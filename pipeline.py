"""
TabletopGen pipeline with single image input.
"""

import os
import sys
import json
import time
import shutil
import logging
import argparse
from pathlib import Path
from google import genai
import yaml

# Import configuration module
from configs.pipeline_config import (
    get_pipeline_dir, get_scene_dir, get_comfy_image_dir,
    get_output_assets_dir, set_working_directory_to_scene,
    restore_working_directory, get_output_scene_dir,
    get_next_scene_id
)


def load_config(config_path: Path | str | None = None):
    """Load the configuration and verify the required API Key."""
    if config_path is None:
        config_path = Path(__file__).resolve().parent / "configs" / "config.yaml"
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}."
        )

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:
        raise RuntimeError(f"Failed to read configuration file: {e}")

    api_cfg = cfg.get("api_keys", {}) or {}
    required = {
        "doubao": "DOUBAO_API_KEY",
        "hunyuan_secret_id": "HY_SECRET_ID",
        "hunyuan_secret_key": "HY_SECRET_KEY",
        "gpt_api_key": "GPT_API_KEY",
    }

    values = {}
    for key, env_name in required.items():
        val = api_cfg.get(key)
        if val is None or str(val).strip() == "":
            raise ValueError(f"Missing or empty configuration: api_keys.{key} ({env_name})")
        values[env_name] = str(val).strip()

    base_url = api_cfg.get("base_url")
    if base_url is None or str(base_url).strip() == "":
        base_url = "https://openrouter.ai/api/v1"
    values["BASE_URL"] = str(base_url).strip()

    proxy_cfg = cfg.get("proxy") or {}
    http_proxy = proxy_cfg.get("http")
    https_proxy = proxy_cfg.get("https")

    if http_proxy:
        os.environ.setdefault("HTTP_PROXY", http_proxy)
    if https_proxy:
        os.environ.setdefault("HTTPS_PROXY", https_proxy)

    return values, cfg

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Pipeline Directory
PIPELINE_DIR = get_pipeline_dir()
print(f"Pipeline directory: {PIPELINE_DIR}")

sys.path.append(PIPELINE_DIR)
os.environ['PIPELINE_DIR'] = PIPELINE_DIR

# Load configuration
CONFIG_VALUES, _RAW_CONFIG = load_config()
HY_SECRET_ID = CONFIG_VALUES["HY_SECRET_ID"]
HY_SECRET_KEY = CONFIG_VALUES["HY_SECRET_KEY"]
DOUBAO_API_KEY = CONFIG_VALUES["DOUBAO_API_KEY"]
GPT_API_KEY = CONFIG_VALUES["GPT_API_KEY"]
BASE_URL = CONFIG_VALUES["BASE_URL"]
PROXY_URL = os.environ.get("HTTP_PROXY")  # If no proxy is configured, this will be None


def setup_scene_directories(scene_id, input_image_path):
    """
    Set up scene directory structure
    """

    output_scene_dir = get_output_scene_dir()
    os.makedirs(output_scene_dir, exist_ok=True)
    
    scene_dir = get_scene_dir(scene_id)
    
    directories = [
        scene_dir,
        get_comfy_image_dir(scene_dir),
        get_output_assets_dir(scene_dir),
        os.path.join(get_output_assets_dir(scene_dir), "image"),
        os.path.join(get_output_assets_dir(scene_dir), "layout_json"),
    ]
    
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
    
    # Copy input image to scene_image.png
    target_image_path = os.path.join(get_comfy_image_dir(scene_dir), "scene_image.png")
    shutil.copy2(input_image_path, target_image_path)
    logger.info(f"Input image copied to: {target_image_path}")
    
    logger.info(f"Scene directory created: {scene_dir}")
    return scene_dir


def step1_get_asset_image(scene_dir):
    """
    Step 1: Generate images of individual objects
    """
    logger.info("=== Step 1: Generate images of individual objects ===")
    
    try:
        from modules.get_asset_image import get_asset_image_main
        
        scene_image_path = os.path.join(get_comfy_image_dir(scene_dir), "scene_image.png")
        if not os.path.exists(scene_image_path):
            raise FileNotFoundError(f"Scene image not found: {scene_image_path}")

        output_dir = get_output_assets_dir(scene_dir)
        
        logger.info("Detecting and segmenting objects in the scene...")
        get_asset_image_main(scene_image_path, GPT_API_KEY, PROXY_URL, BASE_URL, output_dir, box_threshold=0.3, text_threshold=0.2, confidence_threshold=0.4)
        
        logger.info("Step 1 completed: Object images generated")
        return True
        
    except Exception as e:
        logger.error(f"Step 1 failed: {e}")
        return False

def step2_inpaint_occlusion(scene_dir):
    """
    Step 2: Repair occluded objects
    """
    logger.info("=== Step 2: Repair occluded objects ===")
    
    try:
        from modules.inpaint_occlusion import run_inpaint_pipeline
        
        segmentation_json_path = os.path.join(get_output_assets_dir(scene_dir), "image", "segmentation_results.json")
        output_dir = get_comfy_image_dir(scene_dir)
        final_seg_output_dir = os.path.join(get_output_assets_dir(scene_dir), "image")
        
        if not os.path.exists(segmentation_json_path):
            logger.warning("Segmentation results file not found, skipping occlusion repair step")
            return True
        
        original_cwd = set_working_directory_to_scene(scene_dir)
        
        try:
            success, missing = run_inpaint_pipeline(segmentation_json_path, output_dir, final_seg_output_dir, DOUBAO_API_KEY)

            if missing:
                print(f"The following objects are missing: {', '.join(missing)}")
            while not success:
                logger.info("Retrying occlusion repair...")
                success, missing = run_inpaint_pipeline(segmentation_json_path, output_dir, final_seg_output_dir, DOUBAO_API_KEY)

            logger.info("Step 2 completed: Occlusion repair done")
            return True
        finally:
            restore_working_directory(original_cwd)
        
    except Exception as e:
        logger.error(f"Step 2 failed: {e}")
        return False


    
def step3_generate_3d_models_api(scene_dir):
    """
    Step 3: Redraw and generate 3D models
    """
    logger.info("=== Step 3: Redraw and generate 3D models ===")
    
    try:
        from modules.redraw_and_3dgen_with_api_multi import redraw_and_3dgen_api
        segmentation_json_path = os.path.join(get_output_assets_dir(scene_dir), "image", "segmentation_results.json")
        output_dir = get_output_assets_dir(scene_dir)

        result = redraw_and_3dgen_api(
            segmentation_json_path=segmentation_json_path,
            output_dir=output_dir,
            hunyuan_secret_id=HY_SECRET_ID,
            hunyuan_secret_key=HY_SECRET_KEY,
            seedream_api_key=DOUBAO_API_KEY
        )

        if not result:
            print("Batch redraw and 3D generation process failed")
            return False
        logger.info("Step 3 completed: 3D models generated")

        logger.info("Creating backup after step 3 completion...")
        scene_dir_name = os.path.basename(scene_dir)
        backup_dir_name = f"{scene_dir_name}_step3"
        backup_dir_path = os.path.join(scene_dir, backup_dir_name)
        
        # Create backup directory and copy the entire contents of scene_dir
        if os.path.exists(backup_dir_path):
            shutil.rmtree(backup_dir_path)
        os.makedirs(backup_dir_path, exist_ok=True)
        
        # Iterate over all files and folders in scene_dir and copy them to the backup directory
        for item in os.listdir(scene_dir):
            item_path = os.path.join(scene_dir, item)
            backup_item_path = os.path.join(backup_dir_path, item)
            
            # Skip the backup directory itself
            if item == backup_dir_name:
                continue
                
            if os.path.isdir(item_path):
                shutil.copytree(item_path, backup_item_path)
            else:
                shutil.copy2(item_path, backup_item_path)
        
        logger.info(f"Step 3 backup created: {backup_dir_path}")
        return True
        
    except Exception as e:
        logger.error(f"Step 3 failed: {e}")
        return False

def step4_coordinate_canonicalization(scene_dir):
    """
    Step 4: Coordinate canonicalization
    """
    logger.info("=== Step 4: Coordinate canonicalization ===")
    
    try:
        from modules.rotate_glb import process_glb_to_isaac_axis
        from modules.vlm_size_axis import vlm_size_axis_main
        from modules.render_glb_image import render_glb_main, setup_pyrender_offscreen
        from modules.vlm_rotation_analyzer import vlm_rotation_main
        from modules.final_xy_alignment import final_xy_main

        # Set up the pyrender environment
        try:
            backend = setup_pyrender_offscreen()
            logger.info(f"backend: {backend}")
        except RuntimeError as e:
            logger.error(f"Unable to set up the pyrender environment: {e}")
            raise


        original_cwd = set_working_directory_to_scene(scene_dir)
        
        try:
            # 4.1 Preliminary rotation
            logger.info("4.1 Executing preliminary rotation (x90z180)...")
            tex_mesh_dir = os.path.join(get_output_assets_dir(scene_dir), "tex_mesh")
            rotated_mesh_dir = os.path.join(get_output_assets_dir(scene_dir), "rotated_mesh")
            
            if os.path.exists(tex_mesh_dir):
               process_glb_to_isaac_axis(tex_mesh_dir, rotated_mesh_dir)

            json_output_dir = os.path.join(get_output_assets_dir(scene_dir), "layout_json")
            segmentation_json_path = os.path.join(get_output_assets_dir(scene_dir), "image", "segmentation_results.json")

            
            # 4.2 VLM analyze size and coordinates
            logger.info("4.2 VLM analyzing size and coordinates...")
            vlm_size_axis_main(segmentation_json_path, json_output_dir, get_output_assets_dir(scene_dir), get_comfy_image_dir(scene_dir), GPT_API_KEY, PROXY_URL, BASE_URL)

            rotated_images_dir = os.path.join(get_output_assets_dir(scene_dir), "rotated_images")
            render_glb_main(rotated_mesh_dir, rotated_images_dir)
            
            # 4.3 VLM coordinate canonicalization
            logger.info("4.3 VLM coordinate canonicalization...")
            vlm_rotation_main(get_output_assets_dir(scene_dir), GPT_API_KEY, PROXY_URL, BASE_URL)
            
            # 4.4 Final xy alignment
            logger.info("4.4 Final xy alignment...")
            final_xy_main(get_output_assets_dir(scene_dir))
            
            logger.info("Step 4 completed: Coordinate canonicalization finished")
            return True
            
        finally:
            restore_working_directory(original_cwd)
        
    except Exception as e:
        logger.error(f"步骤6失败: {e}")
        return False

def step5_rotation_estimation(scene_dir):
    """
    Step 5: Rotation estimation
    """
    logger.info("=== Step 5: Rotation estimation ===")
    
    try:
        from modules.vlm_scene_view_angle import scene_angle_and_sized_mesh
        from modules.merge_images import merge_images
        import subprocess
        
        scene_image_path = os.path.join(get_comfy_image_dir(scene_dir), "scene_image.png")
        output_assets_dir = get_output_assets_dir(scene_dir)

        original_cwd = set_working_directory_to_scene(scene_dir)
        
        try:
            # 5.1 Analyze top view and render scaled objects
            logger.info("5.1 Analyzing top view and rendering scaled objects...")
            scene_angle_and_sized_mesh(scene_image_path, output_assets_dir, GPT_API_KEY, PROXY_URL, BASE_URL)

            # 5.2 Estimate rotation pose using loss (run in rotation environment)
            logger.info("5.2 Estimating object rotation pose (switching to rotation environment)...")
            
            # Construct command to run in rotation environment
            rotation_script_path = os.path.join(PIPELINE_DIR, "modules", "run_rotation_estimation.py")
            cmd = [
                "conda", "run", "-n", "rotation", 
                "python", rotation_script_path, 
                output_assets_dir,
                GPT_API_KEY,
                PROXY_URL,
                BASE_URL
            ]
            
            logger.info(f"Executing command: {' '.join(cmd)}")
            
            # Execute command
            result = subprocess.run(
                cmd,
                cwd=PIPELINE_DIR,
                capture_output=True,
                text=True,
                timeout=3600
            )
            
            if result.returncode != 0:
                logger.error(f"Rotation estimation failed:")
                logger.error(f"stdout: {result.stdout}")
                logger.error(f"stderr: {result.stderr}")
                return False
            else:
                logger.info("Rotation estimation completed successfully")
                if result.stdout:
                    logger.info(f"Output: {result.stdout}")

            # 5.3 Merge images
            logger.info("5.3 Merging rotation images...")
            merge_images(os.path.join(output_assets_dir, "layout_rotation_images"))
            
            logger.info("Step 5 completed: Rotation estimation finished")
            return True
            
        finally:
            restore_working_directory(original_cwd)
        
    except Exception as e:
        logger.error(f"Step 5 failed: {e}")
        return False

def step6_position_estimation(scene_dir):
    """
    Step 6: Position estimation
    """
    logger.info("=== Step 6: Position estimation ===")
    
    try:
        from modules.vlm_generate_topview_scene_with_check import generate_topview_scene
        from modules.vlm_topview_boundingbox_enhanced import topview_bbox_enhanced_main
        from modules.calculate_vlm_pose_with_ro import calculate_pose_main
        from modules.vlm_placement_order import placement_order_main
        from modules.layout_pose import calculate_final_layout

        scene_image_path = os.path.join(get_comfy_image_dir(scene_dir), "scene_image.png")
        output_assets_dir = get_output_assets_dir(scene_dir)
        comfy_image_dir = get_comfy_image_dir(scene_dir)

        original_cwd = set_working_directory_to_scene(scene_dir)
        
        try:
            # 6.1 Generate top view
            logger.info("6.1 Generating top view...")
            success_6_1 = generate_topview_scene(scene_image_path, os.path.join(output_assets_dir, "image", "topview_scene.png"), DOUBAO_API_KEY, GPT_API_KEY, PROXY_URL, BASE_URL)
            if not success_6_1:
                logger.error("6.1 Generating top view failed")
                return False

            # 6.2 Analyze top view bounding box
            logger.info("6.2 Analyzing top view bounding box...")
            success_6_2 = topview_bbox_enhanced_main(output_assets_dir, comfy_image_dir, DOUBAO_API_KEY, GPT_API_KEY, PROXY_URL, BASE_URL)
            if not success_6_2:
                logger.error("6.2 Analyzing top view bounding box failed")
                return False
            
            # 6.3 Calculate VLM pose (choose method based on number of segmented objects)
            logger.info("6.3 Calculating object pose...")
            seg_json_path = os.path.join(output_assets_dir, "image", "segmentation_results.json")
            num_objects = None
            
            with open(seg_json_path, "r", encoding="utf-8") as f:
                seg = json.load(f)

            num_objects = len(seg["object_names"])

            if num_objects >= 15:
                logger.info("Number of objects >= 15, using multi-object method")
                from modules.calculate_vlm_pose_with_ro_and_multi import calculate_pose_main as calculate_pose_main_multi
                calculate_pose_main_multi(output_assets_dir)
            else:
                calculate_pose_main(output_assets_dir)
            
            # 6.4 Determine placement order
            logger.info("6.4 Determining placement order...")
            success_6_4 = placement_order_main(output_assets_dir, comfy_image_dir, GPT_API_KEY, PROXY_URL, BASE_URL)
            if not success_6_4:
                logger.error("6.4 Determining placement order failed")
                return False
            
            # 6.5 Calculate final pose
            logger.info("6.5 Calculating final pose...")
            calculate_final_layout(output_assets_dir)

            logger.info("Step 6 completed: Position estimation finished")
            return True
            
        finally:
            restore_working_directory(original_cwd)
        
    except Exception as e:
        logger.error(f"Step 6 failed: {e}")
        return False

def step7_glb_scene(scene_dir, scene_id):
    """
    Step 7: Assembly scene GLB
    """
    logger.info("=== Step 7: Assembly scene GLB ===")
    
    try:
        from modules.align_axis_centered import process_all_models
        from modules.assembly_scene_glb import assemble_scene_flexible

        output_assets_dir = get_output_assets_dir(scene_dir)
        
        original_cwd = set_working_directory_to_scene(scene_dir)
        
        try:
            # 7.1 Align coordinate system center
            logger.info("7.1 Aligning coordinate system center to bounding box...")
            process_all_models(output_assets_dir)
            
            # 7.2 Assemble scene GLB
            logger.info("7.2 Assembling scene GLB...")
            assemble_scene_flexible(scene_id, scene_dir)

            logger.info("Step 7 completed: Assembly scene GLB finished")
            return True
            
        finally:
            restore_working_directory(original_cwd)
        
    except Exception as e:
        logger.error(f"Step 7 failed: {e}")
        return False

def list_existing_scenes():
    """
    List all existing scenes
    """
    output_scene_dir = get_output_scene_dir()
    
    if not os.path.exists(output_scene_dir):
        logger.info("No scenes have been created yet")
        return []
    
    scenes = []
    for item in os.listdir(output_scene_dir):
        scene_path = os.path.join(output_scene_dir, item)
        if item.startswith("scene_") and os.path.isdir(scene_path):
            try:
                scene_num = int(item.split("_")[1])
                desc_file = os.path.join(scene_path, "scene_description.txt")
                description = "No description"
                if os.path.exists(desc_file):
                    with open(desc_file, 'r', encoding='utf-8') as f:
                        description = f.read().strip()
                
                scenes.append({
                    'id': scene_num,
                    'folder': item,
                    'description': description,
                    'path': scene_path
                })
            except (ValueError, IndexError):
                continue
    
    # Sort by ID
    scenes.sort(key=lambda x: x['id'])
    
    
    return scenes

def run_full_pipeline(input_image_path, skip_step = [], scene_id=None):
    """
    Run the full 3D scene generation pipeline
    
    Args:
        input_image_path: Input image path
        scene_id: Scene ID, if None, it will be automatically assigned
        use_api: Whether to use the API version (default True), False to use the local version
    """
    # Check if input image exists
    if not os.path.exists(input_image_path):
        logger.error(f"Input image does not exist: {input_image_path}")
        return False
    
    # Automatically assign scene_id
    if scene_id is None:
        scene_id = get_next_scene_id()
    
    logger.info(f"Starting full pipeline - Scene ID: {scene_id}")

    # Set scene directory
    scene_dir = setup_scene_directories(scene_id, input_image_path)
    

    steps = [
        ("Generate asset images", lambda: step1_get_asset_image(scene_dir)),
        ("Inpaint occluded objects", lambda: step2_inpaint_occlusion(scene_dir)),
        ("Repaint and generate 3D models", lambda: step3_generate_3d_models_api(scene_dir)),
        ("Coordinate system canonicalization", lambda: step4_coordinate_canonicalization(scene_dir)),
        ("VLM rotation estimation", lambda: step5_rotation_estimation(scene_dir)),
        ("Position estimation", lambda: step6_position_estimation(scene_dir)),
        ("GLB assembly", lambda: step7_glb_scene(scene_dir, scene_id)),
    ]

    # Execute all steps
    total_steps = len(steps)
    for i, (step_name, step_func) in enumerate(steps, 1):

        if i in skip_step: continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Executing step {i}/{total_steps}: {step_name}")
        logger.info(f"{'='*60}")

        try:
            start_time = time.time()
            success = step_func()
            end_time = time.time()
            
            if success:
                logger.info(f"Step {i} completed successfully, time taken: {end_time - start_time:.2f} seconds")
            else:
                error_msg = f"Step {i} failed, pipeline terminated"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
                
        except Exception as e:
            error_msg = str(e)
            if "step" or "Step" not in error_msg:
                error_msg = f"Step {i} encountered an exception: {error_msg}"
            else:
                error_msg = f"Step {i} encountered an exception: {error_msg}"
            
            logger.error(error_msg)
            logger.error(f"Pipeline terminated at step {i}")
            raise RuntimeError(error_msg)
    
    logger.info(f"\n{'='*60}")
    logger.info("🎉 Full pipeline executed successfully!")
    logger.info(f"Scene results saved at: {scene_dir}, Scene ID: {scene_id}, Input image: {input_image_path}")
    logger.info(f"{'='*60}")
    
    return True

def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description="TabletopGen Pipeline")
    parser.add_argument("--input_image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--scene_id", type=int, default=None, help="Optional Scene ID")
    parser.add_argument("--skip_step", type=int, nargs='*', default=[], help="Steps to skip (space separated integers)")
    
    args = parser.parse_args()

    logger.info(f"Pipeline directory: {PIPELINE_DIR}")
    logger.info(f"Output scene directory: {get_output_scene_dir()}")
    
    # List existing scenes
    existing_scenes = list_existing_scenes()
    
    # Get the next scene ID
    next_scene_id = get_next_scene_id()
    logger.info(f"Next scene will use ID: {next_scene_id}")
    
    # Input image path
    input_image_path = args.input_image
    
    
    logger.info(f"Input image path: {input_image_path}")
    
    # Run full pipeline
    success = run_full_pipeline(
        input_image_path, 
        skip_step=args.skip_step,
        scene_id=args.scene_id,
    )
    
    if success:
        sys.exit(0)
    else:
        logger.error("Pipeline execution failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
