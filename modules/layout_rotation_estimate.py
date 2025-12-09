"""
Use the estimate_rotation_singe_obj function to estimate the rotation angle of an object in the scene.
"""

import os
import json
import logging
import numpy as np
import trimesh
import pyrender
from PIL import Image
from estimate_rotation_single_obj import estimate_theta
from render_glb_image import setup_pyrender_offscreen
import torch

GPU_ID = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("layout_rotation_estimate")

def render_glb_multiview(glb_path: str, out_path: str, view_angle, rotation_angle, img_size):
    """
    Render images of a GLB file rotated along the z-axis.
    """
    try:
        # Load the GLB scene
        trimesh_scene = trimesh.load(glb_path, force='scene')
        
        if len(trimesh_scene.geometry) == 0:
            print("  - Warning: No geometry")
            return False
        
        # Calculate scene bounds
        center = trimesh_scene.centroid
        extents = trimesh_scene.extents
        max_extent = np.max(extents)
        
        # Create a renderer instance
        renderer = pyrender.OffscreenRenderer(img_size, img_size)

        # Create a new scene copy
        scene_copy = trimesh_scene.copy()
        
        # Rotate along the Z-axis
        rotation_matrix = trimesh.transformations.rotation_matrix(
            np.radians(-rotation_angle), [0, 0, 1]
        )
        scene_copy.apply_transform(rotation_matrix)
        
        # Create a pyrender scene
        scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0])
        
        # Add rotated geometry
        added_objects = 0
        for name, geom in scene_copy.geometry.items():
            if not hasattr(geom, 'vertices') or len(geom.vertices) == 0:
                continue
                
            try:
                mesh = pyrender.Mesh.from_trimesh(geom)
                
                transforms = scene_copy.graph.get(name)
                if transforms and len(transforms) > 0:
                    pose = transforms[0]
                else:
                    pose = np.eye(4)
                
                scene.add(mesh, pose=pose)
                added_objects += 1
                
            except Exception as e:
                continue
        
        if added_objects == 0:
            print(f"  - {glb_path}: No valid geometry")
            return False
        
        # Create an orthographic camera
        camera_distance = max_extent * 2.5
        cam = pyrender.OrthographicCamera(
            xmag=max_extent * 1.2,
            ymag=max_extent * 1.2,
            znear=0.01,
            zfar=camera_distance * 3
        )
        
        elevation = np.radians(view_angle)
        azimuth = np.radians(90)
        
        x = camera_distance * np.cos(elevation) * np.cos(azimuth)
        y = camera_distance * np.cos(elevation) * np.sin(azimuth)
        z = camera_distance * np.sin(elevation)
        camera_position = center + np.array([x, y, z])
        
        forward = center - camera_position
        forward = forward / np.linalg.norm(forward)
        
        world_up = np.array([0.0, 0.0, 1.0])

        use_backup_up = False
        if np.abs(np.dot(forward, world_up)) > 0.99:
            world_up = np.array([0.0, 1.0, 0.0])
            use_backup_up = True
        
        right = np.cross(forward, world_up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        camera_pose = np.eye(4)
        camera_pose[:3, 0] = right
        camera_pose[:3, 1] = up
        camera_pose[:3, 2] = -forward
        camera_pose[:3, 3] = camera_position
        
        scene.add(cam, pose=camera_pose)
        
        # Add lighting
        scene.ambient_light = np.array([0.4, 0.4, 0.4, 1.0])
        
        key_light = pyrender.DirectionalLight(
            color=np.array([1.0, 1.0, 1.0]), 
            intensity=2.0
        )
        scene.add(key_light, pose=camera_pose)
        
        top_direction = np.array([0.0, 0.0, -1.0])
        top_position = center + np.array([0, 0, camera_distance])
        
        top_pose = np.eye(4)
        top_pose[:3, 2] = top_direction
        top_pose[:3, 3] = top_position
        
        top_light = pyrender.DirectionalLight(
            color=np.array([1.0, 1.0, 1.0]), 
            intensity=1.0
        )
        scene.add(top_light, pose=top_pose)
        
        # Render a single view
        color, depth = renderer.render(scene)
        
        if color is None or color.size == 0:
            print(f"  - {glb_path}：The rendering result is empty")
            return False

        img = Image.fromarray(color)

        # If the backup coordinate system was used, rotate the image 180 degrees clockwise
        if use_backup_up:
            img = img.rotate(180)
            print("  - Applied 180° rotation due to backup coordinate system")

        img.save(out_path)
        return True
        
    except Exception as e:
        print(f"  - Rendering failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def normalize_angle_to_cardinal(angle):
    """
    Normalize a given angle to the nearest cardinal angle (0°, 90°, 180°, 270°)
    """
    angle = angle % 360
    
    cardinal_angles = [0, 90, 180, 270]
    threshold = 10.01
    
    for cardinal in cardinal_angles:
        if abs(angle - cardinal) <= threshold:
            return cardinal
        
        if cardinal == 0 and angle >= (360 - threshold):
            return 0
    
    return angle


from setup_openai_client import setup_openai_client


def process_layout_rotation_estimation(segmentation_results_path, view_angle_file_path, object_images_dir, 
                                     sized_mesh_dir, output_json_path, output_rotation_image_dir, 
                                     estimate_rotation_dir, api_key, proxy_url, base_url):
    """
    Main process for layout rotation estimation
    """
    try:
        # Step 1: Load segmentation results and scene view angle
        logger.info("Step 1: Loading segmentation results...")
        with open(segmentation_results_path, 'r', encoding='utf-8') as f:
            segmentation_data = json.load(f)

        with open(view_angle_file_path, 'r', encoding='utf-8') as f:
            view_angle = float(f.read().strip())

        logger.info(f"Scene view angle: {view_angle} degrees")

        # Step 1.5: Pre-load DINO model
        logger.info("Step 1.5: Pre-loading DINO model (once for all objects)...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device} (GPU ID: {GPU_ID})")
        dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device).eval()
        logger.info("✓ DINO model loaded successfully")

        # Step 2: Estimate rotation angles for each object
        logger.info("Step 2: Estimating object rotations using computer vision...")
        rotation_results = {}
        rotation_results['view_angle'] = view_angle  # Save view angle information
        
        # Get main object information
        main_object = segmentation_data.get('main_object', None)
        logger.info(f"Main object: {main_object}")
        
        for i, obj in enumerate(segmentation_data['objects']):
            object_full_name = segmentation_data['object_names'][i]
            object_class = obj['class_name']
            
            logger.info(f"Processing {object_full_name} ({object_class})...")
            
            # Check if it is the main object
            if object_class == main_object:
                logger.info(f"✓ {object_full_name} is the main object, setting rotation to 0°")
                rotation_results[object_full_name] = 0
                continue

            object_estimate_dir = os.path.join(estimate_rotation_dir, object_full_name)

            theta_result_path = os.path.join(object_estimate_dir, 'theta_result.json')
            if os.path.exists(theta_result_path):
                try:
                    with open(theta_result_path, 'r') as f:
                        cached_result = json.load(f)
                        cached_theta = cached_result.get('theta_deg', 0)

                        normalized_angle = normalize_angle_to_cardinal(cached_theta)
                        rotation_results[object_full_name] = normalized_angle
                        logger.info(f"✓ {object_full_name}: Using cached result {cached_theta:.1f}° → {normalized_angle}° (skipped estimation)")
                        continue
                except Exception as e:
                    logger.warning(f"Failed to read cached result for {object_full_name}: {e}, will re-estimate")
            
            # Object image path
            if obj.get('is_occluded', False):
                # Use cropped image for occluded objects
                scene_object_image = os.path.join(object_images_dir, f"final_{object_full_name}.png")
            else:
                # Use segmented image for non-occluded objects
                scene_object_image = os.path.join(object_images_dir, f"{object_full_name}.png")
            
            glb_path = os.path.join(sized_mesh_dir, f"{object_full_name}.glb")
            
            if not os.path.exists(scene_object_image):
                logger.warning(f"Scene image not found for {object_full_name}: {scene_object_image}")
                continue
            
            if not os.path.exists(glb_path):
                logger.warning(f"GLB file not found for {object_full_name}: {glb_path}")
                continue
            
            
            try:
                client = setup_openai_client(api_key, proxy_url, base_url)
                result = estimate_theta(
                    image=scene_object_image,
                    mesh=glb_path,
                    pitch_deg=view_angle,
                    fov=45.0,
                    outdir=object_estimate_dir,
                    device_str='cuda',
                    image_size=256,
                    faces_per_pixel=50,
                    batch_chunk=4,
                    max_tex_res=1024,
                    coarse_step=5.0,
                    topk=8,
                    refine_steps=140,
                    pitch_delta=0.0,
                    ws=0.2,
                    we=0.1,
                    wa=3,
                    wr=0.01,
                    camera_model='persp',
                    ortho_mag=1.2,
                    pad_ratio=0,
                    client=client,
                    dino_model=dino_model
                )
                
                rotation_angle = result['theta_deg']
                
                # Add angle normalization
                normalized_angle = normalize_angle_to_cardinal(rotation_angle)
                
                if normalized_angle != rotation_angle:
                    logger.info(f"✓ {object_full_name}: {rotation_angle}° → normalized to {normalized_angle}° (loss: {result['loss']:.6f})")
                else:
                    logger.info(f"✓ {object_full_name}: {rotation_angle}° (loss: {result['loss']:.6f})")
                
                rotation_results[object_full_name] = normalized_angle
                
            except Exception as e:
                logger.error(f"Failed to estimate rotation for {object_full_name}: {e}")
                rotation_results[object_full_name] = 0
                continue
        
        # Step 3: Save results
        logger.info("Step 3: Saving rotation results...")
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(rotation_results, f, indent=4, ensure_ascii=False)
        
        logger.info(f"✓ Layout rotation estimation completed!")
        logger.info(f"✓ Scene view angle: {view_angle} degrees")
        logger.info(f"✓ Rotation results saved to: {output_json_path}")
        logger.info(f"✓ Processed {len([k for k in rotation_results.keys() if k != 'view_angle'])} objects")
        
        # Output results summary
        logger.info("\n=== Layout Rotation Results ===")
        for obj_name, angle in rotation_results.items():
            if obj_name != 'view_angle':
                logger.info(f"{obj_name}: {angle}°")

        # Step 4: Render final results
        logger.info("Step 4: Rendering final results...")

        os.makedirs(output_rotation_image_dir, exist_ok=True)

        for obj_name, angle in rotation_results.items():
            if obj_name == 'view_angle':
                continue
            glb_path = os.path.join(sized_mesh_dir, f"{obj_name}.glb")
            out_path = os.path.join(output_rotation_image_dir, f"{obj_name}.png")
            if not os.path.exists(glb_path):
                logger.warning(f"GLB file not found for {obj_name}: {glb_path}")
                continue
            if not render_glb_multiview(glb_path, out_path, view_angle, angle, 600):
                logger.error(f"Failed to render {obj_name} with angle {angle}°")
        logger.info("✓ All objects rendered successfully.")

        return rotation_results, view_angle
        
    except Exception as e:
        logger.error(f"Layout rotation estimation failed: {e}")
        raise

def layout_rotation_estimate_main(output_assets_dir, api_key, proxy_url, base_url):
    """Main function"""
    try:
        segmentation_results_path = os.path.join(output_assets_dir, "image", "segmentation_results.json")
        view_angle_file_path = os.path.join(output_assets_dir, 'scene_view_angle.txt')
        object_images_dir = os.path.join(output_assets_dir, "image")
        sized_mesh_dir = os.path.join(output_assets_dir, 'sized_mesh')
        
        estimate_rotation_dir = os.path.join(output_assets_dir, 'estimate_rotation')
        output_json_path = os.path.join(output_assets_dir, 'layout_json', 'layout_rotation.json')
        output_rotation_image_dir = os.path.join(output_assets_dir, 'layout_rotation_images')

        os.makedirs(estimate_rotation_dir, exist_ok=True)

        process_layout_rotation_estimation(
            segmentation_results_path, 
            view_angle_file_path, 
            object_images_dir, 
            sized_mesh_dir, 
            output_json_path, 
            output_rotation_image_dir,
            estimate_rotation_dir,
            api_key,
            proxy_url,
            base_url
        )
        
    except Exception as e:
        logger.error(f"Main function execution failed: {e}")

if __name__ == '__main__':
    try:
        backend = setup_pyrender_offscreen()
        print(f"Pyrender backend: {backend}")
    except RuntimeError as e:
        print(f"Unable to set up the pyrender environment: {e}")
        raise
    
    output_assets_dir = "output_scene/scene_1/output_assets"
    layout_rotation_estimate_main(output_assets_dir)