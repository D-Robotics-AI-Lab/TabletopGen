"""
Occlusion inpainting based on segmentation results.
"""

import json
import os
import sys
import shutil
from modules.seg_one_Bi import seg_obj
import requests
import time
from modules.seedream40_API import image_to_base64
from volcenginesdkarkruntime import Ark

def generate_inpaint_prompt(occluded_object, occluding_objects):
    """
    Generate prompt for image inpainting.
    """
    # Convert list of occluding objects to string
    occluding_items = ", ".join(occluding_objects)
    
    prompt = f"remove all things on the {occluded_object}, include {occluding_items}, and other items not mentioned that create occlusion."
    
    return prompt

def generate_redraw_prompt(object_name):
    """
    Generate prompt for refined image redrawing.
    """
    return f"Redraw this {object_name}, keeping its general shape, texture and placement, to make it more realistic and clear. Do not rotate it in any way. You can change the background color and expand it appropriately to show the entire object."


def update_segmentation_results(segmentation_data, object_id, updates, json_path):
    """
    Update inpainting image paths in segmentation results.
    """
    try:
        object_found = False

        for obj in segmentation_data.get("objects", []):
            if obj.get("image_path", ""):
                obj_id = obj["image_path"].split("_")[-1].split(".")[0]
                if obj_id == object_id:
                    for key, value in updates.items():
                        obj[key] = value
                    print(f"Updated object {obj['class_name']}_{object_id} with: {updates}")
                    object_found = True
                    break
        
        if not object_found:
            print(f"Warning: Could not find object with ID {object_id} to update")
            return False

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(segmentation_data, f, indent=4, ensure_ascii=False)
        
        return True

    except Exception as e:
        print(f"Error updating segmentation results: {e}")
        return False
    
def seedream_object_api(image_path1, prompt, output_path, ark_api_key=None):
    """
    Generate object image using Seedream 4.0
    """
    # Initialize Ark client
    client = Ark(
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key=ark_api_key or os.environ.get("ARK_API_KEY"),
    )
    
    image1_base64 = image_to_base64(image_path1)
    
    
    images_response = client.images.generate(
        model="doubao-seedream-4-0-250828",
        prompt=prompt,
        image=[image1_base64],
        size="2K",
        response_format="url",
        watermark=False,
        sequential_image_generation="disabled" 
    )
    
    image_url = images_response.data[0].url
    
    response = requests.get(image_url, timeout=120)
    if response.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"Image saved to: {output_path}")
    else:
        raise RuntimeError(f"Failed to download image, status code: {response.status_code}")
    
    return image_url

def process_occluded_objects(json_path, output_dir, final_seg_output_dir, ark_api_key=None):
    """
    Process all occluded objects.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            segmentation_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Segmentation results file not found: {json_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in: {json_path}")
        return None


    if not segmentation_data or "objects" not in segmentation_data:
        print("Error: Invalid segmentation data")
        return []

    occluded_objects = [obj for obj in segmentation_data["objects"] if obj.get("is_occluded", False)]
    
    if not occluded_objects:
        print("No occluded objects found in segmentation results")
        return []

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(final_seg_output_dir, exist_ok=True)
    
    processed_files = []
    
    for obj in segmentation_data["objects"]:
        if not obj.get("is_occluded", False):
            continue

        class_name = obj["class_name"]
        occluded_by = obj.get("occluded_by", [])
        
        if not occluded_by:
            print(f"Warning: No occluding objects found for {class_name}")
            continue
            
        print(f"\nProcessing occluded object: {class_name}")
        
        # Generate inpainting prompt
        inpaint_prompt = generate_inpaint_prompt(class_name, occluded_by)
        
        # Get input image path
        input_image_path = obj.get("inpainting_image_path")
        if not input_image_path:
            print(f"Error: 'inpainting_image_path' not found for object {class_name}")
            continue

        # Convert relative path to absolute path so ComfyUI can find it
        absolute_input_image_path = os.path.abspath(input_image_path)
        if not os.path.exists(absolute_input_image_path):
            print(f"Error: Input image not found at absolute path: {absolute_input_image_path}")
            continue
            
        # --- Step 1: Inpainting ---
        object_id = obj.get("image_path", "").split("_")[-1].split(".")[0] if obj.get("image_path") else "0"
        inpaint_filename_prefix = f"inpainted_{class_name}_{object_id}"
        expected_inpainted_path = os.path.join(output_dir, f"{inpaint_filename_prefix}.png")

        if os.path.exists(expected_inpainted_path):
            print(f"Skipping inpainting, file already exists: {expected_inpainted_path}")
            inpainted_image_path = expected_inpainted_path
        else:
            seedream_object_api(
                image_path1=absolute_input_image_path,
                prompt=inpaint_prompt,
                output_path=expected_inpainted_path,
                ark_api_key=ark_api_key
            )
        

        if not os.path.exists(expected_inpainted_path):
            print(f"No inpainted image found for {class_name}")
            continue

        print(f"Inpainted image saved: {expected_inpainted_path}")
        update_segmentation_results(segmentation_data, object_id, {"inpainted_image_path": expected_inpainted_path}, json_path)

        # --- Step 2: Refined Redrawing ---
        redraw_prompt = generate_redraw_prompt(class_name)

        redraw_filename_prefix = f"redraw_{class_name}_{object_id}"
        expected_redrawn_path = os.path.join(output_dir, f"{redraw_filename_prefix}.png")

        if os.path.exists(expected_redrawn_path):
            print(f"Skipping redraw, file already exists: {expected_redrawn_path}")
            redrawn_image_path = expected_redrawn_path
        else:
            absolute_inpainted_path = os.path.abspath(expected_inpainted_path)
    
            seedream_object_api(
                image_path1=absolute_inpainted_path,
                prompt=redraw_prompt,
                output_path=expected_redrawn_path,
                ark_api_key=ark_api_key
            )

        if not os.path.exists(expected_redrawn_path):
            print(f"No redrawn image found for {class_name}")
            continue

        print(f"Redrawn image saved: {expected_redrawn_path}")
        update_segmentation_results(segmentation_data, object_id, {"redraw_image_path": expected_redrawn_path}, json_path)

        # --- Step 3: Final Segmentation ---
        final_seg_filename = f"final_{class_name}_{object_id}.png"
        final_seg_path = os.path.join(final_seg_output_dir, final_seg_filename)
        
        seg_success = True
        if os.path.exists(final_seg_path):
            print(f"Skipping final segmentation, file already exists: {final_seg_path}")
        else:
            print(f"Segmenting redrawn image to: {final_seg_path}")
      
            seg_success = seg_obj(
                expected_redrawn_path,
                final_seg_path
            )

        if not seg_success:
            print(f"Final segmentation failed for {class_name}")
            continue
        
        print(f"Final segmented image saved: {final_seg_path}")
        update_segmentation_results(segmentation_data, object_id, {"final_segmented_image_path": final_seg_path}, json_path)
        processed_files.append(final_seg_path)
            
        time.sleep(2)
    
    return processed_files

def run_inpaint_pipeline(segmentation_json_path, output_dir, final_seg_output_dir, ark_api_key):
    """
    Execute the complete occlusion inpainting pipeline.
    """

    # 1. Identify initial occluded objects to process
    try:
        with open(segmentation_json_path, 'r', encoding='utf-8') as f:
            initial_data = json.load(f)
        initial_occluded_objects = [
            obj for obj in initial_data.get("objects", []) if obj.get("is_occluded", False)
        ]
        if not initial_occluded_objects:
            print("No occluded objects to process.")
            return True, []
    except Exception as e:
        print(f"Error reading initial segmentation file: {e}")
        return False, []

    try:
        process_occluded_objects(
            segmentation_json_path,
            output_dir,
            final_seg_output_dir,
            ark_api_key=ark_api_key
        )
    finally:
        pass

    print("\nInpainting process completed!")

    # 3. Final verification
    try:
        with open(segmentation_json_path, 'r', encoding='utf-8') as f:
            final_data = json.load(f)
    except Exception as e:
        print(f"Error reading final segmentation file for verification: {e}")
        return False, [f"{obj['class_name']}_{obj['image_path'].split('_')[-1].split('.')[0]}" for obj in initial_occluded_objects]

    missing_objects = []
    final_objects_map = {
        obj["image_path"].split("_")[-1].split(".")[0]: obj 
        for obj in final_data.get("objects", []) if obj.get("image_path")
    }

    for obj in initial_occluded_objects:
        obj_id = obj["image_path"].split("_")[-1].split(".")[0]
        final_obj = final_objects_map.get(obj_id)

        if not final_obj or "final_segmented_image_path" not in final_obj or not os.path.exists(final_obj["final_segmented_image_path"]):
            missing_info = f"{obj['class_name']}_{obj_id}"
            missing_objects.append(missing_info)
            print(f"Verification failed: Final segmented image is missing for {missing_info}")

    if not missing_objects:
        print("Verification successful: All occluded objects have a final segmented image.")
        return True, []
    else:
        print(f"Verification failed. Missing final images for: {missing_objects}")
        return False, missing_objects



if __name__ == "__main__":
    segmentation_json_path = "output_scene/scene_1/output_assets/segmentation/segmentation_results.json"
    output_dir = "output_scene/scene_1/output_assets/inpainting"
    final_seg_output_dir = "output_scene/scene_1/output_assets/inpainting/final_segmentation"
    ark_api_key = "" 

    success, missing = run_inpaint_pipeline(
        segmentation_json_path,
        output_dir,
        final_seg_output_dir,
        ark_api_key
    )

    if success:
        print("Inpainting pipeline completed successfully.")
    else:
        print(f"Inpainting pipeline completed with missing objects: {missing}")
