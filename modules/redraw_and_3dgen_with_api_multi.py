"""
Redraw the object images and generate 3D models of each object.
"""

import os
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from modules.seedream40_API import generate_object_image
from modules.hunyuan3D_API import gen_single_obj_hy3dapi

def is_multi_object(class_name, all_class_names):
    """
    Determine if the current object is of the same type as others (more than one of the same type)
    """
    if sum(1 for name in all_class_names if name == class_name) > 1:
        return True

    this_words = set(class_name.lower().split())
    for other_name in all_class_names:
        if other_name == class_name:
            continue
        other_words = set(other_name.lower().split())
        if this_words & other_words:
            return True
    return False

def redraw_and_3dgen_api(segmentation_json_path, output_dir, hunyuan_secret_id, hunyuan_secret_key, seedream_api_key):
    """
    Redraw all objects and generate 3D GLB files based on the segmentation result JSON.
    """

    with open(segmentation_json_path, 'r', encoding='utf-8') as f:
        seg_data = json.load(f)
    
    scene_image_path = seg_data['image_path']
    objects = seg_data['objects']
    main_object_name = seg_data.get('main_object', None) 
    all_class_names = [obj['class_name'] for obj in objects]
    
    redraw_dir = os.path.join(output_dir, 'redraw_objects')
    mesh_dir = os.path.join(output_dir, 'tex_mesh')
    os.makedirs(redraw_dir, exist_ok=True)
    os.makedirs(mesh_dir, exist_ok=True)
    
    print(f"=== Starting redraw and 3D generation process ===")
    print(f"Number of objects: {len(objects)}")
    
    # Step 1: Redraw all objects
    print("\n--- a. Redraw all objects ---")
    for obj in objects:
        class_name = obj['class_name']
        object_image_path = obj['image_path']
        
        # Determine if it is the main object
        is_main_obj = (class_name == main_object_name)
        is_multi = is_multi_object(class_name, all_class_names)

        base_name = os.path.basename(object_image_path)
        obj_name_with_id = os.path.splitext(base_name)[0] 
        
        redraw_image_path = os.path.join(redraw_dir, f"{obj_name_with_id}.png")
        
        print(f"\nRedrawing: {class_name} ({obj_name_with_id}){' [Main Object]' if is_main_obj else ''}{' [Multiple Objects]' if is_multi else ''}")
        
        # Check if redraw image already exists
        if os.path.exists(redraw_image_path):
            print(f"Redraw image already exists, skipping: {redraw_image_path}")
            obj['redraw_image_path'] = redraw_image_path
            continue
        
        try:
            generate_object_image(
                image_path1=object_image_path,
                image_path2=scene_image_path,
                object_name=class_name,
                output_path=redraw_image_path,
                is_main_obj=is_main_obj,
                is_multi=is_multi,
                ark_api_key=seedream_api_key
            )
            
            obj['redraw_image_path'] = redraw_image_path
            print(f"Redraw completed: {redraw_image_path}")
            
        except Exception as e:
            print(f"Redraw failed: {str(e)}")
            obj['redraw_image_path'] = None
   
    redraw_json_path = os.path.join(redraw_dir, 'segmentation_result_redraw.json')
    with open(redraw_json_path, 'w', encoding='utf-8') as f:
        json.dump(seg_data, f, indent=4, ensure_ascii=False)
    print(f"\nRedraw result JSON saved: {redraw_json_path}")
    
    # Step 2: Generate 3D models (using thread pool with retries)
    print("\n--- b. Generate 3D models (up to 3 concurrent) ---")
    
    def generate_3d_task(obj):
        """Single 3D generation task"""
        class_name = obj['class_name']
        redraw_image_path = obj.get('redraw_image_path')
        
        if not redraw_image_path or not os.path.exists(redraw_image_path):
            print(f"\nSkipping {class_name}: Redraw image does not exist")
            return obj, None
        
        # Extract object name and ID from redraw image path
        base_name = os.path.basename(redraw_image_path)
        obj_name_with_id = os.path.splitext(base_name)[0]
        
        # GLB file save path
        glb_path = os.path.join(mesh_dir, f"{obj_name_with_id}.glb")
        
        print(f"\nGenerating 3D model: {class_name} ({obj_name_with_id})")

        # Check if GLB file already exists and is valid
        if os.path.exists(glb_path) and os.path.getsize(glb_path) > 0:
            print(f"GLB file already exists, skipping generation: {glb_path}")
            return obj, glb_path
        
        try:
            gen_single_obj_hy3dapi(
                SecretId=hunyuan_secret_id,
                SecretKey=hunyuan_secret_key,
                image_path=redraw_image_path,
                output_glb_path=glb_path
            )
            
            print(f"3D model generation completed: {glb_path}")
            return obj, glb_path
            
        except Exception as e:
            print(f"3D generation failed: {str(e)}")
            return obj, None
    
    # Retry logic: up to 3 attempts
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        print(f"\n{'='*60}")
        print(f"Attempt {attempt} to generate 3D models")
        print(f"{'='*60}")
        
        # Filter objects to process
        if attempt == 1:
            objects_to_process = objects
        else:
            objects_to_process = [
                obj for obj in objects 
                if not obj.get('glb_path') or not os.path.exists(obj['glb_path']) or os.path.getsize(obj['glb_path']) == 0
            ]
            
            if not objects_to_process:
                break
                
            print(f"Number of objects to retry: {len(objects_to_process)}")
            for obj in objects_to_process:
                print(f"  - {obj['class_name']}")
        
        # Use thread pool, up to 3 concurrent
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_obj = {executor.submit(generate_3d_task, obj): obj for obj in objects_to_process}
            
            for future in as_completed(future_to_obj):
                obj, glb_path = future.result()
                obj['glb_path'] = glb_path
        
        # Check object counts
        total_objects = len(objects)
        successful_objects = sum(1 for obj in objects if obj.get('glb_path') and os.path.exists(obj['glb_path']) and os.path.getsize(obj['glb_path']) > 0)
        failed_objects = total_objects - successful_objects
        
        print(f"\n{'='*60}")
        print(f"Attempt {attempt} results summary:")
        print(f"  Total objects: {total_objects}")
        print(f"  Successfully generated: {successful_objects}")
        print(f"  Failed count: {failed_objects}")
        print(f"{'='*60}")
        
        # If all successful, exit retry loop
        if successful_objects == total_objects:
            print(f"\nAll 3D models generated successfully!")
            break
        
        # If there are still failures and max retries not reached
        if attempt < max_retries:
            print(f"\nThere are still {failed_objects} objects not successfully generated, preparing to retry...")
            print("Failed objects:")
            for obj in objects:
                if not obj.get('glb_path') or not os.path.exists(obj['glb_path']) or os.path.getsize(obj['glb_path']) == 0:
                    print(f"  - {obj['class_name']}")
        else:
            # Reached max retries with failures
            print(f"\nReached max retries ({max_retries}), still have {failed_objects} objects failed to generate")
            print("Failed objects:")
            for obj in objects:
                if not obj.get('glb_path') or not os.path.exists(obj['glb_path']) or os.path.getsize(obj['glb_path']) == 0:
                    print(f"  - {obj['class_name']}")
            
            # Save failed results JSON
            failed_json_path = os.path.join(output_dir, 'segmentation_result_failed.json')
            with open(failed_json_path, 'w', encoding='utf-8') as f:
                json.dump(seg_data, f, indent=4, ensure_ascii=False)
            print(f"\nSaved failed results JSON: {failed_json_path}")
            
            return False
    
    final_json_path = os.path.join(output_dir, 'segmentation_result_final.json')
    with open(final_json_path, 'w', encoding='utf-8') as f:
        json.dump(seg_data, f, indent=4, ensure_ascii=False)
    print(f"\nSaved final results JSON: {final_json_path}")
    
    print("\n=== All processing completed ===")
    return seg_data

if __name__ == "__main__":
    # Example usage
    segmentation_json_path = "output_scene/scene_1/output_assets/image/segmentation_results.json"
    output_dir = "output_scene/scene_1/output_assets"

    hunyuan_secret_id = ""
    hunyuan_secret_key = ""
    seedream_api_key = ""
    
    result = redraw_and_3dgen_api(
        segmentation_json_path=segmentation_json_path,
        output_dir=output_dir,
        hunyuan_secret_id=hunyuan_secret_id,
        hunyuan_secret_key=hunyuan_secret_key,
        seedream_api_key=seedream_api_key
    )
