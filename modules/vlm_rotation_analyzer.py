"""
Align coordinate system using VLM.
"""

import os
import json
import base64
import httpx
import openai
import time
import trimesh
import numpy as np
from modules.rotate_glb import rotate_glb
import logging
import shutil
from modules.render_glb_image import render_glb_with_pyrender, setup_pyrender_offscreen
from modules.setup_openai_client import setup_openai_client

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("vlm_rotation_analyzer")


def analyze_rotation_with_gpt(client, image_path, object_info):
    """
    Use GPT to analyze object rotation requirements
    """
    try:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()

        image_base64 = base64.b64encode(image_data).decode('utf-8')

        prompt = f"""You are a 3D model orientation reviewer, responsible for comparing the rendered model with the given description of the actual placement direction of the object, judging and correcting the model coordinate system orientation.

Input:
- A top view of the rendered 3D model, with the local X (red), Y (green), and Z (blue) axes superimposed.
- A description of the actual placement of the object, describing the placement state and xyz orientation of the object, and can also be used as a reference with size.

Please follow the steps below to determine whether the object needs to be rotated:
1. **Z-axis direction**: Determine whether the placement state of the rendered model matches the description. If not, rotate along the x or y axis. For example, a book is described as lying flat, but it is standing up in the rendered model (the z axis points to the height of the book rather than the thickness), so it needs to be rotated.
2. **X and Y direction**: Determine whether the direction of the x and y axes in the rendered model is consistent with the length/width of the description. If not, rotate along the z axis. You need to refer to both the x and y sizes for analysis, and **must** ensure that the rotation is correct.
3. If the rotation was performed in the first step, the second step needs to determine the direction of the new x and y axes after the rotation. Therefore, there may be two rotations.
4. Output ["rotation axis", angle], where the angle is a multiple of ±90.

Output example (strict list):
["x", 90,"z",-90]
If no rotation is required, output: []

The object description list is:
{object_info}"""

        logger.info("Using GPT to analyze rotation requirements...")

        input_messages = [
            {
                "role": "user",
                "content": [
                    { "type": "text",  "text": prompt },
                    { "type": "image_url", "image_url": f"data:image/png;base64,{image_base64}" }
                ]
            }
        ]

        response = client.chat.completions.create(
            model="openai/gpt-5",
            messages=input_messages,
            temperature=1.0,
            top_p=1.0,
            store=True,
            max_tokens=2048
        )
        content = response.choices[0].message.content.strip()
        
        logger.info(f"GPT response: {content}")
        
        try:
            last_line = content.strip().split('\n')[-1].strip()

            if last_line.startswith('```') and last_line.endswith('```'):
                last_line = last_line.strip('```').strip().split('\n')[-1].strip()

            if last_line.startswith('[') and last_line.endswith(']'):
                rotation_list = eval(last_line)
                return rotation_list
            else:
                return []
        except Exception as e:
            logger.error(f"Failed to parse rotation instructions: {e}, lase line: {last_line}")
            return []
            
    except Exception as e:
        logger.error(f"The GPT API call failed: {e}")
        return []
    


def apply_rotations_to_mesh(mesh_path, rotations, output_path):
    """
    Apply rotation to GLB file
    """
    try:
        mesh = trimesh.load(mesh_path, force='mesh')
        
        # Apply rotation sequence
        for i in range(0, len(rotations), 2):
            if i + 1 < len(rotations):
                axis = rotations[i]
                angle = rotations[i + 1]
                mesh = rotate_glb(mesh, axis, angle)
        
        # Save rotated mesh
        mesh.export(output_path)
        return True
        
    except Exception as e:
        logger.error(f"Failed to rotate: {e}")
        return False

def has_xy_rotation(rotations):
    """
    Check if rotation list contains x or y axis rotation
    """
    if not rotations:
        return False
    
    for i in range(0, len(rotations), 2):
        if i + 1 < len(rotations):
            axis = rotations[i]
            if axis.lower() in ['x', 'y']:
                return True
    return False

def process_vlm_rotation_analysis(images_dir, size_axis_path, mesh_input_dir, mesh_output_dir,images_output_dir,api_key,proxy_url,base_url):
    """
    Process VLM rotation analysis for all objects, including secondary verification process
    """
    try:
        client = setup_openai_client(api_key, proxy_url, base_url)

        os.makedirs(mesh_output_dir, exist_ok=True)

        os.makedirs(images_output_dir, exist_ok=True)

        with open(size_axis_path, 'r', encoding='utf-8') as f:
            size_axis_data = json.load(f)
        
        rotation_results = {}
        problematic_objects = []
        
        image_files = [f for f in os.listdir(images_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        logger.info(f"Find {len(image_files)} image files.")
        
        for image_file in image_files:
            object_name = os.path.splitext(image_file)[0]

            if object_name not in size_axis_data:
                logger.warning(f"Skipping {object_name}: Not found in size_axis data")
                continue
            
            logger.info(f"\n {object_name}:")
            
            object_info = {object_name: size_axis_data[object_name]}
            
            # First GPT analysis
            image_path = os.path.join(images_dir, image_file)
            first_rotations = analyze_rotation_with_gpt(client, image_path, object_info)

            logger.info(f"  First GPT suggested rotations: {first_rotations}")

            # Apply first rotation to GLB file
            input_mesh_path = os.path.join(mesh_input_dir, f"{object_name}.glb")
            temp_mesh_path = os.path.join(mesh_output_dir, f"{object_name}_temp.glb")
            output_mesh_path = os.path.join(mesh_output_dir, f"{object_name}.glb")
            final_image_path = os.path.join(images_output_dir, f"{object_name}.png")
            
            if not os.path.exists(input_mesh_path):
                logger.error(f"  ✗ GLB file not found: {input_mesh_path}")
                continue
            
            # Apply first rotation
            if first_rotations:
                success = apply_rotations_to_mesh(input_mesh_path, first_rotations, temp_mesh_path)
                if not success:
                    logger.error(f"  ✗ First rotation failed.")
                    continue

            else:
                # If no rotation needed, copy directly
                shutil.copy2(input_mesh_path, temp_mesh_path)
                logger.info(f"  ✓ No rotation is required for the first time.")
            
            # Check if x or y axis rotation is included, if so, secondary verification is needed
            needs_second_check = has_xy_rotation(first_rotations)
            
            if needs_second_check:
                logger.info(f"  First rotation includes x/y-axis rotation, performing second verification...")

                # Render image after first rotation
                render_success = render_glb_with_pyrender(temp_mesh_path, final_image_path, 1024)
                if not render_success:
                    logger.error(f"  ✗ First rotation image rendering failed.")
                    continue

                time.sleep(2)
                
                # Second GPT analysis
                second_rotations = analyze_rotation_with_gpt(client, final_image_path, object_info)
                logger.info(f"  Second GPT suggested rotations: {second_rotations}")

                # Check second result
                if has_xy_rotation(second_rotations):
                    # If x/y axis rotation is still needed in the second time, mark as problematic object
                    problematic_objects.append(object_name)
                    # Save result of first rotation
                    shutil.copy2(temp_mesh_path, output_mesh_path)
                    rotation_results[object_name] = first_rotations
                else:
                    # If second time no rotation or only z-axis rotation needed, apply second rotation
                    if second_rotations:
                        final_success = apply_rotations_to_mesh(temp_mesh_path, second_rotations, output_mesh_path)
                        if final_success:
                            # Re-render final image
                            render_glb_with_pyrender(output_mesh_path, final_image_path, 1024)
                            # Merge rotation instructions
                            final_rotations = first_rotations + second_rotations
                            rotation_results[object_name] = final_rotations
                        else:
                            logger.error(f"  ✗ Second rotation failed.")
                            shutil.copy2(temp_mesh_path, output_mesh_path)
                            rotation_results[object_name] = first_rotations
                    else:
                        # No rotation needed for the second time
                        shutil.copy2(temp_mesh_path, output_mesh_path)
                        rotation_results[object_name] = first_rotations
                
            else:
                # If first time does not contain x/y axis rotation, save result directly and render image
                shutil.copy2(temp_mesh_path, output_mesh_path)
                rotation_results[object_name] = first_rotations
                
                # Render image for objects without secondary verification
                render_success = render_glb_with_pyrender(output_mesh_path, final_image_path, 1024)
                if render_success:
                    pass
                else:
                    logger.error(f"  ✗ Final image rendering failed.")

            # Delete temporary file
            if os.path.exists(temp_mesh_path):
                os.remove(temp_mesh_path)
            
            time.sleep(2)
        
        output_json_path = os.path.join(os.path.dirname(size_axis_path), 'rotated_vlm.json')
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(rotation_results, f, indent=4, ensure_ascii=False)
        
        logger.info(f"✓ VLM coordinate alignment completed! A total of {len(rotation_results)} objects were processed")
        
        return rotation_results, problematic_objects
        
    except Exception as e:
        logger.error(f"处理过程中出错: {e}")
        raise

def vlm_rotation_main(output_assets_dir,api_key,proxy_url,base_url):
    """Main function"""
    images_dir = os.path.join(output_assets_dir, "rotated_images")
    size_axis_path = os.path.join(output_assets_dir, "layout_json", "size_axis.json")
    mesh_input_dir = os.path.join(output_assets_dir, "rotated_mesh")
    mesh_output_dir = os.path.join(output_assets_dir, "rotated_mesh_vlm")
    images_output_dir = os.path.join(output_assets_dir, "rotated_images_vlm")

    if not os.path.exists(images_dir):
        logger.error(f"图片目录不存在: {images_dir}")
        return
    
    if not os.path.exists(size_axis_path):
        logger.error(f"size_axis.json文件不存在: {size_axis_path}")
        return
    
    if not os.path.exists(mesh_input_dir):
        logger.error(f"输入GLB目录不存在: {mesh_input_dir}")
        return
    
    # Process all objects
    try:
        results,problematic_objects = process_vlm_rotation_analysis(
            images_dir=images_dir,
            size_axis_path=size_axis_path,
            mesh_input_dir=mesh_input_dir,
            mesh_output_dir=mesh_output_dir,
            images_output_dir=images_output_dir,
            api_key=api_key,
            proxy_url=proxy_url,
            base_url=base_url
        )
        
        # Output processing result summary
        logger.info("\n=== VLM Coordinate Alignment Results ===")
        for object_name, rotations in results.items():
            if rotations:
                logger.info(f"{object_name}: {rotations}")
            else:
                logger.info(f"{object_name}: No rotation needed.")

        # Output potentially problematic objects
        if problematic_objects:
            logger.warning(f"\nThe following objects may have problems (both analyses suggest rotation on the x/y axis):")
            for obj in problematic_objects:
                logger.warning(f"  - {obj}")
                
    except Exception as e:
        logger.error(f"Main function execution failed: {e}")

if __name__ == '__main__':
    try:
        backend = setup_pyrender_offscreen()
        logger.info(f"Pyrender backend: {backend}")
    except RuntimeError as e:
        logger.error(f"Unable to set up pyrender environment: {e}")
        raise