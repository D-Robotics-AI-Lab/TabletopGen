"""
Perform final xy-axis alignment.
"""

import os
import json
import trimesh
import numpy as np
from pathlib import Path
import shutil
from modules.rotate_glb import rotate_glb
from modules.render_glb_image import render_glb_with_pyrender, setup_pyrender_offscreen

def extract_glb_ratios(input_dir, output_path):
    """
    Extract XYZ axis ratios of GLB files, normalize X axis to 1.
    """
    ratios_data = {}
    
    # Get all GLB files
    glb_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.glb')]
    
    if not glb_files:
        print("GLB file not found!")
        return ratios_data
    
    for glb_file in glb_files:
        try:
            file_path = os.path.join(input_dir, glb_file)
            
            # Load GLB scene
            scene = trimesh.load(file_path, force='scene')
            
            if len(scene.geometry) == 0:
                print(f"  - Warning: {glb_file} has no geometry")
                continue
            
            # Calculate bounding box of the entire scene
            bounds = scene.bounds
            extents = scene.extents  # [x_size, y_size, z_size]
            
            # Ensure no zero dimensions
            if np.any(extents <= 0):
                print(f"  - Warning: {glb_file} has zero - sized axes: {extents}")
                # Set zero dimensions to a very small value
                extents = np.maximum(extents, 1e-6)
            
            # Normalize X axis to 1, calculate ratios for Y and Z
            x_size = extents[0]
            y_size = extents[1]
            z_size = extents[2]
            
            # Normalized ratios
            ratio_x = 1.0
            ratio_y = y_size / x_size
            ratio_z = z_size / x_size
            
            # Keep 3 decimal places
            ratios = [
                round(ratio_x, 3),
                round(ratio_y, 3), 
                round(ratio_z, 3)
            ]
            
            # Use filename (without extension) as key
            file_key = os.path.splitext(glb_file)[0]
            ratios_data[file_key] = {
                "ratios": ratios,
                "extents": [round(x_size, 3), round(y_size, 3), round(z_size, 3)]
            }
            
        except Exception as e:
            print(f"  - Error processing {glb_file}: {e}")
            continue
    
    # Save results
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(ratios_data, f, indent=4, ensure_ascii=False)

        print(f"\nSuccessfully saved to: {output_path}")

    except Exception as e:
        print(f"Error occurred while saving the file: {e}")
    
    return ratios_data

def compare_xy_sizes(x, y):
    """
    Compare sizes of x and y.
    """
    diff = abs(x - y)
    larger = max(x, y)
    
    # If difference is less than or equal to 20% of the larger value, consider them similar
    if diff <= larger * 0.2:
        return "similar"
    elif x > y:
        return "x_larger"
    else:
        return "y_larger"

def is_thin_object(size_list):
    """
    Determine if the object is a thin type (z dimension is very small).
    """
    if len(size_list) < 3:
        return False
    
    z_size = size_list[2]
    max_size = max(size_list)
    
    # z is the smallest dimension, and is more than 5 times smaller than the largest dimension
    return z_size == min(size_list) and max_size / z_size > 5

def check_z_alignment(glb_extents, axis_sizes):
    """
    Check z-axis alignment and return required rotation operation.
    """
    glb_x, glb_y, glb_z = glb_extents
    glb_sizes = [glb_x, glb_y, glb_z]
    
    # Find the smallest axis in GLB model
    min_glb_value = min(glb_sizes)
    min_glb_index = glb_sizes.index(min_glb_value)
    
    # Find the largest axis in GLB model
    max_glb_value = max(glb_sizes)
    max_glb_index = glb_sizes.index(max_glb_value)
    
    # Find the largest axis in size_axis
    max_axis_value = max(axis_sizes)
    max_axis_index = axis_sizes.index(max_axis_value)
    
    # Case 1: z is already the smallest
    if min_glb_index == 2:
        return False, None
    
    # Case 2: The largest axis is correct, and the difference between z and the smallest is within 1.5 times
    if max_glb_index == max_axis_index:
        ratio = glb_z / min_glb_value
        if ratio <= 1.5:
            # print(1)
            return False, None
    
    # Case 3: Rotation needed
    # If x is the smallest, rotate 90 degrees along y-axis to make x become z
    # If y is the smallest, rotate 90 degrees along x-axis to make y become z
    if min_glb_index == 0:  # x is the smallest
        return True, 'y'
    elif min_glb_index == 1:  # y is the smallest
        return True, 'x'
    
    return False, None

def align_coordinate_systems(input_dir, size_axis_path, rotated_vlm_path, output_mesh_dir, output_image_dir, ratios_output_path, rotation_log_path):
    """
    Align object coordinate systems.
    """
    # Create output directories
    os.makedirs(output_mesh_dir, exist_ok=True)
    os.makedirs(output_image_dir, exist_ok=True)
    
    # Load size_axis data
    with open(size_axis_path, 'r', encoding='utf-8') as f:
        size_axis_data = json.load(f)
    with open(rotated_vlm_path, 'r', encoding='utf-8') as f:
        rotated_vlm_data = json.load(f)
    
    # Extract GLB file ratio data
    ratios_data = extract_glb_ratios(input_dir, ratios_output_path)
    
    rotation_log = {}
    
    print("\nBegin final x-y alignment...")
    
    for object_name in ratios_data.keys():
        if object_name not in size_axis_data:
            print(f"Skipping {object_name}: Not found in size_axis data")
            continue

        # Get xyz dimensions of GLB file
        glb_extents = ratios_data[object_name]["extents"]
        glb_x, glb_y, glb_z = glb_extents[0], glb_extents[1], glb_extents[2]
        
        # Get xyz dimensions in size_axis
        axis_sizes = size_axis_data[object_name]["size"]
        axis_x, axis_y, axis_z = axis_sizes[0], axis_sizes[1], axis_sizes[2]
        
        # Check if rotated along x or y axis in rotated_vlm
        has_xy_rotation = False
        if object_name in rotated_vlm_data:
            rotations = rotated_vlm_data[object_name]
            if rotations and len(rotations) >= 2:
                rotation_axis = rotations[0]
                if rotation_axis in ['x', 'y']:
                    has_xy_rotation = True
        
        # Initialize rotation record
        applied_rotations = []
        
        # If not rotated along x or y axis, and is a thin object, check z-axis alignment first
        if not has_xy_rotation and is_thin_object(axis_sizes):
            needs_z_rotation, z_rotation_axis = check_z_alignment(glb_extents, axis_sizes)
            if needs_z_rotation:
                applied_rotations.append([z_rotation_axis, 90])
                # After applying z-axis alignment rotation, update glb_extents for subsequent xy alignment judgment
                if z_rotation_axis == 'x':
                    glb_x, glb_y, glb_z = glb_x, glb_z, glb_y
                elif z_rotation_axis == 'y':
                    glb_x, glb_y, glb_z = glb_z, glb_y, glb_x
        
        # Continue with xy alignment judgment
        glb_xy_comparison = compare_xy_sizes(glb_x, glb_y)
        axis_xy_comparison = compare_xy_sizes(axis_x, axis_y)
        
        # Determine if xy rotation is needed
        needs_xy_rotation = False
        
        if glb_xy_comparison == "similar" or axis_xy_comparison == "similar":
            pass  # Object length and width are similar, no rotation needed
        elif glb_xy_comparison != axis_xy_comparison:
            needs_xy_rotation = True  # Size order is inconsistent, need to rotate 90 degrees along z-axis
        else:
            pass  # Size order is consistent, no rotation needed
        
        if needs_xy_rotation:
            applied_rotations.append(['z', 90])
        
        # Process GLB file
        input_mesh_path = os.path.join(input_dir, f"{object_name}.glb")
        output_mesh_path = os.path.join(output_mesh_dir, f"{object_name}.glb")
        output_image_path = os.path.join(output_image_dir, f"{object_name}.png")
        
        if not os.path.exists(input_mesh_path):
            print(f"  ✗ GLB file not found: {input_mesh_path}")
            continue
        
        if applied_rotations:
            # Apply all rotations
            try:
                mesh = trimesh.load(input_mesh_path, force='mesh')
                for rotation_axis, angle in applied_rotations:
                    mesh = rotate_glb(mesh, rotation_axis, angle)
                mesh.export(output_mesh_path)
                rotation_log[object_name] = applied_rotations
            except Exception as e:
                print(f"  ✗ Rotation failed: {e}")
                shutil.copy2(input_mesh_path, output_mesh_path)
                rotation_log[object_name] = []
        else:
            shutil.copy2(input_mesh_path, output_mesh_path)
            rotation_log[object_name] = []
        
        # Render image
        try:
            render_success = render_glb_with_pyrender(output_mesh_path, output_image_path, 1024)
        except Exception as e:
            print(f"  ✗ Error in image rendering: {e}")
    
    # Save rotation log
    with open(rotation_log_path, 'w', encoding='utf-8') as f:
        json.dump(rotation_log, f, indent=4, ensure_ascii=False)

    print(f"\n✓ x-y alignment is completed!")


def final_xy_main(output_assets_dir):
    # Set paths
    input_dir = os.path.join(output_assets_dir, "rotated_mesh_vlm")
    size_axis_path = os.path.join(output_assets_dir, "layout_json", "size_axis.json")
    rotated_vlm_path = os.path.join(output_assets_dir, "layout_json", "rotated_vlm.json")
    output_mesh_dir = os.path.join(output_assets_dir, "rotated_mesh_final")
    output_image_dir = os.path.join(output_assets_dir, "rotated_images_final")
    ratios_output_path = os.path.join(output_assets_dir, "layout_json", "glb_ratios.json")
    rotation_log_path = os.path.join(output_assets_dir, "layout_json", "coordinate_alignment_log.json")

    align_coordinate_systems(input_dir, size_axis_path, rotated_vlm_path, output_mesh_dir, output_image_dir, ratios_output_path, rotation_log_path)

if __name__ == '__main__':
    try:
        backend = setup_pyrender_offscreen()
        print(f"backend: {backend}")
    except RuntimeError as e:
        print(f"Unable to set up the pyrender environment: {e}")
        raise

