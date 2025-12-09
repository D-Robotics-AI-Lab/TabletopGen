"""
Model preprocessing before simulation import - align coordinate system center to bounding box center
Output to output_assets/centered_mesh
"""

import os
import trimesh
import numpy as np
from pathlib import Path


def center_model_origin_with_transform(input_path, output_path):
    """Adjust model origin to bounding box center, optionally apply coordinate system transform"""
    try:
        print(f"Processing: {input_path}")
        
        # Load mesh
        mesh = trimesh.load(input_path, force='mesh')
        
        # Handle scene objects
        if hasattr(mesh, 'geometry') and isinstance(mesh.geometry, dict):
            geometries = list(mesh.geometry.values())
            if len(geometries) == 0:
                print(f"Error: No geometries found in {input_path}")
                return False
            elif len(geometries) == 1:
                mesh = geometries[0]
            else:
                try:
                    mesh = trimesh.util.concatenate(geometries)
                except Exception as e:
                    print(f"Error merging geometries, using first geometry: {e}")
                    mesh = geometries[0]
        
        if not isinstance(mesh, trimesh.Trimesh):
            print(f"Error: Loaded object is not a Trimesh, type: {type(mesh)}")
            return False    
        
        # Calculate bounding box center and translate
        bbox_center = mesh.bounding_box.centroid
        translation = -bbox_center
        
        mesh.apply_translation(translation)
        
        # Verify
        new_center = mesh.bounding_box.centroid
        center_distance = np.linalg.norm(new_center)
        
        if center_distance < 0.001:
            pass
        else:
            print("! Origin centering may not be perfect")
        
        # Save
        mesh.export(output_path)
        print(f"✓ Saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        import traceback
        traceback.print_exc()
        return False

def process_all_models(output_assets_dir, apply_transform=False):
    """Process all GLB models in the directory"""
    # Define input and output directories
    input_dir = os.path.join(output_assets_dir, "sized_mesh")
    output_dir = os.path.join(output_assets_dir, "centered_mesh")

    print("Model Origin Centering Preprocessor")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory does not exist: {input_dir}")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to Path object and find all GLB files
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    glb_files = list(input_path.glob("*.glb"))
    
    if not glb_files:
        print(f"No GLB files found in {input_dir}")
        return
    
    print(f"Found {len(glb_files)} GLB files to process")
    if apply_transform:
        print("Will apply coordinate system transformation (Blender -> Isaac)")
    
    successful = 0
    failed = 0
    
    for glb_file in glb_files:
        # Output file path
        output_file = output_path / glb_file.name
        
        # Check if processing is needed
        if output_file.exists() and output_file.stat().st_mtime > glb_file.stat().st_mtime:
            print(f"Skipping {glb_file.name} (already processed and up-to-date)")
            continue
        
        if center_model_origin_with_transform(str(glb_file), str(output_file)):
            successful += 1
        else:
            failed += 1
    
    print(f"\n=== Processing Complete ===")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(glb_files)}")


if __name__ == "__main__":
    output_assets_dir = "output_scene/scene_1/output_assets"

    process_all_models(output_assets_dir)