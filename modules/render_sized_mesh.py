"""
Scale GLB files according to size_axis.json and render from the front-top view
"""

import os
import json
import numpy as np
import trimesh
import pyrender
from PIL import Image, ImageDraw, ImageFont
import shutil
from modules.render_glb_image import setup_pyrender_offscreen

def scale_mesh_by_size_axis(mesh_path, size_axis_data, output_path):
    """
    Scale GLB files according to the size information in size_axis.json.
    """
    try:
        scene = trimesh.load(mesh_path, force='scene')
        
        if len(scene.geometry) == 0:
            print(f"  - Warning: No geometry.")
            return False
        
        current_extents = scene.extents
        
        target_sizes = size_axis_data["size"]  # [x, y, z]
        
        scale_factors = np.array(target_sizes) / current_extents
        
        scale_matrix = np.eye(4)
        scale_matrix[0, 0] = scale_factors[0]  # x scale
        scale_matrix[1, 1] = scale_factors[1]  # y scale
        scale_matrix[2, 2] = scale_factors[2]  # z scale
        
        # Apply scaling transformation
        scene.apply_transform(scale_matrix)
        
        # Save the scaled scene
        scene.export(output_path)
        
        return True
        
    except Exception as e:
        print(f"  - Scaling failed: {e}")
        return False

def render_glb_front_view(glb_path: str, out_path: str, view_angle, img_size):
    """
    Render GLB files from the front-top view using pyrender
    """
    try:
        print(f"Rendering front view: {os.path.basename(glb_path)}")
        
        trimesh_scene = trimesh.load(glb_path, force='scene')
        
        if len(trimesh_scene.geometry) == 0:
            print("  - Warning: No geometry.")
            return False
        
        scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0])
        
        added_objects = 0
        for name, geom in trimesh_scene.geometry.items():
            if not hasattr(geom, 'vertices') or len(geom.vertices) == 0:
                continue
                
            try:
                mesh = pyrender.Mesh.from_trimesh(geom)
                
                transforms = trimesh_scene.graph.get(name)
                if transforms and len(transforms) > 0:
                    pose = transforms[0]
                else:
                    pose = np.eye(4)
                
                scene.add(mesh, pose=pose)
                added_objects += 1
                
            except Exception as e:
                continue
        
        if added_objects == 0:
            print("  - Warning: No valid geometry to render.")
            return False
        
        bounds = trimesh_scene.bounds
        center = trimesh_scene.centroid
        extents = trimesh_scene.extents
        max_extent = np.max(extents)
        
        camera_distance = max_extent * 2.5
        cam = pyrender.OrthographicCamera(
            xmag=max_extent * 1.2,
            ymag=max_extent * 1.2,
            znear=0.01,
            zfar=camera_distance * 3
        )
        
        # Set front-top view angle
        elevation = np.radians(view_angle)
        azimuth = np.radians(90)    
        
        x = camera_distance * np.cos(elevation) * np.cos(azimuth)
        y = camera_distance * np.cos(elevation) * np.sin(azimuth)
        z = camera_distance * np.sin(elevation)
        camera_position = center + np.array([x, y, z])
        
        forward = center - camera_position
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-6:
            print("  - Warning: Camera too close to target.")
            return False
        forward = forward / forward_norm
        
        world_up = np.array([0.0, 0.0, 1.0])
        
        use_backup_up = False
        if np.abs(np.dot(forward, world_up)) > 0.99:
            world_up = np.array([0.0, 1.0, 0.0])
            use_backup_up = True
        
        right = np.cross(forward, world_up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            print("  - Warning: Degenerate camera orientation.")
            return False
        right = right / right_norm
        
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
        
        # Render
        renderer = pyrender.OffscreenRenderer(img_size, img_size)
        color, depth = renderer.render(scene)
        renderer.delete()
        
        if color is None or color.size == 0:
            print("  - Warning: Render result is empty.")
            return False
        
        # After rendering, overlay coordinate axes
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

def process_sized_mesh_rendering(input_mesh_dir, size_axis_path, output_mesh_dir, output_image_dir, view_angle, image_size):
    """
    Process scaling and rendering.
    """
    os.makedirs(output_mesh_dir, exist_ok=True)
    os.makedirs(output_image_dir, exist_ok=True)
    
    with open(size_axis_path, 'r', encoding='utf-8') as f:
        size_axis_data = json.load(f)
    
    glb_files = [f for f in os.listdir(input_mesh_dir) if f.lower().endswith('.glb')]
    
    if not glb_files:
        print("No GLB files found!")
        return

    success_count = 0
    
    for glb_file in glb_files:
        object_name = os.path.splitext(glb_file)[0]
        
        if object_name not in size_axis_data:
            print(f"Skipping {object_name}: Not found in size_axis data")
            continue

        input_mesh_path = os.path.join(input_mesh_dir, glb_file)
        output_mesh_path = os.path.join(output_mesh_dir, glb_file)
        output_image_path = os.path.join(output_image_dir, f"{object_name}.png")
        
        # Scale GLB file
        scale_success = scale_mesh_by_size_axis(
            input_mesh_path, 
            size_axis_data[object_name], 
            output_mesh_path
        )
        
        if not scale_success:
            print(f"  ✗ Scaling failed!")
            continue
        
        # Render front-top view image
        render_success = render_glb_front_view(output_mesh_path, output_image_path, view_angle, image_size)

        if render_success:
            success_count += 1
        else:
            print(f"  ✗ Rendering failed.")

    print(f"✓ Scaled GLB files saved to: {output_mesh_dir}")
    print(f"✓ Front view images saved to: {output_image_dir}")

def main():
    """Main function"""

    input_mesh_dir = 'output_scene/scene_1/output_assets/rotated_mesh_final'
    size_axis_path = 'output_scene/scene_1/output_assets/layout_json/size_axis.json'
    output_mesh_dir = 'output_scene/scene_1/output_assets/sized_mesh'
    output_image_dir = 'output_scene/scene_1/output_assets/sized_images_front'

    print("Start the scaling and front-top view rendering process...")
    process_sized_mesh_rendering(input_mesh_dir, size_axis_path, output_mesh_dir, output_image_dir, 90, 600)

if __name__ == '__main__':

    try:
        backend = setup_pyrender_offscreen()
        print(f"Pyrender backend: {backend}")
    except RuntimeError as e:
        print(f"Unable to set up the pyrender environment: {e}")
        raise

    main()