"""
Use VLM to analyze the top-down view of the scene and calculate the rotation angle of objects.
"""

import os
import base64
import re
import logging
import numpy as np
import trimesh
import pyrender
from PIL import Image
from modules.render_sized_mesh import process_sized_mesh_rendering
from modules.render_glb_image import setup_pyrender_offscreen
from modules.setup_openai_client import setup_openai_client

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("layout_rotation_vlm")

def extract_last_integer(text):
    try:
        numbers = re.findall(r'\b\d+\b', text)
        if numbers:
            return int(numbers[-1])
        return None
    except Exception as e:
        logger.error(f"Failed to extract integer: {e}")
        return None

def analyze_scene_view_angle(client, scene_image_path):
    """
    Analyze the top-down view angle of the scene image.
    """
    try:
        with open(scene_image_path, "rb") as image_file:
            image_data = image_file.read()

        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        prompt = """Please accurately estimate the angle of the top-down view shown in the illustration. Generally speaking, this angle will not be very close to vertical (90 degrees). You can carefully observe the visibility of objects in the scene and make judgments based on visual common sense.
Think carefully, output the detailed reasoning process.
Finally, on a separate line, only output the final degree (in int format, e.g.: 50)"""

        logger.info("Analyzing scene view angle with GPT-4.1...")
        
        input_messages = [
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": prompt },
                    { "type": "image_url", "image_url": f"data:image/webp;base64,{image_base64}" }
                ]
            }
        ]

        response = client.chat.completions.create(
            model="openai/gpt-4.1",
            messages=input_messages
        )
        content = response.choices[0].message.content.strip()
        
        logger.info(f"GPT response for view angle: {content}")

        angle = extract_last_integer(content)
        if angle is not None:
            logger.info(f"Extracted view angle: {angle} degrees")
            return angle
        else:
            logger.warning("Failed to extract view angle, using default 70 degrees")
            return 70
            
    except Exception as e:
        logger.error(f"Scene view angle analysis failed: {e}")
        return 70


def render_glb_multiview(glb_path: str, out_path: str, view_angle, rotation_angle, img_size):
    """
    Render the GLB file image rotated along the z-axis.
    """
    try:
        trimesh_scene = trimesh.load(glb_path, force='scene')
        
        if len(trimesh_scene.geometry) == 0:
            print("  - Warning: No geometry")
            return False
        
        # Calculate scene boundaries
        center = trimesh_scene.centroid
        extents = trimesh_scene.extents
        max_extent = np.max(extents)
        
        # Create a renderer instance to reduce X server pressure by reusing it
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
        
        # Calculate camera position
        x = camera_distance * np.cos(elevation) * np.cos(azimuth)
        y = camera_distance * np.cos(elevation) * np.sin(azimuth)
        z = camera_distance * np.sin(elevation)
        camera_position = center + np.array([x, y, z])
        
        # Construct camera pose matrix
        forward = center - camera_position
        forward = forward / np.linalg.norm(forward)
        
        world_up = np.array([0.0, 0.0, 1.0])
        # Check if forward is parallel to world_up, record if backup coordinate system is used
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

        # If a backup coordinate system was used, rotate the image 180 degrees clockwise
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

def scene_angle_and_sized_mesh(scene_image_path, output_assets_dir, api_key, proxy_url, base_url):
    try:
        input_mesh_dir = os.path.join(output_assets_dir, 'rotated_mesh_final')
        size_axis_path = os.path.join(output_assets_dir, 'layout_json', 'size_axis.json')
        output_mesh_dir = os.path.join(output_assets_dir, 'sized_mesh')
        output_image_dir = os.path.join(output_assets_dir, 'sized_images_front')

        client = setup_openai_client(api_key, proxy_url, base_url)
        
        # Step 1: Analyze scene view angle
        logger.info("Step 1: Analyzing scene view angle...")
        view_angle = analyze_scene_view_angle(client, scene_image_path)
        
        # Save the top-down view angle to a file
        view_angle_file_path = os.path.join(output_assets_dir, 'scene_view_angle.txt')
        with open(view_angle_file_path, 'w') as f:
            f.write(str(view_angle))
        logger.info(f"Scene view angle ({view_angle}) saved to {view_angle_file_path}")

        # Step 2: Render all objects using the view angle
        logger.info(f"Step 2: Rendering objects with view angle {view_angle} degrees...")
        process_sized_mesh_rendering(input_mesh_dir, size_axis_path, output_mesh_dir, output_image_dir, view_angle, 600)


    except Exception as e:
        logger.error(f"Scene angle and sized mesh failed: {e}")
        raise

if __name__ == '__main__':
    try:
        backend = setup_pyrender_offscreen()
        print(f"Pyrender backend: {backend}")
    except RuntimeError as e:
        print(f"Unable to set up the pyrender environment: {e}")
        raise
