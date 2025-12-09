"""Use pyrender to render isometric views of GLB files."""

import os
import numpy as np
import trimesh
import pyrender
from PIL import Image, ImageDraw, ImageFont
import argparse

def setup_pyrender_offscreen():
    """
    Set up the offline rendering environment for pyrender
    """

    try:
        # Start xvfb
        os.system('pkill Xvfb')  # First, kill any existing xvfb processes
        os.system('Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +render -noreset > /dev/null 2>&1 &')
        import time
        time.sleep(2)
        
        os.environ['DISPLAY'] = ':99'
        os.environ.pop('PYOPENGL_PLATFORM', None) 
        
        # Test renderer
        test_renderer = pyrender.OffscreenRenderer(100, 100)
        test_renderer.delete()
        return 'xvfb'
    except Exception as e:
        print(f"xvfb wrong: {e}")
    
    # Fallback: try other backends
    backends = ['egl', 'osmesa']
    
    for backend in backends:
        try:
            os.environ['PYOPENGL_PLATFORM'] = backend
            
            test_renderer = pyrender.OffscreenRenderer(100, 100)
            test_renderer.delete()
            return backend
        except Exception as e:
            print(f"{backend} backend: {e}")
            continue
    
    raise RuntimeError("Unable to set up the pyrender offline rendering environment")


def create_axis_arrow(length=1.0, radius=0.02, color=[1.0, 0.0, 0.0]):
    """
    Create an axis arrow (cylinder + cone)
    """
    # Create cylinder as the shaft
    cylinder = trimesh.creation.cylinder(radius=radius, height=length * 0.8)
    cylinder.apply_translation([0, 0, length * 0.4])
    
    # Create cone as the arrowhead
    cone = trimesh.creation.cone(radius=radius * 2, height=length * 0.2)
    cone.apply_translation([0, 0, length * 0.9])
    
    # Combine geometries
    arrow = cylinder + cone
    
    return arrow

def create_text_mesh(text, size=1.0):
    """
    Create 3D text mesh
    """
    try:
        if text == 'X':
            box1 = trimesh.creation.box([size*0.8, size*0.1, size*0.1])
            box2 = trimesh.creation.box([size*0.1, size*0.8, size*0.1])
            box1.apply_transform(trimesh.transformations.rotation_matrix(np.pi/4, [0, 0, 1]))
            box2.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/4, [0, 0, 1]))
            return box1 + box2
        elif text == 'Y':
            box1 = trimesh.creation.box([size*0.4, size*0.1, size*0.1])
            box2 = trimesh.creation.box([size*0.4, size*0.1, size*0.1])
            box3 = trimesh.creation.box([size*0.1, size*0.4, size*0.1])
            box1.apply_translation([size*0.2, size*0.3, 0])
            box1.apply_transform(trimesh.transformations.rotation_matrix(np.pi/4, [0, 0, 1]))
            box2.apply_translation([-size*0.2, size*0.3, 0])
            box2.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/4, [0, 0, 1]))
            box3.apply_translation([0, -size*0.2, 0])
            return box1 + box2 + box3
        elif text == 'Z':
            box1 = trimesh.creation.box([size*0.8, size*0.1, size*0.1])
            box2 = trimesh.creation.box([size*0.8, size*0.1, size*0.1])
            box3 = trimesh.creation.box([size*0.6, size*0.1, size*0.1])
            box1.apply_translation([0, size*0.35, 0])
            box2.apply_translation([0, -size*0.35, 0])
            box3.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/4, [0, 0, 1]))
            return box1 + box2 + box3
    except Exception as e:
        # If creation fails, return a small box
        return trimesh.creation.box([size*0.3, size*0.3, size*0.1])

def draw_coordinate_axes_on_image(img, center_x=None, center_y=None, axis_length=200):
    """
    Draw coordinate axes at the center of the image
    """
    draw = ImageDraw.Draw(img)
    width, height = img.size
    
    if center_x is None:
        center_x = width // 2
    if center_y is None:
        center_y = height // 2
    
    # Isometric projection axis vectors
    # X axis: down-left
    x_end_x = center_x - int(axis_length * 0.866)  # left
    x_end_y = center_y + int(axis_length * 0.5)    # down
    
    # Y axis: down-right
    y_end_x = center_x + int(axis_length * 0.866)  # right
    y_end_y = center_y + int(axis_length * 0.5)    # down
    
    # Z axis: up
    z_end_x = center_x
    z_end_y = center_y - axis_length
    
    line_width = 8
    
    # X axis - Red, down-left
    draw.line([(center_x, center_y), (x_end_x, x_end_y)], fill=(255, 0, 0), width=line_width)
    draw.polygon([(x_end_x, x_end_y), 
                  (x_end_x + 18, x_end_y - 12), 
                  (x_end_x + 12, x_end_y + 15)], 
                 fill=(255, 0, 0))
    
    # Y axis - Green, down-right
    draw.line([(center_x, center_y), (y_end_x, y_end_y)], fill=(0, 200, 0), width=line_width)
    draw.polygon([(y_end_x, y_end_y), 
                  (y_end_x - 18, y_end_y - 12), 
                  (y_end_x - 12, y_end_y + 15)], 
                 fill=(0, 200, 0))
    
    # Z axis - Blue, up
    draw.line([(center_x, center_y), (z_end_x, z_end_y)], fill=(0, 100, 255), width=line_width)
    draw.polygon([(z_end_x, z_end_y), 
                  (z_end_x - 12, z_end_y + 20), 
                  (z_end_x + 12, z_end_y + 20)], 
                 fill=(0, 100, 255))
    
    # Draw axis labels
    try:
        font_size = max(36, axis_length // 3)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()

    draw.text((x_end_x - 30, x_end_y + 10), "X", fill=(255, 0, 0), font=font)

    draw.text((y_end_x + 15, y_end_y + 10), "Y", fill=(0, 200, 0), font=font)
    
    draw.text((z_end_x - 15, z_end_y - 50), "Z", fill=(0, 100, 255), font=font)
    
    return img


def render_glb_with_pyrender(glb_path: str, out_path: str, img_size: int = 1024):
    """
    Rendering GLB files using pyrender
    """
    try:
        # print(f"Rendering: {os.path.basename(glb_path)}")
        
        trimesh_scene = trimesh.load(glb_path, force='scene')
        
        if len(trimesh_scene.geometry) == 0:
            print("  - Warning: There is no geometric body.")
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
            print("  - No valid geometries to render.")
            return False
        
        bounds = trimesh_scene.bounds
        center = trimesh_scene.centroid
        extents = trimesh_scene.extents
        max_extent = np.max(extents)
        
        # Create orthographic camera (isometric projection)
        camera_distance = max_extent * 2.0
        cam = pyrender.OrthographicCamera(
            xmag=max_extent * 1.0,
            ymag=max_extent * 1.0,
            znear=0.01,
            zfar=camera_distance * 3
        )
        
        elevation = np.radians(35.264)
        azimuth = np.radians(45)
        
        x = camera_distance * np.cos(elevation) * np.cos(azimuth)
        y = camera_distance * np.cos(elevation) * np.sin(azimuth)
        z = camera_distance * np.sin(elevation)
        camera_position = center + np.array([x, y, z])
        
        forward = center - camera_position
        forward = forward / np.linalg.norm(forward)
        
        world_up = np.array([0.0, 0.0, 1.0])
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
        scene.ambient_light = np.array([0.3, 0.3, 0.3, 1.0])
        
        key_light = pyrender.DirectionalLight(
            color=np.array([1.0, 1.0, 1.0]), 
            intensity=2.5
        )
        scene.add(key_light, pose=camera_pose)
        
        fill_direction = np.array([-1.0, -1.0, 1.0])
        fill_direction = fill_direction / np.linalg.norm(fill_direction)
        fill_position = center + fill_direction * camera_distance * 0.7
        
        fill_pose = np.eye(4)
        fill_pose[:3, 2] = -fill_direction
        fill_pose[:3, 3] = fill_position
        
        fill_light = pyrender.DirectionalLight(
            color=np.array([1.0, 1.0, 1.0]), 
            intensity=1.5
        )
        scene.add(fill_light, pose=fill_pose)
        
        renderer = pyrender.OffscreenRenderer(img_size, img_size)
        
        color, depth = renderer.render(scene)
        renderer.delete()
        
        if color is None or color.size == 0:
            print("  - The rendering result is empty.")
            return False
        
        unique_colors = np.unique(color.reshape(-1, color.shape[-1]), axis=0)
        if len(unique_colors) <= 2:
            print(f"  - Warning: The rendering result may be empty. (only {len(unique_colors)} colors)")
        
        img = Image.fromarray(color)
        img = draw_coordinate_axes_on_image(img)
        
        # Save image
        img.save(out_path)
        
        return True
        
    except Exception as e:
        print(f"  - Rendering fail: {e}")
        import traceback
        traceback.print_exc()
        return False


def batch_render(input_dir: str, output_dir: str, img_size: int = 1024):
    """
    Batch rendering of GLB files
    """
    if not os.path.exists(input_dir):
        print(f"No input directory: {input_dir}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    glb_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.glb')]
    
    if not glb_files:
        print("No GLB file was found! ")
        return
    
    success_count = 0
    for i, fname in enumerate(glb_files):
        in_path = os.path.join(input_dir, fname)
        out_name = os.path.splitext(fname)[0] + '.png'
        out_path = os.path.join(output_dir, out_name)
        
        if render_glb_with_pyrender(in_path, out_path, img_size):
            success_count += 1
    
    print(f"\nImage processing is complete! {success_count}/{len(glb_files)}")


def render_glb_main(input_dir=None, output_dir=None, size=1024):

    os.makedirs(output_dir, exist_ok=True)

    single = None

    
    if single:
        if not os.path.exists(single):
            print(f"The file does not exist: {single}")
            return
        
        output_path = single.replace('.glb', '.png')
        render_glb_with_pyrender(single, output_path, size)
    else:
        batch_render(input_dir, output_dir, size)


if __name__ == '__main__':
    try:
        backend = setup_pyrender_offscreen()
        print(f"backend: {backend}")
    except RuntimeError as e:
        print(f"Unable to set up the pyrender environment: {e}")
        raise

    input_dir = 'output_scene/scene_1/output_assets/rotated_mesh' 
    output_dir = 'output_scene/scene_1/output_assets/tex_images' 