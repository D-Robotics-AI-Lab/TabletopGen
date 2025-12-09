"""Rotate GLB files, including preliminary rotation with process_glb_to_isaac_axis"""

import os
import trimesh
import numpy as np

def rotate_glb(mesh, axis,angle_deg):
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)

    # Rotation matrix
    if axis == 'x':
        Rt = np.array([
            [1, 0,  0, 0],
            [0, c, -s, 0],
            [0, s,  c, 0],
            [0, 0,  0, 1]
        ])
    elif axis == 'y':
        Rt = np.array([
            [ c, 0, s, 0],
            [ 0, 1, 0, 0],
            [-s, 0, c, 0],
            [ 0, 0, 0, 1]
        ])
    elif axis == 'z':
        Rt = np.array([
            [c, -s, 0, 0],
            [s,  c, 0, 0],
            [0,  0, 1, 0],
            [0,  0, 0, 1]
        ])
    else:
        raise ValueError(f"Unsupported axis '{axis}'.")

    mesh.apply_transform(Rt)
    return mesh


def process_glb_to_isaac_axis(input_dir, output_dir):
    """
    Preliminary processing of GLB files, uniformly rotating from Blender coordinate system to Isaac coordinate system.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Uniformly rotate 90 degrees around x-axis, then 180 degrees around z-axis
    for filename in os.listdir(input_dir):
            src = os.path.join(input_dir, filename)
            mesh = trimesh.load(src, force='mesh')
            mesh1 = rotate_glb(mesh, axis='x', angle_deg=90)

            mesh2 = rotate_glb(mesh1, axis='z', angle_deg=180)

            dst = os.path.join(output_dir, filename)
            mesh2.export(dst)

    print(f'The GLB has been converted to Isaac coordinates!')

if __name__ == '__main__':
    input_dir = 'output_scene/scene_1/output_assets/tex_mesh'
    output_dir = 'output_scene/scene_1/output_assets/rotated_mesh'

    process_glb_to_isaac_axis(input_dir, output_dir)