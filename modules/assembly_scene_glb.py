"""
Flexible GLB scene assembly.
"""

import os
import json
import math
import numpy as np
import trimesh
from typing import Optional


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def compute_extents_m(scene: trimesh.Scene) -> np.ndarray:
    """Compute axis-aligned extents of a trimesh.Scene in meters."""
    bounds = scene.bounds
    extents = bounds[1] - bounds[0]
    return np.asarray(extents, dtype=float)


def non_uniform_scale_matrix(sx, sy, sz) -> np.ndarray:
    M = np.eye(4, dtype=float)
    M[0, 0] = sx
    M[1, 1] = sy
    M[2, 2] = sz
    return M


def rot_z_matrix(deg: float) -> np.ndarray:
    rad = math.radians(float(deg))
    c, s = math.cos(rad), math.sin(rad)
    R = np.eye(4, dtype=float)
    R[0, 0] = c; R[0, 1] = -s
    R[1, 0] = s; R[1, 1] = c
    return R


def trans_matrix(x, y, z) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, 3] = [x, y, z]
    return T


def safe_scale(extents_m: np.ndarray, target_m: np.ndarray) -> np.ndarray:
    """Compute per-axis scale; if an axis is near zero, fall back to uniform scale."""
    eps = 1e-6
    sx = target_m[0] / extents_m[0] if extents_m[0] > eps else np.nan
    sy = target_m[1] / extents_m[1] if extents_m[1] > eps else np.nan
    sz = target_m[2] / extents_m[2] if extents_m[2] > eps else np.nan

    scale = np.array([sx, sy, sz], dtype=float)
    if np.isnan(scale).any() or np.isinf(scale).any():
        valid = scale[np.isfinite(scale)]
        uni = float(np.mean(valid)) if valid.size > 0 else 1.0
        scale = np.array([uni, uni, uni], dtype=float)
    return scale


def load_object_scene(glb_path: str) -> trimesh.Scene:
    obj = trimesh.load(glb_path, force='scene')
    if isinstance(obj, trimesh.Trimesh):
        obj = trimesh.Scene(obj)
    return obj


def assemble_scene_flexible(
    scene_id: int,
    scene_dir: str = None,
    output_file: Optional[str] = None,
) -> str:
    """
    Flexible GLB scene assembly.
    """
    # default scene directory
    if scene_dir is None:
        scene_dir = os.path.join(os.getcwd(), "output_scene", f"scene_{scene_id}")
    
    pose_json = os.path.join(scene_dir, "output_assets", "layout_json", "layout_pose.json")
    rot_json = os.path.join(scene_dir, "output_assets", "layout_json", "layout_rotation.json")
    centered_mesh = os.path.join(scene_dir, "output_assets", "centered_mesh")
    
    if not os.path.isdir(centered_mesh):
        raise FileNotFoundError(f"centered_mesh not found: {centered_mesh}")
    if not os.path.isfile(pose_json):
        raise FileNotFoundError(f"layout_pose.json not found: {pose_json}")
    if not os.path.isfile(rot_json):
        raise FileNotFoundError(f"layout_rotation.json not found: {rot_json}")

    pose_data = load_json(pose_json)
    rot_data = load_json(rot_json)
    order = pose_data.get("metadata", {}).get("calculation_order", list(pose_data.get("objects", {}).keys()))

    scene_out = trimesh.Scene()

    for name in order:
        obj_info = pose_data["objects"].get(name)
        if obj_info is None:
            print(f"[warn] no pose/size for {name}, skip.")
            continue

        glb_path = os.path.join(centered_mesh, f"{name}.glb")
        if not os.path.isfile(glb_path):
            print(f"[warn] glb missing: {glb_path}, skip.")
            continue

        size_cm = np.array(obj_info["size"], dtype=float)
        pose_cm = np.array(obj_info["pose"], dtype=float)
        size_m = size_cm / 100.0
        pos_m = pose_cm / 100.0

        rot_deg = float(rot_data.get(name, 0.0))

        obj_scene = load_object_scene(glb_path)
        extents_m = compute_extents_m(obj_scene)
        scale_xyz = safe_scale(extents_m, size_m)

        T = trans_matrix(*pos_m) @ rot_z_matrix(rot_deg) @ non_uniform_scale_matrix(*scale_xyz)
        obj_scene.apply_transform(T)

        baked_mesh = obj_scene.to_mesh()
        scene_out.add_geometry(baked_mesh, geom_name=name)
        print(f"[ok] {name}: size(cm)={size_cm.tolist()} pos(cm)={pose_cm.tolist()} rotZ(deg)={rot_deg}")

    if output_file is None:
        output_file = os.path.join(scene_dir, f"scene_{scene_id}.glb")

    scene_out.export(output_file)
    print(f"[done] Exported: {output_file}\n")
    return output_file


if __name__ == "__main__":

    assemble_scene_flexible(
        scene_id=1,
        rotation_dir="output_scene/scene_1/output_assets/layout_json",
        output_file="output_scene/scene_1/scene_1.glb",
    )