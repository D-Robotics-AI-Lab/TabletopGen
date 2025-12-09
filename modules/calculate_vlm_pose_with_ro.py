"""
Calculate the actual XY placement pose for each object.
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import trimesh

GLB_UNIT_TO_CM = 1 # GLB unit -> cm conversion: meter -> cm = 100; if GLB is already in cm, change to 1
REF_INFO = {"name": None, "w_cm": None, "h_cm": None} # Record reference object and its rotated XY dimensions (cm)

def load_json_files(output_assets_dir):
    """Load required JSON files"""

    segmentation_path = os.path.join(output_assets_dir, "image", "segmentation_results.json")
    size_path = os.path.join(output_assets_dir, "layout_json", "size.json")
    boxes_path = os.path.join(output_assets_dir, "layout_json", "doubao_detection_boxes.json")
    rotation_path = os.path.join(output_assets_dir, "layout_json", "layout_rotation.json")
    

    with open(segmentation_path, 'r', encoding='utf-8') as f:
        segmentation_data = json.load(f)
    
    with open(size_path, 'r', encoding='utf-8') as f:
        size_data = json.load(f)
    
    with open(boxes_path, 'r', encoding='utf-8') as f:
        boxes_data = json.load(f)
    
    # Load rotation data
    try:
        with open(rotation_path, 'r', encoding='utf-8') as f:
            rotation_data = json.load(f)
    except FileNotFoundError:
        print(f"Warning: Rotation data file does not exist: {rotation_path}")
        rotation_data = {}
    
    return segmentation_data, size_data, boxes_data, rotation_data

def find_main_object(segmentation_data):
    """Extract main object name"""
    main_object = segmentation_data.get("main_object", "table")
    
    # Find the full name of the main object (including index)
    for object_name in segmentation_data.get("object_names", []):
        if object_name.startswith(main_object):
            return object_name
    
    return None

# ---------- 1) Fuse GLB scene "with node transformations" into a world coordinate mesh ----------
def load_fused_mesh(glb_path: str) -> "trimesh.Trimesh":
    """
    Read .glb and apply all node transformations (including scale/rotation/translation) to geometry,
    finally returning a fused Trimesh (world coordinates).
    """
    # Try to get mesh directly first (some glbs can be unwrapped at once)
    obj = trimesh.load(glb_path, force="mesh")
    if isinstance(obj, trimesh.Trimesh):
        return obj

    # Otherwise handle as scene, manually apply node transformations in scene.graph
    scene = trimesh.load(glb_path, force="scene")
    meshes = []
    mapping = getattr(scene.graph, "nodes_geometry", {})

    if isinstance(mapping, dict):
        pairs = list(mapping.items())
    else:
        pairs = [(n, scene.graph.nodes_geometry[n]) for n in scene.graph.nodes_geometry]

    for node_name, geom_name in pairs:
        if geom_name not in scene.geometry:
            continue
        m = scene.geometry[geom_name].copy()
        # Get world transformation matrix from root to this node
        try:
            T = scene.graph.get(node_name)[0] 
        except Exception:
            T = scene.graph.transforms.node_to_world(node_name)
        m.apply_transform(np.asarray(T))
        meshes.append(m)

    if not meshes:
        return trimesh.util.concatenate(tuple(scene.geometry.values()))

    return trimesh.util.concatenate(meshes)

def measure_rotated_xy_from_glb(glb_dir: str, obj_name: str, rotation_deg: float):
    """
    Read GLB -> Apply scene transformations -> Rotate around Z axis by rotation_deg (top-down task),
    Get width and height of AABB (axis-aligned bounding box) in XY plane (unit: cm).
    """
    glb_path = os.path.join(glb_dir, f"{obj_name}.glb")
    mesh = load_fused_mesh(glb_path)

    # Rotate around Z axis
    R = trimesh.transformations.rotation_matrix(np.deg2rad(rotation_deg), [0, 0, 1])
    mesh = mesh.copy()
    mesh.apply_transform(R)

    xy = mesh.vertices[:, :2]
    w = float(xy[:, 0].max() - xy[:, 0].min())
    h = float(xy[:, 1].max() - xy[:, 1].min())
    return w * GLB_UNIT_TO_CM, h * GLB_UNIT_TO_CM


def find_reference_object(
    glb_dir,
    size_data,
    boxes_data,
    rotation_data,
    main_object_name,
    tau: float = 0.25,
    alpha: float = 10.0 
):
    best = None
    best_score = -1.0
    best_w_cm, best_h_cm = None, None

    for obj_name in boxes_data.keys():
        if obj_name == main_object_name:
            continue
        # Skip if GLB cannot be read
        try:
            rot = float(rotation_data.get(obj_name, 0.0))
            w_cm, h_cm = measure_rotated_xy_from_glb(glb_dir, obj_name, rot)
        except Exception as e:
            print(f"[Ref skip] {obj_name}: {e}")
            continue

        # Detection box size (pixels)
        x1, y1, x2, y2 = boxes_data[obj_name]
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        area_px = bw * bh

        # Aspect ratio: use logarithmic difference & insensitive to 90° swap
        ratio_glb = max(w_cm, 1e-6) / max(h_cm, 1e-6)
        ratio_img = bw / bh
        # Two correspondences: glb(W/H) vs img(W/H), and glb(W/H) vs img(H/W)
        d1 = abs(np.log(ratio_glb) - np.log(ratio_img))
        ratio_err = d1

        angle_off = 0.0

        # Score
        score = area_px / (1.0 + (ratio_err / max(tau, 1e-6))**2 + (angle_off / max(alpha, 1e-6))**2)

        print(f"[Ref-candidate] {obj_name}: rot={rot}°  GLB_xy=({w_cm:.2f},{h_cm:.2f})cm  "
              f"box=({bw:.1f},{bh:.1f})px  area={area_px:.0f}  "
              f"ratio_glb={ratio_glb:.3f} ratio_img={ratio_img:.3f}  "
              f"ratio_err={ratio_err:.3f}  score={score:.2f}")

        if score > best_score:
            best_score = score
            best = obj_name
            best_w_cm, best_h_cm = w_cm, h_cm

    if best is None:
        print("Warning: No usable reference object found (glb missing or read failed), fallback to original logic 'max area'.")
        max_area, reference_object = 0.0, None
        for obj_name, _ in size_data.items():
            if obj_name == main_object_name or obj_name not in boxes_data:
                continue
            x1, y1, x2, y2 = boxes_data[obj_name]
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                reference_object = obj_name
        REF_INFO.update({"name": reference_object, "w_cm": None, "h_cm": None})
        print(f"Reference object (fallback): {reference_object}, area={max_area}")
        return reference_object

    REF_INFO.update({"name": best, "w_cm": best_w_cm, "h_cm": best_h_cm})
    print(f"Reference object chosen: {best}  score={best_score:.2f}  "
          f"GLB_xy=({best_w_cm:.2f},{best_h_cm:.2f})cm")
    return best


def calculate_scale_ratio(reference_object, size_data, boxes_data):
    """
    Calculate scaling using reference object's 'GLB rotated XY dimensions (cm)' and 'detection box pixel dimensions'.
    """
    if reference_object not in boxes_data:
        print(f"Warning: no bbox for reference {reference_object}, use 1.0")
        return 1.0, 1.0

    x1, y1, x2, y2 = boxes_data[reference_object]
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)

    # Prioritize GLB xy measured in find_reference_object; if not available, fallback to size.json (rectangle approximation)
    w_cm = REF_INFO.get("w_cm")
    h_cm = REF_INFO.get("h_cm")
    if w_cm is None or h_cm is None:
        print("Note: REF_INFO has no glb measurement result, fallback to using [L,W] from size.json as approximation.")
        if reference_object in size_data:
            w_cm = float(size_data[reference_object][0])
            h_cm = float(size_data[reference_object][1])
        else:
            print("Warning: reference has no size, use 1.0")
            return 1.0, 1.0

    scale_x = w_cm / bw
    scale_y = h_cm / bh

    print(f"Reference {reference_object}: GLB_xy=({w_cm:.2f},{h_cm:.2f})cm "
          f"bbox=({bw:.1f},{bh:.1f})px  ->  scale_x={scale_x:.6f} cm/px, scale_y={scale_y:.6f} cm/px")
    
    if scale_x >= scale_y:
        return scale_x, scale_x
    else:
        return scale_y, scale_y

def adjust_main_object_size(main_object_name, size_data, boxes_data, scale_x, scale_y):
    """
    Directly multiply table bbox(px) by (cm/px) to get real dimensions (anisotropic);
    Height is also scaled synchronously by XY average scaling factor, and z is saved as integer.
    """
    if main_object_name not in boxes_data:
        print(f"Warning: no bbox for main {main_object_name}, keep original")
        return size_data[main_object_name]

    x1, y1, x2, y2 = boxes_data[main_object_name]
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)

    adjusted_length = bw * float(scale_x)
    adjusted_width  = bh * float(scale_y)

    # Original size (from size_data, no robustness check)
    original_size = size_data[main_object_name]
    original_length = float(original_size[0])
    original_width  = float(original_size[1])
    original_height = float(original_size[2])

    # Calculate scaling factors for XY, take average as Z scaling factor (no robustness check)
    sx_len = adjusted_length / original_length
    sy_wid = adjusted_width  / original_width
    scale_avg = (sx_len + sy_wid) / 2.0

    adjusted_height = int(round(original_height * scale_avg))  # z saved as integer
    adjusted_size = [adjusted_length, adjusted_width, adjusted_height]

    print(f"Main {main_object_name}: bbox=({bw:.1f},{bh:.1f})px  "
          f"* (sx,sy)=({scale_x:.6f},{scale_y:.6f}) -> "
          f"{adjusted_length:.2f} x {adjusted_width:.2f} cm "
          f"(H: {original_height:.1f} -> {adjusted_height} cm, scale_z≈{scale_avg:.4f})")

    return adjusted_size


def calculate_object_poses(main_object_name, boxes_data, scale_x, scale_y):
    """Calculate pose of each object relative to main object center"""
    if main_object_name not in boxes_data:
        print(f"Error: Bounding box data for main object {main_object_name} does not exist")
        return {}
    
    # Get main object bounding box center point
    main_box = boxes_data[main_object_name]
    main_center_x = (main_box[0] + main_box[2]) / 2
    main_center_y = (main_box[1] + main_box[3]) / 2
    
    print(f"Main object center (pixels): ({main_center_x:.1f}, {main_center_y:.1f})")
    
    poses = {}
    
    for obj_name, box in boxes_data.items():
        # Calculate object center point
        obj_center_x = (box[0] + box[2]) / 2
        obj_center_y = (box[1] + box[3]) / 2
        
        # Calculate offset relative to main object center (pixels)
        offset_x_px = obj_center_x - main_center_x
        offset_y_px = obj_center_y - main_center_y
        
        # Convert to actual coordinates (cm), invert x-axis
        actual_x = -offset_x_px * scale_x  # x-axis inverted
        actual_y = offset_y_px * scale_y
        
        poses[obj_name] = [round(actual_x, 2), round(actual_y, 2)]
        
        print(f"{obj_name}: Pixel offset ({offset_x_px:.1f}, {offset_y_px:.1f}) -> Actual position ({actual_x:.2f}, {actual_y:.2f}) cm")
    
    return poses

def create_rotated_rectangle(center_x, center_y, width, height, angle_degrees):
    """Create rotated rectangle vertex coordinates"""
    angle_rad = np.radians(-angle_degrees)
    
    # Four corners of the rectangle (relative to center point)
    corners = np.array([
        [-width/2, -height/2],  # Bottom left
        [width/2, -height/2],   # Bottom right
        [width/2, height/2],    # Top right
        [-width/2, height/2]    # Top left
    ])
    
    # Rotation matrix
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    
    # Apply rotation
    rotated_corners = corners @ rotation_matrix.T
    
    # Translate to actual position
    rotated_corners[:, 0] += center_x
    rotated_corners[:, 1] += center_y
    
    return rotated_corners

def calculate_pose_main(output_assets_dir):
    """Main function"""
    try:
        print("=== Start calculating object pose ===")
        
        # 1. Load data
        print("\n1. Loading JSON files...")
        segmentation_data, size_data, boxes_data, rotation_data = load_json_files(output_assets_dir)
        glb_dir = os.path.join(output_assets_dir, "sized_mesh")
        
        # 2. Identify main object
        print("\n2. Identifying main object...")
        main_object_name = find_main_object(segmentation_data)
        if not main_object_name:
            print("Error: Unable to find main object")
            return
        print(f"Main object: {main_object_name}")
        
        # 3. Find reference object
        print("\n3. Finding reference object...")
        reference_object = find_reference_object(glb_dir,size_data, boxes_data, rotation_data, main_object_name)
        if not reference_object:
            print("Error: Unable to find reference object")
            return
        
        # 4. Calculate scaling ratio
        print("\n4. Calculating scaling ratio...")
        scale_x, scale_y = calculate_scale_ratio(reference_object, size_data, boxes_data)
        
        # 5. Adjust main object size
        print("\n5. Adjusting main object size...")
        adjusted_main_size = adjust_main_object_size(main_object_name, size_data, boxes_data, scale_x, scale_y)
        
        # 6. Calculate object poses
        print("\n6. Calculating object poses...")
        poses = calculate_object_poses(main_object_name, boxes_data, scale_x, scale_y)
        
        # 7. Build final result
        final_vlm_pose = {
            "main_object": main_object_name,
            "reference_object": reference_object,
            "scale_ratio": {
                "x": round(scale_x, 6),
                "y": round(scale_y, 6)
            },
            "adjusted_main_size": [round(x, 2) for x in adjusted_main_size],
            "poses": poses
        }
        
        # 8. Save result
        output_dir = os.path.join(output_assets_dir, "layout_json")
        output_path = os.path.join(output_dir, "final_vlm_pose.json")
        os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_vlm_pose, f, indent=2, ensure_ascii=False)
        
        print(f"\n=== Calculation Complete ===")
        print(f"✓ Result saved to: {output_path}")
        print(f"✓ Main object adjusted size: {[round(x, 2) for x in adjusted_main_size]}")
        print(f"✓ Scaling ratio: x={scale_x:.4f}, y={scale_y:.4f} cm/px")
        print(f"✓ Calculated pose for {len(poses)} objects")
        
        # Print result summary
        print("\n=== Pose Result Summary ===")
        for obj_name, pose in poses.items():
            print(f"{obj_name}: ({pose[0]:>7.2f}, {pose[1]:>7.2f}) cm")
        
            
    except Exception as e:
        print(f"Error during calculation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    output_assets_dir = "output_scene/scene_1/output_assets"
    calculate_pose_main(output_assets_dir)
