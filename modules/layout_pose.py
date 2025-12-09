"""
Calculate final layout pose.
"""

import json
import os

def load_layout_data(output_assets_dir):
    """Load required layout data"""

    placement_order_path = os.path.join(output_assets_dir, "layout_json", "placement_order.json")
    final_vlm_pose_path = os.path.join(output_assets_dir, "layout_json", "final_vlm_pose.json")
    size_path = os.path.join(output_assets_dir, "layout_json", "size.json")
    
    with open(placement_order_path, 'r', encoding='utf-8') as f:
        placement_data = json.load(f)
    
    with open(final_vlm_pose_path, 'r', encoding='utf-8') as f:
        vlm_pose_data = json.load(f)
    
    with open(size_path, 'r', encoding='utf-8') as f:
        size_data = json.load(f)
    
    return placement_data, vlm_pose_data, size_data

def find_main_object(placement_data):
    """Find main object"""
    for obj_name, obj_info in placement_data.items():
        if obj_info.get("order") == 1 and obj_info.get("on_top_of") is None:
            return obj_name
    return None

def calculate_object_z_position(obj_name, placement_data, vlm_pose_data, size_data, calculated_poses):
    """Recursively calculate object Z position"""
    obj_info = placement_data[obj_name]
    on_top_of = obj_info.get("on_top_of")
    
    if on_top_of is None:
        on_top_of_list = []
    elif isinstance(on_top_of, str):
        on_top_of_list = [on_top_of]
    else:
        on_top_of_list = on_top_of
    
    if not on_top_of_list:
        # If not placed on other objects, return half of its own height
        return calculated_poses[obj_name]["size"][2] / 2
    
    # If placed on multiple objects, take the highest Z position
    max_base_z = 0
    for base_obj in on_top_of_list:
        if base_obj in calculated_poses:
            # Base object top position = base object z + base object height / 2
            base_top_z = calculated_poses[base_obj]["pose"][2] + calculated_poses[base_obj]["size"][2] / 2
            max_base_z = max(max_base_z, base_top_z)
        else:
            # If base object not calculated yet, calculate recursively first
            calculate_object_pose(base_obj, placement_data, vlm_pose_data, size_data, calculated_poses)
            base_top_z = calculated_poses[base_obj]["pose"][2] + calculated_poses[base_obj]["size"][2] / 2
            max_base_z = max(max_base_z, base_top_z)
    
    # Current object Z position = highest base object top + own height / 2
    current_obj_height = calculated_poses[obj_name]["size"][2]
    return max_base_z + current_obj_height / 2

def calculate_object_pose(obj_name, placement_data, vlm_pose_data, size_data, calculated_poses):
    """Calculate pose for a single object"""
    if obj_name in calculated_poses:
        return
    
    if obj_name == vlm_pose_data.get("main_object"):
        # Main object uses adjusted size
        obj_size = vlm_pose_data["adjusted_main_size"]
    else:
        obj_size = size_data[obj_name]
    
    # Get XY position
    if obj_name in vlm_pose_data["poses"]:
        vlm_xy = vlm_pose_data["poses"][obj_name]
        obj_x, obj_y = vlm_xy[0], vlm_xy[1]
    else:
        print(f"Warning: {obj_name} not found in vlm_pose, using default position [0,0]")
        obj_x, obj_y = 0.0, 0.0
    
    # Set size first, used for Z position calculation
    calculated_poses[obj_name] = {
        "size": obj_size,
        "pose": [obj_x, obj_y, 0]
    }
    
    # Calculate Z position
    obj_z = calculate_object_z_position(obj_name, placement_data, vlm_pose_data, size_data, calculated_poses)
    calculated_poses[obj_name]["pose"][2] = obj_z
    
    print(f"{obj_name}: size={obj_size}, pose=[{obj_x:.2f}, {obj_y:.2f}, {obj_z:.2f}]")

def calculate_final_layout(output_assets_dir):
    """Calculate final layout"""
    try:
        print("=== Start calculating final layout pose ===")
        
        print("\n1. Loading layout data...")
        placement_data, vlm_pose_data, size_data = load_layout_data(output_assets_dir)
        
        print("\n2. Identifying main object...")
        main_object = find_main_object(placement_data)
        if not main_object:
            print("Error: Unable to find main object")
            return None
        print(f"Main object: {main_object}")
        
        print("\n3. Calculating object poses by placement order...")
        
        sorted_objects = sorted(placement_data.items(), key=lambda x: x[1].get("order", 999))
        
        calculated_poses = {}
        
        # Calculate each object in order
        for obj_name, _ in sorted_objects:

            calculate_object_pose(obj_name, placement_data, vlm_pose_data, size_data, calculated_poses)
        
        # 4. Build final result
        final_layout = {
            "main_object": main_object,
            "objects": calculated_poses,
            "metadata": {
                "total_objects": len(calculated_poses),
                "main_object_size": calculated_poses[main_object]["size"],
                "calculation_order": [obj[0] for obj in sorted_objects]
            }
        }
        
        # 5. Save result
        output_dir = os.path.join(output_assets_dir, "layout_json")
        output_path = os.path.join(output_dir, "layout_pose.json")
        os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_layout, f, indent=4, ensure_ascii=False)
        
        print(f"\n=== Calculation Complete ===")
        print(f"✓ Final layout saved to: {output_path}")
        
        # 6. Output result summary
        print("\n=== Final Layout Result Summary ===")
        for obj_name, obj_data in calculated_poses.items():
            size = obj_data["size"]
            pose = obj_data["pose"]
            print(f"{obj_name}:")
            print(f"  Size: [{size[0]:.1f}, {size[1]:.1f}, {size[2]:.1f}] cm")
            print(f"  Position: [{pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.2f}] cm")
        
        # 7. Validate layout rationality
        print("\n=== Layout Validation ===")
        validate_layout(calculated_poses, placement_data)
        
        return final_layout
        
    except Exception as e:
        print(f"Failed to calculate final layout: {e}")
        import traceback
        traceback.print_exc()
        return None

def validate_layout(calculated_poses, placement_data):
    """Validate layout rationality"""
    try:
        
        for obj_name, obj_info in placement_data.items():
            on_top_of = obj_info.get("on_top_of")
            # Normalize to list
            if on_top_of is None:
                on_top_of_list = []
            elif isinstance(on_top_of, str):
                on_top_of_list = [on_top_of]
            else:
                on_top_of_list = on_top_of

            if not on_top_of_list:
                continue
            
            obj_z = calculated_poses[obj_name]["pose"][2]
            obj_height = calculated_poses[obj_name]["size"][2]
            obj_bottom = obj_z - obj_height / 2
            
            for base_obj in on_top_of_list:
                if base_obj not in calculated_poses:
                    print(f"Warning: Base object {base_obj} not in calculated results, skipping validation")
                    continue
                base_z = calculated_poses[base_obj]["pose"][2]
                base_height = calculated_poses[base_obj]["size"][2]
                base_top = base_z + base_height / 2
                
                if obj_bottom < base_top - 0.1:  # Allow 0.1cm error
                    print(f"Warning: {obj_name} may overlap with {base_obj}")
                else:
                    print(f"✓ {obj_name} correctly placed on top of {base_obj}")
        
        print("Layout validation complete")
        
    except Exception as e:
        print(f"Layout validation failed: {e}")


if __name__ == "__main__":
    try:
        output_assets_dir = "output_scene/scene_1/output_assets"
        final_layout = calculate_final_layout(output_assets_dir)
        if final_layout:
            print("\n✓ Final layout calculation successful!")
        else:
            print("\n✗ Final layout calculation failed!")
    
    except Exception as e:
        print(f"Main function execution failed: {e}")
        import traceback
        traceback.print_exc()
