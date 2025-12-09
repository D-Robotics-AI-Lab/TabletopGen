"""
Isaac Sim 3D Layout Final Module
Based on data from layout_pose.json and layout_rotation.json,
place objects from sized_mesh into Isaac Sim according to the final calculated position, size, and rotation angle.
"""

import os
import time
import json
import sys
import numpy as np
import math
import asyncio
import shutil
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import omni
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from pxr import Usd, UsdGeom, Gf, UsdPhysics, UsdLux
import omni.usd
from omni.isaac.core.utils.extensions import enable_extension

# Enable asset converter extension
enable_extension("omni.kit.asset_converter")


async def convert(in_file, out_file, load_materials=False):
    """Convert asset file to USD format"""
    import omni.kit.asset_converter

    def progress_callback(progress, total_steps):
        pass

    converter_context = omni.kit.asset_converter.AssetConverterContext()
    converter_context.ignore_materials = not load_materials

    instance = omni.kit.asset_converter.get_instance()
    task = instance.create_converter_task(in_file, out_file, progress_callback, converter_context)
    success = True
    while True:
        success = await task.wait_until_finished()
        if not success:
            await asyncio.sleep(0.1)
        else:
            break
    return success

def asset_convert(input_file, output_file):
    """Convert asset file"""
    print(f"Converting {input_file} to {output_file}...")
    status = asyncio.get_event_loop().run_until_complete(
        convert(input_file, output_file, load_materials=True)
    )
    if not status:
        print(f"Error: Failed to convert {input_file}")
        return False
    else:
        print(f"Success: Converted {input_file} to {output_file}")
        return True

def normalize_asset_name(name):
    """Smartly handle asset names"""
    words = name.split()
    capitalized_words = [word.capitalize() for word in words]
    return ''.join(capitalized_words)

def check_asset_exists(asset_path):
    """Check if asset file exists"""
    if not os.path.exists(asset_path):
        print(f"Error: Asset file '{asset_path}' does not exist")
        return False
    return True

def set_angles(prim_path, euler_angles):
    """Set object Euler angle rotation"""
    obj = XFormPrim(prim_path)
    current_position, _ = obj.get_world_pose()
    euler_angles_rad = tuple(math.radians(angle) for angle in euler_angles)
    new_orientation = euler_angles_to_quat(euler_angles_rad)
    obj.set_world_pose(position=current_position, orientation=new_orientation)

def set_position(prim_path, position):
    """Set object position"""
    obj = XFormPrim(prim_path)
    _, current_orientation = obj.get_world_pose()
    new_position = Gf.Vec3d(position[0], position[1], position[2])
    obj.set_world_pose(position=new_position, orientation=current_orientation)

def create_rigid_collision(stage, prim_path):
    """Add rigid body collision properties to object"""
    xform_parent = UsdGeom.Xform.Define(stage, prim_path)
    UsdPhysics.RigidBodyAPI.Apply(xform_parent.GetPrim())
    UsdPhysics.CollisionAPI.Apply(xform_parent.GetPrim())

def calculate_bounding_box(stage, prim_path):
    """Calculate object bounding box, return size and center point"""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        print(f"Error: Invalid prim at path {prim_path}")
        return None, None

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default'])
    bbox = bbox_cache.ComputeWorldBound(prim)
    
    if bbox:
        bounds = bbox.ComputeAlignedBox()
        min_point = bounds.GetMin()
        max_point = bounds.GetMax()
        
        size = [max_point[0] - min_point[0], 
                max_point[1] - min_point[1], 
                max_point[2] - min_point[2]]
        
        center = [(max_point[0] + min_point[0])/2, 
                 (max_point[1] + min_point[1])/2, 
                 (max_point[2] + min_point[2])/2]
        
        return size, center
    else:
        print(f"Cannot calculate bounding box for {prim_path}")
        return None, None

def set_object_scale(stage, prim_path, scale_factors):
    """Set object scale"""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"Error: Invalid prim at path {prim_path}")
        return

    xform = UsdGeom.Xformable(prim)
    scale_op = None

    for op in xform.GetOrderedXformOps():
        if op.GetOpName() == 'xformOp:scale':
            scale_op = op
            break

    if scale_op is None:
        scale_op = xform.AddScaleOp()

    scale_op.Set(Gf.Vec3f(*scale_factors))
    print(f"Applied scale {scale_factors} to {prim_path}")

# new
from pxr import Usd, UsdGeom, UsdPhysics

def _physics_token(name, default=None):
    return getattr(UsdPhysics.Tokens, name, default)

# Dynamically build "best available" mapping
APPROX_CANDIDATES = {
    "box":               _physics_token("boundingCube"),
    "sphere":            _physics_token("boundingSphere"),
    "hull":              _physics_token("convexHull"),
    "vhacd":             _physics_token("convexDecomposition"),
    "mesh_simplify":     _physics_token("meshSimplification"),
    "none":              _physics_token("none"),
}
# Filter out None values
APPROX_MAP = {k: v for k, v in APPROX_CANDIDATES.items() if v is not None}

def print_supported_collision_approximations():
    print("[collider] Supported approximation tokens in this build:")
    for k, v in APPROX_MAP.items():
        print(f"  - {k:>13} -> {v}")

def set_collision_approx_for_submeshes(stage, root_prim_path, mode="box"):
    """Apply collision approximation to all Meshes under the asset (default bounding box)"""
    if mode not in APPROX_MAP:
        # Fallback priority: box -> hull -> vhacd -> sphere -> mesh_simplify -> none
        for fallback in ["box", "hull", "vhacd", "sphere", "mesh_simplify", "none"]:
            if fallback in APPROX_MAP:
                print(f"[collider] mode '{mode}' not supported; fallback to '{fallback}'")
                mode = fallback
                break

    approx_token = APPROX_MAP[mode]
    root = stage.GetPrimAtPath(root_prim_path)
    if not root or not root.IsValid():
        print(f"[collider] invalid prim: {root_prim_path}")
        return

    count = 0
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Mesh):
            coll = UsdPhysics.CollisionAPI.Apply(prim)
            # set physics:approximation
            coll.GetPhysicsApproximationAttr().Set(approx_token)
            count += 1
    print(f"[collider] set '{mode}' on {count} mesh prim(s) under {root_prim_path}")



def add_object_to_scene(sized_mesh_path, usd_cache_dir, stage, object_name, object_data, rotation_angle ,is_main_object):
    """Add object to the scene"""
    # Clean object name, remove special characters
    import re
    clean_object_name = re.sub(r'[^\w\s-]', '', object_name)
    clean_object_name = re.sub(r'^#+', '', clean_object_name).strip()
    
    if not clean_object_name:
        clean_object_name = f"Asset_{hash(object_name) % 10000}"
    
    print(f"Original name: '{object_name}' -> Cleaned name: '{clean_object_name}'")
    

    # Create unique prim path
    normalized_name = normalize_asset_name(clean_object_name)
    prim_path = f"/World/{normalized_name}"
    
    # GLB file path
    glb_path = os.path.join(sized_mesh_path, f"{object_name}.glb")
    
    # Check if GLB file exists
    if not check_asset_exists(glb_path):
        print(f"Skipping asset '{object_name}', file not found: {glb_path}")
        return False
    
    # USD file path
    usd_path = os.path.join(usd_cache_dir, f"{clean_object_name}.usd")
    
    # If USD file does not exist or GLB file is updated, perform conversion
    if not os.path.exists(usd_path) or os.path.getmtime(glb_path) > os.path.getmtime(usd_path):
        if not asset_convert(glb_path, usd_path):
            print(f"Cannot convert asset '{object_name}' to USD format")
            return False
    
    print(f"Adding asset: {object_name}")
    
    try:
        # Step 1: Add USD file to scene
        add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
        print(f"Added asset '{object_name}' to scene")
        
        import omni.kit.app
        omni.kit.app.get_app().update()
        
        # Step 2: Get current size and calculate required scale
        actual_size, actual_center = calculate_bounding_box(stage, prim_path)
        if actual_size:
            expected_size_cm = object_data["size"]  # Expected size in cm
            expected_size_m = [s / 100.0 for s in expected_size_cm]  # Convert to meters
            
            print(f"Expected size (cm): {expected_size_cm}")
            print(f"Expected size (m): {[round(s, 4) for s in expected_size_m]}")
            print(f"Current size (Isaac units): {[round(s, 4) for s in actual_size]}")
            
            # Calculate scale factor: target size (m) / current size
            if actual_size[0] > 0 and actual_size[1] > 0 and actual_size[2] > 0:
                scale_factors = [
                    expected_size_m[0] / actual_size[0],
                    expected_size_m[1] / actual_size[1], 
                    expected_size_m[2] / actual_size[2]
                ]
                
                print(f"Scale factors: {[round(s, 4) for s in scale_factors]}")
                
                # Apply scale
                set_object_scale(stage, prim_path, scale_factors)
                
                # Recalculate scaled size for verification
                omni.kit.app.get_app().update()
                final_size, final_center = calculate_bounding_box(stage, prim_path)
                if final_size:
                    print(f"Final size (m): {[round(s, 4) for s in final_size]}")
        
        # Step 3: Set position (position in layout_pose is already final, convert to meters)
        pose_position = object_data["pose"]  # [x, y, z] in cm
        
        # Convert to Isaac Sim coordinate system: invert x-axis, cm to m, and adjust height to fit room floor
        room_floor_offset = -0.7696  # Room floor height offset
        adjusted_position = [
            pose_position[0] / 100.0,  # invert x-axis, cm to m
            pose_position[1] / 100.0,   # y-axis, cm to m
            pose_position[2] / 100.0 + room_floor_offset    # z-axis, cm to m and subtract floor offset
        ]
        
        print(f"Layout pose (cm): {pose_position}")
        print(f"Adjusted position (m, with room floor offset): {[round(p, 4) for p in adjusted_position]}")
        
        set_position(prim_path, adjusted_position)
        
        # Verify position after setting
        obj = XFormPrim(prim_path)
        final_position, _ = obj.get_world_pose()
        print(f"Final position: {[round(p, 4) for p in final_position]}")
        
        # Step 4: Set rotation angle
        if rotation_angle != 0:
            actual_rotation_angle = rotation_angle
            rotation = [0, 0, actual_rotation_angle]
            set_angles(prim_path, rotation)
            print(f"Applied rotation: {actual_rotation_angle}° around Z-axis (original: {rotation_angle}°)")
        
        # Step 5: Add physical collision
        if is_main_object:
            print_supported_collision_approximations()

            create_rigid_collision(stage, prim_path)
        else:
            create_rigid_collision(stage, prim_path)

        print(f"{object_name} successfully placed")
        return True
        
    except Exception as e:
        print(f"Error adding asset '{object_name}': {e}")
        import traceback
        traceback.print_exc()
        return False

def load_layout_data(layout_pose_path, layout_rotation_path):
    """Load layout data"""
    try:
        # Load pose data
        with open(layout_pose_path, 'r', encoding='utf-8') as f:
            pose_data = json.load(f)
        
        # Load rotation data
        with open(layout_rotation_path, 'r', encoding='utf-8') as f:
            rotation_data = json.load(f)
        
        print(f"Loaded layout data:")
        print(f"  - Pose data for {len(pose_data['objects'])} objects")
        print(f"  - Rotation data for {len(rotation_data)} entries")
        print(f"  - Main object: {pose_data['main_object']}")
        
        return pose_data, rotation_data
        
    except Exception as e:
        print(f"Error loading layout data: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def clear_usd_cache(usd_cache_dir):
    """Clear USD cache directory"""
    if os.path.exists(usd_cache_dir):
        try:
            shutil.rmtree(usd_cache_dir)
            print(f"Cleared USD cache directory: {usd_cache_dir}")
        except Exception as e:
            print(f"Warning: Failed to clear USD cache directory: {e}")
    
    # Recreate directory
    os.makedirs(usd_cache_dir, exist_ok=True)
    print(f"Created fresh USD cache directory: {usd_cache_dir}")

def load_room_environment(stage, pipeline_dir):
    """Load room environment USD file"""
    room_usd_path = os.path.join(pipeline_dir, "background_room/room.usd")
    
    if not os.path.exists(room_usd_path):
        print(f"Warning: Room USD file not found at {room_usd_path}")
        return False
    
    try:
        # Check USD file content first
        print(f"Checking room USD file: {room_usd_path}")
        temp_stage = Usd.Stage.Open(room_usd_path)
        if temp_stage:
            root_prim = temp_stage.GetDefaultPrim()
            if root_prim:
                print(f"Room USD default prim: {root_prim.GetPath()}")
            else:
                print("Room USD has no default prim")
            
            # List all prims
            all_prims = [prim for prim in temp_stage.Traverse()]
            print(f"Room USD contains {len(all_prims)} prims:")
            for prim in all_prims[:10]:  # Show only first 10
                print(f"  - {prim.GetPath()} (type: {prim.GetTypeName()})")
            if len(all_prims) > 10:
                print(f"  ... and {len(all_prims) - 10} more prims")
        
        # Method 1: Use add_reference_to_stage
        room_prim_path = "/World/Room"
        print(f"Trying to load room with add_reference_to_stage...")
        add_reference_to_stage(usd_path=room_usd_path, prim_path=room_prim_path)
        
        import omni.kit.app
        omni.kit.app.get_app().update()
        
        # Check load result
        room_prim = stage.GetPrimAtPath(room_prim_path)
        if room_prim and room_prim.IsValid():
            print(f"Room prim created at {room_prim_path}")
            # Check child prims
            children = room_prim.GetChildren()
            print(f"Room has {len(children)} children:")
            for child in children[:5]:  # Show only first 5
                print(f"  - {child.GetPath()} (type: {child.GetTypeName()})")
            
            if len(children) == 0:
                print("Room prim has no children, trying alternative method...")
                
                # Method 2: Merge stage directly
                try:
                    stage.GetRootLayer().subLayerPaths.append(room_usd_path)
                    omni.kit.app.get_app().update()
                    print("Tried sublayer method")
                except Exception as e2:
                    print(f"Sublayer method failed: {e2}")
                    
                    # Method 3: Use payload
                    try:
                        room_prim.GetPayloads().AddPayload(room_usd_path)
                        omni.kit.app.get_app().update()
                        print("Tried payload method")
                    except Exception as e3:
                        print(f"Payload method failed: {e3}")
        
        print(f"Successfully processed room environment from {room_usd_path}")
        return True
        
    except Exception as e:
        print(f"Error loading room environment: {e}")
        import traceback
        traceback.print_exc()
        return False

def isaac_main(output_assets_dir):
    """Main function"""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    pipeline_dir = script_dir

    sized_mesh_path = os.path.join(output_assets_dir, "centered_mesh")
    layout_pose_path = os.path.join(output_assets_dir, "layout_json", "layout_pose.json")
    layout_rotation_path = os.path.join(output_assets_dir, "layout_json", "layout_rotation.json")

    # Create USD cache directory and clear it
    usd_cache_dir = os.path.join(script_dir, "usd_cache")
    clear_usd_cache(usd_cache_dir)

    print("Isaac Sim 3D Layout Final System")
    print(f"Script directory: {script_dir}")
    print(f"Pipeline directory: {pipeline_dir}")
    print(f"Room USD path: {os.path.join(pipeline_dir, 'room.usd')}")
    print(f"Sized mesh path: {sized_mesh_path}")
    print(f"Layout pose path: {layout_pose_path}")
    print(f"Layout rotation path: {layout_rotation_path}")
    print(f"USD cache directory: {usd_cache_dir}")
    
    # Load layout data
    pose_data, rotation_data = load_layout_data(layout_pose_path, layout_rotation_path)
    if not pose_data or not rotation_data:
        print("Failed to load layout data")
        return
    
    try:
        # Initialize world and physics engine
        my_world = World(stage_units_in_meters=1.0, physics_prim_path="/World/physicsScene")
        stage = omni.usd.get_context().get_stage()
        
        # Load room environment
        print("Loading room environment...")
        if load_room_environment(stage, pipeline_dir):
            print("Room environment loaded successfully")
        else:
            print("Failed to load room environment, adding default ground plane")
            my_world.scene.add_default_ground_plane()
        
        print(f"World stage units: {get_stage_units()} meters per unit")
        
        # Add objects in calculation order
        calculation_order = pose_data["metadata"]["calculation_order"]
        objects_data = pose_data["objects"]
        main_object = pose_data["main_object"]
        
        successful_assets = 0
        is_main_object = False
        
        for object_name in calculation_order:
            if object_name not in objects_data:
                print(f"Warning: {object_name} not found in objects data")
                continue

            if object_name == main_object:
                is_main_object = True
                print(f"\n*** Processing main object: {object_name} ***")
            
            object_data = objects_data[object_name]
            
            # Get rotation angle, skip non-object data like view_angle
            rotation_angle = rotation_data.get(object_name, 0)
            if object_name == "view_angle":
                continue
            
            print(f"\n{'='*60}")
            print(f"Processing object: {object_name}")
            print(f"Size: {object_data['size']} cm")
            print(f"Pose: {object_data['pose']} cm")
            print(f"Rotation angle: {rotation_angle}°")


            
            if add_object_to_scene(sized_mesh_path, usd_cache_dir, stage, object_name, object_data, rotation_angle, is_main_object):
                successful_assets += 1
            
            print(f"{'='*60}")

            is_main_object = False  # Reset flag
        
        print(f"\nSuccessfully added {successful_assets}/{len(calculation_order)} assets")
        print("All assets positioned on room floor (Z offset: -0.7696m)")
        
        # Add lights
        # Dome light
        dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
        dome_light.CreateIntensityAttr(1000)
        
        # Directional light
        distant_light = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
        distant_light.CreateIntensityAttr(500)
        distant_light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
        
        print("\nStarting simulation...")
        print("Use the viewport camera to navigate and check object placement")
        print("Layout based on VLM analysis with calculated stacking relationships")
        
        # Run simulation
        for i in range(55):
            for j in range(5000):
                my_world.step(render=True)
                time.sleep(0.01)
        
        # Wait for user input, keep window open
        input("Press Enter to exit...")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Close simulation
        simulation_app.close()

if __name__ == "__main__":
    output_assets_dir = 'output_scene/scene_1/output_assets'
    isaac_main(output_assets_dir)