"""
Run the independent script for rotation estimation in the rotation virtual environment.
"""
import sys
import os

def main():
    if len(sys.argv) != 5:
        print("Usage: python run_rotation_estimation.py <output_assets_dir> <api_key> <proxy_url> <base_url>")
        sys.exit(1)
    
    output_assets_dir = sys.argv[1]
    api_key = sys.argv[2]
    proxy_url = sys.argv[3]
    base_url = sys.argv[4]
    
    if not os.path.exists(output_assets_dir):
        print(f"Error: output_assets_dir does not exist: {output_assets_dir}")
        sys.exit(1)
    
    try:
        from layout_rotation_estimate import layout_rotation_estimate_main
        
        print(f"Starting rotation estimation in the rotation environment...")
        print(f"Output directory: {output_assets_dir}")
        
        # Run rotation estimation
        layout_rotation_estimate_main(output_assets_dir, api_key, proxy_url, base_url)
        
        print("Rotation estimation completed!")
        
    except Exception as e:
        print(f"Rotation estimation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
