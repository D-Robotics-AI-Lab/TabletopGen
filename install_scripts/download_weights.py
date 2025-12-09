import os
import requests
import sys
from tqdm import tqdm


from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent

def download_file(url, relative_path):
    save_path = ROOT_DIR / relative_path
    
    if save_path.exists():
        print(f"[Skip] File already exists: {save_path.name}")
        return

    print(f"Downloading {save_path.name}...")
    
    os.makedirs(save_path.parent, exist_ok=True)
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status() # Check if the connection is successful
        total_size = int(response.headers.get('content-length', 0))
        
        with open(save_path, 'wb') as file, tqdm(
            desc=save_path.name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
                
    except Exception as e:
        print(f"\n[Error] Failed to download {save_path.name}: {e}")
        # If download fails, remove potentially corrupted file
        if save_path.exists():
            os.remove(save_path)
        sys.exit(1)

# ============================================================
# 🔗 Weights List
# ============================================================
weights_to_download = [
    # --- BiRefNet Weights ---
    {
        "url": "https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/BiRefNet-general-epoch_244.pth",
        "path": "BiRefNet/pth/BiRefNet-general-epoch_244.pth"
    },

    # --- Grounded-SAM-2: SAM 2.1 Checkpoints ---
    {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
        "path": "Grounded-SAM-2/checkpoints/sam2.1_hiera_tiny.pt"
    },
    {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
        "path": "Grounded-SAM-2/checkpoints/sam2.1_hiera_small.pt"
    },
    {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
        "path": "Grounded-SAM-2/checkpoints/sam2.1_hiera_base_plus.pt"
    },
    {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        "path": "Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
    },

    # --- Grounded-SAM-2: Grounding DINO Checkpoints ---
    {
        "url": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
        "path": "Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth"
    },
    {
        "url": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth",
        "path": "Grounded-SAM-2/gdino_checkpoints/groundingdino_swinb_cogcoor.pth"
    }
]

if __name__ == "__main__":
    print("=========================================================")
    print(f"   Downloading Weights to: {ROOT_DIR}")
    print("=========================================================")
    
    for item in weights_to_download:
        download_file(item["url"], item["path"])
        
    print("\n All weights downloaded successfully!")