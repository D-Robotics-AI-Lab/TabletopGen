"""
Pipeline Configuration Module.
"""

import os
from pathlib import Path


# Project root directory (parent of configs)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = PROJECT_ROOT


def get_pipeline_dir() -> str:
    """Get the pipeline root directory (project root)."""
    return str(PIPELINE_DIR)


def get_output_scene_dir() -> str:
    """Get the output scene root directory."""
    return str(PIPELINE_DIR / "output_scene")


def get_scene_dir(scene_id: int) -> str:
    """Get the directory for a specific scene."""
    return str(Path(get_output_scene_dir()) / f"scene_{scene_id}")


def get_next_scene_id() -> int:
    """Automatically get the next available scene ID."""
    output_scene_dir = Path(get_output_scene_dir())

    if not output_scene_dir.exists():
        return 1

    existing_scenes = []
    for item in output_scene_dir.iterdir():
        if item.is_dir() and item.name.startswith("scene_"):
            try:
                scene_num = int(item.name.split("_")[1])
                existing_scenes.append(scene_num)
            except (ValueError, IndexError):
                continue

    if not existing_scenes:
        return 1
    return max(existing_scenes) + 1


def get_comfy_image_dir(scene_dir: str) -> str:
    """Get the image output directory."""
    return str(Path(scene_dir) / "comfy_image")


def get_output_assets_dir(scene_dir: str) -> str:
    """Get the assets output directory."""
    return str(Path(scene_dir) / "output_assets")


def set_working_directory_to_scene(scene_dir: str) -> str:
    """Switch working directory to the scene directory and return the original working directory."""
    original_cwd = os.getcwd()
    os.chdir(scene_dir)
    return original_cwd


def restore_working_directory(original_cwd: str) -> None:
    """Restore the working directory."""
    os.chdir(original_cwd)


def get_configs_dir() -> str:
    """Get the configs directory."""
    return str(PROJECT_ROOT / "configs")
