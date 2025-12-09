import os, sys, importlib.util
from contextlib import contextmanager
from PIL import Image
import torch
import torchvision.transforms as T

MODULES_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(MODULES_DIR, os.pardir))
BIREFNET_DIR = os.path.join(PROJECT_ROOT, "BiRefNet")

@contextmanager
def birefnet_import_context(birefnet_dir: str):
    """
    Only insert birefnet_dir at the front of sys.path during import,
    inject BiRefNet/config.py as the top-level 'config';
    remove pure Python modules loaded from the BiRefNet directory upon exit, and restore 'config'.
    """
    import sys, os, importlib.util

    old_sys_path = sys.path.copy()
    # Record the set of modules before import
    before_keys = set(sys.modules.keys())
    # Record the original 'config' to restore later
    prev_config = sys.modules.get("config", None)

    try:
        sys.path.insert(0, birefnet_dir)

        # Inject BiRefNet/config.py as the top-level 'config'
        cfg_path = os.path.join(birefnet_dir, "config.py")
        spec = importlib.util.spec_from_file_location("config", cfg_path)
        mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(mod)  # type: ignore
        sys.modules["config"] = mod

        yield

    finally:
        # 1) Restore sys.path
        sys.path[:] = old_sys_path

        # 2) Remove only the modules that were newly added during this import and come from the BiRefNet directory
        after_keys = set(sys.modules.keys())
        new_keys = after_keys - before_keys
        for k in list(new_keys):
            m = sys.modules.get(k)
            p = getattr(m, "__file__", None)
            if isinstance(p, str) and os.path.abspath(p).startswith(os.path.abspath(birefnet_dir)):
                sys.modules.pop(k, None)

        # 3) Restore/Clean up 'config'
        if prev_config is not None:
            sys.modules["config"] = prev_config
        else:
            sys.modules.pop("config", None)


_MODEL_CACHE = {"model": None, "device": None, "tfm": None}

def _load_model(weights_path: str = None, image_size=(1024, 1024)):
    if _MODEL_CACHE["model"] is not None:
        return _MODEL_CACHE["model"], _MODEL_CACHE["device"], _MODEL_CACHE["tfm"]

    with birefnet_import_context(BIREFNET_DIR):
        from models.birefnet import BiRefNet
        from utils import check_state_dict

        model = BiRefNet(bb_pretrained=False)
        if weights_path is None:
            weights_path = os.path.join(BIREFNET_DIR, "pth", "BiRefNet-general-epoch_244.pth")

        state = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(check_state_dict(state))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    tfm = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])

    _MODEL_CACHE.update(model=model, device=device, tfm=tfm)
    return model, device, tfm

def seg_obj(img_path: str, out_path: str,
                   weights_path: str = None, image_size=(1024, 1024)):
    model, device, tfm = _load_model(weights_path, image_size)
    img = Image.open(img_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)

    with torch.no_grad():
        if device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred = model(x)[-1]
        else:
            pred = model(x)[-1]
        pred = pred.sigmoid().float().cpu()[0, 0]

    mask = T.ToPILImage()(pred).resize(img.size, Image.BILINEAR)
    rgba = img.copy()
    rgba.putalpha(mask)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rgba.save(out_path)
    return out_path

if __name__ == "__main__":
    src = "output_scene/scene_1/comfy_image/redraw_table_3.png"
    dst = "output_scene/scene_1/comfy_image/seg_table_3.png"
    print("saved:", seg_obj(src, dst))
