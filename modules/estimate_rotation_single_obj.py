"""
Estimate in-plane rotation θ (0–360°) of a textured object.
"""
from __future__ import annotations
import os
import math
import json
import argparse
import tempfile
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import lpips
    HAS_LPIPS = True
except Exception:
    HAS_LPIPS = False

import cv2

# PyTorch3D
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    MeshRenderer,
    MeshRasterizer,
    RasterizationSettings,
    SoftSilhouetteShader,
    SoftPhongShader,
    FoVPerspectiveCameras,
    OrthographicCameras,
    AmbientLights,
)
from pytorch3d.transforms import euler_angles_to_matrix
from pytorch3d.structures import Meshes
from pytorch3d.renderer import FoVPerspectiveCameras, look_at_view_transform

import trimesh


def to_tensor(img: np.ndarray, device: torch.device, float32: bool = True) -> torch.Tensor:
    t = torch.from_numpy(img)
    if float32:
        t = t.float()
    t = t.to(device)
    return t


def imread_rgba(path: str) -> Image.Image:
    return Image.open(path).convert("RGBA")


def rgba_to_np(img_rgba: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.array(img_rgba)[..., :4]
    rgb = arr[..., :3].astype(np.float32) / 255.0
    alpha = arr[..., 3].astype(np.float32) / 255.0
    return rgb, alpha


# Unified cropping: tightest bbox -> pad to square -> resize
def crop_square_by_mask(rgb_or_gray: np.ndarray, mask: np.ndarray,
                        out_size: int = 256, pad_ratio: float = 0,
                        thr: float = 1e-4):
    # rgb_or_gray: (H,W,3) or (H,W); mask: (H,W) in [0..1]
    H, W = mask.shape
    ys, xs = np.where(mask > thr)
    if len(xs) == 0:  # Fallback for all black
        x0, y0, x1, y1 = 0, 0, W-1, H-1
    else:
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
    # Expand and crop within image
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
    side = int(max(bw, bh) * (1.0 + pad_ratio))
    nx0, ny0 = int(round(cx - side/2)), int(round(cy - side/2))
    nx1, ny1 = nx0 + side - 1, ny0 + side - 1
    # Clamp to edges
    dx0, dy0 = max(0, -nx0), max(0, -ny0)
    dx1, dy1 = max(0, nx1 - (W-1)), max(0, ny1 - (H-1))
    nx0, nx1 = nx0 + dx0, nx1 - dx1
    ny0, ny1 = ny0 + dy0, ny1 - dy1

    def _crop_pad_square_resize(img):
        crop = img[ny0:ny1+1, nx0:nx1+1]
        h, w = crop.shape[:2]
        S = max(h, w)
        if crop.ndim == 3:
            canvas = np.zeros((S, S, crop.shape[2]), dtype=crop.dtype)
        else:
            canvas = np.zeros((S, S), dtype=crop.dtype)
        oy, ox = (S - h)//2, (S - w)//2
        canvas[oy:oy+h, ox:ox+w] = crop
        out = cv2.resize((canvas*255).astype(np.uint8), (out_size, out_size), cv2.INTER_AREA)
        return out.astype(np.float32) / 255.0

    rgb_out  = _crop_pad_square_resize(rgb_or_gray)
    mask_out = _crop_pad_square_resize(mask)
    return rgb_out, mask_out


def mask_to_edges_and_dt(mask01: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    m = (mask01 * 255).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)
    edges = cv2.Canny(m, 100, 200)
    inv = np.where(edges > 0, 0, 255).astype(np.uint8)
    dt = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    return edges, dt

# tight bbox from soft silhouette + crop→square→resize
def _bbox_from_soft_mask_np(sil_np: np.ndarray, thr: float = 1e-4) -> tuple[int,int,int,int]:
    """sil_np: (H,W) float [0..1]; returns (x0,y0,x1,y1). If all 0, returns full image."""
    ys, xs = np.where(sil_np > thr)
    H, W = sil_np.shape
    if len(xs) == 0:
        return 0, 0, W-1, H-1
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

def _crop_pad_square_resize_np(img: np.ndarray, bbox: tuple[int,int,int,int],
                               out_size: int = 256, pad_ratio: float = 0) -> np.ndarray:
    """
    img: (H,W) or (H,W,3), range [0..1] or [0..255].
    Crop based on bbox, expand by pad_ratio, pad to square, resize to out_size.
    """
    H, W = img.shape[:2]
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    bw = (x1 - x0 + 1)
    bh = (y1 - y0 + 1)
    side = int(max(bw, bh) * (1.0 + pad_ratio))
    nx0 = max(0, int(round(cx - side/2)))
    ny0 = max(0, int(round(cy - side/2)))
    nx1 = min(W-1, nx0 + side - 1)
    ny1 = min(H-1, ny0 + side - 1)
    crop = img[ny0:ny1+1, nx0:nx1+1]

    # pad to square
    h, w = crop.shape[:2]
    S = max(h, w)
    if img.ndim == 3:
        canvas = np.zeros((S, S, img.shape[2]), dtype=img.dtype)
    else:
        canvas = np.zeros((S, S), dtype=img.dtype)
    oy = (S - h)//2
    ox = (S - w)//2
    canvas[oy:oy+h, ox:ox+w] = crop

    # resize
    if canvas.dtype != np.uint8:
        canvas8 = (np.clip(canvas, 0, 1) * 255).astype(np.uint8)
    else:
        canvas8 = canvas
    out = cv2.resize(canvas8, (out_size, out_size), interpolation=cv2.INTER_AREA)
    if out.dtype != np.float32:
        out = out.astype(np.float32) / 255.0
    return out

def _crop_render_by_silhouette(sil_t: torch.Tensor, rgb_t: torch.Tensor,
                               out_size: int = 256, pad_ratio: float = 0):
    """
    sil_t: (B,1,H,W) torch float in [0,1]
    rgb_t: (B,3,H,W) torch float in [0,1]
    Calculate bbox for each sample using its own silhouette, and synchronously crop RGB and SIL to out_size.
    Returns cropped (sil_t_new, rgb_t_new).
    """
    B, _, H, W = sil_t.shape
    sil_out, rgb_out = [], []
    for i in range(B):
        sil_np = sil_t[i,0].detach().cpu().numpy()
        rgb_np = rgb_t[i].permute(1,2,0).detach().cpu().numpy()
        bb = _bbox_from_soft_mask_np(sil_np, thr=1e-4)
        sil_c = _crop_pad_square_resize_np(sil_np, bb, out_size=out_size, pad_ratio=pad_ratio)
        rgb_c = _crop_pad_square_resize_np(rgb_np, bb, out_size=out_size, pad_ratio=pad_ratio)
        sil_out.append(torch.from_numpy(sil_c)[None, None])
        rgb_out.append(torch.from_numpy(rgb_c).permute(2,0,1)[None])
    sil_out = torch.cat(sil_out, dim=0).to(sil_t.device)
    rgb_out = torch.cat(rgb_out, dim=0).to(rgb_t.device)
    return sil_out, rgb_out



def centroid_from_mask(mask01: np.ndarray) -> Tuple[float, float]:
    m = (mask01 > 0.5).astype(np.uint8)
    M = cv2.moments(m)
    if abs(M['m00']) < 1e-6:
        h, w = mask01.shape
        return (w / 2.0, h / 2.0)
    return (M['m10'] / M['m00'], M['m01'] / M['m00'])


class SoftEdge(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
        self.register_buffer('kx', kx.view(1, 1, 3, 3))
        self.register_buffer('ky', ky.view(1, 1, 3, 3))
    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(x01, self.kx, padding=1)
        gy = F.conv2d(x01, self.ky, padding=1)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-8)
        mag = mag / (mag.amax(dim=[2, 3], keepdim=True) + 1e-6)
        return mag


class AppearanceLoss(nn.Module):
    def __init__(self, device: torch.device, use_lpips: bool = True):
        super().__init__()
        self.use_lpips = use_lpips and HAS_LPIPS
        if self.use_lpips:
            self.lpips = lpips.LPIPS(net='vgg').to(device)
        else:
            self.lpips = None
    def forward(self, imgA: torch.Tensor, imgB: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is not None:
            imgA = imgA * mask
            imgB = imgB * mask
        if self.use_lpips:
            return self.lpips(imgA * 2 - 1, imgB * 2 - 1).mean()
        # Fallback L2
        if mask is not None:
            denom = mask.mean().clamp_min(1e-6)
            return ((imgA - imgB) ** 2 * mask).mean() / denom
        return ((imgA - imgB) ** 2).mean()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
    
class DinoAppearanceLoss(nn.Module):
    def __init__(self, device, backbone='dinov2_vitb14', pretrained_model=None):
        super().__init__()
        self.device = device

        if pretrained_model is not None:
            self.model = pretrained_model
        else:
            self.model = torch.hub.load('facebookresearch/dinov2', backbone).to(device).eval()
        self.transform = transforms.Compose([
            transforms.Resize(224, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5)
        ])
    def encode(self, img):
        B = img.shape[0]
        feats = []
        for i in range(B):
            pil = transforms.ToPILImage()(img[i].cpu())
            t = self.transform(pil).unsqueeze(0).to(self.device)
            f = self.model(t) 
            feats.append(F.normalize(f, dim=-1))
        return torch.cat(feats, dim=0)
    def forward(self, a, b, mask=None):
        fa = self.encode(a)
        fb = self.encode(b)
        # Cosine distance
        return 1.0 - (fa * fb).sum(dim=-1).mean()


# Mesh loading utils

def convert_to_obj_if_needed(mesh_path: str, work_dir: str, max_tex_res: Optional[int] = None) -> str:
    ext = os.path.splitext(mesh_path)[1].lower()
    if ext in ['.obj']:
        if max_tex_res is not None:
            _downscale_textures(os.path.dirname(mesh_path), max_tex_res)
        return mesh_path
    if ext in ['.glb', '.gltf']:
        os.makedirs(work_dir, exist_ok=True)
        obj_path = os.path.join(work_dir, 'converted.obj')
        scene = trimesh.load(mesh_path, force='scene')
        scene.export(obj_path)
        # Also downscale any textures that were exported
        if max_tex_res is not None:
            _downscale_textures(work_dir, max_tex_res)
        return obj_path
    raise ValueError(f"Unsupported mesh format: {ext}. Use .obj/.glb/.gltf")


def load_and_normalize_mesh(obj_path: str, device: torch.device) -> Meshes:
    mesh = load_objs_as_meshes([obj_path], device=device)
    verts = mesh.verts_list()[0]
    center = verts.mean(0)
    verts = verts - center
    scale = 1.0 / (verts.abs().max() + 1e-9)
    verts = verts * scale
    # Enable texture mipmapping to reduce sampling cost if available (PyTorch3D handles internally)
    mesh_norm = Meshes(verts=[verts], faces=mesh.faces_list(), textures=mesh.textures)
    return mesh_norm


# Rendering
@dataclass
class RenderConfig:
    image_size: int = 256
    faces_per_pixel: int = 50
    blur_radius: float = 1e-6
    pitch_deg: float = 60.0
    cam_Tz: float = 3.0      # Distance between camera and object (larger means weaker perspective)
    fov_deg: float = 45.0    # Field of view (larger means stronger perspective)
    camera_model: str = "ortho"   # "ortho" or "persp"
    ortho_mag: float = 1.2        # Orthographic camera xmag/ymag (normalized based on side=2)


class MeshRendererHelper:
    def __init__(self, mesh: Meshes, device: torch.device, cfg: RenderConfig):
        self.mesh = mesh
        self.device = device
        self.cfg = cfg
        self.soft_edge = SoftEdge().to(device)
        # Separate raster settings for silhouette vs RGB to save memory
        self.raster_settings_sil = RasterizationSettings(
            image_size=cfg.image_size,
            blur_radius=cfg.blur_radius,
            faces_per_pixel=cfg.faces_per_pixel,  # can be higher for stable silhouettes
            cull_backfaces=True,
            bin_size=0,  # use naive rasterizer to avoid artifacts (see PyTorch3D docs
        )
        self.raster_settings_rgb = RasterizationSettings(
            image_size=cfg.image_size,
            blur_radius=0.0,
            faces_per_pixel=max(1, cfg.faces_per_pixel // 5),  # much smaller to reduce mem
            cull_backfaces=True,
            bin_size=0,  # use naive rasterizer to avoid artifacts (see PyTorch3D docs
        )

    def _make_cameras(self, theta_rad: torch.Tensor,
                  pitch_delta_rad: Optional[torch.Tensor] = None):
        """
        Explicitly construct camera pose using eye/at/up.
        """

        if theta_rad.ndim == 0:
            theta_rad = theta_rad[None]
        th = theta_rad.reshape(-1)

        B = th.numel()
        device = self.device
        d = torch.full((B,), float(self.cfg.cam_Tz), device=device)

        # pitch: scalar config + optional fine-tuning (unit: degrees -> radians)
        base_pitch_deg = torch.full((B,), float(self.cfg.pitch_deg), device=device)
        if pitch_delta_rad is not None:
            base_pitch_deg = base_pitch_deg + torch.rad2deg(pitch_delta_rad).reshape(-1)
        ph = torch.deg2rad(base_pitch_deg)

        # Calculate camera eye coordinates from (theta, pitch, d)
        r_xy = d * torch.cos(ph)
        z    = d * torch.sin(ph)
        x    = r_xy * torch.sin(th)
        y    = r_xy * torch.cos(th)

        eye = torch.stack([x, y, z], dim=1)
        at  = torch.zeros(B, 3, device=device) 

        forward = (at - eye)
        forward = forward / (forward.norm(dim=1, keepdim=True).clamp_min(1e-8))  # (B,3)

        # 1) First choose a "reference up" to avoid collinearity with forward
        world_up = torch.tensor([0.0, 0.0, 1.0], device=device).expand(B, 3)
        near_pole = forward[:, 2].abs() > 0.98  # Near top/bottom view pole
        # Use Y as reference in pole region, otherwise Z
        up_ref = world_up.clone()
        if near_pole.any():
            up_ref = up_ref.clone()
            up_ref[near_pole] = torch.tensor([0.0, 1.0, 0.0], device=device)

        # 2) Gram-Schmidt orthogonalization into the normal plane of forward
        up = up_ref - (up_ref * forward).sum(dim=1, keepdim=True) * forward
        up = up / up.norm(dim=1, keepdim=True).clamp_min(1e-8)

        # 3) Force up to fall in +Z hemisphere to avoid overall inversion (roll=180)
        flip = (up[:, 2] < 0).unsqueeze(1)   # Flip back if z component is negative
        if flip.any():
            up[flip.expand_as(up)] = -up[flip.expand_as(up)]

        # 4) Generate R, T
        R, T = look_at_view_transform(eye=eye, at=at, up=up, device=device)

        # Get R, T using eye/at/up
        R, T = look_at_view_transform(eye=eye, at=at, up=up, device=device)

        # Assemble camera model
        if self.cfg.camera_model.lower() == "ortho":
            try:
                from pytorch3d.renderer import OrthographicCameras
                scale = torch.ones(B, 3, device=device) * float(self.cfg.ortho_mag)
                cams = OrthographicCameras(device=device, R=R, T=T, scale_xyz=scale)
            except TypeError:
                from pytorch3d.renderer import OrthographicCameras
                f  = torch.full((B, 2), 1.0 / float(self.cfg.ortho_mag), device=device)
                pp = torch.zeros((B, 2), device=device)
                cams = OrthographicCameras(device=device, R=R, T=T,
                                        focal_length=f, principal_point=pp)
        else:
            cams = FoVPerspectiveCameras(device=device, R=R, T=T,
                                        fov=float(self.cfg.fov_deg))

        return cams

    def _build_renderers(self, cams: FoVPerspectiveCameras):
        raster_sil = MeshRasterizer(cameras=cams, raster_settings=self.raster_settings_sil)
        raster_rgb = MeshRasterizer(cameras=cams, raster_settings=self.raster_settings_rgb)
        sil_renderer = MeshRenderer(rasterizer=raster_sil, shader=SoftSilhouetteShader())
        lights = AmbientLights(device=self.device)
        rgb_renderer = MeshRenderer(rasterizer=raster_rgb, shader=SoftPhongShader(device=self.device, cameras=cams, lights=lights))
        return sil_renderer, rgb_renderer

    @staticmethod
    def rotate_mesh_z(mesh: Meshes, thetas_rad: torch.Tensor) -> Meshes:
        """
        Rotate mesh around Z axis by thetas_rad; return batched Meshes
        """
        if thetas_rad.ndim == 0:
            thetas_rad = thetas_rad[None]
        B = thetas_rad.numel()

        V = mesh.verts_padded()          # (1, V, 3)
        F = mesh.faces_padded()          # (1, F, 3)

        tex = mesh.textures.extend(B)

        # Rz(θ)
        Rz = euler_angles_to_matrix(
            torch.stack([
                torch.zeros_like(thetas_rad),   # Rx=0
                torch.zeros_like(thetas_rad),   # Ry=0
                thetas_rad                      # Rz=θ
            ], dim=-1),
            convention="XYZ"
        )                                       # (B,3,3)

        Vb   = V.expand(B, -1, -1)              # (B,V,3)
        Vrot = torch.bmm(Vb, Rz.transpose(1, 2))# (B,V,3)

        return Meshes(verts=Vrot, faces=F.expand(B, -1, -1), textures=tex)
    
    @torch.no_grad()
    def render_batch(self, thetas_rad: torch.Tensor, chunk: int = 4):
        """Chunked rendering in **float32** (no autocast) because PyTorch3D rasterizer expects float.
        This avoids the 'expected scalar type Float but found Half' error under AMP.
        """
        outs_sil = []
        outs_rgb = []
        for i in range(0, thetas_rad.numel(), max(1, chunk)):
            th = thetas_rad[i:i+chunk]
            cams = self._make_cameras(torch.zeros_like(th))
            sil_renderer, rgb_renderer = self._build_renderers(cams)
            
            meshes_rot = self.rotate_mesh_z(self.mesh, th)
            sil = sil_renderer(meshes_rot)[..., 3].unsqueeze(1).clamp(0, 1)
            rgb = rgb_renderer(meshes_rot)[..., :3].permute(0, 3, 1, 2).clamp(0, 1)


            outs_sil.append(sil)
            outs_rgb.append(rgb)
        sil = torch.cat(outs_sil, dim=0)
        rgb = torch.cat(outs_rgb, dim=0)
        return sil, rgb


def affine_warp(img: torch.Tensor, s: torch.Tensor, tx_n: torch.Tensor, ty_n: torch.Tensor) -> torch.Tensor:
    """
    Apply 2D similarity transform (scale s and normalized translations tx_n, ty_n) to an
    image tensor using grid_sample. Robust to s/tx/ty shapes: accepts (B,), (B,1), or (B,1,1).
    """
    B, C, H, W = img.shape
    # Normalize parameter shapes to (B,)
    s = s.reshape(B)
    tx_n = tx_n.reshape(B)
    ty_n = ty_n.reshape(B)
    # Build affine matrices explicitly to avoid bad broadcasting: [[s, 0, tx], [0, s, ty]]
    A = torch.zeros(B, 2, 3, device=img.device, dtype=img.dtype)
    A[:, 0, 0] = s
    A[:, 1, 1] = s
    A[:, 0, 2] = tx_n
    A[:, 1, 2] = ty_n
    grid = F.affine_grid(A, size=(B, C, H, W), align_corners=False)
    out = F.grid_sample(img, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    return out


def soft_iou_loss(pred_sil: torch.Tensor, tgt_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    inter = (pred_sil * tgt_mask).sum(dim=[1, 2, 3])
    union = (pred_sil + tgt_mask - pred_sil * tgt_mask).sum(dim=[1, 2, 3]) + eps
    return (1.0 - inter / union).mean()


def one_sided_chamfer(pred_sil: torch.Tensor,
                      dt_tgt_edges: torch.Tensor,
                      soft_edge: SoftEdge) -> torch.Tensor:
    """
    Normalized version: Normalize DT with image diagonal D, returning magnitude ~[0,1].
    """
    e_pred = soft_edge(pred_sil)                     # Predicted edge intensity (B,1,H,W)
    B, _, H, W = dt_tgt_edges.shape
    # Normalize using geometric scale: D = sqrt(H^2 + W^2)
    D = math.sqrt(H*H + W*W)
    dt_norm = dt_tgt_edges / max(D, 1e-6)           # Normalize to ~[0,1]
    num = (dt_norm * e_pred).sum(dim=[1,2,3])
    den = e_pred.sum(dim=[1,2,3]).clamp_min(1e-6)
    return (num / den).mean()



@dataclass
class Weights:
    w_sil: float = 1.0
    w_edge: float = 0.5
    w_app: float = 0.12
    w_reg: float = 0.01


@dataclass
class Options:
    coarse_step_deg: float = 10.0
    topk: int = 8
    refine_steps: int = 140
    pitch_delta_allow_deg: float = 0.0


@dataclass
class Result:
    theta_deg: float
    loss: float
    sil_render: np.ndarray
    rgb_render: np.ndarray


class ThetaEstimator:
    def __init__(self, renderer: MeshRendererHelper, device: torch.device, weights: Weights, options: Options, pad_ratio: float, dino_model=None):
        self.renderer = renderer
        self.device = device
        self.weights = weights
        self.options = options
        self.app_loss_fn_dino = DinoAppearanceLoss(device, backbone='dinov2_vitb14', pretrained_model=dino_model)
        self.app_loss_fn = AppearanceLoss(device, use_lpips=True)
        self.soft_edge = renderer.soft_edge
        self.pad_ratio = pad_ratio

    @torch.no_grad()
    def coarse_search(self, rgb_tgt: torch.Tensor, mask_tgt: torch.Tensor, dt_tgt: torch.Tensor,
                      init_hint_deg: Optional[float] = None):
        H, W = mask_tgt.shape[-2:]
        step = self.options.coarse_step_deg
        thetas_deg = torch.arange(0.0, 360.0, step, device=self.device)
        if init_hint_deg is not None:
            shift = int((init_hint_deg % 360) // step)
            thetas_deg = torch.remainder(torch.arange(shift, shift + thetas_deg.numel(), device=self.device), thetas_deg.numel()) * step
        thetas_rad = torch.deg2rad(thetas_deg)

        sil_b, rgb_b = self.renderer.render_batch(thetas_rad)

        sil_b, rgb_b = _crop_render_by_silhouette(
            sil_b, rgb_b, out_size=mask_tgt.shape[-1], pad_ratio=self.pad_ratio 
        )


        B = sil_b.shape[0]
        mask_b = mask_tgt.expand(B, -1, -1, -1)
        dt_b   = dt_tgt.expand(B, -1, -1, -1)

        # IoU
        inter = (sil_b * mask_b).sum(dim=[1,2,3])
        union = (sil_b + mask_b - sil_b*mask_b).sum(dim=[1,2,3]) + 1e-6
        sil_l = 1.0 - inter / union

        # One-sided Chamfer
        ch = one_sided_chamfer(sil_b, dt_b, self.soft_edge)

        # Appearance
        app_l = torch.stack([
            self.app_loss_fn(rgb_b[i:i+1] * sil_b[i:i+1], rgb_tgt, mask_b[i:i+1])
            for i in range(B)
        ]).squeeze()

        total_inst = (self.weights.w_sil * sil_l
                    + self.weights.w_edge * ch
                    + self.weights.w_app * app_l)

        topk_vals, topk_idx = torch.topk(-total_inst, k=min(self.options.topk, B))
        return thetas_rad[topk_idx], sil_b[topk_idx], rgb_b[topk_idx], total_inst[topk_idx]


    def refine(self, theta_rad_init: torch.Tensor, rgb_tgt: torch.Tensor, mask_tgt: torch.Tensor, dt_tgt: torch.Tensor) -> Result:
        device = self.device
        best_loss = float('inf'); best = None
        for k in range(theta_rad_init.numel()):
            theta = nn.Parameter(theta_rad_init[k:k+1].clone().detach())
            pitch_delta = nn.Parameter(torch.zeros(1, device=device)) if self.options.pitch_delta_allow_deg > 0 else None
            params = [p for p in [theta, pitch_delta] if p is not None]
            
            opt = torch.optim.Adam(params, lr=3e-2)
            prev = None; stall = 0; patience = 20
            for step in range(self.options.refine_steps):
                opt.zero_grad()

                th1 = theta.view(1)
                zero_th = torch.zeros_like(th1)
                cams = (self.renderer._make_cameras(zero_th, pitch_delta) if pitch_delta is not None else self.renderer._make_cameras(zero_th))
                sil_rend, rgb_rend = self.renderer._build_renderers(cams)

                meshes_rot = self.renderer.rotate_mesh_z(self.renderer.mesh, th1)

                sil = sil_rend(meshes_rot)[..., 3].unsqueeze(1).clamp(0, 1)
                rgb = rgb_rend(meshes_rot)[..., :3].permute(0, 3, 1, 2).clamp(0, 1)

                sil, rgb = _crop_render_by_silhouette(sil, rgb, out_size=mask_tgt.shape[-1], pad_ratio=0)

                L_sil  = soft_iou_loss(sil, mask_tgt)
                L_edge = one_sided_chamfer(sil, dt_tgt, self.soft_edge)
                L_app  = self.app_loss_fn(rgb * sil, rgb_tgt, mask_tgt)

                reg = 0.0
                if pitch_delta is not None:
                    lim = math.radians(self.options.pitch_delta_allow_deg)
                    reg = (pitch_delta / lim) ** 2 * 0.05

                loss = (self.weights.w_sil * L_sil
                        + self.weights.w_edge * L_edge
                        + self.weights.w_app * L_app
                        + reg)

                loss.backward(); nn.utils.clip_grad_norm_(params, 5.0); opt.step()
                cur = loss.item();
                if prev is None or prev - cur > 1e-5: prev = cur; stall = 0
                else: stall += 1
                if stall >= patience: break
           
            with torch.no_grad():
                th1 = theta.view(1)
                zero_th = torch.zeros_like(th1)
                cams = (self.renderer._make_cameras(zero_th, pitch_delta)
                        if pitch_delta is not None else
                        self.renderer._make_cameras(zero_th))
                sil_rend, rgb_rend = self.renderer._build_renderers(cams)
                meshes_rot = self.renderer.rotate_mesh_z(self.renderer.mesh, th1)

                sil = sil_rend(meshes_rot)[..., 3].unsqueeze(1).clamp(0, 1)
                rgb = rgb_rend(meshes_rot)[..., :3].permute(0, 3, 1, 2).clamp(0, 1)
                sil, rgb = _crop_render_by_silhouette(
                    sil, rgb, out_size=mask_tgt.shape[-1], pad_ratio=self.pad_ratio
                )

                L_sil  = soft_iou_loss(sil, mask_tgt).item()
                L_edge = one_sided_chamfer(sil, dt_tgt, self.soft_edge).item()
                L_app  = self.app_loss_fn(rgb * sil, rgb_tgt, mask_tgt).item()
                total  = self.weights.w_sil * L_sil + self.weights.w_edge * L_edge + self.weights.w_app * L_app

                if total < best_loss:
                    best_loss = total
                    theta_deg = float(torch.rad2deg(theta % (2 * math.pi)).item())
                    if theta_deg < 0: theta_deg += 360.0
                    best = Result(
                        theta_deg=theta_deg,
                        loss=float(total),
                        sil_render=sil[0, 0].detach().cpu().numpy(),
                        rgb_render=(rgb[0] * sil[0]).detach().cpu().numpy().transpose(1, 2, 0)
                    )

        return best


# Main

def _downscale_textures(folder: str, max_side: int):
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tga')
    for root, _, files in os.walk(folder):
        for fn in files:
            if fn.lower().endswith(exts):
                p = os.path.join(root, fn)
                try:
                    im = Image.open(p)
                    w, h = im.size
                    s = max(w, h)
                    if s > max_side:
                        scale = max_side / float(s)
                        new = (max(1, int(w * scale)), max(1, int(h * scale)))
                        im = im.resize(new, Image.LANCZOS)
                        im.save(p)
                except Exception:
                    pass

import os, math, base64
import numpy as np
from PIL import Image
import torch

def _save_png(np_img01, path):
    """np_img01: (H,W,3) or (H,W), range [0,1]"""
    arr = np_img01
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(path)

def _b64_of_file(p):
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def extract_last_choice_1_or_2(text: str) -> str | None:
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return None
    return lines[-1] if lines[-1] in ("1", "2") else None

def _render_hq_rgb_np(mesh_norm, device, theta_deg, pitch_deg_for_render,
                      fov_deg, camera_model, ortho_mag,
                      image_size=768, faces_pp=150, base_renderer=None):
    th = torch.tensor([math.radians(theta_deg)], device=device)

    cfg = RenderConfig(
        image_size=image_size, faces_per_pixel=faces_pp, blur_radius=0.0,
        pitch_deg=pitch_deg_for_render, cam_Tz=3.0,
        fov_deg=fov_deg, camera_model=camera_model, ortho_mag=ortho_mag
    )
    hq = MeshRendererHelper(mesh_norm, device, cfg)

    zero = torch.zeros_like(th)
    if base_renderer is not None:
        pitch_delta_deg = float(pitch_deg_for_render) - float(base_renderer.cfg.pitch_deg)
        pitch_delta_rad = torch.tensor([math.radians(pitch_delta_deg)], device=device)
        cams = base_renderer._make_cameras(zero, pitch_delta_rad=pitch_delta_rad)
    else:
        cams = hq._make_cameras(zero)

    _, rgb_rend = hq._build_renderers(cams)

    mesh_rot = hq.rotate_mesh_z(mesh_norm, th)
    rgb = rgb_rend(mesh_rot)[..., :3].permute(0, 3, 1, 2).clamp(0, 1)
    return rgb[0].detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)


def resolve_180_with_gpt(
    *,
    client, 
    outdir: str,
    result, 
    device,
    mesh_norm,
    renderer, 
    estimator, 
    weights, 
    mask_t, rgb_t, dt_t, 
    pad_ratio: float,
    fov: float,
    camera_model: str,
    ortho_mag: float,
    rgb_c: np.ndarray, 
    a_c: np.ndarray
):
    """
    GPT-assisted analysis for 180-degree ambiguity.
    """
    theta_1_deg = float(result.theta_deg)
    theta_2_deg = (theta_1_deg + 180.0) % 360.0

    theta_1_rad = torch.tensor([math.radians(theta_1_deg)], device=device)
    theta_2_rad = torch.tensor([math.radians(theta_2_deg)], device=device)
    print(f"Original angle: {theta_1_deg:.1f}°, 180° flipped: {theta_2_deg:.1f}°")

    # DINO Appearance Loss
    with torch.no_grad():

        zero = torch.zeros_like(theta_1_rad)
        cams_1 = renderer._make_cameras(zero)
        sil_rend_1, rgb_rend_1 = renderer._build_renderers(cams_1)
        mesh1 = renderer.rotate_mesh_z(mesh_norm, theta_1_rad)
        sil_1 = sil_rend_1(mesh1)[..., 3].unsqueeze(1).clamp(0,1)
        rgb_1 = rgb_rend_1(mesh1)[..., :3].permute(0,3,1,2).clamp(0,1)

        cams_2 = renderer._make_cameras(zero)
        sil_rend_2, rgb_rend_2 = renderer._build_renderers(cams_2)
        mesh2 = renderer.rotate_mesh_z(mesh_norm, theta_2_rad)
        sil_2 = sil_rend_2(mesh2)[..., 3].unsqueeze(1).clamp(0,1)
        rgb_2 = rgb_rend_2(mesh2)[..., :3].permute(0,3,1,2).clamp(0,1)


        app_loss_1 = estimator.app_loss_fn_dino(rgb_1 * sil_1, rgb_t, mask_t).item()
        app_loss_2 = estimator.app_loss_fn_dino(rgb_2 * sil_2, rgb_t, mask_t).item()

    print(f"[DINO] App: θ={theta_1_deg:.1f}° → {app_loss_1:.6f}, θ+180={theta_2_deg:.1f}° → {app_loss_2:.6f}")
    dino_prefers_flip = (app_loss_2 < app_loss_1)

    if not dino_prefers_flip:
        print("[DINO] keeps original θ, no GPT verification.")
        return float(result.theta_deg)

    # GPT Review
    print("[DINO] suggests 180° flip → escalating to GPT verification...")

    input_cropped_path = os.path.join(outdir, "input_cropped.png")
    if not os.path.exists(input_cropped_path):
        _save_png(rgb_c * a_c[..., None], input_cropped_path)

    gpt_img1_np = _render_hq_rgb_np(
        mesh_norm, device, theta_1_deg, pitch_deg_for_render=70.0, 
        fov_deg=fov, camera_model=camera_model, ortho_mag=ortho_mag,
        image_size=768, faces_pp=150,
        base_renderer=renderer
    )

    gpt_img2_np = _render_hq_rgb_np(
        mesh_norm, device, theta_2_deg, pitch_deg_for_render=70.0, 
        fov_deg=fov, camera_model=camera_model, ortho_mag=ortho_mag,
        image_size=768, faces_pp=150,
        base_renderer=renderer
    )

    temp_img1_path = os.path.join(outdir, "temp_option1.png")
    temp_img2_path = os.path.join(outdir, "temp_option2.png")
    _save_png(gpt_img1_np, temp_img1_path)
    _save_png(gpt_img2_np, temp_img2_path)

    gpt_prompt = (
        'You are a rotation analyst, please carefully observe Figures 1 and 2. '
        'Which angle is more consistent with Figure 3?\n'
        'Since Figures 1 and 2 are top views, they may be slightly different from Figure 3. Therefore, you need to observe which orientation on the plane is more consistent with Figure 3.'
        'Please analyze carefully, output your thought process, and output the sequence number '
        '"1" or "2" separately on the last line.'
    )
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": gpt_prompt},
            {"type": "image_url", "image_url": f"data:image/png;base64,{_b64_of_file(temp_img1_path)}"},
            {"type": "image_url", "image_url": f"data:image/png;base64,{_b64_of_file(temp_img2_path)}"},
            {"type": "image_url", "image_url": f"data:image/png;base64,{_b64_of_file(input_cropped_path)}"},
        ]
    }]

    try:
        gpt_resp = client.chat.completions.create(model="openai/gpt-4.1", messages=messages)
        gpt_text = gpt_resp.choices[0].message.content.strip()
        print("[GPT raw output]\n" + gpt_text)
        choice = extract_last_choice_1_or_2(gpt_text)
    except Exception as e:
        print(f"[GPT] verification failed: {e}")
        choice = None

    # Final angle
    if choice == "1":
        final_theta_deg = theta_1_deg
        print(f"[GPT] chose option 1 → final angle = {final_theta_deg:.1f}°")
        # Save: Selected is image 1 (original angle), unselected is image 2 (+180)
        _save_png(gpt_img1_np, os.path.join(outdir, "render_rgb_hq_topview.png"))
        _save_png(gpt_img2_np, os.path.join(outdir, "render_rgb_hq_topview_180.png"))
        _save_png(gpt_img2_np, os.path.join(outdir, "render_rgb_hq_180.png"))
    elif choice == "2":
        final_theta_deg = theta_2_deg
        print(f"[GPT] chose option 2 → final angle = {final_theta_deg:.1f}°")
        # Save: Selected is image 2 (+180), unselected is image 1 (original)
        _save_png(gpt_img2_np, os.path.join(outdir, "render_rgb_hq_topview.png"))
        _save_png(gpt_img1_np, os.path.join(outdir, "render_rgb_hq_topview_180.png"))
        _save_png(gpt_img1_np, os.path.join(outdir, "render_rgb_hq_180.png"))
    else:
        print("[GPT] no valid 1/2 on last line → fallback to DINO choice (flip).")
        final_theta_deg = theta_2_deg
        _save_png(gpt_img2_np, os.path.join(outdir, "render_rgb_hq_topview.png"))
        _save_png(gpt_img1_np, os.path.join(outdir, "render_rgb_hq_180.png"))

    # Re-render cropped image consistent with evaluation at final angle, and update result
    with torch.no_grad():
        theta_final_rad = torch.tensor([math.radians(final_theta_deg)], device=device)

        zero = torch.zeros_like(theta_final_rad)

        cams_f = renderer._make_cameras(zero)
        sil_r, rgb_r = renderer._build_renderers(cams_f)

        mesh_f = renderer.rotate_mesh_z(mesh_norm, theta_final_rad)
        sil_f = sil_r(mesh_f)[..., 3].unsqueeze(1).clamp(0, 1)
        rgb_f = rgb_r(mesh_f)[..., :3].permute(0, 3, 1, 2).clamp(0, 1)
        sil_f, rgb_f = _crop_render_by_silhouette(sil_f, rgb_f, out_size=mask_t.shape[-1], pad_ratio=pad_ratio)

        result.sil_render = sil_f[0, 0].detach().cpu().numpy()
        result.rgb_render = (rgb_f[0] * sil_f[0]).detach().cpu().numpy().transpose(1, 2, 0)

        L_sil = soft_iou_loss(sil_f, mask_t).item()
        L_edge = one_sided_chamfer(sil_f, dt_t, estimator.soft_edge).item()
        L_app = estimator.app_loss_fn(rgb_f * sil_f, rgb_t, mask_t).item()
        result.loss = weights.w_sil * L_sil + weights.w_edge * L_edge + weights.w_app * L_app

    result.theta_deg = float(final_theta_deg)
    return float(final_theta_deg)


def estimate_theta(
    image: str,
    mesh: str,
    pitch_deg: float = 60.0,
    fov: float = 45.0,
    outdir: str = 'out_theta',
    device_str: str = 'cuda',
    image_size: int = 256,
    faces_per_pixel: int = 50,
    batch_chunk: int = 4,
    max_tex_res: int = 1024,
    coarse_step: float = 10.0,
    topk: int = 8,
    refine_steps: int = 140,
    pitch_delta: float = 0.0,
    ws: float = 0.2,
    we: float = 0.1,
    wa: float = 3.0,
    wr: float = 0.01,
    camera_model: str = 'persp',
    ortho_mag: float = 1.2,
    pad_ratio: float = 0,
    client = None, 
    dino_model = None 
):
    """
    Estimates the in-plane rotation θ (0–360°) of an object from an RGBA image and a 3D mesh.
    """
    os.makedirs(outdir, exist_ok=True)
    device = torch.device(device_str if torch.cuda.is_available() and device_str.startswith('cuda') else 'cpu')
    print(f"Using device: {device}")

    # 1) Read and preprocess target image
    img_rgba = imread_rgba(image)
    rgb_np, a_np = rgba_to_np(img_rgba)
    rgb_c, a_c = crop_square_by_mask(rgb_np, a_np, pad_ratio=0, out_size=image_size)
    rgb_t = to_tensor(rgb_c.transpose(2, 0, 1)[None, ...], device)
    mask_t = to_tensor(a_c[None, None, ...], device)
    _, dt_np = mask_to_edges_and_dt(a_c)
    dt_t = to_tensor(dt_np[None, None, ...], device)

    # 2) Load mesh & renderer
    with tempfile.TemporaryDirectory() as tmpdir:
        obj_path = convert_to_obj_if_needed(mesh, work_dir=tmpdir, max_tex_res=(max_tex_res or None))
        mesh_norm = load_and_normalize_mesh(obj_path, device)
        rcfg = RenderConfig(
            image_size=image_size,
            faces_per_pixel=faces_per_pixel,
            blur_radius=1e-6,
            pitch_deg=pitch_deg,
            cam_Tz=3.0,
            fov_deg=fov,
            camera_model=camera_model,
            ortho_mag=ortho_mag
        )
        renderer = MeshRendererHelper(mesh_norm, device, rcfg)

        weights = Weights(w_sil=ws, w_edge=we, w_app=wa, w_reg=wr)
        opts = Options(coarse_step_deg=coarse_step, topk=topk, refine_steps=refine_steps, pitch_delta_allow_deg=pitch_delta)
        estimator = ThetaEstimator(renderer, device, weights, opts, pad_ratio, dino_model=dino_model)  

        # 3) Coarse search
        topk_thetas, topk_sil, topk_rgb, topk_losses = estimator.coarse_search(rgb_t, mask_t, dt_t, init_hint_deg=None)
        print("Coarse candidates (deg):", [float(x) for x in torch.rad2deg(topk_thetas)])
        best_idx = torch.argmin(topk_losses)
        best_theta_rad = topk_thetas[best_idx:best_idx+1]
        
        # Construct Result 
        with torch.no_grad():
            zero = torch.zeros_like(best_theta_rad)
            cams = renderer._make_cameras(zero)
            sil_rend, rgb_rend = renderer._build_renderers(cams)
            mesh_rot = renderer.rotate_mesh_z(mesh_norm, best_theta_rad)
            sil = sil_rend(mesh_rot)[..., 3].unsqueeze(1).clamp(0, 1)
            rgb = rgb_rend(mesh_rot)[..., :3].permute(0, 3, 1, 2).clamp(0, 1)
            sil, rgb = _crop_render_by_silhouette(
                sil, rgb, out_size=mask_t.shape[-1], pad_ratio=pad_ratio
            )
            
            L_sil = soft_iou_loss(sil, mask_t).item()
            L_edge = one_sided_chamfer(sil, dt_t, estimator.soft_edge).item()
            L_app = estimator.app_loss_fn(rgb * sil, rgb_t, mask_t).item()
            total = weights.w_sil * L_sil + weights.w_edge * L_edge + weights.w_app * L_app
            
            result = Result(
                theta_deg=float(torch.rad2deg(best_theta_rad % (2 * math.pi)).item()),
                loss=float(total),
                sil_render=sil[0, 0].detach().cpu().numpy(),
                rgb_render=(rgb[0] * sil[0]).detach().cpu().numpy().transpose(1, 2, 0)
            )

        # 4) Iterative refinement
        # result = estimator.refine(topk_thetas, rgb_t, mask_t, dt_t)

    first_theta_deg = float(result.theta_deg)
    print(f"[Pre-FINAL θ] {first_theta_deg:.1f}°")
    first_theta_rad = torch.tensor([math.radians(first_theta_deg)], device=device)
    
    with torch.no_grad():
        zero = torch.zeros_like(first_theta_rad)
        cams = renderer._make_cameras(zero)
        sil_rend, rgb_rend = renderer._build_renderers(cams)
        mesh_rot = renderer.rotate_mesh_z(mesh_norm, first_theta_rad)
        sil_final = sil_rend(mesh_rot)[..., 3].unsqueeze(1).clamp(0, 1)
        rgb_final = rgb_rend(mesh_rot)[..., :3].permute(0, 3, 1, 2).clamp(0, 1)
        
        sil_cropped, rgb_cropped = _crop_render_by_silhouette(
            sil_final, rgb_final, out_size=mask_t.shape[-1], pad_ratio=pad_ratio
        )
        
        edge_pred = renderer.soft_edge(sil_cropped)
        edge_tgt = renderer.soft_edge(mask_t)
        
        # High quality rendering
        hq_config = RenderConfig(
            image_size=512,
            faces_per_pixel=100,
            blur_radius=0.0,
            pitch_deg=pitch_deg,
            cam_Tz=3.0,
            fov_deg=fov,
            camera_model=camera_model,
            ortho_mag=ortho_mag
        )

        hq_renderer = MeshRendererHelper(mesh_norm, device, hq_config)
        hq_cams = hq_renderer._make_cameras(zero)
        _, hq_rgb_rend = hq_renderer._build_renderers(hq_cams)
        mesh_rot_hq = hq_renderer.rotate_mesh_z(mesh_norm, first_theta_rad)
        hq_rgb = hq_rgb_rend(mesh_rot_hq)[..., :3].permute(0, 3, 1, 2).clamp(0, 1)

    hq_rgb_np = hq_rgb[0].detach().cpu().numpy().transpose(1,2,0)
    rgb_cropped_np = rgb_cropped[0].detach().cpu().numpy().transpose(1,2,0)

    # Save all images
    print(first_theta_rad)
    Image.fromarray((np.clip(hq_rgb_np, 0, 1) * 255).astype(np.uint8)).save(
        os.path.join(outdir, 'first_render_rgb_hq.png'))
    Image.fromarray((np.clip(rgb_cropped_np, 0, 1) * 255).astype(np.uint8)).save(
        os.path.join(outdir, 'first_render_rgb_raw.png'))

    # 5) 180 degree angle verification
    final_theta = resolve_180_with_gpt(
        client=client,
        outdir=outdir,
        result=result,
        device=device,
        mesh_norm=mesh_norm,
        renderer=renderer,
        estimator=estimator,
        weights=weights,
        mask_t=mask_t,
        rgb_t=rgb_t,
        dt_t=dt_t,
        pad_ratio=pad_ratio,
        fov=fov,
        camera_model=camera_model,
        ortho_mag=ortho_mag,
        rgb_c=rgb_c,
        a_c=a_c
    )
    print(f"[FINAL θ] {final_theta:.1f}°")


    # 6) Save
    theta_deg = float(result.theta_deg)
    print(f"Estimated θ (deg) = {theta_deg:.3f}; total loss = {result.loss:.6f}")

   
    final_theta_rad = torch.tensor([math.radians(theta_deg)], device=device)

    print(theta_deg,final_theta_rad)
    
    with torch.no_grad():
        zero = torch.zeros_like(final_theta_rad)
        cams = renderer._make_cameras(zero)
        sil_rend, rgb_rend = renderer._build_renderers(cams)
        mesh_rot = renderer.rotate_mesh_z(mesh_norm, final_theta_rad)
        sil_final = sil_rend(mesh_rot)[..., 3].unsqueeze(1).clamp(0, 1)
        rgb_final = rgb_rend(mesh_rot)[..., :3].permute(0, 3, 1, 2).clamp(0, 1)
        
        # Crop to same spec as target image
        sil_cropped, rgb_cropped = _crop_render_by_silhouette(
            sil_final, rgb_final, out_size=mask_t.shape[-1], pad_ratio=pad_ratio
        )
        
        # Calculate edge map
        edge_pred = renderer.soft_edge(sil_cropped)
        edge_tgt = renderer.soft_edge(mask_t)
        
        # High quality rendering
        hq_config = RenderConfig(
            image_size=512,
            faces_per_pixel=100, 
            blur_radius=0.0,
            pitch_deg=pitch_deg,
            cam_Tz=3.0,
            fov_deg=fov,
            camera_model=camera_model,
            ortho_mag=ortho_mag
        )

        hq_renderer = MeshRendererHelper(mesh_norm, device, hq_config)
        hq_cams = hq_renderer._make_cameras(zero)
        _, hq_rgb_rend = hq_renderer._build_renderers(hq_cams)
        mesh_rot_hq = hq_renderer.rotate_mesh_z(mesh_norm, final_theta_rad)
        hq_rgb = hq_rgb_rend(mesh_rot_hq)[..., :3].permute(0, 3, 1, 2).clamp(0, 1)

    sil_cropped_np = sil_cropped[0,0].detach().cpu().numpy()
    rgb_cropped_np = rgb_cropped[0].detach().cpu().numpy().transpose(1,2,0)
    rgb_masked_np = (rgb_cropped_np * sil_cropped_np[..., None])  # Apply mask
    edge_pred_np = edge_pred[0,0].detach().cpu().numpy()
    edge_tgt_np = edge_tgt[0,0].detach().cpu().numpy()
    hq_rgb_np = hq_rgb[0].detach().cpu().numpy().transpose(1,2,0)

    # Save all images
    Image.fromarray((rgb_c * a_c[..., None] * 255).astype(np.uint8)).save(
        os.path.join(outdir, 'input_cropped.png'))
    
    # Target edges
    Image.fromarray((edge_tgt_np * 255).astype(np.uint8)).save(
        os.path.join(outdir, 'target_edges.png'))
    
    # Render silhouette
    Image.fromarray((sil_cropped_np * 255).astype(np.uint8)).save(
        os.path.join(outdir, 'render_sil.png'))
    
    # Render edges
    Image.fromarray((edge_pred_np * 255).astype(np.uint8)).save(
        os.path.join(outdir, 'render_edges.png'))
    
    # Render RGB (masked)
    Image.fromarray((np.clip(rgb_masked_np, 0, 1) * 255).astype(np.uint8)).save(
        os.path.join(outdir, 'render_rgb_masked.png'))
    
    # Render RGB (unmasked)
    Image.fromarray((np.clip(rgb_cropped_np, 0, 1) * 255).astype(np.uint8)).save(
        os.path.join(outdir, 'render_rgb_raw.png'))
    
    # High quality render
    Image.fromarray((np.clip(hq_rgb_np, 0, 1) * 255).astype(np.uint8)).save(
        os.path.join(outdir, 'render_rgb_hq.png'))
    
    # Overlay (target + render)
    overlay = (0.6 * rgb_masked_np + 0.4 * (rgb_c * a_c[..., None])).clip(0, 1)
    Image.fromarray((overlay * 255).astype(np.uint8)).save(
        os.path.join(outdir, 'overlay.png'))

    # Save loss verification info
    with torch.no_grad():
        L_sil_final = soft_iou_loss(sil_cropped, mask_t).item()
        L_edge_final = one_sided_chamfer(sil_cropped, dt_t, renderer.soft_edge).item()
        L_app_final = estimator.app_loss_fn(rgb_cropped * sil_cropped, rgb_t, mask_t).item()
        total_final = weights.w_sil * L_sil_final + weights.w_edge * L_edge_final + weights.w_app * L_app_final

    meta = {
        'theta_deg': theta_deg, 
        'loss': float(result.loss), 
        'loss_final_verification': float(total_final),
        'loss_components': {
            'sil': float(L_sil_final),
            'edge': float(L_edge_final), 
            'app': float(L_app_final)
        },
        'pitch_deg_used': pitch_deg, 
        'fov_deg': fov,
        'weights': {
            'w_sil': weights.w_sil,
            'w_edge': weights.w_edge,
            'w_app': weights.w_app
        }
    }
    
    with open(os.path.join(outdir, 'theta_result.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    return {
        'theta_deg': theta_deg,
        'loss': float(result.loss),
        'render_sil': sil_cropped_np,
        'render_rgb': rgb_masked_np, 
        'render_rgb_raw': rgb_cropped_np, 
        'render_hq': hq_rgb_np, 
        'edge_pred': edge_pred_np,  
        'edge_tgt': edge_tgt_np, 
        'overlay': overlay,
        'meta': meta,
        'outdir': outdir,
    }



if __name__ == '__main__':
    estimate_theta(image="output_scene/scene_1/output_assets/image/toy car_2.png",
        mesh="output_scene/scene_1/output_assets/sized_mesh/toy car_2.glb",
                   pitch_deg=60,
                   image_size=256,
                   outdir="toy_car_2out"
                   )
