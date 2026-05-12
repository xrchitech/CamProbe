import os, json, random
import torch
import math
from torch.utils.data import IterableDataset, Dataset
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
from torch.utils.data import Dataset
from torchvision import transforms
from typing import List, Tuple, Union
from PIL import Image
from torchvision.utils import save_image, make_grid
import os
import functools
import logging
import sys
import imageio
import atexit
import importlib
import torch
import torchvision
import numpy as np
from termcolor import colored
import torchvision.transforms.v2 as T
from einops import rearrange
import subprocess
from torchvision.utils import save_image, make_grid

class LatentStepFlowBank_axis(nn.Module):
    def __init__(self, num_dirs: int, max_angle=0.03
                 ):
        super().__init__()
        self.axis = nn.Parameter(torch.randn(num_dirs, 3))
        self.angle = nn.Parameter(torch.zeros(num_dirs))
        self.max_angle = max_angle

        self.initialization()

    def initialization(self):
        with torch.no_grad():
            a = torch.randn_like(self.axis)
            self.axis.copy_(F.normalize(a, dim=1))

            self.angle.uniform_(0.2, 0.4)

    def so3_exp(self, w):
        
        theta = torch.norm(w, dim=1, keepdim=True) + 1e-8
        k = w / theta

        B = w.shape[0]
        K = torch.zeros(B, 3, 3, device=w.device, dtype=w.dtype)

        K[:, 0, 1] = -k[:, 2]
        K[:, 0, 2] =  k[:, 1]
        K[:, 1, 0] =  k[:, 2]
        K[:, 1, 2] = -k[:, 0]
        K[:, 2, 0] = -k[:, 1]
        K[:, 2, 1] =  k[:, 0]

        I = torch.eye(3, device=w.device, dtype=w.dtype).unsqueeze(0)

        R = I + torch.sin(theta)[:, None] * K \
            + (1 - torch.cos(theta))[:, None] * (K @ K)

        return R
    
    def forward(self, class_ids, depth_ref):
        B, _, H, W = depth_ref.shape

        cy, cx = H // 2, W // 2
        ph, pw = int(H * 0.25), int(W * 0.25)
        center_patch = depth_ref[:, 0, cy-ph:cy+ph, cx-pw:cx+pw]
        mean_depth = center_patch.mean(dim=[1, 2])   

        pivot = torch.stack([
            torch.zeros_like(mean_depth),
            torch.zeros_like(mean_depth),
            mean_depth
        ], dim=1)  

        
        axis = F.normalize(self.axis[class_ids], dim=1)   
        theta = self.max_angle * torch.sigmoid(self.angle[class_ids])  

        w = axis * theta[:, None]                        
        R = self.so3_exp(w)

        
        t = pivot - (R @ pivot.unsqueeze(-1)).squeeze(-1)

        return R, t


class LatentStepFlowBank_axis_xyz(nn.Module):
    def __init__(self, axis="y", direction="left", angle=0.02):
        super().__init__()
        axis_dict = {
            "x": torch.tensor([1.0, 0.0, 0.0]),
            "y": torch.tensor([0.0, 1.0, 0.0]),
            "z": torch.tensor([0.0, 0.0, 1.0]),
        }
        if axis not in axis_dict:
            raise ValueError("axis must be 'x', 'y', or 'z'")
        if direction not in ["left", "right"]:
            raise ValueError("direction must be 'left' or 'right'")

        self.register_buffer("axis", axis_dict[axis].float())
        sign = 1.0 if direction == "left" else -1.0
        self.angle = angle * sign

    def so3_exp(self, w):
        theta = torch.norm(w, dim=1, keepdim=True) + 1e-8
        k = w / theta

        B = w.shape[0]
        K = torch.zeros(B, 3, 3, device=w.device, dtype=w.dtype)

        K[:, 0, 1] = -k[:, 2]
        K[:, 0, 2] = k[:, 1]
        K[:, 1, 0] = k[:, 2]
        K[:, 1, 2] = -k[:, 0]
        K[:, 2, 0] = -k[:, 1]
        K[:, 2, 1] = k[:, 0]

        I = torch.eye(3, device=w.device, dtype=w.dtype).unsqueeze(0)
        R = I + torch.sin(theta)[:, None] * K + (1 - torch.cos(theta))[:, None] * (K @ K)
        return R

    def forward(self, depth_ref):
        B, _, H, W = depth_ref.shape

        cy, cx = H // 2, W // 2
        ph, pw = int(H * 0.25), int(W * 0.25)
        center_patch = depth_ref[:, 0, cy - ph:cy + ph, cx - pw:cx + pw]
        mean_depth = center_patch.mean(dim=[1, 2])

        pivot = torch.stack([
            torch.zeros_like(mean_depth),
            torch.zeros_like(mean_depth),
            mean_depth,
        ], dim=1)

        axis = self.axis.to(depth_ref.device).expand(B, 3)
        w = axis * self.angle
        R = self.so3_exp(w)
        t = pivot - (R @ pivot.unsqueeze(-1)).squeeze(-1)
        return R, t

class LatentStepFlowBank(nn.Module):
    def __init__(self, num_dirs: int
                 ):
        super().__init__()
        self.num_dirs = num_dirs

        self.max_rot = 0.03      
        self.max_trans = 0.03    

        self.flow = nn.Parameter(torch.zeros(num_dirs, 6))
        self.initialization()

    def initialization(self):

        with torch.no_grad():
            K, D = self.flow.shape
            sign = torch.randint(0, 2, (K, D), device=self.flow.device) * 2 - 1
            mag = torch.empty((K, D), device=self.flow.device).uniform_(0.2, 0.4)
            self.flow.copy_(sign * mag)

    def so3_exp(self, w):
        
        theta = torch.norm(w, dim=1, keepdim=True) + 1e-8
        k = w / theta

        B = w.shape[0]
        K = torch.zeros(B, 3, 3, device=w.device, dtype=w.dtype)

        K[:, 0, 1] = -k[:, 2]
        K[:, 0, 2] =  k[:, 1]
        K[:, 1, 0] =  k[:, 2]
        K[:, 1, 2] = -k[:, 0]
        K[:, 2, 0] = -k[:, 1]
        K[:, 2, 1] =  k[:, 0]

        I = torch.eye(3, device=w.device, dtype=w.dtype).unsqueeze(0)

        R = I + torch.sin(theta)[:, None] * K \
            + (1 - torch.cos(theta))[:, None] * (K @ K)

        return R
    
    def forward(self, class_ids,):
        step_flow = self.flow[class_ids]

        w = self.max_rot * torch.tanh(step_flow[:, 0:3])   

        t = self.max_trans * torch.tanh(step_flow[:, 3:6])  

        R = self.so3_exp(w)

        return R, t

class MiDaSSmall:
    def __init__(
        self,
        device,
        depth_mode: str = "inverse",
        inv_depth_min: float = 1e-2,
        depth_min: float = 0.1,
        depth_max: float = None,
        depth_quantile_max: float = None,
    ):
        print(">>> Using MiDaS for GEOMETRIC depth <<<")
        self.device = device

        repo_path = "intel-isl_MiDaS_master"
        self.model = torch.hub.load(
            repo_path,
            "MiDaS_small",
            source="local",
            pretrained=False
        )
        weight_path = "intel-isl_MiDaS_master/weights/midas_v21_small_256.pt"
        state_dict = torch.load(weight_path, map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(device).eval()

        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
        self.std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

        
        self.depth_scale = 1.0     
        self.depth_bias  = 1.0     
        self.depth_mode = depth_mode
        self.inv_depth_min = inv_depth_min
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.depth_quantile_max = depth_quantile_max

    @torch.no_grad()
    def __call__(self, img_rgb):
        B, _, H, W = img_rgb.shape

        x = (img_rgb.to(self.device) - self.mean) / self.std
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)

        inv_depth = self.model(x)              
        inv_depth = F.interpolate(
            inv_depth.unsqueeze(1),
            size=(H, W),
            mode="bicubic",
            align_corners=False
        )  

        if self.depth_mode == "inverse":
            inv_depth_safe = inv_depth.clamp(min=self.inv_depth_min)
            depth = self.depth_scale / inv_depth_safe + self.depth_bias
        elif self.depth_mode == "inverse_quantile":
            inv_depth_safe = inv_depth.clamp(min=self.inv_depth_min)
            depth = self.depth_scale / inv_depth_safe + self.depth_bias
        elif self.depth_mode == "relative":
            depth = inv_depth - inv_depth.amin(dim=(-2, -1), keepdim=True) + self.depth_bias
        elif self.depth_mode == "relative_inverse":
            inv_depth_shifted = inv_depth - inv_depth.amin(dim=(-2, -1), keepdim=True)
            inv_depth_shifted = inv_depth_shifted.clamp(min=self.inv_depth_min)
            depth = self.depth_scale / inv_depth_shifted + self.depth_bias
        elif self.depth_mode == "relative_inverse_quantile":
            inv_depth_shifted = inv_depth - inv_depth.amin(dim=(-2, -1), keepdim=True)
            inv_depth_shifted = inv_depth_shifted.clamp(min=self.inv_depth_min)
            depth = self.depth_scale / inv_depth_shifted + self.depth_bias
            depth = depth.clamp(min=self.depth_min)

            q = self.depth_quantile_max if self.depth_quantile_max is not None else 0.99
            hi = torch.quantile(
                depth.flatten(1),
                q,
                dim=1,
            ).view(B, 1, 1, 1)
            depth = torch.minimum(depth, hi)
        else:
            raise ValueError(
                f"Unsupported MiDaS depth_mode: {self.depth_mode}. "
                "Expected one of ['inverse', 'inverse_quantile', 'relative', "
                "'relative_inverse', 'relative_inverse_quantile']"
            )

        depth = depth.clamp(min=self.depth_min)

        if self.depth_quantile_max is not None:
            hi = torch.quantile(
                depth.flatten(1),
                self.depth_quantile_max,
                dim=1,
            ).view(B, 1, 1, 1)
            depth = torch.minimum(depth, hi)

        if self.depth_max is not None:
            depth = depth.clamp(max=self.depth_max)

        return depth
    

def depth_camera_grid_sequence(
    depth,      
    R,          
    t,          
    H, W,
    scale,      
):
    device = depth.device
    dtype = depth.dtype
    B = depth.shape[0]

    if depth.dim() == 4:
        depth = depth[:, 0]

    if scale.dim() == 1:
        scale_d = scale[:, None, None]
        scale_t = scale[:, None]
    elif scale.dim() == 3:
        scale_d = scale
        scale_t = scale[:, 0, 0][:, None]
    else:
        raise ValueError(f"scale must have shape [B] or [B,1,1], got {scale.shape}")

    scale_d = scale_d.to(device=device, dtype=dtype)
    scale_t = scale_t.to(device=device, dtype=dtype)

    depth = depth / (scale_d + 1e-6)
    t = t / (scale_t + 1e-6)

    
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device, dtype=dtype),
        torch.linspace(-1, 1, W, device=device, dtype=dtype),
        indexing="ij"
    )

    x = x.unsqueeze(0).expand(B, -1, -1)
    y = y.unsqueeze(0).expand(B, -1, -1)

    
    Z = depth
    X = x * Z
    Y = y * Z

    pts = torch.stack([X, Y, Z], dim=-1)   
    pts = pts.view(B, H * W, 3)

    pts = torch.bmm(pts, R.transpose(1, 2)) + t[:, None, :]
    pts = pts.view(B, H, W, 3)

    Xp = pts[..., 0]
    Yp = pts[..., 1]
    Zp = pts[..., 2].clamp(min=1e-4)

    x_proj = Xp / Zp
    y_proj = Yp / Zp

    grid = torch.stack([x_proj, y_proj], dim=-1)
    grid = grid.clamp(-2, 2)
    return grid


def depth_camera_grid_perframe(
    depth,      
    R,          
    t,          
    H, W,
):
    device = depth.device
    dtype = depth.dtype
    B = depth.shape[0]

    if depth.dim() == 4:
        depth = depth[:, 0]

    
    scale_d = depth.flatten(1).median(dim=1, keepdim=True).values[:, None]
    scale_t = scale_d[:, 0, 0][:, None]  

    depth = depth / (scale_d + 1e-6)
    t = t / (scale_t + 1e-6)

    y, x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device, dtype=dtype),
        torch.linspace(-1, 1, W, device=device, dtype=dtype),
        indexing="ij"
    )

    x = x.unsqueeze(0).expand(B, -1, -1)
    y = y.unsqueeze(0).expand(B, -1, -1)

    Z = depth
    X = x * Z
    Y = y * Z

    pts = torch.stack([X, Y, Z], dim=-1).view(B, H * W, 3)
    pts = torch.bmm(pts, R.transpose(1, 2)) + t[:, None, :]
    pts = pts.view(B, H, W, 3)

    Xp = pts[..., 0]
    Yp = pts[..., 1]
    Zp = pts[..., 2].clamp(min=1e-4)

    x_proj = Xp / Zp
    y_proj = Yp / Zp

    grid = torch.stack([x_proj, y_proj], dim=-1)
    return grid.clamp(-2, 2)



def depth_camera_grid(
    depth,      
    R,          
    t,          
    H, W,
):

    device = depth.device
    dtype = depth.dtype
    B = depth.shape[0]

    if depth.dim() == 4:
        depth = depth[:, 0]

    
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device, dtype=dtype),
        torch.linspace(-1, 1, W, device=device, dtype=dtype),
        indexing="ij"
    )

    x = x.unsqueeze(0).expand(B, -1, -1)
    y = y.unsqueeze(0).expand(B, -1, -1)

    
    Z = depth
    X = x * Z
    Y = y * Z

    pts = torch.stack([X, Y, Z], dim=-1)   

    pts = pts.view(B, H*W, 3)
    pts = torch.bmm(pts, R.transpose(1,2)) + t[:,None,:]
    pts = pts.view(B, H, W, 3)

    Xp = pts[...,0]
    Yp = pts[...,1]
    Zp = pts[...,2].clamp(min=1e-4)

    
    x_proj = Xp / Zp
    y_proj = Yp / Zp

    grid = torch.stack([x_proj, y_proj], dim=-1)

    
    grid = grid.clamp(-2, 2)

    return grid

def make_checkerboard(B, C, H, W, device, tile=8):
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    )
    board = ((yy // tile + xx // tile) % 2).float()
    board = board * 2 - 1   
    board = board.unsqueeze(0).unsqueeze(0)  
    board = board.repeat(B, C, 1, 1)
    return board

def save_checkerboard_camera_layered(step_flow_bank,
                                     latent_c, latent_h, latent_w,
                                     video_length, camera_vector_dim,
                                     local_rank, output_dir):

    B = 1
    C = latent_c
    H = latent_h
    W = latent_w
    T = video_length

    device = torch.device(f"{local_rank}")
    bank = step_flow_bank.module if hasattr(step_flow_bank, "module") else step_flow_bank

    os.makedirs(output_dir, exist_ok=True)

    
    x0 = make_checkerboard(B, C, H, W, device=device)   

    depth = torch.full((1, H, W), 1.2, device=device)

    depth[:, H//3:2*H//3, W//3:2*W//3] = 0.6

    depth[:, H//4:3*H//4, W//4:3*W//4] = 0.9

    for k in range(camera_vector_dim):
        class_ids = torch.tensor([k], device=device)

        
        R_delta, t_delta = bank(class_ids)
        R_delta = R_delta.to(x0.dtype)
        t_delta = t_delta.to(x0.dtype)

        
        R_cum = torch.eye(3, device=device, dtype=x0.dtype).unsqueeze(0)
        t_cum = torch.zeros(1, 3, device=device, dtype=x0.dtype)

        R_list, t_list = [], []
        for f in range(T):
            R_list.append(R_cum)
            t_list.append(t_cum)

            t_cum = (R_delta @ t_cum.unsqueeze(-1)).squeeze(-1) + t_delta
            R_cum = R_delta @ R_cum
        
        frames = []
        for f in range(T):
            grid = depth_camera_grid(
                depth=depth,
                R=R_list[f],
                t=t_list[f],
                H=H,
                W=W,
            )

            warped = F.grid_sample(
                x0,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )

            img = warped[0, :3] if C >= 3 else warped[0].repeat(3, 1, 1)
            img = (img.clamp(-1, 1) + 1) / 2.0
            frames.append(img.cpu())

        gif = torch.stack(frames, dim=0)   
        save_path = os.path.join(output_dir, f"k_{k:02d}.gif")

        save_videos_grid(
            rearrange(gif, 't c h w -> 1 c t h w'),
            save_path,
            rescale=False,
        )

    print(f"[Checkerboard layered-depth sanity] saved to {output_dir}")

def save_checkerboard_camera_layered_depth(step_flow_bank,
                                     latent_c, latent_h, latent_w,
                                     video_length, camera_vector_dim,
                                     local_rank, output_dir):

    B = 1
    C = latent_c
    H = latent_h
    W = latent_w
    T = video_length

    device = torch.device(f"cuda:{local_rank}")
    bank = step_flow_bank.module if hasattr(step_flow_bank, "module") else step_flow_bank

    os.makedirs(output_dir, exist_ok=True)
    x0 = make_checkerboard(B, C, H, W, device=device)   
    depth = torch.full((1, H, W), 1.2, device=device)
    depth[:, H//3:2*H//3, W//3:2*W//3] = 0.6
    depth[:, H//4:3*H//4, W//4:3*W//4] = 0.9

    for k in range(camera_vector_dim):
        class_ids = torch.tensor([k], device=device)

        
        R_delta, t_delta = bank(class_ids, depth.unsqueeze(0))
        R_delta = R_delta.to(x0.dtype)
        t_delta = t_delta.to(x0.dtype)

        
        R_cum = torch.eye(3, device=device, dtype=x0.dtype).unsqueeze(0)
        t_cum = torch.zeros(1, 3, device=device, dtype=x0.dtype)

        R_list, t_list = [], []
        for f in range(T):
            R_list.append(R_cum)
            t_list.append(t_cum)

            t_cum = (R_delta @ t_cum.unsqueeze(-1)).squeeze(-1) + t_delta
            R_cum = R_delta @ R_cum

        
        frames = []
        for f in range(T):
            grid = depth_camera_grid(
                depth=depth,
                R=R_list[f],
                t=t_list[f],
                H=H,
                W=W,
            )

            warped = F.grid_sample(
                x0,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )

            img = warped[0, :3] if C >= 3 else warped[0].repeat(3, 1, 1)
            img = (img.clamp(-1, 1) + 1) / 2.0
            frames.append(img.cpu())

        gif = torch.stack(frames, dim=0)   
        save_path = os.path.join(output_dir, f"k_{k:02d}.gif")

        save_videos_grid(
            rearrange(gif, 't c h w -> 1 c t h w'),
            save_path,
            rescale=False,
        )

    print(f"[Checkerboard layered-depth sanity] saved to {output_dir}")

def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  
        x = (x.detach() * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)

def save_x0pred_grid_step(
    x0_pred: torch.Tensor,
    i: int,
    save_dir: str,
    nrow: int = 7,
    frame_indices=None,
    name: str=None,
):
    os.makedirs(save_dir, exist_ok=True)
    assert x0_pred.ndim == 5, f"Expected [B, C, T, H, W], got {x0_pred.shape}"
    x = x0_pred.detach()[0]           
    x = x[:3]                         
    if frame_indices is not None:
        x = x[:, frame_indices]        
    imgs = x.permute(1, 0, 2, 3)
    imgs = imgs.float()
    minv = imgs.min()
    maxv = imgs.max()
    imgs = (imgs - minv) / (maxv - minv + 1e-6)
    grid = make_grid(imgs, nrow=nrow)
    save_path = os.path.join(save_dir, f"i-{i}-{name}.png")
    save_image(grid, save_path)

    print(f"[x0pred debug] saved {save_path}")

def _normalize_for_vis(x: torch.Tensor):
    x = x.float()
    minv = x.min()
    maxv = x.max()
    return (x - minv) / (maxv - minv + 1e-6)

def save_video_frame_grid(
    video: torch.Tensor,
    save_path: str,
    nrow: int = 7,
    frame_indices=None,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    assert video.ndim == 5, f"Expected [B, C, T, H, W], got {video.shape}"

    x = video.detach()[0]
    if frame_indices is not None:
        x = x[:, frame_indices]
    imgs = x.permute(1, 0, 2, 3).clamp(0, 1)
    grid = make_grid(imgs, nrow=nrow)
    save_image(grid, save_path)

def save_depth_grid(
    depth_list,
    save_path: str,
    nrow: int = 7,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    imgs = []
    for depth in depth_list:
        d = depth.detach()[0, 0]
        d = _normalize_for_vis(d)
        imgs.append(d.unsqueeze(0).repeat(3, 1, 1))

    if len(imgs) == 0:
        return

    grid = make_grid(torch.stack(imgs, dim=0), nrow=nrow)
    save_image(grid, save_path)

def save_debug_stats(
    save_path: str,
    stats: dict,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

class I2VPromptDataset(Dataset):
    def __init__(self, json_file):
        self.samples = []
        with open(json_file, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.samples.append(obj)
        assert len(self.samples) > 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image_path = sample["image"]
        prompt = sample["prompt"]

        return {
            "image_path": image_path,   
            "prompt": prompt,
        }
    
def save_tensor_as_loop_gif(
    video,              
    save_path,
    fps=6,
    size=200,
    max_frames=8,       
):
    if isinstance(video, torch.Tensor):
        video = video.detach().cpu()

    
    if video.shape[0] == 3:
        video = video.permute(1, 2, 3, 0)

    T, H, W, C = video.shape
    assert C == 3, "Video must be RGB"

    
    if T > max_frames:
        idx = torch.linspace(0, T - 1, max_frames).long()
        video = video[idx]

    
    frames = (video * 255.0).clamp(0, 255).byte().numpy()
    T = frames.shape[0]

    cmd = [
        "ffmpeg",
        "-y",

        
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}",
        "-r", str(fps),
        "-i", "-",

        
        "-vf",
        (
            f"fps={fps},scale={size}:-1:flags=lanczos,"
            f"split[a][b];"
            f"[a]palettegen[p];"
            f"[b][p]paletteuse=dither=sierra2_4a"
        ),

        "-loop", "0",
        str(save_path),
    ]

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    for f in frames:
        proc.stdin.write(f.tobytes())

    proc.stdin.close()
    proc.wait()

    if proc.returncode != 0:
        err = proc.stderr.read().decode()
        raise RuntimeError(f"ffmpeg failed:\n{err}")
