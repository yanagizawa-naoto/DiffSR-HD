#!/usr/bin/env python3
# inference.py
# Super-Resolution Inference Script for trained Conditional UNet diffusion model
# Supports image files, glob patterns, and indexed images within tar archives (syntax: archive.tar:idx)

import os
import glob
import argparse
import math
import io
import tarfile

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from model import ConditionalUNet

# ——— Config ———
TIMESTEPS = 1000
# (height, width) must match training: 1920×1080
HR_SIZE   = (1920, 1080)
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ——— Diffusion schedule ———
def cosine_beta_schedule(T, s=0.008):
    x  = torch.linspace(0, T, T+1, device=DEVICE)
    cp = torch.cos(((x / T) + s) / (1 + s) * math.pi / 2)**2
    cp = cp / cp[0]
    return (1 - cp[1:] / cp[:-1]).clamp(0, 0.999)

BETAS          = cosine_beta_schedule(TIMESTEPS)
ALPHAS         = 1 - BETAS
ALPHAS_CUMPROD = torch.cumprod(ALPHAS, dim=0)

# ——— Load model ———
def load_model(ckpt_path):
    model = ConditionalUNet().to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model

# ——— Sampling function ———
def sample(model, lr_tensor):
    b, c, h, w = lr_tensor.shape
    x = torch.randn_like(lr_tensor, device=DEVICE)
    for t in reversed(range(TIMESTEPS)):
        t_batch = torch.full((b,), t, device=DEVICE, dtype=torch.long)
        with torch.no_grad():
            eps = model(lr_tensor, x, t_batch)
        a_t  = ALPHAS[t]
        ab_t = ALPHAS_CUMPROD[t]
        x = (1.0 / a_t.sqrt()) * (x - ((1 - a_t) / (1 - ab_t).sqrt()) * eps)
        if t > 0:
            x = x + BETAS[t].sqrt() * torch.randn_like(x)
    return x

# ——— Main ———
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Super-Resolution Diffusion Inference")
    parser.add_argument("--ckpt",   type=str, required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--input",  type=str, required=True,
                        help="Input image, glob (e.g. '*.png'), directory, or tar with index 'archive.tar:idx'")
    parser.add_argument("--output", type=str, default="outputs",
                        help="Directory to save super-res images")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    model = load_model(args.ckpt)

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    inputs = []

    # Handle tar:idx syntax
    if ":" in args.input:
        tar_path, idx_str = args.input.rsplit(":", 1)
        if tar_path.lower().endswith(".tar"):
            idx = int(idx_str)
            with tarfile.open(tar_path, "r") as tar:
                members = sorted([m for m in tar.getmembers() if m.isfile()],
                                 key=lambda m: m.name)
                if idx < 0 or idx >= len(members):
                    raise IndexError(f"Index {idx} out of range for archive {tar_path}")
                m = members[idx]
                f = tar.extractfile(m)
                img = Image.open(io.BytesIO(f.read())).convert("RGB")
                inputs.append((img, os.path.basename(m.name)))
    else:
        # Directory or glob
        paths = []
        if os.path.isdir(args.input):
            paths = glob.glob(os.path.join(args.input, "*"))
        else:
            paths = glob.glob(args.input)
        for path in paths:
            try:
                img = Image.open(path).convert("RGB")
                inputs.append((img, os.path.basename(path)))
            except:
                continue

    if not inputs:
        print("No valid inputs found.")
        exit(1)

    for img, name in inputs:
        # to tensor and normalize
        lr = to_tensor(img).unsqueeze(0).to(DEVICE)
        # upsample to HR_SIZE
        lr_up = F.interpolate(lr, size=HR_SIZE, mode="bilinear", align_corners=False)
        # run reverse diffusion
        out = sample(model, lr_up)
        # denormalize
        out = (out * 0.5 + 0.5).clamp(0, 1)
        # to PIL and save
        out_img = transforms.ToPILImage()(out.squeeze(0).cpu())
        base, ext = os.path.splitext(name)
        save_path = os.path.join(args.output, f"{base}_sr{ext}")
        out_img.save(save_path)
        print(f"Saved super-res image: {save_path}")
