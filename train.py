# Updated train.py: only the loss function changed from L1 to L2 (MSE)

import os
import tarfile
import tempfile
import random
import math
import argparse
import logging
import shutil
import pickle
import io
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFile, UnidentifiedImageError

from model import ConditionalUNet
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Config
TIMESTEPS        = 1000
BATCH_SIZE       = 1
ACCUM_STEPS      = 32
LR               = 1e-4
NUM_EPOCHS       = 10
SEED             = 42
DRIVE_FOLDER_ID  = '15cYAaK0aKkOroXAWNAPvqd6jYLLhBuny'
CREDENTIALS_PATH = '/home/6/uj05056/my_tool/credentials.json'
TOKEN_PATH       = '/home/6/uj05056/my_tool/token.pickle'
DEVICE           = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Logging
logging.basicConfig(
    filename='train.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

# Diffusion schedule
def cosine_beta_schedule(T, s=0.008):
    x = torch.linspace(0, T, T+1, device=DEVICE)
    cp = torch.cos(((x / T) + s) / (1 + s) * math.pi / 2)**2
    cp = cp / cp[0]
    return (1 - cp[1:] / cp[:-1]).clamp(0, 0.999)

BETAS = cosine_beta_schedule(TIMESTEPS)
ALPHAS_CUMPROD = torch.cumprod(1 - BETAS, dim=0)

# Dataset
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return []
    lr, hr = zip(*batch)
    return torch.stack(lr, 0), torch.stack(hr, 0)

class TarDataset(Dataset):
    def __init__(self, tar_path, lr_size=(576,1024), hr_size=(1080,1920)):
        self.tar = tarfile.open(tar_path)
        self.members = [m for m in self.tar.getmembers() if m.isfile()]
        self.lr_tf = transforms.Compose([
            transforms.Resize((lr_size[1], lr_size[0])),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        self.hr_tf = transforms.Compose([
            transforms.Resize((hr_size[1], hr_size[0])),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
    def __len__(self): return len(self.members)
    def __getitem__(self, idx):
        m = self.members[idx]
        try:
            f = self.tar.extractfile(m)
            img = Image.open(f).convert('RGB')
        except:
            return None
        hr = self.hr_tf(img)
        lr = self.lr_tf(img)
        lr_up = F.interpolate(lr.unsqueeze(0),
                              size=hr.shape[1:],
                              mode='bilinear',
                              align_corners=False).squeeze(0)
        return lr_up, hr
    def close(self): self.tar.close()

# Drive API helpers
SCOPES = ['https://www.googleapis.com/auth/drive']
def get_drive_service():
    creds = None
    if os.path.exists(TOKEN_PATH):
        with open(TOKEN_PATH, 'rb') as f:
            creds = pickle.load(f)
    if creds and creds.expired and getattr(creds, 'refresh_token', None):
        creds.refresh(Request())
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
        creds = flow.run_local_server(port=0, open_browser=False)
    with open(TOKEN_PATH, 'wb') as f:
        pickle.dump(creds, f)
    return build('drive','v3',credentials=creds)

def download_tar(service, file_id, local_path):
    req = service.files().get_media(fileId=file_id)
    fh = io.FileIO(local_path, 'wb')
    dl = MediaIoBaseDownload(fh, req)
    done = False
    while not done:
        _, done = dl.next_chunk()
    fh.close()

# Training
def train(use_dream=False, resume_ckpt=None):
    mode = 'DREAM' if use_dream else 'DDPM'
    logging.info(f"Starting training in {mode} mode")

    random.seed(SEED); torch.manual_seed(SEED)
    model = ConditionalUNet().to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler()
    drive = get_drive_service()

    # determine resume position
    start_epoch = 1
    start_part = 1
    if resume_ckpt:
        resume_name = Path(resume_ckpt).stem
        state = torch.load(resume_ckpt, map_location=DEVICE)
        model.load_state_dict(state)
        if resume_name.startswith("part_"):
            # format: part_{part_idx}_epoch_{epoch}
            _, p_str, _, e_str = resume_name.split("_")
            start_part = int(p_str) + 1
            start_epoch = int(e_str)
        elif resume_name.startswith("epoch_"):
            _, e_str = resume_name.split("_")
            start_epoch = int(e_str) + 1
        logging.info(f"Resuming from {resume_ckpt}, epoch {start_epoch}, part {start_part}")

    # list tar files
    resp = drive.files().list(
        q=f"'{DRIVE_FOLDER_ID}' in parents and mimeType='application/x-tar' and trashed=false",
        fields='files(id,name)'
    ).execute()
    tar_files = resp.get('files', [])

    best_loss, best_ckpt = float('inf'), None

    for epoch in range(start_epoch, NUM_EPOCHS+1):
        logging.info(f"Epoch {epoch}/{NUM_EPOCHS} start")
        epoch_loss, batch_count = 0.0, 0
        opt.zero_grad()

        for part_idx, tar_info in enumerate(tar_files, 1):
            # skip earlier parts if resuming
            if epoch == start_epoch and part_idx < start_part:
                continue

            tmpdir = tempfile.mkdtemp()
            path = os.path.join(tmpdir, tar_info['name'])
            download_tar(drive, tar_info['id'], path)
            ds = TarDataset(path)
            loader = DataLoader(ds, batch_size=BATCH_SIZE,
                                shuffle=True, num_workers=4,
                                collate_fn=collate_fn)

            model.train()
            for idx, data in enumerate(loader, 1):
                if not isinstance(data, tuple): continue
                lr_up, hr = [x.to(DEVICE) for x in data]
                t = torch.randint(0, TIMESTEPS, (1,), device=DEVICE)
                noise = torch.randn_like(hr)
                ab = ALPHAS_CUMPROD[t][:,None,None,None]
                yt = ab.sqrt()*hr + (1-ab).sqrt()*noise

                with torch.cuda.amp.autocast():
                    eps_pred = model(lr_up, yt, t)
                    if use_dream:
                        delta = noise - eps_pred.detach()
                        lam = (1-ab).sqrt()
                        yt_bar = yt + lam*delta
                        loss = F.mse_loss(eps_pred + lam*delta,
                                         model(lr_up, yt_bar, t))
                    else:
                        loss = F.mse_loss(noise, eps_pred)
                    loss = loss / ACCUM_STEPS

                scaler.scale(loss).backward()
                if idx % ACCUM_STEPS == 0:
                    scaler.step(opt); scaler.update(); opt.zero_grad()

                epoch_loss += loss.item() * ACCUM_STEPS
                batch_count += 1
                logging.info(f"Epoch {epoch} Part {part_idx} Batch {batch_count} Loss {loss.item():.6f}")

            ds.close(); shutil.rmtree(tmpdir)

            # part-level checkpoint
            ckpt_name = f"part_{part_idx}_epoch_{epoch}.pt"
            torch.save(model.state_dict(), ckpt_name)
            logging.info(f"Saved part checkpoint: {ckpt_name}")
            drive.files().create(
                body={'name': ckpt_name, 'parents':[DRIVE_FOLDER_ID]},
                media_body=MediaFileUpload(ckpt_name, resumable=True),
                fields='id'
            ).execute()

        avg_loss = epoch_loss / batch_count if batch_count else float('nan')
        logging.info(f"Epoch {epoch} Average Loss: {avg_loss:.6f}")

        # epoch-level checkpoint
        epoch_ckpt = f"epoch_{epoch}.pt"
        torch.save(model.state_dict(), epoch_ckpt)
        if avg_loss < best_loss:
            if best_ckpt and os.path.exists(best_ckpt):
                os.remove(best_ckpt)
            best_loss, best_ckpt = avg_loss, epoch_ckpt
        else:
            os.remove(epoch_ckpt)

        torch.cuda.empty_cache()
        logging.info(f"Epoch {epoch} complete. Best loss so far: {best_loss:.6f}")

    logging.info(f"Training finished. Best checkpoint: {best_ckpt}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dream',  action='store_true', help='Enable DREAM loss')
    parser.add_argument('--resume', type=str,       help='Checkpoint path to resume from')
    args = parser.parse_args()
    train(use_dream=args.dream, resume_ckpt=args.resume)

# Save this as train.py in your working directory.
