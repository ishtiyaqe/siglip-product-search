#!/usr/bin/env python3
"""
Training script for image embedding model using Google SigLIP base-patch16-224
- Backbone: google/siglip-base-patch16-224 (~86M params, excellent quality)
- Projects 768-dim features → 512-dim normalized embeddings
- Uses ABO small dataset with metadata
- Trains on GPU (if available), saves lightweight .pt for CPU server inference
"""

import os
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler

import pandas as pd
from PIL import Image
from transformers import AutoProcessor, AutoModel
from torchvision import transforms as T
from tqdm.auto import tqdm

from pathlib import Path
import pickle


print(torch.__version__)
print("CUDA available?     ", torch.cuda.is_available())
print("CUDA device count:  ", torch.cuda.device_count())
print("Current device:     ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")



# ─── Paths (adjust if needed) ─────────────────────────────────────────────
BASE_DIR = Path(r"/mnt/d/nobel model traning/Image Embedding model training/images")
METADATA_PATH = BASE_DIR / "metadata" / "images.csv.gz"
IMAGES_ROOT = BASE_DIR / "small"


# ─── ABO Dataset ──────────────────────────────────────────────────────────
CACHE_FILE = Path("valid_abo_indices.pkl")

class ABOSmallDataset(Dataset):
    def __init__(self, metadata_path, images_root, transform=None, use_cache=True):
        print("Loading ABO metadata...")
        self.df = pd.read_csv(metadata_path, compression='gzip')
        
        if 'path' not in self.df.columns:
            raise ValueError("CSV missing 'path' column. Check ABO metadata.")
        
        self.images_root = Path(images_root)
        self.transform = transform
        
        total_rows = len(self.df)
        
        # ─── Cache handling ────────────────────────────────────────────────
        if use_cache and CACHE_FILE.exists():
            print("→ Using cached valid indices (fast startup)")
            with open(CACHE_FILE, 'rb') as f:
                self.valid_indices = pickle.load(f)
        else:
            print("→ Checking which images exist (this may take 20–90 seconds)...")
            self.valid_indices = []
            
            # Progress bar for scanning files
            for idx, row in tqdm(
                self.df.iterrows(),
                total=total_rows,
                desc="Scanning image files",
                unit="img",
                ncols=100,
                miniters=500  # update less often to be faster
            ):
                full_path = self.images_root / row['path']
                if full_path.is_file():
                    self.valid_indices.append(idx)
            
            # Save cache for next runs
            with open(CACHE_FILE, 'wb') as f:
                pickle.dump(self.valid_indices, f)
            print(f"→ Cache created: {CACHE_FILE}")

        # Apply filtering
        self.df = self.df.iloc[self.valid_indices].reset_index(drop=True)
        
        valid_count = len(self.valid_indices)
        print(f"\nMetadata entries: {total_rows:,}")
        print(f"Valid images found: {valid_count:,} ({valid_count/total_rows*100:.1f}%)")
        print(f"Missing/invalid images: {total_rows - valid_count:,}")
    
    def __len__(self):
        """Required by DataLoader - returns number of samples"""
        return len(self.valid_indices)  # or len(self.df) - both are the same now

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.images_root / row['path']
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Warning: Failed to load {img_path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))  # fallback black image

        if self.transform:
            image = self.transform(image)
            
        return image, 0  # dummy label for self-supervised training    
        
        
        
# ─── SigLIP-based Model with Projection to 512-dim ────────────────────────
class ImageEmbeddingModel(nn.Module):
    def __init__(self, model_name: str = "google/siglip-base-patch16-224", output_dim: int = 512):
        super().__init__()
        print(f"Loading SigLIP backbone: {model_name}")
        self.backbone = AutoModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)

        # SigLIP base → 768-dim image features (pooled)
        self.projection = nn.Sequential(
            nn.Linear(768, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, pixel_values):
        # Get image features (pooled output after projection in SigLIP)
        outputs = self.backbone.get_image_features(pixel_values=pixel_values)
        emb = self.projection(outputs)
        return F.normalize(emb, p=2, dim=-1)  # L2 normalized 512-dim


# ─── Main Training ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train SigLIP-based embedding on ABO small")
    parser.add_argument('--batch_size', type=int, default=96)   # lower than MobileCLIP due to ~86M params
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=5e-5)       # SigLIP likes smaller LR
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_abo_siglip_base')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=6)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device} | Batch size: {args.batch_size}")

    # ─── Transforms (SigLIP uses mean/std 0.5) ────────────────────────────
    train_transform = T.Compose([
        T.RandomResizedCrop(224, scale=(0.75, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # SigLIP standard
    ])

    dataset = ABOSmallDataset(
        metadata_path=METADATA_PATH,
        images_root=IMAGES_ROOT,
        transform=train_transform
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True
    )

    # ─── Model + Optimizer ────────────────────────────────────────────────
    model = ImageEmbeddingModel(output_dim=512).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(loader), eta_min=1e-6)
    
    scaler = GradScaler(enabled=(device.type == "cuda"))

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_loss = float('inf')

    print("\n" + "="*80)
    print("Starting training with google/siglip-base-patch16-224 → 512-dim embeddings")
    print("="*80 + "\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        start_time = time.time()

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")

        for images, _ in pbar:
            images = images.to(device)
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=(device.type == "cuda")):
                emb = model(images)
                # Symmetric InfoNCE (temperature-scaled)
                logits = emb @ emb.T * 20.0
                labels = torch.arange(len(emb), device=device)
                loss = (F.cross_entropy(logits, labels) + 
                        F.cross_entropy(logits.T, labels)) / 2

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.5f}")

        avg_loss = total_loss / len(loader)
        epoch_time = time.time() - start_time

        print(f"Epoch {epoch:3d} | Loss: {avg_loss:.5f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e} | Time: {epoch_time:.0f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = Path(args.checkpoint_dir) / "best_model.pt"
            torch.save(model.state_dict(), save_path)
            print(f"  → New best saved: {save_path} (loss: {avg_loss:.5f})")

    print("\n" + "="*80)
    print("Training finished!")
    print(f"Best model: {save_path}")
    print("Load on CPU server like this:")
    print("""
model = ImageEmbeddingModel("google/siglip-base-patch16-224", 512)
model.load_state_dict(torch.load("best_model.pt", map_location="cpu"))
model.eval()
# Then use model.processor and model(pixel_values) for inference
""")
    print("="*80)


if __name__ == "__main__":
    main()