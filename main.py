#!/usr/bin/env python3
"""
Fast production-ready training script for product image embeddings
- Backbone: google/siglip-base-patch16-224 + LoRA
- Output: 512-dim L2-normalized embeddings
- Dataset: Amazon Berkeley Objects (ABO) small
- With train/validation split + validation loss monitoring
"""

import os
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler

import pandas as pd
from PIL import Image
from transformers import AutoProcessor, AutoModel
from torchvision import transforms as T
from tqdm.auto import tqdm
from peft import LoraConfig, get_peft_model

import pickle

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("-" * 60)

# ─── Paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(r"/mnt/d/nobel model traning/Image Embedding model training/images")
METADATA_PATH = BASE_DIR / "metadata" / "images.csv.gz"
IMAGES_ROOT = BASE_DIR / "small"
CACHE_FILE = Path("valid_abo_indices.pkl")


# ─── Dataset ──────────────────────────────────────────────────────────────
class ABOSmallDataset(Dataset):
    def __init__(self, metadata_path, images_root, transform=None, use_cache=True, sample_fraction=1.0):
        print("Loading ABO metadata...")
        self.df = pd.read_csv(metadata_path, compression='gzip')
        
        if 'path' not in self.df.columns:
            raise ValueError("CSV missing 'path' column")

        if sample_fraction < 1.0:
            print(f"Sampling {sample_fraction*100:.1f}% of data...")
            self.df = self.df.sample(frac=sample_fraction, random_state=42).reset_index(drop=True)

        self.images_root = Path(images_root)
        self.transform = transform
        total_rows = len(self.df)

        if use_cache and CACHE_FILE.exists():
            print("→ Using cached valid indices")
            with open(CACHE_FILE, 'rb') as f:
                self.valid_indices = pickle.load(f)
        else:
            print("→ Scanning files...")
            self.valid_indices = []
            for idx, row in tqdm(self.df.iterrows(), total=total_rows, desc="Scanning", unit="img", miniters=500):
                if (self.images_root / row['path']).is_file():
                    self.valid_indices.append(idx)
            with open(CACHE_FILE, 'wb') as f:
                pickle.dump(self.valid_indices, f)
            print(f"Cache saved: {CACHE_FILE}")

        self.df = self.df.iloc[self.valid_indices].reset_index(drop=True)
        print(f"Valid images: {len(self.df):,} / {total_rows:,}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            img = Image.open(self.images_root / row['path']).convert("RGB")
        except:
            img = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            img = self.transform(img)
        return img, 0


# ─── Model with LoRA ──────────────────────────────────────────────────────
class ImageEmbeddingModel(nn.Module):
    def __init__(self, model_name="google/siglip-base-patch16-224", output_dim=512):
        super().__init__()
        print(f"Loading {model_name} with LoRA...")
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # Safe LoRA config - no task_type
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        
        try:
            self.backbone.vision_model = get_peft_model(self.backbone.vision_model, lora_config)
        except Exception as e:
            print(f"LoRA wrap failed: {e}")
            print("Falling back to full fine-tuning (slower but stable)...")
            # Optional fallback: no LoRA if it crashes

        self.processor = AutoProcessor.from_pretrained(model_name)

        self.projection = nn.Sequential(
            nn.Linear(768, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, pixel_values):
        features = self.backbone.get_image_features(pixel_values)
        emb = self.projection(features)
        return F.normalize(emb, p=2, dim=-1)

@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    
    pbar = tqdm(val_loader, desc="Validation", leave=False)
    for images, _ in pbar:
        images = images.to(device)
        with autocast(device_type='cuda', dtype=torch.float16):
            emb = model(images)
            logits = emb @ emb.T * 20.0
            labels = torch.arange(len(emb), device=device)
            loss = (F.cross_entropy(logits, labels) + 
                    F.cross_entropy(logits.T, labels)) / 2
        
        total_loss += loss.item()
    
    return total_loss / len(val_loader)


# ─── Main Training ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fast SigLIP product embedding training with validation")
    parser.add_argument('--batch_size', type=int, default=96)
    parser.add_argument('--accum_steps', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_siglip_lora')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--sample_frac', type=float, default=1.0)
    parser.add_argument('--val_split', type=float, default=0.2, help="Fraction for validation set")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Batch: {args.batch_size} | Accum: ×{args.accum_steps}")

    # Transforms
    train_transform = T.Compose([
        T.RandomResizedCrop(224, scale=(0.75, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2, 0.2),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3),
    ])

    val_transform = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3),
    ])

    # Full dataset (with transform=None first)
    full_dataset = ABOSmallDataset(
        METADATA_PATH, IMAGES_ROOT, transform=None,
        sample_fraction=args.sample_frac
    )

    # Split into train/val
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Apply transforms after split
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,  # larger batch for validation (faster)
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"Train samples: {len(train_dataset):,} | Val samples: {len(val_dataset):,}")

    model = ImageEmbeddingModel(output_dim=512).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader), eta_min=1e-6)
    
    scaler = GradScaler('cuda') if device.type == "cuda" else None

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf')
    patience = 4
    patience_counter = 0

    print("="*70)
    print("Starting training with validation")
    print("="*70 + "\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0
        start = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")

        for step, (images, _) in enumerate(pbar):
            images = images.to(device)

            with autocast(device_type='cuda', dtype=torch.float16):
                emb = model(images)
                logits = emb @ emb.T * 20.0
                labels = torch.arange(len(emb), device=device)
                loss = (F.cross_entropy(logits, labels) + 
                        F.cross_entropy(logits.T, labels)) / 2

            loss = loss / args.accum_steps
            scaler.scale(loss).backward()

            if (step + 1) % args.accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            total_train_loss += loss.item() * args.accum_steps
            pbar.set_postfix(loss=f"{loss.item()*args.accum_steps:.5f}")

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        val_loss = validate(model, val_loader, device)

        epoch_time = time.time() - start
        print(f"Epoch {epoch:2d} | Train Loss: {avg_train_loss:.5f} | Val Loss: {val_loss:.5f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e} | {epoch_time:.0f}s")

        # Save best based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = Path(args.checkpoint_dir) / "best_model.pt"
            torch.save(model.state_dict(), save_path)
            print(f"  → New best model saved! (Val Loss: {val_loss:.5f})")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break

    print("\nTraining completed!")
    print(f"Best checkpoint (lowest val loss): {save_path}")
    print("\nInference example:")
    print("""
model = ImageEmbeddingModel()
model.load_state_dict(torch.load("best_model.pt", map_location="cpu"))
model.eval()
# Use model.processor and model(pixel_values) for 512-dim embeddings
""")


if __name__ == "__main__":
    main()