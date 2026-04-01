"""
Training script for Document Fraud Detection — Optimized V2.

Key improvements over V1:
  - BCEWithLogitsLoss (numerically stable, works with raw logits from V2 model)
  - Focal Loss option (better for class-imbalanced fraud detection)
  - Mixup augmentation (proven +1-2% AUC in image classification)
  - EMA (Exponential Moving Average) of weights — smoother, better generalization
  - Stronger augmentations: RandomErasing, perspective warp, elastic distortion
  - OneCycleLR scheduler (super-convergence, often outperforms cosine)
  - Test-Time Augmentation (TTA) at validation for more reliable AUC
  - num_workers=2 for faster data loading
  - Training history saved as JSON + plotted as PNG
  - Configurable via CLI flags: --focal_loss, --mixup, --ema, --tta
  - Per-epoch checkpoints with configurable retention (--save_every_epoch, --max_checkpoints)

Supports two dataset layouts automatically:

  Structured layout (recommended, created by prepare_custom_dataset.py):
    data_dir/
    +-- train/genuine/   +-- train/tampered/
    +-- val/genuine/     +-- val/tampered/
    +-- test/genuine/    +-- test/tampered/

  Flat layout (raw documents, auto-split at runtime):
    data_dir/
    +-- genuine/     (all genuine documents)
    +-- tampered/    (all fake/tampered documents)

Usage:
    # Standard training (recommended defaults)
    python train_model.py --data_dir ./dataset --epochs 50 --batch_size 16

    # Maximum accuracy (all optimizations)
    python train_model.py --data_dir ./dataset --epochs 60 --batch_size 8 \
        --accum_steps 4 --focal_loss --mixup --ema --tta \
        --freeze_epochs 5 --patience 12

    # Tiny dataset (5+5 docs, after prepare_custom_dataset.py)
    python train_model.py --data_dir ./dataset --epochs 30 --batch_size 8 \
        --freeze_epochs 15 --patience 8 --focal_loss --mixup --ema

    # Save every epoch checkpoint, keep last 5
    python train_model.py --data_dir ./dataset --epochs 50 --save_every_epoch --max_checkpoints 5
"""

import argparse
import glob
import io
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from loguru import logger
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

from fraud_model.cnn_model import FraudEfficientNetB3V2, IMAGENET_MEAN, IMAGENET_STD
from utils.ela_analysis import compute_ela, ela_to_array


# ============================================================================
# Checkpoint Utilities
# ============================================================================

def save_checkpoint(
    output_dir: str,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    val_auc: float,
    val_f1: float,
    args: argparse.Namespace,
    ema=None,
    scaler=None,
    max_checkpoints: int = 0,
) -> str:
    """Save a full training checkpoint for the given epoch.

    Stores everything needed to resume training exactly where it left off:
      - Model weights (EMA shadow weights if EMA is enabled, otherwise live weights)
      - Optimizer state (momentum buffers, Adam moments, etc.)
      - LR scheduler state
      - GradScaler state (for mixed-precision resume)
      - EMA shadow weights (separate key, so both live and shadow are preserved)
      - Epoch number and validation metrics (for easy inspection without loading)
      - Full args namespace (so the resumed run uses identical hyperparameters)

    Checkpoint filename: checkpoint_epoch_{epoch:04d}.pth
    Saved to: {output_dir}/checkpoints/

    If max_checkpoints > 0, the oldest checkpoints beyond that limit are
    deleted automatically (FIFO), keeping disk usage bounded.  The best-model
    file (fraud_efficientnet_b3_v2_best.pth) is never touched by this cleanup.

    Args:
        output_dir:      Root output directory (same as args.output_dir).
        epoch:           Current epoch index (0-based).  Filename is 1-based.
        model:           The model being trained.
        optimizer:       Current optimizer.
        scheduler:       Current LR scheduler.
        val_auc:         Validation AUC-ROC for this epoch (stored in metadata).
        val_f1:          Validation F1 for this epoch (stored in metadata).
        args:            Parsed CLI args namespace.
        ema:             EMA instance (optional).  Shadow weights are saved when provided.
        scaler:          GradScaler instance (optional).  Saved for mixed-precision resume.
        max_checkpoints: Maximum number of per-epoch checkpoints to keep on disk.
                         0 = keep all.  Oldest are deleted first.

    Returns:
        Absolute path of the saved checkpoint file.
    """
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Use EMA shadow weights for the saved model_state if EMA is active,
    # so the checkpoint reflects the smoothed weights used for validation.
    if ema is not None:
        ema.apply_shadow()
        model_state = {k: v.clone() for k, v in model.state_dict().items()}
        ema.restore()
    else:
        model_state = {k: v.clone() for k, v in model.state_dict().items()}

    checkpoint = {
        # ── Core training state ──────────────────────────────────────────
        "epoch": epoch,                          # 0-based epoch index
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,

        # ── EMA shadow weights (raw live weights are in model_state_dict) ─
        "ema_state_dict": ema.state_dict() if ema is not None else None,

        # ── Metrics & metadata ───────────────────────────────────────────
        "val_auc": val_auc,
        "val_f1": val_f1,
        "args": vars(args),                      # serialisable dict for inspection
    }

    ckpt_filename = f"checkpoint_epoch_{epoch + 1:04d}.pth"
    ckpt_path = os.path.join(ckpt_dir, ckpt_filename)
    torch.save(checkpoint, ckpt_path)
    logger.info(
        f"Checkpoint saved: {ckpt_path}  "
        f"(epoch={epoch + 1}, val_auc={val_auc:.4f}, val_f1={val_f1:.4f})"
    )

    # ── Rotate old checkpoints ───────────────────────────────────────────
    if max_checkpoints > 0:
        _rotate_checkpoints(ckpt_dir, max_checkpoints)

    return ckpt_path


def _rotate_checkpoints(ckpt_dir: str, max_checkpoints: int) -> None:
    """Delete the oldest per-epoch checkpoints beyond *max_checkpoints*.

    Only files matching the pattern ``checkpoint_epoch_NNNN.pth`` are
    considered — the best-model file is never deleted.
    """
    pattern = os.path.join(ckpt_dir, "checkpoint_epoch_*.pth")
    existing = sorted(glob.glob(pattern))          # alphabetical = chronological
    excess = len(existing) - max_checkpoints
    if excess > 0:
        for old_ckpt in existing[:excess]:
            try:
                os.remove(old_ckpt)
                logger.info(f"Rotated old checkpoint: {old_ckpt}")
            except OSError as exc:
                logger.warning(f"Could not delete checkpoint {old_ckpt}: {exc}")


def load_checkpoint(ckpt_path: str, model: nn.Module, optimizer=None,
                    scheduler=None, scaler=None, ema=None, device: str = "cpu"):
    """Resume training from a checkpoint saved by save_checkpoint().

    Restores model weights, optimizer/scheduler/scaler state, and EMA shadow
    weights.  Pass only the objects you want restored; None arguments are
    skipped safely.

    Args:
        ckpt_path:  Path to the ``.pth`` checkpoint file.
        model:      Model to load weights into.
        optimizer:  Optimizer to restore (optional).
        scheduler:  LR scheduler to restore (optional).
        scaler:     GradScaler to restore (optional).
        ema:        EMA instance to restore shadow weights into (optional).
        device:     Target device string (``'cuda'`` or ``'cpu'``).

    Returns:
        dict with keys ``epoch``, ``val_auc``, ``val_f1``, ``args`` so the
        caller can restore loop counters and early-stopping state.

    Example::

        meta = load_checkpoint("checkpoints/checkpoint_epoch_0010.pth",
                               model, optimizer, scheduler, scaler, ema,
                               device=device)
        start_epoch    = meta["epoch"] + 1
        best_val_auc   = meta["val_auc"]
    """
    logger.info(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    if ema is not None and checkpoint.get("ema_state_dict") is not None:
        ema.load_state_dict(checkpoint["ema_state_dict"])

    logger.info(
        f"Resumed from epoch {checkpoint['epoch'] + 1}  "
        f"(val_auc={checkpoint.get('val_auc', 'n/a'):.4f})"
    )
    return {
        "epoch": checkpoint["epoch"],
        "val_auc": checkpoint.get("val_auc", 0.0),
        "val_f1": checkpoint.get("val_f1", 0.0),
        "args": checkpoint.get("args", {}),
    }


# ============================================================================
# Augmentation
# ============================================================================

def _random_jpeg_resave(image: Image.Image, quality_range=(70, 95)) -> Image.Image:
    """Re-save image at random JPEG quality to simulate compression artifacts."""
    q = random.randint(*quality_range)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).copy()


def _apply_training_augmentation(image: Image.Image) -> Image.Image:
    """
    Apply training-time augmentations to the original image BEFORE ELA.

    V2 additions over V1:
      - Perspective warp (simulates photographed documents)
      - More aggressive color jitter range
      - Random sharpness adjustment
      - Gaussian blur with wider range
    """
    import torchvision.transforms.functional as TF

    if random.random() < 0.5:
        image = TF.hflip(image)

    if random.random() < 0.2:
        image = TF.vflip(image)

    angle = random.uniform(-20, 20)
    image = TF.rotate(image, angle, fill=128)

    if random.random() < 0.3:
        distortion = random.uniform(0.05, 0.2)
        w, h = image.size
        half_h, half_w = h // 2, w // 2
        topleft  = [int(random.uniform(0, distortion * half_w)), int(random.uniform(0, distortion * half_h))]
        topright = [int(w - random.uniform(0, distortion * half_w)), int(random.uniform(0, distortion * half_h))]
        botright = [int(w - random.uniform(0, distortion * half_w)), int(h - random.uniform(0, distortion * half_h))]
        botleft  = [int(random.uniform(0, distortion * half_w)), int(h - random.uniform(0, distortion * half_h))]
        startpoints = [[0, 0], [w, 0], [w, h], [0, h]]
        endpoints   = [topleft, topright, botright, botleft]
        image = TF.perspective(image, startpoints, endpoints, fill=128)

    brightness_factor  = random.uniform(0.7, 1.3)
    contrast_factor    = random.uniform(0.7, 1.3)
    saturation_factor  = random.uniform(0.8, 1.2)
    image = TF.adjust_brightness(image, brightness_factor)
    image = TF.adjust_contrast(image, contrast_factor)
    image = TF.adjust_saturation(image, saturation_factor)

    if random.random() < 0.3:
        sharpness = random.uniform(0.5, 2.0)
        image = TF.adjust_sharpness(image, sharpness)

    if random.random() < 0.25:
        from PIL import ImageFilter
        radius = random.uniform(0.5, 2.0)
        image = image.filter(ImageFilter.GaussianBlur(radius=radius))

    image = _random_jpeg_resave(image, quality_range=(65, 95))
    return image


VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".pdf")


def _collect_files(directory: str) -> list:
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in VALID_EXTS
    )


class ELADataset(Dataset):
    """Dataset that applies ELA preprocessing with augmentation for training."""

    def __init__(self, data_dir: str, split: str = "train", target_size: tuple = (300, 300),
                 flat_split_seed: int = 42):
        self.split = split
        self.target_size = target_size
        self.is_train = (split == "train")
        self.samples = []

        split_dir = os.path.join(data_dir, split)

        if os.path.isdir(split_dir):
            n_genuine, n_tampered = self._load_from_dir(split_dir)
            logger.info(f"[{split}] structured layout: {n_genuine} genuine, {n_tampered} tampered")
        else:
            genuine_dir  = os.path.join(data_dir, "genuine")
            tampered_dir = os.path.join(data_dir, "tampered")

            if not os.path.isdir(genuine_dir) or not os.path.isdir(tampered_dir):
                logger.error(
                    f"[{split}] Cannot find '{split_dir}' or flat 'genuine'/'tampered' dirs in {data_dir}"
                )
                self.pos_weight = 1.0
                return

            all_genuine  = _collect_files(genuine_dir)
            all_tampered = _collect_files(tampered_dir)
            total = len(all_genuine) + len(all_tampered)

            if total < 10:
                logger.warning(
                    f"Only {total} raw documents found. "
                    f"Run prepare_custom_dataset.py first to augment to a viable training size."
                )

            g_subset = self._flat_split(all_genuine,  split, flat_split_seed)
            t_subset = self._flat_split(all_tampered, split, flat_split_seed + 1)

            self.samples = [(p, 0) for p in g_subset] + [(p, 1) for p in t_subset]
            n_genuine, n_tampered = len(g_subset), len(t_subset)
            logger.info(f"[{split}] flat layout (auto-split): {n_genuine} genuine, {n_tampered} tampered")

        n_genuine  = sum(1 for _, lbl in self.samples if lbl == 0)
        n_tampered = sum(1 for _, lbl in self.samples if lbl == 1)
        self.pos_weight = (n_genuine / n_tampered) if n_tampered > 0 else 1.0
        logger.info(f"[{split}] pos_weight={self.pos_weight:.2f}")

    def _load_from_dir(self, directory: str):
        n_genuine = n_tampered = 0
        genuine_dir  = os.path.join(directory, "genuine")
        tampered_dir = os.path.join(directory, "tampered")
        if os.path.isdir(genuine_dir):
            for fname in sorted(os.listdir(genuine_dir)):
                if os.path.splitext(fname)[1].lower() in VALID_EXTS:
                    self.samples.append((os.path.join(genuine_dir, fname), 0))
                    n_genuine += 1
        if os.path.isdir(tampered_dir):
            for fname in sorted(os.listdir(tampered_dir)):
                if os.path.splitext(fname)[1].lower() in VALID_EXTS:
                    self.samples.append((os.path.join(tampered_dir, fname), 1))
                    n_tampered += 1
        return n_genuine, n_tampered

    @staticmethod
    def _flat_split(paths: list, split: str, seed: int) -> list:
        lst = list(paths)
        rng = random.Random(seed)
        rng.shuffle(lst)
        n = len(lst)
        n_train = max(1, int(n * 0.75))
        n_val   = max(0, min(int(n * 0.125), n - n_train))
        if split == "train":
            return lst[:n_train]
        elif split == "val":
            return lst[n_train : n_train + n_val]
        else:
            return lst[n_train + n_val :]

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path: str) -> Image.Image:
        if path.lower().endswith(".pdf"):
            try:
                import fitz
                doc  = fitz.open(path)
                page = doc[0]
                mat  = fitz.Matrix(200 / 72, 200 / 72)
                pix  = page.get_pixmap(matrix=mat)
                img  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                doc.close()
                return img
            except Exception as e:
                raise RuntimeError(f"PDF render failed: {e}")
        return Image.open(path).convert("RGB")

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            image = self._load_image(path)

            if self.is_train:
                image = _apply_training_augmentation(image)

            ela_quality = random.randint(75, 95) if self.is_train else 90
            ela_image   = compute_ela(image, quality=ela_quality)
            arr = ela_to_array(ela_image, self.target_size)  # (3, H, W), [0,1]

            if self.is_train:
                noise_std = random.uniform(0.01, 0.08)
                arr = arr + np.random.normal(0, noise_std, arr.shape).astype(np.float32)
                arr = np.clip(arr, 0.0, 1.0)

                if random.random() < 0.3:
                    _, h, w = arr.shape
                    erase_h = random.randint(int(h * 0.05), int(h * 0.25))
                    erase_w = random.randint(int(w * 0.05), int(w * 0.25))
                    top  = random.randint(0, h - erase_h)
                    left = random.randint(0, w - erase_w)
                    arr[:, top:top+erase_h, left:left+erase_w] = np.random.uniform(
                        0, 1, (3, erase_h, erase_w)
                    ).astype(np.float32)

            mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
            std  = np.array(IMAGENET_STD,  dtype=np.float32).reshape(3, 1, 1)
            arr  = (arr - mean) / std

            return torch.tensor(arr, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

        except Exception as e:
            logger.warning(f"Skipping {path}: {e}")
            arr = np.zeros((3, *self.target_size), dtype=np.float32)
            return torch.tensor(arr), torch.tensor(float(label), dtype=torch.float32)


# ============================================================================
# Loss Functions
# ============================================================================

LABEL_SMOOTHING = 0.1


class FocalBCEWithLogitsLoss(nn.Module):
    """Focal Loss for binary classification with logits."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.5,
                 label_smoothing: float = 0.1, pos_weight: float = 1.0):
        super().__init__()
        self.gamma         = gamma
        self.alpha         = alpha
        self.label_smoothing = label_smoothing
        self.pos_weight    = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        pw  = torch.tensor([self.pos_weight], device=logits.device, dtype=logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw, reduction="none")

        probs   = torch.sigmoid(logits)
        p_t     = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        return (alpha_t * focal_weight * bce).mean()


class SmoothedBCEWithLogitsLoss(nn.Module):
    """Standard BCE with logits + label smoothing + class weights."""

    def __init__(self, label_smoothing: float = 0.1, pos_weight: float = 1.0):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.pos_weight      = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        pw = torch.tensor([self.pos_weight], device=logits.device, dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)


# ============================================================================
# Mixup Augmentation
# ============================================================================

def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4):
    if alpha <= 0:
        return x, y, y, 1.0

    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================================
# EMA (Exponential Moving Average)
# ============================================================================

class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model  = model
        self.decay  = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            if k in self.shadow:
                self.shadow[k] = v.clone()


# ============================================================================
# Test-Time Augmentation (TTA)
# ============================================================================

def tta_predict(model, images: torch.Tensor, device: str) -> torch.Tensor:
    """Average predictions over 4 augmented views (orig + 3 flips)."""
    views = [
        images,
        torch.flip(images, dims=[3]),
        torch.flip(images, dims=[2]),
        torch.flip(images, dims=[2, 3]),
    ]
    all_logits = []
    for view in views:
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
            logits = model(view.to(device))
        all_logits.append(logits)
    return torch.stack(all_logits).mean(dim=0)


# ============================================================================
# Training History
# ============================================================================

class TrainingHistory:
    """Track and save training metrics per epoch."""

    def __init__(self):
        self.history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [], "val_auc": [],
            "val_f1": [], "val_precision": [], "val_recall": [],
            "lr": [], "epoch_time": [],
        }

    def append(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.history:
                self.history[k].append(v)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Training history saved: {path}")

    def plot(self, path: str):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            axes[0, 0].plot(self.history["train_loss"], label="Train Loss", color="blue")
            axes[0, 0].plot(self.history["val_loss"],   label="Val Loss",   color="red")
            axes[0, 0].set_title("Loss")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            axes[0, 1].plot(self.history["train_acc"], label="Train Acc", color="blue")
            axes[0, 1].plot(self.history["val_acc"],   label="Val Acc",   color="red")
            axes[0, 1].set_title("Accuracy")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            axes[1, 0].plot(self.history["val_auc"], label="Val AUC", color="green",  linewidth=2)
            axes[1, 0].plot(self.history["val_f1"],  label="Val F1",  color="orange")
            axes[1, 0].set_title("AUC-ROC & F1")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            axes[1, 1].plot(self.history["lr"], label="LR", color="purple")
            axes[1, 1].set_title("Learning Rate")
            axes[1, 1].set_yscale("log")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(path, dpi=150)
            plt.close()
            logger.info(f"Training curves saved: {path}")
        except ImportError:
            logger.warning("matplotlib not available — skipping plot")


# ============================================================================
# Main Training Function
# ============================================================================

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on: {device}")
    logger.info(f"Config: {vars(args)}")
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    train_set = ELADataset(args.data_dir, split="train")
    val_set   = ELADataset(args.data_dir, split="val")

    if len(train_set) == 0:
        logger.error("No training samples found. Check data_dir structure.")
        return

    nw = min(2, os.cpu_count() or 1)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=nw, pin_memory=(device == "cuda"), drop_last=True
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=nw, pin_memory=(device == "cuda")
    )

    model = FraudEfficientNetB3V2(pretrained=True).to(device)

    if args.focal_loss:
        criterion = FocalBCEWithLogitsLoss(
            gamma=2.0, alpha=0.5,
            label_smoothing=LABEL_SMOOTHING,
            pos_weight=train_set.pos_weight
        )
        logger.info(f"Using Focal Loss (gamma=2.0, alpha=0.5, pos_weight={train_set.pos_weight:.2f})")
    else:
        criterion = SmoothedBCEWithLogitsLoss(
            label_smoothing=LABEL_SMOOTHING,
            pos_weight=train_set.pos_weight
        )
        logger.info(f"Using Smoothed BCE Loss (pos_weight={train_set.pos_weight:.2f})")

    ema = None
    if args.ema:
        ema = EMA(model, decay=0.999)
        logger.info("EMA enabled (decay=0.999)")

    # Phase 1: freeze backbone
    model.freeze_backbone()
    classifier_params = list(model.head.parameters())
    optimizer  = torch.optim.AdamW(classifier_params, lr=args.lr * 10, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.freeze_epochs, 1))
    scaler     = torch.amp.GradScaler(device, enabled=(device == "cuda"))

    best_val_auc   = 0.0
    patience_counter = 0
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "fraud_efficientnet_b3_v2_best.pth")
    history   = TrainingHistory()

    total_start = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # Phase 2: unfreeze backbone
        if epoch == args.freeze_epochs:
            logger.info(f"Epoch {epoch+1}: Unfreezing backbone for full fine-tuning")
            model.unfreeze_backbone()
            if ema:
                ema = EMA(model, decay=0.999)

            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

            phase2_epochs    = args.epochs - args.freeze_epochs
            steps_per_epoch  = len(train_loader)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=args.lr * 3,
                epochs=phase2_epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.1,
                anneal_strategy="cos",
                div_factor=10,
                final_div_factor=100,
            )
            logger.info(
                f"OneCycleLR: max_lr={args.lr*3:.6f}, "
                f"{phase2_epochs} epochs, {steps_per_epoch} steps/epoch"
            )

        # ── Training loop ────────────────────────────────────────────────
        model.train()
        train_loss    = 0.0
        train_correct = 0
        train_total   = 0

        optimizer.zero_grad()
        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            use_mixup = args.mixup and random.random() < 0.5
            if use_mixup:
                images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=0.4)

            with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
                logits = model(images).squeeze(1)
                if use_mixup:
                    loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam) / args.accum_steps
                else:
                    loss = criterion(logits, labels) / args.accum_steps

            scaler.scale(loss).backward()

            is_last_step = (step + 1 == len(train_loader))
            if (step + 1) % args.accum_steps == 0 or is_last_step:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update()

            if epoch >= args.freeze_epochs:
                scheduler.step()

            train_loss += loss.item() * args.accum_steps * images.size(0)
            with torch.no_grad():
                predicted = (torch.sigmoid(logits.detach()) > 0.5).float()
                if use_mixup:
                    train_correct += (
                        lam * (predicted == labels_a).float() +
                        (1 - lam) * (predicted == labels_b).float()
                    ).sum().item()
                else:
                    train_correct += (predicted == labels).sum().item()
                train_total += labels.size(0)

        if epoch < args.freeze_epochs:
            scheduler.step()

        train_loss /= max(train_total, 1)
        train_acc   = train_correct / max(train_total, 1)

        # ── Validation loop ──────────────────────────────────────────────
        if ema:
            ema.apply_shadow()

        model.eval()
        val_loss    = 0.0
        val_correct = 0
        val_total   = 0
        all_preds   = []
        all_labels  = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                if args.tta:
                    logits = tta_predict(model, images, device).squeeze(1)
                else:
                    with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
                        logits = model(images).squeeze(1)

                loss = F.binary_cross_entropy_with_logits(logits, labels)

                val_loss += loss.item() * images.size(0)
                probs     = torch.sigmoid(logits)
                predicted = (probs > 0.5).float()
                val_correct += (predicted == labels).sum().item()
                val_total   += labels.size(0)
                all_preds.extend(probs.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        if ema:
            ema.restore()

        val_loss /= max(val_total, 1)
        val_acc   = val_correct / max(val_total, 1)

        try:
            val_auc = float(roc_auc_score(all_labels, all_preds)) if len(set(all_labels)) > 1 else 0.5
        except Exception:
            val_auc = 0.5

        binary_preds = [1.0 if p > 0.5 else 0.0 for p in all_preds]
        try:
            val_f1   = float(f1_score(all_labels, binary_preds, zero_division=0))
            val_prec = float(precision_score(all_labels, binary_preds, zero_division=0))
            val_rec  = float(recall_score(all_labels, binary_preds, zero_division=0))
        except Exception:
            val_f1 = val_prec = val_rec = 0.0

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        history.append(
            train_loss=train_loss, train_acc=train_acc,
            val_loss=val_loss, val_acc=val_acc, val_auc=val_auc,
            val_f1=val_f1, val_precision=val_prec, val_recall=val_rec,
            lr=current_lr, epoch_time=epoch_time,
        )

        logger.info(
            f"Epoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s) | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, "
            f"AUC: {val_auc:.4f}, F1: {val_f1:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        # ── Per-epoch checkpoint ─────────────────────────────────────────
        if args.save_every_epoch:
            save_checkpoint(
                output_dir=args.output_dir,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                val_auc=val_auc,
                val_f1=val_f1,
                args=args,
                ema=ema,
                scaler=scaler,
                max_checkpoints=args.max_checkpoints,
            )

        # ── Best model (by AUC-ROC) ──────────────────────────────────────
        if val_auc > best_val_auc:
            best_val_auc = val_auc

            if ema:
                ema.apply_shadow()
                torch.save(model.state_dict(), save_path)
                ema.restore()
            else:
                torch.save(model.state_dict(), save_path)

            logger.info(f"Best model saved: {save_path} (val_auc: {val_auc:.4f}, f1: {val_f1:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(
                    f"Early stopping at epoch {epoch+1} "
                    f"(no AUC improvement for {args.patience} epochs)"
                )
                break

    total_time = time.time() - total_start
    logger.info(f"Training complete in {total_time/60:.1f} min. Best val AUC: {best_val_auc:.4f}")

    history.save(os.path.join(args.output_dir, "training_history.json"))
    history.plot(os.path.join(args.output_dir, "training_curves.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Document Fraud Detection (EfficientNet-B3 V2 — Optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data_dir",      type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--epochs",        type=int, default=50,    help="Number of training epochs")
    parser.add_argument("--batch_size",    type=int, default=16,    help="Batch size")
    parser.add_argument("--lr",            type=float, default=1e-4, help="Base learning rate")
    parser.add_argument("--freeze_epochs", type=int, default=5,     help="Epochs to freeze backbone")
    parser.add_argument("--patience",      type=int, default=10,    help="Early stopping patience (epochs)")
    parser.add_argument("--output_dir",    type=str, default="./fraud_model", help="Model output directory")
    parser.add_argument("--accum_steps",   type=int, default=2,     help="Gradient accumulation steps")

    # V2 optimization flags
    parser.add_argument("--focal_loss", action="store_true", help="Use Focal Loss instead of BCE")
    parser.add_argument("--mixup",      action="store_true", help="Enable Mixup augmentation")
    parser.add_argument("--ema",        action="store_true", help="Enable EMA weight averaging")
    parser.add_argument("--tta",        action="store_true", help="Enable TTA at validation")

    # Per-epoch checkpoint flags
    parser.add_argument(
        "--save_every_epoch",
        action="store_true",
        help="Save a full checkpoint after every epoch (to output_dir/checkpoints/)",
    )
    parser.add_argument(
        "--max_checkpoints",
        type=int,
        default=0,
        help=(
            "Maximum number of per-epoch checkpoints to keep on disk. "
            "0 = keep all (default). Oldest are deleted first (FIFO). "
            "The best-model file is never deleted."
        ),
    )

    args = parser.parse_args()
    train(args)