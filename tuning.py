"""
Fine-tuning code for BT-SurgSAM.
Trains the model on scarce-label surgical video segmentation.
Supports:
- Sparse labels (masks with -1 for unlabeled frames)
- Uncertainty loss (BURE) + DRA alignment loss + Dice loss
- Validation with full metrics
- Model checkpointing
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm
import wandb  # optional
from typing import Dict, Tuple

# Import our modules (assuming they are in the same directory or accessible)
from model import BTSurgSAM
from data_loader import SurgicalVideoDataset, create_dataloaders

# ================================
# 1. Loss Functions (with sparse label handling)
# ================================

def dice_loss_with_ignore(pred_logits, target, num_classes, ignore_index=-1):
    """
    Dice loss that ignores pixels with ignore_index.
    pred_logits: (B, K, H, W)
    target: (B, H, W) long
    """
    B, K, H, W = pred_logits.shape
    pred_softmax = F.softmax(pred_logits, dim=1)  # (B, K, H, W)
    target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()  # (B, K, H, W)
    # Mask out ignore_index
    valid_mask = (target != ignore_index).float().unsqueeze(1)  # (B, 1, H, W)
    pred_softmax = pred_softmax * valid_mask
    target_one_hot = target_one_hot * valid_mask
    
    intersection = (pred_softmax * target_one_hot).sum(dim=(2,3))
    union = pred_softmax.sum(dim=(2,3)) + target_one_hot.sum(dim=(2,3))
    dice = (2 * intersection + 1e-6) / (union + 1e-6)
    # Mean over classes, then mean over batch
    dice_loss = 1 - dice.mean()
    return dice_loss


def cross_entropy_with_ignore(pred_logits, target, ignore_index=-1):
    """Cross-entropy loss ignoring specified index."""
    return F.cross_entropy(pred_logits, target, ignore_index=ignore_index)


def compute_supervised_loss(pred_logits, target, num_classes, ignore_index=-1):
    """
    Combined supervised loss: cross-entropy + dice.
    Used for frames that have ground truth labels.
    """
    ce = cross_entropy_with_ignore(pred_logits, target, ignore_index)
    dice = dice_loss_with_ignore(pred_logits, target, num_classes, ignore_index)
    return ce + dice

# ================================
# 2. Training Function
# ================================

def train_one_epoch(model, train_loader, optimizer, device, num_classes, 
                    epoch, use_dra=True, log_interval=10):
    """
    Train for one epoch.
    Returns: average loss and metrics.
    """
    model.train()
    total_loss = 0
    total_dice_loss = 0
    total_bure_loss = 0
    total_dra_loss = 0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch_idx, batch in enumerate(pbar):
        frames = batch['frames'].to(device)      # (B, T, 3, H, W)
        masks = batch['masks'].to(device)        # (B, T, H, W) with -1 for unlabeled
        
        B, T, H, W = masks.shape
        # For each clip, we will only supervise the frames that have labels.
        # The model's forward expects labels for the last frame (as per paper design)
        # but we can also provide labels for all frames? The paper's BURE uses prior from previous frames.
        # However, for scarce labels, only some frames have GT. We'll adapt:
        # - For the last frame, if it has label, we compute supervised dice loss (L'_dice).
        # - For BURE's posterior loss, we need labels for the last frame (as in equation).
        # - For DRA, we need class masks for the frames we want to align. We'll use available labels if present,
        #   otherwise we skip DRA update for that frame (or use pseudo-labels).
        
        # We'll use the last frame's label if available (most common in video segmentation)
        last_masks = masks[:, -1]  # (B, H, W)
        # Only compute supervised loss on samples where last frame has at least one labeled pixel
        # (ignore -1 pixels in the loss calculation)
        
        # Forward pass with BURE and DRA
        # The model returns seg_logits (for last frame), total_loss (combined), and individual losses.
        # We need to provide labels for the last frame (if any) and class_masks for DRA.
        # For class_masks, we can create binary masks per class from the last frame's label (if available).
        # For frames without labels, we set class_masks = None, and DRA will skip.
        if (last_masks != -1).any():
            # Create class masks (B, K, H, W) from last_masks (ignoring -1)
            class_masks = torch.zeros(B, num_classes, H, W, device=device)
            for c in range(num_classes):
                class_masks[:, c] = (last_masks == c).float()
            # Also pass labels for supervised loss (last_masks)
            labels_for_loss = last_masks
        else:
            class_masks = None
            labels_for_loss = None
        
        # Forward
        seg_logits, total_loss_val, loss_u, loss_dra, loss_dice = model(
            frames, 
            labels=labels_for_loss,  # used for posterior loss in BURE and for supervised dice
            class_masks=class_masks,  # used for DRA similarity loss and memory update
            other_memory=None,
            update_dra=use_dra and (class_masks is not None)  # only update if we have labels
        )
        # total_loss_val already includes loss_u + loss_dra + loss_dice
        
        optimizer.zero_grad()
        total_loss_val.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += total_loss_val.item()
        total_dice_loss += loss_dice.item()
        total_bure_loss += loss_u.item()
        total_dra_loss += loss_dra.item()
        num_batches += 1
        
        if batch_idx % log_interval == 0:
            pbar.set_postfix({
                'loss': total_loss_val.item(),
                'dice': loss_dice.item(),
                'bure': loss_u.item(),
                'dra': loss_dra.item()
            })
    
    avg_loss = total_loss / num_batches
    avg_dice = total_dice_loss / num_batches
    avg_bure = total_bure_loss / num_batches
    avg_dra = total_dra_loss / num_batches
    
    return {
        'loss': avg_loss,
        'dice_loss': avg_dice,
        'bure_loss': avg_bure,
        'dra_loss': avg_dra
    }

# ================================
# 3. Validation Function
# ================================

@torch.no_grad()
def validate(model, val_loader, device, num_classes, ignore_index=-1):
    """
    Validation: compute Dice, IoU, accuracy on all labeled frames (full labels).
    """
    model.eval()
    all_dice = []
    all_iou = []
    all_acc = []
    
    for batch in tqdm(val_loader, desc="Validation"):
        frames = batch['frames'].to(device)
        masks = batch['masks'].to(device)  # (B, T, H, W) with -1 for unlabeled
        
        B, T, H, W = masks.shape
        # Evaluate on all frames that have labels (not just last)
        # We'll run the model on each clip, but the model only outputs segmentation for the last frame.
        # To evaluate on all frames, we would need to run the model for each frame as the last frame of a sliding window.
        # For simplicity, we evaluate only on the last frame (common practice).
        # If needed, we could slide the window and accumulate.
        last_masks = masks[:, -1]
        valid = (last_masks != ignore_index)
        if not valid.any():
            continue
        
        # Forward pass (no DRA update, no labels needed for validation)
        seg_logits, _, _, _, _ = model(frames, labels=None, class_masks=None, update_dra=False)
        # seg_logits: (B, K, H, W)
        
        # Only keep valid samples (where last frame has label)
        seg_logits = seg_logits[valid]
        gt = last_masks[valid]
        
        # Compute metrics
        pred = torch.argmax(seg_logits, dim=1)  # (B, H, W)
        pred_np = pred.cpu().numpy()
        gt_np = gt.cpu().numpy()
        
        for i in range(len(pred_np)):
            # Dice per class
            dice_per_class = np.zeros(num_classes)
            iou_per_class = np.zeros(num_classes)
            for c in range(num_classes):
                pred_c = (pred_np[i] == c)
                gt_c = (gt_np[i] == c)
                if pred_c.sum() == 0 and gt_c.sum() == 0:
                    dice = 1.0
                    iou = 1.0
                else:
                    inter = (pred_c & gt_c).sum()
                    dice = 2 * inter / (pred_c.sum() + gt_c.sum() + 1e-8)
                    iou = inter / (pred_c.sum() + gt_c.sum() - inter + 1e-8)
                dice_per_class[c] = dice
                iou_per_class[c] = iou
            all_dice.append(dice_per_class)
            all_iou.append(iou_per_class)
            # Pixel accuracy
            acc = (pred_np[i] == gt_np[i]).mean()
            all_acc.append(acc)
    
    if len(all_dice) == 0:
        return {'mean_dice': 0, 'mean_iou': 0, 'pixel_acc': 0}
    
    all_dice = np.stack(all_dice, axis=0)  # (N, K)
    all_iou = np.stack(all_iou, axis=0)
    mean_dice = all_dice.mean(axis=0).mean()  # overall mean
    mean_iou = all_iou.mean(axis=0).mean()
    mean_acc = np.mean(all_acc)
    
    return {
        'mean_dice': mean_dice,
        'mean_iou': mean_iou,
        'pixel_acc': mean_acc,
        'per_class_dice': all_dice.mean(axis=0),
        'per_class_iou': all_iou.mean(axis=0)
    }

# ================================
# 4. Main Training Script
# ================================

def main():
    parser = argparse.ArgumentParser(description='Fine-tune BT-SurgSAM for scarce-label surgical video segmentation')
    # Data arguments
    parser.add_argument('--data_root', type=str, required=True, help='Dataset root directory')
    parser.add_argument('--dataset', type=str, default='nephrectomy', choices=['nephrectomy', 'ophthalmology'])
    parser.add_argument('--target_size', type=int, nargs=2, default=[256, 256], help='Height Width')
    parser.add_argument('--clip_length', type=int, default=8, help='Number of frames per clip (T)')
    parser.add_argument('--frame_step', type=int, default=1, help='Step between consecutive frames')
    parser.add_argument('--label_density', type=float, default=0.1, help='Fraction of frames with labels (scarce-label)')
    
    # Model arguments
    parser.add_argument('--sam_checkpoint', type=str, required=True, help='SAM pretrained weights path')
    parser.add_argument('--sam_model_type', type=str, default='vit_b', choices=['vit_b', 'vit_l', 'vit_h'])
    parser.add_argument('--num_classes', type=int, default=None, help='Override number of classes (auto-detect if None)')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--lr_scheduler', type=str, default='cosine', choices=['cosine', 'plateau', 'none'])
    parser.add_argument('--use_dra', action='store_true', default=True, help='Enable DRA module')
    parser.add_argument('--no_dra', dest='use_dra', action='store_false')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--eval_interval', type=int, default=5, help='Validate every N epochs')
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='bt-surgsam')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device)
    
    # Create datasets and loaders
    train_transform = SurgicalAugmentation(
        crop_size=args.target_size,
        horizontal_flip_prob=0.5,
        rotate_degree=5,
        brightness_jitter=0.1,
        contrast_jitter=0.1
    )
    val_transform = None  # only resize
    
    train_dataset = SurgicalVideoDataset(
        root_dir=args.data_root,
        dataset_name=args.dataset,
        split='train',
        clip_length=args.clip_length,
        frame_step=args.frame_step,
        label_density=args.label_density,
        transform=train_transform,
        target_size=tuple(args.target_size)
    )
    val_dataset = SurgicalVideoDataset(
        root_dir=args.data_root,
        dataset_name=args.dataset,
        split='val',
        clip_length=args.clip_length,
        frame_step=args.frame_step,
        label_density=1.0,  # validation uses all labels
        transform=val_transform,
        target_size=tuple(args.target_size)
    )
    
    # Collate function (same as in data_loader)
    def collate_fn(batch):
        frames = torch.stack([item['frames'] for item in batch])
        masks = torch.stack([item['masks'] for item in batch])
        video_idx = torch.tensor([item['video_idx'] for item in batch])
        return {'frames': frames, 'masks': masks, 'video_idx': video_idx}
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
    )
    
    num_classes = args.num_classes if args.num_classes is not None else train_dataset.num_classes
    print(f"Number of classes: {num_classes}")
    print(f"Training clips: {len(train_dataset)}, Validation clips: {len(val_dataset)}")
    
    # Initialize model
    model = BTSurgSAM(
        sam_model_type=args.sam_model_type,
        checkpoint_path=args.sam_checkpoint,
        num_classes=num_classes,
        num_frames=args.clip_length,
        device=device
    )
    model.to(device)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer (only trainable params)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    if args.lr_scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.lr_scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    else:
        scheduler = None
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_dice = 0.0
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_dice = checkpoint.get('best_dice', 0.0)
            print(f"Resumed from epoch {start_epoch}, best Dice: {best_dice:.4f}")
        else:
            print(f"Checkpoint {args.resume} not found, starting from scratch.")
    
    # Setup logging
    if args.wandb:
        wandb.init(project=args.wandb_project, config=vars(args))
        wandb.watch(model)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Train one epoch
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, num_classes,
            epoch, use_dra=args.use_dra, log_interval=args.log_interval
        )
        print(f"Train Loss: {train_metrics['loss']:.4f}, Dice Loss: {train_metrics['dice_loss']:.4f}, "
              f"BURE: {train_metrics['bure_loss']:.4f}, DRA: {train_metrics['dra_loss']:.4f}")
        
        # Validation
        if (epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1:
            val_metrics = validate(model, val_loader, device, num_classes)
            print(f"Validation: Mean Dice = {val_metrics['mean_dice']:.4f}, "
                  f"Mean IoU = {val_metrics['mean_iou']:.4f}, Acc = {val_metrics['pixel_acc']:.4f}")
            
            # Save best model
            if val_metrics['mean_dice'] > best_dice:
                best_dice = val_metrics['mean_dice']
                checkpoint_path = os.path.join(args.save_dir, f'best_model_epoch{epoch+1}_dice{best_dice:.4f}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_dice': best_dice,
                    'val_metrics': val_metrics
                }, checkpoint_path)
                print(f"Saved best model to {checkpoint_path}")
            
            # Log to wandb
            if args.wandb:
                wandb.log({
                    'train_loss': train_metrics['loss'],
                    'train_dice_loss': train_metrics['dice_loss'],
                    'train_bure_loss': train_metrics['bure_loss'],
                    'train_dra_loss': train_metrics['dra_loss'],
                    'val_dice': val_metrics['mean_dice'],
                    'val_iou': val_metrics['mean_iou'],
                    'val_acc': val_metrics['pixel_acc'],
                    'epoch': epoch
                })
        
        # Save checkpoint every N epochs (e.g., every 10)
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        
        # Update scheduler
        if args.lr_scheduler == 'cosine':
            scheduler.step()
        elif args.lr_scheduler == 'plateau':
            scheduler.step(val_metrics['mean_dice'])
    
    print(f"\nTraining completed. Best validation Dice: {best_dice:.4f}")
    if args.wandb:
        wandb.finish()

# ================================
# 5. Helper: SurgicalAugmentation (if not already imported)
# ================================

class SurgicalAugmentation:
    """Simple augmentation for surgical video frames (same as in data_loader)."""
    def __init__(self, crop_size=(256,256), horizontal_flip_prob=0.5, rotate_degree=5,
                 brightness_jitter=0.1, contrast_jitter=0.1):
        self.crop_size = crop_size
        self.horizontal_flip_prob = horizontal_flip_prob
        self.rotate_degree = rotate_degree
        self.brightness_jitter = brightness_jitter
        self.contrast_jitter = contrast_jitter
        
    def __call__(self, frames, masks=None):
        # Simple implementation: just resize and normalize, skip complex augmentation
        # For full augmentation, refer to data_loader.py
        if isinstance(frames, np.ndarray):
            if frames.ndim == 3:
                frames = np.expand_dims(frames, 0)
                was_single = True
            else:
                was_single = False
            T, H, W, C = frames.shape
            # Resize
            frames_resized = np.zeros((T, self.crop_size[0], self.crop_size[1], C), dtype=np.uint8)
            for t in range(T):
                frames_resized[t] = cv2.resize(frames[t], (self.crop_size[1], self.crop_size[0]))
            frames = frames_resized
            if masks is not None:
                if masks.ndim == 2:
                    masks = np.expand_dims(masks, 0)
                masks_resized = np.zeros((T, self.crop_size[0], self.crop_size[1]), dtype=np.int64)
                for t in range(T):
                    masks_resized[t] = cv2.resize(masks[t], (self.crop_size[1], self.crop_size[0]), 
                                                  interpolation=cv2.INTER_NEAREST)
                masks = masks_resized
            # To tensor
            frames_tensor = torch.from_numpy(frames).float() / 255.0
            frames_tensor = frames_tensor.permute(0, 3, 1, 2)
            if masks is not None:
                masks_tensor = torch.from_numpy(masks).long()
            else:
                masks_tensor = None
            if was_single:
                frames_tensor = frames_tensor[0]
                if masks_tensor is not None:
                    masks_tensor = masks_tensor[0]
            return frames_tensor, masks_tensor
        else:
            # Already tensor, just return
            return frames, masks

if __name__ == "__main__":
    import cv2  # for resize in augmentation
    main()