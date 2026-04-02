"""
Testing code for BT-SurgSAM.
Evaluates segmentation performance on nephrectomy and ophthalmology datasets.
Metrics: Dice score, IoU (Jaccard), Pixel accuracy.
Supports scarce-label test setting (using all available labels for evaluation).
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
from collections import defaultdict

# Import our modules
from model import BTSurgSAM  # from previous code
from data_loader import SurgicalVideoDataset, create_dataloaders

# ================================
# 1. Evaluation Metrics
# ================================

def compute_dice(pred_mask, gt_mask, num_classes, ignore_index=-1):
    """
    Compute Dice score per class.
    Args:
        pred_mask: (H, W) or (B, H, W) predicted class indices
        gt_mask: (H, W) or (B, H, W) ground truth indices
        num_classes: int
        ignore_index: label value to ignore (e.g., -1 for unlabeled)
    Returns:
        dice_per_class: (num_classes,) array
    """
    if pred_mask.dim() == 2:
        pred_mask = pred_mask.unsqueeze(0)
        gt_mask = gt_mask.unsqueeze(0)
    B, H, W = pred_mask.shape
    dice_per_class = np.zeros(num_classes)
    valid = (gt_mask != ignore_index)
    for c in range(num_classes):
        pred_c = (pred_mask == c) & valid
        gt_c = (gt_mask == c) & valid
        intersection = (pred_c & gt_c).sum().item()
        union_pred = pred_c.sum().item()
        union_gt = gt_c.sum().item()
        if union_pred + union_gt == 0:
            dice = 1.0 if intersection == 0 else 0.0  # no GT and no pred -> perfect
        else:
            dice = 2 * intersection / (union_pred + union_gt + 1e-8)
        dice_per_class[c] = dice
    return dice_per_class


def compute_iou(pred_mask, gt_mask, num_classes, ignore_index=-1):
    """Compute IoU (Jaccard) per class."""
    if pred_mask.dim() == 2:
        pred_mask = pred_mask.unsqueeze(0)
        gt_mask = gt_mask.unsqueeze(0)
    B, H, W = pred_mask.shape
    iou_per_class = np.zeros(num_classes)
    valid = (gt_mask != ignore_index)
    for c in range(num_classes):
        pred_c = (pred_mask == c) & valid
        gt_c = (gt_mask == c) & valid
        intersection = (pred_c & gt_c).sum().item()
        union = (pred_c | gt_c).sum().item()
        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / (union + 1e-8)
        iou_per_class[c] = iou
    return iou_per_class


def compute_pixel_accuracy(pred_mask, gt_mask, ignore_index=-1):
    """Compute pixel accuracy (overall)."""
    if pred_mask.dim() == 2:
        pred_mask = pred_mask.unsqueeze(0)
        gt_mask = gt_mask.unsqueeze(0)
    valid = (gt_mask != ignore_index)
    correct = (pred_mask == gt_mask) & valid
    acc = correct.sum().item() / (valid.sum().item() + 1e-8)
    return acc


def compute_all_metrics(pred_logits, gt_mask, num_classes, ignore_index=-1):
    """
    Compute all metrics from logits.
    Args:
        pred_logits: (B, K, H, W) logits
        gt_mask: (B, H, W) long tensor
    Returns:
        dice: dict per class + mean
        iou: dict
        acc: float
    """
    pred_mask = torch.argmax(pred_logits, dim=1).cpu().numpy()
    gt_mask = gt_mask.cpu().numpy()
    dice_per_class = compute_dice(pred_mask, gt_mask, num_classes, ignore_index)
    iou_per_class = compute_iou(pred_mask, gt_mask, num_classes, ignore_index)
    acc = compute_pixel_accuracy(pred_mask, gt_mask, ignore_index)
    # Exclude ignore_index from mean (already handled by compute functions)
    valid_classes = [c for c in range(num_classes) if np.any(gt_mask == c)]
    if len(valid_classes) > 0:
        mean_dice = dice_per_class[valid_classes].mean()
        mean_iou = iou_per_class[valid_classes].mean()
    else:
        mean_dice = 0.0
        mean_iou = 0.0
    return {
        'dice_per_class': dice_per_class,
        'iou_per_class': iou_per_class,
        'mean_dice': mean_dice,
        'mean_iou': mean_iou,
        'pixel_accuracy': acc
    }

# ================================
# 2. Testing Function
# ================================

@torch.no_grad()
def test_model(model, test_loader, device, num_classes, save_predictions_dir=None):
    """
    Run evaluation on test set.
    Args:
        model: BTSurgSAM instance
        test_loader: DataLoader returning batches with 'frames' and 'masks'
        device: cuda/cpu
        num_classes: int
        save_predictions_dir: optional path to save visualization
    Returns:
        results: dict with aggregated metrics
    """
    model.eval()
    
    # Accumulate metrics
    all_metrics = defaultdict(list)
    
    # For per-class aggregation over all samples
    all_dice = []
    all_iou = []
    
    pbar = tqdm(test_loader, desc="Testing")
    for batch_idx, batch in enumerate(pbar):
        frames = batch['frames'].to(device)        # (B, T, 3, H, W)
        masks = batch['masks'].to(device)          # (B, T, H, W) with -1 for unlabeled
        
        B, T, C, H, W = frames.shape
        # We only evaluate on the last frame of each clip (or all frames that have labels)
        # For simplicity, evaluate on the last frame (most common in literature)
        # But if the last frame has no label (=-1), we skip.
        last_masks = masks[:, -1]   # (B, H, W)
        # Only keep samples where last frame has label
        valid_samples = (last_masks != -1).any(dim=(1,2))  # at least one pixel labeled
        if not valid_samples.any():
            continue
        
        # Forward pass
        # Note: model forward returns seg_logits, total_loss, ... but we only need seg_logits
        seg_logits, _, _, _, _ = model(frames, labels=None, class_masks=None, update_dra=False)
        # seg_logits: (B, K, H, W)
        
        # Filter valid samples
        seg_logits = seg_logits[valid_samples]
        valid_masks = last_masks[valid_samples]
        
        # Compute metrics
        metrics = compute_all_metrics(seg_logits, valid_masks, num_classes, ignore_index=-1)
        
        all_metrics['mean_dice'].append(metrics['mean_dice'])
        all_metrics['mean_iou'].append(metrics['mean_iou'])
        all_metrics['pixel_accuracy'].append(metrics['pixel_accuracy'])
        all_dice.append(metrics['dice_per_class'])
        all_iou.append(metrics['iou_per_class'])
        
        # Optional: save predictions
        if save_predictions_dir is not None and batch_idx < 10:  # save first few batches
            os.makedirs(save_predictions_dir, exist_ok=True)
            pred_masks = torch.argmax(seg_logits, dim=1).cpu().numpy()  # (B, H, W)
            for i in range(len(pred_masks)):
                pred_mask = pred_masks[i].astype(np.uint8)
                gt_mask = valid_masks[i].cpu().numpy().astype(np.uint8)
                # Create color overlay (simple)
                h, w = pred_mask.shape
                vis = np.zeros((h, w, 3), dtype=np.uint8)
                # Color palette (random but fixed per class)
                colors = np.random.randint(0, 255, (num_classes, 3))
                for c in range(num_classes):
                    vis[pred_mask == c] = colors[c]
                # Save
                cv2.imwrite(os.path.join(save_predictions_dir, f'pred_{batch_idx}_{i}.png'), 
                            cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                # Also save GT
                vis_gt = np.zeros((h, w, 3), dtype=np.uint8)
                for c in range(num_classes):
                    vis_gt[gt_mask == c] = colors[c]
                cv2.imwrite(os.path.join(save_predictions_dir, f'gt_{batch_idx}_{i}.png'),
                            cv2.cvtColor(vis_gt, cv2.COLOR_RGB2BGR))
    
    # Aggregate results
    mean_dice = np.mean(all_metrics['mean_dice'])
    mean_iou = np.mean(all_metrics['mean_iou'])
    mean_acc = np.mean(all_metrics['pixel_accuracy'])
    
    # Per-class dice across all samples
    if len(all_dice) > 0:
        all_dice = np.stack(all_dice, axis=0)  # (N_samples, num_classes)
        per_class_dice = np.mean(all_dice, axis=0)
        per_class_iou = np.mean(np.stack(all_iou, axis=0), axis=0)
    else:
        per_class_dice = np.zeros(num_classes)
        per_class_iou = np.zeros(num_classes)
    
    results = {
        'mean_dice': mean_dice,
        'mean_iou': mean_iou,
        'pixel_accuracy': mean_acc,
        'per_class_dice': per_class_dice,
        'per_class_iou': per_class_iou
    }
    return results

# ================================
# 3. Main Testing Script
# ================================

def main():
    parser = argparse.ArgumentParser(description='Test BT-SurgSAM on surgical video datasets')
    parser.add_argument('--data_root', type=str, required=True, help='Path to dataset root')
    parser.add_argument('--dataset', type=str, default='nephrectomy', choices=['nephrectomy', 'ophthalmology'])
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--sam_checkpoint', type=str, default='sam_vit_b_01ec64.pth', help='SAM pretrained weights')
    parser.add_argument('--sam_model_type', type=str, default='vit_b', choices=['vit_b', 'vit_l', 'vit_h'])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--clip_length', type=int, default=8)
    parser.add_argument('--frame_step', type=int, default=1)
    parser.add_argument('--target_size', type=int, nargs=2, default=[256, 256], help='Height Width')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_preds', type=str, default=None, help='Directory to save prediction visualizations')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # Create test dataset and loader
    test_dataset = SurgicalVideoDataset(
        root_dir=args.data_root,
        dataset_name=args.dataset,
        split='test',
        clip_length=args.clip_length,
        frame_step=args.frame_step,
        label_density=1.0,  # test uses all labels
        transform=None,      # no augmentation
        target_size=tuple(args.target_size)
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=lambda batch: {
            'frames': torch.stack([b['frames'] for b in batch]),
            'masks': torch.stack([b['masks'] for b in batch]),
            'video_idx': torch.tensor([b['video_idx'] for b in batch])
        }, pin_memory=True
    )
    
    num_classes = test_dataset.num_classes
    print(f"Testing on {args.dataset} dataset with {num_classes} classes")
    print(f"Test samples: {len(test_dataset)}")
    
    # Initialize model
    model = BTSurgSAM(
        sam_model_type=args.sam_model_type,
        checkpoint_path=args.sam_checkpoint,
        num_classes=num_classes,
        num_frames=args.clip_length,
        device=device
    )
    # Load checkpoint (we saved full model or state_dict)
    if os.path.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {args.checkpoint}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    model.to(device)
    
    # Run test
    results = test_model(model, test_loader, device, num_classes, save_predictions_dir=args.save_preds)
    
    # Print results
    print("\n" + "="*50)
    print(f"Test Results on {args.dataset}:")
    print(f"Mean Dice: {results['mean_dice']:.4f}")
    print(f"Mean IoU:  {results['mean_iou']:.4f}")
    print(f"Pixel Acc: {results['pixel_accuracy']:.4f}")
    print("\nPer-class Dice:")
    for c in range(num_classes):
        print(f"  Class {c}: {results['per_class_dice'][c]:.4f}")
    print("="*50)
    
    # Optionally save results to file
    output_file = os.path.join(os.path.dirname(args.checkpoint), 'test_results.txt')
    with open(output_file, 'w') as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Mean Dice: {results['mean_dice']:.4f}\n")
        f.write(f"Mean IoU: {results['mean_iou']:.4f}\n")
        f.write(f"Pixel Accuracy: {results['pixel_accuracy']:.4f}\n")
        f.write("Per-class Dice:\n")
        for c, d in enumerate(results['per_class_dice']):
            f.write(f"  Class {c}: {d:.4f}\n")
    print(f"Results saved to {output_file}")

# ================================
# 4. Additional: Test with Different Label Densities
# ================================

def test_with_label_density(model, data_root, dataset_name, checkpoint, label_densities=[0.01, 0.05, 0.1, 0.5]):
    """
    Evaluate model under different scarce-label training settings.
    This assumes the model has been trained with a specific label density.
    We just evaluate on full test set (all labels).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dict = {}
    for ld in label_densities:
        # Create test dataset (with full labels for evaluation)
        test_dataset = SurgicalVideoDataset(
            root_dir=data_root,
            dataset_name=dataset_name,
            split='test',
            clip_length=8,
            frame_step=1,
            label_density=1.0,  # full labels for evaluation
            transform=None,
            target_size=(256,256)
        )
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4,
                                 collate_fn=lambda batch: {
                                     'frames': torch.stack([b['frames'] for b in batch]),
                                     'masks': torch.stack([b['masks'] for b in batch])
                                 })
        # Load model trained with label_density ld
        # (Assume checkpoint naming includes ld)
        ckpt_path = checkpoint.format(ld)
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint for label density {ld} not found, skipping")
            continue
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device)
        results = test_model(model, test_loader, device, test_dataset.num_classes)
        results_dict[ld] = results
        print(f"Label density {ld}: Dice = {results['mean_dice']:.4f}")
    return results_dict

if __name__ == "__main__":
    main()