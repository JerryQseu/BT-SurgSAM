"""
Data loading module for BT-SurgSAM.
Supports:
- Nephrectomy dataset (public benchmark)
- Ophthalmology dataset (newly collected)
- Scarce-label setting: only a fraction of frames have segmentation masks.
"""

import os
import glob
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from typing import Tuple, List, Dict, Optional, Union
import json

# ================================
# 1. Data Augmentation Utilities
# ================================

class SurgicalAugmentation:
    """Online augmentation for surgical video frames."""
    def __init__(self, 
                 crop_size=(256, 256),
                 horizontal_flip_prob=0.5,
                 vertical_flip_prob=0.0,
                 rotate_degree=10,
                 brightness_jitter=0.1,
                 contrast_jitter=0.1,
                 saturation_jitter=0.1):
        self.crop_size = crop_size
        self.horizontal_flip_prob = horizontal_flip_prob
        self.vertical_flip_prob = vertical_flip_prob
        self.rotate_degree = rotate_degree
        self.brightness_jitter = brightness_jitter
        self.contrast_jitter = contrast_jitter
        self.saturation_jitter = saturation_jitter
        
    def __call__(self, frames: np.ndarray, masks: Optional[np.ndarray] = None):
        """
        Args:
            frames: (T, H, W, 3) or (H, W, 3) numpy array uint8
            masks: (T, H, W) or (H, W) numpy int64 (class indices)
        Returns:
            frames_tensor: (T, 3, H, W) or (3, H, W) float32 normalized to [0,1]
            masks_tensor: (T, H, W) or (H, W) long
        """
        is_single = (frames.ndim == 3)
        if is_single:
            frames = np.expand_dims(frames, axis=0)
            if masks is not None:
                masks = np.expand_dims(masks, axis=0)
        
        T, H, W, C = frames.shape
        
        # Random crop
        if self.crop_size is not None and (H > self.crop_size[0] and W > self.crop_size[1]):
            top = random.randint(0, H - self.crop_size[0])
            left = random.randint(0, W - self.crop_size[1])
            frames = frames[:, top:top+self.crop_size[0], left:left+self.crop_size[1], :]
            if masks is not None:
                masks = masks[:, top:top+self.crop_size[0], left:left+self.crop_size[1]]
        
        # Random flip
        if random.random() < self.horizontal_flip_prob:
            frames = np.flip(frames, axis=2).copy()
            if masks is not None:
                masks = np.flip(masks, axis=2).copy()
        if random.random() < self.vertical_flip_prob:
            frames = np.flip(frames, axis=1).copy()
            if masks is not None:
                masks = np.flip(masks, axis=1).copy()
        
        # Random rotation
        if self.rotate_degree > 0:
            angle = random.uniform(-self.rotate_degree, self.rotate_degree)
            # Rotate each frame (simple rotation using OpenCV)
            for t in range(T):
                M = cv2.getRotationMatrix2D((frames.shape[2]/2, frames.shape[1]/2), angle, 1)
                frames[t] = cv2.warpAffine(frames[t], M, (frames.shape[2], frames.shape[1]))
                if masks is not None:
                    masks[t] = cv2.warpAffine(masks[t].astype(np.float32), M, 
                                             (masks.shape[2], masks.shape[1]), 
                                             flags=cv2.INTER_NEAREST).astype(np.int64)
        
        # Color jitter (apply same to all frames in clip)
        if self.brightness_jitter > 0 or self.contrast_jitter > 0 or self.saturation_jitter > 0:
            # Convert to PIL tensor for color jitter (or use numpy)
            frames_tensor = torch.from_numpy(frames).float() / 255.0  # (T, H, W, C)
            frames_tensor = frames_tensor.permute(0, 3, 1, 2)  # (T, C, H, W)
            # Apply same color jitter parameters across all frames
            brightness = 1 + random.uniform(-self.brightness_jitter, self.brightness_jitter)
            contrast = 1 + random.uniform(-self.contrast_jitter, self.contrast_jitter)
            saturation = 1 + random.uniform(-self.saturation_jitter, self.saturation_jitter)
            frames_tensor = frames_tensor * brightness
            mean = frames_tensor.mean(dim=[2,3], keepdim=True)
            frames_tensor = (frames_tensor - mean) * contrast + mean
            # Saturation: convert to HSV? Simpler: use torchvision's adjust_saturation
            # We'll just do a simple scaling in RGB space (not accurate but fine)
            frames_tensor = frames_tensor * saturation
            frames_tensor = torch.clamp(frames_tensor, 0, 1)
            frames = (frames_tensor.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
        
        # Convert to tensor and normalize
        frames_tensor = torch.from_numpy(frames).float() / 255.0  # (T, H, W, C)
        frames_tensor = frames_tensor.permute(0, 3, 1, 2)  # (T, C, H, W)
        
        if masks is not None:
            masks_tensor = torch.from_numpy(masks).long()  # (T, H, W)
        else:
            masks_tensor = None
        
        if is_single:
            frames_tensor = frames_tensor[0]
            if masks_tensor is not None:
                masks_tensor = masks_tensor[0]
        
        return frames_tensor, masks_tensor

# ================================
# 2. Base Surgical Video Dataset
# ================================

class SurgicalVideoDataset(Dataset):
    """
    Generic dataset for surgical video segmentation.
    Supports two dataset types: 'nephrectomy' and 'ophthalmology'.
    Scarce-label setting: only a subset of frames have ground truth masks.
    """
    def __init__(self,
                 root_dir: str,
                 dataset_name: str = 'nephrectomy',
                 split: str = 'train',
                 clip_length: int = 8,
                 frame_step: int = 1,
                 label_density: float = 0.1,   # fraction of frames with labels (scarce-label)
                 transform: Optional[SurgicalAugmentation] = None,
                 target_size: Tuple[int, int] = (256, 256),
                 use_optical_flow: bool = False):
        """
        Args:
            root_dir: Root directory containing videos and masks.
            dataset_name: 'nephrectomy' or 'ophthalmology'.
            split: 'train', 'val', 'test'.
            clip_length: Number of frames per clip (T).
            frame_step: Step between consecutive frames in the clip.
            label_density: Proportion of frames that have annotations (0..1). 
                           For scarce-label experiments, use small values (e.g., 0.05, 0.1).
            transform: Augmentation transform.
            target_size: Resize frames and masks to this size.
            use_optical_flow: Whether to compute optical flow between frames (not used in base version).
        """
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.split = split
        self.clip_length = clip_length
        self.frame_step = frame_step
        self.label_density = label_density
        self.transform = transform
        self.target_size = target_size
        self.use_optical_flow = use_optical_flow
        
        # Build file list
        self.video_list = self._get_video_list()
        # For each video, store available frames and which frames have labels
        self.video_info = []  # list of dict: {'video_path': ..., 'frames': [...], 'labeled_frames': set(...)}
        for video_path in self.video_list:
            frame_paths, labeled_frames = self._load_frame_info(video_path)
            if len(frame_paths) >= clip_length:
                self.video_info.append({
                    'video_path': video_path,
                    'frame_paths': frame_paths,
                    'labeled_frames': labeled_frames,
                    'num_frames': len(frame_paths)
                })
        
        # Class mapping (depends on dataset)
        self.num_classes, self.class_names = self._get_class_info()
        
    def _get_video_list(self) -> List[str]:
        """Return list of video identifiers or paths."""
        if self.dataset_name == 'nephrectomy':
            # Assumed structure: root_dir/videos/*.mp4 or root_dir/frames/video_001/frame_*.png
            # We'll support two modes: video files or frame folders.
            video_dir = os.path.join(self.root_dir, self.split, 'videos')
            if os.path.exists(video_dir):
                videos = glob.glob(os.path.join(video_dir, '*.mp4')) + \
                         glob.glob(os.path.join(video_dir, '*.avi'))
                return videos
            else:
                # Frame-based: each video is a folder with frames
                frame_dir = os.path.join(self.root_dir, self.split, 'frames')
                videos = [d for d in os.listdir(frame_dir) if os.path.isdir(os.path.join(frame_dir, d))]
                return [os.path.join(frame_dir, v) for v in videos]
        elif self.dataset_name == 'ophthalmology':
            # New ophthalmology dataset: similar structure
            video_dir = os.path.join(self.root_dir, self.split, 'videos')
            if os.path.exists(video_dir):
                videos = glob.glob(os.path.join(video_dir, '*.mp4')) + \
                         glob.glob(os.path.join(video_dir, '*.avi'))
                return videos
            else:
                frame_dir = os.path.join(self.root_dir, self.split, 'frames')
                videos = [d for d in os.listdir(frame_dir) if os.path.isdir(os.path.join(frame_dir, d))]
                return [os.path.join(frame_dir, v) for v in videos]
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def _load_frame_info(self, video_path: str) -> Tuple[List[str], set]:
        """
        Load list of frame file paths and determine which frames have labels.
        Returns:
            frame_paths: sorted list of paths to image files (or frame indices)
            labeled_frames: set of indices (0-based) that have segmentation masks.
        """
        if os.path.isdir(video_path):
            # Frame-based video
            frame_files = sorted(glob.glob(os.path.join(video_path, '*.png')) + 
                                 glob.glob(os.path.join(video_path, '*.jpg')))
            # Check for corresponding mask files
            # Masks are assumed to be in a parallel directory: masks/video_name/frame_*.png
            mask_dir = os.path.join(self.root_dir, self.split, 'masks', os.path.basename(video_path))
            labeled_frames = set()
            for idx, fpath in enumerate(frame_files):
                fname = os.path.basename(fpath)
                mask_path = os.path.join(mask_dir, fname)
                if os.path.exists(mask_path):
                    labeled_frames.add(idx)
            # Scarce-label: we may further subsample labeled frames if label_density < 1
            # But we keep all original labeled frames; the training loop will ignore most of them
            # based on label_density sampling.
            return frame_files, labeled_frames
        else:
            # Video file: we need to extract frames on the fly or pre-extract.
            # For simplicity, we assume frames have been pre-extracted to a folder.
            # If not, we raise an error.
            raise NotImplementedError("Video file reading not implemented. Please pre-extract frames to folders.")
    
    def _get_class_info(self):
        """Return number of classes and class names for each dataset."""
        if self.dataset_name == 'nephrectomy':
            # Example classes: background, kidney, tumor, instrument, etc.
            # Adjust based on actual dataset.
            num_classes = 6  # placeholder
            class_names = ['background', 'kidney', 'tumor', 'instrument', 'suture', 'vessel']
        elif self.dataset_name == 'ophthalmology':
            # Cataract surgery: iris, pupil, lens, cornea, instrument, etc.
            num_classes = 8
            class_names = ['background', 'iris', 'pupil', 'lens', 'cornea', 'instrument', 'capsule', 'vitreous']
        else:
            num_classes = 10  # default
            class_names = [f'class_{i}' for i in range(num_classes)]
        return num_classes, class_names
    
    def _get_frame_and_mask(self, video_info: dict, frame_idx: int):
        """
        Load frame image and its mask (if available) at given index.
        Returns:
            frame: (H, W, 3) numpy uint8
            mask: (H, W) numpy int64 or None if no label.
        """
        frame_path = video_info['frame_paths'][frame_idx]
        # Load image
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize to target size
        if self.target_size is not None:
            frame = cv2.resize(frame, (self.target_size[1], self.target_size[0]))
        
        # Load mask if frame is labeled
        mask = None
        if frame_idx in video_info['labeled_frames']:
            # Determine mask path
            video_name = os.path.basename(video_info['video_path'])
            mask_filename = os.path.basename(frame_path)
            mask_dir = os.path.join(self.root_dir, self.split, 'masks', video_name)
            mask_path = os.path.join(mask_dir, mask_filename)
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if self.target_size is not None:
                    mask = cv2.resize(mask, (self.target_size[1], self.target_size[0]), 
                                     interpolation=cv2.INTER_NEAREST)
                mask = mask.astype(np.int64)
        return frame, mask
    
    def __len__(self):
        return len(self.video_info)
    
    def __getitem__(self, idx):
        video_info = self.video_info[idx]
        total_frames = video_info['num_frames']
        # Randomly select starting frame for the clip
        max_start = total_frames - self.clip_length * self.frame_step
        if max_start < 0:
            # Not enough frames, pad? For simplicity, skip or use smaller clip.
            # Here we just take the first frame repeatedly (should not happen)
            start = 0
        else:
            start = random.randint(0, max_start)
        
        # Collect frames and masks for the clip
        frames = []
        masks = []  # list of masks (some may be None)
        has_label = False
        for t in range(self.clip_length):
            frame_idx = start + t * self.frame_step
            frame, mask = self._get_frame_and_mask(video_info, frame_idx)
            frames.append(frame)
            masks.append(mask)
            if mask is not None:
                has_label = True
        
        # Apply same augmentation to all frames (and masks)
        # Convert to numpy arrays: (T, H, W, C) and (T, H, W)
        frames_np = np.stack(frames, axis=0)
        masks_np = np.stack(masks, axis=0) if masks[0] is not None else None
        
        if self.transform is not None:
            frames_tensor, masks_tensor = self.transform(frames_np, masks_np)
        else:
            # Default: to tensor and normalize
            frames_tensor = torch.from_numpy(frames_np).float() / 255.0
            frames_tensor = frames_tensor.permute(0, 3, 1, 2)
            masks_tensor = torch.from_numpy(masks_np).long() if masks_np is not None else None
        
        # For scarce-label setting, we may need to indicate which frames have labels.
        # We'll create a boolean mask indicating labeled positions (only for the last frame? 
        # The paper uses labels only on the last frame of the clip for supervised loss.
        # But BURE uses previous frames for prior, so we need labels for previous frames? 
        # In scarce-label, previous frames may not have labels. We can still use prior loss without labels.
        # For simplicity, we will only use the label of the last frame if it exists.
        # The DRA uses class masks derived from predictions or sparse labels.
        # We'll return the full clip and also a 'has_label' flag per clip.
        
        # For the last frame only
        last_frame_label = masks_tensor[-1] if masks_tensor is not None else None
        # Also return the full masks tensor (may contain None entries, but we can't have None in batch)
        # To handle sparse labels, we return a separate tensor with -1 for unlabeled frames.
        if masks_tensor is not None:
            # Create a label tensor where unlabeled frames have -1
            sparse_labels = masks_tensor.clone()
            # For frames without label, set to -1 (ignored in loss)
            # In our current implementation, if a frame didn't have a mask, masks_tensor is None for that frame.
            # But we stacked only labeled frames? Actually we stacked all, but those without label became None.
            # We need to modify the stacking: for unlabeled frames, we put a placeholder -1.
            # Let's redo the stacking:
            mask_array = -np.ones((self.clip_length, self.target_size[0], self.target_size[1]), dtype=np.int64)
            for t, m in enumerate(masks):
                if m is not None:
                    mask_array[t] = m
            masks_tensor = torch.from_numpy(mask_array).long()
        else:
            masks_tensor = torch.full((self.clip_length, self.target_size[0], self.target_size[1]), 
                                     -1, dtype=torch.long)
        
        return {
            'frames': frames_tensor,          # (T, 3, H, W)
            'masks': masks_tensor,            # (T, H, W) with -1 for unlabeled frames
            'video_idx': idx,
            'start_frame': start
        }

# ================================
# 3. Scarce-Label Sampler (optional)
# ================================

class ScarceLabelSampler:
    """
    Sampler that controls the density of labeled clips.
    Since our dataset already has sparse labels per frame, we can also sample clips 
    that have at least one labeled frame. Or we can use a strategy to only use 
    a fraction of labeled frames during training (the label_density parameter).
    """
    def __init__(self, dataset, labeled_ratio=0.1):
        self.dataset = dataset
        self.labeled_ratio = labeled_ratio
        # Precompute which clips have labels (at least one labeled frame in the clip)
        # This can be done by scanning the dataset; we'll do it lazily.
        self._labeled_indices = []
        self._unlabeled_indices = []
        for i in range(len(dataset)):
            # Check if dataset has any labeled frame in the clip? Hard to know without loading.
            # Instead, we'll use a simpler approach: during training, we randomly sample with replacement
            # and apply a Bernoulli to decide whether to return the label (if available).
            pass
    
    def __iter__(self):
        # Not fully implemented; we rely on the dataset's sparse mask with -1.
        pass

# ================================
# 4. Create DataLoaders
# ================================

def create_dataloaders(config: dict):
    """
    Create train/val/test dataloaders for BT-SurgSAM.
    Config should contain:
        data_root: str
        dataset_name: str ('nephrectomy' or 'ophthalmology')
        batch_size: int
        clip_length: int (T)
        frame_step: int
        label_density: float (e.g., 0.05, 0.1)
        num_workers: int
        target_size: tuple (H, W)
    """
    # Training transforms
    train_transform = SurgicalAugmentation(
        crop_size=config.get('crop_size', config['target_size']),
        horizontal_flip_prob=0.5,
        rotate_degree=5,
        brightness_jitter=0.1,
        contrast_jitter=0.1
    )
    val_transform = None  # only resize and to tensor
    
    train_dataset = SurgicalVideoDataset(
        root_dir=config['data_root'],
        dataset_name=config['dataset_name'],
        split='train',
        clip_length=config['clip_length'],
        frame_step=config.get('frame_step', 1),
        label_density=config.get('label_density', 0.1),
        transform=train_transform,
        target_size=config['target_size']
    )
    
    val_dataset = SurgicalVideoDataset(
        root_dir=config['data_root'],
        dataset_name=config['dataset_name'],
        split='val',
        clip_length=config['clip_length'],
        frame_step=config.get('frame_step', 1),
        label_density=1.0,  # validation uses all available labels
        transform=val_transform,
        target_size=config['target_size']
    )
    
    test_dataset = SurgicalVideoDataset(
        root_dir=config['data_root'],
        dataset_name=config['dataset_name'],
        split='test',
        clip_length=config['clip_length'],
        frame_step=config.get('frame_step', 1),
        label_density=1.0,
        transform=val_transform,
        target_size=config['target_size']
    )
    
    # Collate function to handle variable label presence
    def collate_fn(batch):
        frames = torch.stack([item['frames'] for item in batch])
        masks = torch.stack([item['masks'] for item in batch])
        video_idx = torch.tensor([item['video_idx'] for item in batch])
        return {
            'frames': frames,      # (B, T, 3, H, W)
            'masks': masks,        # (B, T, H, W) with -1 for unlabeled
            'video_idx': video_idx
        }
    
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True,
        num_workers=config['num_workers'], collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers'], collate_fn=collate_fn, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers'], collate_fn=collate_fn, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_dataset.num_classes

# ================================
# 5. Example Usage
# ================================

if __name__ == "__main__":
    # Configuration for scarce-label experiment
    config = {
        'data_root': '/path/to/surgical_datasets',
        'dataset_name': 'nephrectomy',   # or 'ophthalmology'
        'batch_size': 4,
        'clip_length': 8,
        'frame_step': 2,
        'label_density': 0.1,            # only 10% of frames have labels
        'num_workers': 4,
        'target_size': (256, 256),
        'crop_size': (224, 224)
    }
    
    train_loader, val_loader, test_loader, num_classes = create_dataloaders(config)
    print(f"Number of classes: {num_classes}")
    print(f"Train batches: {len(train_loader)}")
    
    # Test a batch
    for batch in train_loader:
        frames = batch['frames']      # (B, T, 3, H, W)
        masks = batch['masks']        # (B, T, H, W) with -1 for unlabeled
        print(f"Frames shape: {frames.shape}")
        print(f"Masks shape: {masks.shape}")
        # Count labeled frames in batch
        labeled_mask = (masks != -1)
        num_labeled = labeled_mask.sum().item()
        print(f"Labeled frames in batch: {num_labeled} / {masks.numel()}")
        break