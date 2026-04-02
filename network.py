import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
import math

# Try importing SAM (segment_anything)
try:
    from segment_anything import sam_model_registry, SamPredictor
    from segment_anything.modeling import Sam, ImageEncoderViT, PromptEncoder, MaskDecoder
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("Warning: segment_anything not installed. Install with: pip install segment-anything")

# Try importing Mamba
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba_ssm not installed. Using simplified SSM placeholder.")
    # Define a simplified Mamba placeholder (linear + conv)
    class Mamba(nn.Module):
        def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
            super().__init__()
            self.d_model = d_model
            self.inner_dim = d_model * expand
            self.conv1d = nn.Conv1d(d_model, self.inner_dim, kernel_size=d_conv, padding=d_conv-1, groups=self.inner_dim)
            self.linear_in = nn.Linear(d_model, self.inner_dim)
            self.linear_out = nn.Linear(self.inner_dim, d_model)
        def forward(self, x):
            # x: (B, L, D)
            B, L, D = x.shape
            x_in = self.linear_in(x)
            x_conv = self.conv1d(x_in.transpose(1,2)).transpose(1,2)[:,:L,:]
            out = self.linear_out(x_conv)
            return out

class CrossAttention(nn.Module):
    """Cross-attention module for recovering fine-grained details."""
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query, key, value):
        # query, key, value: (B, L, D)
        B, L_q, D = query.shape
        L_k = key.shape[1]
        
        q = self.q_proj(query).reshape(B, L_q, self.num_heads, self.head_dim).transpose(1,2)
        k = self.k_proj(key).reshape(B, L_k, self.num_heads, self.head_dim).transpose(1,2)
        v = self.v_proj(value).reshape(B, L_k, self.num_heads, self.head_dim).transpose(1,2)
        
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1,2).reshape(B, L_q, D)
        out = self.out_proj(out)
        return out

class SpatialTemporalEnhancement(nn.Module):
    def __init__(self, embed_dim, d_state=16, d_conv=4, expand=2, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.mamba = Mamba(d_model=embed_dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.cross_attn = CrossAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, E):
        B, T, D, H, W = E.shape
        # Reshape to (B*H*W, T, D)
        E_flat = E.permute(0,1,3,4,2).reshape(B*H*W, T, D)  # (N, T, D)
        
        # Mamba processes temporal sequence
        mamba_out = self.mamba(E_flat)  # (N, T, D)
        
        # Cross-attention: query = original, key/value = mamba_out to recover details
        attn_out = self.cross_attn(query=E_flat, key=mamba_out, value=mamba_out)
        attn_out = self.norm(attn_out + E_flat)
        
        # Reshape back
        C = attn_out.reshape(B, H, W, T, D).permute(0,3,4,1,2)  # (B, T, D, H, W)
        return C

class EDLPriorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, y_n, e, S):
        K = y_n.shape[-1]
        if S.dim() == 1:
            S = S.unsqueeze(0).expand_as(y_n)
        loss = (y_n * (torch.log(S + 1e-8) - torch.log(e + 1))).sum(dim=-1).mean()
        return loss

class MultiLevelPosteriorLoss(nn.Module):
    def __init__(self, num_levels=3):
        super().__init__()
        self.num_levels = num_levels
        # Learnable weights theta (initialized to 0.5)
        self.log_theta = nn.Parameter(torch.zeros(num_levels))  # log(theta) for numerical stability
        self.theta = torch.sigmoid(self.log_theta) * 0.5 + 0.25  # range [0.25, 0.75]
        
    def forward(self, pred, target, focal_gamma=2.0):
        # Pixel-level: Focal Loss
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** focal_gamma * ce_loss).mean()
        
        # Region-level: Dice Loss
        pred_softmax = F.softmax(pred, dim=1)
        dice_loss = 0
        for c in range(pred.shape[1]):
            pred_c = pred_softmax[:, c].reshape(pred.shape[0], -1)
            target_c = (target == c).float().reshape(target.shape[0], -1)
            intersection = (pred_c * target_c).sum(dim=1)
            union = pred_c.sum(dim=1) + target_c.sum(dim=1)
            dice = (2 * intersection + 1e-6) / (union + 1e-6)
            dice_loss += (1 - dice.mean())
        dice_loss = dice_loss / pred.shape[1]
        
        # Distribution-level: Cross-Entropy
        dist_ce = F.cross_entropy(pred, target)
        
        losses = [focal_loss, dice_loss, dist_ce]
        theta = torch.sigmoid(self.log_theta)  # range (0,1)
        
        total_loss = 0
        for i, L_i in enumerate(losses):
            theta_i = theta[i] + 1e-8
            total_loss += (1 / (2 * theta_i**2)) * L_i + torch.log(1 + theta_i**2)
        return total_loss

class BURE(nn.Module):
    def __init__(self, embed_dim, num_classes, num_heads=8, d_state=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        self.spatial_temporal = SpatialTemporalEnhancement(embed_dim, d_state=d_state, num_heads=num_heads)
        self.edl_prior_loss = EDLPriorLoss()
        self.multi_posterior_loss = MultiLevelPosteriorLoss(num_levels=3)
        
        # Evidence function: g(z) = log(1+exp(z))
        self.evidence_fn = lambda z: torch.log(1 + torch.exp(z))
        
    def compute_prior_loss(self, C_prev, C_curr, labels_prev):
        B, T_prev, D, H, W = C_prev.shape

        prev_feat_avg = C_prev.mean(dim=[1,3,4])  # (B, D)

        if not hasattr(self, 'temp_classifier'):
            self.temp_classifier = nn.Linear(D, self.num_classes).to(C_prev.device)
        prev_logits = self.temp_classifier(prev_feat_avg)  # (B, K)
        y_n = F.softmax(prev_logits, dim=-1)  # probability distribution from previous frames
        

        curr_feat_avg = C_curr.mean(dim=[2,3])  # (B, D)
        curr_evidence = self.evidence_fn(self.temp_classifier(curr_feat_avg))  # (B, K)
        

        alpha_prev = torch.exp(prev_logits) + 1  # approximate
        S = alpha_prev.mean(dim=0)  # (K,)
        
        loss = self.edl_prior_loss(y_n, curr_evidence, S)
        return loss
    
    def forward(self, E, labels=None):

        B, T, D, H, W = E.shape

        C = self.spatial_temporal(E)  # (B, T, D, H, W)
        

        if T > 1:
            C_prev = C[:, :-1]  # (B, T-1, D, H, W)
            C_curr = C[:, -1]   # (B, D, H, W)
            prior_loss = self.compute_prior_loss(C_prev, C_curr, labels)
        else:
            prior_loss = torch.tensor(0.0, device=E.device)

        if labels is not None:

            if not hasattr(self, 'seg_head'):

                self.seg_head = nn.Sequential(
                    nn.Conv2d(D, D//2, kernel_size=3, padding=1),
                    nn.BatchNorm2d(D//2),
                    nn.ReLU(),
                    nn.Conv2d(D//2, self.num_classes, kernel_size=1)
                ).to(E.device)
            pred_logits = self.seg_head(C_curr)  # (B, K, H, W)
            posterior_loss = self.multi_posterior_loss(pred_logits, labels)
        else:
            posterior_loss = torch.tensor(0.0, device=E.device)
        
        loss_u = prior_loss + posterior_loss
        return C, loss_u


class DynamicRepresentationAlignment(nn.Module):
    def __init__(self, embed_dim, num_classes, memory_update_momentum=0.9):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.momentum = memory_update_momentum
        
        # Memory banks for each class: (num_classes, embed_dim)
        self.register_buffer('memory', torch.zeros(num_classes, embed_dim))
        self.register_buffer('memory_counter', torch.zeros(num_classes))
        
        # Projection layers to map enhanced features to a compact representation
        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim//2, kernel_size=1),
            nn.BatchNorm2d(embed_dim//2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(embed_dim//2, embed_dim)
        
    def update_intra_video(self, features, class_masks):
        B, D, H, W = features.shape
        # Project features to compact representation
        proj_feat = self.proj(features).squeeze(-1).squeeze(-1)  # (B, D//2)
        proj_feat = self.fc(proj_feat)  # (B, D)
        
        # For each class, compute mean feature of the region
        for b in range(B):
            for c in range(self.num_classes):
                mask = class_masks[b, c]  # (H, W)
                if mask.sum() > 0:
                    region_feat = (features[b] * mask.unsqueeze(0)).sum(dim=(1,2)) / mask.sum()
                    # Update memory using moving average
                    if self.memory_counter[c] == 0:
                        self.memory[c] = region_feat
                    else:
                        self.memory[c] = self.momentum * self.memory[c] + (1 - self.momentum) * region_feat
                    self.memory_counter[c] += 1
                    
    def update_inter_video(self, other_memory):
        self.memory = (self.memory + other_memory) / 2
        
    def compute_similarity_loss(self, features, class_masks):
        B, D, H, W = features.shape
        proj_feat = self.proj(features).squeeze(-1).squeeze(-1)
        proj_feat = self.fc(proj_feat)  # (B, D)
        
        total_loss = 0
        count = 0
        for b in range(B):
            for c in range(self.num_classes):
                mask = class_masks[b, c]
                if mask.sum() > 0:
                    region_feat = (features[b] * mask.unsqueeze(0)).sum(dim=(1,2)) / mask.sum()
                    region_feat = F.normalize(region_feat, dim=0)
                    mem = F.normalize(self.memory[c], dim=0)
                    cos_sim = (region_feat * mem).sum()
                    loss = 1 - cos_sim
                    total_loss += loss
                    count += 1
        if count > 0:
            total_loss = total_loss / count
        return total_loss
    
    def forward(self, features, class_masks=None, other_memory=None, update=True):
        loss = torch.tensor(0.0, device=features.device)
        if class_masks is not None:
            loss = self.compute_similarity_loss(features, class_masks)
            if update:
                self.update_intra_video(features, class_masks)
        if other_memory is not None:
            self.update_inter_video(other_memory)
        return loss

class BTSurgSAM(nn.Module):
    def __init__(self, sam_model_type='vit_b', checkpoint_path=None, num_classes=10, 
                 embed_dim=768, num_frames=8, device='cuda'):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.device = device
        
        if not SAM_AVAILABLE:
            raise ImportError("segment_anything is required. Install with: pip install segment-anything")
        
        # Load SAM model
        self.sam = sam_model_registry[sam_model_type](checkpoint=checkpoint_path)
        self.sam.to(device)
        
        # Freeze image encoder and prompt encoder
        for param in self.sam.image_encoder.parameters():
            param.requires_grad = False
        for param in self.sam.prompt_encoder.parameters():
            param.requires_grad = False
        
        # Only mask decoder is trainable
        # But we will replace/add BURE and DRA, and optionally add a learnable prompt embedding
        for param in self.sam.mask_decoder.parameters():
            param.requires_grad = True
        
        # Get actual embedding dimension from SAM's image encoder
        # For vit_b, it's 768; for vit_l: 1024; for vit_h: 1280
        if sam_model_type == 'vit_b':
            self.embed_dim = 768
        elif sam_model_type == 'vit_l':
            self.embed_dim = 1024
        elif sam_model_type == 'vit_h':
            self.embed_dim = 1280
        else:
            self.embed_dim = embed_dim
        
        # BURE module
        self.bure = BURE(embed_dim=self.embed_dim, num_classes=num_classes, num_heads=8, d_state=16)
        
        # DRA module
        self.dra = DynamicRepresentationAlignment(embed_dim=self.embed_dim, num_classes=num_classes)
        
        # Learnable prompt embedding for mask decoder (since we freeze prompt encoder, we need a prompt)
        # SAM's mask decoder expects a prompt embedding from prompt encoder.
        # We create a dummy prompt embedding (a learned token) for each class? Or just a global token.
        # Here we create a learnable prompt token of shape (1, embed_dim)
        self.learnable_prompt = nn.Parameter(torch.randn(1, self.embed_dim))
        
        # Additional segmentation head for auxiliary outputs (if needed)
        self.aux_head = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.embed_dim//2),
            nn.ReLU(),
            nn.Conv2d(self.embed_dim//2, num_classes, kernel_size=1)
        )
        
    def get_image_embeddings(self, frames):
        B, T, C, H, W = frames.shape
        # SAM image encoder expects (B, 3, H, W). Process each frame individually.
        embeddings = []
        with torch.no_grad():
            for t in range(T):
                frame_t = frames[:, t]  # (B, 3, H, W)
                # Image encoder output shape: (B, D, H_feat, W_feat) where H_feat=H/16, W_feat=W/16
                emb_t = self.sam.image_encoder(frame_t)
                embeddings.append(emb_t)
        embeddings = torch.stack(embeddings, dim=1)  # (B, T, D, H_feat, W_feat)
        return embeddings
    
    def forward(self, frames, labels=None, class_masks=None, other_memory=None, update_dra=True):
        B, T, C, H, W = frames.shape
        # Get image embeddings
        img_embs = self.get_image_embeddings(frames)  # (B, T, D, H_feat, W_feat)
        
        # Apply BURE to enhance representations
        # For DRA, we need the enhanced features before mask decoder
        C_enhanced, loss_u = self.bure(img_embs, labels=labels)  # C_enhanced: (B, T, D, H_feat, W_feat)
        
        # Use the last frame's enhanced feature for segmentation
        last_feat = C_enhanced[:, -1]  # (B, D, H_feat, W_feat)
        
        # DRA loss: we need class masks for the last frame (could be derived from labels or pseudo-labels)
        loss_dra = torch.tensor(0.0, device=frames.device)
        if class_masks is not None:
            loss_dra = self.dra(last_feat, class_masks=class_masks, other_memory=other_memory, update=update_dra)

        
        # Option 1: Use auxiliary head
        seg_logits = self.aux_head(last_feat)  # (B, K, H_feat, W_feat)
        # Upsample to original resolution
        seg_logits = F.interpolate(seg_logits, size=(H, W), mode='bilinear', align_corners=False)
        
        # Option 2 (more faithful): Use SAM's mask decoder with a learnable prompt
        # We'll also compute a second output using SAM's decoder for completeness
        # This requires that we have the original image embeddings (img_embs) for the last frame.
        orig_emb = img_embs[:, -1]  # (B, D, H_feat, W_feat)
        # Create a dummy sparse prompt (no points) and dense prompt (zeros)
        B, D, Hf, Wf = orig_emb.shape
        # Learnable prompt embedding repeated for batch
        prompt_emb = self.learnable_prompt.unsqueeze(0).expand(B, -1, -1)  # (B, 1, D)
        # Dense prompt: zeros
        dense_prompt = torch.zeros(B, D, Hf, Wf, device=orig_emb.device)
        # Image positional encoding (SAM's default)
        image_pe = self.sam.prompt_encoder.get_dense_pe()  # (1, D, Hf, Wf)
        # Forward through mask decoder
        masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=orig_emb,
            image_pe=image_pe,
            sparse_prompt_embeddings=prompt_emb,
            dense_prompt_embeddings=dense_prompt,
            multimask_output=False
        )
        # Compute supervised Dice loss if labels provided
        loss_dice = torch.tensor(0.0, device=frames.device)
        if labels is not None:
            # Dice loss on seg_logits
            pred_softmax = F.softmax(seg_logits, dim=1)
            dice_loss = 0
            for c in range(self.num_classes):
                pred_c = pred_softmax[:, c].reshape(B, -1)
                target_c = (labels == c).float().reshape(B, -1)
                intersection = (pred_c * target_c).sum(dim=1)
                union = pred_c.sum(dim=1) + target_c.sum(dim=1)
                dice = (2 * intersection + 1e-6) / (union + 1e-6)
                dice_loss += (1 - dice.mean())
            loss_dice = dice_loss / self.num_classes
        
        total_loss = loss_u + loss_dra + loss_dice
        return seg_logits, total_loss, loss_u, loss_dra, loss_dice

