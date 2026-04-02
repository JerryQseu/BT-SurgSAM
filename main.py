
def train_epoch(model, dataloader, optimizer, device, num_classes, use_dra=True):
    model.train()
    total_loss = 0
    for batch in dataloader:
        # batch: video frames (B, T, 3, H, W), labels for last frame (B, H, W), optional class_masks
        frames = batch['frames'].to(device)
        labels = batch['labels'].to(device)  # for last frame
        # For DRA, we need class masks for the last frame (binary masks per class)
        if use_dra:
            B, H, W = labels.shape
            class_masks = torch.zeros(B, num_classes, H, W, device=device)
            for c in range(num_classes):
                class_masks[:, c] = (labels == c).float()
        else:
            class_masks = None
        
        optimizer.zero_grad()
        seg_logits, loss, loss_u, loss_dra, loss_dice = model(frames, labels=labels, class_masks=class_masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def main():
    # Example configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_classes = 10  # e.g., instruments + tissues
    num_frames = 8
    batch_size = 2
    learning_rate = 1e-4
    
    # Initialize model (need SAM checkpoint)
    # Download SAM checkpoint from https://github.com/facebookresearch/segment-anything
    checkpoint_path = "sam_vit_b_01ec64.pth"
    model = BTSurgSAM(sam_model_type='vit_b', checkpoint_path=checkpoint_path, 
                      num_classes=num_classes, num_frames=num_frames, device=device)
    model.to(device)
    
    # Optimizer: only train mask decoder, BURE, DRA parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
    
    # Dummy dataloader (replace with actual surgical video dataset)
    # The dataset should return: frames (B,T,3,H,W), labels (B,H,W) for the last frame
    # For scarce-label setting, only a small fraction of videos have labels.
    # Here we assume all frames in the batch have labels for simplicity.
    
    # Example dummy dataset
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples=100, T=8, H=256, W=256, C=3, num_classes=10):
            self.num_samples = num_samples
            self.T = T
            self.H = H
            self.W = W
            self.C = C
            self.num_classes = num_classes
        def __len__(self):
            return self.num_samples
        def __getitem__(self, idx):
            frames = torch.randn(self.T, self.C, self.H, self.W)
            labels = torch.randint(0, self.num_classes, (self.H, self.W))
            return {'frames': frames, 'labels': labels}
    
    dataset = DummyDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    epochs = 10
    for epoch in range(epochs):
        loss = train_epoch(model, dataloader, optimizer, device, num_classes, use_dra=True)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
    
    print("Training completed.")

if __name__ == "__main__":
    main()