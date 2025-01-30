import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# If you're on Transformers >= 4.26, SegformerFeatureExtractor may be renamed to SegformerImageProcessor
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

# For confusion matrix (if you want it later)
from sklearn.metrics import confusion_matrix

# We'll need functional for upsampling
import torch.nn.functional as F

class MySegDataset(Dataset):
    """
    Loads images + semantic segmentation masks from a folder structure like:
      my_seg_dataset/
        train/
          images/
          masks/
        val/
          images/
          masks/
    Each mask pixel is a class ID (0 = background, 1..N = other classes).
    """
    def __init__(self, images_dir, masks_dir, transform=None, mask_transform=None):
        self.images_dir = images_dir
        self.masks_dir  = masks_dir
        self.transform = transform
        self.mask_transform = mask_transform

        self.image_names = sorted(os.listdir(images_dir))
        # Filter for actual image files if needed
        # self.image_names = [f for f in self.image_names if f.endswith(".jpg") or f.endswith(".png")]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.images_dir, img_name)

        # The mask must have the same base name but .png (adjust if your extension is different)
        mask_name = os.path.splitext(img_name)[0] + ".png"
        mask_path = os.path.join(self.masks_dir, mask_name)

        # Load image (RGB) and mask
        image = Image.open(img_path).convert("RGB")
        mask  = Image.open(mask_path)  # grayscale or paletted

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        # Convert mask to tensor (long) so each pixel is an integer class ID
        mask_tensor = torch.as_tensor(np.array(mask), dtype=torch.long)

        # If the image transforms didn't already return a tensor, do so here
        if not isinstance(image, torch.Tensor):
            image = transforms.ToTensor()(image)

        return image, mask_tensor


def main():
    # ---------------------------
    # 1) Path to your dataset
    # ---------------------------
    data_root = "/workspace/my_seg_dataset"
    train_img_dir = os.path.join(data_root, "train", "images")
    train_mask_dir= os.path.join(data_root, "train", "masks")
    val_img_dir   = os.path.join(data_root, "val",   "images")
    val_mask_dir  = os.path.join(data_root, "val",   "masks")

    # Number of classes (including background)
    num_classes = 6

    # ---------------------------
    # 2) Dataset & DataLoader
    # ---------------------------
    # Example: just resize everything to 512x512
    image_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
    ])

    train_dataset = MySegDataset(train_img_dir, train_mask_dir, image_transform, mask_transform)
    val_dataset   = MySegDataset(val_img_dir,   val_mask_dir,   image_transform, mask_transform)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=2, shuffle=False)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")

    # ---------------------------
    # 3) Load SegFormer w/ custom num_labels
    # ---------------------------
    model_name = "nvidia/segformer-b0-finetuned-ade-512-512"

    # SegformerFeatureExtractor is deprecated in Transformers v5+,
    # so you might see "SegformerImageProcessor" if you're on a newer version.
    feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)

    # The key is setting 'num_labels' + 'ignore_mismatched_sizes=True'
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ---------------------------
    # 4) Training Setup
    # ---------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    num_epochs = 500

    # Track the best validation loss
    best_val_loss = float('inf')
    best_epoch = 0
    best_model_state = None

    # ---------------------------
    # 5) Training Loop
    # ---------------------------
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device)  # (B, 3, 512, 512)
            masks = masks.to(device)    # (B, 512, 512)

            outputs = model(pixel_values=images)
            logits = outputs.logits     # (B, num_classes, h_out, w_out), e.g. 128x128

            # 1) Upsample logits to match (512, 512)
            logits_upsampled = F.interpolate(
                logits,
                size=(masks.shape[-2], masks.shape[-1]),  # (512, 512)
                mode="bilinear",
                align_corners=False
            )

            loss = criterion(logits_upsampled, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(pixel_values=images)
                logits = outputs.logits  # e.g. (B, num_classes, 128, 128)

                # Upsample
                logits_upsampled = F.interpolate(
                    logits,
                    size=(masks.shape[-2], masks.shape[-1]),
                    mode="bilinear",
                    align_corners=False
                )

                loss = criterion(logits_upsampled, masks)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Check if this is the best model so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            # Save the best model weights in memory
            best_model_state = {
                'model_state': model.state_dict(),
                'epoch': best_epoch,
                'val_loss': best_val_loss
            }
            print(f"** Best model so far! Saving (val_loss={best_val_loss:.4f})")

    print("Training complete!")
    print(f"Best model found at epoch {best_epoch} with val_loss={best_val_loss:.4f}")

    # ---------------------------
    # 6) Load and Save the Best Model Weights
    # ---------------------------
    model.load_state_dict(best_model_state['model_state'])
    save_dir = "segformer_model_best"
    model.save_pretrained(save_dir)
    feature_extractor.save_pretrained(save_dir)
    print(f"Best model saved to: {save_dir}/")

    # Optional: Print the confusion matrix using the best model
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(pixel_values=images)
            logits = outputs.logits
            # Upsample for predictions
            logits_upsampled = F.interpolate(
                logits,
                size=(masks.shape[-2], masks.shape[-1]),
                mode="bilinear",
                align_corners=False
            )

            pred_mask = logits_upsampled.argmax(dim=1)  # (B, 512, 512)
            preds_np = pred_mask.cpu().numpy().ravel()
            labels_np = masks.cpu().numpy().ravel()

            all_preds.append(preds_np)
            all_labels.append(labels_np)

    all_preds = np.concatenate(all_preds)
    all_labels= np.concatenate(all_labels)

    # Compute confusion matrix for up to 'num_classes' classes
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    print("\nConfusion Matrix (pixel-level):")
    print(cm)

    # Quick pixel accuracy
    pixel_acc = np.trace(cm) / cm.sum()
    print(f"Pixel Accuracy: {pixel_acc:.2%}")


if __name__ == "__main__":
    main()
