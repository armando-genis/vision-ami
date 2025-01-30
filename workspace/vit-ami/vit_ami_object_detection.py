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

# For confusion matrix
from sklearn.metrics import confusion_matrix

# We'll need functional for upsampling and other ops
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

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.images_dir, img_name)

        # The mask must have the same base name but .png
        mask_name = os.path.splitext(img_name)[0] + ".png"
        mask_path = os.path.join(self.masks_dir, mask_name)

        # Load image (RGB) and mask
        image = Image.open(img_path).convert("RGB")
        mask  = Image.open(mask_path)

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        # Convert mask to tensor (long) so each pixel is an integer class ID
        mask_tensor = torch.as_tensor(np.array(mask), dtype=torch.long)

        # Convert image if not already a tensor
        if not isinstance(image, torch.Tensor):
            image = transforms.ToTensor()(image)

        return image, mask_tensor


# ---------------------------
# 1) mIoU Helpers
# ---------------------------
def compute_mIoU_from_cm(cm):
    """
    Given a confusion matrix `cm` (num_classes x num_classes),
    compute mean Intersection-over-Union (mIoU).
    row = true class, col = predicted class
    IoU_c = TP_c / (TP_c + FP_c + FN_c)
    """
    num_classes = cm.shape[0]
    IoUs = []
    for c in range(num_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp  # sum of row c except cm[c,c]
        fp = cm[:, c].sum() - tp  # sum of col c except cm[c,c]
        denom = (tp + fp + fn)
        iou_c = (tp / denom) if denom != 0 else 0
        IoUs.append(iou_c)
    return np.mean(IoUs)

def compute_confusion_matrix(model, loader, num_classes, device):
    """
    Runs inference on the loader, accumulates a confusion matrix for the entire dataset.
    Returns: cm (num_classes x num_classes)
    """
    model.eval()
    all_preds = []
    all_labels= []
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(pixel_values=images)
            logits = outputs.logits

            # Upsample to match the mask size
            logits_upsampled = F.interpolate(
                logits,
                size=(masks.shape[-2], masks.shape[-1]),
                mode="bilinear",
                align_corners=False
            )
            pred_mask = logits_upsampled.argmax(dim=1)

            all_preds.append(pred_mask.cpu().numpy().ravel())
            all_labels.append(masks.cpu().numpy().ravel())

    all_preds = np.concatenate(all_preds)
    all_labels= np.concatenate(all_labels)

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    return cm


def main():
    # ---------------------------
    # 2) Paths and Setup
    # ---------------------------
    data_root = "/workspace/my_seg_dataset"
    train_img_dir = os.path.join(data_root, "train", "images")
    train_mask_dir= os.path.join(data_root, "train", "masks")
    val_img_dir   = os.path.join(data_root, "val",   "images")
    val_mask_dir  = os.path.join(data_root, "val",   "masks")

    num_classes = 6  # includes background

    # Transforms
    image_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
    ])

    # Datasets
    train_dataset = MySegDataset(train_img_dir, train_mask_dir, image_transform, mask_transform)
    val_dataset   = MySegDataset(val_img_dir,   val_mask_dir,   image_transform, mask_transform)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=2, shuffle=False)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")

    # ---------------------------
    # 3) Model
    # ---------------------------
    model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
    feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)
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
    num_epochs = 200

    # We will pick best model by highest mIoU (not by val loss)
    best_mIoU = 0.0
    best_epoch = 0
    best_model_state = None

    # ---------------------------
    # 5) Training Loop
    # ---------------------------
    for epoch in range(num_epochs):
        # --- Training ---
        model.train()
        total_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(pixel_values=images)
            logits = outputs.logits
            # Upsample to match mask size
            logits_upsampled = F.interpolate(
                logits,
                size=(masks.shape[-2], masks.shape[-1]),
                mode="bilinear",
                align_corners=False
            )

            loss = criterion(logits_upsampled, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        # --- Validation Loss ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(pixel_values=images)
                logits = outputs.logits
                logits_upsampled = F.interpolate(
                    logits,
                    size=(masks.shape[-2], masks.shape[-1]),
                    mode="bilinear",
                    align_corners=False
                )
                loss = criterion(logits_upsampled, masks)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        # --- Compute mIoU this epoch ---
        cm = compute_confusion_matrix(model, val_loader, num_classes, device)
        mIoU = compute_mIoU_from_cm(cm)

        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, mIoU: {mIoU:.4f}")

        # If this is the best mIoU so far, save model
        if mIoU > best_mIoU:
            best_mIoU = mIoU
            best_epoch = epoch + 1
            best_model_state = {
                'model_state': model.state_dict(),
                'epoch': best_epoch,
                'mIoU': best_mIoU
            }
            print(f"** Best model so far (mIoU={mIoU:.4f}), saving checkpoint...")

    print("Training complete!")
    print(f"Best model found at epoch {best_epoch} with mIoU={best_mIoU:.4f}")

    # ---------------------------
    # 6) Load & Save Best Model
    # ---------------------------
    if best_model_state is not None:
        model.load_state_dict(best_model_state['model_state'])
    save_dir = "segformer_model_best_mIoU"
    model.save_pretrained(save_dir)
    feature_extractor.save_pretrained(save_dir)
    print(f"Best model (mIoU) saved to: {save_dir}/")

    # ---------------------------
    # 7) Final Confusion Matrix & mIoU with Best Model
    # ---------------------------
    final_cm = compute_confusion_matrix(model, val_loader, num_classes, device)
    final_mIoU = compute_mIoU_from_cm(final_cm)
    print("\nConfusion Matrix (pixel-level) for best model:")
    print(final_cm)
    print(f"Final mIoU: {final_mIoU:.4f}")

if __name__ == "__main__":
    main()
