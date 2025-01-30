import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm


# For confusion matrix
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# ---------------------------
# 1. Dataset paths
# ---------------------------
train_dir = "/workspace/dataset/train"
val_dir   = "/workspace/dataset/val"

# ---------------------------
# 2. Data transforms
# ---------------------------
# For training, we typically include data augmentation
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# For validation, use deterministic resizing and cropping
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------------------
# 3. Create Datasets
# ---------------------------
train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
val_dataset   = datasets.ImageFolder(root=val_dir,   transform=val_transforms)

# ---------------------------
# 4. Create DataLoaders
# ---------------------------
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=4)

# Print basic dataset info
print("Classes found:", train_dataset.classes)
print("Number of training samples:", len(train_dataset))
print("Number of validation samples:", len(val_dataset))


# ---------------------------
# 5. Create/Load Model
# ---------------------------
# If you have 10 classes, for example, the final layer should match num_classes=10.
num_classes = len(train_dataset.classes)
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)


# ---------------------------
# 6. Define Loss and Optimizer
# ---------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ---------------------------
# 7. Training Setup
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 10  # You can change this

# ---------------------------
# 8. Training Loop
# ---------------------------
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    
    for images, labels in train_loader:
        # Move images and labels to GPU (if available)
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Compute average training loss for the epoch
    epoch_loss = running_loss / len(train_loader)
    
    # Validation step
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_accuracy = 100.0 * correct / total
    
    print(f"Epoch [{epoch+1}/{num_epochs}]  "
            f"Train Loss: {epoch_loss:.4f}  "
            f"Val Accuracy: {val_accuracy:.2f}%")
    
# ---------------------------
# 9. Save Model Weights
# ---------------------------
save_path = "vit_base_patch16_224.pth"
torch.save(model.state_dict(), save_path)
print(f"Model weights saved to {save_path}")

print("Training complete!")


# ---------------------------
# 10. Compute Confusion Matrix
# ---------------------------
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        # Move tensors to CPU numpy arrays
        all_preds.append(predicted.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

# Flatten the lists of arrays
all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
print("\nConfusion Matrix:")
print(cm)

# Optional: Classification report (precision, recall, F1)
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))

print("Training + Confusion Matrix computation complete!")



# ---------------------------
# pip install scikit-learn
# pip install timm

