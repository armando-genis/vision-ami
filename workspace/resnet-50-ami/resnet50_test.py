import torch
import torchvision
import torchvision.transforms as T
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights


# ------------------------------
# 1. Load a Pretrained Model
# ------------------------------
# This is DeepLabV3 with a ResNet-50 backbone, pretrained on COCO
model = torchvision.models.segmentation.deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT, progress=True)
model.eval()  # set to evaluation mode

print("Model loaded successfully!")

# ------------------------------
# 2. Download Images/Dataset
# ------------------------------

