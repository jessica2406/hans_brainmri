import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes=5):
    # Load a pre-trained ResNet18
    model = models.resnet18(weights='DEFAULT')
    
    # Change the first layer from 3 channels (RGB) to 1 channel (Grayscale)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # Change the last layer to output our specific number of classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model