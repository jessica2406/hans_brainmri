import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

def generate_heatmap(model, img_tensor, original_img):
    model.eval()
    
    # We want the output of the very last convolutional layer
    target_layer = model.layer4[-1].conv2
    
    # Hook to get the gradients
    gradients = []
    def save_gradient(grad):
        gradients.append(grad)
    
    target_layer.register_backward_hook(lambda module, grad_in, grad_out: save_gradient(grad_out[0]))
    
    # Forward pass
    output = model(img_tensor)
    category = torch.argmax(output)
    
    # Backward pass
    model.zero_grad()
    output[0, category].backward()
    
    # Process gradients
    grads = gradients[0].cpu().data.numpy()[0]
    weights = np.mean(grads, axis=(1, 2))
    
    # Get feature map
    # Note: This is a simplified version of Grad-CAM
    # For a beginner project, we can visualize the activations directly
    feature_map = model.layer4[-1].conv2.weight.data[0].cpu().numpy()[0]
    heatmap = cv2.resize(feature_map, (224, 224))
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    
    return heatmap

# This will be integrated into your app.py next!