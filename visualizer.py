import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

def get_gradcam(model, img_tensor, target_class):
    model.eval()
    
    # We target the last convolutional layer of ResNet-18
    feature_map = None
    gradient = None

    def hook_feature(module, input, output):
        nonlocal feature_map
        feature_map = output.detach()

    def hook_gradient(module, grad_in, grad_out):
        nonlocal gradient
        gradient = grad_out[0].detach()

    # Register hooks on the last conv layer
    target_layer = model.layer4[-1].conv2
    handle_f = target_layer.register_forward_hook(hook_feature)
    handle_g = target_layer.register_backward_hook(hook_gradient)

    # Forward pass
    output = model(img_tensor)
    score = output[:, target_class]
    
    # Backward pass
    model.zero_grad()
    score.backward()

    # Process Grad-CAM
    weights = torch.mean(gradient, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * feature_map, dim=1).squeeze().cpu().numpy()
    
    # Normalize Heatmap
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-10)
    
    # Cleanup hooks
    handle_f.remove()
    handle_g.remove()
    
    return cam

def overlay_heatmap(img, heatmap):
    # Convert grayscale image to BGR for color overlay
    img_bgr = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Combine original image and heatmap
    result = cv2.addWeighted(img_bgr, 0.6, heatmap_color, 0.4, 0)
    return result