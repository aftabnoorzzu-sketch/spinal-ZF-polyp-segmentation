"""
Explainability and Interpretability Utilities

Implements Grad-CAM style visualization for understanding model decisions
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt


class GradCAMExplainer:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for segmentation models
    
    Helps visualize which regions of the input image are important for the model's prediction.
    """
    
    def __init__(self, model, target_layer):
        """
        Args:
            model: The segmentation model
            target_layer: The layer to compute Grad-CAM for (typically last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Hook to save layer activations"""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Hook to save layer gradients"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, target_class=1):
        """
        Generate CAM for input image
        
        Args:
            input_image: Input tensor [1, C, H, W]
            target_class: Target class (1 for polyp)
        
        Returns:
            cam: Class Activation Map
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_image)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Compute gradients for target class
        if target_class is not None:
            target = output[:, 0, :, :].mean()
        else:
            target = output.mean()
        
        target.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Compute weights (global average pooling of gradients)
        weights = gradients.mean(dim=(1, 2), keepdim=True)  # [C, 1, 1]
        
        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=0)  # [H, W]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy()
    
    def overlay_cam(self, image, cam, alpha=0.5, colormap=cv2.COLORMAP_JET):
        """
        Overlay CAM on original image
        
        Args:
            image: Original image (numpy array, RGB)
            cam: Class Activation Map
            alpha: Transparency
            colormap: OpenCV colormap
        
        Returns:
            overlay: Image with CAM overlay
        """
        # Resize CAM to image size
        cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))
        
        # Convert to uint8
        cam_uint8 = (cam_resized * 255).astype(np.uint8)
        
        # Apply colormap
        cam_colored = cv2.applyColorMap(cam_uint8, colormap)
        cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)
        
        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Overlay
        overlay = cv2.addWeighted(image, 1 - alpha, cam_colored, alpha, 0)
        
        return overlay


class SaliencyMapGenerator:
    """
    Generate saliency maps using gradient-based methods
    """
    
    def __init__(self, model):
        self.model = model
    
    def generate_saliency(self, input_image):
        """
        Generate saliency map for input image
        
        Args:
            input_image: Input tensor [1, C, H, W]
        
        Returns:
            saliency: Saliency map
        """
        self.model.eval()
        
        # Enable gradient computation for input
        input_image.requires_grad = True
        
        # Forward pass
        output = self.model(input_image)
        
        # Compute gradient of output mean w.r.t. input
        output.mean().backward()
        
        # Get gradient
        gradient = input_image.grad[0]  # [C, H, W]
        
        # Compute saliency (magnitude of gradient)
        saliency = gradient.abs().max(dim=0)[0]  # [H, W]
        
        # Normalize
        saliency = saliency - saliency.min()
        saliency = saliency / (saliency.max() + 1e-8)
        
        return saliency.cpu().numpy()


def generate_explainability_map(model, input_tensor, method='gradcam', target_layer=None):
    """
    Generate explainability map using specified method
    
    Args:
        model: Segmentation model
        input_tensor: Input image tensor
        method: 'gradcam' or 'saliency'
        target_layer: Target layer for Grad-CAM
    
    Returns:
        explain_map: Explainability map
    """
    if method == 'gradcam':
        if target_layer is None:
            # Find last convolutional layer
            for name, module in reversed(list(model.named_modules())):
                if isinstance(module, torch.nn.Conv2d):
                    target_layer = module
                    break
        
        explainer = GradCAMExplainer(model, target_layer)
        return explainer.generate_cam(input_tensor)
    
    elif method == 'saliency':
        explainer = SaliencyMapGenerator(model)
        return explainer.generate_saliency(input_tensor)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def create_explainability_visualization(image, mask, explain_map, title="Model Explanation"):
    """
    Create comprehensive explainability visualization
    
    Args:
        image: Original image
        mask: Predicted mask
        explain_map: Explainability map (Grad-CAM or saliency)
        title: Plot title
    
    Returns:
        fig: Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Prediction mask
    axes[0, 1].imshow(mask, cmap='gray')
    axes[0, 1].set_title('Predicted Mask', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Explainability map
    axes[1, 0].imshow(explain_map, cmap='hot')
    axes[1, 0].set_title('Attention Map (Grad-CAM)', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Overlay
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    explain_resized = cv2.resize(explain_map, (image.shape[1], image.shape[0]))
    explain_uint8 = (explain_resized * 255).astype(np.uint8)
    explain_colored = cv2.applyColorMap(explain_uint8, cv2.COLORMAP_HOT)
    explain_colored = cv2.cvtColor(explain_colored, cv2.COLOR_BGR2RGB)
    
    overlay = cv2.addWeighted(image, 0.6, explain_colored, 0.4, 0)
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Overlay on Image', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def compute_feature_importance(model, input_tensor, n_segments=10):
    """
    Compute feature importance by occluding parts of the input
    
    Args:
        model: Segmentation model
        input_tensor: Input image tensor
        n_segments: Number of segments to divide image into
    
    Returns:
        importance_map: Feature importance map
    """
    model.eval()
    
    with torch.no_grad():
        # Get baseline prediction
        baseline_output = model(input_tensor)
        baseline_score = baseline_output.mean().item()
        
        # Get image dimensions
        _, _, h, w = input_tensor.shape
        segment_h = h // n_segments
        segment_w = w // n_segments
        
        importance_map = np.zeros((n_segments, n_segments))
        
        # Test each segment
        for i in range(n_segments):
            for j in range(n_segments):
                # Create occluded version
                occluded = input_tensor.clone()
                occluded[:, :, i*segment_h:(i+1)*segment_h, j*segment_w:(j+1)*segment_w] = 0
                
                # Get prediction with occlusion
                occluded_output = model(occluded)
                occluded_score = occluded_output.mean().item()
                
                # Importance is the drop in score
                importance_map[i, j] = baseline_score - occluded_score
    
    return importance_map


if __name__ == "__main__":
    # Test explainability
    from models.spinal_zf_model import create_model
    
    model = create_model()
    test_input = torch.randn(1, 3, 256, 256)
    
    # Test Grad-CAM
    cam = generate_explainability_map(model, test_input, method='gradcam')
    print(f"Grad-CAM shape: {cam.shape}")
    
    # Test saliency
    saliency = generate_explainability_map(model, test_input, method='saliency')
    print(f"Saliency shape: {saliency.shape}")
