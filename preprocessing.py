"""
Image Preprocessing and Postprocessing Utilities

Handles:
- Image resizing and normalization
- Tensor conversion for model input
- Mask postprocessing and thresholding
"""

import numpy as np
import torch
from PIL import Image
import cv2


def preprocess_image(image, target_size=(256, 256)):
    """
    Preprocess input image for model inference
    
    Args:
        image: PIL Image or numpy array
        target_size: Target size for resizing (default: 256x256 as per paper)
    
    Returns:
        tensor: Preprocessed image tensor [1, 3, H, W]
        original_size: Original image size for postprocessing
    """
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        original_size = image.size
        image = np.array(image)
    else:
        original_size = (image.shape[1], image.shape[0])
    
    # Ensure RGB format
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[2] == 4:
        image = image[:, :, :3]
    
    # Resize to target size (256x256 as specified in paper)
    image_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Normalize to [0, 1] range (as per paper: "pixel intensities are normalized to [0, 1]")
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    # Convert to tensor [H, W, C] -> [C, H, W]
    image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).float()
    
    # Add batch dimension [1, C, H, W]
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor, original_size


def postprocess_mask(mask_tensor, original_size, threshold=0.5):
    """
    Postprocess model output mask
    
    Args:
        mask_tensor: Model output tensor [1, 1, H, W]
        original_size: Original image size (W, H) for resizing
        threshold: Threshold for binary mask (default: 0.5)
    
    Returns:
        binary_mask: Binary mask as numpy array
        probability_mask: Probability mask as numpy array
    """
    # Remove batch and channel dimensions
    mask = mask_tensor.squeeze().cpu().numpy()
    
    # Resize to original size
    binary_mask = cv2.resize(mask, original_size, interpolation=cv2.INTER_LINEAR)
    
    # Keep probability mask
    probability_mask = binary_mask.copy()
    
    # Apply threshold for binary mask
    binary_mask = (binary_mask >= threshold).astype(np.uint8) * 255
    
    return binary_mask, probability_mask


def apply_colormap(mask, colormap='jet'):
    """
    Apply colormap to segmentation mask for visualization
    
    Args:
        mask: Binary or probability mask
        colormap: OpenCV colormap name
    
    Returns:
        colored_mask: RGB mask with colormap applied
    """
    # Normalize to 0-255 if needed
    if mask.max() <= 1.0:
        mask = (mask * 255).astype(np.uint8)
    
    # Apply colormap
    colored = cv2.applyColorMap(mask, getattr(cv2, f'COLORMAP_{colormap.upper()}'))
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    
    return colored


def compute_mask_statistics(binary_mask):
    """
    Compute statistics about the predicted mask
    
    Args:
        binary_mask: Binary mask (0 or 255)
    
    Returns:
        stats: Dictionary with mask statistics
    """
    # Normalize to 0-1
    mask_normalized = binary_mask / 255.0
    
    # Compute statistics
    total_pixels = mask_normalized.size
    polyp_pixels = np.sum(mask_normalized)
    polyp_percentage = (polyp_pixels / total_pixels) * 100
    
    # Find contours for polyp count and size
    contours, _ = cv2.findContours(
        binary_mask, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    num_polyps = len(contours)
    
    # Compute areas
    areas = [cv2.contourArea(cnt) for cnt in contours]
    total_polyp_area = sum(areas) if areas else 0
    
    stats = {
        'total_pixels': total_pixels,
        'polyp_pixels': int(polyp_pixels),
        'polyp_percentage': polyp_percentage,
        'num_detected_regions': num_polyps,
        'total_polyp_area_pixels': int(total_polyp_area),
        'average_polyp_area_pixels': int(np.mean(areas)) if areas else 0
    }
    
    return stats


def enhance_contrast(image, clip_limit=2.0, grid_size=8):
    """
    Apply CLAHE contrast enhancement for colonoscopy images
    
    Args:
        image: Input image (numpy array)
        clip_limit: CLAHE clip limit
        grid_size: CLAHE grid size
    
    Returns:
        enhanced: Contrast-enhanced image
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    l_enhanced = clahe.apply(l)
    
    # Merge back
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    
    return enhanced


def remove_specular_highlights(image, threshold=240):
    """
    Remove specular highlights common in colonoscopy images
    
    Args:
        image: Input image
        threshold: Brightness threshold for highlight detection
    
    Returns:
        inpainted: Image with highlights removed
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Create mask for bright regions
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Dilate mask slightly
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    # Inpaint
    inpainted = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    
    return inpainted


def preprocess_pipeline(image, enhance=True, remove_highlights=False):
    """
    Complete preprocessing pipeline for colonoscopy images
    
    Args:
        image: Input PIL Image or numpy array
        enhance: Whether to apply contrast enhancement
        remove_highlights: Whether to remove specular highlights
    
    Returns:
        tensor: Preprocessed tensor
        original_size: Original image size
        processed_image: Processed numpy image (for display)
    """
    # Convert to numpy if PIL
    if isinstance(image, Image.Image):
        original_size = image.size
        image_np = np.array(image)
    else:
        original_size = (image.shape[1], image.shape[0])
        image_np = image.copy()
    
    # Ensure RGB
    if len(image_np.shape) == 2:
        image_np = np.stack([image_np] * 3, axis=-1)
    elif image_np.shape[2] == 4:
        image_np = image_np[:, :, :3]
    
    # Optional preprocessing steps
    if remove_highlights:
        image_np = remove_specular_highlights(image_np)
    
    if enhance:
        image_np = enhance_contrast(image_np)
    
    # Standard preprocessing
    tensor, _ = preprocess_image(image_np, target_size=(256, 256))
    
    return tensor, original_size, image_np


if __name__ == "__main__":
    # Test preprocessing
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    tensor, orig_size = preprocess_image(test_image)
    print(f"Input shape: {test_image.shape}")
    print(f"Output tensor shape: {tensor.shape}")
    print(f"Original size: {orig_size}")
