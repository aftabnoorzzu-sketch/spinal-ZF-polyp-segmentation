"""
Visualization Utilities for Segmentation Results

Provides functions for:
- Creating overlay visualizations
- Plotting segmentation results
- Generating comparison figures
- Computing and displaying metrics
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import cv2
from PIL import Image
import io
import base64


def create_overlay(image, mask, alpha=0.5, color=(255, 0, 0)):
    """
    Create segmentation overlay on original image
    
    Args:
        image: Original image (numpy array, RGB)
        mask: Binary mask (numpy array)
        alpha: Transparency for overlay (0-1)
        color: Overlay color (R, G, B)
    
    Returns:
        overlay: Image with segmentation overlay
    """
    # Ensure image is uint8
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # Ensure mask is binary
    if mask.max() <= 1.0:
        mask = (mask * 255).astype(np.uint8)
    
    # Create colored mask
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 127] = color
    
    # Apply overlay
    overlay = cv2.addWeighted(image, 1.0, colored_mask, alpha, 0)
    
    # Add contour outline
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = cv2.drawContours(overlay, contours, -1, color, 2)
    
    return overlay


def create_probability_heatmap(image, probability_mask, alpha=0.6):
    """
    Create heatmap visualization of prediction probabilities
    
    Args:
        image: Original image
        probability_mask: Probability mask (0-1 values)
        alpha: Transparency for heatmap
    
    Returns:
        heatmap_overlay: Image with probability heatmap
    """
    # Ensure image is uint8
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # Convert probability to colormap
    prob_uint8 = (probability_mask * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(prob_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Apply overlay
    heatmap_overlay = cv2.addWeighted(image, 1.0 - alpha, heatmap, alpha, 0)
    
    return heatmap_overlay


def plot_segmentation_results(image, mask, probability_mask=None, figsize=(16, 4)):
    """
    Create comprehensive segmentation result plot
    
    Args:
        image: Original image
        mask: Binary segmentation mask
        probability_mask: Probability mask (optional)
        figsize: Figure size
    
    Returns:
        fig: Matplotlib figure
    """
    if probability_mask is not None:
        fig, axes = plt.subplots(1, 4, figsize=figsize)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Binary mask
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Segmentation Mask', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Overlay
    overlay = create_overlay(image, mask, alpha=0.5, color=(255, 0, 0))
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay (Red = Polyp)', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    # Probability heatmap (if provided)
    if probability_mask is not None:
        heatmap = create_probability_heatmap(image, probability_mask, alpha=0.6)
        axes[3].imshow(heatmap)
        axes[3].set_title('Probability Heatmap', fontsize=12, fontweight='bold')
        axes[3].axis('off')
    
    plt.tight_layout()
    return fig


def create_comparison_figure(image, mask, ground_truth=None, figsize=(14, 4)):
    """
    Create comparison figure with optional ground truth
    
    Args:
        image: Original image
        mask: Predicted mask
        ground_truth: Ground truth mask (optional)
        figsize: Figure size
    
    Returns:
        fig: Matplotlib figure
    """
    if ground_truth is not None:
        fig, axes = plt.subplots(1, 4, figsize=figsize)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Original
    axes[0].imshow(image)
    axes[0].set_title('Input Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Prediction
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Predicted Mask', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Prediction overlay
    overlay_pred = create_overlay(image, mask, alpha=0.5, color=(0, 255, 0))
    axes[2].imshow(overlay_pred)
    axes[2].set_title('Prediction Overlay (Green)', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    # Ground truth comparison (if provided)
    if ground_truth is not None:
        overlay_gt = create_overlay(image, ground_truth, alpha=0.5, color=(255, 0, 0))
        axes[3].imshow(overlay_gt)
        axes[3].set_title('Ground Truth Overlay (Red)', fontsize=12, fontweight='bold')
        axes[3].axis('off')
    
    plt.tight_layout()
    return fig


def compute_metrics_display(pred_mask, gt_mask=None):
    """
    Compute and format metrics for display
    
    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth mask (optional)
    
    Returns:
        metrics: Dictionary of computed metrics
    """
    metrics = {}
    
    # Normalize masks
    pred = (pred_mask > 127).astype(np.float32)
    
    # Basic statistics
    metrics['polyp_pixels'] = int(np.sum(pred))
    metrics['polyp_percentage'] = float(np.sum(pred) / pred.size * 100)
    
    # Find contours
    contours, _ = cv2.findContours(
        (pred * 255).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    metrics['num_regions'] = len(contours)
    
    if gt_mask is not None:
        gt = (gt_mask > 127).astype(np.float32)
        
        # Compute IoU
        intersection = np.sum(pred * gt)
        union = np.sum((pred + gt) > 0)
        metrics['iou'] = float(intersection / (union + 1e-7))
        
        # Compute Dice
        dice = 2 * intersection / (np.sum(pred) + np.sum(gt) + 1e-7)
        metrics['dice'] = float(dice)
        
        # Compute Precision and Recall
        tp = np.sum(pred * gt)
        fp = np.sum(pred * (1 - gt))
        fn = np.sum((1 - pred) * gt)
        
        metrics['precision'] = float(tp / (tp + fp + 1e-7))
        metrics['recall'] = float(tp / (tp + fn + 1e-7))
        metrics['f1'] = float(2 * metrics['precision'] * metrics['recall'] / 
                            (metrics['precision'] + metrics['recall'] + 1e-7))
    
    return metrics


def create_metrics_bar_chart(metrics, figsize=(10, 5)):
    """
    Create bar chart of segmentation metrics
    
    Args:
        metrics: Dictionary of metrics
        figsize: Figure size
    
    Returns:
        fig: Matplotlib figure
    """
    metric_names = ['Dice', 'IoU', 'Precision', 'Recall', 'F1-Score']
    metric_keys = ['dice', 'iou', 'precision', 'recall', 'f1']
    
    values = [metrics.get(k, 0) for k in metric_keys]
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']
    
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(metric_names, values, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Segmentation Performance Metrics', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_confidence_gauge(confidence, figsize=(6, 4)):
    """
    Create a gauge visualization for confidence score
    
    Args:
        confidence: Confidence score (0-1)
        figsize: Figure size
    
    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create color gradient based on confidence
    if confidence >= 0.9:
        color = '#2ecc71'  # Green
        label = 'High Confidence'
    elif confidence >= 0.7:
        color = '#f39c12'  # Orange
        label = 'Moderate Confidence'
    else:
        color = '#e74c3c'  # Red
        label = 'Low Confidence'
    
    # Draw gauge
    theta = np.linspace(0, np.pi, 100)
    r = 1.0
    
    # Background arc
    ax.fill_between(np.cos(theta), np.sin(theta), 0, alpha=0.1, color='gray')
    
    # Value arc
    value_theta = theta[:int(confidence * 100)]
    ax.fill_between(np.cos(value_theta), np.sin(value_theta), 0, alpha=0.7, color=color)
    
    # Add needle
    needle_angle = np.pi * (1 - confidence)
    ax.arrow(0, 0, 0.8 * np.cos(needle_angle), 0.8 * np.sin(needle_angle),
             head_width=0.05, head_length=0.05, fc='black', ec='black')
    
    # Add text
    ax.text(0, -0.3, f'{confidence:.1%}', ha='center', va='center',
            fontsize=24, fontweight='bold')
    ax.text(0, -0.6, label, ha='center', va='center', fontsize=12)
    
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.8, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    return fig


def fig_to_array(fig):
    """
    Convert matplotlib figure to numpy array
    
    Args:
        fig: Matplotlib figure
    
    Returns:
        arr: Numpy array representation
    """
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    return np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)


def fig_to_base64(fig):
    """
    Convert matplotlib figure to base64 string
    
    Args:
        fig: Matplotlib figure
    
    Returns:
        base64_str: Base64 encoded string
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str


def create_side_by_side_comparison(images, titles, figsize=(16, 4)):
    """
    Create side-by-side comparison of multiple images
    
    Args:
        images: List of images
        titles: List of titles
        figsize: Figure size
    
    Returns:
        fig: Matplotlib figure
    """
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    
    if n == 1:
        axes = [axes]
    
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    return fig


def create_dataset_comparison_chart():
    """
    Create comparison chart showing model performance across datasets
    Based on paper results
    """
    datasets = ['CVC-Clinic', 'CVC-Colon', 'Kvasir-SEG', 'ETIS-LARIB']
    dice_scores = [0.96, 0.96, 0.95, 0.95]
    iou_scores = [0.91, 0.90, 0.89, 0.89]
    fps = [70, 68, 70, 68]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Dice and IoU comparison
    x = np.arange(len(datasets))
    width = 0.35
    
    axes[0].bar(x - width/2, dice_scores, width, label='Dice', color='#2ecc71', edgecolor='black')
    axes[0].bar(x + width/2, iou_scores, width, label='IoU', color='#3498db', edgecolor='black')
    
    axes[0].set_ylabel('Score', fontsize=12)
    axes[0].set_title('Performance Across Datasets', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(datasets, rotation=15, ha='right')
    axes[0].legend()
    axes[0].set_ylim(0.8, 1.0)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (d, iou) in enumerate(zip(dice_scores, iou_scores)):
        axes[0].text(i - width/2, d + 0.005, f'{d:.2f}', ha='center', fontsize=9)
        axes[0].text(i + width/2, iou + 0.005, f'{iou:.2f}', ha='center', fontsize=9)
    
    # FPS comparison
    axes[1].bar(datasets, fps, color='#e74c3c', edgecolor='black')
    axes[1].set_ylabel('Frames Per Second (FPS)', fontsize=12)
    axes[1].set_title('Inference Speed', fontsize=14, fontweight='bold')
    axes[1].set_xticklabels(datasets, rotation=15, ha='right')
    axes[1].set_ylim(0, 80)
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].axhline(y=30, color='green', linestyle='--', label='Real-time threshold (30 FPS)')
    axes[1].legend()
    
    # Add value labels
    for i, f in enumerate(fps):
        axes[1].text(i, f + 1, f'{f} FPS', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Test visualization
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    test_mask = np.random.randint(0, 2, (256, 256)) * 255
    test_prob = np.random.rand(256, 256)
    
    fig = plot_segmentation_results(test_image, test_mask, test_prob)
    plt.savefig('/tmp/test_viz.png')
    print("Test visualization saved")
