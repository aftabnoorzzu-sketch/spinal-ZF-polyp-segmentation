"""
Spinal ZF-inspired Segmentation Demo App
========================================

A Streamlit web application for demonstrating the Spinal ZF-inspired deep learning
framework for accurate and real-time colorectal polyp segmentation.

Paper Reference:
"A Spinal ZF–Inspired Deep Learning Framework for Accurate and Real-Time 
Colorectal Polyp Segmentation"

Features:
- Image upload and preprocessing
- Real-time polyp segmentation
- Segmentation mask visualization with overlays
- Explainability (Grad-CAM) visualization
- Performance metrics display
- Demo mode (works without pretrained weights)

Author: Research Team
For research and demonstration purposes only. Not for clinical use.
"""

import os
import sys
import time
import warnings
from pathlib import Path

import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Page configuration must be first Streamlit command
st.set_page_config(
    page_title="Spinal ZF Polyp Segmentation",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import project modules
from spinal_zf_model import create_model
from utils.preprocessing import preprocess_image, postprocess_mask, compute_mask_statistics
from utils.visualization import (
    create_overlay, 
    create_probability_heatmap,
    create_metrics_bar_chart,
    create_dataset_comparison_chart,
    fig_to_array
)
from utils.explainability import generate_explainability_map, create_explainability_visualization

# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

CHECKPOINT_DIR = Path("./checkpoints")
SUPPORTED_FORMATS = ['jpg', 'jpeg', 'png', 'bmp', 'tiff']
TARGET_SIZE = (256, 256)

# Paper-reported metrics for reference
PAPER_METRICS = {
    'CVC-Clinic DB': {'dice': 0.96, 'iou': 0.91, 'precision': 0.97, 'recall': 0.88, 'fps': 70},
    'CVC-Colon DB': {'dice': 0.96, 'iou': 0.90, 'precision': 0.97, 'recall': 0.88, 'fps': 68},
    'Kvasir-SEG': {'dice': 0.95, 'iou': 0.89, 'precision': 0.96, 'recall': 0.87, 'fps': 70},
    'ETIS-LARIB': {'dice': 0.95, 'iou': 0.89, 'precision': 0.97, 'recall': 0.88, 'fps': 68}
}

# =============================================================================
# CACHED FUNCTIONS
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_model(checkpoint_path=None, device='cpu'):
    """
    Load the Spinal ZF segmentation model with caching for efficiency.
    
    Args:
        checkpoint_path: Path to model checkpoint (optional)
        device: Device to load model on ('cpu' or 'cuda')
    
    Returns:
        model: Loaded model
        is_demo_mode: Whether running in demo mode (no checkpoint)
    """
    model = create_model(in_channels=3, num_classes=1, base_channels=64)
    
    is_demo_mode = True
    
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Remove 'module.' prefix if present (from DataParallel)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace('module.', '') if k.startswith('module.') else k
                new_state_dict[name] = v
            
            model.load_state_dict(new_state_dict, strict=False)
            is_demo_mode = False
            
        except Exception as e:
            st.warning(f"⚠️ Could not load checkpoint: {e}")
            st.info("Running in demo mode with random weights.")
    
    model.to(device)
    model.eval()
    
    return model, is_demo_mode


@st.cache_data(show_spinner=False)
def process_image_cached(image_bytes):
    """Cache image processing to avoid reprocessing"""
    image = Image.open(image_bytes)
    return image


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar():
    """Render the application sidebar with controls and information"""
    
    st.sidebar.title("🔬 Spinal ZF Segmentation")
    st.sidebar.markdown("---")
    
    # Model Status Section
    st.sidebar.subheader("📊 Model Status")
    
    # Check for checkpoint
    checkpoint_files = list(CHECKPOINT_DIR.glob("*.pth")) + list(CHECKPOINT_DIR.glob("*.pt"))
    
    if checkpoint_files:
        st.sidebar.success(f"✅ Found {len(checkpoint_files)} checkpoint(s)")
        selected_checkpoint = st.sidebar.selectbox(
            "Select checkpoint:",
            [f.name for f in checkpoint_files],
            key="checkpoint_select"
        )
        checkpoint_path = CHECKPOINT_DIR / selected_checkpoint
    else:
        st.sidebar.warning("⚠️ No checkpoint found")
        st.sidebar.info("Running in **demo mode** with random weights")
        checkpoint_path = None
    
    st.sidebar.markdown("---")
    
    # Device Selection
    st.sidebar.subheader("⚙️ Settings")
    
    device_options = ['cpu']
    if torch.cuda.is_available():
        device_options.append('cuda')
    
    device = st.sidebar.selectbox(
        "Device:",
        device_options,
        index=0,
        help="CPU is recommended for Streamlit Cloud deployment"
    )
    
    # Inference Settings
    threshold = st.sidebar.slider(
        "Segmentation Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Threshold for converting probability map to binary mask"
    )
    
    st.sidebar.markdown("---")
    
    # Paper Information
    st.sidebar.subheader("📄 Paper Information")
    st.sidebar.markdown("""
    **Title:**  
    A Spinal ZF–Inspired Deep Learning Framework for Accurate and Real-Time Colorectal Polyp Segmentation
    
    **Key Metrics:**
    - Dice: 0.96
    - IoU: 0.90
    - Precision: 0.97
    - Recall: 0.88
    - FPS: 70
    
    **Datasets:**
    - CVC-Clinic DB
    - CVC-Colon DB
    - Kvasir-SEG
    - ETIS-LARIB
    """)
    
    st.sidebar.markdown("---")
    
    # Disclaimer
    st.sidebar.warning("""
    **⚠️ Disclaimer**
    
    For research and demonstration purposes only.  
    **Not for clinical use.**
    """)
    
    return checkpoint_path, device, threshold


# =============================================================================
# MAIN TABS
# =============================================================================

def render_segmentation_tab(model, device, threshold, is_demo_mode):
    """Render the segmentation demo tab"""
    
    st.header("🖼️ Polyp Segmentation Demo")
    
    if is_demo_mode:
        st.info("""
        **Demo Mode Active:** The model is running with randomly initialized weights.  
        For accurate predictions, please upload a pretrained checkpoint to `./checkpoints/`.
        """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a colonoscopy image (JPG, PNG)",
        type=SUPPORTED_FORMATS,
        help="Upload a colonoscopy image for polyp segmentation"
    )
    
    if uploaded_file is not None:
        try:
            # Load and display original image
            image = Image.open(uploaded_file)
            
            # Validate image
            if image.mode not in ['RGB', 'L', 'RGBA']:
                st.error("❌ Unsupported image mode. Please upload an RGB image.")
                return
            
            # Create columns for display
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📷 Original Image")
                st.image(image, use_container_width=True)
                
                # Image info
                st.markdown(f"""
                **Image Info:**
                - Size: {image.size[0]} × {image.size[1]} pixels
                - Mode: {image.mode}
                - Format: {uploaded_file.type}
                """)
            
            # Process image
            with st.spinner("🔍 Running segmentation..."):
                start_time = time.time()
                
                # Preprocess
                input_tensor, original_size = preprocess_image(image, TARGET_SIZE)
                input_tensor = input_tensor.to(device)
                
                # Run inference
                with torch.no_grad():
                    output = model(input_tensor)
                
                # Postprocess
                binary_mask, prob_mask = postprocess_mask(output, original_size, threshold)
                
                inference_time = time.time() - start_time
                fps = 1.0 / inference_time if inference_time > 0 else 0
                
                # Convert for visualization
                image_np = np.array(image)
                if image_np.shape[2] == 4:
                    image_np = image_np[:, :, :3]
                
                # Create visualizations
                overlay = create_overlay(image_np, binary_mask, alpha=0.5, color=(255, 0, 0))
                heatmap = create_probability_heatmap(image_np, prob_mask, alpha=0.6)
                
                # Compute statistics
                stats = compute_mask_statistics(binary_mask)
            
            with col2:
                st.subheader("🎯 Segmentation Result")
                st.image(overlay, use_container_width=True)
                
                # Inference stats
                st.markdown(f"""
                **Inference Stats:**
                - Time: {inference_time*1000:.1f} ms
                - FPS: {fps:.1f}
                - Polyp Area: {stats['polyp_percentage']:.2f}%
                - Detected Regions: {stats['num_detected_regions']}
                """)
            
            # Detailed results
            st.markdown("---")
            st.subheader("📊 Detailed Results")
            
            res_col1, res_col2, res_col3 = st.columns(3)
            
            with res_col1:
                st.markdown("**Binary Mask**")
                st.image(binary_mask, use_container_width=True, clamp=True)
            
            with res_col2:
                st.markdown("**Probability Heatmap**")
                st.image(heatmap, use_container_width=True)
            
            with res_col3:
                st.markdown("**Polyp Probability**")
                # Create probability histogram
                fig, ax = plt.subplots(figsize=(4, 3))
                prob_flat = prob_mask.flatten()
                ax.hist(prob_flat, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
                ax.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold ({threshold})')
                ax.set_xlabel('Probability')
                ax.set_ylabel('Pixel Count')
                ax.legend()
                st.pyplot(fig)
            
            # Store results in session state for other tabs
            st.session_state['last_image'] = image_np
            st.session_state['last_mask'] = binary_mask
            st.session_state['last_prob'] = prob_mask
            st.session_state['last_tensor'] = input_tensor
            
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
            st.info("Please try uploading a different image.")


def render_explainability_tab(model, device):
    """Render the explainability tab with Grad-CAM visualization"""
    
    st.header("🔍 Explainability Analysis")
    
    st.markdown("""
    This section provides **Grad-CAM visualizations** to help understand which regions 
    of the input image the model focuses on when making segmentation decisions.
    
    *Note: Run segmentation on an image first to enable explainability analysis.*
    """)
    
    if 'last_image' not in st.session_state:
        st.info("👆 Please run segmentation on an image in the **Segmentation Demo** tab first.")
        return
    
    image = st.session_state['last_image']
    mask = st.session_state['last_mask']
    prob_mask = st.session_state['last_prob']
    input_tensor = st.session_state['last_tensor']
    
    if st.button("🔍 Generate Explainability Map", type="primary"):
        with st.spinner("Generating Grad-CAM visualization..."):
            try:
                # Generate explainability map
                explain_map = generate_explainability_map(
                    model, 
                    input_tensor.to(device),
                    method='gradcam'
                )
                
                # Create visualization
                fig = create_explainability_visualization(
                    image, 
                    mask, 
                    explain_map,
                    title="Grad-CAM: Regions Important for Polyp Detection"
                )
                
                st.pyplot(fig)
                
                st.markdown("""
                **Interpretation:**
                - **Red/Yellow regions**: Areas the model focuses on for polyp detection
                - **Dark regions**: Areas with less influence on the model's decision
                - The overlay shows how the model's attention aligns with the predicted mask
                """)
                
            except Exception as e:
                st.error(f"Error generating explainability map: {e}")
                st.info("Explainability requires a valid model architecture. Ensure the model is properly loaded.")


def render_performance_tab():
    """Render the performance metrics tab"""
    
    st.header("📈 Performance Metrics")
    
    st.markdown("""
    This section presents the **quantitative performance** of the Spinal ZF framework 
    across four benchmark datasets, as reported in the paper.
    """)
    
    # Dataset comparison
    st.subheader("Performance Across Datasets")
    fig = create_dataset_comparison_chart()
    st.pyplot(fig)
    
    # Detailed metrics table
    st.subheader("Detailed Metrics")
    
    metrics_data = []
    for dataset, metrics in PAPER_METRICS.items():
        metrics_data.append({
            'Dataset': dataset,
            'Dice': f"{metrics['dice']:.2f}",
            'IoU': f"{metrics['iou']:.2f}",
            'Precision': f"{metrics['precision']:.2f}",
            'Recall': f"{metrics['recall']:.2f}",
            'FPS': f"{metrics['fps']}"
        })
    
    st.table(metrics_data)
    
    # Key highlights
    st.subheader("🌟 Key Highlights")
    
    highlight_col1, highlight_col2, highlight_col3 = st.columns(3)
    
    with highlight_col1:
        st.metric("Mean Dice", "0.96", "State-of-the-art")
    
    with highlight_col2:
        st.metric("Mean IoU", "0.90", "+5% vs baselines")
    
    with highlight_col3:
        st.metric("Inference Speed", "70 FPS", "Real-time capable")
    
    # Comparison with other methods
    st.subheader("Comparison with State-of-the-Art Methods")
    
    comparison_data = {
        'Method': ['U-Net', 'Attention U-Net', 'Pra-Net', 'Polyp-PVT', 'Spinal ZF (Ours)'],
        'CVC-Clinic Dice': [0.91, 0.90, 0.89, 0.93, 0.96],
        'Kvasir-SEG Dice': [0.54, 0.53, 0.89, 0.91, 0.95],
        'FPS': [14, 16, 50, 15, 70]
    }
    
    st.dataframe(comparison_data, use_container_width=True)


def render_about_tab():
    """Render the about tab with paper summary"""
    
    st.header("📄 About Spinal ZF Framework")
    
    st.markdown("""
    ## Overview
    
    The **Spinal ZF–inspired Deep Learning Framework** is a novel architecture designed 
    for accurate and real-time colorectal polyp segmentation in colonoscopy images.
    
    ## Key Innovation
    
    The framework introduces a **hierarchical and progressive feature learning strategy** 
    inspired by the biological processing paradigm of the human spinal cord, where information 
    is propagated and refined stage by stage.
    
    ## Architecture Components
    
    ### 1. Hierarchical Feature Learning
    - Multi-stage convolutional feature extraction
    - Progressive abstraction from low-level edges to high-level semantics
    - Captures both fine-grained details and global context
    
    ### 2. Spinal Net Progressive Refinement
    - Sequential feature processing inspired by spinal cord information flow
    - Features are split into segments and progressively refined
    - Improves robustness to variations in polyp size, shape, and texture
    
    ### 3. Spinal ZF Feature Fusion
    - Integrates hierarchical convolutional features with Spinal Net representations
    - Uses 1×1 convolutions and residual connections
    - Balances spatial detail with contextual information
    
    ### 4. ZF-Net-based Refinement
    - Enhances fused features through structured convolutional transformations
    - Improves spatial coherence and suppresses noise
    - Generates precise boundary delineation
    
    ### 5. Decoder with Skip Connections
    - Progressive upsampling with skip connections
    - Recovers fine spatial details
    - Generates accurate segmentation masks
    
    ## Training Configuration
    
    | Parameter | Value |
    |-----------|-------|
    | Optimizer | Adam |
    | Learning Rate | 0.0001 |
    | Batch Size | 16 |
    | Epochs | 50 |
    | Input Size | 256 × 256 |
    | Loss Function | BCE + Dice |
    
    ## Datasets
    
    The model was evaluated on four publicly available benchmark datasets:
    
    1. **CVC-Clinic DB**: 612 images from patient video sequences
    2. **CVC-Colon DB**: 300 images with high variability
    3. **ETIS-LARIB Polyp DB**: 196 challenging images with small polyps
    4. **Kvasir-SEG**: 1,000 images reflecting real-world clinical variability
    
    ## Citation
    
    If you use this framework in your research, please cite:
    
    ```
    [Paper citation will be added upon publication]
    ```
    
    ## Acknowledgments
    
    This research was supported by:
    - National Natural Science Foundation of China (82473228, 82472998)
    - Fundamental Research Project of Key Scientific Research in Henan Province (23ZX007)
    - Science and Technology Project of Henan Province (242102311082)
    - Central Plains Science and Technology Innovation Leading Talents (224200510015)
    """)


def render_demo_mode_tab(model, device):
    """Render the demo mode tab with sample visualizations"""
    
    st.header("🎮 Demo Mode")
    
    st.markdown("""
    This tab demonstrates the segmentation pipeline using synthetic data.  
    Useful for testing the app without uploading real medical images.
    """)
    
    if st.button("🎲 Generate Synthetic Test Case", type="primary"):
        with st.spinner("Generating synthetic colonoscopy image and running segmentation..."):
            # Generate synthetic image (simulating colonoscopy with polyp-like structure)
            np.random.seed(42)
            
            # Create base tissue background
            h, w = 256, 256
            synthetic = np.zeros((h, w, 3), dtype=np.uint8)
            
            # Pinkish tissue background
            synthetic[:, :, 0] = 180 + np.random.randint(-20, 20, (h, w))
            synthetic[:, :, 1] = 120 + np.random.randint(-15, 15, (h, w))
            synthetic[:, :, 2] = 120 + np.random.randint(-15, 15, (h, w))
            
            # Add polyp-like structure (darker red region)
            center_y, center_x = h // 2, w // 2
            y, x = np.ogrid[:h, :w]
            
            # Create elliptical polyp
            polyp_mask = ((x - center_x) ** 2 / 2500 + (y - center_y) ** 2 / 1600) <= 1
            synthetic[polyp_mask] = [140, 60, 60]  # Darker red for polyp
            
            # Add noise and texture
            noise = np.random.randint(-10, 10, (h, w, 3))
            synthetic = np.clip(synthetic + noise, 0, 255).astype(np.uint8)
            
            # Add specular highlights
            highlight_mask = np.random.random((h, w)) > 0.98
            synthetic[highlight_mask] = [255, 255, 255]
            
            # Run segmentation
            input_tensor, _ = preprocess_image(synthetic, TARGET_SIZE)
            input_tensor = input_tensor.to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
            
            binary_mask, prob_mask = postprocess_mask(output, (w, h), 0.5)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Synthetic Image")
                st.image(synthetic, use_container_width=True)
                st.markdown("""
                **Synthetic Features:**
                - Pinkish mucosal tissue background
                - Central elliptical polyp structure
                - Added noise and texture
                - Simulated specular highlights
                """)
            
            with col2:
                st.subheader("Segmentation Result")
                overlay = create_overlay(synthetic, binary_mask, alpha=0.5, color=(0, 255, 0))
                st.image(overlay, use_container_width=True)
                
                stats = compute_mask_statistics(binary_mask)
                st.markdown(f"""
                **Detected:**
                - Polyp Area: {stats['polyp_percentage']:.2f}%
                - Regions: {stats['num_detected_regions']}
                - Pixels: {stats['polyp_pixels']:,}
                """)
            
            st.info("""
            **Note:** In demo mode with random weights, the segmentation may not accurately 
            detect the synthetic polyp. For accurate results, use a trained checkpoint.
            """)


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point"""
    
    # Title and introduction
    st.title("🔬 Spinal ZF Polyp Segmentation")
    st.markdown("""
    **A Deep Learning Framework for Accurate and Real-Time Colorectal Polyp Segmentation**
    
    This demo showcases the Spinal ZF-inspired framework for automatic polyp detection 
    and segmentation in colonoscopy images.
    """)
    
    # Render sidebar and get settings
    checkpoint_path, device, threshold = render_sidebar()
    
    # Load model
    with st.spinner("Loading model..."):
        model, is_demo_mode = load_model(checkpoint_path, device)
    
    # Display mode banner
    if is_demo_mode:
        st.warning("""
        ⚠️ **Demo Mode**: No checkpoint found. The model is using randomly initialized weights.  
        Predictions will not be accurate. For proper inference, upload a checkpoint to `./checkpoints/`.
        """)
    else:
        st.success("✅ Model loaded successfully with pretrained weights.")
    
    # Create tabs
    tab_seg, tab_exp, tab_perf, tab_about, tab_demo = st.tabs([
        "🖼️ Segmentation Demo",
        "🔍 Explainability",
        "📈 Performance",
        "📄 About",
        "🎮 Demo Mode"
    ])
    
    with tab_seg:
        render_segmentation_tab(model, device, threshold, is_demo_mode)
    
    with tab_exp:
        render_explainability_tab(model, device)
    
    with tab_perf:
        render_performance_tab()
    
    with tab_about:
        render_about_tab()
    
    with tab_demo:
        render_demo_mode_tab(model, device)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p><strong>Spinal ZF Segmentation Framework</strong> | Research Demo</p>
        <p style="font-size: 0.8em;">
            For research and demonstration purposes only. Not for clinical use.<br>
            Data Availability: 
            <a href="https://polyp.grand-challenge.org/CVCClinicDB/">CVC-Clinic</a> | 
            <a href="https://datasets.simula.no/kvasir-seg">Kvasir-SEG</a>
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
