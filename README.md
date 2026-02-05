# Spinal ZF Polyp Segmentation Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

A **Streamlit web application** demonstrating the Spinal ZF-inspired deep learning framework for accurate and real-time colorectal polyp segmentation.

![Framework Overview](https://via.placeholder.com/800x400?text=Spinal+ZF+Framework)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Deployment Guide](#deployment-guide)
- [Model Architecture](#model-architecture)
- [Citation](#citation)

## 🔬 Overview

This demo application showcases the **Spinal ZF–inspired Deep Learning Framework** for colorectal polyp segmentation, as described in the research paper. The framework achieves:

| Metric | Value |
|--------|-------|
| **Dice Coefficient** | 0.96 |
| **IoU** | 0.90 |
| **Precision** | 0.97 |
| **Recall** | 0.88 |
| **Inference Speed** | 70 FPS |

### Key Innovation

The framework introduces a **hierarchical and progressive feature learning strategy** inspired by the biological processing paradigm of the human spinal cord:

1. **Hierarchical Feature Learning** - Multi-stage feature extraction
2. **Spinal Net Progressive Refinement** - Sequential feature processing
3. **Spinal ZF Feature Fusion** - Integration of complementary features
4. **ZF-Net-based Refinement** - Enhanced spatial coherence
5. **Decoder with Skip Connections** - Precise boundary delineation

## ✨ Features

- 🖼️ **Image Upload**: Support for JPG, PNG, and other formats
- 🎯 **Real-time Segmentation**: Polyp detection and boundary delineation
- 📊 **Visualization**: Overlay, heatmap, and probability display
- 🔍 **Explainability**: Grad-CAM visualization for model interpretation
- 📈 **Performance Metrics**: Dataset comparison and benchmark results
- 🎮 **Demo Mode**: Works without pretrained weights for testing
- ⚡ **Fast Inference**: Optimized for CPU and GPU

## 📁 Repository Structure

```
spinal-zf-demo/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .gitignore               # Git ignore rules
├── models/                   # Model architecture
│   ├── __init__.py
│   └── spinal_zf_model.py   # Spinal ZF model implementation
├── utils/                    # Utility functions
│   ├── __init__.py
│   ├── preprocessing.py     # Image preprocessing
│   ├── visualization.py     # Visualization tools
│   └── explainability.py    # Grad-CAM and explainability
├── data/                     # Data directory (optional)
│   └── __init__.py
└── checkpoints/              # Model checkpoints
    └── .gitkeep
```

## 🚀 Quick Start

### Local Development

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/spinal-zf-demo.git
   cd spinal-zf-demo
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser:** Navigate to `http://localhost:8501`

### Adding a Checkpoint (Optional)

To use a trained model instead of random weights:

1. Place your checkpoint file (`.pth` or `.pt`) in the `checkpoints/` directory
2. Restart the application
3. The app will automatically detect and load the checkpoint

## 📦 Deployment Guide

### Deploying to Streamlit Community Cloud

#### Step 1: Prepare Your Repository

1. **Ensure your repository is on GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/yourusername/spinal-zf-demo.git
   git push -u origin main
   ```

2. **Verify repository structure:**
   - `app.py` must be at the root
   - `requirements.txt` must be at the root (do not rename!)
   - All `__init__.py` files are in place

#### Step 2: Deploy on Streamlit Cloud

1. **Go to [Streamlit Cloud](https://streamlit.io/cloud)** and sign in with GitHub

2. **Click "New app"** and select your repository

3. **Configure deployment:**
   - **Repository:** `yourusername/spinal-zf-demo`
   - **Branch:** `main`
   - **Main file path:** `app.py`

4. **Click "Deploy"**

5. **Wait for deployment** (typically 2-5 minutes)

#### Step 3: Upload Checkpoint (Optional)

For large checkpoint files, you have two options:

**Option A: Git LFS (Recommended for >100MB)**
```bash
# Install Git LFS
git lfs install

# Track checkpoint files
git lfs track "checkpoints/*.pth"
git lfs track "checkpoints/*.pt"

# Add and commit
git add .gitattributes
git add checkpoints/your-model.pth
git commit -m "Add model checkpoint"
git push
```

**Option B: Manual Upload**
1. Deploy the app without the checkpoint (demo mode)
2. Use Streamlit Cloud's file upload feature
3. Or use cloud storage (AWS S3, Google Cloud Storage) and modify `app.py` to download

#### Step 4: Configure Secrets (Optional)

If using cloud storage for checkpoints:
1. Go to your app settings on Streamlit Cloud
2. Click "Secrets"
3. Add your storage credentials:
   ```toml
   [aws]
   access_key = "your-access-key"
   secret_key = "your-secret-key"
   ```

### Troubleshooting Deployment

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Ensure `__init__.py` exists in all subdirectories |
| `opencv-python` error | Use `opencv-python-headless` in requirements.txt |
| Checkpoint not found | Verify file is in `checkpoints/` directory |
| Memory error | Reduce batch size or use CPU instead of GPU |
| Slow loading | Enable `@st.cache_resource` for model loading |

### Important Deployment Notes

⚠️ **Critical for Stability:**

1. **Do NOT rename `requirements.txt`** - This is required by Streamlit Cloud
2. **Keep each app in its own repository** - Avoid conflicts between apps
3. **Use relative paths only** - No hardcoded local paths
4. **Include `__init__.py`** in all subdirectories
5. **Use `opencv-python-headless`** instead of `opencv-python`

## 🏗️ Model Architecture

```
Input Image (256×256×3)
    ↓
Initial Convolution
    ↓
Encoder Downsampling
    ↓
Hierarchical Feature Learning (Stage 1 & 2)
    ↓
Spinal Net Progressive Refinement
    ↓
Spinal ZF Feature Fusion
    ↓
ZF-Net-based Refinement
    ↓
Decoder with Skip Connections
    ↓
Segmentation Mask (256×256×1)
```

### Loss Function

The model uses a hybrid loss combining:
- **Binary Cross-Entropy (BCE)**: Pixel-level classification
- **Dice Loss**: Region-level spatial overlap

```python
Loss = BCE_Loss + Dice_Loss
```

## 📊 Datasets

The model was evaluated on four benchmark datasets:

| Dataset | Images | Characteristics |
|---------|--------|-----------------|
| CVC-Clinic DB | 612 | Standard benchmark |
| CVC-Colon DB | 300 | High variability |
| ETIS-LARIB | 196 | Small polyps, challenging |
| Kvasir-SEG | 1,000 | Real-world clinical data |

## 📄 Citation

If you use this framework in your research, please cite:

```bibtex
@article{spinalzf2024,
  title={A Spinal ZF–Inspired Deep Learning Framework for Accurate and Real-Time Colorectal Polyp Segmentation},
  author={[Author Names]},
  journal={[Journal Name]},
  year={2024},
  publisher={[Publisher]}
}
```

## ⚠️ Disclaimer

**For research and demonstration purposes only. Not for clinical use.**

This demo application is intended for research evaluation only. It should not be used for medical diagnosis or clinical decision-making without proper validation and regulatory approval.

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Contact the research team

## 🙏 Acknowledgments

This research was supported by:
- National Natural Science Foundation of China (82473228, 82472998)
- Fundamental Research Project of Key Scientific Research in Henan Province (23ZX007)
- Science and Technology Project of Henan Province (242102311082)
- Central Plains Science and Technology Innovation Leading Talents (224200510015)

## 📜 License

[Add your license information here]

---

<p align="center">
  <strong>Spinal ZF Segmentation Framework</strong> | Research Demo
</p>
