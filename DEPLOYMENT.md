# Streamlit Cloud Deployment Instructions

## Step-by-Step Guide for Deploying Spinal ZF Demo

### Prerequisites

- GitHub account
- Streamlit Cloud account (free tier available)
- Your model checkpoint file (optional, for trained model)

---

## Part 1: Repository Setup (5 minutes)

### 1.1 Create a New GitHub Repository

1. Go to [GitHub](https://github.com) and sign in
2. Click the **+** icon → **New repository**
3. Name it `spinal-zf-demo` (or your preferred name)
4. Choose **Public** or **Private**
5. Do NOT initialize with README (we have our own)
6. Click **Create repository**

### 1.2 Push Your Code to GitHub

Open terminal in your project directory:

```bash
# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit of Spinal ZF demo app"

# Rename branch to main
git branch -M main

# Add remote (replace with your repository URL)
git remote add origin https://github.com/YOUR_USERNAME/spinal-zf-demo.git

# Push to GitHub
git push -u origin main
```

---

## Part 2: Deploy on Streamlit Cloud (5 minutes)

### 2.1 Sign In to Streamlit Cloud

1. Go to [Streamlit Cloud](https://streamlit.io/cloud)
2. Click **Sign in with GitHub**
3. Authorize Streamlit to access your repositories

### 2.2 Create New App

1. Click **New app** button
2. Select your GitHub repository: `YOUR_USERNAME/spinal-zf-demo`
3. Select branch: `main`
4. Main file path: `app.py`
5. Click **Deploy**

### 2.3 Wait for Deployment

- Deployment typically takes 2-5 minutes
- You'll see build logs in real-time
- Once complete, your app will be live at: `https://your-app-name.streamlit.app`

---

## Part 3: Add Model Checkpoint (Optional, 10 minutes)

### Option A: Using Git LFS (For checkpoints < 2GB)

```bash
# Install Git LFS
brew install git-lfs          # macOS
# or
git lfs install              # Windows/Linux with git-lfs installed

# Initialize Git LFS
git lfs install

# Track checkpoint files
git lfs track "checkpoints/*.pth"
git lfs track "checkpoints/*.pt"

# Add .gitattributes
git add .gitattributes

# Add your checkpoint
cp /path/to/your/model.pth checkpoints/
git add checkpoints/model.pth

# Commit and push
git commit -m "Add model checkpoint via Git LFS"
git push origin main
```

### Option B: Direct Upload (For checkpoints < 100MB)

```bash
# Simply copy and push
cp /path/to/your/model.pth checkpoints/
git add checkpoints/model.pth
git commit -m "Add model checkpoint"
git push origin main
```

### Option C: Cloud Storage Integration (For large checkpoints)

Modify `app.py` to download checkpoint from cloud storage:

```python
import requests
import os

def download_checkpoint():
    url = "https://your-storage.com/model.pth"
    checkpoint_path = "checkpoints/model.pth"
    
    if not os.path.exists(checkpoint_path):
        response = requests.get(url)
        with open(checkpoint_path, 'wb') as f:
            f.write(response.content)
    
    return checkpoint_path
```

---

## Part 4: Configure App Settings (Optional)

### 4.1 Set Python Version

Create `runtime.txt` in the root directory:

```
python-3.9
```

### 4.2 Add Streamlit Configuration

Create `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#F63366"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 200
enableCORS = false
```

### 4.3 Add Secrets (For API keys, etc.)

1. Go to your app on Streamlit Cloud
2. Click **⋮** → **Settings**
3. Go to **Secrets** tab
4. Add your secrets in TOML format:

```toml
[general]
model_url = "https://your-storage.com/model.pth"
```

Access in code:
```python
import streamlit as st
model_url = st.secrets["general"]["model_url"]
```

---

## Part 5: Verify Deployment

### 5.1 Check App is Running

1. Visit your app URL: `https://your-app-name.streamlit.app`
2. Verify all tabs work:
   - [ ] Segmentation Demo
   - [ ] Explainability
   - [ ] Performance
   - [ ] About
   - [ ] Demo Mode

### 5.2 Test Image Upload

1. Go to **Segmentation Demo** tab
2. Upload a test image (JPG/PNG)
3. Verify segmentation runs without errors

### 5.3 Check Model Loading

If you added a checkpoint:
- You should see "✅ Model loaded successfully with pretrained weights"

If running in demo mode:
- You should see "⚠️ Demo Mode" warning

---

## Troubleshooting

### Build Failed

| Error | Solution |
|-------|----------|
| `ModuleNotFoundError` | Check `requirements.txt` has all dependencies |
| `No module named 'models'` | Ensure `models/__init__.py` exists |
| `opencv-python` error | Use `opencv-python-headless` |
| `torch` installation fails | Pin version: `torch==2.0.0` |

### Runtime Errors

| Error | Solution |
|-------|----------|
| Checkpoint not found | Verify file path is relative (`./checkpoints/`) |
| Out of memory | Use CPU instead of GPU; reduce image size |
| Slow inference | Enable caching with `@st.cache_resource` |

### App Crashes

1. Check logs in Streamlit Cloud dashboard
2. Test locally first: `streamlit run app.py`
3. Verify all file paths are correct
4. Check for missing `__init__.py` files

---

## Updating Your App

### Push Updates

```bash
# Make changes to code
git add .
git commit -m "Update: description of changes"
git push origin main
```

Streamlit Cloud will automatically redeploy on push.

### Manual Redeploy

1. Go to Streamlit Cloud dashboard
2. Click **⋮** → **Reboot**

---

## Best Practices

1. **Test locally before deploying**
   ```bash
   streamlit run app.py
   ```

2. **Use pinned versions for critical dependencies**
   ```
   torch==2.0.0
   streamlit==1.28.0
   ```

3. **Keep checkpoint files out of git if >100MB**
   - Use Git LFS or cloud storage

4. **Use relative paths only**
   ```python
   # Good
   "./checkpoints/model.pth"
   
   # Bad
   "/home/user/project/checkpoints/model.pth"
   ```

5. **Cache model loading**
   ```python
   @st.cache_resource
   def load_model():
       ...
   ```

6. **Handle errors gracefully**
   ```python
   try:
       result = model.predict(image)
   except Exception as e:
       st.error(f"Error: {e}")
   ```

---

## Getting Help

- [Streamlit Documentation](https://docs.streamlit.io)
- [Streamlit Community Forum](https://discuss.streamlit.io)
- [GitHub Issues](https://github.com/streamlit/streamlit/issues)

---

## Quick Reference Card

```bash
# Local testing
streamlit run app.py

# Deploy to GitHub
git add .
git commit -m "Update"
git push origin main

# Add checkpoint with Git LFS
git lfs track "checkpoints/*.pth"
git add checkpoints/model.pth
git commit -m "Add checkpoint"
git push
```

**Streamlit Cloud URL:** `https://share.streamlit.io`

**Your App URL:** `https://your-app-name.streamlit.app`
