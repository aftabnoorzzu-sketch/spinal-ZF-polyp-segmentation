"""
Test script for Spinal ZF Model

Validates:
- Model can be instantiated
- Forward pass works
- Model can handle different input sizes
- Checkpoint loading (if available)
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.spinal_zf_model import create_model
from utils.preprocessing import preprocess_image
from utils.visualization import create_overlay


def test_model_creation():
    """Test model can be created"""
    print("Testing model creation...")
    try:
        model = create_model(in_channels=3, num_classes=1, base_channels=64)
        print(f"✅ Model created successfully")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        return model
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return None


def test_forward_pass(model):
    """Test forward pass with different input sizes"""
    print("\nTesting forward pass...")
    
    test_sizes = [
        (1, 3, 256, 256),  # Standard size
        (1, 3, 512, 512),  # Larger size
        (2, 3, 256, 256),  # Batch size 2
    ]
    
    model.eval()
    with torch.no_grad():
        for size in test_sizes:
            try:
                x = torch.randn(size)
                output = model(x)
                print(f"✅ Input {size} -> Output {output.shape}")
            except Exception as e:
                print(f"❌ Failed for size {size}: {e}")


def test_checkpoint_loading():
    """Test checkpoint loading if available"""
    print("\nTesting checkpoint loading...")
    
    checkpoint_dir = Path("./checkpoints")
    checkpoint_files = list(checkpoint_dir.glob("*.pth")) + list(checkpoint_dir.glob("*.pt"))
    
    if not checkpoint_files:
        print("ℹ️ No checkpoints found - this is OK for demo mode")
        return
    
    for ckpt_file in checkpoint_files:
        try:
            model = create_model(pretrained_path=str(ckpt_file))
            print(f"✅ Loaded checkpoint: {ckpt_file.name}")
        except Exception as e:
            print(f"❌ Failed to load {ckpt_file.name}: {e}")


def test_preprocessing():
    """Test image preprocessing"""
    print("\nTesting preprocessing...")
    
    try:
        # Create synthetic image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Preprocess
        tensor, original_size = preprocess_image(image, target_size=(256, 256))
        
        print(f"✅ Preprocessing successful")
        print(f"   Input: {image.shape} -> Tensor: {tensor.shape}")
        print(f"   Original size: {original_size}")
        
        return tensor, original_size
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        return None, None


def test_end_to_end():
    """Test complete pipeline"""
    print("\nTesting end-to-end pipeline...")
    
    try:
        # Create model
        model = create_model()
        model.eval()
        
        # Create synthetic image
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Preprocess
        tensor, original_size = preprocess_image(image)
        
        # Run inference
        with torch.no_grad():
            output = model(tensor)
        
        # Postprocess
        from utils.preprocessing import postprocess_mask
        binary_mask, prob_mask = postprocess_mask(output, original_size, threshold=0.5)
        
        # Create overlay
        overlay = create_overlay(image, binary_mask)
        
        print(f"✅ End-to-end pipeline successful")
        print(f"   Output mask shape: {binary_mask.shape}")
        print(f"   Polyp pixels: {np.sum(binary_mask > 0)}")
        
    except Exception as e:
        print(f"❌ End-to-end test failed: {e}")
        import traceback
        traceback.print_exc()


def test_explainability():
    """Test explainability functions"""
    print("\nTesting explainability...")
    
    try:
        from utils.explainability import generate_explainability_map
        
        model = create_model()
        input_tensor = torch.randn(1, 3, 256, 256)
        
        # Test Grad-CAM
        cam = generate_explainability_map(model, input_tensor, method='gradcam')
        print(f"✅ Grad-CAM generated: {cam.shape}")
        
        # Test saliency
        saliency = generate_explainability_map(model, input_tensor, method='saliency')
        print(f"✅ Saliency map generated: {saliency.shape}")
        
    except Exception as e:
        print(f"❌ Explainability test failed: {e}")


def main():
    """Run all tests"""
    print("=" * 60)
    print("Spinal ZF Model Test Suite")
    print("=" * 60)
    
    # Test model creation
    model = test_model_creation()
    if model is None:
        print("\n❌ Critical failure: Model cannot be created")
        return
    
    # Test forward pass
    test_forward_pass(model)
    
    # Test checkpoint loading
    test_checkpoint_loading()
    
    # Test preprocessing
    test_preprocessing()
    
    # Test end-to-end
    test_end_to_end()
    
    # Test explainability
    test_explainability()
    
    print("\n" + "=" * 60)
    print("Test Suite Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
