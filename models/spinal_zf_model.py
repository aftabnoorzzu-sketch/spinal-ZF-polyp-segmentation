"""
Spinal ZF-inspired Deep Learning Framework for Colorectal Polyp Segmentation

Based on the paper:
"A Spinal ZF–Inspired Deep Learning Framework for Accurate and Real-Time Colorectal Polyp Segmentation"

Architecture Components:
- Hierarchical Feature Learning (Stage 1 & 2)
- Spinal Net Progressive Refinement
- Spinal ZF Feature Fusion
- ZF-Net-based Feature Refinement
- Decoder with Skip Connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvBlock(nn.Module):
    """Basic convolutional block with Conv -> BatchNorm -> ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class HierarchicalFeatureBlock(nn.Module):
    """
    Hierarchical Feature Learning Block
    Extracts multi-level contextual representations using 3x3 convolutions
    """
    def __init__(self, in_channels, out_channels):
        super(HierarchicalFeatureBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class SpinalNetBlock(nn.Module):
    """
    Spinal Net Progressive Refinement Block
    Sequentially processes feature segments for progressive refinement
    Inspired by biological spinal cord information processing
    """
    def __init__(self, in_features, num_segments=4, hidden_dim=64):
        super(SpinalNetBlock, self).__init__()
        self.num_segments = num_segments
        self.segment_size = in_features // num_segments
        
        # Transformation layers for each segment
        self.transform_layers = nn.ModuleList([
            nn.Linear(self.segment_size + (hidden_dim if i > 0 else 0), hidden_dim)
            for i in range(num_segments)
        ])
        
        self.activation = nn.ReLU(inplace=True)
        self.output_proj = nn.Linear(hidden_dim * num_segments, in_features)
    
    def forward(self, x):
        # x: [B, C, H, W] -> flatten to [B, C*H*W]
        batch_size = x.size(0)
        original_shape = x.shape
        x_flat = x.view(batch_size, -1)
        
        # Split into segments
        segments = torch.split(x_flat, self.segment_size, dim=1)
        
        # Progressive refinement
        refined_segments = []
        prev_output = None
        
        for i, segment in enumerate(segments[:self.num_segments]):
            if prev_output is not None:
                # Concatenate current segment with previous output
                combined = torch.cat([segment, prev_output], dim=1)
            else:
                combined = segment
            
            # Transform
            refined = self.transform_layers[i](combined)
            refined = self.activation(refined)
            prev_output = refined
            refined_segments.append(refined)
        
        # Concatenate all refined segments
        spinal_output = torch.cat(refined_segments, dim=1)
        
        # Project back to original dimension
        output = self.output_proj(spinal_output)
        
        # Reshape back to spatial format
        output = output.view(original_shape)
        
        return output


class SpinalNetRefinement(nn.Module):
    """Stacked Spinal Net blocks for progressive refinement"""
    def __init__(self, channels, spatial_size=64, num_blocks=2):
        super(SpinalNetRefinement, self).__init__()
        in_features = channels * spatial_size * spatial_size
        self.blocks = nn.ModuleList([
            SpinalNetBlock(in_features, num_segments=4, hidden_dim=64)
            for _ in range(num_blocks)
        ])
        self.channels = channels
        self.spatial_size = spatial_size
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class SpinalZFFusion(nn.Module):
    """
    Spinal ZF Feature Fusion Layer
    Integrates hierarchical convolutional features with Spinal Net representations
    """
    def __init__(self, in_channels):
        super(SpinalZFFusion, self).__init__()
        # 1x1 convolution for channel reduction and feature integration
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, hierarchical_features, spinal_features):
        # Concatenate features along channel dimension
        fused = torch.cat([hierarchical_features, spinal_features], dim=1)
        # Apply 1x1 convolution
        fused = self.fusion_conv(fused)
        # Residual connection with spinal features
        output = fused + spinal_features
        return output


class ZFNetRefinement(nn.Module):
    """
    ZF-Net-based Feature Refinement Module
    Applies convolutional transformations to enhance spatial coherence
    """
    def __init__(self, in_channels):
        super(ZFNetRefinement, self).__init__()
        self.refinement = nn.Sequential(
            ConvBlock(in_channels, in_channels),
            ConvBlock(in_channels, in_channels // 2),
            ConvBlock(in_channels // 2, in_channels)
        )
    
    def forward(self, x):
        return self.refinement(x)


class DecoderBlock(nn.Module):
    """
    Decoder block with upsampling and skip connections
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, skip):
        x = self.upsample(x)
        # Handle size mismatch
        if x.size() != skip.size():
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class SpinalZFSegmentationModel(nn.Module):
    """
    Complete Spinal ZF-inspired Segmentation Model
    
    Architecture Pipeline:
    1. Initial Convolutional Feature Extraction
    2. Hierarchical Feature Learning (Stage 1 & 2)
    3. Spinal Net Progressive Refinement
    4. Spinal ZF Feature Fusion
    5. ZF-Net-based Feature Refinement
    6. Decoder with Skip Connections
    """
    def __init__(self, in_channels=3, num_classes=1, base_channels=64):
        super(SpinalZFSegmentationModel, self).__init__()
        
        # Initial feature extraction
        self.initial_conv = ConvBlock(in_channels, base_channels)
        
        # Encoder downsampling
        self.enc_pool1 = nn.MaxPool2d(2)
        self.enc_pool2 = nn.MaxPool2d(2)
        
        # Hierarchical Feature Learning
        self.hierarchical_stage1 = HierarchicalFeatureBlock(base_channels, base_channels * 2)
        self.hierarchical_stage2 = HierarchicalFeatureBlock(base_channels * 2, base_channels * 4)
        
        # Spinal Net Progressive Refinement
        # After two poolings: 256 -> 64, so spatial size is 64x64
        self.spinal_refinement = SpinalNetRefinement(
            channels=base_channels * 4,
            spatial_size=64,
            num_blocks=2
        )
        
        # Spinal ZF Feature Fusion
        self.spinal_zf_fusion = SpinalZFFusion(base_channels * 4)
        
        # ZF-Net-based Feature Refinement
        self.zf_refinement = ZFNetRefinement(base_channels * 4)
        
        # Decoder with skip connections
        self.dec_block1 = DecoderBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.dec_block2 = DecoderBlock(base_channels * 2, base_channels, base_channels)
        
        # Final segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, num_classes, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Store input size for final interpolation
        input_size = x.shape[2:]
        
        # Initial convolution
        x0 = self.initial_conv(x)
        
        # Encoder path with downsampling
        x1 = self.enc_pool1(x0)
        x1 = self.hierarchical_stage1(x1)
        
        x2 = self.enc_pool2(x1)
        x2 = self.hierarchical_stage2(x2)
        
        # Spinal Net progressive refinement
        x2_refined = self.spinal_refinement(x2)
        
        # Spinal ZF feature fusion
        x2_fused = self.spinal_zf_fusion(x2, x2_refined)
        
        # ZF-Net-based refinement
        x2_refined_final = self.zf_refinement(x2_fused)
        
        # Decoder path with skip connections
        d1 = self.dec_block1(x2_refined_final, x1)
        d2 = self.dec_block2(d1, x0)
        
        # Final segmentation output
        output = self.seg_head(d2)
        
        # Interpolate to input size
        output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=True)
        
        return output
    
    def predict(self, x):
        """Inference mode prediction"""
        self.eval()
        with torch.no_grad():
            return self.forward(x)


def create_model(in_channels=3, num_classes=1, base_channels=64, pretrained_path=None):
    """
    Factory function to create Spinal ZF model
    
    Args:
        in_channels: Number of input channels (default: 3 for RGB)
        num_classes: Number of output classes (default: 1 for binary segmentation)
        base_channels: Base number of channels (default: 64)
        pretrained_path: Path to pretrained checkpoint (optional)
    
    Returns:
        model: SpinalZFSegmentationModel instance
    """
    model = SpinalZFSegmentationModel(in_channels, num_classes, base_channels)
    
    if pretrained_path is not None:
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"Successfully loaded pretrained model from {pretrained_path}")
        except Exception as e:
            print(f"Warning: Could not load pretrained model: {e}")
            print("Initializing with random weights (demo mode)")
    
    return model


# For compatibility with different PyTorch versions
def load_checkpoint_compat(model, checkpoint_path):
    """Load checkpoint with compatibility for different save formats"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
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
        return True
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return False


if __name__ == "__main__":
    # Test model
    model = create_model()
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
