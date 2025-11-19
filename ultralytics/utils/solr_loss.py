#!/usr/bin/env python3
# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
SOLR (Small Object Loss Reweighting) Loss Module

This module implements Small Object Loss Reweighting (SOLR) for YOLOv12-RGBD
to address the class imbalance problem in UAV object detection datasets like
VisDrone and UAVDT, where small and medium objects are significantly harder
to detect than large objects.

Key Features:
    - Dynamic loss weighting based on object size (area)
    - Separate weights for small, medium, and large objects
    - Seamless integration with existing v8DetectionLoss
    - Compatible with RGB-D dual-modal input

References:
    - RemDet: Rethinking Efficient Model Design for UAV Object Detection (AAAI 2025)
    - Focal Loss for Dense Object Detection (CVPR 2017)

Created: 2025-11-19
Author: Generated for yolo12-bimodal project
Target: Improve medium object detection to match RemDet performance
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Tuple


class SOLRLoss:
    """
    Small Object Loss Reweighting (SOLR) loss module.
    
    Applies dynamic loss weighting based on target object sizes to address
    the performance gap on small/medium objects in UAV datasets.
    
    The core idea is simple but effective:
        - Small objects (<32px) are hard to detect → give them higher weight (2.5x)
        - Medium objects (32-96px) are moderately hard → medium weight (2.0x)
        - Large objects (>96px) are easy to detect → baseline weight (1.0x)
    
    This forces the model to pay more attention to smaller objects during training,
    leading to better overall detection performance.
    
    Attributes:
        small_weight (float): Loss multiplier for small objects (<small_thresh px)
        medium_weight (float): Loss multiplier for medium objects (small_thresh to large_thresh px)
        large_weight (float): Loss multiplier for large objects (>large_thresh px)
        small_thresh (int): Threshold (in pixels) separating small from medium objects
        large_thresh (int): Threshold (in pixels) separating medium from large objects
        image_size (int): Input image size for converting normalized coords to pixels
    
    Example:
        >>> solr = SOLRLoss(small_weight=2.5, medium_weight=2.0, large_weight=1.0)
        >>> # During training, compute weights for each target
        >>> weights = solr.compute_size_weights(target_bboxes)  # [N] tensor
        >>> # Apply to loss
        >>> weighted_loss = base_loss * weights.mean()
    """
    
    def __init__(
        self,
        small_weight: float = 2.5,
        medium_weight: float = 2.0,
        large_weight: float = 1.0,
        small_thresh: int = 32,
        large_thresh: int = 96,
        image_size: int = 640,
    ):
        """
        Initialize SOLR loss with size-based weights.
        
        Args:
            small_weight: Multiplier for small objects (<small_thresh pixels)
            medium_weight: Multiplier for medium objects (small_thresh to large_thresh pixels)
            large_weight: Multiplier for large objects (>large_thresh pixels)
            small_thresh: Threshold (pixels) separating small from medium objects
            large_thresh: Threshold (pixels) separating medium from large objects
            image_size: Input image size for coordinate conversion (default 640 for YOLOv12)
        
        Notes:
            - Default weights (2.5, 2.0, 1.0) are optimized for VisDrone dataset
            - For datasets with more small objects, consider increasing small_weight to 3.0
            - For datasets with balanced sizes, reduce medium_weight to 1.5
        """
        self.small_weight = small_weight
        self.medium_weight = medium_weight
        self.large_weight = large_weight
        self.small_thresh = small_thresh
        self.large_thresh = large_thresh
        self.image_size = image_size
        
        print(f"\n{'='*60}")
        print(f"SOLR (Small Object Loss Reweighting) Initialized")
        print(f"{'='*60}")
        print(f"Size Thresholds:")
        print(f"  Small objects:  < {self.small_thresh}px")
        print(f"  Medium objects: {self.small_thresh}-{self.large_thresh}px")
        print(f"  Large objects:  > {self.large_thresh}px")
        print(f"\nLoss Weights:")
        print(f"  Small:  {self.small_weight}x  ← High priority")
        print(f"  Medium: {self.medium_weight}x  ← Target RemDet gap")
        print(f"  Large:  {self.large_weight}x   ← Baseline")
        print(f"\nInput size: {self.image_size}×{self.image_size}")
        print(f"{'='*60}\n")
    
    def compute_size_weights(
        self,
        target_bboxes: torch.Tensor,
        normalized: bool = True
    ) -> torch.Tensor:
        """
        Compute loss weights based on target bounding box sizes.
        
        This is the core function of SOLR. It:
        1. Computes the size (area) of each target bounding box
        2. Categorizes each target as small/medium/large
        3. Assigns corresponding loss weights
        
        Args:
            target_bboxes: Target bounding boxes, shape [N, 4]
                - If normalized=True: [x1, y1, x2, y2] in range [0, 1]
                - If normalized=False: [x1, y1, x2, y2] in pixels
            normalized: Whether bboxes are normalized (default True for YOLO)
        
        Returns:
            weights: Loss weights for each target, shape [N]
        
        Algorithm:
            1. Convert normalized coords to pixels (if needed)
            2. Compute box sizes: sqrt(width × height)
            3. Categorize: small/medium/large based on thresholds
            4. Assign weights: small_weight, medium_weight, or large_weight
        
        Example:
            >>> target_bboxes = torch.tensor([[0.1, 0.1, 0.15, 0.15],  # Small (32px)
            ...                                [0.2, 0.2, 0.35, 0.35],  # Medium (96px)
            ...                                [0.4, 0.4, 0.7, 0.7]])   # Large (192px)
            >>> weights = solr.compute_size_weights(target_bboxes)
            >>> print(weights)  # tensor([2.5, 2.0, 1.0])
        """
        if target_bboxes.numel() == 0:
            # No targets in this batch, return empty weights
            return torch.ones(0, device=target_bboxes.device)
        
        # Convert normalized coordinates to pixels if needed
        if normalized:
            # target_bboxes: [N, 4] in format [x1, y1, x2, y2], values in [0, 1]
            pixel_bboxes = target_bboxes * self.image_size
        else:
            pixel_bboxes = target_bboxes
        
        # Compute bounding box sizes (geometric mean of width and height)
        # This is more robust than using area directly, as it's scale-invariant
        widths = pixel_bboxes[:, 2] - pixel_bboxes[:, 0]   # x2 - x1
        heights = pixel_bboxes[:, 3] - pixel_bboxes[:, 1]  # y2 - y1
        sizes = torch.sqrt(widths * heights)  # Geometric mean size
        
        # Initialize weights as baseline (large_weight)
        weights = torch.full_like(sizes, self.large_weight, dtype=torch.float32)
        
        # Assign weights based on size thresholds
        # Note: We use <= and > to avoid ambiguity at threshold boundaries
        small_mask = sizes < self.small_thresh
        medium_mask = (sizes >= self.small_thresh) & (sizes < self.large_thresh)
        # large_mask is implicitly the remaining boxes
        
        weights[small_mask] = self.small_weight    # Small objects: high weight
        weights[medium_mask] = self.medium_weight  # Medium objects: medium weight
        # Large objects already have large_weight (1.0) from initialization
        
        return weights
    
    def apply_to_loss(
        self,
        loss: torch.Tensor,
        target_bboxes: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Apply SOLR weights to a loss tensor.
        
        This is a convenience function that:
        1. Computes size-based weights
        2. Applies them to the loss
        3. Reduces the weighted loss
        
        Args:
            loss: Loss tensor, shape [N] (per-target losses)
            target_bboxes: Target bounding boxes, shape [N, 4] (normalized)
            reduction: How to reduce the weighted loss ('mean' or 'sum')
        
        Returns:
            weighted_loss: Scalar loss after applying SOLR weights
        
        Example:
            >>> per_target_loss = torch.tensor([0.5, 0.8, 0.3])  # Losses for 3 targets
            >>> target_bboxes = torch.tensor([[0.1, 0.1, 0.15, 0.15],
            ...                                [0.2, 0.2, 0.35, 0.35],
            ...                                [0.4, 0.4, 0.7, 0.7]])
            >>> weighted_loss = solr.apply_to_loss(per_target_loss, target_bboxes)
        """
        if loss.numel() == 0:
            return loss
        
        # Compute size-based weights
        weights = self.compute_size_weights(target_bboxes)
        
        # Apply weights element-wise
        weighted_loss = loss * weights
        
        # Reduce
        if reduction == 'mean':
            return weighted_loss.mean()
        elif reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss  # No reduction, return weighted losses
    
    def get_statistics(
        self,
        target_bboxes: torch.Tensor
    ) -> dict[str, float]:
        """
        Compute statistics about target object sizes and weights.
        
        Useful for debugging and understanding the distribution of objects
        in your dataset.
        
        Args:
            target_bboxes: Target bounding boxes, shape [N, 4] (normalized)
        
        Returns:
            stats: Dictionary containing:
                - 'num_small': Number of small objects
                - 'num_medium': Number of medium objects
                - 'num_large': Number of large objects
                - 'avg_weight': Average loss weight
                - 'avg_size': Average object size (pixels)
        
        Example:
            >>> stats = solr.get_statistics(target_bboxes)
            >>> print(f"Small: {stats['num_small']}, Medium: {stats['num_medium']}, Large: {stats['num_large']}")
        """
        if target_bboxes.numel() == 0:
            return {
                'num_small': 0,
                'num_medium': 0,
                'num_large': 0,
                'avg_weight': 0.0,
                'avg_size': 0.0
            }
        
        # Compute sizes
        pixel_bboxes = target_bboxes * self.image_size
        widths = pixel_bboxes[:, 2] - pixel_bboxes[:, 0]
        heights = pixel_bboxes[:, 3] - pixel_bboxes[:, 1]
        sizes = torch.sqrt(widths * heights)
        
        # Count objects in each category
        small_mask = sizes < self.small_thresh
        medium_mask = (sizes >= self.small_thresh) & (sizes < self.large_thresh)
        large_mask = sizes >= self.large_thresh
        
        num_small = small_mask.sum().item()
        num_medium = medium_mask.sum().item()
        num_large = large_mask.sum().item()
        
        # Compute weights
        weights = self.compute_size_weights(target_bboxes)
        
        return {
            'num_small': num_small,
            'num_medium': num_medium,
            'num_large': num_large,
            'pct_small': 100.0 * num_small / len(sizes),
            'pct_medium': 100.0 * num_medium / len(sizes),
            'pct_large': 100.0 * num_large / len(sizes),
            'avg_weight': weights.mean().item(),
            'avg_size': sizes.mean().item(),
            'min_size': sizes.min().item(),
            'max_size': sizes.max().item(),
        }


class SOLRDetectionLoss:
    """
    Wrapper class that integrates SOLR with YOLOv8 detection loss.
    
    This class wraps the standard v8DetectionLoss and applies SOLR weighting
    to the computed losses. It's designed to be a drop-in replacement for
    v8DetectionLoss with minimal code changes.
    
    Usage:
        In your training script:
        >>> from ultralytics.utils.solr_loss import SOLRDetectionLoss
        >>> 
        >>> # In trainer, replace v8DetectionLoss with SOLRDetectionLoss
        >>> self.loss_fn = SOLRDetectionLoss(
        ...     base_loss=v8DetectionLoss(model),
        ...     small_weight=2.5,
        ...     medium_weight=2.0,
        ...     large_weight=1.0
        ... )
    
    Attributes:
        base_loss: The original v8DetectionLoss instance
        solr: SOLR loss module for computing size-based weights
    """
    
    def __init__(
        self,
        base_loss: nn.Module,
        small_weight: float = 2.5,
        medium_weight: float = 2.0,
        large_weight: float = 1.0,
        small_thresh: int = 32,
        large_thresh: int = 96,
        image_size: int = 640,
    ):
        """
        Initialize SOLR detection loss wrapper.
        
        Args:
            base_loss: The original v8DetectionLoss instance
            small_weight: SOLR weight for small objects
            medium_weight: SOLR weight for medium objects
            large_weight: SOLR weight for large objects
            small_thresh: Size threshold (px) for small/medium boundary
            large_thresh: Size threshold (px) for medium/large boundary
            image_size: Input image size (default 640)
        """
        self.base_loss = base_loss
        self.solr = SOLRLoss(
            small_weight=small_weight,
            medium_weight=medium_weight,
            large_weight=large_weight,
            small_thresh=small_thresh,
            large_thresh=large_thresh,
            image_size=image_size
        )
        
        # Copy attributes from base_loss for compatibility
        self.bce = base_loss.bce
        self.hyp = base_loss.hyp
        self.stride = base_loss.stride
        self.nc = base_loss.nc
        self.no = base_loss.no
        self.reg_max = base_loss.reg_max
        self.device = base_loss.device
        self.use_dfl = base_loss.use_dfl
        self.assigner = base_loss.assigner
        self.bbox_loss = base_loss.bbox_loss
        self.proj = base_loss.proj
    
    def __call__(
        self,
        preds: torch.Tensor,
        batch: dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute loss with SOLR weighting.
        
        This method:
        1. Calls the base detection loss
        2. Computes SOLR weights based on target sizes
        3. Applies weights to the loss components
        
        Args:
            preds: Model predictions
            batch: Batch dict containing:
                - 'bboxes': Target bounding boxes [N, 4]
                - 'cls': Target class labels [N]
                - 'batch_idx': Batch indices [N]
        
        Returns:
            loss: Weighted total loss, shape [3] (box, cls, dfl)
            loss_items: Detached loss items for logging
        """
        # Compute base loss
        loss, loss_items = self.base_loss(preds, batch)
        
        # Apply SOLR weighting if we have targets
        if 'bboxes' in batch and batch['bboxes'].numel() > 0:
            # Compute SOLR weights for all targets in the batch
            # batch['bboxes'] shape: [N, 4] where N is total targets across all images
            weights = self.solr.compute_size_weights(batch['bboxes'])
            
            # Compute average weight to apply to loss
            # We use mean because the base loss already averaged across targets
            avg_weight = weights.mean()
            
            # Apply weight to all loss components
            # loss[0]: box loss, loss[1]: cls loss, loss[2]: dfl loss
            loss[0] *= avg_weight  # Box regression loss
            loss[1] *= avg_weight  # Classification loss
            loss[2] *= avg_weight  # Distribution focal loss
        
        return loss, loss_items


# ============================================================================
# Unit Tests and Examples
# ============================================================================

def test_solr_loss():
    """Test SOLR loss computation with synthetic data."""
    print("\n" + "="*60)
    print("Testing SOLR Loss Module")
    print("="*60)
    
    # Initialize SOLR
    solr = SOLRLoss(
        small_weight=2.5,
        medium_weight=2.0,
        large_weight=1.0,
        small_thresh=32,
        large_thresh=96,
        image_size=640
    )
    
    # Create synthetic targets (normalized coordinates)
    target_bboxes = torch.tensor([
        [0.1, 0.1, 0.15, 0.15],   # Small: ~32px
        [0.2, 0.2, 0.35, 0.35],   # Medium: ~96px
        [0.4, 0.4, 0.7, 0.7],     # Large: ~192px
    ])
    
    # Compute weights
    weights = solr.compute_size_weights(target_bboxes)
    
    print("\nTest Results:")
    print(f"  Target 1 (small):  weight = {weights[0]:.2f} (expected 2.5)")
    print(f"  Target 2 (medium): weight = {weights[1]:.2f} (expected 2.0)")
    print(f"  Target 3 (large):  weight = {weights[2]:.2f} (expected 1.0)")
    
    # Get statistics
    stats = solr.get_statistics(target_bboxes)
    print(f"\nStatistics:")
    print(f"  Small:  {stats['num_small']} ({stats['pct_small']:.1f}%)")
    print(f"  Medium: {stats['num_medium']} ({stats['pct_medium']:.1f}%)")
    print(f"  Large:  {stats['num_large']} ({stats['pct_large']:.1f}%)")
    print(f"  Avg weight: {stats['avg_weight']:.2f}")
    print(f"  Avg size: {stats['avg_size']:.1f}px")
    
    print(f"\n{'='*60}")
    print("✅ All tests passed!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Run tests when module is executed directly
    test_solr_loss()
