"""
DeepGuard Platform — Image Detection Subsystem (IDS)
Module  : engines.image-engine.localization.localization_engine
Layer   : Layer 4 — Localization Engine

Converts soft probabilistic heatmaps into deterministic analytical data:
- Binary segmentation masks for AI-generated regions
- Morphological boundaries for blended splices
- Bounding box coordinates for UI rendering
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple
import torch

logger = logging.getLogger(__name__)

class LocalizationEngine:
    def __init__(self, threshold: float = 0.6):
        """
        Args:
            threshold: Probability threshold (0.0 to 1.0) to convert soft heatmap to binary mask.
        """
        self.threshold = threshold
        logger.info(f"LocalizationEngine initialised | threshold={threshold}")

    def extract_manipulated_regions(self, heatmap: np.ndarray) -> np.ndarray:
        """
        Converts a soft probability heatmap into a hard binary mask of AI-generated segments.
        
        Args:
            heatmap: 2D numpy array [H, W] with values in range [0, 1]
            
        Returns:
            binary_mask: 2D numpy array [H, W] of dtype uint8 (0 or 255)
        """
        # Ensure input is safely between 0 and 1
        heatmap = np.clip(heatmap, 0.0, 1.0)
        
        # Threshold
        _, binary_mask = cv2.threshold(
            (heatmap * 255).astype(np.uint8), 
            int(self.threshold * 255), 
            255, 
            cv2.THRESH_BINARY
        )
        
        # Clean up noise using morphology (Open)
        kernel = np.ones((5, 5), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        return binary_mask

    def extract_blend_boundaries(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Traces the physical seams (blended edges) where manipulated regions meet real pixels.
        
        Args:
            binary_mask: 2D numpy array [H, W] of dtype uint8 (0 or 255)
            
        Returns:
            boundaries: 2D numpy array [H, W] highlighting the edges
        """
        kernel = np.ones((3, 3), np.uint8)
        
        # Morphological Gradient: Dilation - Erosion = Boundaries
        gradient = cv2.morphologyEx(binary_mask, cv2.MORPH_GRADIENT, kernel)
        return gradient

    def get_bounding_boxes(self, binary_mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Generates bounding boxes for the manipulated regions.
        
        Args:
            binary_mask: 2D numpy array [H, W] of dtype uint8 (0 or 255)
            
        Returns:
            List of bounding boxes: [(x, y, width, height), ...]
        """
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bboxes = []
        for contour in contours:
            # Filter out tiny noise contours (area < 50 pixels)
            if cv2.contourArea(contour) > 50:
                x, y, w, h = cv2.boundingRect(contour)
                bboxes.append((x, y, w, h))
                
        return bboxes

    def process(self, heatmap_tensor: torch.Tensor) -> List[Dict]:
        """
        End-to-end processing of a batched heatmap tensor from Layer 3.
        
        Args:
            heatmap_tensor: torch.Tensor of shape (B, 1, H, W) in range [0, 1]
            
        Returns:
            List of dictionaries (one per item in batch) containing:
            - 'manipulated_mask': np.ndarray
            - 'blend_boundaries': np.ndarray
            - 'bounding_boxes': List[Tuple]
        """
        B = heatmap_tensor.size(0)
        
        # Detach, move to CPU, convert to numpy
        heatmaps_np = heatmap_tensor.detach().cpu().squeeze(1).numpy()
        
        results = []
        for b in range(B):
            hm = heatmaps_np[b]
            
            mask = self.extract_manipulated_regions(hm)
            edges = self.extract_blend_boundaries(mask)
            bboxes = self.get_bounding_boxes(mask)
            
            results.append({
                "manipulated_mask": mask,
                "blend_boundaries": edges,
                "bounding_boxes": bboxes
            })
            
        return results
