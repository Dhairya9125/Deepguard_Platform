"""
DeepGuard Platform — Image Detection Subsystem (IDS)
Master Pipeline API

Wraps Layers 1-5 into a single, clean inference pipeline.
"""

import os
import sys
import torch
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any

_CURRENT_DIR = Path(__file__).resolve().parent
if str(_CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(_CURRENT_DIR))

from img_preprocessing.face_detector import FaceDetector
from img_preprocessing.face_aligner import FaceAligner
from img_preprocessing.image_normalizer import ImageNormalizer
from img_preprocessing.image_augmenter import ImageAugmenter
from img_preprocessing.frequency_analyzer import FrequencyAnalyzer

from img_feature_extraction.spatial_branch import ClipSpatialBranch
from img_feature_extraction.frequency_branch import EfficientNetFrequencyBranch
from img_feature_extraction.discrepancy_branch import DiscrepancyBranch
from img_feature_extraction.noise_branch import NoiseResidualBranch
from img_feature_extraction.fingerprint_branch import FingerprintBranch

from img_fusion.ids_fusion_engine import IDSFusionEngine
from img_localization.localization_engine import LocalizationEngine
from img_explainability.explainability_engine import ExplainabilityEngine

logger = logging.getLogger(__name__)

class IDSPipeline:
    def __init__(self, device: str = 'cpu'):
        logger.info("Initializing Full IDS Pipeline...")
        self.device = torch.device(device)
        
        # --- Layer 1: Preprocessing ---
        self.detector = FaceDetector()
        self.aligner = FaceAligner()
        self.normalizer = ImageNormalizer()
        self.freq_analyzer = FrequencyAnalyzer()
        
        # --- Layer 2: Feature Extraction ---
        try:
            self.branch_spatial = ClipSpatialBranch(proj_dim=512).to(self.device).eval()
            self.branch_freq = EfficientNetFrequencyBranch(proj_dim=512).to(self.device).eval()
            self.branch_disc = DiscrepancyBranch(proj_dim=512).to(self.device).eval()
            self.branch_noise = NoiseResidualBranch(proj_dim=512).to(self.device).eval()
            self.branch_fingerprint = FingerprintBranch(proj_dim=512).to(self.device).eval()
            self.models_loaded = True
        except Exception as e:
            logger.warning(f"Could not load full engine weights. Running in DEMO/MOCK mode: {e}")
            self.models_loaded = False
            
        # --- Layer 3: Fusion Engine ---
        self.fusion = IDSFusionEngine(embed_dim=512, num_branches=5).to(self.device).eval()
        
        # --- Layer 4: Localization ---
        self.localization = LocalizationEngine(threshold=0.6)
        
        # --- Layer 5: Explainability ---
        self.explainability = ExplainabilityEngine()
        
        logger.info("IDS Pipeline Successfully Initialized.")

    @torch.no_grad()
    def predict(self, image_path: str) -> Dict[str, Any]:
        """Runs an image through the entire IDS pipeline."""
        logger.info(f"Processing image: {image_path}")
        
        # 1. PREPROCESSING
        # Detect face
        face_boxes = self.detector.detect(image_path)
        if not face_boxes:
            return {"error": "No face detected in image."}
        
        # Align (takes first face)
        aligned_face = self.aligner.align(image_path, face_boxes[0])
        
        # Normalize for spatial branches
        spatial_tensor = self.normalizer.normalize(aligned_face).unsqueeze(0).to(self.device)
        
        # Frequency analysis (FFT/DCT/Wavelet stack)
        freq_stack = self.freq_analyzer.process(aligned_face).unsqueeze(0).to(self.device)
        
        # 2. FEATURE EXTRACTION
        if self.models_loaded:
            # Generate 512-d embeddings
            spatial_emb = self.branch_spatial(spatial_tensor)
            freq_emb = self.branch_freq(freq_stack)
            disc_emb = self.branch_disc(spatial_tensor)
            noise_emb = self.branch_noise(spatial_tensor)
            finger_emb = self.branch_fingerprint(spatial_tensor)
        else:
            # Mock embeddings if models failed to load
            batch_size = spatial_tensor.size(0)
            spatial_emb = torch.randn(batch_size, 512, device=self.device)
            freq_emb = torch.randn(batch_size, 512, device=self.device)
            disc_emb = torch.randn(batch_size, 512, device=self.device)
            noise_emb = torch.randn(batch_size, 512, device=self.device)
            finger_emb = torch.randn(batch_size, 512, device=self.device)
            
        embeddings = {
            "spatial": spatial_emb,
            "frequency": freq_emb,
            "discrepancy": disc_emb,
            "noise": noise_emb,
            "fingerprint": finger_emb
        }
        
        # 3. FUSION
        fusion_out = self.fusion(embeddings)
        fake_prob = fusion_out["fake_probability"].item()
        ood_score = fusion_out["OOD_score"].item()
        heatmap_tensor = fusion_out["manipulation_heatmap"]
        
        # 4. LOCALIZATION
        loc_results = self.localization.process(heatmap_tensor)[0]
        bboxes = loc_results["bounding_boxes"]
        
        # 5. EXPLAINABILITY
        scores = {
            'fake_probability': fake_prob,
            'ood_score': ood_score,
            'spatial_signal': float(spatial_emb.mean()),
            'frequency_signal': float(freq_emb.mean()),
            'discrepancy_signal': float(disc_emb.mean()),
            'noise_signal': float(noise_emb.mean()),
            'fingerprint_signal': float(finger_emb.mean())
        }
        text_report = self.explainability.generate_text_report(scores)
        
        # Prepare final output dictionary
        return {
            "status": "success",
            "fake_probability": fake_prob,
            "ood_score": ood_score,
            "bounding_boxes": bboxes,
            "forensic_report": text_report,
            "heatmap_tensor": heatmap_tensor # Raw tensor if UI needs it
        }
