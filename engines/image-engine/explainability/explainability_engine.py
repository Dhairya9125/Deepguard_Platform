"""
DeepGuard Platform — Image Detection Subsystem (IDS)
Module  : engines.image-engine.explainability.explainability_engine
Layer   : Layer 5 — Explainability Engine

Translates mathematical arrays (FFT, SRM, Probabilities) into
human-readable forensic evidence: Spectral images, Noise maps, and Text Reports.
"""

import cv2
import numpy as np
import torch
import logging
from typing import Dict, Any

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

logger = logging.getLogger(__name__)


class ExplainabilityEngine:
    def __init__(self):
        logger.info("ExplainabilityEngine initialised.")

    def generate_gradcam(self, model: torch.nn.Module, target_layers: list, input_tensor: torch.Tensor, original_image: np.ndarray) -> np.ndarray:
        """
        Generates a GradCAM visualization indicating which pixels triggered the network.
        
        Args:
            model: The PyTorch module (e.g., Branch C Discrepancy ResNet).
            target_layers: List of target layers to hook into.
            input_tensor: Tensor of shape (1, 3, H, W) to pass to the model.
            original_image: RGB image normalized between [0, 1] of shape (H, W, 3).
            
        Returns:
            RGB numpy array showing the heatmap overlaid on the original image.
        """
        cam = GradCAM(model=model, target_layers=target_layers)
        
        # We don't specify targets so it defaults to the highest scoring category
        grayscale_cam = cam(input_tensor=input_tensor)
        
        # Take the first image in the batch
        grayscale_cam = grayscale_cam[0, :]
        
        # Overlay on original image
        visualization = show_cam_on_image(original_image, grayscale_cam, use_rgb=True)
        return visualization

    def visualize_spectrum(self, fft_tensor: torch.Tensor) -> np.ndarray:
        """
        Converts raw 2D FFT data into a visible Inferno colormap.
        
        Args:
            fft_tensor: Magnitude spectrum tensor of shape (H, W).
            
        Returns:
            RGB numpy array showing the frequency spectrum.
        """
        # Ensure it's numpy
        if isinstance(fft_tensor, torch.Tensor):
            fft_np = fft_tensor.detach().cpu().numpy()
        else:
            fft_np = fft_tensor
            
        # Logarithmic scaling to bring out subtle high-frequency details
        # Add a small epsilon to avoid log(0)
        fft_log = np.log(fft_np + 1e-8)
        
        # Normalize to [0, 255]
        fft_norm = cv2.normalize(fft_log, None, 0, 255, cv2.NORM_MINMAX)
        fft_norm = np.uint8(fft_norm)
        
        # Apply Inferno colormap
        spectral_img = cv2.applyColorMap(fft_norm, cv2.COLORMAP_INFERNO)
        
        # OpenCV uses BGR for ColorMap, convert to RGB for standard processing
        spectral_img = cv2.cvtColor(spectral_img, cv2.COLOR_BGR2RGB)
        return spectral_img

    def visualize_residual(self, srm_tensor: torch.Tensor) -> np.ndarray:
        """
        Normalizes microscopic SRM high-pass noise into a high-contrast grayscale image.
        
        Args:
            srm_tensor: Tensor of shape (C, H, W).
            
        Returns:
            Grayscale numpy array of the sensor noise.
        """
        if isinstance(srm_tensor, torch.Tensor):
            srm_np = srm_tensor.detach().cpu().numpy()
        else:
            srm_np = srm_tensor
            
        # If multi-channel, average them to get a single noise map
        if len(srm_np.shape) == 3:
            srm_np = np.mean(srm_np, axis=0)
            
        # Normalize strictly to expose micro-details
        srm_norm = cv2.normalize(np.abs(srm_np), None, 0, 255, cv2.NORM_MINMAX)
        srm_norm = np.uint8(srm_norm)
        
        return srm_norm

    def generate_text_report(self, scores: Dict[str, Any]) -> str:
        """
        Generates a natural language forensic summary based on network outputs.
        
        Args:
            scores: Dictionary containing probabilities and branch signals.
                Expected keys: 'fake_probability', 'ood_score', 'spatial_signal', 'frequency_signal', etc.
                
        Returns:
            A string containing the forensic explanation.
        """
        prob = scores.get('fake_probability', 0.0)
        ood = scores.get('ood_score', 0.0)
        
        report = []
        
        # 1. Primary Decision
        if prob > 0.85:
            report.append(f"CRITICAL: The media exhibits extremely high likelihood of manipulation (Confidence: {prob*100:.1f}%).")
        elif prob > 0.5:
            report.append(f"WARNING: The media shows moderate signs of manipulation (Confidence: {prob*100:.1f}%).")
        else:
            report.append(f"CLEAN: The media appears authentic (Fake Confidence: {prob*100:.1f}%).")
            
        # 2. Out of Distribution Check
        if ood > 0.7:
            report.append("Notice: High Out-of-Distribution (OOD) score detected. This signature does not perfectly match known generators, implying a novel or heavily adversarial deepfake technique.")
            
        # 3. Branch Specific Explanations
        if prob > 0.5:
            report.append("Forensic Evidence Breakdown:")
            
            if scores.get('frequency_signal', 0.0) > 0.6:
                report.append("- Frequency Branch: Detected anomalous spectral grids and high-frequency noise consistent with GAN upsampling or Diffusion latent decoding.")
            if scores.get('spatial_signal', 0.0) > 0.6:
                report.append("- Spatial/Semantic Branch: Flagged unnatural structural textures and contextual semantic mismatches.")
            if scores.get('noise_signal', 0.0) > 0.6:
                report.append("- Noise Residual Branch: Discovered PRNU sensor inconsistencies, strongly indicating multiple images were spliced together.")
            if scores.get('discrepancy_signal', 0.0) > 0.6:
                report.append("- Discrepancy Branch: Localized patches violated universal continuity invariants.")
                
        return "\n".join(report)
