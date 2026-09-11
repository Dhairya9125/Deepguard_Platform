"""
DeepGuard Platform — Image Detection Subsystem (IDS)
Module  : engines.image-engine.preprocessing.frequency_analyzer
Layer   : Layer 1 — Preprocessing
Task    : Frequency Domain Preparation (FFT / DCT)

Architecture Reference:
  - Branch B (Frequency Domain Features): computes 2D FFT and DCT representations 
    to expose GAN grid artifacts, diffusion noise patterns, and compression 
    inconsistencies invisible in the spatial pixel domain.
  - Tool: NumPy, SciPy

Usage:
    from preprocessing.frequency_analyzer import FrequencyAnalyzer
    
    analyzer = FrequencyAnalyzer()
    fft_spectrum = analyzer.get_fft_spectrum(aligned_face_bgr)
    dct_spectrum = analyzer.get_dct_spectrum(aligned_face_bgr)
"""

import logging
from typing import Tuple, Union

import cv2
import numpy as np
import pywt
import torch
from scipy.fft import dctn

logger = logging.getLogger(__name__)


class FrequencyAnalyzer:
    """
    Computes 2D Fast Fourier Transform (FFT) and Discrete Cosine Transform (DCT)
    of images to expose frequency-domain artifacts left by Generative AI models.
    """

    def __init__(self, log_scale: bool = True) -> None:
        """
        Args:
            log_scale: If True, applies log scaling to the magnitude spectrum 
                       (20 * log10(abs(X) + eps)) which is standard for CNN inputs.
        """
        self.log_scale = log_scale
        logger.info("FrequencyAnalyzer initialised | log_scale=%s", log_scale)

    def _prepare_grayscale(self, image: Union[np.ndarray, str]) -> np.ndarray:
        """Ensures the image is a 2D grayscale float32 array."""
        if isinstance(image, str):
            img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Image not found: {image}")
        else:
            # If 3-channel, convert to grayscale
            if len(image.shape) == 3:
                # Assume BGR or RGB; using standard OpenCV conversion (defaults to BGR2GRAY but 
                # practically RGB2GRAY yields similar frequency profiles for deepfakes)
                img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif len(image.shape) == 2:
                img = image
            else:
                raise ValueError(f"Unsupported image shape: {image.shape}")
                
        # Normalize pixel values to [0, 1] for stable transforms
        if img.dtype == np.uint8:
            return img.astype(np.float32) / 255.0
        return img.astype(np.float32)

    def get_fft_spectrum(self, image: Union[np.ndarray, str]) -> np.ndarray:
        """
        Computes the 2D FFT magnitude spectrum.
        Low frequencies are shifted to the center.
        """
        gray = self._prepare_grayscale(image)
        
        # Compute 2D Fast Fourier Transform
        f_transform = np.fft.fft2(gray)
        
        # Shift the zero-frequency component to the center of the spectrum
        f_shift = np.fft.fftshift(f_transform)
        
        # Compute magnitude
        magnitude = np.abs(f_shift)
        
        if self.log_scale:
            # 20 * log10(magnitude) to match dB scale (add epsilon to avoid log(0))
            magnitude = 20 * np.log10(magnitude + 1e-8)
            
        return magnitude

    def get_dct_spectrum(self, image: Union[np.ndarray, str]) -> np.ndarray:
        """
        Computes the 2D Discrete Cosine Transform (DCT) magnitude spectrum.
        DCT concentrates most of the signal energy in the top-left corner.
        """
        gray = self._prepare_grayscale(image)
        
        # Compute 2D DCT (Type-II, orthogonal normalization for CNN stability)
        dct_coeffs = dctn(gray, type=2, norm='ortho')
        
        # Compute magnitude
        magnitude = np.abs(dct_coeffs)
        
        if self.log_scale:
            magnitude = 20 * np.log10(magnitude + 1e-8)
            
        return magnitude

    def get_wavelet_spectrum(self, image: Union[np.ndarray, str]) -> np.ndarray:
        """
        Computes the 2D Discrete Wavelet Transform (DWT) using the Haar wavelet.
        Returns the magnitude of the high-frequency detail coefficients (HH),
        which is highly sensitive to checkerboard artifacts from CNN upsampling.
        """
        gray = self._prepare_grayscale(image)
        
        # Compute 2D DWT
        # Returns: (cA, (cH, cV, cD)) 
        # cA = Approximation, cH = Horizontal detail, cV = Vertical detail, cD = Diagonal detail
        coeffs2 = pywt.dwt2(gray, 'haar')
        cA, (cH, cV, cD) = coeffs2
        
        # We focus on the high-frequency diagonal details (cD) for artifacts,
        # but combining all details gives a robust artifact map.
        details_magnitude = np.abs(cH) + np.abs(cV) + np.abs(cD)
        
        # Resize the wavelet output back to the original image size 
        # (since DWT halves the resolution)
        h, w = gray.shape
        details_resized = cv2.resize(details_magnitude, (w, h), interpolation=cv2.INTER_LINEAR)
        
        if self.log_scale:
            details_resized = 20 * np.log10(details_resized + 1e-8)
            
        return details_resized

    def get_stacked_frequency_tensor(self, image: Union[np.ndarray, str]) -> torch.Tensor:
        """
        Computes FFT, DCT, and Wavelet spectra, normalizes them, and stacks them 
        into a 3-channel (C, H, W) PyTorch tensor suitable for EfficientNet input.
        
        Channel 0: FFT
        Channel 1: DCT
        Channel 2: Wavelet
        """
        fft_mag = self.get_fft_spectrum(image)
        dct_mag = self.get_dct_spectrum(image)
        wave_mag = self.get_wavelet_spectrum(image)
        
        # Normalize each channel individually to [0, 1] for neural network stability
        def normalize(arr):
            min_v = np.min(arr)
            max_v = np.max(arr)
            if max_v == min_v:
                return np.zeros_like(arr, dtype=np.float32)
            return ((arr - min_v) / (max_v - min_v)).astype(np.float32)
            
        c0 = normalize(fft_mag)
        c1 = normalize(dct_mag)
        c2 = normalize(wave_mag)
        
        # Stack into (3, H, W) NumPy array
        stacked_np = np.stack([c0, c1, c2], axis=0)
        
        # Convert to PyTorch Tensor
        return torch.from_numpy(stacked_np)

    def to_visual_tensor(self, magnitude_spectrum: np.ndarray) -> np.ndarray:
        """
        Normalizes a raw magnitude spectrum to [0, 255] uint8 format 
        so it can be visualised or saved as an image.
        """
        # Min-Max scaling
        min_val = np.min(magnitude_spectrum)
        max_val = np.max(magnitude_spectrum)
        
        if max_val == min_val:
            return np.zeros_like(magnitude_spectrum, dtype=np.uint8)
            
        norm = (magnitude_spectrum - min_val) / (max_val - min_val)
        return (norm * 255).astype(np.uint8)
