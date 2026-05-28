import os
from pathlib import Path

import cv2
import numpy as np

from preprocessing.frequency_analyzer import FrequencyAnalyzer

def test():
    print("Testing FrequencyAnalyzer (FFT / DCT)...")
    
    # We can use the aligned face output from the FaceAligner test
    img_path = Path("test_outputs/aligned_mediapipe_1.jpg")
    
    if img_path.exists():
        img = cv2.imread(str(img_path))
    else:
        # Create a synthetic image with a clear frequency pattern (grid)
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        img[::10, ::10] = 255
        print("Using synthetic grid image for test.")
        
    print(f"Input image shape: {img.shape}")
    
    analyzer = FrequencyAnalyzer(log_scale=True)
    
    # Compute FFT
    fft_mag = analyzer.get_fft_spectrum(img)
    fft_vis = analyzer.to_visual_tensor(fft_mag)
    
    # Compute DCT
    dct_mag = analyzer.get_dct_spectrum(img)
    dct_vis = analyzer.to_visual_tensor(dct_mag)
    
    print(f"FFT output shape: {fft_mag.shape}, dtype: {fft_mag.dtype}")
    print(f"DCT output shape: {dct_mag.shape}, dtype: {dct_mag.dtype}")
    
    # Save outputs
    out_dir = Path("test_outputs")
    out_dir.mkdir(exist_ok=True)
    
    # Apply a pseudo-color map for better visualization of frequency intensities
    fft_color = cv2.applyColorMap(fft_vis, cv2.COLORMAP_JET)
    dct_color = cv2.applyColorMap(dct_vis, cv2.COLORMAP_JET)
    
    cv2.imwrite(str(out_dir / "freq_fft_spectrum.jpg"), fft_color)
    cv2.imwrite(str(out_dir / "freq_dct_spectrum.jpg"), dct_color)
    
    print("Frequency spectra computed successfully! Outputs saved to test_outputs/freq_*.jpg")

if __name__ == "__main__":
    test()
