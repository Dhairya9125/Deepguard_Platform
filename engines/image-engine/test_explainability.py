import numpy as np
import torch
import cv2
from pathlib import Path

from explainability.explainability_engine import ExplainabilityEngine

def test():
    print("Testing Layer 5: Explainability Engine...")
    
    engine = ExplainabilityEngine()
    
    out_dir = Path("test_outputs")
    out_dir.mkdir(exist_ok=True)
    
    # 1. Test Spectral Visualizer
    print("Visualizing simulated FFT spectrum...")
    # Simulate a 224x224 FFT map with a cross pattern (typical JPEG/grid artifact)
    fft_sim = np.zeros((224, 224))
    fft_sim[112, :] = 1000
    fft_sim[:, 112] = 1000
    spectral_vis = engine.visualize_spectrum(fft_sim)
    cv2.imwrite(str(out_dir / "spectral_vis.jpg"), cv2.cvtColor(spectral_vis, cv2.COLOR_RGB2BGR))
    
    # 2. Test Residual Visualizer
    print("Visualizing simulated SRM noise...")
    srm_sim = np.random.randn(3, 224, 224) * 0.1 # Soft noise
    srm_sim[:, 100:150, 100:150] += 5.0 # Spike in the middle (splice)
    residual_vis = engine.visualize_residual(srm_sim)
    cv2.imwrite(str(out_dir / "residual_vis.jpg"), residual_vis)
    
    # 3. Test Text Report
    print("Generating forensic text report...")
    dummy_scores = {
        'fake_probability': 0.94,
        'ood_score': 0.8,
        'frequency_signal': 0.75,
        'spatial_signal': 0.4,
        'noise_signal': 0.88,
        'discrepancy_signal': 0.91
    }
    report = engine.generate_text_report(dummy_scores)
    print("\n--- FORENSIC REPORT ---")
    print(report)
    print("-----------------------\n")
    
    # Check shapes
    assert spectral_vis.shape == (224, 224, 3)
    assert residual_vis.shape == (224, 224)
    assert "CRITICAL" in report
    assert "Notice: High Out-of-Distribution" in report
    
    print("Test passed! Explainability components successfully generated.")

if __name__ == "__main__":
    test()
