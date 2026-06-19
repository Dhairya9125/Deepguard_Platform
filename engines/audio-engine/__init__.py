"""
Audio Detection Subsystem (ADS) — Engine Package
Detects AI voice cloning, synthetic speech, voice conversion,
diffusion-based audio attacks, and audio splicing.
"""
from .pipeline_ads import ADSPipeline
__all__ = ["ADSPipeline"]
