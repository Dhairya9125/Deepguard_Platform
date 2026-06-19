"""
DeepGuard Platform — Image Detection Subsystem (IDS)
Module  : engines.image-engine.ingestion.media_ingestion
Layer   : Layer 0 — Media Ingestion

Extracts EXIF metadata, computes cryptographic & perceptual hashes,
and scans for known AI-generation software tags.
"""

import os
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

from PIL import Image
import imagehash
import exifread

logger = logging.getLogger(__name__)

# Known metadata tags left by generative models
AI_SIGNATURES = [
    "Midjourney", "Stable Diffusion", "DALL-E", "Adobe Firefly", 
    "ComfyUI", "Automatic1111", "Fooocus", "NovelAI"
]

class IngestionEngine:
    def __init__(self):
        logger.info("Layer 0: IngestionEngine initialized.")
        
    def _compute_sha256(self, file_path: Path) -> str:
        """Computes the cryptographic SHA-256 hash of the raw file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256.update(byte_block)
        return sha256.hexdigest()

    def _compute_phash(self, img: Image.Image) -> str:
        """Computes the Perceptual Hash (pHash) for visual deduplication."""
        return str(imagehash.phash(img))

    def _extract_exif(self, file_path: Path) -> Dict[str, str]:
        """Extracts readable EXIF metadata from the file."""
        exif_data = {}
        with open(file_path, "rb") as f:
            tags = exifread.process_file(f, details=False)
            for tag, value in tags.items():
                if tag not in ('JPEGThumbnail', 'TIFFThumbnail', 'Filename', 'EXIF MakerNote'):
                    exif_data[tag] = str(value).strip()
        return exif_data

    def _scan_for_ai_signatures(self, exif_data: Dict[str, str]) -> Tuple[bool, str]:
        """Scans the EXIF metadata for hardcoded AI generator tags."""
        for tag, value in exif_data.items():
            for sig in AI_SIGNATURES:
                if sig.lower() in value.lower():
                    return True, sig
        return False, ""

    def process(self, image_path: str) -> Dict[str, Any]:
        """
        Ingests the image and returns all Layer 0 metadata and hashes.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found at {path}")
            
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > 50.0:
            logger.warning(f"File size ({file_size_mb:.2f} MB) exceeds 50MB limit.")

        logger.info(f"Ingesting image: {path.name}")
        
        # 1. Cryptographic Hash
        sha256_hash = self._compute_sha256(path)
        
        # 2. Image Loading & pHash
        try:
            img = Image.open(path)
            # Ensure image is fully loaded to catch truncation errors
            img.verify()
            img = Image.open(path) # Reopen after verify
            
            phash = self._compute_phash(img)
            img_format = img.format
            img_size = img.size
            img_mode = img.mode
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            raise
            
        # 3. EXIF Extraction
        exif = self._extract_exif(path)
        
        # 4. AI Signature Scan
        has_ai_sig, ai_model = self._scan_for_ai_signatures(exif)
        
        return {
            "file_name": path.name,
            "format": img_format,
            "resolution": f"{img_size[0]}x{img_size[1]}",
            "color_mode": img_mode,
            "size_mb": round(file_size_mb, 2),
            "sha256": sha256_hash,
            "phash": phash,
            "exif_metadata": exif,
            "ai_signature_detected": has_ai_sig,
            "ai_model_tag": ai_model if has_ai_sig else None
        }
