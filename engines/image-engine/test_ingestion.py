import cv2
import numpy as np
import piexif
from PIL import Image
from pathlib import Path
import json

from img_ingestion.media_ingestion import IngestionEngine

def create_dummy_ai_image(path):
    # Create dummy image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(path, img)
    
    # Inject fake EXIF metadata simulating Midjourney
    zeroth_ifd = {
        piexif.ImageIFD.Software: "Midjourney v6.0",
        piexif.ImageIFD.Make: "AI Generator"
    }
    exif_dict = {"0th": zeroth_ifd}
    exif_bytes = piexif.dump(exif_dict)
    
    # Reload and save with EXIF
    im = Image.open(path)
    im.save(path, "jpeg", exif=exif_bytes)


def test():
    print("Testing Layer 0: Media Ingestion Engine...")
    
    test_img = "test_outputs/dummy_ingest.jpg"
    Path("test_outputs").mkdir(exist_ok=True)
    create_dummy_ai_image(test_img)
    
    engine = IngestionEngine()
    results = engine.process(test_img)
    
    print("\n--- Ingestion Results ---")
    print(json.dumps(results, indent=4))
    
    assert results["ai_signature_detected"] is True
    assert results["ai_model_tag"] == "Midjourney"
    assert "sha256" in results
    assert "phash" in results
    
    print("\nTest passed! Layer 0 successfully extracted metadata, calculated hashes, and identified the AI signature.")

if __name__ == "__main__":
    test()
