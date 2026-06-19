import os
from pathlib import Path
import re

ROOT = Path(r"C:\Users\prana\OneDrive\Documents\Desktop\TRUX\Deepguard_Platform-main")

ENGINES = {
    "image-engine": {
        "prefix": "img_",
        "packages": [
            "data", "explainability", "feature_extraction", "fusion", 
            "ingestion", "localization", "models", "preprocessing"
        ]
    },
    "audio-engine": {
        "prefix": "aud_",
        "packages": [
            "config", "domain", "explainability", "fusion", 
            "localization", "models", "preprocessing", "signal_processing"
        ]
    },
    "video-engine": {
        "prefix": "vid_",
        "packages": [
            "explainability", "feature_extraction", "localization", "preprocessing"
        ]
    },
    "fusion-engine": {
        "prefix": "fusion_",
        "packages": [
            "core"
        ]
    }
}

def update_imports():
    for engine_name, config in ENGINES.items():
        engine_path = ROOT / "engines" / engine_name
        
        for py_file in engine_path.rglob("*.py"):
            with open(py_file, "r", encoding="utf-8") as f:
                content = f.read()
            
            new_content = content
            for pkg in config["packages"]:
                # relative imports
                pattern_from = rf"from \.{pkg}\b"
                replacement_from = f"from .{config['prefix']}{pkg}"
                new_content = re.sub(pattern_from, replacement_from, new_content)
                
                pattern_from_2 = rf"from \.\.{pkg}\b"
                replacement_from_2 = f"from ..{config['prefix']}{pkg}"
                new_content = re.sub(pattern_from_2, replacement_from_2, new_content)
            
            if new_content != content:
                print(f"Updating relative imports in {py_file}")
                with open(py_file, "w", encoding="utf-8") as f:
                    f.write(new_content)

if __name__ == "__main__":
    update_imports()
    print("Relative imports fixed.")
