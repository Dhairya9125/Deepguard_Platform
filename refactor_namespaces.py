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

def rename_directories():
    for engine_name, config in ENGINES.items():
        engine_path = ROOT / "engines" / engine_name
        for pkg in config["packages"]:
            old_path = engine_path / pkg
            new_path = engine_path / f"{config['prefix']}{pkg}"
            if old_path.exists() and old_path.is_dir():
                print(f"Renaming {old_path} -> {new_path}")
                os.rename(old_path, new_path)

def update_imports():
    """
    Update import statements in all Python files.
    """
    # 1. Build a mapping of old import paths to new import paths.
    #    For example: 'from preprocessing.' -> 'from img_preprocessing.' (if in image engine)
    
    # We must scan files engine by engine to apply the correct prefix locally.
    for engine_name, config in ENGINES.items():
        engine_path = ROOT / "engines" / engine_name
        
        for py_file in engine_path.rglob("*.py"):
            with open(py_file, "r", encoding="utf-8") as f:
                content = f.read()
            
            new_content = content
            for pkg in config["packages"]:
                # Replace absolute imports from the old package name
                # e.g., `from preprocessing.face_detector import ...` -> `from img_preprocessing.face_detector import ...`
                # e.g., `import preprocessing.face_detector` -> `import img_preprocessing.face_detector`
                
                # We need to use regex to ensure we match whole words
                pattern_from = rf"from {pkg}\b"
                replacement_from = f"from {config['prefix']}{pkg}"
                new_content = re.sub(pattern_from, replacement_from, new_content)
                
                pattern_import = rf"import {pkg}\b"
                replacement_import = f"import {config['prefix']}{pkg}"
                new_content = re.sub(pattern_import, replacement_import, new_content)
            
            if new_content != content:
                print(f"Updating imports in {py_file}")
                with open(py_file, "w", encoding="utf-8") as f:
                    f.write(new_content)

    # 2. Update imports in apps/api/
    api_path = ROOT / "apps" / "api"
    for py_file in api_path.rglob("*.py"):
        with open(py_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        new_content = content
        
        # In apps/api, they might import from the engine directories
        # But wait, apps/api just imports from `preprocessing` because they added the engine to sys.path!
        # Which engine does it belong to?
        # routers/image.py or pipeline_ids.py -> image engine packages
        # routers/audio.py or services/ads_pipeline.py -> audio engine packages
        # routers/video.py or services/vds_pipeline.py -> video engine packages
        
        # We can just apply the appropriate mapping based on filename or context
        if "audio" in py_file.name or "ads" in py_file.name:
            for pkg in ENGINES["audio-engine"]["packages"]:
                new_content = re.sub(rf"from {pkg}\b", f"from aud_{pkg}", new_content)
                new_content = re.sub(rf"import {pkg}\b", f"import aud_{pkg}", new_content)
        elif "video" in py_file.name or "vds" in py_file.name:
            for pkg in ENGINES["video-engine"]["packages"]:
                new_content = re.sub(rf"from {pkg}\b", f"from vid_{pkg}", new_content)
                new_content = re.sub(rf"import {pkg}\b", f"import vid_{pkg}", new_content)
        elif "image" in py_file.name or "ids" in py_file.name:
            for pkg in ENGINES["image-engine"]["packages"]:
                new_content = re.sub(rf"from {pkg}\b", f"from img_{pkg}", new_content)
                new_content = re.sub(rf"import {pkg}\b", f"import img_{pkg}", new_content)
                
        # Also handle fusion engine
        for pkg in ENGINES["fusion-engine"]["packages"]:
            new_content = re.sub(rf"from {pkg}\b", f"from fusion_{pkg}", new_content)
            new_content = re.sub(rf"import {pkg}\b", f"import fusion_{pkg}", new_content)
                
        if new_content != content:
            print(f"Updating API imports in {py_file}")
            with open(py_file, "w", encoding="utf-8") as f:
                f.write(new_content)

if __name__ == "__main__":
    rename_directories()
    update_imports()
    print("Refactoring completed.")
