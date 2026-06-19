"""
DeepGuard Platform — IDS Layer 1 Face Detection: Quick Test Script

Run from the image-engine directory:
    cd engines/image-engine
    python test_face_detector.py

Or from the repo root:
    python engines/image-engine/test_face_detector.py

Requirements:
    pip install insightface onnxruntime opencv-python numpy Pillow

What this script does:
  1. Downloads a public-domain portrait image for testing.
  2. Runs the full detect_and_align() pipeline.
  3. Prints per-face detection results (bbox, landmarks, score).
  4. Saves a visualized output image (annotated bounding boxes + landmarks).
  5. Saves individual aligned face crops to disk.
  6. Serializes detection results to JSON.
"""

import sys
import os
import logging
import json
import urllib.request

# Allow import from local engine package without installing
_ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ENGINE_DIR)

import cv2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

# ─── Import our module ─────────────────────────────────────────────────────────
from img_preprocessing.face_detector import FaceDetector, FaceDetectionResult

# ─── Configuration ─────────────────────────────────────────────────────────────
# Sample face image from InsightFace's own test data (same library we use)
# Multiple fallback URLs in case one is unavailable
FACE_IMAGE_URLS = [
    "https://raw.githubusercontent.com/deepinsight/insightface/master/examples/sample-images/t1.jpg",
    "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/obama.jpg",
    "https://github.com/deepinsight/insightface/raw/master/examples/sample-images/t1.jpg",
]

OUTPUT_DIR = os.path.join(_ENGINE_DIR, "test_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
SAMPLE_PATH = os.path.join(OUTPUT_DIR, "sample_face.jpg")

# ─── Download sample image ─────────────────────────────────────────────────────
if not os.path.exists(SAMPLE_PATH):
    import requests as _req
    downloaded = False
    for url in FACE_IMAGE_URLS:
        try:
            print(f"\n[INFO] Downloading sample image ...\n  -> {url}")
            _headers = {"User-Agent": "Mozilla/5.0 (compatible; DeepGuard-Test/1.0)"}
            _resp = _req.get(url, headers=_headers, timeout=30)
            _resp.raise_for_status()
            with open(SAMPLE_PATH, "wb") as _f:
                _f.write(_resp.content)
            print(f"[INFO] Saved to: {SAMPLE_PATH}")
            downloaded = True
            break
        except Exception as e:
            print(f"[WARN] Failed ({e}) -- trying next URL...")
    if not downloaded:
        sys.exit("[ERROR] All download URLs failed. Please place a face image at:\n" +
                 f"        {SAMPLE_PATH}")
else:
    print(f"\n[INFO] Using cached sample image: {SAMPLE_PATH}")

# ─── Load image ────────────────────────────────────────────────────────────────
img_bgr = cv2.imread(SAMPLE_PATH)
if img_bgr is None:
    sys.exit(f"[ERROR] Could not read image: {SAMPLE_PATH}")

print(f"[INFO] Image loaded — shape: {img_bgr.shape}  (H × W × BGR)")

# ─── Initialise detector ───────────────────────────────────────────────────────
print("\n" + "═" * 62)
print("  DeepGuard IDS — Layer 1: Face Detection & Alignment Test")
print("═" * 62)

detector = FaceDetector(
    confidence_threshold=0.50,   # lowered for test/debug — production default is 0.90
    target_size=(224, 224),      # ViT / EfficientNet standard input size
    align=True,
    model_pack="buffalo_l",     # small+fast; swap to "buffalo_l" for accuracy
)
print(f"\n  Detector: {detector}\n")

# ─── Run full pipeline ─────────────────────────────────────────────────────────
print("[INFO] Running detect_and_align() ...")
results = detector.detect_and_align(img_bgr)

if not results:
    print("\n[WARNING] No faces detected above confidence threshold.")
    print("          Try lowering confidence_threshold or use a different image.")
    sys.exit(0)

# ─── Print results ─────────────────────────────────────────────────────────────
print(f"\n  ✓  Detected {len(results)} face(s):\n")
for face in results:
    print(f"  ┌─ {face.face_id}")
    print(f"  │  Confidence  : {face.score:.4f}  ({face.score:.2%})")
    print(f"  │  Bounding box: {face.bbox}   (x1, y1, x2, y2)")
    print(f"  │  Dimensions  : {face.width}px × {face.height}px")
    print(f"  │  Landmarks:")
    print(f"  │    right_eye  : {tuple(round(v,1) for v in face.landmarks.right_eye)}")
    print(f"  │    left_eye   : {tuple(round(v,1) for v in face.landmarks.left_eye)}")
    print(f"  │    nose       : {tuple(round(v,1) for v in face.landmarks.nose)}")
    print(f"  │    mouth_right: {tuple(round(v,1) for v in face.landmarks.mouth_right)}")
    print(f"  │    mouth_left : {tuple(round(v,1) for v in face.landmarks.mouth_left)}")
    if face.aligned_crop is not None:
        print(f"  │  Aligned crop : shape={face.aligned_crop.shape}  dtype={face.aligned_crop.dtype}")
    print(f"  └{'─' * 50}")
    print()

# ─── Save JSON results ─────────────────────────────────────────────────────────
json_path = os.path.join(OUTPUT_DIR, "detection_results.json")
with open(json_path, "w") as jf:
    json.dump([f.to_dict() for f in results], jf, indent=2)
print(f"[OUT] JSON results        → {json_path}")

# ─── Save annotated visualization ─────────────────────────────────────────────
vis_img = detector.visualize(img_bgr, results, draw_landmarks=True, draw_score=True)
vis_path = os.path.join(OUTPUT_DIR, "visualized_detections.jpg")
cv2.imwrite(vis_path, vis_img)
print(f"[OUT] Annotated image     → {vis_path}")

# ─── Save aligned face crops ───────────────────────────────────────────────────
for i, face in enumerate(results):
    if face.aligned_crop is not None:
        # Convert RGB → BGR for OpenCV saving
        crop_bgr = cv2.cvtColor(face.aligned_crop, cv2.COLOR_RGB2BGR)
        crop_path = os.path.join(OUTPUT_DIR, f"aligned_face_{i + 1}.jpg")
        cv2.imwrite(crop_path, crop_bgr)
        print(f"[OUT] Aligned crop {i+1}       → {crop_path}")

# ─── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "═" * 62)
print(f"  ✓  All outputs saved to: {OUTPUT_DIR}")
print("═" * 62 + "\n")
