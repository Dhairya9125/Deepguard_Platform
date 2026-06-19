"""
DeepGuard Platform — IDS Layer 1: Face Alignment (MediaPipe) Test Script

Run from the image-engine directory:
    cd engines/image-engine
    python -X utf8 test_face_aligner.py

What this script does:
  1. Loads the cached sample face image (downloaded by test_face_detector.py).
  2. Runs RetinaFace detection (FaceDetector) to get bounding boxes.
  3. Runs MediaPipe Face Mesh alignment (FaceAligner) for dense 468-pt landmarks.
  4. Prints all key points, roll angle, landmark count, and detection score.
  5. Saves:
       aligned_mediapipe.jpg  — the MediaPipe-aligned 224x224 crop
       landmarks_vis.jpg      — full image with 468 mesh + 5 key anchors drawn
       alignment_result.json  — serialised result metadata
  6. Compares RetinaFace vs MediaPipe eye-centre estimates side by side.
"""

import sys
import os
import logging
import json

_ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ENGINE_DIR)

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

from img_preprocessing.face_detector import FaceDetector
from img_preprocessing.face_aligner  import FaceAligner, AlignmentResult

# ─── Paths ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR  = os.path.join(_ENGINE_DIR, "test_outputs")
SAMPLE_PATH = os.path.join(OUTPUT_DIR, "sample_face.jpg")
os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(SAMPLE_PATH):
    sys.exit(
        f"[ERROR] Sample image not found: {SAMPLE_PATH}\n"
        "        Run test_face_detector.py first to download it."
    )

# ─── Load image ────────────────────────────────────────────────────────────────
img_bgr = cv2.imread(SAMPLE_PATH)
if img_bgr is None:
    sys.exit(f"[ERROR] Could not read: {SAMPLE_PATH}")

print(f"\n[INFO] Image loaded: {img_bgr.shape}  (H x W x BGR)")

# ─── Step 1: RetinaFace Detection ──────────────────────────────────────────────
print("\n" + "=" * 62)
print("  Step 1  |  RetinaFace Detection (InsightFace)")
print("=" * 62)

detector = FaceDetector(
    confidence_threshold=0.50,
    target_size=(224, 224),
    align=True,
    model_pack="buffalo_l",
)
detected_faces = detector.detect_and_align(img_bgr)

if not detected_faces:
    sys.exit("[ERROR] RetinaFace found no faces. Check the image.")

print(f"\n  RetinaFace found {len(detected_faces)} face(s).\n")
for face in detected_faces:
    print(f"  face_id     : {face.face_id}")
    print(f"  score       : {face.score:.4f}")
    print(f"  bbox        : {face.bbox}")
    print(f"  eye_right   : {tuple(round(v,1) for v in face.landmarks.right_eye)}  (RetinaFace 5-pt)")
    print(f"  eye_left    : {tuple(round(v,1) for v in face.landmarks.left_eye)}   (RetinaFace 5-pt)")
    print()

# ─── Step 2: MediaPipe Face Mesh Alignment ─────────────────────────────────────
print("=" * 62)
print("  Step 2  |  MediaPipe Face Mesh Alignment (468 landmarks)")
print("=" * 62 + "\n")

aligner = FaceAligner(
    target_size=(224, 224),
    padding=0.30,
)

# Align all detected faces
bboxes  = [f.bbox for f in detected_faces]
results = aligner.align_batch(img_bgr, bboxes)

# ─── Print results ─────────────────────────────────────────────────────────────
for i, (face, result) in enumerate(zip(detected_faces, results)):
    if result is None:
        print(f"  [WARNING] MediaPipe failed for face {i+1} — skipping.")
        continue

    print(f"  Face {i+1}")
    print(f"  {'─' * 56}")
    print(f"  Landmarks detected : {len(result.landmarks_2d)} points "
          f"({'iris-refined' if len(result.landmarks_2d) >= 478 else 'standard'})")
    print(f"  Detection score    : {result.detection_score:.4f}")
    print(f"  In-plane roll      : {result.roll_deg:+.2f} degrees  "
          f"({'level' if abs(result.roll_deg) < 2 else 'tilted'})")
    print(f"  Aligned crop shape : {result.aligned_crop.shape}")
    print()
    print(f"  Key anchor points (MediaPipe):")
    print(f"    eye_left   : {tuple(round(v,1) for v in result.eye_left)}")
    print(f"    eye_right  : {tuple(round(v,1) for v in result.eye_right)}")
    print(f"    nose_tip   : {tuple(round(v,1) for v in result.nose_tip)}")
    print(f"    mouth_left : {tuple(round(v,1) for v in result.mouth_left)}")
    print(f"    mouth_right: {tuple(round(v,1) for v in result.mouth_right)}")
    print()
    print(f"  Comparison  (RetinaFace 5-pt  vs  MediaPipe 468-pt):")
    rf_re = face.landmarks.right_eye
    rf_le = face.landmarks.left_eye
    mp_re = result.eye_right
    mp_le = result.eye_left
    print(f"    eye_right  RF={tuple(round(v,1) for v in rf_re)}  "
          f"MP={tuple(round(v,1) for v in mp_re)}  "
          f"diff={round(np.linalg.norm(np.array(rf_re)-np.array(mp_re)),1)}px")
    print(f"    eye_left   RF={tuple(round(v,1) for v in rf_le)}  "
          f"MP={tuple(round(v,1) for v in mp_le)}  "
          f"diff={round(np.linalg.norm(np.array(rf_le)-np.array(mp_le)),1)}px")
    print(f"  Interocular distance: {result.interocular_distance:.1f}px")
    print()

# ─── Save outputs ──────────────────────────────────────────────────────────────
for i, result in enumerate(results):
    if result is None:
        continue

    # Aligned crop
    crop_path = os.path.join(OUTPUT_DIR, f"aligned_mediapipe_{i+1}.jpg")
    crop_bgr  = cv2.cvtColor(result.aligned_crop, cv2.COLOR_RGB2BGR)
    cv2.imwrite(crop_path, crop_bgr)
    print(f"[OUT] MediaPipe aligned crop {i+1} -> {crop_path}")

    # JSON metadata
    json_path = os.path.join(OUTPUT_DIR, f"alignment_result_{i+1}.json")
    with open(json_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"[OUT] Alignment metadata     -> {json_path}")

# Visualization — full mesh + key points overlaid
vis = aligner.visualize(
    img_bgr,
    [r for r in results if r is not None],
    draw_mesh=True,
    draw_keypoints=True,
    draw_roll=True,
)
vis_path = os.path.join(OUTPUT_DIR, "landmarks_vis.jpg")
cv2.imwrite(vis_path, vis)
print(f"[OUT] Landmark visualization -> {vis_path}")

# Comparison grid: RetinaFace crop | MediaPipe crop
if detected_faces[0].aligned_crop is not None and results[0] is not None:
    rf_crop_rgb = detected_faces[0].aligned_crop        # RGB
    mp_crop_rgb = results[0].aligned_crop               # RGB
    rf_crop_bgr = cv2.cvtColor(rf_crop_rgb, cv2.COLOR_RGB2BGR)
    mp_crop_bgr = cv2.cvtColor(mp_crop_rgb, cv2.COLOR_RGB2BGR)

    # Add labels
    for img, label in [(rf_crop_bgr, "RetinaFace 5-pt"), (mp_crop_bgr, "MediaPipe 468-pt")]:
        cv2.putText(img, label, (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

    # Stack side by side with a thin white divider
    divider = np.full((224, 4, 3), 255, dtype=np.uint8)
    comparison = np.hstack([rf_crop_bgr, divider, mp_crop_bgr])
    cmp_path = os.path.join(OUTPUT_DIR, "crop_comparison.jpg")
    cv2.imwrite(cmp_path, comparison)
    print(f"[OUT] Crop comparison        -> {cmp_path}")

aligner.close()

print("\n" + "=" * 62)
print(f"  All outputs saved to: {OUTPUT_DIR}")
print("=" * 62 + "\n")
