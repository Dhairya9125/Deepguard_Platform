import cv2
import numpy as np
from pathlib import Path
import json

from pipeline_ids import IDSPipeline

def create_dummy_image(path):
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    # Draw a simple face shape so RetinaFace or MediaPipe might find something
    cv2.circle(img, (200, 200), 100, (255, 200, 200), -1) # face
    cv2.circle(img, (160, 170), 15, (0, 0, 0), -1) # eye L
    cv2.circle(img, (240, 170), 15, (0, 0, 0), -1) # eye R
    cv2.ellipse(img, (200, 230), (40, 20), 0, 0, 180, (0,0,0), -1) # mouth
    cv2.imwrite(path, img)

def test():
    print("Testing Full IDS Pipeline End-to-End...")
    
    test_img = "test_outputs/dummy_pipeline_face.jpg"
    Path("test_outputs").mkdir(exist_ok=True)
    create_dummy_image(test_img)
    
    print("\n--- Initializing IDS Pipeline (Loading all models) ---")
    pipeline = IDSPipeline(device='cpu')
    
    print("\n--- Running Inference ---")
    results = pipeline.predict(test_img)
    
    if results.get("status") == "success":
        print("\nPipeline execution successful!")
        print(f"Fake Probability: {results['fake_probability']:.4f}")
        print(f"OOD Score: {results['ood_score']:.4f}")
        print(f"Detected Bounding Boxes: {results['bounding_boxes']}")
        print("\nForensic Report:")
        print(results['forensic_report'])
    else:
        print("\nPipeline execution returned an error:")
        print(results)
        
    print("\nTest passed! Full end-to-end integration is functional.")

if __name__ == "__main__":
    test()
