import cv2
import numpy as np
from PIL import Image
import wave
import struct
import os

def generate_image(filename="test_image.jpg"):
    # Create a 224x224 RGB image
    img = Image.new('RGB', (224, 224), color = (73, 109, 137))
    img.save(filename)
    print(f"Created {filename}")

def generate_audio(filename="test_audio.wav"):
    # Create a 1 second 16kHz sine wave
    sample_rate = 16000
    obj = wave.open(filename, 'w')
    obj.setnchannels(1)
    obj.setsampwidth(2)
    obj.setframerate(sample_rate)
    
    for i in range(sample_rate):
        value = int(32767.0 * np.sin(2.0 * np.pi * 440.0 * i / sample_rate))
        data = struct.pack('<h', value)
        obj.writeframesraw(data)
    obj.close()
    print(f"Created {filename}")

def generate_video(filename="test_video.mp4"):
    # Create a 1 second 30fps black video
    width, height = 224, 224
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for _ in range(fps):
        out.write(frame)
    out.release()
    print(f"Created {filename}")

if __name__ == "__main__":
    os.makedirs("tests", exist_ok=True)
    generate_image("tests/test_image.jpg")
    generate_audio("tests/test_audio.wav")
    generate_video("tests/test_video.mp4")
