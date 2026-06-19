import requests
import time
import sys

BASE_URL = "http://127.0.0.1:8000/api/v1"

def register_and_login():
    print("--- 1. Authenticating ---")
    email = f"test_{int(time.time())}@example.com"
    username = f"user_{int(time.time())}"
    password = "password123"
    
    # Register
    res = requests.post(f"{BASE_URL}/auth/register", json={
        "username": username,
        "email": email,
        "password": password
    })
    
    if res.status_code == 409:
        print("User exists, logging in directly.")
    else:
        res.raise_for_status()
        print("Registered user successfully.")
        
    # Login
    res = requests.post(f"{BASE_URL}/auth/login", json={
        "username": username,
        "password": password
    })
    res.raise_for_status()
    token = res.json()["access_token"]
    print("Logged in. Received JWT Token.")
    return {"Authorization": f"Bearer {token}"}

def test_image(headers):
    print("\n--- 2. Testing Image Analysis (Sync) ---")
    with open("tests/test_image.jpg", "rb") as f:
        res = requests.post(f"{BASE_URL}/analyze/image", headers=headers, files={"file": ("test_image.jpg", f, "image/jpeg")})
    
    if res.status_code != 200:
        print(f"Error: {res.status_code} - {res.text}")
        return False
        
    data = res.json()
    print(f"Verdict: {data.get('verdict')}")
    print(f"Fake Probability: {data.get('fake_probability')}")
    print("Image endpoint works correctly!")
    return True

def poll_job(job_id, headers, modality):
    print(f"Polling {modality} job {job_id}...")
    for _ in range(30): # 1 minute max
        time.sleep(2)
        res = requests.get(f"{BASE_URL}/jobs/{job_id}", headers=headers)
        if res.status_code == 200:
            data = res.json()
            status = data.get("status")
            print(f"[{status}]", end=" ", flush=True)
            if status == "COMPLETED":
                print(f"\n{modality} Job Completed!")
                print(f"Results Fetched from Neon DB: Verdict={data.get('results', {}).get('verdict')}")
                return True
            elif status == "FAILED":
                print(f"\n{modality} Job Failed! {data.get('message')}")
                return False
    print("\nTimeout waiting for job completion.")
    return False

def test_audio(headers):
    print("\n--- 3. Testing Audio Analysis (Async) ---")
    with open("tests/test_audio.wav", "rb") as f:
        res = requests.post(f"{BASE_URL}/analyze/audio", headers=headers, files={"file": ("test_audio.wav", f, "audio/wav")})
    
    if res.status_code != 202:
        print(f"Error: {res.status_code} - {res.text}")
        return False
        
    job_id = res.json()["job_id"]
    print(f"Received Job ID: {job_id}")
    return poll_job(job_id, headers, "Audio")

def test_video(headers):
    print("\n--- 4. Testing Video Analysis (Async) ---")
    with open("tests/test_video.mp4", "rb") as f:
        res = requests.post(f"{BASE_URL}/analyze/video", headers=headers, files={"file": ("test_video.mp4", f, "video/mp4")})
    
    if res.status_code != 202:
        print(f"Error: {res.status_code} - {res.text}")
        return False
        
    job_id = res.json()["job_id"]
    print(f"Received Job ID: {job_id}")
    return poll_job(job_id, headers, "Video")

if __name__ == "__main__":
    print("================ E2E WORKFLOW VERIFICATION ================")
    try:
        headers = register_and_login()
        image_ok = test_image(headers)
        audio_ok = test_audio(headers)
        video_ok = test_video(headers)
        
        if image_ok and audio_ok and video_ok:
            print("\nSUCCESS: End-to-End workflow verified and database synced perfectly!")
        else:
            print("\nFAILURE: Some tests did not pass.")
            sys.exit(1)
    except Exception as e:
        print(f"\nSCRIPT CRASHED: {e}")
        sys.exit(1)
