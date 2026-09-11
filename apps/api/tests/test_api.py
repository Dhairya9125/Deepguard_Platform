"""
Integration tests for the Unified REST API using FastAPI TestClient.
"""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from apps.api.main import app
from apps.api.core.config import settings

client = TestClient(app)

# We will mock the orchestrator run_vds_pipeline to avoid running heavy DL models in API tests
import apps.api.routers.video as video_router

def mock_run_vds_pipeline(job_id: str, video_path: str):
    """
    Mock pipeline that simulates processing and immediately resolves the job as COMPLETED.
    """
    import asyncio
    from apps.api.services.vds_pipeline import JOB_STORE
    
    # Simulate processing delay
    JOB_STORE[job_id]["status"] = "PROCESSING"
    
    # Simulate successful completion
    output_dir = settings.ARTIFACTS_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a dummy forensic_summary.json
    results = {
        "overall_verdict": "FAKE",
        "overall_confidence": 0.99,
        "duration_s": 5.0,
        "manipulated_frames": [0, 1, 2],
        "audio_timestamps": [],
        "tracks": {}
    }
    
    with open(output_dir / "forensic_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f)
        
    # Write a dummy image to verify static mounting
    with open(output_dir / "dummy_graph.png", "wb") as f:
        f.write(b"PNG IMAGE DATA")
        
    JOB_STORE[job_id] = {
        "status": "COMPLETED",
        "message": "Analysis completed successfully.",
        "results": results
    }

# Patch the pipeline in the router
video_router.run_vds_pipeline = mock_run_vds_pipeline


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome to DeepGuard VDS API" in response.json()["message"]


def test_analyze_video_endpoint():
    # Create a dummy video file
    dummy_video_content = b"fake video bytes"
    
    # Post it to the endpoint
    response = client.post(
        f"{settings.API_V1_STR}/analyze/video",
        files={"file": ("test_video.mp4", dummy_video_content, "video/mp4")}
    )
    
    assert response.status_code == 202
    data = response.json()
    assert "job_id" in data
    assert "successfully" in data["message"]
    
    job_id = data["job_id"]
    
    # Immediately check job status. Note: BackgroundTasks run *after* the response is returned.
    # However, TestClient executes BackgroundTasks immediately synchronously.
    # So the status should be COMPLETED immediately.
    status_response = client.get(f"{settings.API_V1_STR}/jobs/{job_id}")
    assert status_response.status_code == 200
    status_data = status_response.json()
    
    assert status_data["job_id"] == job_id
    assert status_data["status"] == "COMPLETED"
    assert status_data["results"]["overall_verdict"] == "FAKE"
    
    # Test Artifacts Mounting
    artifact_resp = client.get(f"/artifacts/{job_id}/forensic_summary.json")
    assert artifact_resp.status_code == 200
    
    artifact_img = client.get(f"/artifacts/{job_id}/dummy_graph.png")
    assert artifact_img.status_code == 200
    assert artifact_img.content == b"PNG IMAGE DATA"


def test_get_job_not_found():
    response = client.get(f"{settings.API_V1_STR}/jobs/non-existent-job-id")
    assert response.status_code == 404
    assert response.json()["detail"] == "Job ID not found."
