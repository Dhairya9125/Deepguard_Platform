# Comprehensive Results Visualization Plan

The deepfake models **already** pinpoint the exact manipulated portions, but the frontend UI is currently ignoring this data! 

Here is exactly what the models currently return:
- **Image**: Returns `bounding_boxes` (x, y, width, height) highlighting the exact regions of the image where spatial anomalies (like blending boundaries or generative artifacts) were found.
- **Audio**: Returns `manipulation_segments` (start time, end time in seconds) pinpointing the exact moments where the voice was synthesized or spliced.
- **Video**: Returns `manipulated_frames`, spatial `manipulated_regions` per frame, and audio desync `audio_timestamps`.

Since the models already do the heavy lifting, we just need to update the platform to visually present this data! 

## Proposed Changes

### 1. Backend (API)
We need a way for the frontend to fetch the original media so it can draw over it.
- **[MODIFY] `apps/api/routers/jobs.py`** (or create a new endpoint):
  - Add a `GET /api/v1/jobs/{job_id}/media` endpoint that returns the saved file (image/video/audio) from the server's `UPLOAD_DIR`.

### 2. Frontend (UI)
- **[MODIFY] `ADS/frontend/src/app/results/[id]/page.tsx`**:
  - Add a dedicated **Media Viewer Section**.
  - **For Images**: Load the image and use an HTML `<canvas>` to draw red bounding boxes over the manipulated regions returned in `job.results.bounding_boxes`.
  - **For Audio**: Implement a custom audio player that renders a timeline, using `job.results.manipulation_segments` to paint the timeline red during deepfake portions.
  - **For Video**: Implement a video player that highlights the timeline (similar to audio) using `audio_timestamps` and `manipulated_frames`.

## User Review Required
> [!IMPORTANT]
> The backend ML installation is still running in the background. Once I implement this plan, you will have a fully comprehensive results page that visually pinpoints the deepfakes! Do you approve of this plan?
