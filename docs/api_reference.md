# DeepGuard API Reference

The DeepGuard FastAPI server exposes endpoints for authentication, media analysis, and job polling.

## Authentication Endpoints (`/api/v1/auth`)

### `POST /register`
Creates a new user.
- **Body**: `{"username": "str", "email": "str", "password": "str"}`
- **Returns**: `UserResponse` (201 Created)

### `POST /login`
Authenticates a user and returns a JWT.
- **Body**: `{"username": "str", "password": "str"}`
- **Returns**: `TokenResponse` (200 OK)

## Analysis Endpoints (`/api/v1/analyze`)

> **Note:** Media endpoints accept `multipart/form-data` uploads.

### `POST /image`
Synchronously analyzes an uploaded image.
- **Form Data**: `file` (image/jpeg, image/png)
- **Returns**: `ImageAnalysisResponse` containing authenticity score and heatmap paths.

### `POST /audio`
Asynchronously analyzes an uploaded audio file.
- **Form Data**: `file` (audio/wav, audio/flac)
- **Returns**: `{"job_id": "uuid", "message": "str"}` (202 Accepted)

### `POST /video`
Asynchronously analyzes an uploaded video file.
- **Form Data**: `file` (video/mp4)
- **Returns**: `{"job_id": "uuid", "message": "str"}` (202 Accepted)

## Job Polling (`/api/v1/jobs`)

### `GET /{job_id}`
Retrieves the current status of an async analysis job.
- **Returns**: `JobResponse` detailing status (`PENDING`, `RUNNING`, `COMPLETED`, `FAILED`) and results if completed.
