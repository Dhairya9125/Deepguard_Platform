# DeepGuard Platform

DeepGuard is an enterprise-grade multimodal deepfake detection platform. It uses state-of-the-art AI to detect synthetic manipulations in audio, video, and image media.

## Architecture

DeepGuard utilizes a **Fusion Engine** architecture, integrating specialized sub-engines:
- **Audio Engine (ADS)**: Detects voice cloning and adversarial noise.
- **Image Engine (IDS)**: Detects spatial anomalies and GAN artifacts.
- **Video Engine (VDS)**: Analyzes temporal anomalies across sequential frames.
- **Fusion Engine**: Cross-attention transformer that blends the individual modalities into a single overarching confidence score and deepfake verdict.

The platform is designed to be **lightweight for local development (4GB RAM requirement)**:
- **FastAPI** backend with built-in `BackgroundTasks` instead of heavy message brokers like Celery/RabbitMQ.
- **SQLite (or Neon Serverless Postgres)** instead of Dockerized PostgreSQL.
- **File-based MLflow** instead of an active tracking server.
- **Next.js** frontend with a premium dark-mode aesthetic utilizing TailwindCSS, Framer Motion, and Lucide React.

## Getting Started

### 1. Running the API Backend

Navigate to the root directory and ensure the Python virtual environment is activated. We provide a helper script for Windows:

```powershell
# 1. Setup the environment and install dependencies
.\infra\deployment\setup_env.ps1

# 2. Start the FastAPI server using Uvicorn
.\infra\deployment\start_dev.ps1
# Alternatively: uvicorn apps.api.main:app --reload --port 8000
```
The API documentation will be available at `http://127.0.0.1:8000/docs`.

### 2. Running the Frontend Dashboard

The Next.js frontend provides a comprehensive suite of tools including an overview Dashboard, History view, Settings, and Pinpoint Analysis results. It also features a custom interactive cursor and fully stabilized glassmorphic UI.

```powershell
# Navigate to the frontend directory
cd ADS/frontend

# Install node dependencies
npm install

# Start the Next.js development server
npm run dev
```
The frontend application will be available at `http://localhost:3000`.

## Documentation
- Detailed API specs are located in `docs/api_reference.md`.
- Architecture and system diagrams are located in `docs/architecture.md`.
- ML benchmarks and logs are tracked in the `research/` directory.

## Project Structure
- `apps/api/`: FastAPI application, routers, schemas, and core services.
- `engines/`: Core PyTorch deepfake detection engines (Audio, Image, Video, Fusion).
- `ADS/frontend/`: Next.js web application.
- `ml/`: Model training utilities, MLflow setup, and dataset loaders.
- `infra/`: Local deployment and environment setup scripts.