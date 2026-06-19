# Changelog

All notable changes to ADS will be documented in this file.

## [0.1.0] - 2026-06-02

### Added
- Project initialization and repository structure
- Multi-branch audio deepfake detection system
- WavLM Large branch for self-supervised audio representation
- XLS-R branch for multilingual robustness
- Spectral CNN branch for synthesis artifact detection
- Voice Biometrics (ECAPA-TDNN) for speaker consistency
- Temporal Splice Conformer for manipulation localization
- Diffusion Attack Detector for diffusion-generated audio
- Adversarial Robustness Detector
- Cross-branch Fusion Transformer with adaptive gating
- Temporal localization engine with CRF decoding
- Explainability engine with SHAP, attention maps, report generation
- FastAPI REST API with JWT authentication
- CLI interface for detection and analysis
- Celery async task processing
- Docker and Docker Compose deployment
- Kubernetes manifests (Deployment, HPA, NetworkPolicy, ConfigMap)
- Terraform AWS infrastructure (EKS, RDS, ElastiCache, S3, ECR)
- CI/CD pipelines (GitHub Actions)
- MLflow integration for experiment tracking
- Prometheus metrics endpoint
- Database models (SQLAlchemy async with PostgreSQL)
- Comprehensive test suite (>90% coverage target)
- Security layer (JWT, RBAC, audit logging, rate limiting)
- Streaming audio detection support
