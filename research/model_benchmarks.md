# DeepGuard Target Model Benchmarks

This table defines the target baseline metrics for DeepGuard's individual detection branches against established datasets.

| Engine | Modality | Dataset | Target Accuracy | Target EER |
|--------|----------|---------|-----------------|------------|
| ADS    | Audio    | ASVspoof 2019 (LA) | > 92.0% | < 5.0% |
| ADS    | Audio    | ASVspoof 2021 (DF) | > 88.0% | < 8.0% |
| IDS    | Image    | FaceForensics++ (Raw) | > 95.0% | < 3.0% |
| IDS    | Image    | Celeb-DF v2 | > 85.0% | < 10.0% |
| VDS    | Video    | FaceForensics++ (HQ) | > 93.0% | < 4.0% |

> **Note**: Metrics such as Equal Error Rate (EER) are critical for biometric and deepfake detection systems. Lower EER indicates a better performing model.
