# Shared Core

Cross-engine shared library for the DeepGuard Platform.

## Modules

| Module | Purpose |
|---|---|
| `result_types.py` | `DeepfakeVerdict`, `MediaModality`, `AnalysisResult`, `EngineResult` enums/dataclasses |
| `base_pipeline.py` | `BasePipeline` abstract class — all engines implement this |
| `exceptions.py` | Shared exception hierarchy (`DeepGuardError` and subclasses) |

## Usage

```python
from core.shared-core import DeepfakeVerdict, AnalysisResult, BasePipeline
```

## Engine Implementations

- `engines/image-engine/pipeline_ids.py` → `IDSPipeline(BasePipeline)`
- `engines/video-engine/` → `VDSPipeline(BasePipeline)` (planned)
- `engines/audio-engine/` → `ADSPipeline(BasePipeline)`
