"""
DeepGuard Platform — Shared Exception Hierarchy
Module  : core.shared-core.exceptions

All custom exceptions used across the platform derive from DeepGuardError.
This allows callers to catch all platform errors with a single except clause.
"""


class DeepGuardError(Exception):
    """Base exception for all DeepGuard Platform errors."""


class EngineNotReadyError(DeepGuardError):
    """Raised when a detection engine is called before it has been loaded."""
    def __init__(self, engine_name: str) -> None:
        super().__init__(
            f"Engine '{engine_name}' is not loaded. Call .load() or .ensure_loaded() first."
        )


class UnsupportedMediaError(DeepGuardError):
    """Raised when a pipeline receives a media file it cannot process."""
    def __init__(self, file_path: str, reason: str = "") -> None:
        msg = f"Unsupported media file: '{file_path}'"
        if reason:
            msg += f" — {reason}"
        super().__init__(msg)


class AnalysisError(DeepGuardError):
    """Raised when the detection pipeline fails during inference."""


class ModelLoadError(DeepGuardError):
    """Raised when a model checkpoint or weight file cannot be loaded."""
    def __init__(self, model_name: str, path: str, reason: str = "") -> None:
        msg = f"Failed to load model '{model_name}' from '{path}'"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)
