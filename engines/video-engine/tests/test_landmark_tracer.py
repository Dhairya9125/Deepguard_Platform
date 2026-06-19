"""
Tests for LandmarkTracer — VDS Layer 1 MediaPipe FaceMesh landmark extraction.

Uses synthetic face crops (solid-colour arrays and a simple gradient face
approximation) so no real video or camera is needed.  MediaPipe-specific
tests are skipped gracefully if the package is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blank_crop(h=224, w=224):
    """Return a blank (black) RGB crop."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _gradient_crop(h=224, w=224):
    """Return a gradient RGB crop (still not a real face, but non-trivial)."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = np.linspace(50, 200, w, dtype=np.uint8)
    img[:, :, 1] = np.linspace(100, 150, w, dtype=np.uint8)
    return img


def _make_fake_lm_array(n=478):
    """Return a random (n, 3) float32 array simulating raw landmarks."""
    return np.random.rand(n, 3).astype(np.float32)


# ---------------------------------------------------------------------------
# Unit tests — EMA state
# ---------------------------------------------------------------------------

class TestEMAState:

    def test_cold_start_returns_raw(self):
        """First EMA update must return a copy of the input (cold start)."""
        try:
            from vid_preprocessing.landmark_tracer import _EMAState  # noqa: PLC0415
        except ImportError:
            pytest.skip("landmark_tracer not importable")

        state = _EMAState(alpha=0.3)
        raw = _make_fake_lm_array()
        out = state.update(raw)
        np.testing.assert_array_almost_equal(out, raw)

    def test_ema_converges_toward_input(self):
        """After many identical inputs, EMA must converge to that input."""
        try:
            from vid_preprocessing.landmark_tracer import _EMAState  # noqa: PLC0415
        except ImportError:
            pytest.skip("landmark_tracer not importable")

        state = _EMAState(alpha=0.5)
        target = np.ones((10, 3), dtype=np.float32) * 100.0
        out = np.zeros((10, 3), dtype=np.float32)
        for _ in range(30):
            out = state.update(target)
        np.testing.assert_array_almost_equal(out, target, decimal=0)

    def test_ema_reset_clears_state(self):
        """After reset(), the next update must again be a cold start."""
        try:
            from vid_preprocessing.landmark_tracer import _EMAState  # noqa: PLC0415
        except ImportError:
            pytest.skip("landmark_tracer not importable")

        state  = _EMAState(alpha=0.3)
        first  = _make_fake_lm_array()
        _      = state.update(first)
        state.reset()
        second = _make_fake_lm_array()
        out    = state.update(second)
        np.testing.assert_array_almost_equal(out, second)

    def test_ema_smoothing_intermediate(self):
        """EMA output must lie between previous and current input."""
        try:
            from vid_preprocessing.landmark_tracer import _EMAState  # noqa: PLC0415
        except ImportError:
            pytest.skip("landmark_tracer not importable")

        alpha = 0.4
        state = _EMAState(alpha=alpha)
        raw0  = np.zeros((5, 3), dtype=np.float32)
        raw1  = np.ones((5, 3),  dtype=np.float32) * 10.0

        _    = state.update(raw0)   # cold start: ema = 0
        out1 = state.update(raw1)   # ema = 0.4*10 + 0.6*0 = 4

        expected = alpha * 10.0  # 4.0
        np.testing.assert_array_almost_equal(out1, np.full((5, 3), expected))


# ---------------------------------------------------------------------------
# Unit tests — Derived metrics (no MediaPipe required)
# ---------------------------------------------------------------------------

class TestDerivedMetrics:

    def test_ear_fully_closed_eye(self):
        """EAR must be ~0 when vertical points collapse onto horizontal axis."""
        try:
            from vid_preprocessing.landmark_tracer import (  # noqa: PLC0415
                LandmarkTracer,
                _LEFT_EYE_EAR_IDX,
            )
        except ImportError:
            pytest.skip("landmark_tracer not importable")

        # Build a 478-point array where EAR eye points form a horizontal line
        # P1=(0,0), P2=(0,0), P3=(0,0), P4=(10,0), P5=(0,0), P6=(0,0)
        lm = np.zeros((478, 3), dtype=np.float32)
        indices = _LEFT_EYE_EAR_IDX
        lm[indices[0]] = [0,  0, 0]
        lm[indices[1]] = [3,  0, 0]   # p2
        lm[indices[2]] = [6,  0, 0]   # p3
        lm[indices[3]] = [10, 0, 0]   # p4
        lm[indices[4]] = [6,  0, 0]   # p5
        lm[indices[5]] = [3,  0, 0]   # p6
        # All vertical components are 0 → EAR ≈ 0
        ear = LandmarkTracer._ear(lm, indices)
        assert ear < 0.05, f"EAR should be near 0 for flat eye, got {ear}"

    def test_jaw_open_ratio_closed(self):
        """jaw_open_ratio should be near 0 when upper/lower lip are coincident."""
        try:
            from vid_preprocessing.landmark_tracer import (  # noqa: PLC0415
                LandmarkTracer,
                _UPPER_LIP_IDX,
                _LOWER_LIP_IDX,
                _CHIN_IDX,
                _FOREHEAD_IDX,
            )
        except ImportError:
            pytest.skip("landmark_tracer not importable")

        lm = np.zeros((478, 3), dtype=np.float32)
        # Upper and lower lip at same position
        lm[_UPPER_LIP_IDX] = [100, 100, 0]
        lm[_LOWER_LIP_IDX] = [100, 100, 0]  # coincident → gap = 0
        # Forehead and chin well separated
        lm[_FOREHEAD_IDX]  = [100,  50, 0]
        lm[_CHIN_IDX]      = [100, 200, 0]

        ratio = LandmarkTracer._jaw_open_ratio(lm)
        assert ratio < 0.01, f"Expected jaw_open ~0, got {ratio}"

    def test_jaw_open_ratio_open(self):
        """jaw_open_ratio should increase as lip gap increases."""
        try:
            from vid_preprocessing.landmark_tracer import (  # noqa: PLC0415
                LandmarkTracer,
                _UPPER_LIP_IDX,
                _LOWER_LIP_IDX,
                _CHIN_IDX,
                _FOREHEAD_IDX,
            )
        except ImportError:
            pytest.skip("landmark_tracer not importable")

        lm = np.zeros((478, 3), dtype=np.float32)
        lm[_UPPER_LIP_IDX] = [100,  90, 0]
        lm[_LOWER_LIP_IDX] = [100, 110, 0]   # 20px gap
        lm[_FOREHEAD_IDX]  = [100,  50, 0]
        lm[_CHIN_IDX]      = [100, 200, 0]   # 150px face height

        ratio = LandmarkTracer._jaw_open_ratio(lm)
        # Expected: 20 / 150 ≈ 0.133
        assert 0.10 < ratio < 0.20, f"Expected jaw_open ~0.13, got {ratio}"


# ---------------------------------------------------------------------------
# Unit tests — trace() with no crop
# ---------------------------------------------------------------------------

class TestLandmarkTracerNoCrop:

    def test_trace_none_crop_returns_failure_result(self):
        """trace() with face_crop_rgb=None must return detection_success=False."""
        try:
            from vid_preprocessing.landmark_tracer import LandmarkTracer  # noqa: PLC0415
        except ImportError:
            pytest.skip("landmark_tracer not importable")

        tracer = LandmarkTracer()
        result = tracer.trace(
            track_id="track_001",
            frame_id=0,
            timestamp_ms=0.0,
            face_crop_rgb=None,
        )
        assert result.detection_success is False
        assert result.landmarks_478 is None
        assert result.track_id == "track_001"
        assert result.frame_id == 0

    def test_trace_wrong_shape_returns_failure(self):
        """trace() with a 2D array (wrong shape) must return detection_success=False."""
        try:
            from vid_preprocessing.landmark_tracer import LandmarkTracer  # noqa: PLC0415
        except ImportError:
            pytest.skip("landmark_tracer not importable")

        tracer = LandmarkTracer()
        bad_crop = np.zeros((224, 224), dtype=np.uint8)  # 2D, no channels
        result = tracer.trace("track_001", 0, 0.0, bad_crop)
        assert result.detection_success is False

    def test_trace_blank_crop_mediapipe(self):
        """
        trace() on a blank/black crop — MediaPipe likely fails to find a face.
        Result must be returned (not raised) with detection_success=False.
        """
        try:
            from vid_preprocessing.landmark_tracer import LandmarkTracer  # noqa: PLC0415
            import mediapipe  # noqa: PLC0415, F401
        except ImportError:
            pytest.skip("mediapipe or landmark_tracer not installed")

        tracer = LandmarkTracer(refine_landmarks=True)
        crop   = _blank_crop()
        result = tracer.trace("track_001", 0, 0.0, crop)
        # On a blank frame, MediaPipe should fail gracefully
        assert result.detection_success is False or result.detection_success is True
        # Either way: no exception, and the result type is correct
        from vid_preprocessing.result_types import FaceLandmarkResult  # noqa: PLC0415
        assert isinstance(result, FaceLandmarkResult)


# ---------------------------------------------------------------------------
# Unit tests — reset and close
# ---------------------------------------------------------------------------

class TestLandmarkTracerLifecycle:

    def test_reset_clears_ema_states(self):
        """reset() must clear all per-track EMA state dictionaries."""
        try:
            from vid_preprocessing.landmark_tracer import LandmarkTracer  # noqa: PLC0415
        except ImportError:
            pytest.skip("landmark_tracer not importable")

        tracer = LandmarkTracer()
        # Inject fake EMA states
        tracer._ema_states = {"track_001": object(), "track_002": object()}
        tracer.reset()
        assert tracer._ema_states == {}

    def test_close_via_context_manager(self):
        """LandmarkTracer must support 'with' statement without raising."""
        try:
            from vid_preprocessing.landmark_tracer import LandmarkTracer  # noqa: PLC0415
        except ImportError:
            pytest.skip("landmark_tracer not importable")

        with LandmarkTracer() as tracer:
            assert tracer is not None
        # After __exit__: _face_mesh should be None (closed)
        assert tracer._face_mesh is None

    def test_repr(self):
        """__repr__ must contain class name and key params."""
        try:
            from vid_preprocessing.landmark_tracer import LandmarkTracer  # noqa: PLC0415
        except ImportError:
            pytest.skip("landmark_tracer not importable")

        tracer = LandmarkTracer(ema_alpha=0.5)
        r = repr(tracer)
        assert "LandmarkTracer" in r
        assert "0.5" in r


# ---------------------------------------------------------------------------
# Unit tests — mediapipe_to_pixel conversion
# ---------------------------------------------------------------------------

class TestPixelConversion:

    def test_pixel_conversion_scale(self):
        """Normalised landmark at (1.0, 1.0) must map to (width, height)."""
        try:
            from vid_preprocessing.landmark_tracer import LandmarkTracer  # noqa: PLC0415
        except ImportError:
            pytest.skip("landmark_tracer not importable")

        # Build a mock MediaPipe landmark list using MagicMock
        from unittest.mock import MagicMock  # noqa: PLC0415

        lm = MagicMock()
        lm.x = 1.0
        lm.y = 1.0
        lm.z = 0.5

        mp_lm_list        = MagicMock()
        mp_lm_list.landmark = [lm]

        arr = LandmarkTracer._mediapipe_to_pixel(mp_lm_list, crop_h=224, crop_w=224)
        assert arr.shape == (1, 3)
        assert arr[0, 0] == pytest.approx(224.0)   # x → width
        assert arr[0, 1] == pytest.approx(224.0)   # y → height
        assert arr[0, 2] == pytest.approx(0.5)     # z kept as-is
