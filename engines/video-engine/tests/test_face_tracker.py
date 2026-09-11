"""
Tests for FaceTracker — VDS Layer 1 DeepSORT multi-face tracking.

Uses blank RGB numpy arrays as synthetic frames so no real video or
model weights are required for the structural tests. Tests that require
the DNN / DeepSORT models are skipped gracefully if dependencies are absent.
"""

from __future__ import annotations

from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_blank_frame(height=240, width=320, color=(80, 120, 200)):
    """Return a plain-colour RGB uint8 numpy array."""
    return np.full((height, width, 3), color, dtype=np.uint8)


def _make_frame_packet(frame_id: int, rgb=None):
    """Return a minimal FramePacket for testing."""
    try:
        from preprocessing.result_types import FramePacket  # noqa: PLC0415
    except ImportError:
        pytest.skip("result_types not importable")
    return FramePacket(
        frame_id=frame_id,
        timestamp_ms=frame_id * 125.0,   # 8 fps spacing
        rgb_array=rgb if rgb is not None else _make_blank_frame(),
    )


# ---------------------------------------------------------------------------
# Unit tests — no real models
# ---------------------------------------------------------------------------

class TestFaceTrackerInit:

    def test_valid_backends(self):
        """FaceTracker accepts 'dnn' and 'retinaface' as backend values."""
        try:
            from preprocessing.face_tracker import FaceTracker  # noqa: PLC0415
        except ImportError:
            pytest.skip("face_tracker not importable")
        FaceTracker(backend="dnn")
        FaceTracker(backend="retinaface")

    def test_invalid_backend_raises(self):
        """FaceTracker must raise ValueError for unknown backend."""
        try:
            from preprocessing.face_tracker import FaceTracker  # noqa: PLC0415
        except ImportError:
            pytest.skip("face_tracker not importable")
        with pytest.raises(ValueError, match="backend must be"):
            FaceTracker(backend="yolo_ultra_v99")

    def test_repr(self):
        """__repr__ must mention key parameters."""
        try:
            from preprocessing.face_tracker import FaceTracker  # noqa: PLC0415
        except ImportError:
            pytest.skip("face_tracker not importable")
        tracker = FaceTracker(backend="dnn", max_age=15)
        r = repr(tracker)
        assert "FaceTracker" in r
        assert "dnn" in r

    def test_reset_clears_state(self):
        """reset() must clear the track ID map and reset the counter."""
        try:
            from preprocessing.face_tracker import FaceTracker  # noqa: PLC0415
        except ImportError:
            pytest.skip("face_tracker not importable")
        tracker = FaceTracker(backend="dnn")
        # Manually inject a fake track ID mapping
        tracker._track_id_map     = {1: "track_001", 2: "track_002"}
        tracker._next_stable_id   = 3
        tracker.reset()
        assert tracker._track_id_map  == {}
        assert tracker._next_stable_id == 1
        assert tracker._tracker is None


class TestFaceTrackerStableIDs:

    def test_same_deepsort_id_gives_same_stable_id(self):
        """The same DeepSORT integer ID must always resolve to the same string."""
        try:
            from preprocessing.face_tracker import FaceTracker  # noqa: PLC0415
        except ImportError:
            pytest.skip("face_tracker not importable")
        tracker = FaceTracker(backend="dnn")
        id_a = tracker._get_stable_id(42)
        id_b = tracker._get_stable_id(42)
        assert id_a == id_b

    def test_different_deepsort_ids_give_different_stable_ids(self):
        """Two different DeepSORT IDs must map to different stable strings."""
        try:
            from preprocessing.face_tracker import FaceTracker  # noqa: PLC0415
        except ImportError:
            pytest.skip("face_tracker not importable")
        tracker = FaceTracker(backend="dnn")
        id_a = tracker._get_stable_id(1)
        id_b = tracker._get_stable_id(2)
        assert id_a != id_b

    def test_stable_id_format(self):
        """Stable IDs must be zero-padded 'track_XXX' strings."""
        try:
            from preprocessing.face_tracker import FaceTracker  # noqa: PLC0415
        except ImportError:
            pytest.skip("face_tracker not importable")
        tracker = FaceTracker(backend="dnn")
        sid = tracker._get_stable_id(99)
        assert sid.startswith("track_")
        numeric_part = sid.replace("track_", "")
        assert numeric_part.isdigit()
        assert len(numeric_part) == 3   # zero-padded to 3 digits


class TestFaceTrackerNullFrame:

    def test_update_with_no_rgb_returns_empty(self):
        """update() on a FramePacket with rgb_array=None must return []."""
        try:
            from preprocessing.face_tracker import FaceTracker  # noqa: PLC0415
            from preprocessing.result_types  import FramePacket  # noqa: PLC0415
        except ImportError:
            pytest.skip("modules not importable")

        tracker = FaceTracker(backend="dnn")
        fp = FramePacket(frame_id=0, timestamp_ms=0.0, rgb_array=None)
        result = tracker.update(fp)
        assert result == []
        assert fp.tracked_faces == []


class TestFaceTrackerCropDNN:

    def test_crop_dnn_returns_target_size(self):
        """_crop_dnn must return an array of exactly target_size."""
        try:
            from preprocessing.face_tracker import FaceTracker  # noqa: PLC0415
        except ImportError:
            pytest.skip("face_tracker not importable")

        rgb = _make_blank_frame(480, 640)
        # Simulated bbox well inside the frame
        bbox = (50, 50, 200, 200)
        target = (112, 112)
        crop = FaceTracker._crop_dnn(rgb, bbox, target)
        assert crop.shape == (112, 112, 3)

    def test_crop_dnn_clamps_padding(self):
        """_crop_dnn must not raise when padded bbox exceeds frame boundary."""
        try:
            from preprocessing.face_tracker import FaceTracker  # noqa: PLC0415
        except ImportError:
            pytest.skip("face_tracker not importable")

        rgb = _make_blank_frame(100, 100)
        # bbox very near the edge
        bbox = (80, 80, 100, 100)
        crop = FaceTracker._crop_dnn(rgb, bbox, (224, 224))
        assert crop.shape == (224, 224, 3)


class TestFaceTrackerIoUMatching:

    def test_match_track_to_face_high_iou(self):
        """_match_track_to_face must return the face with highest IoU above threshold."""
        try:
            from preprocessing.face_tracker import FaceTracker  # noqa: PLC0415
        except ImportError:
            pytest.skip("face_tracker not importable")

        # Create a mock face result with a known bbox
        mock_face = MagicMock()
        mock_face.bbox = (10, 10, 110, 110)   # 100×100 box

        track_bbox = (15, 15, 105, 105)   # ~81% IoU with mock_face.bbox
        result = FaceTracker._match_track_to_face(track_bbox, [mock_face], iou_threshold=0.3)
        assert result is mock_face

    def test_match_track_to_face_low_iou_returns_none(self):
        """_match_track_to_face must return None when best IoU < threshold."""
        try:
            from preprocessing.face_tracker import FaceTracker  # noqa: PLC0415
        except ImportError:
            pytest.skip("face_tracker not importable")

        mock_face = MagicMock()
        mock_face.bbox = (200, 200, 300, 300)   # far from track

        track_bbox = (10, 10, 50, 50)   # ~0% IoU
        result = FaceTracker._match_track_to_face(track_bbox, [mock_face], iou_threshold=0.3)
        assert result is None


class TestFaceTrackerProcessVideo:

    def test_process_video_populates_tracked_faces(self):
        """
        process_video must populate tracked_faces on each FramePacket.
        We mock the DeepSORT tracker to return zero confirmed tracks
        (blank video = no real faces) and verify no exception is raised.
        """
        try:
            from preprocessing.face_tracker import FaceTracker  # noqa: PLC0415
            from preprocessing.result_types  import FramePacket  # noqa: PLC0415
        except ImportError:
            pytest.skip("modules not importable")

        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort  # noqa: PLC0415
        except ImportError:
            pytest.skip("deep-sort-realtime not installed")

        frames = [_make_frame_packet(i) for i in range(5)]
        tracker = FaceTracker(backend="dnn", min_hits=1)

        # Patch the DNN detector to return no detections
        with patch.object(
            tracker._get_dnn_detector().__class__, "detect", return_value=[]
        ):
            tracker.process_video(frames)

        # Each frame should have been processed (tracked_faces is a list, not None)
        for fp in frames:
            assert isinstance(fp.tracked_faces, list)
