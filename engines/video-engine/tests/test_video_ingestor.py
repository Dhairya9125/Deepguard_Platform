"""
Tests for VideoIngestor — VDS Layer 1 Frame & Audio Extraction.

Uses a synthetic blank video generated in-memory via OpenCV VideoWriter
so no real video file is required for CI.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_synthetic_video(
    path: Path,
    fps: float = 25.0,
    duration_s: float = 3.0,
    width: int = 320,
    height: int = 240,
    color: tuple = (80, 120, 200),
    with_audio: bool = False,
) -> None:
    """
    Write a short solid-colour video to `path` using OpenCV VideoWriter.
    NOTE: OpenCV VideoWriter does not support audio — the produced file
    is video-only. This is intentional for testing has_audio=False path.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    n_frames = int(fps * duration_s)
    frame = np.full((height, width, 3), color, dtype=np.uint8)
    for _ in range(n_frames):
        writer.write(frame)
    writer.release()


@pytest.fixture(scope="module")
def synthetic_video_dir(tmp_path_factory):
    """Create a temporary directory with a synthetic video file."""
    tmp = tmp_path_factory.mktemp("video_ingestor_test")
    video_path = tmp / "test_video.mp4"
    _make_synthetic_video(video_path, fps=25.0, duration_s=3.0)
    return tmp, video_path


@pytest.fixture(scope="module")
def ingestor():
    """Instantiate VideoIngestor — this validates FFmpeg availability."""
    pytest.importorskip("cv2",       reason="opencv-python not installed")
    try:
        from vid_preprocessing.video_ingestor import VideoIngestor  # noqa: PLC0415
    except RuntimeError as exc:
        pytest.skip(f"FFmpeg not available: {exc}")
    return VideoIngestor(target_fps=8.0, save_frames=False)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestVideoIngestorBasic:

    def test_ffmpeg_validation(self):
        """FFmpeg presence is checked at import / instantiation time."""
        try:
            from vid_preprocessing.video_ingestor import VideoIngestor  # noqa: PLC0415
            _ = VideoIngestor()
        except RuntimeError as exc:
            # Expected if FFmpeg is not installed in CI — mark as skip, not fail
            pytest.skip(f"FFmpeg not installed: {exc}")
        except ImportError:
            pytest.skip("video_ingestor module not importable")

    def test_ingest_returns_tuple(self, ingestor, synthetic_video_dir, tmp_path):
        """ingest() must return (VideoMetadata, List[FramePacket])."""
        _, video_path = synthetic_video_dir
        meta, frames = ingestor.ingest(video_path, workspace_dir=tmp_path / "ws")
        assert meta is not None
        assert isinstance(frames, list)

    def test_frame_count_approx(self, ingestor, synthetic_video_dir, tmp_path):
        """At 8 fps, a 3-second video should yield ~24 frames (±2 tolerance)."""
        _, video_path = synthetic_video_dir
        meta, frames = ingestor.ingest(video_path, workspace_dir=tmp_path / "ws2")
        # 3 s × 8 fps = 24 frames expected
        assert 20 <= len(frames) <= 28, f"Unexpected frame count: {len(frames)}"

    def test_frame_packet_fields(self, ingestor, synthetic_video_dir, tmp_path):
        """Each FramePacket must have frame_id, timestamp_ms, and rgb_array."""
        _, video_path = synthetic_video_dir
        _, frames = ingestor.ingest(video_path, workspace_dir=tmp_path / "ws3")
        fp = frames[0]
        assert fp.frame_id == 0
        assert fp.timestamp_ms >= 0.0
        assert fp.rgb_array is not None
        assert fp.rgb_array.ndim == 3
        assert fp.rgb_array.shape[2] == 3  # RGB channels

    def test_frame_ids_sequential(self, ingestor, synthetic_video_dir, tmp_path):
        """frame_id values must be 0, 1, 2, … without gaps."""
        _, video_path = synthetic_video_dir
        _, frames = ingestor.ingest(video_path, workspace_dir=tmp_path / "ws4")
        ids = [fp.frame_id for fp in frames]
        assert ids == list(range(len(frames)))

    def test_timestamps_monotone(self, ingestor, synthetic_video_dir, tmp_path):
        """Timestamps must be strictly increasing."""
        _, video_path = synthetic_video_dir
        _, frames = ingestor.ingest(video_path, workspace_dir=tmp_path / "ws5")
        timestamps = [fp.timestamp_ms for fp in frames]
        for i in range(1, len(timestamps)):
            assert timestamps[i] > timestamps[i - 1], \
                f"Non-monotone timestamps at index {i}: {timestamps[i-1]} ≥ {timestamps[i]}"

    def test_rgb_array_shape(self, ingestor, synthetic_video_dir, tmp_path):
        """Extracted frames must match the video resolution."""
        _, video_path = synthetic_video_dir
        meta, frames = ingestor.ingest(video_path, workspace_dir=tmp_path / "ws6")
        for fp in frames:
            h, w, c = fp.rgb_array.shape
            assert c == 3
            assert h == meta.height
            assert w == meta.width

    def test_no_audio_flag(self, ingestor, synthetic_video_dir, tmp_path):
        """Synthetic video (no audio stream) must report has_audio=False."""
        _, video_path = synthetic_video_dir
        meta, _ = ingestor.ingest(video_path, workspace_dir=tmp_path / "ws7")
        # OpenCV VideoWriter produces video-only files
        assert meta.has_audio is False
        assert meta.audio_path is None

    def test_metadata_fields(self, ingestor, synthetic_video_dir, tmp_path):
        """VideoMetadata must have valid resolution and duration fields."""
        _, video_path = synthetic_video_dir
        meta, frames = ingestor.ingest(video_path, workspace_dir=tmp_path / "ws8")
        assert meta.width == 320
        assert meta.height == 240
        assert meta.duration_s > 0.0
        assert meta.native_fps > 0.0
        assert meta.target_fps == 8.0
        assert meta.total_frames_extracted == len(frames)


class TestVideoIngestorSaveFrames:

    def test_save_frames_creates_jpegs(self, synthetic_video_dir, tmp_path):
        """save_frames=True must create JPEG files in workspace/frames/."""
        pytest.importorskip("cv2", reason="opencv-python not installed")
        try:
            from vid_preprocessing.video_ingestor import VideoIngestor  # noqa: PLC0415
        except (RuntimeError, ImportError) as exc:
            pytest.skip(str(exc))

        _, video_path = synthetic_video_dir
        ingestor = VideoIngestor(target_fps=4.0, save_frames=True, jpeg_quality=80)
        _, frames = ingestor.ingest(video_path, workspace_dir=tmp_path / "save_ws")

        frames_dir = tmp_path / "save_ws" / "frames"
        assert frames_dir.exists()
        jpeg_files = list(frames_dir.glob("frame_*.jpg"))
        assert len(jpeg_files) == len(frames)
        for fp in frames:
            assert fp.frame_path is not None
            assert fp.frame_path.exists()


class TestVideoIngestorEdgeCases:

    def test_missing_video_raises(self, tmp_path):
        """ingest() must raise FileNotFoundError for a non-existent file."""
        try:
            from vid_preprocessing.video_ingestor import VideoIngestor  # noqa: PLC0415
        except (RuntimeError, ImportError) as exc:
            pytest.skip(str(exc))

        ingestor = VideoIngestor(target_fps=8.0)
        with pytest.raises(FileNotFoundError):
            ingestor.ingest("non_existent_video.mp4", workspace_dir=tmp_path)

    def test_metadata_to_dict(self, ingestor, synthetic_video_dir, tmp_path):
        """VideoMetadata.to_dict() must be JSON-serialisable."""
        import json  # noqa: PLC0415
        _, video_path = synthetic_video_dir
        meta, _ = ingestor.ingest(video_path, workspace_dir=tmp_path / "dict_ws")
        d = meta.to_dict()
        json.dumps(d)   # must not raise

    def test_repr(self, ingestor):
        """VideoIngestor must have a useful __repr__."""
        r = repr(ingestor)
        assert "VideoIngestor" in r
        assert "target_fps" in r
