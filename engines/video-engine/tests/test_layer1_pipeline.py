"""
Integration test for VideoLayer1Pipeline — VDS Layer 1 orchestrator.

Runs the full four-stage pipeline (Ingest → Scene → Track → Landmark)
on a synthetic 5-second, 25fps, two-scene video generated in-test.
All heavy models (DeepSORT, MediaPipe) are used if available; the test
degrades gracefully to structural validation if dependencies are absent.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_video(
    path: Path,
    fps: float = 25.0,
    duration_s: float = 5.0,
    width: int = 320,
    height: int = 240,
) -> None:
    """
    Write a synthetic video with two distinct scenes:
    - Scene 1 (0–2.5s) : dark blue frames
    - Scene 2 (2.5–5s) : bright white frames
    No audio stream (OpenCV limitation).
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    n_each = int(fps * duration_s / 2)

    dark   = np.full((height, width, 3), (30,  30, 160), dtype=np.uint8)
    bright = np.full((height, width, 3), (220, 220, 220), dtype=np.uint8)

    for _ in range(n_each):
        writer.write(dark)
    for _ in range(n_each):
        writer.write(bright)
    writer.release()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory):
    d = tmp_path_factory.mktemp("pipeline_test")
    p = d / "synthetic.mp4"
    _make_synthetic_video(p)
    return p


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    return tmp_path_factory.mktemp("pipeline_workspace")


@pytest.fixture(scope="module")
def pipeline():
    """Instantiate the pipeline — skip if FFmpeg is unavailable."""
    try:
        from vid_preprocessing.video_layer1_pipeline import VideoLayer1Pipeline  # noqa: PLC0415
    except ImportError:
        pytest.skip("video_layer1_pipeline not importable")

    try:
        return VideoLayer1Pipeline(
            target_fps=4.0,       # Fast for testing
            tracker_backend="dnn",
            tracker_min_hits=1,   # Confirm tracks faster on short clips
            scene_threshold=15.0, # Sensitive to synthetic hard cut
            scene_min_len=5,
            landmark_refine=False, # Faster (468 landmarks) for tests
            save_frames=False,
        )
    except RuntimeError as exc:
        # FFmpeg not available
        pytest.skip(str(exc))


@pytest.fixture(scope="module")
def pipeline_result(pipeline, synthetic_video, workspace):
    """Run the pipeline once; cache result for all tests in this module."""
    return pipeline.run(
        video_path=synthetic_video,
        workspace_dir=workspace / "run_01",
    )


# ---------------------------------------------------------------------------
# Structural tests (no model weight assertions)
# ---------------------------------------------------------------------------

class TestPipelineResult:

    def test_result_type(self, pipeline_result):
        """pipeline.run() must return a VideoPreprocessingResult."""
        try:
            from vid_preprocessing.result_types import VideoPreprocessingResult  # noqa: PLC0415
        except ImportError:
            pytest.skip("result_types not importable")
        assert isinstance(pipeline_result, VideoPreprocessingResult)

    def test_processing_ok(self, pipeline_result):
        """processing_ok must be True for a valid synthetic video."""
        assert pipeline_result.processing_ok is True

    def test_n_frames_positive(self, pipeline_result):
        """At least one frame must be extracted."""
        assert pipeline_result.n_frames > 0

    def test_n_scenes_positive(self, pipeline_result):
        """At least one scene must be returned."""
        assert pipeline_result.n_scenes >= 1

    def test_metadata_populated(self, pipeline_result):
        """VideoMetadata must have valid resolution and fps."""
        meta = pipeline_result.metadata
        assert meta.width  == 320
        assert meta.height == 240
        assert meta.target_fps == pytest.approx(4.0)
        assert meta.duration_s > 0.0

    def test_all_frames_have_scene_ids(self, pipeline_result):
        """Every FramePacket must have scene_id != -1 after pipeline."""
        for fp in pipeline_result.frames:
            assert fp.scene_id != -1, \
                f"frame_id={fp.frame_id} still has scene_id=-1"

    def test_frame_packets_have_rgb_arrays(self, pipeline_result):
        """All FramePackets must have rgb_array populated (save_frames=False)."""
        for fp in pipeline_result.frames:
            assert fp.rgb_array is not None
            assert fp.rgb_array.ndim == 3

    def test_summary_is_serialisable(self, pipeline_result):
        """summary() must return a JSON-serialisable dict."""
        import json  # noqa: PLC0415
        d = pipeline_result.summary()
        json.dumps(d)   # must not raise

    def test_frames_for_scene_method(self, pipeline_result):
        """frames_for_scene() must return a subset of frames with that scene_id."""
        for sc in pipeline_result.scenes:
            subset = pipeline_result.frames_for_scene(sc.scene_id)
            assert all(fp.scene_id == sc.scene_id for fp in subset)

    def test_scene_boundaries_cover_all_frames(self, pipeline_result):
        """Union of all scene frame ranges must cover all extracted frames."""
        all_scene_ids = {fp.scene_id for fp in pipeline_result.frames}
        result_scene_ids = {sc.scene_id for sc in pipeline_result.scenes}
        # Every scene_id on frames must correspond to a known SceneBoundary
        assert all_scene_ids.issubset(result_scene_ids)

    def test_unique_track_ids_format(self, pipeline_result):
        """All track IDs must be strings starting with 'track_'."""
        for tid in pipeline_result.unique_track_ids:
            assert isinstance(tid, str)
            assert tid.startswith("track_"), f"Unexpected track_id format: {tid}"

    def test_landmarks_dict_keys_match_track_ids(self, pipeline_result):
        """landmarks dict keys must be a subset of unique_track_ids."""
        track_set = set(pipeline_result.unique_track_ids)
        for lm_key in pipeline_result.landmarks:
            assert lm_key in track_set, \
                f"landmark key '{lm_key}' not in unique_track_ids"

    def test_no_audio_on_synthetic_video(self, pipeline_result):
        """Synthetic video (OpenCV, no audio) must not produce audio_path."""
        # OpenCV VideoWriter has no audio support
        assert pipeline_result.metadata.has_audio is False


class TestPipelineContextManager:

    def test_context_manager_no_error(self, synthetic_video, workspace):
        """VideoLayer1Pipeline used as a context manager must not raise."""
        try:
            from vid_preprocessing.video_layer1_pipeline import VideoLayer1Pipeline  # noqa: PLC0415
        except ImportError:
            pytest.skip("video_layer1_pipeline not importable")

        try:
            with VideoLayer1Pipeline(target_fps=2.0, save_frames=False) as p:
                result = p.run(
                    video_path=synthetic_video,
                    workspace_dir=workspace / "ctx_run",
                )
        except RuntimeError as exc:
            pytest.skip(str(exc))   # FFmpeg not available

        assert result is not None
        assert p.landmark_tracer._face_mesh is None  # closed by __exit__


class TestPipelineMissingVideo:

    def test_missing_video_raises(self, workspace):
        """run() on a non-existent video must raise FileNotFoundError."""
        try:
            from vid_preprocessing.video_layer1_pipeline import VideoLayer1Pipeline  # noqa: PLC0415
        except ImportError:
            pytest.skip("video_layer1_pipeline not importable")

        try:
            pipeline = VideoLayer1Pipeline(target_fps=4.0)
        except RuntimeError as exc:
            pytest.skip(str(exc))

        with pytest.raises(FileNotFoundError):
            pipeline.run("totally_fake_video.mp4", workspace_dir=workspace / "err")


class TestPipelineRepr:

    def test_repr_contains_key_info(self):
        """VideoLayer1Pipeline.__repr__ must mention fps and tracker backend."""
        try:
            from vid_preprocessing.video_layer1_pipeline import VideoLayer1Pipeline  # noqa: PLC0415
        except ImportError:
            pytest.skip("video_layer1_pipeline not importable")

        try:
            p = VideoLayer1Pipeline(target_fps=12.0, tracker_backend="dnn")
        except RuntimeError as exc:
            pytest.skip(str(exc))

        r = repr(p)
        assert "VideoLayer1Pipeline" in r
        assert "12.0" in r
        assert "dnn" in r
