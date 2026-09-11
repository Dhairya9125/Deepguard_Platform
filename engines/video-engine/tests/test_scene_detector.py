"""
Tests for SceneDetector — VDS Layer 1 PySceneDetect scene segmentation.

Uses a synthetic multi-scene video generated via OpenCV VideoWriter
(abrupt brightness change between scenes simulates a hard cut).
Tests requiring PySceneDetect are skipped if the package is absent.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_two_scene_video(path: Path, fps: float = 25.0) -> None:
    """
    Write a 4-second video with a hard scene cut at 2 seconds:
    - Scene 1 (0–2s)  : dark blue frames (RGB ≈ 30, 30, 150)
    - Scene 2 (2–4s)  : bright white frames (RGB ≈ 220, 220, 220)
    The abrupt colour change simulates a hard cut for ContentDetector.
    """
    fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
    h, w    = 240, 320
    writer  = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    n_each  = int(fps * 2.0)

    dark_frame   = np.full((h, w, 3), (30,  30,  150), dtype=np.uint8)
    bright_frame = np.full((h, w, 3), (220, 220, 220), dtype=np.uint8)

    for _ in range(n_each):
        writer.write(dark_frame)
    for _ in range(n_each):
        writer.write(bright_frame)
    writer.release()


def _make_single_scene_video(path: Path, fps: float = 25.0) -> None:
    """Write a 3-second uniform video — no hard cuts."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w   = 240, 320
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    frame  = np.full((h, w, 3), (100, 150, 200), dtype=np.uint8)
    for _ in range(int(fps * 3)):
        writer.write(frame)
    writer.release()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def two_scene_video(tmp_path_factory):
    d = tmp_path_factory.mktemp("scene_det_test")
    p = d / "two_scenes.mp4"
    _make_two_scene_video(p)
    return p


@pytest.fixture(scope="module")
def single_scene_video(tmp_path_factory):
    d = tmp_path_factory.mktemp("scene_det_single")
    p = d / "single.mp4"
    _make_single_scene_video(p)
    return p


# ---------------------------------------------------------------------------
# SceneDetector unit tests
# ---------------------------------------------------------------------------

class TestSceneDetectorInit:

    def test_init_defaults(self):
        """SceneDetector initialises with expected defaults."""
        try:
            from preprocessing.scene_detector import SceneDetector  # noqa: PLC0415
        except ImportError:
            pytest.skip("scene_detector not importable")
        sd = SceneDetector()
        assert sd.threshold      == 27.0
        assert sd.min_scene_len  == 15
        assert sd.show_progress  is False

    def test_repr(self):
        """__repr__ must mention threshold and class name."""
        try:
            from preprocessing.scene_detector import SceneDetector  # noqa: PLC0415
        except ImportError:
            pytest.skip("scene_detector not importable")
        sd = SceneDetector(threshold=30.0)
        r  = repr(sd)
        assert "SceneDetector" in r
        assert "30.0" in r

    def test_detect_missing_video(self, tmp_path):
        """detect() on a non-existent file must raise FileNotFoundError."""
        try:
            from preprocessing.scene_detector import SceneDetector  # noqa: PLC0415
        except ImportError:
            pytest.skip("scene_detector not importable")
        sd = SceneDetector()
        with pytest.raises(FileNotFoundError):
            sd.detect(tmp_path / "does_not_exist.mp4")


class TestSceneDetectorDetect:

    def test_detect_returns_list(self, single_scene_video):
        """detect() must always return a list (at minimum with one scene)."""
        try:
            from preprocessing.scene_detector import SceneDetector  # noqa: PLC0415
            import scenedetect  # noqa: PLC0415, F401
        except ImportError:
            pytest.skip("scene_detector or scenedetect not available")

        sd     = SceneDetector(threshold=27.0, min_scene_len=5)
        scenes = sd.detect(single_scene_video)
        assert isinstance(scenes, list)
        assert len(scenes) >= 1   # at least one scene always returned

    def test_single_scene_video_returns_one_scene(self, single_scene_video):
        """A uniform video should produce exactly 1 scene."""
        try:
            from preprocessing.scene_detector import SceneDetector  # noqa: PLC0415
            import scenedetect  # noqa: PLC0415, F401
        except ImportError:
            pytest.skip("scene_detector or scenedetect not available")

        sd     = SceneDetector(threshold=27.0, min_scene_len=5)
        scenes = sd.detect(single_scene_video)
        assert len(scenes) == 1, f"Expected 1 scene, got {len(scenes)}"

    def test_two_scene_video_returns_two_scenes(self, two_scene_video):
        """A video with one hard cut should produce 2 scenes."""
        try:
            from preprocessing.scene_detector import SceneDetector  # noqa: PLC0415
            import scenedetect  # noqa: PLC0415, F401
        except ImportError:
            pytest.skip("scene_detector or scenedetect not available")

        # Use low threshold to ensure the hard cut is detected
        sd     = SceneDetector(threshold=15.0, min_scene_len=5)
        scenes = sd.detect(two_scene_video)
        assert len(scenes) == 2, f"Expected 2 scenes, got {len(scenes)}: {scenes}"

    def test_scene_ids_sequential(self, single_scene_video):
        """Scene IDs must be 0, 1, 2, … without gaps."""
        try:
            from preprocessing.scene_detector import SceneDetector  # noqa: PLC0415
            import scenedetect  # noqa: PLC0415, F401
        except ImportError:
            pytest.skip("scene_detector or scenedetect not available")

        sd     = SceneDetector()
        scenes = sd.detect(single_scene_video)
        for i, sc in enumerate(scenes):
            assert sc.scene_id == i

    def test_scene_boundary_fields(self, single_scene_video):
        """SceneBoundary must have all required fields with valid values."""
        try:
            from preprocessing.scene_detector import SceneDetector  # noqa: PLC0415
            import scenedetect  # noqa: PLC0415, F401
        except ImportError:
            pytest.skip("scene_detector or scenedetect not available")

        sd     = SceneDetector()
        scenes = sd.detect(single_scene_video)
        sc     = scenes[0]
        assert sc.start_ms  >= 0.0
        assert sc.end_ms    >  sc.start_ms
        assert sc.end_frame >= sc.start_frame
        assert sc.frame_count > 0
        assert sc.duration_ms > 0.0


class TestSceneDetectorAssignIDs:

    def _make_frames_with_timestamps(self, n: int, fps: float = 8.0):
        """Create minimal FramePacket list for assignment testing."""
        try:
            from preprocessing.result_types import FramePacket  # noqa: PLC0415
        except ImportError:
            pytest.skip("result_types not importable")
        return [
            FramePacket(frame_id=i, timestamp_ms=i * (1000.0 / fps))
            for i in range(n)
        ]

    def test_assign_sets_scene_id_on_all_frames(self, single_scene_video):
        """After assign_scene_ids, no frame should have scene_id == -1."""
        try:
            from preprocessing.scene_detector import SceneDetector  # noqa: PLC0415
            import scenedetect  # noqa: PLC0415, F401
        except ImportError:
            pytest.skip("scene_detector or scenedetect not available")

        sd     = SceneDetector()
        scenes = sd.detect(single_scene_video)
        frames = self._make_frames_with_timestamps(16, fps=8.0)
        sd.assign_scene_ids(frames, scenes)

        for fp in frames:
            assert fp.scene_id != -1, f"frame_id={fp.frame_id} has scene_id=-1"

    def test_assign_without_detect_raises(self):
        """assign_scene_ids with no scenes and no cached result must raise."""
        try:
            from preprocessing.scene_detector import SceneDetector  # noqa: PLC0415
            from preprocessing.result_types    import FramePacket  # noqa: PLC0415
        except ImportError:
            pytest.skip("modules not importable")

        sd = SceneDetector()
        frames = [FramePacket(frame_id=0, timestamp_ms=0.0)]
        with pytest.raises(ValueError, match="No scene list available"):
            sd.assign_scene_ids(frames, scenes=None)

    def test_assign_empty_frames_no_error(self, single_scene_video):
        """assign_scene_ids on an empty frame list must not raise."""
        try:
            from preprocessing.scene_detector import SceneDetector  # noqa: PLC0415
            import scenedetect  # noqa: PLC0415, F401
        except ImportError:
            pytest.skip("scene_detector or scenedetect not available")

        sd     = SceneDetector()
        scenes = sd.detect(single_scene_video)
        sd.assign_scene_ids([], scenes)   # must not raise

    def test_scene_ids_monotone_with_cut(self, two_scene_video):
        """
        In a two-scene video, scene_id on frames before the cut must be 0
        and frames after the cut must be 1.
        """
        try:
            from preprocessing.scene_detector import SceneDetector  # noqa: PLC0415
            import scenedetect  # noqa: PLC0415, F401
        except ImportError:
            pytest.skip("scene_detector or scenedetect not available")

        sd     = SceneDetector(threshold=15.0, min_scene_len=5)
        scenes = sd.detect(two_scene_video)

        if len(scenes) < 2:
            pytest.skip("Scene detection did not produce 2 scenes on this machine.")

        # Create frames spanning 4 seconds at 8 fps = 32 frames
        frames = self._make_frames_with_timestamps(32, fps=8.0)
        sd.assign_scene_ids(frames, scenes)

        # Frames in first 2 s → scene 0; frames after 2 s → scene 1
        cut_ms = 2000.0
        for fp in frames:
            if fp.timestamp_ms < cut_ms - 200:  # 200 ms buffer
                assert fp.scene_id == 0, \
                    f"frame_id={fp.frame_id} t={fp.timestamp_ms}ms expected scene 0"
            elif fp.timestamp_ms > cut_ms + 200:
                assert fp.scene_id == 1, \
                    f"frame_id={fp.frame_id} t={fp.timestamp_ms}ms expected scene 1"


class TestSceneDetectorSummary:

    def test_summary_returns_string(self, single_scene_video):
        """summary() must return a non-empty string."""
        try:
            from preprocessing.scene_detector import SceneDetector  # noqa: PLC0415
            import scenedetect  # noqa: PLC0415, F401
        except ImportError:
            pytest.skip("scene_detector or scenedetect not available")

        sd     = SceneDetector()
        scenes = sd.detect(single_scene_video)
        s      = sd.summary(scenes)
        assert isinstance(s, str)
        assert len(s) > 0
        assert "SceneDetector" in s
