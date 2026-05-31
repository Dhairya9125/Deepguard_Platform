"""
Tests for VDS Layer 2 Branch A — Frame-Level Visual Analysis.

Test strategy:
  - All IDS model loading is skipped if PyTorch / HuggingFace deps are absent
  - Structural and dataclass tests run with zero heavy dependencies
  - Batch logic and path-management tests use mocks to isolate Branch A logic
  - Integration-level tests (real IDS forward pass) are gated on torch + transformers
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blank_crop(h=224, w=224) -> np.ndarray:
    """Return a blank (black) RGB uint8 crop."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _random_crop(h=224, w=224) -> np.ndarray:
    """Return a random-noise RGB uint8 crop."""
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def _make_frame_visual_result(
    track_id="track_001",
    frame_id=0,
    timestamp_ms=0.0,
    fake_probability=0.2,
    detection_success=True,
):
    try:
        from feature_extraction.result_types_l2 import FrameVisualResult  # noqa: PLC0415
    except ImportError:
        pytest.skip("result_types_l2 not importable")
    return FrameVisualResult(
        track_id=track_id,
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        fake_probability=fake_probability,
        ood_score=0.05,
        branch_embeddings={
            "spatial":     np.zeros(512, dtype=np.float32),
            "frequency":   np.zeros(512, dtype=np.float32),
            "discrepancy": np.zeros(512, dtype=np.float32),
            "noise":       np.zeros(512, dtype=np.float32),
            "fingerprint": np.zeros(512, dtype=np.float32),
        },
        artifact_embedding=np.zeros(2560, dtype=np.float32),
        manipulation_heatmap=np.zeros((224, 224), dtype=np.float32),
        manipulated_mask=np.zeros((224, 224), dtype=np.uint8),
        blend_boundaries=np.zeros((224, 224), dtype=np.uint8),
        bounding_boxes=[],
        detection_success=detection_success,
    )


# ---------------------------------------------------------------------------
# FrameVisualResult unit tests (zero dependencies)
# ---------------------------------------------------------------------------

class TestFrameVisualResultDataclass:

    def test_fields_populated(self):
        result = _make_frame_visual_result(fake_probability=0.8)
        assert result.track_id == "track_001"
        assert result.frame_id == 0
        assert result.fake_probability == pytest.approx(0.8)

    def test_is_fake_property_true(self):
        r = _make_frame_visual_result(fake_probability=0.85, detection_success=True)
        assert r.is_fake is True

    def test_is_fake_property_false_below_threshold(self):
        r = _make_frame_visual_result(fake_probability=0.3, detection_success=True)
        assert r.is_fake is False

    def test_is_fake_false_when_detection_failed(self):
        r = _make_frame_visual_result(fake_probability=0.99, detection_success=False)
        assert r.is_fake is False

    def test_n_manipulated_regions_empty(self):
        r = _make_frame_visual_result()
        assert r.n_manipulated_regions == 0

    def test_n_manipulated_regions_with_boxes(self):
        try:
            from feature_extraction.result_types_l2 import FrameVisualResult  # noqa: PLC0415
        except ImportError:
            pytest.skip("result_types_l2 not importable")
        r = FrameVisualResult(
            track_id="track_001", frame_id=0, timestamp_ms=0.0,
            bounding_boxes=[(10, 10, 50, 50), (100, 100, 30, 30)],
        )
        assert r.n_manipulated_regions == 2

    def test_to_dict_is_serialisable(self):
        import json  # noqa: PLC0415
        r = _make_frame_visual_result()
        d = r.to_dict()
        json.dumps(d)  # Must not raise

    def test_to_dict_keys(self):
        r = _make_frame_visual_result()
        d = r.to_dict()
        expected_keys = {
            "track_id", "frame_id", "timestamp_ms", "fake_probability",
            "ood_score", "is_fake", "n_manipulated_regions", "bounding_boxes",
            "branch_embedding_keys", "artifact_embedding_shape",
            "heatmap_shape", "detection_success", "failure_reason",
        }
        assert expected_keys.issubset(set(d.keys()))

    def test_to_dict_none_arrays_when_failed(self):
        try:
            from feature_extraction.result_types_l2 import FrameVisualResult  # noqa: PLC0415
        except ImportError:
            pytest.skip("result_types_l2 not importable")
        r = FrameVisualResult(
            track_id="track_001", frame_id=0, timestamp_ms=0.0,
            detection_success=False, failure_reason="test error",
        )
        d = r.to_dict()
        assert d["artifact_embedding_shape"] is None
        assert d["heatmap_shape"] is None
        assert d["detection_success"] is False
        assert d["failure_reason"] == "test error"


# ---------------------------------------------------------------------------
# BranchAResult unit tests (zero dependencies)
# ---------------------------------------------------------------------------

class TestBranchAResult:

    def _make_branch_a(self):
        try:
            from feature_extraction.result_types_l2 import BranchAResult  # noqa: PLC0415
        except ImportError:
            pytest.skip("result_types_l2 not importable")

        r1 = _make_frame_visual_result("track_001", 0, 0.0,   fake_probability=0.2)
        r2 = _make_frame_visual_result("track_001", 1, 125.0, fake_probability=0.8)
        r3 = _make_frame_visual_result("track_002", 0, 0.0,   fake_probability=0.1)

        return BranchAResult(
            frame_results={
                "track_001": [r1, r2],
                "track_002": [r3],
            },
            n_frames_processed=10,
            n_faces_total=3,
            n_faces_successful=3,
        )

    def test_unique_track_ids_sorted(self):
        ba = self._make_branch_a()
        assert ba.unique_track_ids == ["track_001", "track_002"]

    def test_n_tracks(self):
        ba = self._make_branch_a()
        assert ba.n_tracks == 2

    def test_results_for_track(self):
        ba = self._make_branch_a()
        r  = ba.results_for_track("track_001")
        assert len(r) == 2

    def test_results_for_missing_track(self):
        ba = self._make_branch_a()
        assert ba.results_for_track("track_999") == []

    def test_fake_probability_timeline(self):
        ba = self._make_branch_a()
        tl = ba.fake_probability_timeline("track_001")
        assert len(tl) == 2
        assert tl[0] == pytest.approx(0.2)
        assert tl[1] == pytest.approx(0.8)

    def test_mean_fake_probability(self):
        ba = self._make_branch_a()
        mean = ba.mean_fake_probability("track_001")
        assert mean == pytest.approx(0.5)

    def test_mean_fake_probability_missing_track(self):
        ba = self._make_branch_a()
        assert ba.mean_fake_probability("track_999") is None

    def test_artifact_embedding_timeline(self):
        ba = self._make_branch_a()
        tl = ba.artifact_embedding_timeline("track_001")
        assert len(tl) == 2
        assert tl[0].shape == (2560,)

    def test_summary_serialisable(self):
        import json  # noqa: PLC0415
        ba = self._make_branch_a()
        s  = ba.summary()
        json.dumps(s)

    def test_summary_contains_track_summaries(self):
        ba = self._make_branch_a()
        s  = ba.summary()
        assert "track_summaries" in s
        assert "track_001" in s["track_summaries"]
        assert "track_002" in s["track_summaries"]

    def test_verdict_fake(self):
        ba = self._make_branch_a()
        # track_001 mean_prob=0.5, which is not > 0.5, so REAL
        # make it FAKE by using prob 0.9 and 0.9
        try:
            from feature_extraction.result_types_l2 import BranchAResult  # noqa: PLC0415
        except ImportError:
            pytest.skip("result_types_l2 not importable")
        r1 = _make_frame_visual_result("track_001", 0, 0.0,   fake_probability=0.9)
        r2 = _make_frame_visual_result("track_001", 1, 125.0, fake_probability=0.85)
        ba2 = BranchAResult(frame_results={"track_001": [r1, r2]})
        assert ba2.summary()["track_summaries"]["track_001"]["verdict"] == "FAKE"


# ---------------------------------------------------------------------------
# IDSAdapter path resolution tests (no model loading)
# ---------------------------------------------------------------------------

class TestIDSAdapterPathResolution:

    def test_resolve_ids_path_finds_image_engine(self):
        try:
            from feature_extraction.ids_adapter import _resolve_ids_path  # noqa: PLC0415
        except ImportError:
            pytest.skip("ids_adapter not importable")
        ids_path = _resolve_ids_path()
        assert ids_path.exists(), f"image-engine not found at {ids_path}"
        assert (ids_path / "preprocessing").exists()
        assert (ids_path / "feature_extraction").exists()

    def test_ensure_ids_on_path_idempotent(self):
        """Calling _ensure_ids_on_path() twice must not duplicate path entries."""
        try:
            from feature_extraction.ids_adapter import (  # noqa: PLC0415
                _ensure_ids_on_path,
                _resolve_ids_path,
            )
        except ImportError:
            pytest.skip("ids_adapter not importable")
        ids_str = str(_resolve_ids_path())
        _ensure_ids_on_path()
        _ensure_ids_on_path()
        count = sys.path.count(ids_str)
        assert count <= 2, f"Path inserted too many times: {count}"


# ---------------------------------------------------------------------------
# FrameVisualAnalyzer structural tests (mocked IDS)
# ---------------------------------------------------------------------------

class TestFrameVisualAnalyzerStructural:

    def _make_mock_ids(self):
        """Create a minimal mock IDSComponents that produces plausible outputs."""
        import torch  # noqa: PLC0415
        pytest.importorskip("torch", reason="PyTorch not installed")

        mock_ids = MagicMock()

        # Preprocessing mocks
        mock_ids.normalizer.normalize.return_value = np.zeros((3, 224, 224), dtype=np.float32)
        mock_ids.freq_analyzer.get_stacked_frequency_tensor.return_value = torch.zeros(3, 224, 224)
        mock_ids.device = torch.device("cpu")

        # Branch model mocks: return (1, 512) tensors
        emb = torch.zeros(1, 512)
        for branch in ["branch_spatial", "branch_freq", "branch_disc", "branch_noise", "branch_fingerprint"]:
            getattr(mock_ids, branch).return_value = emb

        # Fusion mock
        mock_ids.fusion.return_value = {
            "fake_probability":    torch.tensor([[0.3]]),
            "OOD_score":           torch.tensor([[0.01]]),
            "artifact_embedding":  torch.zeros(1, 2560),
            "manipulation_heatmap": torch.zeros(1, 1, 224, 224),
        }

        # Localization mock
        mock_ids.localization.process.return_value = [{
            "manipulated_mask": np.zeros((224, 224), dtype=np.uint8),
            "blend_boundaries": np.zeros((224, 224), dtype=np.uint8),
            "bounding_boxes":   [],
        }]

        return mock_ids

    def test_analyze_crop_none_returns_failure(self):
        try:
            from feature_extraction.frame_visual_analyzer import FrameVisualAnalyzer  # noqa: PLC0415
        except ImportError:
            pytest.skip("frame_visual_analyzer not importable")
        analyzer = FrameVisualAnalyzer()
        result = analyzer.analyze_crop("track_001", 0, 0.0, None)
        assert result.detection_success is False
        assert result.failure_reason is not None

    def test_analyze_crop_wrong_shape_returns_failure(self):
        try:
            from feature_extraction.frame_visual_analyzer import FrameVisualAnalyzer  # noqa: PLC0415
        except ImportError:
            pytest.skip("frame_visual_analyzer not importable")
        analyzer = FrameVisualAnalyzer()
        bad_crop = np.zeros((224, 224), dtype=np.uint8)  # 2D, no channels
        result = analyzer.analyze_crop("track_001", 0, 0.0, bad_crop)
        assert result.detection_success is False

    def test_analyze_crop_with_mocked_ids(self):
        """analyze_crop should return a valid FrameVisualResult when IDS is mocked."""
        try:
            from feature_extraction.frame_visual_analyzer import FrameVisualAnalyzer  # noqa: PLC0415
        except ImportError:
            pytest.skip("frame_visual_analyzer not importable")
        pytest.importorskip("torch", reason="PyTorch not installed")

        analyzer = FrameVisualAnalyzer()
        analyzer._ids = self._make_mock_ids()

        crop = _blank_crop()
        result = analyzer.analyze_crop("track_001", 5, 625.0, crop)

        assert result.detection_success is True
        assert result.track_id == "track_001"
        assert result.frame_id == 5
        assert result.timestamp_ms == pytest.approx(625.0)
        assert 0.0 <= result.fake_probability <= 1.0
        assert set(result.branch_embeddings.keys()) == {
            "spatial", "frequency", "discrepancy", "noise", "fingerprint"
        }
        assert result.artifact_embedding.shape == (2560,)
        assert result.manipulation_heatmap.shape == (224, 224)

    def test_analyze_crop_resizes_non_224(self):
        """analyze_crop must handle crops that are not 224x224."""
        try:
            from feature_extraction.frame_visual_analyzer import FrameVisualAnalyzer  # noqa: PLC0415
        except ImportError:
            pytest.skip("frame_visual_analyzer not importable")
        pytest.importorskip("torch", reason="PyTorch not installed")

        analyzer = FrameVisualAnalyzer()
        analyzer._ids = self._make_mock_ids()

        crop_112 = _blank_crop(h=112, w=112)
        result = analyzer.analyze_crop("track_001", 0, 0.0, crop_112)
        assert result.detection_success is True

    def test_repr(self):
        try:
            from feature_extraction.frame_visual_analyzer import FrameVisualAnalyzer  # noqa: PLC0415
        except ImportError:
            pytest.skip("frame_visual_analyzer not importable")
        analyzer = FrameVisualAnalyzer(device="cpu", batch_size=4)
        r = repr(analyzer)
        assert "FrameVisualAnalyzer" in r
        assert "cpu" in r
        assert "4" in r


# ---------------------------------------------------------------------------
# Batch processing tests (mocked IDS)
# ---------------------------------------------------------------------------

class TestBatchProcessing:

    def _make_mock_ids_batch(self, batch_size: int):
        """Mock that returns correct batch-sized tensors."""
        import torch  # noqa: PLC0415
        mock_ids = MagicMock()
        mock_ids.normalizer.normalize.return_value = np.zeros((3, 224, 224), dtype=np.float32)
        mock_ids.freq_analyzer.get_stacked_frequency_tensor.return_value = torch.zeros(3, 224, 224)
        mock_ids.device = torch.device("cpu")

        B = batch_size
        emb = torch.zeros(B, 512)
        for branch in ["branch_spatial", "branch_freq", "branch_disc", "branch_noise", "branch_fingerprint"]:
            getattr(mock_ids, branch).return_value = emb

        mock_ids.fusion.return_value = {
            "fake_probability":    torch.full((B, 1), 0.35),
            "OOD_score":           torch.full((B, 1), 0.02),
            "artifact_embedding":  torch.zeros(B, 2560),
            "manipulation_heatmap": torch.zeros(B, 1, 224, 224),
        }
        mock_ids.localization.process.return_value = [
            {"manipulated_mask": np.zeros((224, 224), dtype=np.uint8),
             "blend_boundaries": np.zeros((224, 224), dtype=np.uint8),
             "bounding_boxes": []}
            for _ in range(B)
        ]
        return mock_ids

    def test_batch_of_four_all_succeed(self):
        try:
            from feature_extraction.frame_visual_analyzer import FrameVisualAnalyzer  # noqa: PLC0415
        except ImportError:
            pytest.skip("frame_visual_analyzer not importable")
        pytest.importorskip("torch", reason="PyTorch not installed")

        analyzer = FrameVisualAnalyzer(batch_size=4)
        analyzer._ids = self._make_mock_ids_batch(4)

        batch = [("track_001", i, float(i * 125), _blank_crop()) for i in range(4)]
        results = analyzer._run_ids_batch(batch)

        assert len(results) == 4
        for r in results:
            assert r.detection_success is True
            assert r.artifact_embedding.shape == (2560,)

    def test_batch_with_mixed_valid_invalid(self):
        """Batch with 2 valid crops and 1 None should produce 2 success, 1 failure."""
        try:
            from feature_extraction.frame_visual_analyzer import FrameVisualAnalyzer  # noqa: PLC0415
        except ImportError:
            pytest.skip("frame_visual_analyzer not importable")
        pytest.importorskip("torch", reason="PyTorch not installed")

        analyzer = FrameVisualAnalyzer(batch_size=8)
        analyzer._ids = self._make_mock_ids_batch(2)

        batch = [
            ("track_001", 0, 0.0,   _blank_crop()),  # valid
            ("track_001", 1, 125.0, None),            # invalid
            ("track_002", 0, 0.0,   _blank_crop()),  # valid
        ]
        results = analyzer._run_ids_batch(batch)

        assert len(results) == 3
        assert results[0].detection_success is True
        assert results[1].detection_success is False
        assert results[2].detection_success is True

    def test_batch_single_crop(self):
        """Batch of size 1 must work correctly."""
        try:
            from feature_extraction.frame_visual_analyzer import FrameVisualAnalyzer  # noqa: PLC0415
        except ImportError:
            pytest.skip("frame_visual_analyzer not importable")
        pytest.importorskip("torch", reason="PyTorch not installed")

        analyzer = FrameVisualAnalyzer(batch_size=1)
        analyzer._ids = self._make_mock_ids_batch(1)

        batch = [("track_001", 0, 0.0, _blank_crop())]
        results = analyzer._run_ids_batch(batch)

        assert len(results) == 1
        assert results[0].detection_success is True


# ---------------------------------------------------------------------------
# analyze() integration test with mocked Layer 1 result
# ---------------------------------------------------------------------------

class TestAnalyzeFromLayer1:

    def _make_layer1_result(self, n_tracks=2, n_frames=4):
        """
        Build a minimal synthetic VideoPreprocessingResult with tracked faces.
        Does NOT require real video or FFmpeg — pure in-memory stub.
        """
        try:
            from preprocessing.result_types import (  # noqa: PLC0415
                FramePacket, TrackedFace, VideoMetadata,
                VideoPreprocessingResult, SceneBoundary,
            )
        except ImportError:
            pytest.skip("preprocessing.result_types not importable")

        from pathlib import Path  # noqa: PLC0415

        meta = VideoMetadata(
            source_path=Path("synthetic.mp4"),
            duration_s=float(n_frames) / 8.0,
            native_fps=25.0,
            width=320,
            height=240,
            total_frames_native=n_frames * 3,
            total_frames_extracted=n_frames,
            target_fps=8.0,
            audio_path=None,
            has_audio=False,
            audio_sample_rate=16000,
        )

        track_ids = [f"track_{i+1:03d}" for i in range(n_tracks)]
        frames = []
        for fi in range(n_frames):
            tracked = [
                TrackedFace(
                    track_id=tid,
                    track_age=fi,
                    detection_score=0.95,
                    bbox=(10, 10, 110, 110),
                    frame_id=fi,
                    timestamp_ms=float(fi * 125),
                    aligned_crop=_blank_crop(),
                )
                for tid in track_ids
            ]
            fp = FramePacket(
                frame_id=fi,
                timestamp_ms=float(fi * 125),
                rgb_array=_blank_crop(240, 320),
                tracked_faces=tracked,
                scene_id=0,
            )
            frames.append(fp)

        scenes = [SceneBoundary(
            scene_id=0, start_frame=0, end_frame=n_frames-1,
            start_ms=0.0, end_ms=float(n_frames*125), frame_count=n_frames,
        )]

        return VideoPreprocessingResult(
            metadata=meta,
            frames=frames,
            scenes=scenes,
            landmarks={},
            unique_track_ids=track_ids,
        )

    def test_analyze_returns_branch_a_result(self):
        try:
            from feature_extraction.frame_visual_analyzer import FrameVisualAnalyzer  # noqa: PLC0415
            from feature_extraction.result_types_l2       import BranchAResult       # noqa: PLC0415
        except ImportError:
            pytest.skip("modules not importable")
        pytest.importorskip("torch", reason="PyTorch not installed")

        layer1 = self._make_layer1_result(n_tracks=2, n_frames=4)

        # Patch load_ids_components to avoid real model loading
        mock_ids = MagicMock()
        import torch  # noqa: PLC0415
        mock_ids.device = torch.device("cpu")
        mock_ids.normalizer.normalize.return_value = np.zeros((3, 224, 224), dtype=np.float32)
        mock_ids.freq_analyzer.get_stacked_frequency_tensor.return_value = torch.zeros(3, 224, 224)
        B = 8
        emb = torch.zeros(B, 512)
        for b in ["branch_spatial", "branch_freq", "branch_disc", "branch_noise", "branch_fingerprint"]:
            getattr(mock_ids, b).return_value = emb
        mock_ids.fusion.return_value = {
            "fake_probability":    torch.full((B, 1), 0.25),
            "OOD_score":           torch.full((B, 1), 0.01),
            "artifact_embedding":  torch.zeros(B, 2560),
            "manipulation_heatmap": torch.zeros(B, 1, 224, 224),
        }
        mock_ids.localization.process.return_value = [
            {"manipulated_mask": np.zeros((224,224), dtype=np.uint8),
             "blend_boundaries": np.zeros((224,224), dtype=np.uint8),
             "bounding_boxes": []}
            for _ in range(B)
        ]

        analyzer = FrameVisualAnalyzer(batch_size=8)
        analyzer._ids = mock_ids

        result = analyzer.analyze(layer1)
        assert isinstance(result, BranchAResult)

    def test_analyze_result_has_all_tracks(self):
        """BranchAResult must have an entry for each unique track ID."""
        try:
            from feature_extraction.frame_visual_analyzer import FrameVisualAnalyzer  # noqa: PLC0415
        except ImportError:
            pytest.skip("modules not importable")
        pytest.importorskip("torch", reason="PyTorch not installed")

        layer1 = self._make_layer1_result(n_tracks=3, n_frames=3)

        import torch  # noqa: PLC0415
        mock_ids = MagicMock()
        mock_ids.device = torch.device("cpu")
        mock_ids.normalizer.normalize.return_value = np.zeros((3, 224, 224), dtype=np.float32)
        mock_ids.freq_analyzer.get_stacked_frequency_tensor.return_value = torch.zeros(3, 224, 224)
        B = 8
        emb = torch.zeros(B, 512)
        for b in ["branch_spatial", "branch_freq", "branch_disc", "branch_noise", "branch_fingerprint"]:
            getattr(mock_ids, b).return_value = emb
        mock_ids.fusion.return_value = {
            "fake_probability":    torch.full((B, 1), 0.15),
            "OOD_score":           torch.full((B, 1), 0.01),
            "artifact_embedding":  torch.zeros(B, 2560),
            "manipulation_heatmap": torch.zeros(B, 1, 224, 224),
        }
        mock_ids.localization.process.return_value = [
            {"manipulated_mask": np.zeros((224,224), dtype=np.uint8),
             "blend_boundaries": np.zeros((224,224), dtype=np.uint8),
             "bounding_boxes": []}
            for _ in range(B)
        ]

        analyzer = FrameVisualAnalyzer(batch_size=8)
        analyzer._ids = mock_ids
        result = analyzer.analyze(layer1)

        assert result.n_tracks == 3
        for i in range(1, 4):
            assert f"track_{i:03d}" in result.frame_results

    def test_analyze_empty_layer1_returns_empty(self):
        """analyze() on a Layer 1 result with no tracked faces returns empty BranchAResult."""
        try:
            from feature_extraction.frame_visual_analyzer import FrameVisualAnalyzer  # noqa: PLC0415
            from preprocessing.result_types import (  # noqa: PLC0415
                FramePacket, VideoMetadata, VideoPreprocessingResult, SceneBoundary,
            )
        except ImportError:
            pytest.skip("modules not importable")

        from pathlib import Path  # noqa: PLC0415
        meta = VideoMetadata(
            source_path=Path("s.mp4"), duration_s=1.0, native_fps=25.0,
            width=320, height=240, total_frames_native=25, total_frames_extracted=4,
            target_fps=8.0, audio_path=None, has_audio=False, audio_sample_rate=16000,
        )
        frames = [
            FramePacket(frame_id=i, timestamp_ms=float(i*125), tracked_faces=[])
            for i in range(4)
        ]
        scenes = [SceneBoundary(0, 0, 3, 0.0, 375.0, 4)]
        layer1 = VideoPreprocessingResult(
            metadata=meta, frames=frames, scenes=scenes, landmarks={}, unique_track_ids=[],
        )

        analyzer = FrameVisualAnalyzer()
        result = analyzer.analyze(layer1)

        assert result.n_faces_total == 0
        assert result.n_tracks == 0
        assert "No tracked faces" in " ".join(result.processing_warnings)

    def test_unload_clears_ids(self):
        try:
            from feature_extraction.frame_visual_analyzer import FrameVisualAnalyzer  # noqa: PLC0415
        except ImportError:
            pytest.skip("frame_visual_analyzer not importable")
        analyzer = FrameVisualAnalyzer()
        analyzer._ids = MagicMock()  # Inject fake IDS
        analyzer.unload()
        assert analyzer._ids is None
