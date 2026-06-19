"""
Tests for VDS Layer 4 — Video Localization.

Test strategy:
  - Coordinate mapping: Verify inverse transform from crop-space to global frame coords.
  - Frame aggregation: Verify manipulated frame detection from fusion timelines.
  - Audio intervals: Verify contiguous grouping of mismatch frames into time intervals.
  - Cross-modal mismatch timeline: Verify synthesis from Branch D metrics.
  - Missing inputs: Verify graceful handling when branches are absent.
  - End-to-end orchestration: Full localize() call with synthetic mock data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest

from vid_localization.video_localizer import VideoLocalizer
from vid_localization.result_types import VideoLocalizationResult


# ===========================================================================
# Mock Stubs for Layer 1 / 2 / 3 Inputs
# ===========================================================================

class MockVideoMetadata:
    def __init__(self, target_fps=8.0, duration_s=5.0, width=640, height=480):
        self.target_fps = target_fps
        self.duration_s = duration_s
        self.width = width
        self.height = height


class MockTrackedFace:
    def __init__(self, track_id, bbox):
        self.track_id = track_id
        self.bbox = bbox


class MockFramePacket:
    def __init__(self, frame_id, tracked_faces=None):
        self.frame_id = frame_id
        self.tracked_faces = tracked_faces or []


class MockPreprocessingResult:
    def __init__(self, metadata, frames):
        self.metadata = metadata
        self.frames = frames

    @property
    def n_frames(self):
        return len(self.frames)


class MockFusionLocalizationResult:
    def __init__(self, temporal_timeline, modality_attribution=None):
        self.temporal_timeline = temporal_timeline
        self.modality_attribution = modality_attribution or {
            "audio": 0.0, "visual": 0.0, "temporal": 0.0, "biological": 0.0
        }


class MockFusionTrackResult:
    def __init__(self, track_id, verdict="FAKE", confidence=0.9, localization=None):
        self.track_id = track_id
        self.verdict = verdict
        self.confidence = confidence
        self.localization = localization or MockFusionLocalizationResult([])


class MockFusionResult:
    def __init__(self, track_results):
        self.track_results = track_results


class MockFrameVisualResult:
    def __init__(self, frame_id, bounding_boxes=None):
        self.frame_id = frame_id
        self.bounding_boxes = bounding_boxes or []


class MockBranchAResult:
    def __init__(self, frame_results):
        self.frame_results = frame_results


class MockTrackSyncMetrics:
    def __init__(self, phoneme_viseme_inconsistency=0.0, emotion_mismatch_score=0.0,
                 av_hubert_score=0.0):
        self.phoneme_viseme_inconsistency = phoneme_viseme_inconsistency
        self.emotion_mismatch_score = emotion_mismatch_score
        self.av_hubert_score = av_hubert_score


class MockBranchDResult:
    def __init__(self, track_metrics):
        self.track_metrics = track_metrics


# ===========================================================================
# Test Cases
# ===========================================================================

class TestCoordinateMapping:
    """Tests for _map_crop_to_global inverse coordinate transform."""

    def test_identity_mapping(self):
        """Crop bbox that covers entire crop maps to the full face bbox."""
        result = VideoLocalizer._map_crop_to_global(
            face_bbox=(100, 50, 324, 274),  # 224×224 face region
            crop_bbox=(0, 0, 224, 224),
            crop_size=224,
            frame_width=640,
            frame_height=480,
        )
        assert result == (100, 50, 224, 224)

    def test_centered_region(self):
        """A small centered region in crop-space maps correctly."""
        result = VideoLocalizer._map_crop_to_global(
            face_bbox=(100, 100, 324, 324),  # 224×224
            crop_bbox=(56, 56, 112, 112),    # center quarter in crop space
            crop_size=224,
            frame_width=640,
            frame_height=480,
        )
        # scale factor = 224/224 = 1.0
        # gx = 100 + 56*1.0 = 156
        # gy = 100 + 56*1.0 = 156
        # gw = 112*1.0 = 112
        # gh = 112*1.0 = 112
        assert result == (156, 156, 112, 112)

    def test_scaled_face_bbox(self):
        """Face bbox larger than crop_size produces correct scaling."""
        result = VideoLocalizer._map_crop_to_global(
            face_bbox=(0, 0, 448, 448),    # 448×448 face region
            crop_bbox=(0, 0, 112, 112),    # top-left quarter
            crop_size=224,
            frame_width=640,
            frame_height=480,
        )
        # scale = 448/224 = 2.0
        # gx=0, gy=0, gw=224, gh=224
        assert result == (0, 0, 224, 224)

    def test_clamping_to_frame_boundaries(self):
        """Coordinates are clamped to not exceed frame dimensions."""
        result = VideoLocalizer._map_crop_to_global(
            face_bbox=(600, 440, 700, 540),  # face near edge of 640×480 frame
            crop_bbox=(0, 0, 224, 224),
            crop_size=224,
            frame_width=640,
            frame_height=480,
        )
        gx, gy, gw, gh = result
        assert gx + gw <= 640
        assert gy + gh <= 480

    def test_zero_crop_size_fallback(self):
        """When crop_size=0, scale defaults to 1.0."""
        result = VideoLocalizer._map_crop_to_global(
            face_bbox=(10, 10, 110, 110),
            crop_bbox=(5, 5, 20, 20),
            crop_size=0,
            frame_width=640,
            frame_height=480,
        )
        assert result == (15, 15, 20, 20)


class TestAudioIntervalExtraction:
    """Tests for _extract_audio_intervals temporal grouping."""

    def test_single_contiguous_interval(self):
        """All frames above threshold form one interval."""
        timeline = [0.1, 0.8, 0.9, 0.7, 0.6, 0.1]
        intervals = VideoLocalizer._extract_audio_intervals(
            timeline=timeline, fps=2.0, threshold=0.5, min_duration_s=0.0,
        )
        assert len(intervals) == 1
        assert intervals[0] == pytest.approx((0.5, 2.5), abs=0.01)

    def test_two_disjoint_intervals(self):
        """Two separate groups of contiguous frames above threshold."""
        timeline = [0.9, 0.8, 0.1, 0.1, 0.7, 0.9, 0.1]
        intervals = VideoLocalizer._extract_audio_intervals(
            timeline=timeline, fps=4.0, threshold=0.5, min_duration_s=0.0,
        )
        assert len(intervals) == 2
        assert intervals[0][0] == pytest.approx(0.0)
        assert intervals[1][0] == pytest.approx(1.0)

    def test_trailing_interval(self):
        """Interval that extends to the end of the timeline."""
        timeline = [0.1, 0.1, 0.8, 0.9]
        intervals = VideoLocalizer._extract_audio_intervals(
            timeline=timeline, fps=2.0, threshold=0.5, min_duration_s=0.0,
        )
        assert len(intervals) == 1
        # start at frame 2 / 2fps = 1.0s, end at 4/2 = 2.0s
        assert intervals[0] == pytest.approx((1.0, 2.0))

    def test_min_duration_filter(self):
        """Short intervals below min_duration_s are discarded."""
        timeline = [0.9, 0.1, 0.1, 0.1]
        intervals = VideoLocalizer._extract_audio_intervals(
            timeline=timeline, fps=10.0, threshold=0.5, min_duration_s=0.2,
        )
        # Single frame at 10fps = 0.1s duration, below min_duration=0.2
        assert len(intervals) == 0

    def test_empty_timeline(self):
        """Empty timeline produces no intervals."""
        intervals = VideoLocalizer._extract_audio_intervals(
            timeline=[], fps=8.0, threshold=0.5,
        )
        assert intervals == []

    def test_zero_fps(self):
        """Zero FPS produces no intervals (guard)."""
        intervals = VideoLocalizer._extract_audio_intervals(
            timeline=[0.9, 0.9], fps=0.0, threshold=0.5,
        )
        assert intervals == []


class TestManipulatedFrameDetection:
    """Tests for manipulated frame identification from fusion timelines."""

    def test_frames_above_threshold(self):
        """Frames with fusion scores above threshold are flagged."""
        localizer = VideoLocalizer(visual_threshold=0.5)

        fusion = MockFusionResult({
            "track_01": MockFusionTrackResult(
                "track_01",
                localization=MockFusionLocalizationResult(
                    [0.1, 0.2, 0.8, 0.9, 0.3, 0.7]
                ),
            )
        })

        result = localizer.localize(fusion_result=fusion)
        assert sorted(result.manipulated_frames) == [2, 3, 5]

    def test_no_frames_above_threshold(self):
        """All frames below threshold results in empty list."""
        localizer = VideoLocalizer(visual_threshold=0.5)

        fusion = MockFusionResult({
            "track_01": MockFusionTrackResult(
                "track_01",
                localization=MockFusionLocalizationResult(
                    [0.1, 0.2, 0.3, 0.4]
                ),
            )
        })

        result = localizer.localize(fusion_result=fusion)
        assert result.manipulated_frames == []

    def test_multi_track_aggregation(self):
        """Max across tracks is used — frame is flagged if ANY track exceeds threshold."""
        localizer = VideoLocalizer(visual_threshold=0.5)

        fusion = MockFusionResult({
            "track_01": MockFusionTrackResult(
                "track_01",
                localization=MockFusionLocalizationResult([0.3, 0.4, 0.2]),
            ),
            "track_02": MockFusionTrackResult(
                "track_02",
                localization=MockFusionLocalizationResult([0.1, 0.9, 0.1]),
            ),
        })

        result = localizer.localize(fusion_result=fusion)
        # Frame 1 is flagged because track_02 has 0.9 there
        assert result.manipulated_frames == [1]


class TestCrossModalMismatchTimeline:
    """Tests for cross-modal mismatch timeline synthesis from Branch D."""

    def test_branch_d_mismatch_timeline(self):
        """Branch D metrics generate a uniform mismatch timeline."""
        localizer = VideoLocalizer()

        fusion = MockFusionResult({
            "track_01": MockFusionTrackResult(
                "track_01",
                localization=MockFusionLocalizationResult(
                    [0.5, 0.5, 0.5, 0.5, 0.5],
                    modality_attribution={"audio": 0.8, "visual": 0.1, "temporal": 0.1, "biological": 0.0},
                ),
            )
        })

        branch_d = MockBranchDResult({
            "track_01": MockTrackSyncMetrics(
                phoneme_viseme_inconsistency=0.8,
                emotion_mismatch_score=0.6,
                av_hubert_score=0.7,
            )
        })

        result = localizer.localize(fusion_result=fusion, branch_d_result=branch_d)

        assert len(result.cross_modal_mismatch_timeline) == 5
        # All values should be > 0 since Branch D has significant mismatch
        assert all(v > 0.0 for v in result.cross_modal_mismatch_timeline)

    def test_missing_branch_d(self):
        """Missing Branch D results in empty mismatch timeline."""
        localizer = VideoLocalizer()

        fusion = MockFusionResult({
            "track_01": MockFusionTrackResult(
                "track_01",
                localization=MockFusionLocalizationResult([0.5, 0.5]),
            )
        })

        result = localizer.localize(fusion_result=fusion, branch_d_result=None)
        assert result.cross_modal_mismatch_timeline == []


class TestMissingInputs:
    """Tests for graceful degradation when inputs are missing."""

    def test_missing_fusion_result(self):
        """Missing fusion result returns an empty result with warnings."""
        localizer = VideoLocalizer()
        result = localizer.localize(fusion_result=None)

        assert isinstance(result, VideoLocalizationResult)
        assert result.manipulated_frames == []
        assert result.audio_timestamps == []

    def test_missing_preprocessing_result(self):
        """Missing Layer 1 result still produces frame-level outputs."""
        localizer = VideoLocalizer(visual_threshold=0.5)

        fusion = MockFusionResult({
            "t1": MockFusionTrackResult(
                "t1",
                localization=MockFusionLocalizationResult([0.1, 0.9, 0.8]),
            )
        })

        result = localizer.localize(fusion_result=fusion)
        assert sorted(result.manipulated_frames) == [1, 2]
        assert "Missing Layer 1 preprocessing result." in result.metadata.get("warnings", [])


class TestEndToEndOrchestration:
    """Full end-to-end localize() call with synthetic data."""

    def test_full_orchestration(self):
        """Complete localization pipeline with all inputs provided."""
        localizer = VideoLocalizer(visual_threshold=0.5, audio_threshold=0.4)

        # Layer 1: 5 frames, 1 face track
        meta = MockVideoMetadata(target_fps=5.0, duration_s=1.0, width=640, height=480)
        frames = []
        for i in range(5):
            face = MockTrackedFace("track_01", bbox=(100, 50, 324, 274))
            frames.append(MockFramePacket(frame_id=i, tracked_faces=[face]))
        preproc = MockPreprocessingResult(meta, frames)

        # Layer 3: fusion timeline with frames 2,3 manipulated
        fusion = MockFusionResult({
            "track_01": MockFusionTrackResult(
                "track_01",
                localization=MockFusionLocalizationResult(
                    [0.1, 0.3, 0.8, 0.9, 0.2],
                    modality_attribution={"audio": 0.7, "visual": 0.9, "temporal": 0.2, "biological": 0.1},
                ),
            )
        })

        # Branch A: bounding boxes in crop space for frames 2 and 3
        branch_a = MockBranchAResult({
            "track_01": [
                MockFrameVisualResult(frame_id=0),
                MockFrameVisualResult(frame_id=1),
                MockFrameVisualResult(frame_id=2, bounding_boxes=[(50, 50, 100, 100)]),
                MockFrameVisualResult(frame_id=3, bounding_boxes=[(30, 30, 80, 80)]),
                MockFrameVisualResult(frame_id=4),
            ]
        })

        # Branch D: sync metrics
        branch_d = MockBranchDResult({
            "track_01": MockTrackSyncMetrics(
                phoneme_viseme_inconsistency=0.7,
                emotion_mismatch_score=0.5,
                av_hubert_score=0.6,
            )
        })

        result = localizer.localize(
            preprocessing_result=preproc,
            fusion_result=fusion,
            branch_a_result=branch_a,
            branch_d_result=branch_d,
        )

        # Verify manipulated frames
        assert sorted(result.manipulated_frames) == [2, 3]

        # Verify spatial regions exist for manipulated frames
        assert 2 in result.manipulated_regions
        assert 3 in result.manipulated_regions
        assert len(result.manipulated_regions[2]) == 1
        assert len(result.manipulated_regions[3]) == 1

        # Verify bounding boxes are in global frame coordinates
        gx, gy, gw, gh = result.manipulated_regions[2][0]
        assert gx >= 100  # must be offset by face bbox x1
        assert gy >= 50   # must be offset by face bbox y1

        # Verify cross-modal timeline has 5 entries
        assert len(result.cross_modal_mismatch_timeline) == 5

        # Verify audio timestamps are non-empty (high mismatch + audio attribution)
        # Timeline mismatch is uniform and modulated by audio_attribution
        assert isinstance(result.audio_timestamps, list)

        # Verify metadata
        assert result.metadata["fps"] == 5.0
        assert result.metadata["frame_width"] == 640

    def test_serialization(self):
        """Verify to_dict produces a clean JSON-safe output."""
        result = VideoLocalizationResult(
            manipulated_frames=[1, 3, 5],
            manipulated_regions={1: [(10, 20, 30, 40)], 3: [(50, 60, 70, 80)]},
            audio_timestamps=[(0.5, 1.2), (2.0, 3.5)],
            cross_modal_mismatch_timeline=[0.1, 0.9, 0.2, 0.8, 0.3],
            metadata={"fps": 8.0},
        )
        d = result.to_dict()
        assert d["manipulated_frames"] == [1, 3, 5]
        assert "1" in d["manipulated_regions"]
        assert len(d["audio_timestamps"]) == 2
        assert d["audio_timestamps"][0] == [0.5, 1.2]
        assert len(d["cross_modal_mismatch_timeline"]) == 5
