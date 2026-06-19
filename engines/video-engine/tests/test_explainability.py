"""
Tests for VDS Layer 5 — Explainability.

Test strategy:
  - Generate synthetic mock data for Layers 1-4.
  - Verify that VDSVisualizer correctly creates plot files.
  - Verify that ForensicReportGenerator correctly outputs MD and JSON files.
  - Verify that ExplainabilityEngine end-to-end orchestration works without exceptions
    and returns valid paths.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vid_explainability.explainability_engine import ExplainabilityEngine
from vid_explainability.result_types import ExplainabilityResult


# ===========================================================================
# Mock Stubs for Layer Inputs
# ===========================================================================

class MockFramePacket:
    def __init__(self, frame_id, rgb_array=None):
        self.frame_id = frame_id
        # Provide a synthetic 100x100 RGB image for cv2 tests
        self.rgb_array = rgb_array if rgb_array is not None else np.zeros((100, 100, 3), dtype=np.uint8)


class MockPreprocessingResult:
    def __init__(self):
        self.frames = [
            MockFramePacket(frame_id=0),
            MockFramePacket(frame_id=1),
            MockFramePacket(frame_id=2),
        ]


class MockTrackMetrics:
    def __init__(self, bvp_signal, estimated_heart_rate):
        self.bvp_signal = bvp_signal
        self.estimated_heart_rate = estimated_heart_rate


class MockBranchEResult:
    def __init__(self):
        self.track_metrics = {
            "track_01": MockTrackMetrics(
                bvp_signal=[0.1, 0.5, 0.9, 0.4, 0.2, 0.1, 0.6],
                estimated_heart_rate=72.5
            )
        }


class MockLocalizationTrack:
    def __init__(self, temporal_timeline):
        self.localization = type("Loc", (), {"temporal_timeline": temporal_timeline})
        self.verdict = "FAKE"
        self.confidence = 0.95
        self.modality_culprits = {"audio": 0.8, "visual": 0.1}


class MockFusionResult:
    def __init__(self):
        self.overall_verdict = "FAKE"
        self.overall_confidence = 0.92
        self.track_results = {
            "track_01": MockLocalizationTrack([0.1, 0.2, 0.9, 0.8, 0.1])
        }


class MockLocalizationResult:
    def __init__(self):
        self.metadata = {"fps": 2.0, "duration_s": 2.5, "n_frames_analyzed": 5}
        self.manipulated_frames = [2, 3]
        self.manipulated_regions = {
            2: [(10, 10, 40, 40)],
            3: [(15, 15, 45, 45)],
        }
        self.cross_modal_mismatch_timeline = [0.1, 0.1, 0.8, 0.9, 0.2]
        self.audio_timestamps = [(1.0, 2.0)]


# ===========================================================================
# Test Cases
# ===========================================================================

@pytest.fixture
def output_dir(tmp_path):
    return tmp_path / "explain_output"


def test_visualizer_lip_sync_plot(output_dir):
    from vid_explainability.visualizer import VDSVisualizer
    viz = VDSVisualizer(output_dir)
    
    path = viz.plot_lip_sync_timeline(
        timeline=[0.1, 0.2, 0.9, 0.9, 0.1],
        audio_timestamps=[(1.0, 2.0)],
        fps=2.0
    )
    assert path is not None
    assert path.exists()
    assert path.suffix == ".png"


def test_visualizer_rppg_plot(output_dir):
    from vid_explainability.visualizer import VDSVisualizer
    viz = VDSVisualizer(output_dir)
    
    path = viz.plot_rppg_signals([0.1, 0.8, 0.2, 0.9], 65.0)
    assert path is not None
    assert path.exists()
    assert path.suffix == ".png"


def test_visualizer_temporal_plot(output_dir):
    from vid_explainability.visualizer import VDSVisualizer
    viz = VDSVisualizer(output_dir)
    
    path = viz.plot_temporal_inconsistencies([0.1, 0.2, 0.8, 0.3], 2.0)
    assert path is not None
    assert path.exists()
    assert path.suffix == ".png"


def test_visualizer_heatmaps(output_dir):
    from vid_explainability.visualizer import VDSVisualizer
    viz = VDSVisualizer(output_dir)
    
    preproc = MockPreprocessingResult()
    loc = MockLocalizationResult()
    
    paths = viz.render_frame_heatmaps(
        preprocessing_result=preproc,
        manipulated_regions=loc.manipulated_regions,
        manipulated_frames=loc.manipulated_frames,
    )
    
    # We mocked frames 0, 1, 2.
    # manipulated_frames are [2, 3].
    # Frame 2 exists in preproc, Frame 3 does not.
    # So we should only get 1 heatmap for frame 2.
    assert len(paths) == 1
    assert paths[0].exists()
    assert paths[0].name == "heatmap_frame_0002.png"


def test_report_generator(output_dir):
    from vid_explainability.report_generator import ForensicReportGenerator
    gen = ForensicReportGenerator(output_dir)
    
    fusion = MockFusionResult()
    loc = MockLocalizationResult()
    graphs = {
        "temporal": Path("temp.png"),
        "lip_sync": Path("lip.png"),
    }
    
    report_path = gen.generate_report(fusion, loc, graphs)
    assert report_path.exists()
    assert report_path.name == "forensic_summary.md"
    
    json_path = output_dir / "forensic_summary.json"
    assert json_path.exists()
    
    # Verify JSON content
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["overall_verdict"] == "FAKE"
        assert data["duration_s"] == 2.5
        assert len(data["audio_timestamps"]) == 1
        assert "track_01" in data["tracks"]


def test_explainability_engine_orchestration(output_dir):
    engine = ExplainabilityEngine(output_dir)
    
    result = engine.generate_explanation(
        preprocessing_result=MockPreprocessingResult(),
        branch_e_result=MockBranchEResult(),
        fusion_result=MockFusionResult(),
        localization_result=MockLocalizationResult(),
    )
    
    assert isinstance(result, ExplainabilityResult)
    
    # Check that paths were populated
    assert result.temporal_graph_path is not None
    assert result.lip_sync_graph_path is not None
    assert result.rppg_graph_path is not None
    assert result.report_path is not None
    assert len(result.heatmap_paths) == 1
    
    # Check that files actually exist on disk
    assert result.temporal_graph_path.exists()
    assert result.lip_sync_graph_path.exists()
    assert result.rppg_graph_path.exists()
    assert result.report_path.exists()
    assert result.heatmap_paths[0].exists()
    
    # Verify serialization
    d = result.to_dict()
    assert d["report_path"] == str(result.report_path)
    assert len(d["heatmap_paths"]) == 1
