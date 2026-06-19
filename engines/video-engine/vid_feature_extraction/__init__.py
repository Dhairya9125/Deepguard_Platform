"""
DeepGuard Platform — Video Detection Subsystem (VDS)
Package : engines.video-engine.feature_extraction
Layer   : Layer 2 — Temporal Feature Extraction

Branch A — Frame-Level Visual Analysis:
    FrameVisualAnalyzer        — IDS pipeline runner over every tracked face crop
    IDSComponents              — IDS model component container
    load_ids_components        — Factory: initialise all 9 IDS components for inference
    FrameVisualResult          — Per-face, per-frame IDS verdict dataclass
    BranchAResult              — Aggregated Branch A output (top-level)

Branch B — Temporal Consistency Analysis:
    TemporalConsistencyAnalyzer— Branch B main class (12-signal temporal analysis)
    TrackTemporalMetrics       — All temporal signals for one face track
    BranchBResult              — Aggregated Branch B output (top-level)

Branch C — Temporal Motion Modeling:
    TemporalMotionAnalyzer     — Branch C main class (cross-model + geometric motion analysis)
    TrackMotionMetrics         — All temporal motion metrics for one face track
    BranchCResult              — Aggregated Branch C output (top-level)

Branch D — Cross-Modal Synchronization:
    TemporalSyncAnalyzer       — Branch D main class (SyncNet + AV-HuBERT cross-modal analysis)
    TrackSyncMetrics           — All cross-modal sync metrics for one face track
    BranchDResult              — Aggregated Branch D output (top-level)

Branch E — rPPG Biological Signal Analysis:
    TemporalRPPGAnalyzer       — Branch E main class (PhysNet + DeepPhys rPPG analysis)
    TrackRPPGMetrics           — All rPPG metrics for one face track
    BranchEResult              — Aggregated Branch E output (top-level)

Branch F — Identity Continuity Tracking:
    TemporalContinuityAnalyzer — Branch F main class (Facenet identity + trajectory continuity analysis)
    TrackContinuityMetrics     — All continuity metrics for one face track
    BranchFResult              — Aggregated Branch F output (top-level)

Branch G — Scene Semantic Consistency:
    TemporalSemanticAnalyzer   — Branch G main class (WorldModelNet scene consistency analysis)
    SceneSemanticMetrics       — All semantic consistency metrics for one scene
    BranchGResult              — Aggregated Branch G output (top-level)
"""

# Branch A
from .frame_visual_analyzer         import FrameVisualAnalyzer
from .ids_adapter                   import IDSComponents, load_ids_components
from .result_types_l2               import BranchAResult, FrameVisualResult

# Branch B
from .temporal_consistency_analyzer import TemporalConsistencyAnalyzer
from .result_types_l2b              import BranchBResult, TrackTemporalMetrics

# Branch C
from .temporal_motion_analyzer      import TemporalMotionAnalyzer
from .result_types_l2c              import BranchCResult, TrackMotionMetrics

# Branch D
from .temporal_sync_analyzer        import TemporalSyncAnalyzer
from .result_types_l2d              import BranchDResult, TrackSyncMetrics

# Branch E
from .temporal_rppg_analyzer        import TemporalRPPGAnalyzer
from .result_types_l2e              import BranchEResult, TrackRPPGMetrics

# Branch F
from .temporal_continuity_analyzer  import TemporalContinuityAnalyzer
from .result_types_l2f              import BranchFResult, TrackContinuityMetrics

# Branch G
from .temporal_semantic_analyzer    import TemporalSemanticAnalyzer
from .result_types_l2g              import BranchGResult, SceneSemanticMetrics

__all__ = [
    # Branch A
    "FrameVisualAnalyzer",
    "IDSComponents",
    "load_ids_components",
    "FrameVisualResult",
    "BranchAResult",
    # Branch B
    "TemporalConsistencyAnalyzer",
    "TrackTemporalMetrics",
    "BranchBResult",
    # Branch C
    "TemporalMotionAnalyzer",
    "TrackMotionMetrics",
    "BranchCResult",
    # Branch D
    "TemporalSyncAnalyzer",
    "TrackSyncMetrics",
    "BranchDResult",
    # Branch E
    "TemporalRPPGAnalyzer",
    "TrackRPPGMetrics",
    "BranchEResult",
    # Branch F
    "TemporalContinuityAnalyzer",
    "TrackContinuityMetrics",
    "BranchFResult",
    # Branch G
    "TemporalSemanticAnalyzer",
    "SceneSemanticMetrics",
    "BranchGResult",
]
