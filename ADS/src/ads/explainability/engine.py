"""Explainability Engine for forensic analysis.

Provides:
- SHAP feature importance
- Attention visualization
- Spectrogram heatmaps
- Forensic timeline
- Report generation (PDF, JSON, HTML)
"""

from __future__ import annotations

import io
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ads.config.settings import settings
from ads.domain.entities import (
    AttentionMap,
    BranchOutput,
    ForensicReport,
    ManipulatedSegment,
    RiskLevel,
    ShapExplanation,
    ForensicTimeline,
)

logger = logging.getLogger(__name__)


class ExplainabilityEngine:
    """Generates explainable forensic reports for deepfake detection results.

    Combines SHAP analysis, attention visualization, spectrogram heatmaps,
    and structured report generation to provide actionable forensic evidence.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or {}
        self.use_shap = self.config.get("use_shap", settings.explainability.use_shap)
        self.use_attention = self.config.get("use_attention", settings.explainability.use_attention)
        self.use_heatmap = self.config.get("use_heatmap", settings.explainability.use_spectrogram_heatmap)
        self.max_features_shap = self.config.get("max_features_shap", settings.explainability.max_features_shap)
        self.background_samples = self.config.get("background_samples", settings.explainability.background_samples)
        self.generate_pdf = self.config.get("generate_pdf", settings.explainability.generate_pdf)
        self.generate_html = self.config.get("generate_html", settings.explainability.generate_html)
        self.generate_json = self.config.get("generate_json", settings.explainability.generate_json)

    def generate_report(
        self,
        analysis_data: Dict[str, Any],
        output_dir: Optional[str] = None,
    ) -> ForensicReport:
        """Generate a comprehensive forensic report.

        Args:
            analysis_data: Complete analysis results from detector
            output_dir: Optional directory to save report files

        Returns:
            ForensicReport with explanations, timeline, and findings
        """
        fake_prob = analysis_data.get("fake_probability", 0.0)
        branch_outputs = analysis_data.get("branch_outputs", {})
        manipulated_segments = analysis_data.get("manipulated_segments", [])
        timeline = analysis_data.get("timeline")
        analysis_id = analysis_data.get("analysis_id", "unknown")

        risk_score = fake_prob
        risk_level = self._compute_risk_level(fake_prob)

        shap_explanations = []
        if self.use_shap:
            shap_explanations = self._compute_shap_explanations(branch_outputs)

        attention_maps = []
        if self.use_attention:
            attention_maps = self._extract_attention_maps(branch_outputs)

        branch_summaries = {}
        for name, output in branch_outputs.items():
            if isinstance(output, dict):
                branch_summaries[name] = BranchOutput(
                    branch_name=name,
                    fake_probability=float(output.get("probability", output.get("confidence", 0.5))),
                    confidence=float(output.get("confidence", 0.5)),
                    metadata={k: v for k, v in output.items() if k != "embedding"},
                )

        executive_summary = self._generate_executive_summary(
            fake_prob, risk_level, len(manipulated_segments), branch_outputs
        )

        report = ForensicReport(
            analysis_id=analysis_id,
            executive_summary=executive_summary,
            risk_assessment=risk_level,
            risk_score=risk_score,
            technical_findings=self._build_technical_findings(analysis_data),
            branch_summaries=branch_summaries,
            manipulated_segments=manipulated_segments,
            timeline=timeline,
            shap_explanations=shap_explanations,
            attention_maps=attention_maps,
            confidence_score=1.0 - abs(fake_prob - 0.5) * 2,
            processing_time_ms=analysis_data.get("processing_time_ms", 0),
            model_version=analysis_data.get("model_version", "0.1.0"),
        )

        if output_dir:
            self._save_report(report, output_dir)

        return report

    def _compute_shap_explanations(
        self, branch_outputs: Dict[str, Any]
    ) -> List[ShapExplanation]:
        """Compute SHAP-style feature importance from branch outputs."""
        explanations = []

        for branch_name, output in branch_outputs.items():
            if isinstance(output, dict) and "confidence" in output:
                importance = float(output["confidence"])
                direction = "increases_fake" if importance > 0.5 else "increases_real"
                explanations.append(
                    ShapExplanation(
                        feature_name=branch_name,
                        importance=abs(importance - 0.5) * 2,
                        direction=direction,
                    )
                )

        explanations.sort(key=lambda x: x.importance, reverse=True)
        return explanations[: self.max_features_shap]

    def _extract_attention_maps(self, branch_outputs: Dict[str, Any]) -> List[AttentionMap]:
        """Extract attention weights from temporal branches."""
        attention_maps = []

        if "temporal" in branch_outputs:
            temporal_out = branch_outputs["temporal"]
            if isinstance(temporal_out, dict) and "attention_weights" in temporal_out:
                weights = temporal_out["attention_weights"]
                if isinstance(weights, torch.Tensor):
                    weights = weights.cpu().numpy()
                for t, w in enumerate(weights.flatten()[:50]):
                    attention_maps.append(
                        AttentionMap(
                            timestep=float(t),
                            attention_weight=float(w),
                            branch_name="temporal",
                        )
                    )

        return attention_maps

    def _generate_executive_summary(
        self,
        fake_prob: float,
        risk_level: RiskLevel,
        num_segments: int,
        branch_outputs: Dict[str, Any],
    ) -> str:
        """Generate human-readable executive summary."""
        verdict = "DEEPFAKE DETECTED" if fake_prob > 0.5 else "AUTHENTIC"
        confidence_desc = "high confidence" if abs(fake_prob - 0.5) > 0.3 else "moderate confidence"

        strong_branches = []
        for name, out in branch_outputs.items():
            prob = 0.0
            if isinstance(out, dict):
                prob = float(out.get("probability", out.get("confidence", 0.0)))
            if prob > 0.7:
                strong_branches.append(name)

        summary_parts = [
            f"Audio Forensic Analysis: {verdict}",
            f"The analysis indicates {verdict.lower()} with {confidence_desc}.",
        ]

        if strong_branches:
            summary_parts.append(
                f"Strong indicators from: {', '.join(strong_branches)}."
            )

        if num_segments > 0:
            summary_parts.append(
                f"{num_segments} manipulated segment(s) detected and localized."
            )

        if risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            summary_parts.append("IMMEDIATE REVIEW RECOMMENDED.")
        elif risk_level == RiskLevel.MEDIUM:
            summary_parts.append("FURTHER ANALYSIS RECOMMENDED.")

        return " ".join(summary_parts)

    def _build_technical_findings(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build structured technical findings from analysis data."""
        return {
            "fake_probability": analysis_data.get("fake_probability", 0.0),
            "is_deepfake": analysis_data.get("is_deepfake", False),
            "speaker_consistency": analysis_data.get("speaker_consistency_score", 1.0),
            "num_branches_activated": len(analysis_data.get("branch_outputs", {})),
            "audio_duration_seconds": analysis_data.get("duration_seconds", 0),
            "num_speech_segments": len(analysis_data.get("speech_segments", [])),
            "num_speakers": len(analysis_data.get("speaker_segments", [])),
        }

    def _compute_risk_level(self, fake_prob: float) -> RiskLevel:
        if fake_prob < 0.1:
            return RiskLevel.SAFE
        if fake_prob < 0.3:
            return RiskLevel.LOW
        if fake_prob < 0.6:
            return RiskLevel.MEDIUM
        if fake_prob < 0.85:
            return RiskLevel.HIGH
        return RiskLevel.CRITICAL

    def _save_report(self, report: ForensicReport, output_dir: str) -> None:
        """Save report in multiple formats."""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)

        base_name = f"report_{report.analysis_id[:8]}"

        if self.generate_json:
            json_path = path / f"{base_name}.json"
            with open(json_path, "w") as f:
                json.dump(report.model_dump(mode="json"), f, indent=2, default=str)
            logger.info(f"Report saved: {json_path}")

        if self.generate_html:
            html_path = path / f"{base_name}.html"
            with open(html_path, "w") as f:
                f.write(report.to_html())
            logger.info(f"HTML report saved: {html_path}")

        if self.generate_pdf:
            try:
                from fpdf import FPDF

                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                for line in report.to_markdown().split("\n"):
                    pdf.cell(200, 10, txt=line[:200], ln=True, align="L")
                pdf_path = path / f"{base_name}.pdf"
                pdf.output(str(pdf_path))
                logger.info(f"PDF report saved: {pdf_path}")
            except Exception as e:
                logger.warning(f"PDF generation failed: {e}")

    def generate_spectrogram_heatmap(
        self,
        spectrogram: torch.Tensor,
        manipulated_segments: List[ManipulatedSegment],
        sample_rate: int = 16000,
        hop_length: int = 256,
    ) -> np.ndarray:
        """Generate a spectrogram heatmap highlighting manipulated regions.

        Args:
            spectrogram: (freq, time) tensor
            manipulated_segments: List of detected manipulated segments
            sample_rate: Audio sample rate
            hop_length: STFT hop length

        Returns:
            RGB heatmap array (height, width, 3)
        """
        spec = spectrogram.cpu().numpy() if isinstance(spectrogram, torch.Tensor) else spectrogram
        spec = np.abs(spec)

        spec_norm = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
        heatmap = np.stack([spec_norm] * 3, axis=-1)

        for seg in manipulated_segments:
            start_frame = int(seg.start_time * sample_rate / hop_length)
            end_frame = int(seg.end_time * sample_rate / hop_length)
            start_frame = max(0, min(start_frame, spec.shape[1] - 1))
            end_frame = max(0, min(end_frame, spec.shape[1] - 1))

            heatmap[:, start_frame:end_frame, 0] = 1.0
            heatmap[:, start_frame:end_frame, 1] = 0.0
            heatmap[:, start_frame:end_frame, 2] = 0.0

        return heatmap
