"""
ADS — Forensic Report Generator
Generates a natural-language forensic summary of the ADS detection result.
"""
from typing import List
from ..aud_domain.entities import AudioDetectionResult, BranchResult


class ForensicReportGenerator:
    """Generates a human-readable forensic narrative from detection results."""

    BRANCH_DESCRIPTIONS = {
        "spectral_cnn":        "Spectral CNN (vocoder/TTS artifact analysis)",
        "temporal_splice":     "Temporal Splice Detector (edit-boundary analysis)",
        "voice_biometrics":    "Voice Biometrics (pitch and prosody analysis)",
        "wavlm":               "WavLM Deep Representations (self-supervised features)",
        "xlsr":                "XLS-R Cross-Lingual Analysis (multilingual detection)",
        "diffusion_detector":  "Diffusion Model Artifact Detector",
        "adversarial_detector": "Adversarial Perturbation Detector",
    }

    def generate(self, result: AudioDetectionResult) -> str:
        lines = [
            "# DeepGuard — Audio Forensic Analysis Report",
            "",
            f"**Verdict**: {result.verdict}",
            f"**Fake Probability**: {result.fake_probability:.1%}",
            f"**Authenticity Score**: {result.authenticity_score:.1f}/100",
            f"**Confidence**: {result.confidence:.1f}%",
            f"**Duration**: {result.duration_s:.2f}s",
            f"**Processing Time**: {result.processing_time_s:.3f}s",
            "",
            "## Branch Analysis",
            "",
        ]

        for branch in result.branch_results:
            desc = self.BRANCH_DESCRIPTIONS.get(branch.branch_name, branch.branch_name)
            status = "⚠️ ERROR" if branch.error else (
                "🔴 HIGH RISK" if branch.fake_score >= 0.75 else
                "🟡 UNCERTAIN" if branch.fake_score >= 0.45 else
                "🟢 LOW RISK"
            )
            lines.append(f"- **{desc}**: {status} (score={branch.fake_score:.3f}, confidence={branch.confidence:.2f})")
            if branch.error:
                lines.append(f"  - Error: {branch.error}")

        if result.manipulation_segments:
            lines += [
                "",
                "## Detected Manipulation Segments",
                "",
            ]
            for seg in result.manipulation_segments:
                lines.append(
                    f"- [{seg['start_s']:.2f}s – {seg['end_s']:.2f}s]: "
                    f"fake_score={seg['fake_score']:.3f}"
                )

        lines += [
            "",
            "## Assessment",
            "",
        ]

        if result.verdict == "FAKE":
            lines.append(
                "This audio sample shows strong indicators of AI synthesis or manipulation. "
                "Multiple forensic branches have flagged anomalies consistent with "
                "voice cloning, TTS generation, or audio splicing."
            )
        elif result.verdict == "UNCERTAIN":
            lines.append(
                "This audio sample shows ambiguous signals. Some branches detected "
                "potential manipulation artifacts, but the evidence is not conclusive. "
                "Manual review is recommended."
            )
        else:
            lines.append(
                "This audio sample shows no significant indicators of AI manipulation. "
                "Spectral, temporal, and biometric features are consistent with authentic speech."
            )

        return "\n".join(lines)
