"""CLI interface for the Audio Deepfake Detection System.

Provides command-line access to detection, analysis batch processing,
and system management.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import click

from ads.application.detector import DeepfakeDetector
from ads.config.settings import settings


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """ADS - Audio Deepfake Detection System CLI."""
    pass


@cli.command()
@click.argument("audio_path", type=click.Path(exists=True))
@click.option("--report", "-r", is_flag=True, help="Generate forensic report")
@click.option("--output", "-o", type=click.Path(), help="Output directory for report")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
def detect(audio_path: str, report: bool, output: Optional[str], json_output: bool):
    """Detect deepfake in audio file."""
    detector = DeepfakeDetector()
    result = detector.analyze(audio_path, return_report=report, output_dir=output)

    if json_output:
        click.echo(json.dumps(result.model_dump(mode="json"), indent=2, default=str))
    else:
        click.echo(f"\n{'='*60}")
        click.echo(f"  ADS - Audio Deepfake Detection Result")
        click.echo(f"{'='*60}")
        click.echo(f"  File:         {audio_path}")
        click.echo(f"  Status:       {result.status.value}")
        click.echo(f"  Deepfake:     {'YES' if result.is_deepfake else 'NO'}")
        click.echo(f"  Probability:  {result.fake_probability:.2%}")
        click.echo(f"  Confidence:   {result.confidence_score:.2%}")
        click.echo(f"  Speaker Cons: {result.speaker_consistency_score:.2%}")
        click.echo(f"  Segments:     {len(result.manipulated_segments)}")
        click.echo(f"  Time:         {result.processing_time_ms:.0f}ms")
        click.echo(f"{'='*60}\n")


@cli.command()
@click.argument("audio_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output directory")
def preprocess(audio_path: str, output: Optional[str]):
    """Preprocess audio and show intermediate results."""
    from ads.preprocessing.pipeline import AudioPreprocessingPipeline

    pipeline = AudioPreprocessingPipeline()
    result = pipeline.process(audio_path)

    click.echo(f"\nPreprocessing Results:")
    click.echo(f"  Duration: {result['metadata'].duration_seconds:.2f}s")
    click.echo(f"  Sample Rate: {result['metadata'].sample_rate}Hz")
    click.echo(f"  Channels: {result['metadata'].channels}")
    click.echo(f"  Speech Segments: {len(result['speech_segments'])}")
    click.echo(f"  Speaker Segments: {len(result['speaker_segments'])}")
    click.echo(f"  Chunks: {len(result['chunks'])}")
    click.echo(f"  Preprocessing Time: {result['preprocessing_time_ms']:.0f}ms")


@cli.command()
@click.argument("audio_path", type=click.Path(exists=True))
def features(audio_path: str):
    """Extract and show signal features."""
    import torch
    import torchaudio

    from ads.signal_processing.features import SignalProcessor

    waveform, sr = torchaudio.load(audio_path)
    processor = SignalProcessor()
    features = processor.compute_all_features(waveform, sr)

    click.echo(f"\nSignal Features:")
    click.echo(f"  Waveform:      {features.waveform.shape}")
    click.echo(f"  Mel Spec:      {features.mel_spectrogram.shape}")
    click.echo(f"  MFCC:          {features.mfcc.shape}")
    click.echo(f"  LFCC:          {features.lfcc.shape}")
    click.echo(f"  CQCC:          {features.cqcc.shape}")
    click.echo(f"  Chroma:        {features.chroma.shape}")
    click.echo(f"  Spectral Cont: {features.spectral_contrast.shape}")
    click.echo(f"  Pitch:         {features.pitch.shape}")
    click.echo(f"  Harmonic:      {features.harmonic.shape}")


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind")
@click.option("--port", default=8000, help="Port to bind")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
def serve(host: str, port: int, reload: bool):
    """Start the ADS API server."""
    import uvicorn

    click.echo(f"Starting ADS API on {host}:{port}")
    uvicorn.run("src.api.app:app", host=host, port=port, reload=reload)


@cli.command()
@click.argument("paths", nargs=-1, type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output JSON file")
def batch(paths: tuple, output: Optional[str]):
    """Batch process multiple audio files."""
    detector = DeepfakeDetector()
    results = []

    with click.progressbar(paths, label="Processing") as bar:
        for path in bar:
            result = detector.analyze(path)
            results.append({
                "file": path,
                "is_deepfake": result.is_deepfake,
                "probability": result.fake_probability,
                "confidence": result.confidence_score,
            })

    if output:
        Path(output).write_text(json.dumps(results, indent=2))
        click.echo(f"Results saved to {output}")
    else:
        click.echo(json.dumps(results, indent=2))


if __name__ == "__main__":
    cli()
