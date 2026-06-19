"""
DeepGuard Platform — Video Detection Subsystem (VDS)
Module  : engines.video-engine.preprocessing.video_ingestor
Layer   : Layer 1 — Temporal Preprocessing
Task    : Frame extraction & Audio extraction via FFmpeg

Architecture Reference (DeepGuard_Platform_Architecture.docx — VDS Layer 1):
  - Frame extraction : FFmpeg pipe → raw RGB24 numpy arrays (no per-frame disk I/O
                       in fast mode; optional JPEG flush when save_frames=True)
  - Audio extraction : FFmpeg transcode → WAV 16 kHz, mono, PCM s16le
  - Metadata probe   : `ffprobe` JSON query for native fps, resolution, duration,
                       codec strings, and audio stream presence

Design decisions:
  1. Uses `ffmpeg-python` (thin Python wrapper around the FFmpeg binary) for clean
     subprocess management — no manual pipe byte-wrangling.
  2. FFmpeg binary is validated at import time with a helpful error if missing.
  3. Frame sub-sampling is achieved via the `fps` filter in the FFmpeg command —
     we do NOT decode all frames and then discard; this is far faster.
  4. With save_frames=True, frames are written as JPEG (quality=95) under
     workspace_dir/frames/. With save_frames=False (default), only the numpy
     array is returned and no disk I/O occurs per-frame.
  5. Audio is always extracted to workspace_dir/audio.wav when an audio stream
     exists.  Downstream audio-engine modules consume this WAV directly.

Usage:
    from vid_preprocessing.video_ingestor import VideoIngestor

    ingestor = VideoIngestor(target_fps=8.0, audio_sample_rate=16000)
    meta, frames = ingestor.ingest("path/to/video.mp4", workspace_dir="tmp/run_01")

    for fp in frames:
        print(fp.frame_id, fp.timestamp_ms, fp.rgb_array.shape)

    if meta.has_audio:
        print("Audio extracted to:", meta.audio_path)
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .result_types import FramePacket, VideoMetadata

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FFmpeg binary validation
# ---------------------------------------------------------------------------

def _require_ffmpeg() -> Tuple[str, str]:
    """
    Verify that both `ffmpeg` and `ffprobe` binaries are reachable on PATH.

    Returns:
        Tuple of (ffmpeg_path, ffprobe_path) as resolved strings.

    Raises:
        RuntimeError: with clear install instructions if either binary is absent.
    """
    ffmpeg  = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")

    missing = []
    if ffmpeg  is None: missing.append("ffmpeg")
    if ffprobe is None: missing.append("ffprobe")

    if missing:
        binaries = " and ".join(missing)
        raise RuntimeError(
            f"[VideoIngestor] Required binary not found in PATH: {binaries}\n\n"
            "Install FFmpeg (includes ffprobe) from your package manager:\n"
            "  Windows  : winget install --id=Gyan.FFmpeg  -e\n"
            "             OR  https://www.gyan.dev/ffmpeg/builds/\n"
            "  macOS    : brew install ffmpeg\n"
            "  Ubuntu   : sudo apt install ffmpeg\n\n"
            "After installing, ensure the bin/ folder is in your system PATH,\n"
            "then restart your terminal / Python process."
        )

    logger.debug("FFmpeg binary : %s", ffmpeg)
    logger.debug("FFprobe binary: %s", ffprobe)
    return ffmpeg, ffprobe


# ---------------------------------------------------------------------------
# VideoIngestor
# ---------------------------------------------------------------------------

class VideoIngestor:
    """
    FFmpeg-based frame and audio extractor for the VDS Layer 1 pipeline.

    Produces:
        - VideoMetadata  : static properties of the source video.
        - List[FramePacket] : extracted frames as RGB numpy arrays,
          optionally flushed to JPEG on disk.

    Args:
        target_fps (float):
            Frames per second to extract. Values lower than native_fps cause
            sub-sampling (e.g. 8.0 for forensic scan mode). Use 0.0 to extract
            at the native frame rate.
        audio_sample_rate (int):
            Sample rate for the extracted WAV file. Default: 16 000 Hz — the
            canonical rate for speech / audio deepfake models.
        save_frames (bool):
            If True, each extracted frame is saved as a JPEG file under
            `workspace_dir/frames/frame_XXXXXX.jpg`. The FramePacket will
            have both `rgb_array` and `frame_path` populated.
            If False (default), only `rgb_array` is populated; no disk I/O.
        jpeg_quality (int):
            JPEG compression quality [1–100] used when save_frames=True. Default: 95.
        ffmpeg_loglevel (str):
            FFmpeg subprocess log level. 'error' suppresses most output.
            Use 'info' or 'verbose' for debugging FFmpeg issues.
    """

    def __init__(
        self,
        target_fps:        float = 8.0,
        audio_sample_rate: int   = 16_000,
        save_frames:       bool  = False,
        jpeg_quality:      int   = 95,
        ffmpeg_loglevel:   str   = "error",
    ) -> None:
        self.target_fps        = target_fps
        self.audio_sample_rate = audio_sample_rate
        self.save_frames       = save_frames
        self.jpeg_quality      = jpeg_quality
        self.ffmpeg_loglevel   = ffmpeg_loglevel

        # Validate FFmpeg immediately — fail fast, not during ingest()
        self._ffmpeg_bin, self._ffprobe_bin = _require_ffmpeg()

        logger.info(
            "VideoIngestor ready | target_fps=%.2f | audio_sr=%d | save_frames=%s",
            target_fps, audio_sample_rate, save_frames,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(
        self,
        video_path:    str | Path,
        workspace_dir: str | Path,
    ) -> Tuple[VideoMetadata, List[FramePacket]]:
        """
        Full ingestion pipeline: probe → extract frames → extract audio.

        Args:
            video_path    : Path to the source video file (mp4, avi, mkv, mov …).
            workspace_dir : Directory where audio.wav (and optionally frames/)
                            will be written. Created if it does not exist.

        Returns:
            (VideoMetadata, List[FramePacket])

        Raises:
            FileNotFoundError : video_path does not exist.
            RuntimeError      : FFmpeg subprocess fails.
        """
        video_path    = Path(video_path).resolve()
        workspace_dir = Path(workspace_dir).resolve()

        if not video_path.exists():
            raise FileNotFoundError(f"[VideoIngestor] Video not found: {video_path}")

        workspace_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Ingesting video: %s", video_path)

        # ── 1. Probe metadata ────────────────────────────────────────────
        probe = self._probe(video_path)
        width, height, native_fps, duration_s, has_audio, codec_v, codec_a = probe

        # ── 2. Extract frames ────────────────────────────────────────────
        frames = self._extract_frames(
            video_path=video_path,
            width=width,
            height=height,
            native_fps=native_fps,
            workspace_dir=workspace_dir,
        )

        # ── 3. Extract audio ─────────────────────────────────────────────
        audio_path: Optional[Path] = None
        if has_audio:
            audio_path = self._extract_audio(video_path, workspace_dir)
        else:
            logger.info("No audio stream detected — skipping audio extraction.")

        # ── 4. Assemble metadata ─────────────────────────────────────────
        effective_fps = self.target_fps if self.target_fps > 0 else native_fps
        meta = VideoMetadata(
            source_path=video_path,
            duration_s=duration_s,
            native_fps=native_fps,
            width=width,
            height=height,
            total_frames_native=int(round(native_fps * duration_s)),
            total_frames_extracted=len(frames),
            target_fps=effective_fps,
            audio_path=audio_path,
            has_audio=has_audio and audio_path is not None,
            audio_sample_rate=self.audio_sample_rate,
            codec_video=codec_v,
            codec_audio=codec_a,
        )

        logger.info(
            "Ingestion complete | frames=%d | duration=%.2fs | audio=%s",
            len(frames), duration_s, meta.has_audio,
        )
        return meta, frames

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _probe(self, video_path: Path) -> Tuple[int, int, float, float, bool, str, str]:
        """
        Run ffprobe to extract video properties as a JSON dict.

        Returns:
            (width, height, native_fps, duration_s, has_audio, codec_video, codec_audio)
        """
        cmd = [
            self._ffprobe_bin,
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            "-show_format",
            str(video_path),
        ]
        logger.debug("ffprobe command: %s", " ".join(cmd))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=60,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"[VideoIngestor] ffprobe failed for {video_path}:\n{exc.stderr}"
            ) from exc

        info    = json.loads(result.stdout)
        streams = info.get("streams", [])
        fmt     = info.get("format", {})

        # Locate video and audio streams
        video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
        audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

        if video_stream is None:
            raise ValueError(f"[VideoIngestor] No video stream found in: {video_path}")

        # Width / height
        width  = int(video_stream.get("width",  0))
        height = int(video_stream.get("height", 0))

        # Native FPS — stored as fraction string e.g. "30000/1001"
        fps_str = video_stream.get("r_frame_rate", "25/1")
        try:
            num, den = fps_str.split("/")
            native_fps = float(num) / float(den)
        except (ValueError, ZeroDivisionError):
            native_fps = 25.0
            logger.warning("Could not parse fps '%s' — defaulting to 25.0", fps_str)

        # Duration
        duration_s = float(
            video_stream.get("duration")
            or fmt.get("duration")
            or 0.0
        )

        codec_v = video_stream.get("codec_name", "unknown")
        codec_a = audio_stream.get("codec_name", "unknown") if audio_stream else "none"
        has_audio = audio_stream is not None

        logger.info(
            "Probe result | %dx%d @ %.3f fps | %.2fs | audio=%s | vcodec=%s",
            width, height, native_fps, duration_s, has_audio, codec_v,
        )
        return width, height, native_fps, duration_s, has_audio, codec_v, codec_a

    def _extract_frames(
        self,
        video_path:   Path,
        width:        int,
        height:       int,
        native_fps:   float,
        workspace_dir: Path,
    ) -> List[FramePacket]:
        """
        Extract frames from the video via FFmpeg pipe → numpy arrays.

        FFmpeg outputs raw RGB24 bytes, which are directly re-shaped into
        (H, W, 3) uint8 arrays — no intermediate image encoding/decoding.
        This is substantially faster than writing JPEG and reading back.

        The `fps` filter performs server-side sub-sampling so we only
        decode the frames we actually need.
        """
        effective_fps = self.target_fps if self.target_fps > 0 else native_fps

        # Build the ffmpeg command: video → raw RGB24 pipe
        fps_filter = f"fps={effective_fps}"
        cmd = [
            self._ffmpeg_bin,
            "-loglevel", self.ffmpeg_loglevel,
            "-i", str(video_path),
            "-vf", fps_filter,
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-",  # output to stdout
        ]
        logger.info(
            "Extracting frames | effective_fps=%.2f | resolution=%dx%d",
            effective_fps, width, height,
        )
        logger.debug("ffmpeg frame cmd: %s", " ".join(cmd))

        # Optionally prepare frames directory
        frames_dir: Optional[Path] = None
        if self.save_frames:
            frames_dir = workspace_dir / "frames"
            frames_dir.mkdir(exist_ok=True)

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            raise RuntimeError(
                f"[VideoIngestor] Failed to launch ffmpeg: {exc}"
            ) from exc

        frame_size_bytes = width * height * 3  # RGB24
        frames: List[FramePacket] = []
        frame_id = 0
        frame_interval_ms = 1000.0 / effective_fps

        try:
            while True:
                raw = proc.stdout.read(frame_size_bytes)
                if len(raw) < frame_size_bytes:
                    break  # EOF

                rgb = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
                timestamp_ms = frame_id * frame_interval_ms

                # Optionally flush JPEG to disk
                frame_path: Optional[Path] = None
                if self.save_frames and frames_dir is not None:
                    frame_path = frames_dir / f"frame_{frame_id:06d}.jpg"
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(
                        str(frame_path),
                        bgr,
                        [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
                    )

                frames.append(
                    FramePacket(
                        frame_id=frame_id,
                        timestamp_ms=timestamp_ms,
                        rgb_array=rgb,
                        frame_path=frame_path,
                    )
                )
                frame_id += 1

                if frame_id % 100 == 0:
                    logger.debug("Extracted %d frames so far …", frame_id)

        finally:
            proc.stdout.close()
            proc.wait()

        if proc.returncode not in (0, None):
            stderr_text = proc.stderr.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"[VideoIngestor] ffmpeg exited with code {proc.returncode}:\n{stderr_text}"
            )

        logger.info("Frame extraction complete | total_frames=%d", len(frames))
        return frames

    def _extract_audio(self, video_path: Path, workspace_dir: Path) -> Optional[Path]:
        """
        Transcode the audio stream to 16 kHz mono PCM WAV.

        Returns the path to the extracted WAV file, or None on failure.
        """
        audio_out = workspace_dir / "audio.wav"
        cmd = [
            self._ffmpeg_bin,
            "-loglevel", self.ffmpeg_loglevel,
            "-i", str(video_path),
            "-vn",                          # no video
            "-acodec", "pcm_s16le",         # PCM 16-bit signed little-endian
            "-ar", str(self.audio_sample_rate),  # sample rate
            "-ac", "1",                     # mono
            "-y",                           # overwrite output if exists
            str(audio_out),
        ]
        logger.info("Extracting audio → %s", audio_out)
        logger.debug("ffmpeg audio cmd: %s", " ".join(cmd))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=300,
            )
        except subprocess.CalledProcessError as exc:
            logger.warning(
                "Audio extraction failed (non-fatal): %s", exc.stderr[-500:]
            )
            return None

        if not audio_out.exists():
            logger.warning("Audio extraction produced no output file (non-fatal).")
            return None

        size_kb = audio_out.stat().st_size / 1024
        logger.info("Audio extracted | path=%s | size=%.1f KB", audio_out, size_kb)
        return audio_out

    def __repr__(self) -> str:
        return (
            f"VideoIngestor("
            f"target_fps={self.target_fps}, "
            f"audio_sample_rate={self.audio_sample_rate}, "
            f"save_frames={self.save_frames})"
        )
