"""FakeAVCeleb dataset loader.

FakeAVCeleb contains real and deepfake audio-visual content
with both voice cloning and face manipulation.
"""

from __future__ import annotations

import logging
from pathlib import Path

from datasets.base import AudioDeepfakeDataset, AudioSample

logger = logging.getLogger(__name__)


class FakeAVCelebDataset(AudioDeepfakeDataset):
    """FakeAVCeleb dataset loader.

    Directory structure:
        FakeAVCeleb/
            Real/
                *.mp4 or *.wav
            Fake/
                voice_clone/
                    *.mp4 or *.wav
                face_swap/
                    ...
    """

    def _load_filelist(self) -> None:
        root = Path(self.root_dir)

        real_dir = root / "Real"
        if real_dir.exists():
            for fpath in sorted(real_dir.glob("**/*.wav")):
                self.samples.append(
                    AudioSample(
                        audio_path=str(fpath),
                        label=0,
                        sample_id=fpath.stem,
                        dataset_name="FakeAVCeleb",
                        manipulation_type="bonafide",
                    )
                )

        fake_dir = root / "Fake"
        if fake_dir.exists():
            for subdir in sorted(fake_dir.iterdir()):
                if subdir.is_dir():
                    for fpath in sorted(subdir.glob("**/*.wav")):
                        self.samples.append(
                            AudioSample(
                                audio_path=str(fpath),
                                label=1,
                                sample_id=fpath.stem,
                                dataset_name="FakeAVCeleb",
                                manipulation_type=subdir.name,
                            )
                        )

        if not self.samples:
            logger.warning(f"No audio files found in {root}. "
                           "FakeAVCeleb contains .mp4 files — extracting audio recommended")
            for fpath in sorted(root.glob("**/*.mp4")):
                self.samples.append(
                    AudioSample(
                        audio_path=str(fpath),
                        label=1 if "Fake" in str(fpath) else 0,
                        sample_id=fpath.stem,
                        dataset_name="FakeAVCeleb",
                        manipulation_type=fpath.parent.name,
                    )
                )
