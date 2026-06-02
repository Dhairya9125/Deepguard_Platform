"""ASVspoof 2021 dataset loader.

The ASVspoof 2021 dataset includes logical access (LA) and
deepfake (DF) subsets for audio deepfake detection.

Expected directory structure:
    ASVspoof2021/
        LA/
            ASVspoof2021_LA_eval/
                flac/
                ...
            ASVspoof2021_LA_eval.trn.txt (protocol)
        DF/
            ...
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from datasets.base import AudioDeepfakeDataset, AudioSample

logger = logging.getLogger(__name__)


class ASVspoofDataset(AudioDeepfakeDataset):
    """ASVspoof 2021 dataset loader.

    Supports LA (Logical Access) and DF (Deepfake) subsets.
    """

    PROTOCOL_FILES = {
        "train": {
            "LA": "ASVspoof2021_LA_train.trn.txt",
            "DF": "ASVspoof2021_DF_train.trn.txt",
        },
        "dev": {
            "LA": "ASVspoof2021_LA_dev.trn.txt",
            "DF": "ASVspoof2021_DF_dev.trn.txt",
        },
        "eval": {
            "LA": "ASVspoof2021_LA_eval.trn.txt",
            "DF": "ASVspoof2021_DF_eval.trn.txt",
        },
    }

    def __init__(
        self,
        root_dir: str,
        subset: str = "train",
        task: str = "LA",
        **kwargs,
    ) -> None:
        self.task = task.upper()
        super().__init__(root_dir=root_dir, subset=subset, **kwargs)

    def _load_filelist(self) -> None:
        subset_dir = {"train": "train", "dev": "dev", "eval": "eval"}.get(
            self.subset, self.subset
        )
        protocol_name = self.PROTOCOL_FILES.get(self.subset, {}).get(self.task)
        if not protocol_name:
            protocol_path = Path(self.root_dir) / self.task / f"{self.subset}" / f"ASVspoof2021_{self.task}_{self.subset}.trn.txt"
        else:
            protocol_path = Path(self.root_dir) / self.task / subset_dir / protocol_name

        audio_dir = Path(self.root_dir) / self.task / subset_dir / "flac"

        if protocol_path.exists():
            self._load_from_protocol(protocol_path, audio_dir)
        else:
            logger.info(f"Protocol file not found at {protocol_path}, scanning directory")
            self._scan_directory(audio_dir)

    def _load_from_protocol(self, protocol_path: Path, audio_dir: Path) -> None:
        with open(protocol_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                speaker_id = parts[0]
                filename = parts[1]
                label_str = parts[3]

                if self.task == "LA":
                    label = 0 if label_str == "bonafide" else 1
                else:
                    label = 0 if label_str == "bonafide" else 1

                audio_path = audio_dir / f"{filename}.flac"
                if audio_path.exists():
                    self.samples.append(
                        AudioSample(
                            audio_path=str(audio_path),
                            label=label,
                            sample_id=filename,
                            speaker_id=speaker_id,
                            dataset_name="ASVspoof2021",
                            manipulation_type=label_str,
                        )
                    )

    def _scan_directory(self, audio_dir: Path) -> None:
        if not audio_dir.exists():
            logger.warning(f"Audio directory not found: {audio_dir}")
            return

        for fpath in sorted(audio_dir.glob("*.flac")):
            self.samples.append(
                AudioSample(
                    audio_path=str(fpath),
                    label=-1,
                    sample_id=fpath.stem,
                    dataset_name="ASVspoof2021",
                )
            )

    @property
    def num_real(self) -> int:
        return sum(1 for s in self.samples if s.label == 0)

    @property
    def num_fake(self) -> int:
        return sum(1 for s in self.samples if s.label == 1)
