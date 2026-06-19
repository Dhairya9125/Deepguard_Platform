import torch
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
import pytorch_lightning as pl
from datasets import load_dataset, Features, Value
import torchvision.transforms as T
import logging
import io
try:
    import av
except ImportError:
    av = None


logger = logging.getLogger(__name__)

# Custom features that bypass HuggingFace's Video decoder (which needs torchcodec + FFmpeg).
# The dataset stores the video column as a plain string file path — not a dict.
# The label column is also a string ('deepfake' / 'real'), not an integer.
_VIDEO_FEATURES = Features({
    "video": Value("string"),
    "label": Value("string"),
})

# Map string labels to binary floats (1.0 = fake, 0.0 = real)
_LABEL_MAP = {
    "deepfake": 1.0, "fake": 1.0, "synthetic": 1.0,
    "real": 0.0,     "authentic": 0.0, "original": 0.0,
}


# ---------------------------------------------------------------------------
# Synthetic (offline) dataset — used when HF_TOKEN is not set or for fast CI
# ---------------------------------------------------------------------------

class SyntheticDeepfakeDataset(IterableDataset):
    """
    Generates random tensors of the correct shape for a given modality.
    Zero network I/O — validates the full training loop without any downloads.

    Shapes produced:
        audio  : [1, 16000]
        image  : [3, 224, 224]
        video  : [T, 3, 224, 224]  (T = num_frames)
    """

    def __init__(self, modality: str = "video", num_samples: int = 64, num_frames: int = 8):
        super().__init__()
        self.modality = modality
        self.num_samples = num_samples
        self.num_frames = num_frames

    def __iter__(self):
        for i in range(self.num_samples):
            label = torch.tensor(float(i % 2), dtype=torch.float32)  # alternating 0/1

            if self.modality == "audio":
                yield torch.randn(1, 16000), label
            elif self.modality == "image":
                yield torch.randn(3, 224, 224), label
            elif self.modality == "video":
                yield torch.randn(self.num_frames, 3, 224, 224), label
            else:
                raise ValueError(f"Unknown modality: {self.modality}")


# ---------------------------------------------------------------------------
# Real HuggingFace streaming dataset
# ---------------------------------------------------------------------------

class StreamingDeepfakeDataset(IterableDataset):
    def __init__(self, modality='audio', split='train', max_samples=None):
        super().__init__()
        self.modality = modality
        self.split = split
        self.max_samples = max_samples
        
        if modality == 'audio':
            # Binary classification dataset: 0=Real, 1=Fake
            self.dataset = load_dataset("garystafford/deepfake-audio-detection", split=split, streaming=True)
        elif modality == 'image':
            # Binary classification dataset: 0=Real, 1=Fake
            self.dataset = load_dataset("insanescw/20K_real_and_deepfake_images", split=split, streaming=True)
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor()
            ])
        elif modality == 'video':
            # Load with raw-bytes features so HuggingFace does NOT call torchcodec internally.
            # We decode frames ourselves with PyAV below.
            try:
                self.dataset = load_dataset(
                    "UniDataPro/deepfake-videos-dataset",
                    split=split,
                    streaming=True,
                    features=_VIDEO_FEATURES,
                )
            except Exception:
                # Fallback: load without explicit features override (may work on some dataset revisions)
                self.dataset = load_dataset(
                    "UniDataPro/deepfake-videos-dataset",
                    split=split,
                    streaming=True,
                )
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor()
            ])
            self.num_frames = 8
        else:
            raise ValueError(f"Unsupported modality: {modality}")

    def __iter__(self):
        count = 0
        for item in self.dataset:
            if self.max_samples and count >= self.max_samples:
                break
                
            try:
                if self.modality == 'audio':
                    audio_array = item['audio']['array']
                    target_length = 16000 # 1 sec at 16kHz
                    if len(audio_array) > target_length:
                        audio_array = audio_array[:target_length]
                    else:
                        audio_array = np.pad(audio_array, (0, target_length - len(audio_array)))
                    
                    tensor = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0)
                    label = float(item['label'])
                    yield tensor, torch.tensor(label, dtype=torch.float32)
                    
                elif self.modality == 'image':
                    img = item['image'].convert('RGB')
                    tensor = self.transform(img)
                    label = float(item['label'])
                    yield tensor, torch.tensor(label, dtype=torch.float32)
                    
                elif self.modality == 'video':
                    if av is None:
                        raise ImportError("av is required for video streaming. Install with `pip install av`")
                    
                    # Dataset provides video as a local cached file path (string).
                    # Fall back to checking other common column names just in case.
                    video_data = item.get('video') or item.get('file') or item.get('path')

                    if isinstance(video_data, str) and video_data:
                        container = av.open(video_data)
                    elif isinstance(video_data, (bytes, bytearray)) and video_data:
                        container = av.open(io.BytesIO(video_data))
                    elif isinstance(video_data, dict):
                        raw_bytes = video_data.get('bytes')
                        path      = video_data.get('path')
                        if raw_bytes:
                            container = av.open(io.BytesIO(raw_bytes))
                        elif path:
                            container = av.open(path)
                        else:
                            raise ValueError("video dict has neither bytes nor path")
                    else:
                        raise ValueError(f"Unrecognized video data: type={type(video_data)} value={repr(video_data)[:80]}")

                    stream = container.streams.video[0]
                    total_frames = stream.frames
                    if total_frames <= 0:
                        total_frames = 60 # Fallback estimate if container doesn't know

                    indices = set(np.linspace(0, total_frames - 1, self.num_frames, dtype=int).tolist())
                    frames = []
                    frame_idx = 0
                    
                    for frame in container.decode(stream):
                        if frame_idx in indices:
                            img = frame.to_image().convert('RGB')
                            tensor = self.transform(img)
                            frames.append(tensor)
                            if len(frames) == self.num_frames:
                                break
                        frame_idx += 1
                    
                    container.close()
                        
                    while len(frames) < self.num_frames:
                        frames.append(frames[-1] if len(frames) > 0 else torch.zeros(3, 224, 224))
                        
                    video_tensor = torch.stack(frames) # [T, C, H, W]
                    
                    # Labels may be string ('deepfake'/'real') or numeric
                    raw_label = item.get('label', item.get('fake', 0))
                    if isinstance(raw_label, str):
                        label_val = _LABEL_MAP.get(raw_label.lower().strip(), 0.0)
                    else:
                        label_val = float(raw_label)
                    yield video_tensor, torch.tensor(label_val, dtype=torch.float32)
                    
                count += 1
            except Exception as e:
                logger.debug(f"Skipping corrupted sample: {e}")
                continue


# ---------------------------------------------------------------------------
# Lightning DataModule — picks synthetic vs. real based on `synthetic` flag
# ---------------------------------------------------------------------------

class StreamingDataModule(pl.LightningDataModule):
    def __init__(self, modality: str = 'audio', batch_size: int = 16, synthetic: bool = False):
        super().__init__()
        self.modality = modality
        self.batch_size = batch_size
        self.synthetic = synthetic

    def setup(self, stage=None):
        if self.synthetic:
            logger.info("Using SYNTHETIC dataset (no network I/O) for pipeline validation.")
            self.train_dataset = SyntheticDeepfakeDataset(self.modality, num_samples=256)
            self.val_dataset   = SyntheticDeepfakeDataset(self.modality, num_samples=64)
        else:
            self.train_dataset = StreamingDeepfakeDataset(self.modality, 'train')
            self.val_dataset   = StreamingDeepfakeDataset(self.modality, 'train', max_samples=100)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
