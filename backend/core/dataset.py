import os
import torch
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Any, Optional
from collections import Counter


def default_training_jpeg_dir(data_root: str) -> Optional[str]:
    """
    Directory where ``prepare_finebadminton_20k.py --extract-training-frames`` writes
    ``{video_stem}_{frame_index}.jpg`` (same linspace as training). First existing path wins.
    """
    for rel in (
        os.path.join(data_root, "FineBadminton-20K", "dataset", "image"),
        os.path.join(data_root, "dataset", "image"),
    ):
        if os.path.isdir(rel):
            return rel
    return None


def get_class_weights(
    data_root: str,
    list_file: str,
    task: str = "stroke_type",
    cap_unseen: float = 2.0,
    inverse_freq_max: float = 5.0,
    normalize_mean_one: bool = True,
) -> Tuple[List[float], Dict[str, float]]:
    """
    Compute per-class loss weights from annotation counts (inverse frequency).
    Uses the same loading and mapping as FineBadmintonDataset so counts match training.

    Args:
        data_root: Same as for the dataset.
        list_file: Path to the annotations JSON.
        task: Task name (e.g. 'stroke_type', 'position').
        cap_unseen: Weight for classes with 0 count (avoids huge weights).
        inverse_freq_max: Cap on 1/(count+1) before normalization.
        normalize_mean_one: If True, scale weights so mean is 1.

    Returns:
        weights_list: List of length num_classes, same order as dataset.classes[task].
        name_to_weight: Dict mapping class name -> weight.
    """
    if not os.path.exists(list_file):
        return [], {}
    ds = FineBadmintonDataset(data_root, list_file, transform=None)
    if task not in ds.classes:
        return [], {}
    class_list = ds.classes[task]
    counts = Counter()
    for sample in ds.samples:
        labels = ds._map_labels(sample)
        counts[labels[task]] += 1
    weights = []
    for i in range(len(class_list)):
        c = counts.get(i, 0)
        if c == 0:
            w = cap_unseen
        else:
            w = min(1.0 / (c + 1.0), inverse_freq_max)
        weights.append(w)
    if normalize_mean_one and weights:
        mean_w = sum(weights) / len(weights)
        if mean_w > 0:
            weights = [w / mean_w for w in weights]
    name_to_weight = {class_list[i]: weights[i] for i in range(len(class_list))}
    return weights, name_to_weight


# Label order is the inference contract. The annotation JSON is a training clip
# index and is not required to serve the model.
FINEBADMINTON_LABELS = {
    "stroke_type": [
        "Serve", "Clear", "Smash", "Drop", "Drive",
        "Net_Shot", "Lob", "Defensive_Shot", "Other",
    ],
    "stroke_subtype": [
        "None", "Short_Serve", "Flick_Serve", "High_Serve",
        "Common_Smash", "Jump_Smash", "Full_Smash", "Stick_Smash", "Slice_Smash",
        "Slice_Drop", "Stop_Drop", "Reverse_Slice_Drop", "Blocked_Drop",
        "Flat_Lift", "High_Lift",
        "Attacking_Clear", "Spinning_Net", "Flat_Drive", "High_Drive",
        "High_Block", "Continuous_Net_Kills",
    ],
    "technique": ["Forehand", "Backhand", "Unknown"],
    "placement": [
        "Straight", "Cross-court", "Body_Hit", "Over_Head", "Passing_Shot",
        "Wide", "Net_Fault", "Out", "Repeat", "Unknown",
    ],
    "position": [
        "Mid_Front", "Mid_Court", "Mid_Back",
        "Left_Front", "Left_Mid", "Left_Back",
        "Right_Front", "Right_Mid", "Right_Back", "Unknown",
    ],
    "intent": [
        "Intercept", "Passive", "Defensive", "To_Create_Depth",
        "Move_To_Net", "Early_Net_Shot", "Deception", "Hesitation", "Seamlessly", "None",
    ],
    "quality": ["Developing", "Emerging", "Competent", "Proficient", "Advanced", "Expert", "Elite"],
}


class FineBadmintonDataset(Dataset):
    """
    Dataset class for FineBadminton data.

    Multiclass setup: each task is single-label multiclass. Each sample has one
    true class index per task (e.g. stroke_type in 0..8). Labels are class indices
    (long); CrossEntropyLoss(logits, labels) with optional per-class weights
    handles imbalance. Use get_class_weights() for data-driven weights.

    Supports Multi-Task training for:
    - stroke_type (Main hit_type)
    - stroke_subtype (Detailed variation)
    - technique (Backhand/Forehand)
    - placement (Direction/Characteristics)
    - position (Court Area)
    - intent (Strategy/Tactics)
    - quality (Execution Score)
    """
    def __init__(
        self,
        data_root: str,
        list_file: str,
        transform=None,
        sequence_length: int = 16,
        frame_interval: int = 2,
        sampling_mode: str = "span_linspace",
        image_dir: Optional[str] = None,
        prefer_video: bool = False,
    ):
        self.data_root = data_root
        self.transform = transform
        self.sequence_length = sequence_length
        self.frame_interval = frame_interval
        if sampling_mode not in ("span_linspace", "hit_centered"):
            raise ValueError(
                f"sampling_mode must be span_linspace|hit_centered, got {sampling_mode!r}"
            )
        self.sampling_mode = sampling_mode
        self.prefer_video = prefer_video
        if prefer_video:
            self.image_dir: Optional[str] = None
        elif image_dir is not None:
            self.image_dir = image_dir if os.path.isdir(image_dir) else None
        else:
            self.image_dir = default_training_jpeg_dir(data_root)

        # Per-dataset (per DataLoader process) open captures — not a class var, so
        # multiprocess workers do not share broken fd state across processes.
        self._video_caps: dict = {}

        self.classes = {k: list(v) for k, v in FINEBADMINTON_LABELS.items()}
        
        # Build index maps
        self.maps = {k: {cls: i for i, cls in enumerate(v)} for k, v in self.classes.items()}

        if os.path.exists(list_file):
            self.samples = self._load_annotations(list_file)
        else:
            print(f"Warning: Annotation file {list_file} not found.")
            self.samples = []

    def _resolve_video_path(self, video_filename: str) -> str:
        """Find the .mp4 on disk; search common locations."""
        candidates = [
            os.path.join(self.data_root, video_filename),
            os.path.join(self.data_root, "FineBadminton-20K", "videos", video_filename),
            os.path.join(self.data_root, "videos", video_filename),
        ]
        for p in candidates:
            if os.path.isfile(p):
                return p
        return candidates[0]

    def _load_annotations(self, list_file: str) -> List[Dict[str, Any]]:
        import json
        with open(list_file, 'r') as f:
            data = json.load(f)
            
        samples = []
        for video_item in data:
            video_filename = video_item['video']
            if 'hitting' not in video_item: continue
            video_path = self._resolve_video_path(video_filename)
                
            for hit in video_item['hitting']:
                if 'start_frame' not in hit or 'end_frame' not in hit: continue

                hit_frame = hit.get("hit_frame")
                if hit_frame is None:
                    hit_frame = (int(hit["start_frame"]) + int(hit["end_frame"])) // 2

                samples.append({
                    'video_path': video_path,
                    'start_frame': hit['start_frame'],
                    'end_frame': hit['end_frame'],
                    'hit_frame': int(hit_frame),
                    'hit_type': hit.get('hit_type', 'Other'),
                    'subtype': hit.get('subtype', []),
                    'player_actions': hit.get('player_actions', []),
                    'shot_characteristics': hit.get('shot_characteristics', []),
                    'ball_area': hit.get('ball_area', 'Unknown'),
                    'strategies': hit.get('strategies', []),
                    'quality': hit.get('quality', 1)
                })
        return samples

    def _map_labels(self, sample: Dict) -> Dict[str, int]:
        # 1. Map Stroke Type
        type_map = {
            "serve": "Serve", "clear": "Clear", "smash": "Smash", "kill": "Smash", 
            "net kill": "Smash", "drop": "Drop", "drop shot": "Drop", "drive": "Drive",
            "net shot": "Net_Shot", "cross-court net shot": "Net_Shot", "lob": "Lob", 
            "push shot": "Lob", "net lift": "Lob", "block": "Defensive_Shot", "defensive shot": "Defensive_Shot"
        }
        raw_type = sample['hit_type'].lower()
        type_mapped = type_map.get(raw_type, "Other")
        
        # 2. Map Subtype (Take first if multiple, else None)
        st_map = {
            "short serve": "Short_Serve", "flick serve": "Flick_Serve", "high serve": "High_Serve",
            "common smash": "Common_Smash", "jump smash": "Jump_Smash", "full smash": "Full_Smash",
            "stick smash": "Stick_Smash", "slice smash": "Slice_Smash", "slice drop shot": "Slice_Drop",
            "stop drop shot": "Stop_Drop", "reverse slice drop shot": "Reverse_Slice_Drop", "blocked drop shot": "Blocked_Drop",
            "flat lift": "Flat_Lift", "high lift": "High_Lift",
            "attacking clear": "Attacking_Clear", "spinning net": "Spinning_Net", "flat drive": "Flat_Drive", "high drive": "High_Drive",
            "high block": "High_Block", "continuous net kills": "Continuous_Net_Kills",
        }
        subtypes = [s.lower() for s in sample['subtype']]
        st_mapped = st_map.get(subtypes[0], "None") if subtypes else "None"

        # 3. Map Technique (Player Action — scan all positions, not just first)
        pa_map = {"forehand": "Forehand", "backhand": "Backhand"}
        actions = [a.lower() for a in sample['player_actions']]
        pa_mapped = "Unknown"
        for a in actions:
            if a in pa_map:
                pa_mapped = pa_map[a]
                break

        # 4. Map Placement (Shot Characteristic)
        char_map = {
            "straight": "Straight", "cross-court": "Cross-court", "body hit": "Body_Hit",
            "over head": "Over_Head", "passing shot": "Passing_Shot", "wide placement": "Wide",
            "net fault": "Net_Fault", "out": "Out", "repeat shot": "Repeat",
        }
        chars = [c.lower() for c in sample['shot_characteristics']]
        ch_mapped = char_map.get(chars[0], "Unknown") if chars else "Unknown"

        # 5. Map Position (Ball Area)
        pos_map = {
            "mid front court": "Mid_Front", "mid court": "Mid_Court", "mid back court": "Mid_Back",
            "left front court": "Left_Front", "left mid court": "Left_Mid", "left back court": "Left_Back",
            "right front court": "Right_Front", "right mid court": "Right_Mid", "right back court": "Right_Back"
        }
        pos_mapped = pos_map.get(sample['ball_area'].lower(), "Unknown")

        # 6. Map Intent (from player_actions — strategies is always empty in 20K)
        intent_map = {
            "intercept": "Intercept", "passive": "Passive", "defensive": "Defensive",
            "to create depth": "To_Create_Depth", "move to the net": "Move_To_Net",
            "a high net early shot": "Early_Net_Shot", "deception": "Deception",
            "hesitation": "Hesitation", "seamlessly": "Seamlessly"
        }
        strat_mapped = "None"
        for a in actions:
            if a in intent_map:
                strat_mapped = intent_map[a]
                break

        # Convert back to indices
        return {
            "stroke_type": self.maps["stroke_type"][type_mapped],
            "stroke_subtype": self.maps["stroke_subtype"][st_mapped],
            "technique": self.maps["technique"][pa_mapped],
            "placement": self.maps["placement"][ch_mapped],
            "position": self.maps["position"][pos_mapped],
            "intent": self.maps["intent"][strat_mapped],
            "quality": max(0, min(int(sample['quality']) - 1, 6))
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        labels_dict = self._map_labels(sample)

        frames = self._load_frames(
            sample['video_path'],
            sample['start_frame'],
            sample['end_frame'],
            sample.get('hit_frame'),
        )

        if self.transform:
            frames = self.transform(frames)

        if isinstance(frames, list):
            frames = torch.stack(frames)

        tensor_labels = {k: torch.tensor(v, dtype=torch.long) for k, v in labels_dict.items()}

        return frames, tensor_labels

    # ------------------------------------------------------------------
    # Frame loading: decode directly from source .mp4 via cv2
    # ------------------------------------------------------------------

    def _get_cap(self, video_path: str):
        """Return an open cv2.VideoCapture, reusing across calls for the same file."""
        if video_path not in self._video_caps:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            self._video_caps[video_path] = cap
        return self._video_caps[video_path]

    def _frame_indices(
        self, start_frame: int, end_frame: int, hit_frame: Optional[int] = None
    ) -> np.ndarray:
        """Return ``sequence_length`` frame indices for the stroke clip."""
        duration = end_frame - start_frame
        if duration <= 0:
            return np.zeros(self.sequence_length, dtype=int)
        if self.sampling_mode == "span_linspace" or hit_frame is None:
            return np.linspace(start_frame, end_frame - 1, self.sequence_length).astype(int)
        center = self.sequence_length // 2
        hit = int(np.clip(hit_frame, start_frame, end_frame - 1))
        indices = np.linspace(start_frame, end_frame - 1, self.sequence_length).astype(int)
        indices[center] = hit
        return indices

    def _load_frames(
        self, video_path: str, start_frame: int, end_frame: int,
        hit_frame: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """Sample sequence_length frames (JPEG cache or MP4 decode)."""
        if self.image_dir is not None:
            return self._load_frames_from_jpeg(
                video_path, start_frame, end_frame, hit_frame
            )
        return self._load_frames_from_video(
            video_path, start_frame, end_frame, hit_frame
        )

    def _load_frames_from_jpeg(
        self,
        video_path: str,
        start_frame: int,
        end_frame: int,
        hit_frame: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """Frame indices follow ``sampling_mode`` (linspace or hit-centered)."""
        blank = [torch.zeros((3, 224, 224)) for _ in range(self.sequence_length)]
        duration = end_frame - start_frame
        if duration <= 0:
            return blank
        assert self.image_dir is not None
        stem = Path(video_path).stem
        indices = self._frame_indices(start_frame, end_frame, hit_frame)
        frames: List[torch.Tensor] = []
        for idx in indices:
            fp = os.path.join(self.image_dir, f"{stem}_{int(idx)}.jpg")
            bgr = cv2.imread(fp, cv2.IMREAD_COLOR)
            if bgr is None or bgr.size == 0:
                frames.append(torch.zeros((3, 224, 224)))
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (224, 224))
            frames.append(torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0)
        return frames

    def _load_frames_from_video(
        self,
        video_path: str,
        start_frame: int,
        end_frame: int,
        hit_frame: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """Sample sequence_length frames from the video clip."""
        blank = [torch.zeros((3, 224, 224)) for _ in range(self.sequence_length)]
        duration = end_frame - start_frame
        if duration <= 0:
            return blank

        indices = self._frame_indices(start_frame, end_frame, hit_frame)

        cap = self._get_cap(video_path)
        if cap is None:
            return blank

        frames: List[torch.Tensor] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, bgr = cap.read()
            if ok and bgr is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (224, 224))
                frames.append(
                    torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
                )
            else:
                frames.append(torch.zeros((3, 224, 224)))

        return frames

if __name__ == "__main__":
    print("Dataset class refined with Multi-Task support.")
