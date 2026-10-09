"""
Model card and config for the paper JVC checkpoint on the Hugging Face Hub.

JVC is the joint-to-vision cross-attention model. The published weight is the
K-STViT run (registry category ``k_st_vit``), not the no-cross-attention ablations.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Mapping

# Hub repo published by scripts/publish_jvc_to_hf.py and the GitHub Action.
DEFAULT_HF_USER = "navneethdg"
DEFAULT_HF_REPO_NAME = "JVC"
# Previous id, moved on the next publish.
PREVIOUS_HF_REPO_NAME = "isocourt-jvc"
DEMO_URL = "https://isocourt.fit"
DATASET_ID = "Moujuruo/Finebadminton-20K"
# Training artifact in this repo. The Hub file is published under a separate name.
LOCAL_CHECKPOINT_FILENAME = "badminton_model_k_st_vit.pth"
HUB_CHECKPOINT_FILENAME = "jvc.pth"
# Older Hub filename, removed on the next publish.
PREVIOUS_HUB_CHECKPOINT_FILENAME = "badminton_model_k_st_vit.pth"
PAPER_TITLE = "JVC: Joint Vision Cross-Attention for Fine-Grained Badminton Stroke Recognition"
PAPER_URL = "https://openreview.net/forum?id=XJEhcXfwEe"
PAPER_PDF_URL = "https://openreview.net/pdf?id=XJEhcXfwEe"
PAPER_AUTHORS = ("Navneeth Dhamotharan", "Bin Han")
PAPER_YEAR = 2026
PAPER_VENUE = (
    "ECCV 2026 Workshop: Human Motion-Informed World Models and Socially Intelligent Action"
)
PAPER_BIBTEX = """@inproceedings{
dhamotharan2026jvc,
title={{JVC}: Joint Vision Cross-Attention for Fine-Grained Badminton Stroke Recognition},
author={Navneeth Dhamotharan and Bin Han},
booktitle={ECCV26 Workshop: Human Motion-Informed World Models and Socially Intelligent Action},
year={2026},
url={https://openreview.net/forum?id=XJEhcXfwEe}
}"""

# Label order matches FineBadmintonDataset (backend/core/dataset.py).
# stroke_subtype is not a head on this checkpoint.
TASK_LABELS: Dict[str, tuple[str, ...]] = {
    "stroke_type": (
        "Serve",
        "Clear",
        "Smash",
        "Drop",
        "Drive",
        "Net_Shot",
        "Lob",
        "Defensive_Shot",
        "Other",
    ),
    "technique": ("Forehand", "Backhand", "Unknown"),
    "placement": (
        "Straight",
        "Cross-court",
        "Body_Hit",
        "Over_Head",
        "Passing_Shot",
        "Wide",
        "Net_Fault",
        "Out",
        "Repeat",
        "Unknown",
    ),
    "position": (
        "Mid_Front",
        "Mid_Court",
        "Mid_Back",
        "Left_Front",
        "Left_Mid",
        "Left_Back",
        "Right_Front",
        "Right_Mid",
        "Right_Back",
        "Unknown",
    ),
    "intent": (
        "Intercept",
        "Passive",
        "Defensive",
        "To_Create_Depth",
        "Move_To_Net",
        "Early_Net_Shot",
        "Deception",
        "Hesitation",
        "Seamlessly",
        "None",
    ),
    "quality": (
        "Developing",
        "Emerging",
        "Competent",
        "Proficient",
        "Advanced",
        "Expert",
        "Elite",
    ),
}


def default_repo_id() -> str:
    user = (os.environ.get("HF_USERNAME") or DEFAULT_HF_USER).strip() or DEFAULT_HF_USER
    return f"{user}/{DEFAULT_HF_REPO_NAME}"


def checkpoint_path(repo_root: str) -> str:
    return os.path.join(os.path.abspath(repo_root), "backend", "models", LOCAL_CHECKPOINT_FILENAME)


def is_lfs_pointer(path: str) -> bool:
    """True when Git LFS has not smudged the checkpoint into a real zip archive."""
    if not os.path.isfile(path):
        return True
    if os.path.getsize(path) > 1024:
        return False
    with open(path, "rb") as f:
        head = f.read(64)
    return head.startswith(b"version https://git-lfs.github.com/spec/v1")


def meta_from_checkpoint(path: str) -> Dict[str, Any]:
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "k_st_vit" not in ckpt:
        raise ValueError(f"{path} is not a JVC K-STViT checkpoint (missing 'k_st_vit')")
    meta = {k: v for k, v in ckpt.items() if k != "k_st_vit"}
    meta["four_stream"] = True
    meta["num_frames"] = 16
    meta["num_joints"] = 33
    meta["num_heads"] = int(meta.get("num_heads", 4))
    return meta


def build_config(meta: Mapping[str, Any], repo_id: str) -> Dict[str, Any]:
    task_classes = {k: int(v) for k, v in dict(meta["task_classes"]).items()}
    labels = {}
    for task, n_cls in task_classes.items():
        names = TASK_LABELS.get(task)
        if names is None or len(names) != n_cls:
            raise ValueError(f"No label list of length {n_cls} for task {task!r}")
        labels[task] = list(names)
    acc = round(float(meta["best_acc"]), 2)
    return {
        "architecture": "jvc",
        "paper_name": "JVC",
        "repo_id": repo_id,
        "checkpoint": HUB_CHECKPOINT_FILENAME,
        "embed_dim": int(meta["embed_dim"]),
        "st_depth": int(meta["st_depth"]),
        "num_cross_layers": int(meta["num_cross_layers"]),
        "num_heads": int(meta.get("num_heads", 4)),
        "skel_embed_dim": 64,
        "skel_num_heads": 16,
        "four_stream": bool(meta.get("four_stream", True)),
        "vision_backbone": str(meta["vision_backbone"]),
        "video_backbone": str(meta["video_backbone"]),
        "spatial_size": int(meta["spatial_size"]),
        "num_frames": int(meta.get("num_frames", 16)),
        "num_joints": int(meta.get("num_joints", 33)),
        "sampling_mode": str(meta["sampling_mode"]),
        "use_shuttle": bool(meta.get("use_shuttle", False)),
        "contact_pool": True,
        "task_classes": task_classes,
        "labels": labels,
        "id2label": {str(i): name for i, name in enumerate(labels["stroke_type"])},
        "label2id": {name: i for i, name in enumerate(labels["stroke_type"])},
        "metrics": {
            "stroke_type_val_acc": acc,
            "epoch": int(meta["epoch"]),
            "split": "video_level_80_20_seed_42",
        },
        "dataset": DATASET_ID,
        "demo": DEMO_URL,
        "paper": {
            "title": PAPER_TITLE,
            "url": PAPER_URL,
            "pdf": PAPER_PDF_URL,
            "authors": list(PAPER_AUTHORS),
            "year": PAPER_YEAR,
            "venue": PAPER_VENUE,
        },
    }


def render_model_card(config: Mapping[str, Any]) -> str:
    acc = config["metrics"]["stroke_type_val_acc"]
    epoch = config["metrics"]["epoch"]
    repo_id = config["repo_id"]
    labels = ", ".join(f"`{name}`" for name in config["labels"]["stroke_type"])
    return f"""---
tags:
  - pytorch
  - badminton
  - action-recognition
  - video-classification
  - skeleton
datasets:
  - {DATASET_ID}
library_name: pytorch
model-index:
  - name: JVC
    results:
      - task:
          type: video-classification
          name: Badminton stroke recognition
        dataset:
          name: FineBadminton-20K (video-level 80/20 split, seed 42)
          type: {DATASET_ID}
        metrics:
          - name: Val stroke_type accuracy
            type: accuracy
            value: {acc}
---

# JVC

Main checkpoint for [{PAPER_TITLE}]({PAPER_URL}) ([PDF]({PAPER_PDF_URL})). Navneeth Dhamotharan and Bin Han, {PAPER_VENUE}, {PAPER_YEAR}.

The released weight is the joint-vision cross-attention model.

A 16-frame hit clip goes through two encoders: R(2+1)D Conv3D for RGB patches, and a four-stream SkateFormer for MediaPipe joints. Joint tokens cross-attend to visual patches, a divided space–time transformer mixes the clip, and contact-weighted pooling feeds multitask heads. The paper metric is **9-way `stroke_type`**.

| | |
| --- | --- |
| Val `stroke_type` accuracy | **{acc}%** (epoch {epoch}) |
| Split | Video-level 80/20, seed 42. No clip from a validation video is in training. |
| Dataset | [FineBadminton-20K](https://huggingface.co/datasets/{DATASET_ID}) |
| Frames | 16, `span_linspace` over the hit span, 224×224, ImageNet normalization |
| Skeleton | MediaPipe, 33 joints × (x, y, z), four-stream (joint, bone, joint-motion, bone-motion) |
| Paper | [{PAPER_TITLE}]({PAPER_URL}) |
| Demo | [isocourt.fit]({DEMO_URL}) |

Stroke classes, in logit order: {labels}.

The checkpoint also emits logits for `technique`, `placement`, `position`, `intent`, and `quality`. `stroke_type` is the number reported above.

## Try the demo

Upload a clip at [isocourt.fit]({DEMO_URL}).

## Weights

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download("{repo_id}", "{HUB_CHECKPOINT_FILENAME}")
```

`frames` is `(batch, 16, 3, 224, 224)` float RGB after ImageNet normalization
(`mean = (0.485, 0.456, 0.406)`, `std = (0.229, 0.224, 0.225)`).
`pose` is `(batch, 16, 33, 3)` in the same layout as the training MediaPipe cache.
Heads return logits, not probabilities.

## Files

| File | Role |
| --- | --- |
| `{HUB_CHECKPOINT_FILENAME}` | Checkpoint. Constructor metadata sits beside the weights. |
| `config.json` | Architecture, label names, and metric. |

## What this weight is

Vision backbone `{config["video_backbone"]}` (`{config["vision_backbone"]}`), embed dim {config["embed_dim"]}, {config["num_cross_layers"]} cross-attention layers, {config["st_depth"]} divided space–time blocks, four-stream skeleton. Shuttle features are off.

No-cross-attention JVC ablations are separate runs and are not this file.

## Citation

```bibtex
{PAPER_BIBTEX}
```
"""


def load_published_jvc(checkpoint: str | Dict[str, Any], *, device: str = "cpu"):
    """Build JVC (K-STViT) and load this checkpoint. Conv3D starts uninitialized.

    The checkpoint already contains the trained R(2+1)D trunk, so Kinetics
    weights are not downloaded.
    """
    import torch
    from core.k_st_vit import build_k_st_vit, load_k_st_vit_partial

    if isinstance(checkpoint, dict):
        ckpt = checkpoint
    else:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    if "k_st_vit" not in ckpt:
        raise KeyError("Expected a JVC checkpoint with a 'k_st_vit' state dict")

    model = build_k_st_vit(
        {k: int(v) for k, v in ckpt["task_classes"].items()},
        window_size=int(ckpt.get("num_frames", 16)),
        embed_dim=int(ckpt["embed_dim"]),
        st_depth=int(ckpt["st_depth"]),
        num_cross_layers=int(ckpt["num_cross_layers"]),
        num_heads=int(ckpt.get("num_heads", 4)),
        vision_backbone=str(ckpt["vision_backbone"]),
        video_backbone=str(ckpt["video_backbone"]),
        spatial_size=int(ckpt["spatial_size"]),
        vit_model_name=str(ckpt.get("vit_model_name", "vit_small_patch16_224")),
        vit_unfreeze_last_n=int(ckpt.get("vit_unfreeze_last_n", 4)),
        conv_pretrained=False,
        use_shuttle=bool(ckpt.get("use_shuttle", False)),
        four_stream=bool(ckpt.get("four_stream", True)),
    )
    load_k_st_vit_partial(model, ckpt, device=device)
    model.to(device)
    model.eval()
    return model
