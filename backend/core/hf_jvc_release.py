"""
Model card and config for the paper JVC checkpoint on the Hugging Face Hub.

JVC is the joint-to-vision cross-attention model. The published weight is the
K-STViT run (registry category ``k_st_vit``), not the no-cross-attention ablations.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Mapping

# Hub repo published by scripts/publish_jvc_to_hf.py and the GitHub Action.
DEFAULT_HF_USER = "navneethdg"
DEFAULT_HF_REPO_NAME = "isocourt-jvc"
DEMO_SPACE_URL = "https://huggingface.co/spaces/navneethdg/BadCoach"
CODE_URL = "https://github.com/Navneethd8/IsoCourt"
DATASET_ID = "Moujuruo/Finebadminton-20K"
CHECKPOINT_FILENAME = "badminton_model_k_st_vit.pth"

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
    return os.path.join(os.path.abspath(repo_root), "backend", "models", CHECKPOINT_FILENAME)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    meta["checkpoint_sha256"] = sha256_file(path)
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
        "code_architecture": "k_st_vit",
        "paper_name": "JVC",
        "implementation": "K-STViT",
        "repo_id": repo_id,
        "checkpoint": CHECKPOINT_FILENAME,
        "checkpoint_key": "k_st_vit",
        "checkpoint_sha256": meta["checkpoint_sha256"],
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
        "demo_space": DEMO_SPACE_URL,
        "code": CODE_URL,
    }


def render_model_card(config: Mapping[str, Any]) -> str:
    acc = config["metrics"]["stroke_type_val_acc"]
    epoch = config["metrics"]["epoch"]
    repo_id = config["repo_id"]
    labels = ", ".join(f"`{name}`" for name in config["labels"]["stroke_type"])
    stroke_literal = json.dumps(list(config["labels"]["stroke_type"]))
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

# JVC (joint-to-vision cross-attention)

This is the main **JVC** checkpoint from the IsoCourt badminton paper, implemented as **K-STViT**.

A 16-frame hit clip goes through two encoders: R(2+1)D Conv3D for RGB patches, and a four-stream SkateFormer for MediaPipe joints. Joint tokens cross-attend to visual patches, a divided space–time transformer mixes the clip, and contact-weighted pooling feeds multitask heads. The paper metric is **9-way `stroke_type`**.

| | |
| --- | --- |
| Val `stroke_type` accuracy | **{acc}%** (epoch {epoch}) |
| Split | Video-level 80/20, seed 42. No clip from a validation video is in training. |
| Dataset | [FineBadminton-20K](https://huggingface.co/datasets/{DATASET_ID}) |
| Frames | 16, `span_linspace` over the hit span, 224×224, ImageNet normalization |
| Skeleton | MediaPipe BlazePose, 33 joints × (x, y, z), four-stream (joint, bone, joint-motion, bone-motion) |
| Demo | [{DEMO_SPACE_URL}]({DEMO_SPACE_URL}) |
| Code | [{CODE_URL}]({CODE_URL}) |

Stroke classes, in logit order: {labels}.

The checkpoint also emits logits for `technique`, `placement`, `position`, `intent`, and `quality`. `stroke_type` is the number reported above.

## Try the demo

The [BadCoach Space]({DEMO_SPACE_URL}) serves this same weight for video upload and live analysis.

## Load the checkpoint

```bash
git clone {CODE_URL}
cd IsoCourt
pip install -r backend/requirements.txt
pip install torch torchvision huggingface_hub
```

```python
import sys
sys.path.insert(0, "backend")

import torch
from huggingface_hub import hf_hub_download
from core.hf_jvc_release import load_published_jvc

STROKE_TYPE = {stroke_literal}

path = hf_hub_download("{repo_id}", "{CHECKPOINT_FILENAME}")
model = load_published_jvc(path)  # eval mode; Conv3D weights come from the checkpoint

frames = torch.zeros(1, 16, 3, 224, 224)  # RGB, ImageNet-normalized
pose = torch.zeros(1, 16, 33, 3)          # MediaPipe x, y, z
with torch.no_grad():
    logits = model(frames, pose)

stroke_id = int(logits["stroke_type"].argmax(dim=-1))
print(STROKE_TYPE[stroke_id])
```

`frames` is `(batch, 16, 3, 224, 224)` float RGB after ImageNet normalization
(`mean = (0.485, 0.456, 0.406)`, `std = (0.229, 0.224, 0.225)`).
`pose` is `(batch, 16, 33, 3)` in the same layout as the training MediaPipe cache.
Heads return logits, not probabilities.

## Files

| File | Role |
| --- | --- |
| `{CHECKPOINT_FILENAME}` | Training checkpoint. State dict is under the `k_st_vit` key, with constructor metadata beside it. |
| `config.json` | Architecture, label names, metric, and SHA-256 of the checkpoint. |

SHA-256 of `{CHECKPOINT_FILENAME}`: `{config["checkpoint_sha256"]}`

## What this weight is

Registry category `k_st_vit`, file `{CHECKPOINT_FILENAME}`. Vision backbone `{config["video_backbone"]}` (`{config["vision_backbone"]}`), embed dim {config["embed_dim"]}, {config["num_cross_layers"]} cross-attention layers, {config["st_depth"]} divided space–time blocks, four-stream skeleton. Shuttle features are off.

No-cross-attention JVC ablations are separate runs and are not this file.
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
