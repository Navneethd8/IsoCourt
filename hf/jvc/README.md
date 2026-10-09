---
tags:
  - pytorch
  - badminton
  - action-recognition
  - video-classification
  - skeleton
datasets:
  - Moujuruo/Finebadminton-20K
library_name: pytorch
model-index:
  - name: JVC
    results:
      - task:
          type: video-classification
          name: Badminton stroke recognition
        dataset:
          name: FineBadminton-20K (video-level 80/20 split, seed 42)
          type: Moujuruo/Finebadminton-20K
        metrics:
          - name: Val stroke_type accuracy
            type: accuracy
            value: 80.61
---

# JVC

Main checkpoint for [JVC: Joint Vision Cross-Attention for Fine-Grained Badminton Stroke Recognition](https://openreview.net/forum?id=XJEhcXfwEe) ([PDF](https://openreview.net/pdf?id=XJEhcXfwEe)). Navneeth Dhamotharan and Bin Han, ECCV 2026 Workshop: Human Motion-Informed World Models and Socially Intelligent Action, 2026.

The released weight is the joint-vision cross-attention model (code name K-STViT).

A 16-frame hit clip goes through two encoders: R(2+1)D Conv3D for RGB patches, and a four-stream SkateFormer for MediaPipe joints. Joint tokens cross-attend to visual patches, a divided space–time transformer mixes the clip, and contact-weighted pooling feeds multitask heads. The paper metric is **9-way `stroke_type`**.

| | |
| --- | --- |
| Val `stroke_type` accuracy | **80.61%** (epoch 18) |
| Split | Video-level 80/20, seed 42. No clip from a validation video is in training. |
| Dataset | [FineBadminton-20K](https://huggingface.co/datasets/Moujuruo/Finebadminton-20K) |
| Frames | 16, `span_linspace` over the hit span, 224×224, ImageNet normalization |
| Skeleton | MediaPipe BlazePose, 33 joints × (x, y, z), four-stream (joint, bone, joint-motion, bone-motion) |
| Paper | [JVC: Joint Vision Cross-Attention for Fine-Grained Badminton Stroke Recognition](https://openreview.net/forum?id=XJEhcXfwEe) |
| Demo | [https://huggingface.co/spaces/navneethdg/BadCoach](https://huggingface.co/spaces/navneethdg/BadCoach) |
| Code | [https://github.com/Navneethd8/IsoCourt](https://github.com/Navneethd8/IsoCourt) |

Stroke classes, in logit order: `Serve`, `Clear`, `Smash`, `Drop`, `Drive`, `Net_Shot`, `Lob`, `Defensive_Shot`, `Other`.

The checkpoint also emits logits for `technique`, `placement`, `position`, `intent`, and `quality`. `stroke_type` is the number reported above.

## Try the demo

The [BadCoach Space](https://huggingface.co/spaces/navneethdg/BadCoach) serves this same weight for video upload and live analysis.

## Load the checkpoint

```bash
git clone https://github.com/Navneethd8/IsoCourt
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

STROKE_TYPE = ["Serve", "Clear", "Smash", "Drop", "Drive", "Net_Shot", "Lob", "Defensive_Shot", "Other"]

path = hf_hub_download("navneethdg/JVC", "badminton_model_k_st_vit.pth")
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
| `badminton_model_k_st_vit.pth` | Training checkpoint. State dict is under the `k_st_vit` key, with constructor metadata beside it. |
| `config.json` | Architecture, label names, metric, and SHA-256 of the checkpoint. |

SHA-256 of `badminton_model_k_st_vit.pth`: `5f9d4d062baf3a77dcce835f919ce072d59aad294d624e728d37fe84625cf2d8`

## What this weight is

Registry category `k_st_vit`, file `badminton_model_k_st_vit.pth`. Vision backbone `r2plus1d_18` (`conv3d`), embed dim 128, 2 cross-attention layers, 4 divided space–time blocks, four-stream skeleton. Shuttle features are off.

No-cross-attention JVC ablations are separate runs and are not this file.

## Citation

```bibtex
@inproceedings{
dhamotharan2026jvc,
title={{JVC}: Joint Vision Cross-Attention for Fine-Grained Badminton Stroke Recognition},
author={Navneeth Dhamotharan and Bin Han},
booktitle={ECCV26 Workshop: Human Motion-Informed World Models and Socially Intelligent Action},
year={2026},
url={https://openreview.net/forum?id=XJEhcXfwEe}
}
```
