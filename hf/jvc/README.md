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

The released weight is the joint-vision cross-attention model.

A 16-frame hit clip goes through two encoders: R(2+1)D Conv3D for RGB patches, and a four-stream SkateFormer for MediaPipe joints. Joint tokens cross-attend to visual patches, a divided space–time transformer mixes the clip, and contact-weighted pooling feeds multitask heads. The paper metric is **9-way `stroke_type`**.

| | |
| --- | --- |
| Val `stroke_type` accuracy | **80.61%** (epoch 18) |
| Split | Video-level 80/20, seed 42. No clip from a validation video is in training. |
| Dataset | [FineBadminton-20K](https://huggingface.co/datasets/Moujuruo/Finebadminton-20K) |
| Frames | 16, `span_linspace` over the hit span, 224×224, ImageNet normalization |
| Skeleton | MediaPipe, 33 joints × (x, y, z), four-stream (joint, bone, joint-motion, bone-motion) |
| Paper | [JVC: Joint Vision Cross-Attention for Fine-Grained Badminton Stroke Recognition](https://openreview.net/forum?id=XJEhcXfwEe) |
| Demo | [isocourt.fit](https://isocourt.fit) |

Stroke classes, in logit order: `Serve`, `Clear`, `Smash`, `Drop`, `Drive`, `Net_Shot`, `Lob`, `Defensive_Shot`, `Other`.

The checkpoint also emits logits for `technique`, `placement`, `position`, `intent`, and `quality`. `stroke_type` is the number reported above.

## Try the demo

Upload a clip at [isocourt.fit](https://isocourt.fit).

## Weights

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download("navneethdg/JVC", "jvc.pth")
```

`frames` is `(batch, 16, 3, 224, 224)` float RGB after ImageNet normalization
(`mean = (0.485, 0.456, 0.406)`, `std = (0.229, 0.224, 0.225)`).
`pose` is `(batch, 16, 33, 3)` in the same layout as the training MediaPipe cache.
Heads return logits, not probabilities.

## Files

| File | Role |
| --- | --- |
| `jvc.pth` | Checkpoint. Constructor metadata sits beside the weights. |
| `config.json` | Architecture, label names, metric, and SHA-256 of the checkpoint. |

SHA-256 of `jvc.pth`: `5f9d4d062baf3a77dcce835f919ce072d59aad294d624e728d37fe84625cf2d8`

## What this weight is

Vision backbone `r2plus1d_18` (`conv3d`), embed dim 128, 2 cross-attention layers, 4 divided space–time blocks, four-stream skeleton. Shuttle features are off.

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
