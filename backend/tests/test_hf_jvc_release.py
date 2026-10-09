"""Card + loader contract for the published JVC checkpoint."""
from __future__ import annotations

import json
import os
import sys

import pytest

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(BACKEND)
sys.path.insert(0, BACKEND)

from core.hf_jvc_release import (  # noqa: E402
    HUB_CHECKPOINT_FILENAME,
    TASK_LABELS,
    build_config,
    checkpoint_path,
    is_lfs_pointer,
    render_model_card,
)
from core.label_maps import STROKE_TYPE_CLASSES  # noqa: E402


def _sample_meta() -> dict:
    return {
        "task_classes": {task: len(names) for task, names in TASK_LABELS.items()},
        "best_acc": 80.61415846225972,
        "epoch": 18,
        "embed_dim": 128,
        "st_depth": 4,
        "num_cross_layers": 2,
        "vision_backbone": "conv3d",
        "video_backbone": "r2plus1d_18",
        "spatial_size": 224,
        "sampling_mode": "span_linspace",
        "use_shuttle": False,
        "checkpoint_sha256": "abc",
        "num_heads": 4,
        "four_stream": True,
    }


def test_stroke_labels_match_shared_map():
    assert TASK_LABELS["stroke_type"] == STROKE_TYPE_CLASSES


def test_config_and_card_describe_the_paper_checkpoint():
    config = build_config(_sample_meta(), "navneethdg/JVC")
    assert config["architecture"] == "jvc"
    assert config["repo_id"] == "navneethdg/JVC"
    assert config["paper"]["url"] == "https://openreview.net/forum?id=XJEhcXfwEe"
    assert config["metrics"]["stroke_type_val_acc"] == 80.61
    assert config["labels"]["stroke_type"][2] == "Smash"
    assert config["task_classes"]["stroke_type"] == 9
    card = render_model_card(config)
    assert 'hf_hub_download("navneethdg/JVC"' in card
    assert "dhamotharan2026jvc" in card
    assert "https://openreview.net/pdf?id=XJEhcXfwEe" in card
    assert HUB_CHECKPOINT_FILENAME in card
    assert "80.61%" in card
    assert "k_st_vit" not in card.lower()
    assert "kstvit" not in card.lower()
    assert "https://isocourt.fit" in card
    assert "BadCoach" not in card
    assert "badcoach" not in card.lower()
    assert "github.com" not in card
    assert "IsoCourt" not in card


def test_committed_card_matches_checkpoint_when_weights_are_present():
    path = checkpoint_path(REPO)
    if is_lfs_pointer(path):
        pytest.skip("JVC checkpoint is a Git LFS pointer in this checkout")
    from core.hf_jvc_release import meta_from_checkpoint

    config = build_config(meta_from_checkpoint(path), "navneethdg/JVC")
    card_dir = os.path.join(REPO, "hf", "jvc")
    with open(os.path.join(card_dir, "config.json"), encoding="utf-8") as f:
        committed = json.load(f)
    assert committed == config
    with open(os.path.join(card_dir, "README.md"), encoding="utf-8") as f:
        readme = f.read()
    assert readme == render_model_card(config)
