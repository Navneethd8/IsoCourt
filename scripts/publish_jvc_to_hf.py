#!/usr/bin/env python3
"""Publish the paper JVC (K-STViT) checkpoint to the Hugging Face Hub.

Writes ``hf/jvc/config.json`` and ``README.md`` from the checkpoint, and
uploads that folder plus the weight when ``HF_TOKEN`` is set. The Hub repo
is ``<user>/JVC``. An older ``<user>/isocourt-jvc`` repo is renamed first.

  python scripts/publish_jvc_to_hf.py --write-card
  HF_TOKEN=... python scripts/publish_jvc_to_hf.py
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "backend"))

from core.hf_jvc_release import (  # noqa: E402
    HUB_CHECKPOINT_FILENAME,
    LOCAL_CHECKPOINT_FILENAME,
    PREVIOUS_HF_REPO_NAME,
    PREVIOUS_HUB_CHECKPOINT_FILENAME,
    build_config,
    checkpoint_path,
    default_repo_id,
    is_lfs_pointer,
    meta_from_checkpoint,
    render_model_card,
)

CARD_DIR = os.path.join(REPO_ROOT, "hf", "jvc")


def write_card_files(dest: str, *, include_checkpoint: bool) -> str:
    src = checkpoint_path(REPO_ROOT)
    if is_lfs_pointer(src):
        raise SystemExit(
            f"{src} is still a Git LFS pointer. Run "
            f"`git lfs pull --include=backend/models/{LOCAL_CHECKPOINT_FILENAME}` first."
        )
    meta = meta_from_checkpoint(src)
    config = build_config(meta, default_repo_id())
    os.makedirs(dest, exist_ok=True)
    with open(os.path.join(dest, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
    with open(os.path.join(dest, "README.md"), "w", encoding="utf-8") as f:
        f.write(render_model_card(config))
    if include_checkpoint:
        shutil.copy2(src, os.path.join(dest, HUB_CHECKPOINT_FILENAME))
    return config["checkpoint_sha256"]


def _rename_previous_repo(api, repo_id: str) -> None:
    """Move ``user/isocourt-jvc`` to ``user/JVC`` when the old repo is still there.

    A redirect from the old name still reports the repo as existing after the
    move, so skip the move once the JVC repo itself is present.
    """
    from huggingface_hub.errors import RepositoryNotFoundError

    user = repo_id.split("/", 1)[0]
    old_id = f"{user}/{PREVIOUS_HF_REPO_NAME}"
    if old_id.lower() == repo_id.lower():
        return
    if api.repo_exists(repo_id, repo_type="model"):
        return
    if not api.repo_exists(old_id, repo_type="model"):
        return
    try:
        api.move_repo(from_id=old_id, to_id=repo_id, repo_type="model")
    except RepositoryNotFoundError:
        return
    print(f"Renamed {old_id} -> {repo_id}")


def upload_folder(folder: str, repo_id: str, token: str) -> str:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    _rename_previous_repo(api, repo_id)
    api.create_repo(repo_id, repo_type="model", exist_ok=True, private=False)
    api.upload_folder(
        folder_path=folder,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Update the JVC model card",
        delete_patterns=PREVIOUS_HUB_CHECKPOINT_FILENAME,
    )
    return f"https://huggingface.co/{repo_id}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-card",
        action="store_true",
        help="Regenerate hf/jvc/config.json and README.md from the checkpoint.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Stage the Hub folder under a temp directory and print the path. Do not upload.",
    )
    args = parser.parse_args()

    if args.write_card:
        digest = write_card_files(CARD_DIR, include_checkpoint=False)
        print(f"Wrote {CARD_DIR} (checkpoint sha256 {digest})")
        if not args.dry_run and not os.environ.get("HF_TOKEN"):
            return

    token = os.environ.get("HF_TOKEN", "").strip()
    repo_id = default_repo_id()
    if args.dry_run or not token:
        if not token and not args.dry_run and not args.write_card:
            raise SystemExit("Set HF_TOKEN to upload, or pass --write-card / --dry-run.")
        if args.dry_run:
            stage = tempfile.mkdtemp(prefix="jvc-")
            digest = write_card_files(stage, include_checkpoint=True)
            print(f"Staged {repo_id} at {stage} (sha256 {digest})")
        return

    stage = tempfile.mkdtemp(prefix="jvc-")
    try:
        digest = write_card_files(stage, include_checkpoint=True)
        url = upload_folder(stage, repo_id, token)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    print(f"Published {url} (sha256 {digest})")


if __name__ == "__main__":
    main()
