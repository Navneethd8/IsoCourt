# Model registry and inference selection

This folder holds stroke-model weights and two pieces of configuration:


| File                       | Role                                                                                                                                                             |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model_registry.json`      | **Training catalog (v2).** Per-architecture `primary` checkpoint plus optional `registrations` (experiments). Updated when training saves a new best checkpoint. |
| `inference_selection.json` | **Prod / API switch only.** Names which architecture **category** the server loads. Not written by training scripts.                                             |
| `*.pth`                    | Checkpoints. Inference always uses the `**primary.file`** for the selected category—not arbitrary paths from `registrations`.                                    |


Implementation: `backend/core/model_registry.py`, `backend/api/model_loader.py`, CLI `python -m api.inference_model_cli` (run from `backend/`).

---

## Architecture categories (canonical keys)

These are the **only** valid keys under `models.architectures.`* and for `inference_selection.json` → `category`. They map one-to-one to training scripts and loader logic.


| Category               | Research group    | Training script                                    | Core module                        | Notes                                                                                                                               |
| ---------------------- | ----------------- | -------------------------------------------------- | ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `cnn_lstm`             | cnn               | `pipelines/training/train_full.py`                 | `core/model.py` (`CNN_LSTM_Model`) | ResNet50 frame features + LSTM; pose optional via cache.                                                                            |
| `conv3d_pose`          | video_cnn         | `pipelines/training/train_conv3d.py`               | `core/conv3d_pose.py`              | Torchvision R(2+1)D / R3D / MC3 + late-fused MediaPipe joints; optional Grad-CAM on `layer4`.                                       |
| `st_tr`                | graph             | `pipelines/training/train_st_tr.py`                | `core/st_tr.py` (`STTRModel`)      | ST-TR dual transformer on MediaPipe pose. **Not supported** by `/analyze` (pose-only).                                              |
| `timesformer`          | video_transformer | `pipelines/training/train_timesformer.py`          | `core/timesformer.py`              | Divided space–time attention + pose tokens; ViT-style stem when `backbone=vit`.                                                     |
| `k_st_vit`             | graph             | `pipelines/training/train_k_st_vit.py`             | `core/k_st_vit.py`                 | SkateFormer joints + Conv3D/ViT patches, divided ST fusion; default `hit_centered` sampling.                                        |
| `vit_gcn`              | graph             | `pipelines/training/train_vit_gcn.py`              | `core/vit_gcn.py`                  | Per-frame timm ViT CLS + fixed skeleton GCN on MediaPipe joints.                                                                    |


**Loader architecture strings** (checkpoint + registry; used by `api/model_loader.py`): `cnn_lstm`, `conv3d_pose`, `st_tr`, `timesformer`, `k_st_vit`, `vit_gcn`.

---

## JSON schema (v2)

```json
{
  "schema_version": 2,
  "models": {
    "architectures": {
      "<category>": {
        "primary": {
          "file": "relative_or_absolute_checkpoint.pth",
          "experiment": false,
          "accuracy": 0.0,
          "epoch": 0,
          "script": "train_....py",
          "architecture": "<same as category when set>",
          "inference": { },
          "dataset": {
            "id": "finebadminton_20k",
            "source": "Moujuruo/Finebadminton-20K",
            "list_file": "data/transformed_combined_rounds_output_en_evals_translated.json",
            "data_root_relative": "data/FineBadminton-20K/videos",
            "frames": "uniform_16_in_hit_span"
          }
        },
        "registrations": [
          { "file": "...", "experiment": true, "...": "..." }
        ]
      }
    }
  }
}
```

- `**primary**`: At most one row per category. Metadata here (especially `inference`) is merged with the checkpoint envelope when building the model for the API.
- `**dataset**`: Which label/video tree this weight was trained against and what the API should use for **annotation paths** at startup (class vocab is still the fixed multitask schema in `FineBadmintonDataset`). Defaults are **FineBadminton-20K** relative to the `backend/` directory. Override per checkpoint by writing a custom `dataset` object on that row.
  - `**id`**: Stable id (default `finebadminton_20k`).
  - `**list_file`**: Merged JSON list (default `data/transformed_combined_rounds_output_en_evals_translated.json` from `prepare_finebadminton_20k.py`).
  - `**data_root_relative`**: Folder containing `video` filenames from the JSON (default `data/FineBadminton-20K/videos`).
  - `**frames**`: `uniform_16_in_hit_span` (model samples 16 indices across each hit’s `[start_frame, end_frame)`; JPEGs must exist for those indices when using disk). Use `jpeg_all_video_frames` after a full extract (see below).
- `**registrations**`: Historical or A/B weights. They do **not** affect inference unless you **promote** one to `primary` (CLI below).

### FineBadminton-20K layout and frame extraction

1. **Prepare labels** (writes merged list + downloads Hub snapshot):
  `python backend/pipelines/vlm/common/prepare_finebadminton_20k.py`
2. **Contact frames only** (one JPEG per hit at `hit_frame`):
  `python .../prepare_finebadminton_20k.py --extract-frames`
3. **Every video frame** (full decode; huge disk use):
  `python .../prepare_finebadminton_20k.py --extract-all-frames`  
   Then set registry `dataset.frames` to `jpeg_all_video_frames` for that checkpoint if you trained assuming dense JPEGs.

The API resolves `list_file` / `data_root_relative` against the **parent of `backend/models/`** (i.e. `backend/`). Optional environment overrides (absolute or `~` paths):


| Variable                        | Effect                                                                             |
| ------------------------------- | ---------------------------------------------------------------------------------- |
| `ISOCOURT_INFERENCE_LIST_FILE`  | Overrides merged JSON path for startup.                                            |
| `ISOCOURT_INFERENCE_DATA_ROOT`  | Overrides video directory for `FineBadmintonDataset` construction.                 |
| `ISOCOURT_INFERENCE_DATASET_ID` | Overrides logged `dataset_id` only (paths unchanged unless the two above are set). |


Legacy **v1** flat `models: { "file.pth": { ... } }` plus `active_model` is still read once and normalized to v2 in memory when loading; saving from training always writes v2.

---

## Training vs inference


| Action                                    | Effect on `model_registry.json`                                                                                | Effect on `inference_selection.json`    |
| ----------------------------------------- | -------------------------------------------------------------------------------------------------------------- | --------------------------------------- |
| Normal training best save                 | Overwrites `**primary`** for that script’s category (unless `--registry-experiment`).                          | None.                                   |
| Training with `**--registry-experiment`** | Appends to `**registrations`** only.                                                                           | None.                                   |
| You want API on another lane              | Optionally update `**primary**` for that category (e.g. after promote), then point inference at that category. | `**set**` CLI or edit JSON (see below). |


---

## Copy-paste: train every stroke model

Run these from the **repository root** (the folder that contains `backend/`). Each script resolves `backend/data/transformed_combined_rounds_output_en_evals_translated.json` and `backend/data/` on its own. **JPEG contact frames** should already exist under `backend/data/FineBadminton-20K/dataset/image/` (see *FineBadminton-20K layout* above); most trainers also build or reuse `backend/models/pose_cache_mediapipe.pt` unless you pass `--no-pose`.

Device is picked automatically (`cuda` → `mps` → `cpu`).

**Optional:** append `--registry-experiment` to any command so the best checkpoint is recorded under `registrations` instead of overwriting that architecture’s `primary`.

```bash
# --- Optional: labels + Hub snapshot + contact JPEGs (once per machine) ---
# python backend/pipelines/vlm/common/prepare_finebadminton_20k.py --skip-download --local-dir backend/data/FineBadminton-20K --extract-frames

# --- cnn_lstm (CNN+LSTM, default registry: badminton_model.pth) ---
python backend/pipelines/training/train_full.py
# RGB only (no MediaPipe / no pose cache):
# python backend/pipelines/training/train_full.py --no-pose

# --- conv3d_pose (R(2+1)D / R3D / MC3 + pose, default: badminton_model_conv3d_pose.pth) ---
python backend/pipelines/training/train_conv3d.py
# python backend/pipelines/training/train_conv3d.py --epochs 60 --batch-size 4 --video-backbone r2plus1d_18 --spatial-size 224 --aug strong --no-pose

# --- st_tr (ST-TR pose dual-stream; default: badminton_model_st_tr.pth; not for /analyze API) ---
python backend/pipelines/training/train_st_tr.py

# --- timesformer (divided ST + pose; default: badminton_model_timesformer.pth) ---
python backend/pipelines/training/train_timesformer.py
# ViT per-frame backbone instead of scratch conv stem:
# python backend/pipelines/training/train_timesformer.py --backbone vit --vit-model vit_small_patch16_224
# python backend/pipelines/training/train_timesformer.py --no-pose

# --- vit_gcn (timm ViT CLS + skeleton GCN; default: badminton_model_vit_gcn.pth) ---
python backend/pipelines/training/train_vit_gcn.py
# python backend/pipelines/training/train_vit_gcn.py --vit-model vit_small_patch16_224 --no-pose
```

Heavy scripts (`train_conv3d.py`, `train_timesformer.py`, `train_vit_gcn.py`) accept shared-style flags such as `--epochs`, `--batch-size`, `--lr`, `--pose-cache /path/to.pt`, `--max-train-batches N` (smoke tests), `--aug {strong,medium,mild}`, `--accum-steps`, `--stroke-loss-weight`, `--registry-experiment`, and `--no-pose` where listed. Use each file’s `--help` for the full list.

---

## Switching the API model (inference only)

From `**backend/**`:

```bash
python3 -m api.inference_model_cli list
python3 -m api.inference_model_cli show
python3 -m api.inference_model_cli set timesformer
python3 -m api.inference_model_cli promote vit_gcn 0
```

`**set**` also rewrites `**backend/deploy/docker-inference.env**` (used by `**docker compose**` via `docker-compose.yml`) and `**backend/deploy/ci_inference_category**` (one line; the HF deploy workflow loads it into `ISOCOURT_INFERENCE_CATEGORY` before copying the single checkpoint). Commit those files with `models/inference_selection.json` so you are not hand-editing Docker or GitHub YAML.

**Environment (overrides the JSON file):** `ISOCOURT_INFERENCE_CATEGORY=timesformer`

**Resolution order for category:** env → `inference_selection.json` → legacy v1 `active_model` hint (migration) → first category in the built-in fallback list that has a valid `primary` file on disk.

**Resolved checkpoint:** always `models_dir / primary.file` for that category—never an experiment path unless it has been promoted to `primary`.

---

## Related assets

- `**pose_cache_mediapipe.pt`**: Shared MediaPipe-derived pose tensor `(N, T, 33, 3)` for trainers that fuse pose.
- `**pose_landmarker_lite.task`**: MediaPipe task file used by `PoseEstimator`.

VLMs (Qwen, JSONL, etc.) live under `backend/pipelines/vlm/` and are **not** entries in this stroke registry.

---

## Deploy (Hugging Face Space)

GitHub Actions **does not rsync every** `models/*.pth` into the Space (that would waste runner time and LFS). It **excludes all** `.pth` during `rsync`, then **copies only** the checkpoint resolved by `inference_selection.json` / `ISOCOURT_INFERENCE_CATEGORY` (same rules as local inference). A follow-up step still **slims** `model_registry.json` to that category’s `primary` only.

Set `backend/models/inference_selection.json` on `main` to the architecture you want shipped before each deploy (or set `ISOCOURT_INFERENCE_CATEGORY` on the workflow if you add that env to the job).

## Paper checkpoint (Hugging Face model)

The main JVC weight is the `k_st_vit` primary (`badminton_model_k_st_vit.pth`, 80.61% val `stroke_type`). It is published as a model repo, separate from the BadCoach Space, so people can download and run it:

`https://huggingface.co/navneethdg/JVC`

Card and config live in `hf/jvc/`. The card links the paper and includes the OpenReview BibTeX. Regenerate them from the checkpoint with `python scripts/publish_jvc_to_hf.py --write-card`. Upload with `HF_TOKEN` set, or run the **Publish JVC model to Hugging Face** workflow (uses the `HF_TOKEN` / `HF_USERNAME` secrets). Load path: `core.hf_jvc_release.load_published_jvc`.