"""
CLI for prod inference model selection (not used by training).

Examples:
  python -m api.inference_model_cli show
  python -m api.inference_model_cli set timesformer
  python -m api.inference_model_cli list
  python -m api.inference_model_cli promote vit_gcn badminton_model_vit_gcn.pth

``set`` writes ``models/inference_selection.json`` plus ``deploy/docker-inference.env``
and ``deploy/ci_inference_category`` (Docker Compose + GitHub Actions deploy).
Override at runtime without those files: ``ISOCOURT_INFERENCE_CATEGORY=timesformer``.
"""
from __future__ import annotations

import argparse
import os
import sys

_backend = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _backend not in sys.path:
    sys.path.insert(0, _backend)

from core.deploy_artifacts import sync_inference_category_for_deploy  # noqa: E402
from core.model_registry import (  # noqa: E402
    ARCHITECTURE_CATEGORIES,
    CATEGORY_GROUPS,
    ENV_INFERENCE_CATEGORY,
    inference_selection_path,
    load_inference_selection,
    load_registry_file,
    normalize_registry,
    promote_experiment_by_filename,
    promote_experiment_to_primary,
    primary_entry_for_category,
    resolve_inference_category,
    resolve_inference_model_path,
    save_inference_selection,
)


def _models_dir() -> str:
    return os.path.normpath(os.path.join(_backend, "models"))


def cmd_show(models_dir: str) -> int:
    reg_path = os.path.join(models_dir, "model_registry.json")
    reg = load_registry_file(reg_path)
    reg = normalize_registry(reg)
    try:
        cat = resolve_inference_category(models_dir, reg)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    path = resolve_inference_model_path(models_dir, reg)
    sel = load_inference_selection(models_dir)
    env = os.environ.get(ENV_INFERENCE_CATEGORY, "")
    print(f"Registry: {reg_path}")
    print(f"{ENV_INFERENCE_CATEGORY} (env): {env or '(unset)'}")
    print(f"inference_selection.json: {sel}")
    print(f"Resolved category: {cat}")
    print(f"Resolved checkpoint (primary only): {path}")
    return 0 if path and os.path.isfile(path) else 1


def cmd_list(models_dir: str) -> int:
    reg = normalize_registry(load_registry_file(os.path.join(models_dir, "model_registry.json")))
    arch = reg["models"]["architectures"]
    for cat in ARCHITECTURE_CATEGORIES:
        slot = arch.get(cat) or {}
        group = CATEGORY_GROUPS.get(cat, "?")
        prim = slot.get("primary")
        regs = slot.get("registrations") or []
        pfile = prim.get("file") if isinstance(prim, dict) else None
        print(f"[{group:18}] {cat:22} primary={pfile!r}  experiments={len(regs)}")
        if isinstance(regs, list):
            for i, r in enumerate(regs):
                if isinstance(r, dict):
                    print(f"                      [{i}] {r.get('file')!r} experiment={r.get('experiment')}")
    return 0


def cmd_set(models_dir: str, category: str) -> int:
    save_inference_selection(models_dir, category)
    print(f"Wrote {inference_selection_path(models_dir)} with category={category!r}")
    written = sync_inference_category_for_deploy(_backend, category)
    print("Updated deploy/Docker sync files:")
    for p in written:
        print(f"  {p}")
    print("Commit these paths together with models/inference_selection.json for HF + local docker compose.")
    return cmd_show(models_dir)


def cmd_promote(models_dir: str, category: str, target: str) -> int:
    if target.isdigit():
        promote_experiment_to_primary(models_dir, category, int(target))
    else:
        promote_experiment_by_filename(models_dir, category, target)
    print(f"Promoted in {category!r}. Run `show` to verify.")
    return 0


def main() -> int:
    models_dir = _models_dir()
    p = argparse.ArgumentParser(description="IsoCourt inference model (primary checkpoint per category).")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("show", help="Print env, inference_selection.json, resolved category and checkpoint path.")
    sub.add_parser("list", help="List architecture categories, primary file, and experiment registrations.")

    sp_set = sub.add_parser(
        "set",
        help=(
            f"Write inference_selection.json, deploy/docker-inference.env, and deploy/ci_inference_category "
            f"(categories: {', '.join(ARCHITECTURE_CATEGORIES)})."
        ),
    )
    sp_set.add_argument("category", choices=ARCHITECTURE_CATEGORIES)

    sp_pr = sub.add_parser("promote", help="Move an experiment registration to primary for a category.")
    sp_pr.add_argument("category", choices=ARCHITECTURE_CATEGORIES)
    sp_pr.add_argument("target", help="Experiment list index (integer) or checkpoint basename.")

    args = p.parse_args()
    if args.cmd == "show":
        return cmd_show(models_dir)
    if args.cmd == "list":
        return cmd_list(models_dir)
    if args.cmd == "set":
        return cmd_set(models_dir, args.category)
    if args.cmd == "promote":
        try:
            return cmd_promote(models_dir, args.category, args.target)
        except (ValueError, IndexError) as e:
            print(e, file=sys.stderr)
            return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
