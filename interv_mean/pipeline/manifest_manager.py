#!/usr/bin/env python3
"""Preset manager for sec5.1 manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from interv_mean.pipeline import utils


def _expand_numeric_expr(expr: str) -> List[int]:
    values: List[int] = []
    for token in expr.replace(",", " ").split():
        if "-" in token:
            left, right = token.split("-", 1)
            start = int(left)
            end = int(right)
            if end < start:
                raise ValueError(f"Invalid range '{token}'")
            values.extend(list(range(start, end + 1)))
        else:
            values.append(int(token))
    # stable dedupe
    seen = set()
    out: List[int] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _load_presets(path: Path) -> Dict[str, Dict[str, object]]:
    raw = path.read_text()
    data = None
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(raw)
    except Exception:
        data = json.loads(raw)

    if not isinstance(data, dict) or not isinstance(data.get("presets"), dict):
        raise ValueError(f"Invalid presets file: {path}")
    presets = data["presets"]
    return {str(name): cfg for name, cfg in presets.items() if isinstance(cfg, dict)}


def _manifest_path_for_preset(generated_dir: Path, preset_cfg: Dict[str, object], preset_name: str) -> Path:
    manifest_name = str(preset_cfg.get("manifest_name", f"{preset_name}.json"))
    return generated_dir / manifest_name


def _ensure_single_preset(
    *,
    preset_name: str,
    preset_cfg: Dict[str, object],
    generated_dir: Path,
    train_dir: Path,
    test_dir: Path,
    stage2_dir: Path,
    bridge_dir: Path,
    allow_missing_stage2: bool,
    force: bool,
) -> Path:
    output_path = _manifest_path_for_preset(generated_dir, preset_cfg, preset_name)
    if output_path.exists() and not force:
        print(f"[skip] {preset_name}: {output_path} already exists")
        return output_path

    codes = [str(c) for c in preset_cfg.get("codes", [])]
    if not codes:
        raise ValueError(f"Preset '{preset_name}' has empty codes list")

    train_sizes = [int(x) for x in preset_cfg.get("train_sizes", utils.TRAIN_SIZES)]
    seeds_expr = str(preset_cfg.get("seeds_expr", ""))
    if not seeds_expr:
        raise ValueError(f"Preset '{preset_name}' missing seeds_expr")
    seeds = _expand_numeric_expr(seeds_expr)

    runs = utils.build_runs(
        train_sizes=train_sizes,
        seeds=seeds,
        train_dir=train_dir,
        test_dir=test_dir,
        stage2_dir=stage2_dir,
        require_stage2=not allow_missing_stage2,
        codes=codes,
    )
    utils.save_manifest(
        runs,
        output_path,
        train_dir=train_dir,
        test_dir=test_dir,
        stage2_dir=stage2_dir,
        default_bridge_dir=bridge_dir,
    )
    print(f"[ok] {preset_name}: wrote {output_path} ({len(runs)} runs)")
    return output_path


def _add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--presets-path",
        type=Path,
        default=utils.DEFAULT_MANIFESTS_DIR / "presets.yaml",
        help="Path to presets.yaml",
    )
    parser.add_argument(
        "--generated-dir",
        type=Path,
        default=utils.DEFAULT_MANIFEST_GENERATED_DIR,
        help="Manifest output directory for presets",
    )
    parser.add_argument("--train-dir", type=Path, default=utils.DEFAULT_TRAIN_DIR)
    parser.add_argument("--test-dir", type=Path, default=utils.DEFAULT_TEST_DIR)
    parser.add_argument("--stage2-dir", type=Path, default=utils.DEFAULT_STAGE2_DIR)
    parser.add_argument("--bridge-dir", type=Path, default=utils.DEFAULT_BRIDGE_DIR)


def cmd_list_presets(args: argparse.Namespace) -> None:
    presets = _load_presets(args.presets_path)
    for name in sorted(presets.keys()):
        cfg = presets[name]
        codes = cfg.get("codes", [])
        sizes = cfg.get("train_sizes", [])
        seeds_expr = cfg.get("seeds_expr", "")
        output = _manifest_path_for_preset(args.generated_dir, cfg, name)
        print(
            f"{name:16s} codes={len(codes):2d} sizes={list(sizes)} seeds='{seeds_expr}' -> {output.name}"
        )


def cmd_show(args: argparse.Namespace) -> None:
    presets = _load_presets(args.presets_path)
    if args.preset not in presets:
        raise SystemExit(f"Unknown preset: {args.preset}")
    cfg = dict(presets[args.preset])
    cfg["name"] = args.preset
    cfg["manifest_path"] = str(_manifest_path_for_preset(args.generated_dir, presets[args.preset], args.preset))
    print(json.dumps(cfg, indent=2))


def cmd_ensure(args: argparse.Namespace) -> None:
    presets = _load_presets(args.presets_path)
    if args.preset not in presets:
        raise SystemExit(f"Unknown preset: {args.preset}")

    _ensure_single_preset(
        preset_name=args.preset,
        preset_cfg=presets[args.preset],
        generated_dir=args.generated_dir,
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        stage2_dir=args.stage2_dir,
        bridge_dir=args.bridge_dir,
        allow_missing_stage2=args.allow_missing_stage2,
        force=args.force,
    )


def cmd_ensure_all(args: argparse.Namespace) -> None:
    presets = _load_presets(args.presets_path)
    for name in sorted(presets.keys()):
        _ensure_single_preset(
            preset_name=name,
            preset_cfg=presets[name],
            generated_dir=args.generated_dir,
            train_dir=args.train_dir,
            test_dir=args.test_dir,
            stage2_dir=args.stage2_dir,
            bridge_dir=args.bridge_dir,
            allow_missing_stage2=args.allow_missing_stage2,
            force=args.force,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage sec5.1 manifest presets.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list-presets", help="List available presets")
    _add_common_paths(p_list)
    p_list.set_defaults(func=cmd_list_presets)

    p_show = sub.add_parser("show", help="Show one preset")
    _add_common_paths(p_show)
    p_show.add_argument("--preset", required=True)
    p_show.set_defaults(func=cmd_show)

    p_ensure = sub.add_parser("ensure", help="Ensure one preset manifest exists")
    _add_common_paths(p_ensure)
    p_ensure.add_argument("--preset", required=True)
    p_ensure.add_argument("--force", action="store_true")
    p_ensure.add_argument("--allow-missing-stage2", action="store_true")
    p_ensure.set_defaults(func=cmd_ensure)

    p_ensure_all = sub.add_parser("ensure-all", help="Ensure all preset manifests exist")
    _add_common_paths(p_ensure_all)
    p_ensure_all.add_argument("--force", action="store_true")
    p_ensure_all.add_argument("--allow-missing-stage2", action="store_true")
    p_ensure_all.set_defaults(func=cmd_ensure_all)

    args = parser.parse_args()
    args.generated_dir.mkdir(parents=True, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
