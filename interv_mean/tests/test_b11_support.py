from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = REPO_ROOT / "tabcf_core"
PIPELINE_DIR = REPO_ROOT / "interv_mean" / "pipeline"

for candidate in (CORE_DIR, PIPELINE_DIR, REPO_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import dgp as dgp
import stage2_outcome as stage2
import utils
import plot_div_style as vis


def test_b11_helpers_match_core_and_stage2_paths():
    cfg = dgp.DGPConfig(first_stage="A3", second_stage="B11")
    x = np.array([-0.25, 0.5, 1.75], dtype=float)
    h = np.array([0.1, -0.4, 0.8], dtype=float)
    eps = np.array([-0.2, 0.3, -0.5], dtype=float)
    expected = 1.0 + 2.0 * x + np.cos(2.0 * x) + x * h - h + eps

    stage2_vals = np.array(
        [
            stage2.simulate_y_given_x_eps(
                cfg,
                float(x_i),
                np.array([eps_i], dtype=float),
                h_draws=np.array([h_i], dtype=float),
            )[0]
            for x_i, h_i, eps_i in zip(x, h, eps)
        ]
    )
    core_vals = dgp.simulate_second_stage(
        cfg,
        x,
        np.zeros_like(x),
        eps,
        latent_h=h,
    )
    mc_vals = utils.compute_y_clean_monte_carlo(cfg, x, n_samples=256, rng_seed=17)
    expected_mc = 1.0 + 2.0 * x + np.cos(2.0 * x) + x * 0.0

    assert np.allclose(core_vals, expected)
    assert np.allclose(stage2_vals, expected)
    assert np.allclose(mc_vals, expected_mc, atol=0.2)


def test_b11_codes_manifest_support_and_vis_labels():
    assert "A3_B11" in utils.CODE_SCENARIO_MAP
    assert "A9_B11" in utils.CODE_SCENARIO_MAP
    assert utils.make_dgp_config("A3_B11").second_stage == "B11"
    assert utils.make_dgp_config("A9_B11").first_stage == "A9"
    assert vis._table_second_stage_label("B11") == "Periodic,\nNonadditive"

    repo_tmp = REPO_ROOT / "tmp" / "test_b11_codes_manifest_support_and_vis_labels"
    if repo_tmp.exists():
        for child in sorted(repo_tmp.rglob("*"), reverse=True):
            if child.is_file():
                child.unlink()
            else:
                child.rmdir()
        repo_tmp.rmdir()

    train_dir = repo_tmp / "train"
    test_dir = repo_tmp / "test"
    stage2_dir = repo_tmp / "stage2"
    bridge_dir = repo_tmp / "bridge"
    manifest_path = repo_tmp / "manifest.json"

    train_dir.mkdir(parents=True)
    test_dir.mkdir()
    stage2_dir.mkdir()
    bridge_dir.mkdir()

    (train_dir / "train_data_A3_B11_n4000_seed1.csv").touch()
    (test_dir / "test_data_A3_B11.csv").touch()

    runs = utils.build_runs(
        train_sizes=[4000],
        seeds=[1],
        train_dir=train_dir,
        test_dir=test_dir,
        stage2_dir=stage2_dir,
        require_stage2=False,
        codes=["A3_B11"],
    )
    utils.save_manifest(
        runs,
        manifest_path,
        train_dir=train_dir,
        test_dir=test_dir,
        stage2_dir=stage2_dir,
        default_bridge_dir=bridge_dir,
    )

    manifest = utils.load_manifest(manifest_path)
    assert manifest["runs"][0]["code"] == "A3_B11"
