import json

import numpy as np
import pytest
from statsmodels.stats.inter_rater import fleiss_kappa as sm_fleiss
from statsmodels.stats.proportion import proportion_confint

from taxorec.common import Run, config_hash, load_config, read_json, write_json
from taxorec.stats import (fleiss_kappa, holm, krippendorff_alpha_nominal, paired_bootstrap,
                           two_way_bootstrap_diff, wilson_ci)


def test_config_inheritance_and_overrides(tmp_path):
    (tmp_path / "base.yaml").write_text("a: 1\nb: {c: 2, d: 3}\nlst: [1, 2]\n", encoding="utf-8")
    (tmp_path / "child.yaml").write_text("_base_: base.yaml\nb: {d: 4}\nlst: [9]\n", encoding="utf-8")
    cfg = load_config(tmp_path / "child.yaml", ["b.c=7", "new.key=[1, 2]"])
    assert cfg == {"a": 1, "b": {"c": 7, "d": 4}, "lst": [9], "new": {"key": [1, 2]}}
    assert config_hash(cfg) == config_hash(json.loads(json.dumps(cfg)))


def test_json_handles_numpy(tmp_path):
    path = write_json(tmp_path / "x.json", {"i": np.int64(3), "f": np.float32(0.5), "a": np.arange(3)})
    assert read_json(path) == {"i": 3, "f": 0.5, "a": [0, 1, 2]}


def test_run_records_config_env_and_inputs(tmp_path):
    inp = tmp_path / "input.txt"
    inp.write_text("data", encoding="utf-8")
    run = Run(tmp_path / "runs", "demo", {"x": 1}, inputs=[inp], seed=1)
    run.save_json("result.json", {"ok": True})
    run.finish({"done": 1})
    for name in ("config.yaml", "env.json", "inputs.json", "log.txt", "result.json", "status.json"):
        assert (run.dir / name).exists()
    assert len(read_json(run.dir / "inputs.json")[0]["sha256"]) == 64


def test_wilson_matches_statsmodels():
    lo, hi = wilson_ci(37, 60)
    ref = proportion_confint(37, 60, method="wilson")
    assert lo == pytest.approx(ref[0]) and hi == pytest.approx(ref[1])


def test_fleiss_matches_statsmodels():
    rng = np.random.default_rng(0)
    counts = rng.multinomial(10, [0.6, 0.3, 0.1], size=40)
    assert fleiss_kappa(counts) == pytest.approx(sm_fleiss(counts))


def _alpha_bruteforce(data):
    pairs_obs, values = [], []
    for unit in data.T:
        v = unit[~np.isnan(unit)]
        m = len(v)
        if m < 2:
            continue
        values.extend(v)
        for i in range(m):
            for j in range(m):
                if i != j:
                    pairs_obs.append((v[i] != v[j]) / (m - 1))
    n = len(values)
    d_o = sum(pairs_obs) / n
    vals = np.array(values)
    d_e = sum((a != b) for i, a in enumerate(vals) for j, b in enumerate(vals) if i != j) / (n * (n - 1))
    return 1 - d_o / d_e


def test_krippendorff_alpha():
    rng = np.random.default_rng(1)
    data = rng.integers(0, 3, size=(4, 12)).astype(float)
    data[0, 3] = np.nan
    data[2, 7] = np.nan
    assert krippendorff_alpha_nominal(data) == pytest.approx(_alpha_bruteforce(data))
    assert krippendorff_alpha_nominal(np.tile([0, 1, 1, 0, 2], (5, 1)).astype(float)) == pytest.approx(1.0)


def test_paired_bootstrap_and_holm():
    rng = np.random.default_rng(0)
    a = rng.random(500)
    same = paired_bootstrap(a, a, n_boot=500)
    assert same["diff"] == 0 and same["p_value"] == 1.0
    better = paired_bootstrap(a + 0.2, a, n_boot=500)
    assert better["p_value"] < 0.01 and better["ci_low"] > 0
    assert holm([0.01, 0.04, 0.03]).tolist() == pytest.approx([0.03, 0.06, 0.06])


def test_two_way_bootstrap_detects_difference():
    rng = np.random.default_rng(0)
    a = (rng.random((40, 9)) < 0.85).astype(float)
    b = (rng.random((40, 9)) < 0.40).astype(float)
    res = two_way_bootstrap_diff(a, b, n_boot=400)
    assert res["diff"] > 0.3 and res["p_value"] < 0.01
