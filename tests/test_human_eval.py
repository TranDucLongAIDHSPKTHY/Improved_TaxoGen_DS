from collections import Counter

import numpy as np
import pandas as pd
import pytest

from taxorec.human_eval import (SampleConfig, analyze, build_batch, forms_rows, import_form_export,
                                integrity_report, level_quotas, load_responses, sample_items,
                                validate_responses, write_batch)
from taxorec.taxonomy.builders import BuilderConfig, TaxonomyBuilder


@pytest.fixture(scope="module")
def batch(synthetic_corpus, synthetic_encoder):
    def build(**kw):
        cfg = dict(clusterer="kmeans", k=3, relation="none", local="none", max_depth=2,
                   min_terms_to_split=6, min_docs_to_split=10, min_child_terms=2)
        cfg.update(kw)
        return TaxonomyBuilder(BuilderConfig.from_dict(cfg), synthetic_corpus, synthetic_encoder,
                               log=lambda m: None).build()
    taxes = {"A": build(), "B": build(k=2)}
    samples = sample_items(taxes, SampleConfig(n_relation=6, n_intrusion=6, display_k=3, seed=1))
    return samples, build_batch(samples, "synthetic", seed=1)


def test_level_quotas_are_common_and_capped():
    q = level_quotas([Counter({1: 3, 2: 9}), Counter({1: 5, 2: 40, 3: 7})], 10)
    assert set(q) == {1, 2} and sum(q.values()) == 10 and q[1] <= 3
    assert level_quotas([Counter({1: 2}), Counter({1: 9})], 10) == {1: 2}


def test_samples_are_level_matched(batch):
    samples, _ = batch
    per_method = {m: Counter(it["level"] for it in items) for m, items in samples["relation"].items()}
    assert per_method["A"] == per_method["B"]
    for items in samples["intrusion"].values():
        for it in items:
            assert it["terms"][it["intruder_index"]] == it["intruder"]


def test_sheets_are_blinded(batch, tmp_path):
    _, b = batch
    out = write_batch(b, tmp_path)
    rel = pd.read_csv(out / "relation_sheet.csv")
    intr = pd.read_csv(out / "intrusion_sheet.csv")
    assert list(rel.columns) == ["item_id", "child_terms", "parent_terms", "answer"]
    assert "code" not in intr.columns and "method" not in intr.columns
    assert rel["child_terms"].ne(rel["parent_terms"]).all()   # no 'identical lists' check items


def _simulate(b, task, n_ann=12, seed=0, fabricate=False):
    rng = np.random.default_rng(seed)
    items = b[task]
    rows = []
    base_p = {it["item_id"]: rng.uniform(0.2, 0.95) for it in items}
    if fabricate:
        # the pattern found in the old data: every annotator has the same number of 'yes'
        real = [it for it in items if not it["is_check"]]
        base = np.array([rng.random() < base_p[it["item_id"]] for it in real])
        for a in range(n_ann):
            ans = base.copy()
            yes, no = np.where(ans)[0], np.where(~ans)[0]
            k = min(len(yes), len(no), 3)
            ans[rng.choice(yes, k, replace=False)] = False
            ans[rng.choice(no, k, replace=False)] = True
            rows += [{"annotator_id": f"a{a}", "item_id": it["item_id"],
                      "answer": "yes" if v else "no"} for it, v in zip(real, ans)]
            rows += [{"annotator_id": f"a{a}", "item_id": it["item_id"], "answer": it["expected"]}
                     for it in items if it["is_check"]]
        return pd.DataFrame(rows)
    for a in range(n_ann):
        leniency = rng.normal(0, 0.8)
        for it in items:
            p = 1 / (1 + np.exp(-(np.log(base_p[it["item_id"]] / (1 - base_p[it["item_id"]])) + leniency)))
            if it["is_check"]:
                truth = it["expected"] if task == "relation" else str(it["intruder_index"] + 1)
                rows.append({"annotator_id": f"a{a}", "item_id": it["item_id"], "answer": truth})
                continue
            if task == "relation":
                ans = "yes" if rng.random() < p else "no"
            else:
                right = str(it["intruder_index"] + 1)
                ans = right if rng.random() < p else str(rng.choice([x for x in "123456" if x != right]))
            rows.append({"annotator_id": f"a{a}", "item_id": it["item_id"], "answer": ans})
    return pd.DataFrame(rows)


def test_integrity_passes_independent_annotators(batch):
    _, b = batch
    for task in ("relation", "intrusion"):
        rep = integrity_report(_simulate(b, task), task, b)
        assert rep["ok"], rep["flags"]
        assert not rep["failed_attention"]


def test_form_timestamps_in_vietnamese_locale():
    from taxorec.human_eval import form_timestamp
    assert form_timestamp("2026/10/27 1:29:59 CH GMT+7") == "2026-10-27T13:29:59+07:00"
    assert form_timestamp("2026/10/27 12:05:00 SA GMT+7") == "2026-10-27T00:05:00+07:00"
    assert form_timestamp("2026/11/02 10:00:00") == "2026-11-02T10:00:00"
    assert form_timestamp("yesterday") == ""


def test_two_wrong_attention_checks_exclude(batch):
    _, b = batch
    long = _simulate(b, "intrusion")
    checks = [it for it in b["intrusion"] if it["is_check"]]
    assert len(checks) >= 2

    def wrong(ann, n):
        for it in checks[:n]:
            right = str(it["intruder_index"] + 1)
            long.loc[(long["annotator_id"] == ann) & (long["item_id"] == it["item_id"]), "answer"] = \
                "1" if right != "1" else "2"
    wrong("a0", 1)
    wrong("a1", 2)
    assert set(integrity_report(long, "intrusion", b)["failed_attention"]) == {"a1"}


def test_integrity_flags_uniform_marginals(batch):
    _, b = batch
    rep = integrity_report(_simulate(b, "relation", fabricate=True), "relation", b)
    assert not rep["ok"]
    assert rep["distinct_count_vectors"] == 1
    with pytest.raises(ValueError, match="integrity"):
        analyze(_simulate(b, "relation", fabricate=True), "relation", b, n_boot=50)


def test_forms_rows_and_export_roundtrip(batch, tmp_path):
    _, b = batch
    rows = forms_rows(b)
    assert len(rows) == len(b["relation"]) + len(b["intrusion"])
    assert all(r["title"].startswith(r["item_id"]) for r in rows)
    # a Google Forms export: one row per annotator, one column per question title
    titles = {r["item_id"]: r["title"] for r in rows}
    # the third person declined consent, so the form ended with no ID and no answers
    export = {"Timestamp": ["2026/11/02 10:00:00", "2026/11/02 11:30:00", "2026/11/02 12:00:00"],
              "Đồng ý tham gia / Consent": ["Đồng ý tham gia / I agree to take part"] * 2
              + ["Không đồng ý / I do not agree"],
              "Mã người chấm / Annotator ID (A01-A20)": ["A01", "A02", ""]}
    for it in b["relation"]:
        export[titles[it["item_id"]]] = ["yes", "no", ""]
    for it in b["intrusion"]:
        export[titles[it["item_id"]]] = [f"{it['intruder_index'] + 1}. {it['intruder']}", "1. x", ""]
    pd.DataFrame(export).to_csv(tmp_path / "export.csv", index=False)
    out = import_form_export(tmp_path / "export.csv", tmp_path / "responses")
    rel = load_responses([out["relation"]], "relation")
    intr = load_responses([out["intrusion"]], "intrusion")
    assert set(rel["annotator_id"]) == {"A01", "A02"} and set(rel["answer"]) == {"yes", "no"}
    assert not validate_responses(rel, "relation", [it["item_id"] for it in b["relation"]])
    assert not validate_responses(intr, "intrusion", [it["item_id"] for it in b["intrusion"]])
    a01 = intr[intr["annotator_id"] == "A01"].set_index("item_id")["answer"]
    assert all(a01[it["item_id"]] == str(it["intruder_index"] + 1) for it in b["intrusion"])


def test_analyze_and_validate(batch, tmp_path):
    _, b = batch
    long = _simulate(b, "relation", seed=3)
    for ann, grp in long.groupby("annotator_id"):
        grp[["item_id", "answer"]].to_csv(tmp_path / f"{ann}_relation.csv", index=False)
    loaded = load_responses(sorted(tmp_path.glob("*_relation.csv")), "relation")
    assert not validate_responses(loaded, "relation", [it["item_id"] for it in b["relation"]])
    res = analyze(loaded, "relation", b, n_boot=100)
    assert set(res["methods"]) == {"A", "B"}
    assert "A vs B" in res["comparisons"]
    assert -1 <= res["krippendorff_alpha"] <= 1
    assert res["krippendorff_alpha_band"] in {"reliable", "tentative", "low"}
    votes = res["comparisons"]["A vs B"]["annotators"]
    assert votes["favour_a"] + votes["favour_b"] + votes["ties"] == res["n_annotators"]
    assert 0 <= votes["sign_test_p"] <= 1


def test_primary_holm_adjusts_the_whole_family():
    from taxorec.human_eval import primary_holm

    def res(diff, p):
        return {"integrity": {"flags": []},
                "comparisons": {"improved vs taxogen": {"acc_a": 0.5 + diff, "acc_b": 0.5, "diff": diff,
                                                        "ci_low": diff - 0.1, "ci_high": diff + 0.1,
                                                        "p_value": p}}}
    rows = primary_holm({"x/relation": res(0.2, 0.001), "x/intrusion": res(0.1, 0.02),
                         "y/relation": res(-0.2, 0.001), "y/intrusion": res(0.05, 0.03)})
    assert len(rows) == 4
    assert [r["p_holm"] for r in rows] == pytest.approx([0.004, 0.04, 0.004, 0.04])
    # a significant difference in favour of the baseline is not 'a better'
    assert [r["conclusion"] for r in rows] == ["a better", "a better", "not established", "a better"]
    assert primary_holm({"x/relation": res(0.1, 0.02)})[0]["p_holm"] == pytest.approx(0.02)


def test_agreement_bands_are_fixed():
    from taxorec.stats import alpha_label, kappa_label, sign_test
    assert [kappa_label(k) for k in (-0.1, 0.2, 0.21, 0.6, 0.61, 0.81)] == \
        ["poor", "slight", "fair", "moderate", "substantial", "almost perfect"]
    assert [alpha_label(a) for a in (0.8, 0.7, 0.66)] == ["reliable", "tentative", "low"]
    assert sign_test(15, 5) == pytest.approx(0.0414, abs=1e-3) and sign_test(0, 0) == 1.0
