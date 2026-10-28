"""End-to-end smoke test: every pipeline step on the synthetic toy dataset."""
import zipfile

import pandas as pd
import pytest

from taxorec import pipelines as P
from taxorec.common import load_config, read_json
from taxorec.taxonomy.tree import Taxonomy


@pytest.fixture(scope="module")
def toy_cfg(tmp_path_factory):
    root = tmp_path_factory.mktemp("toy")
    return load_config("configs/toy.yaml", [
        f"paths.out_dir={(root / 'out').as_posix()}",
        f"amazon.reviews={(root / 'data' / 'Toy.jsonl.gz').as_posix()}",
        f"amazon.meta={(root / 'data' / 'meta_Toy.jsonl.gz').as_posix()}",
    ])


def test_test_documents_are_read_once(tmp_path):
    cfg = {"paths": {"out_dir": str(tmp_path)}, "evaluation": {"test_once": True}}
    P._open_test(cfg, "eval-taxonomy")
    P._open_test(cfg, "ablation")                      # each step once
    with pytest.raises(RuntimeError, match="once"):
        P._open_test(cfg, "eval-taxonomy")
    P._open_test(cfg, "eval-taxonomy", force=True)     # debugging only, and recorded
    log = read_json(tmp_path / "eval" / "test_access.json")
    assert [h["forced"] for h in log["history"]] == [False, False, True]


def test_taxonomy_names_skip_fingerprint_sidecars(tmp_path):
    L = P.Layout({"paths": {"out_dir": str(tmp_path)}})
    L.taxonomies.mkdir(parents=True)
    for name in ("improved.json", "improved.json.sha256.json", "improved.json.sha256.json.sha256.json",
                 "taxogen.json", "taxogen.txt"):
        (L.taxonomies / name).write_text("{}")
    assert P.taxonomy_names(L) == ["improved", "taxogen"]


def test_corpus_splits_are_disjoint(synthetic_corpus):
    s = P.corpus_splits({"seed": 42, "taxonomy": {"split": {"train": 0.2, "val": 0.3}}}, synthetic_corpus)
    tr, va, te = (set(s[k].doc_map.tolist()) for k in ("train", "val", "test"))
    assert not (tr & va or tr & te or va & te)
    assert len(tr | va | te) == synthetic_corpus.n_docs and set(s["build"].doc_map.tolist()) == tr | va
    assert abs(len(te) - 0.5 * synthetic_corpus.n_docs) <= 1


def test_select_params_rule():
    rows = [{"params": {"a": 1}, "npmi": 0.1, "subsumption": 0.5, "sibling_overlap": 0.2,
             "term_coverage": 0.9, "n_nodes": 20},
            {"params": {"a": 2}, "npmi": 0.3, "subsumption": 0.6, "sibling_overlap": 0.1,
             "term_coverage": 0.9, "n_nodes": 20},
            {"params": {"a": 3}, "npmi": 0.9, "subsumption": 0.9, "sibling_overlap": 0.0,
             "term_coverage": 0.05, "n_nodes": 20}]          # best metrics but too little coverage
    best, eligible = P.select_params(rows, {"min_term_coverage": 0.3, "min_nodes": 10})
    assert best["params"] == {"a": 2} and len(eligible) == 2


def test_full_pipeline_on_toy_data(toy_cfg):
    P.make_toy(toy_cfg)
    stats = P.prepare_rec(toy_cfg)
    assert stats["n_eval_users"] > 100

    L = P.Layout(toy_cfg)
    P.prepare_corpus(toy_cfg)
    held = pd.read_csv(L.heldout, dtype=str)
    docs = pd.read_csv(L.corpus / "docs_meta.csv", dtype=str)
    assert not set(zip(held["user_id"], held["item_id"])) & set(zip(docs["user_id"], docs["item_id"]))

    chosen = P.tune_taxonomy(toy_cfg)
    assert set(chosen) == {"improved", "taxogen", "sbert_kmeans"}

    sens = P.sensitivity(toy_cfg)                # one factor at a time, on val
    assert [(r["param"], r["value"]) for r in sens] == [("builder.tau", 0.05), ("builder.tau", 0.1),
                                                        ("encoders.sbert_ctx.drop_df", None),
                                                        ("encoders.sbert_ctx.drop_df", 0.3)]
    assert sum(r["is_tuned"] for r in sens if r["param"] == "builder.tau") == 1
    assert not (L.eval / "test_access.json").exists()     # sensitivity never reads test

    built = P.build_taxonomy(toy_cfg)          # bertopic is optional and skipped if not installed
    main = {k: v for k, v in built.items() if not k.endswith("bertopic")}
    assert all(s["n_degenerate_edges"] == 0 and s["n_nodes"] > 0 for s in main.values())
    P.build_taxonomy(toy_cfg, ["improved", "taxogen"], all_seeds=True)
    assert len(list((L.taxonomies / "seeds").glob("*.json"))) == 4

    summary_abl = P.ablation(toy_cfg)
    assert len(summary_abl) == 2 * 2 * 2 + 6 and all(r["n_seeds"] == 2 for r in summary_abl)
    assert [r["center"] for r in summary_abl].count(False) == 1       # the uncentred variant
    assert [r["reduce"] for r in summary_abl].count("none") == 1      # the no-UMAP variant
    assert [r["encoder"] for r in summary_abl].count("sbert") == 1    # term-only SBERT, split by terms
    assert {r["split"] for r in summary_abl} == {"docs", "terms"}

    summary = P.evaluate_taxonomies(toy_cfg, ["improved", "taxogen", "sbert_kmeans"])
    assert "pairF1@1" in summary["improved"] and summary["taxogen"]["n_seeds"] == 2
    # the main comparison: all methods pruned to the same size, every metric
    sizes = {m: summary[m]["n_nodes@matched_size"] for m in ("improved", "taxogen", "sbert_kmeans")}
    assert len(set(sizes.values())) == 1
    assert all(f"{k}@matched_size" in summary["improved"] for k in ("npmi", "subsumption", "pairF1@1"))

    batch_dir = P.make_human_eval(toy_cfg)
    assert (batch_dir / "relation_sheet.csv").exists() and (batch_dir / "forms_toy.csv").exists()
    key = read_json(batch_dir / "KEY_DO_NOT_SHARE.json")       # items come from same-size taxonomies
    taxes = {m: Taxonomy.load(L.taxonomy(m)) for m in ("improved", "taxogen")}
    n_human = min(t.size for t in taxes.values())
    kept = {m: set(t.prune_to_size(n_human).nodes) for m, t in taxes.items()}
    items = [i for task in ("relation", "intrusion") for i in key[task] if i["method"]]
    assert items and all(i["node_id"] in kept[i["method"]] for i in items)
    P.freeze(toy_cfg)
    with pytest.raises(RuntimeError, match="frozen"):
        P.make_human_eval(toy_cfg)
    with pytest.raises(FileNotFoundError, match="analyze-human-eval"):   # no partial Holm family
        P.human_eval_holm([toy_cfg])

    results = P.run_rec(toy_cfg)
    assert "Sem+Tax(Gold)" in results
    for res in results.values():
        assert 0 <= res["test"]["ndcg@10"] <= 1
    assert read_json(L.results / "rec_significance.json")

    report = P.report(toy_cfg).read_text(encoding="utf-8")
    assert "Recommendation (test split)" in report and "Ablation" in report and "Tuned" in report

    archive = P.archive(toy_cfg)
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
    assert "MANIFEST.json" in names and "taxonomies/improved.json" in names
    assert not any(n.endswith("docs.txt") or n.startswith("cache/") for n in names)
