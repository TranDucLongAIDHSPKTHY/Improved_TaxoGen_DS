"""Fidelity check of the TaxoGen re-implementation on DBLP, following the rules
fixed in docs/KIEM_TRA_TAXOGEN_GOC.md. Outputs go to outputs/dblp/fidelity/.

    python tools/taxogen_fidelity.py convert --root <orig>/data/dblp/our-l3-0.15 --compressed <compress.py output>
    python tools/taxogen_fidelity.py build-c       # re-implementation with the original word2vec settings
    python tools/taxogen_fidelity.py compare       # val documents, verdict
    python tools/taxogen_fidelity.py test          # one extra, logged reading of the test documents
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np

from taxorec import data as D
from taxorec import pipelines as P
from taxorec.common import ensure_dir, load_config, write_json
from taxorec.taxonomy.evaluate import evaluate_intrinsic
from taxorec.taxonomy.tree import ROOT, Taxonomy

CONFIG = "configs/dblp.yaml"
# word2vec as compiled in the original repository (code/word2vec.c defaults; main.py passes only -threads)
ORIGINAL_W2V = {"sg": 0, "vector_size": 50, "epochs": 50, "negative": 15, "sample": 1e-4,
                "min_count": 10, "window": 5}
METRICS = ("npmi", "subsumption", "sibling_overlap", "term_coverage")


def out_dir(cfg: dict) -> Path:
    return ensure_dir(P.Layout(cfg).out / "fidelity")


def convert(root: Path, compressed: Path) -> Taxonomy:
    """compress.py writes '*/a/b<TAB>term,term,...' per node; the node's documents
    are the lines of its folder's doc_ids.txt."""
    tax = Taxonomy("taxogen_orig", {"source": str(compressed), "root": str(root)})
    by_path = {"*": ROOT}
    rows = [line.rstrip("\n").split("\t") for line in open(compressed, encoding="utf-8") if "\t" in line]
    for path, terms in sorted(rows, key=lambda r: (r[0].count("/"), r[0])):
        ids = root / Path(*path.split("/")[1:]) / "doc_ids.txt"
        n_docs = sum(1 for line in open(ids, encoding="utf-8") if line.strip()) if ids.exists() else 0
        node = tax.add_child(by_path[path.rsplit("/", 1)[0]], [t for t in terms.split(",") if t], n_docs)
        by_path[path] = node.id
    return tax


def build_c(cfg: dict) -> Taxonomy:
    L = P.Layout(cfg)
    spec = copy.deepcopy(P._method_spec(cfg, "taxogen", print))   # with the paper tree's tuned values
    spec["taxogen"]["w2v"] = {**(spec["taxogen"].get("w2v") or {}), **ORIGINAL_W2V}
    corpus = P.corpus_splits(cfg, D.Corpus.load(L.corpus))["build"]
    tax = P._build(cfg, spec, corpus, "taxogen_c", print)
    tax.save(out_dir(cfg) / "taxogen_c.json")
    return tax


def level1_overlap(ref: Taxonomy, other: Taxonomy, k: int = 10) -> float:
    """Mean over ref's level-1 nodes of the best Jaccard with one of other's (top-k terms)."""
    a = [set(n.terms[:k]) for n in ref.children(ROOT)]
    b = [set(n.terms[:k]) for n in other.children(ROOT)]
    return float(np.mean([max(len(x & y) / len(x | y) for y in b) for x in a])) if a and b else 0.0


def _metrics(corpus, tax: Taxonomy, top_k: int) -> dict:
    res = evaluate_intrinsic(corpus, tax, top_k)
    return {**P._flat_metrics(res), "nodes_per_level": res["structure"]["nodes_per_level"]}


def compare(cfg: dict) -> dict:
    L, top_k = P.Layout(cfg), cfg["evaluation"].get("top_k", 10)
    val = P.corpus_splits(cfg, D.Corpus.load(L.corpus))["val"]
    trees = {"T_paper": Taxonomy.load(L.taxonomy("taxogen")),
             "T_orig": Taxonomy.load(out_dir(cfg) / "taxogen_orig.json"),
             "T_C": Taxonomy.load(out_dir(cfg) / "taxogen_c.json")}
    n = min(t.size for t in trees.values())
    seeds = [Taxonomy.load(p) for p in sorted((L.taxonomies / "seeds").glob("taxogen@seed*.json"))
             if not p.name.endswith(".sha256.json")]
    seed_rows = [_metrics(val, t.prune_to_size(n), top_k) for t in seeds]
    margin = {m: max(2 * float(np.std([r[m] for r in seed_rows], ddof=1)), 0.01) for m in METRICS}
    matched = {k: _metrics(val, t.prune_to_size(n), top_k) for k, t in trees.items()}
    full = {k: _metrics(val, t, top_k) for k, t in trees.items()}

    def weaker(name: str) -> dict:
        o, p = matched["T_orig" if name == "T_paper" else name], matched["T_paper"]
        if name == "T_paper":       # is the paper's tree weaker than the original code's?
            hits = {"npmi": o["npmi"] - p["npmi"] > margin["npmi"],
                    "subsumption": o["subsumption"] - p["subsumption"] > margin["subsumption"],
                    "sibling_overlap": p["sibling_overlap"] - o["sibling_overlap"] > margin["sibling_overlap"]}
        else:                       # descriptive: does the original word2vec change the tree?
            hits = {m: abs(o[m] - p[m]) > margin[m] for m in ("npmi", "subsumption", "sibling_overlap")}
        return {"criteria": hits, "verdict": any(hits.values())}

    out = {"rules": "docs/KIEM_TRA_TAXOGEN_GOC.md", "documents": "val", "matched_size": n,
           "n_seed_trees": len(seed_rows), "margin": margin, "matched": matched, "full": full,
           "level1_overlap_with_T_orig": {k: level1_overlap(trees["T_orig"], t)
                                          for k, t in trees.items() if k != "T_orig"},
           "T_paper_weaker_than_original": weaker("T_paper"),
           "T_C_differs_from_T_paper": weaker("T_C")}
    write_json(out_dir(cfg) / "compare_val.json", out)
    return out


def test(cfg: dict) -> dict:
    """The extra test reading fixed in section 5 of the rules; logged under its own step."""
    P._open_test(cfg, "taxogen-fidelity")
    L, top_k = P.Layout(cfg), cfg["evaluation"].get("top_k", 10)
    corpus = P.corpus_splits(cfg, D.Corpus.load(L.corpus))["test"]
    trees = {"improved": Taxonomy.load(L.taxonomy("improved")),
             "T_paper": Taxonomy.load(L.taxonomy("taxogen")),
             "T_orig": Taxonomy.load(out_dir(cfg) / "taxogen_orig.json"),
             "T_C": Taxonomy.load(out_dir(cfg) / "taxogen_c.json")}
    n = min(t.size for t in trees.values())
    out = {"documents": "test", "matched_size": n,
           "matched": {k: _metrics(corpus, t.prune_to_size(n), top_k) for k, t in trees.items()}}
    write_json(out_dir(cfg) / "test.json", out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["convert", "build-c", "compare", "test"])
    ap.add_argument("--root", type=Path, help="convert: the original code's taxonomy folder")
    ap.add_argument("--compressed", type=Path, help="convert: compress.py output")
    args = ap.parse_args()
    cfg = load_config(CONFIG)
    if args.step == "convert":
        tax = convert(args.root, args.compressed)
        tax.save(out_dir(cfg) / "taxogen_orig.json")
        print(f"[fidelity] T_orig: {tax.size} nodes, {tax.depth} levels")
    elif args.step == "build-c":
        print(f"[fidelity] T_C: {build_c(cfg).size} nodes")
    else:
        res = compare(cfg) if args.step == "compare" else test(cfg)
        print(res)


if __name__ == "__main__":
    main()
