"""Automatic taxonomy evaluation and document/item-to-path assignment.

Intrinsic (corpus only): NPMI term coherence, parent-child subsumption,
sibling overlap, term/document coverage, structure statistics.
External (silver standard): items are assigned to taxonomy paths from their
text alone; the induced item partition at each depth is compared with gold
category paths (pairwise P/R/F1, NMI, ARI).

Assignment is lexical (term matching), deliberately independent of any
embedding model, so the taxonomy signal used downstream is not a disguised
copy of the semantic signal.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix

from taxorec.data import Corpus
from taxorec.taxonomy.tree import ROOT, Taxonomy


# ---------------------------------------------------------------- assignment

def taxonomy_vocab(tax: Taxonomy) -> list[str]:
    return sorted(set().union(*(n.terms for n in tax.non_root()))) if tax.size else []


def assign_paths(tax: Taxonomy, counts: sp.csr_matrix, vocab: list[str], top_k: int = 20,
                 min_score: float = 0.0) -> list[tuple[str, ...]]:
    """Top-down greedy assignment. counts: (items x vocab) term counts. At each
    node an item moves to the child whose top_k ranked terms carry the most
    tf-idf mass (rank-discounted); it stops when no child scores > min_score."""
    index = {t: i for i, t in enumerate(vocab)}
    n_items = counts.shape[0]
    df = np.diff(counts.tocsc().indptr)
    idf = np.log((1 + n_items) / (1 + df)) + 1.0
    weighted = sp.csr_matrix(counts.multiply(idf[None, :]))
    paths: list[list[str]] = [[] for _ in range(n_items)]
    frontier = {ROOT: np.arange(n_items)}
    while frontier:
        nxt = {}
        for node_id, items in frontier.items():
            kids = tax.children(node_id)
            if not kids or len(items) == 0:
                continue
            w = np.zeros((len(vocab), len(kids)))
            for c, kid in enumerate(kids):
                for rank, term in enumerate(kid.terms[:top_k]):
                    if term in index:
                        w[index[term], c] = 1.0 / np.log2(rank + 2)
            scores = np.asarray(weighted[items] @ w)
            best = scores.argmax(axis=1)
            ok = scores.max(axis=1) > min_score
            for c, kid in enumerate(kids):
                chosen = items[ok & (best == c)]
                for i in chosen:
                    paths[i].append(kid.id)
                if len(chosen):
                    nxt[kid.id] = chosen
        frontier = nxt
    return [tuple(p) for p in paths]


# ---------------------------------------------------------------- intrinsic

def npmi(corpus: Corpus, term_ids) -> float:
    term_ids = np.asarray(term_ids, dtype=np.int64)
    if len(term_ids) < 2:
        return float("nan")
    b = (corpus.term_doc[:, term_ids] > 0).astype(np.float64)
    co = (b.T @ b).toarray()
    n = corpus.n_docs
    p = np.diag(co) / n
    vals = []
    for i, j in combinations(range(len(term_ids)), 2):
        pij = co[i, j] / n
        if pij == 0 or p[i] == 0 or p[j] == 0:
            vals.append(-1.0)
        elif pij == 1:
            vals.append(1.0)
        else:
            vals.append(np.log(pij / (p[i] * p[j])) / -np.log(pij))
    return float(np.mean(vals))


def _top_docs(corpus: Corpus, terms: list[str], k: int) -> np.ndarray:
    return corpus.docs_with_any(corpus.ids(terms[:k]))


def evaluate_intrinsic(corpus: Corpus, tax: Taxonomy, top_k: int = 10) -> dict:
    nodes = list(tax.non_root())
    coherence = {n.id: npmi(corpus, corpus.ids(n.terms[:top_k])) for n in nodes}
    docs = {n.id: _top_docs(corpus, n.terms, top_k) for n in nodes}

    subsumption = []
    for parent, child in tax.edges():
        if len(docs[child.id]):
            inter = np.intersect1d(docs[child.id], docs[parent.id], assume_unique=True)
            subsumption.append(len(inter) / len(docs[child.id]))

    overlaps = []
    for node_id in [ROOT] + [n.id for n in nodes]:
        kids = tax.children(node_id)
        for a, b in combinations(kids, 2):
            da, db = docs[a.id], docs[b.id]
            union = len(np.union1d(da, db))
            if union:
                overlaps.append(len(np.intersect1d(da, db, assume_unique=True)) / union)

    placed_ids = corpus.ids(sorted(set().union(*(n.terms for n in nodes)))) if nodes else np.zeros(0, int)
    level1 = tax.children(ROOT)
    covered_docs = corpus.docs_with_any(corpus.ids(sorted(set().union(*(n.terms for n in level1))))) \
        if level1 else np.zeros(0)
    finite = [v for v in coherence.values() if np.isfinite(v)]
    by_level = defaultdict(list)
    for n in nodes:
        if np.isfinite(coherence[n.id]):
            by_level[n.level].append(coherence[n.id])
    return {
        "structure": tax.stats(),
        "npmi_mean": float(np.mean(finite)) if finite else float("nan"),
        "npmi_by_level": {lvl: float(np.mean(v)) for lvl, v in sorted(by_level.items())},
        "subsumption_mean": float(np.mean(subsumption)) if subsumption else float("nan"),
        "sibling_overlap_mean": float(np.mean(overlaps)) if overlaps else float("nan"),
        "term_coverage": len(placed_ids) / corpus.n_terms if corpus.n_terms else 0.0,
        "doc_coverage": len(covered_docs) / corpus.n_docs if corpus.n_docs else 0.0,
        "per_node_npmi": coherence,
    }


# ---------------------------------------------------------------- external

def labels_at_depth(paths: list[tuple], depth: int, unassigned_singletons: bool = True) -> list:
    """Cluster label of each item at a depth. Items whose path stops earlier
    keep their deepest node; unassigned items become singletons."""
    out = []
    for i, p in enumerate(paths):
        if not p:
            out.append(f"__single_{i}" if unassigned_singletons else None)
        else:
            out.append("/".join(map(str, p[:depth])))
    return out


def pairwise_prf(pred: list, gold: list) -> dict:
    cm = contingency_matrix(gold, pred, sparse=True).astype(np.float64)

    def pairs(x):
        return float((x * (x - 1) / 2).sum())

    tp = pairs(cm.data)
    pred_pairs = pairs(np.asarray(cm.sum(axis=0)).ravel())
    gold_pairs = pairs(np.asarray(cm.sum(axis=1)).ravel())
    p = tp / pred_pairs if pred_pairs else 0.0
    r = tp / gold_pairs if gold_pairs else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def evaluate_external(pred_paths: list[tuple], gold_paths: list[tuple], depths=(1, 2, 3)) -> dict:
    out = {"n_items": len(pred_paths),
           "assigned": float(np.mean([bool(p) for p in pred_paths])) if pred_paths else 0.0}
    for d in depths:
        keep = [i for i, g in enumerate(gold_paths) if len(g) >= d]
        if len(keep) < 2:
            continue
        gold = ["/".join(gold_paths[i][:d]) for i in keep]
        pred = labels_at_depth([pred_paths[i] for i in keep], d)
        out[f"depth{d}"] = {"n_items": len(keep), **pairwise_prf(pred, gold),
                            "nmi": float(normalized_mutual_info_score(gold, pred)),
                            "ari": float(adjusted_rand_score(gold, pred))}
    return out
