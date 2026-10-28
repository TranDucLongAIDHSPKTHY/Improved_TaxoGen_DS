"""Taxonomy construction methods.

TaxonomyBuilder   recursive (encoder x clusterer x relation scorer) builder.
                  TaxoGen-Improved = SBERT + HDBSCAN + BM25-cosine; every other
                  combination is an ablation or a simpler baseline
                  (e.g. SBERT + spherical k-means, no relation scoring).
TaxoGen           re-implementation of Zhang et al. (KDD 2018): spherical
                  k-means, CaseOLAP representativeness filtering, local
                  word2vec embeddings.
build_bertopic    flat BERTopic topics as a depth-1 taxonomy (optional extra).
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass, fields
from typing import Callable

import numpy as np
import scipy.sparse as sp

from taxorec.data import Corpus
from taxorec.taxonomy.components import (
    Encoder, RelationScorer, Word2VecEncoder, adaptive_min_cluster_size, caseolap_scores,
    cell_statistics, hdbscan_labels, kl_representativeness, rank_terms, recenter,
    spherical_kmeans, umap_reduce,
)
from taxorec.taxonomy.tree import ROOT, Taxonomy


def _from_dict(cls, d: dict):
    known = {f.name for f in fields(cls)}
    unknown = set(d) - known
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**d)


def _mean_vector(emb: np.ndarray, ids) -> np.ndarray:
    v = emb[np.asarray(ids, dtype=np.int64)].mean(axis=0)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _invalid_issues(tax: Taxonomy) -> list[str]:
    return [msg for sev, msg in tax.validate() if sev == "error"]


# ---------------------------------------------------------------- generic builder

@dataclass
class BuilderConfig:
    clusterer: str = "hdbscan"          # hdbscan | kmeans
    split: str = "terms"                # terms: cluster term vectors | docs: cluster the node's documents
    split_max_docs: int = 20000         # docs: documents sampled per node for clustering
    split_min_kl: float = 0.25          # docs: a term joins a child only if its CaseOLAP scores are this far from uniform
    split_min_df: int = 0               # docs: a term joins a child only if it occurs in this many of its sampled documents
    split_min_df_frac: float = 0.0      # docs: ... and in at least this fraction of them (scales with the cluster)
    split_attach_min_sim: float | None = None  # docs: rare terms join the child with the closest anchor centroid (cosine >= this)
    center: bool = False                # subtract the mean term vector before any cosine (context vectors share one direction)
    reduce: str = "none"                # none | umap (before clustering, at every node)
    umap_dim: int = 5
    umap_neighbors: int = 15
    k: int = 5                          # kmeans only
    hdb_divisor: float = 15.0           # min_cluster_size = pool / divisor, clamped
    hdb_floor: int = 15
    hdb_ceiling: int = 2000
    hdb_min_samples: int = 3
    hdb_selection: str = "leaf"
    relation: str = "bm25cos"           # none | cos | bm25 | bm25cos
    w: float = 0.5
    tau: float = 0.3
    profile_k: int = 10
    score_root_children: bool = False   # the root has no topic to be related to
    local: str = "recenter"             # none | recenter | w2v
    max_depth: int = 3
    min_terms_to_split: int = 30
    min_docs_to_split: int = 20
    min_child_terms: int = 5
    min_children: int = 2
    max_local_docs: int = 20000
    w2v: dict | None = None             # word2vec params for local="w2v"
    seed: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "BuilderConfig":
        return _from_dict(cls, d)


class TaxonomyBuilder:
    def __init__(self, cfg: BuilderConfig, corpus: Corpus, encoder: Encoder,
                 method: str = "improved", log: Callable[[str], None] = print):
        self.cfg, self.corpus, self.encoder, self.method, self.log = cfg, corpus, encoder, method, log
        self.scorer = RelationScorer(cfg.relation, cfg.w, cfg.tau, cfg.profile_k)

    def cluster(self, x: np.ndarray) -> np.ndarray:
        c = self.cfg
        if c.clusterer not in ("kmeans", "hdbscan"):
            raise ValueError(f"unknown clusterer {c.clusterer!r}")
        if c.reduce not in ("none", "umap"):
            raise ValueError(f"unknown reduce {c.reduce!r}")
        unit = c.reduce == "none"
        if not unit:
            x = umap_reduce(x, c.umap_dim, c.umap_neighbors, seed=c.seed)
        if c.clusterer == "kmeans":
            return (spherical_kmeans(x, c.k, seed=c.seed, normalize=unit) if len(x) > c.k
                    else np.full(len(x), -1))
        mcs = adaptive_min_cluster_size(len(x), c.hdb_divisor, c.hdb_floor, c.hdb_ceiling)
        return hdbscan_labels(x, mcs, c.hdb_min_samples, c.hdb_selection, normalize=unit)

    def build(self) -> Taxonomy:
        corpus = self.corpus
        emb = self.encoder.encode_terms(corpus.terms)
        pool = np.where(np.linalg.norm(emb, axis=1) > 0)[0]
        if self.cfg.center:
            # averaged context vectors all point one way (random term pairs have
            # cosine ~0.6 on reviews), which makes every cosine threshold pass
            emb = emb.copy()
            emb[pool] = recenter(emb[pool])
        self.log(f"[{self.method}] {len(pool)}/{corpus.n_terms} terms have a representation")
        tax = Taxonomy(self.method, {**asdict(self.cfg), "encoder": self.encoder.name})
        all_docs = np.arange(corpus.n_docs)
        ranked = rank_terms(corpus, pool, all_docs)
        tax.root.terms = [corpus.terms[i] for i in ranked]
        tax.root.n_docs = corpus.n_docs
        self._expand(tax, ROOT, ranked, emb, all_docs, level=0)
        problems = _invalid_issues(tax)
        if problems:
            raise RuntimeError(f"invalid taxonomy: {problems[:5]}")
        self.log(f"[{self.method}] done: {tax.stats()}")
        return tax

    def _expand(self, tax: Taxonomy, node_id: str, pool_ranked: np.ndarray, emb: np.ndarray,
                docs: np.ndarray, level: int) -> None:
        c, corpus = self.cfg, self.corpus
        node = tax.nodes[node_id]
        node.general_terms = list(node.terms)  # a leaf keeps all of its terms
        if level >= c.max_depth or len(pool_ranked) < c.min_terms_to_split or len(docs) < c.min_docs_to_split:
            return
        pool = np.sort(pool_ranked)
        if c.split == "docs":
            groups, child_docs, n_noise = self._split_docs(pool, emb, docs, level)
        elif c.split == "terms":
            groups, child_docs, n_noise = self._split_terms(pool, emb, docs, level)
        else:
            raise ValueError(f"unknown split {c.split!r}")
        if len(groups) < c.min_children:
            return  # one (or no) dense group: the node is a leaf, never a copy of itself
        ranked = [rank_terms(corpus, g, child_docs[i], [d for j, d in enumerate(child_docs) if j != i])
                  for i, g in enumerate(groups)]

        if level == 0 and not c.score_root_children:
            decisions = [{"s_rel": None, "accept": True} for _ in groups]
        else:
            parent_vec = _mean_vector(emb, pool_ranked[:c.profile_k])
            decisions = self.scorer.score(
                node.terms, parent_vec,
                [[corpus.terms[i] for i in r] for r in ranked],
                [_mean_vector(emb, r[:c.profile_k]) for r in ranked])
        accepted = [i for i, d in enumerate(decisions) if d["accept"] and len(child_docs[i]) > 0]
        if len(accepted) < c.min_children:
            return
        placed = set(np.concatenate([groups[i] for i in accepted]).tolist())
        node.general_terms = [t for t in node.terms if corpus.term_index[t] not in placed]
        self.log(f"[{self.method}] {node_id} level={level} pool={len(pool)} clusters={len(groups)} "
                 f"noise={n_noise} accepted={len(accepted)}")

        order = sorted(accepted, key=lambda i: (-len(groups[i]), int(ranked[i][0])))
        for i in order:
            child = tax.add_child(node_id, [corpus.terms[j] for j in ranked[i]], int(len(child_docs[i])),
                                  {k: v for k, v in decisions[i].items() if k != "accept"})
            child_emb = self._local_w2v(emb, groups[i], child_docs[i]) if c.local == "w2v" else emb
            self._expand(tax, child.id, ranked[i], child_emb, child_docs[i], level + 1)

    def _split_terms(self, pool: np.ndarray, emb: np.ndarray, docs: np.ndarray, level: int):
        """Children = dense groups of term vectors; a child's documents contain any of its terms."""
        c, corpus = self.cfg, self.corpus
        x = emb[pool]
        if c.local == "recenter" and level > 0:
            x = recenter(x)
        labels = self.cluster(x)
        groups = [pool[labels == lab] for lab in sorted(set(labels.tolist()) - {-1})]
        groups = [g for g in groups if len(g) >= c.min_child_terms]
        child_docs = [np.intersect1d(corpus.docs_with_any(g), docs, assume_unique=True) for g in groups]
        return groups, child_docs, int((labels == -1).sum())

    def _split_docs(self, pool: np.ndarray, emb: np.ndarray, docs: np.ndarray, level: int):
        """Children = dense groups of the node's documents (a sample of them,
        embedded by the encoder's base model). Embedding single terms groups
        synonyms and spelling variants (cheap/cheapy/cheep), which never
        co-occur; document groups are what the reviews (titles) talk about.
        A term joins the group where its CaseOLAP score is highest when its
        scores are far enough from uniform (KL), else it stays at the parent;
        every document then goes to the group whose terms carry the most idf
        mass in it, as in TaxoGen."""
        c, corpus = self.cfg, self.corpus
        if not hasattr(self.encoder, "encode_docs"):
            raise ValueError("split: docs needs a context encoder (it embeds the node's documents)")
        rng = np.random.default_rng((c.seed, level, len(docs)))
        sample = np.sort(rng.choice(docs, min(len(docs), c.split_max_docs), replace=False))
        labels = self.cluster(self.encoder.encode_docs(sample))
        labs = sorted(set(labels.tolist()) - {-1})
        n_noise = int((labels == -1).sum())
        if len(labs) < c.min_children:
            return [], [], n_noise
        tf, df = cell_statistics(corpus, [sample[labels == lab] for lab in labs], pool)
        scores = caseolap_scores(tf, df)
        kl = kl_representativeness(scores)
        keep = (scores.max(axis=0) > 0) & (kl > c.split_min_kl)
        best = scores.argmax(axis=0)
        if c.split_min_df > 0 or c.split_min_df_frac > 0:
            # a term seen in a handful of documents cannot describe a subtopic;
            # the fraction keeps the bar proportional to the cluster's size
            sizes = np.array([(labels == lab).sum() for lab in labs])
            need = np.maximum(c.split_min_df, c.split_min_df_frac * sizes)
            keep &= df[best, np.arange(len(pool))] >= need[best]
        groups = [pool[(best == i) & keep] for i in range(len(labs))]
        groups = [g for g in groups if len(g) >= c.min_child_terms]
        if len(groups) < c.min_children:
            return [], [], n_noise
        anchors = groups                 # the anchors decide which documents go where
        terms = np.concatenate(anchors)
        if c.split_attach_min_sim is not None:
            # long-tail terms are too rare for document statistics; they join
            # the child whose anchors they are closest to in the encoder space.
            # A term seen often enough but spread evenly over the children
            # (low KL) describes the parent and stays there.
            general = (kl <= c.split_min_kl) & (df.sum(axis=0) >= max(c.split_min_df, 1))
            rest = np.setdiff1d(pool[~general], terms)
            rest = rest[np.linalg.norm(emb[rest], axis=1) > 0]
            if len(rest):
                cent = np.stack([_mean_vector(emb, g) for g in groups])
                sims = emb[rest] @ cent.T
                to, ok = sims.argmax(axis=1), sims.max(axis=1) >= c.split_attach_min_sim
                groups = [np.sort(np.concatenate([g, rest[ok & (to == i)]])) for i, g in enumerate(groups)]
        idf = np.log(1.0 + corpus.n_docs / np.maximum(corpus.df[terms], 1))
        weights = sp.csr_matrix((idf, (np.arange(len(terms)), np.repeat(np.arange(len(anchors)),
                                                                          [len(g) for g in anchors]))),
                                shape=(len(terms), len(anchors)))
        mass = (corpus.doc_term[docs][:, terms].astype(np.float64) @ weights).toarray()
        has = mass.max(axis=1) > 0
        best_doc = mass.argmax(axis=1)
        child_docs = [docs[has & (best_doc == i)] for i in range(len(groups))]
        return groups, child_docs, n_noise

    def _local_w2v(self, emb: np.ndarray, term_ids: np.ndarray, docs: np.ndarray) -> np.ndarray:
        c = self.cfg
        if len(docs) > c.max_local_docs:
            docs = np.sort(np.random.default_rng(c.seed).choice(docs, c.max_local_docs, replace=False))
        local = Word2VecEncoder.train(self.corpus.get_docs(docs), seed=c.seed,
                                      **{"min_count": 3, **(c.w2v or {})})
        out = emb.copy()
        terms = [self.corpus.terms[i] for i in term_ids]
        vecs = local.encode_terms(terms)
        known = np.linalg.norm(vecs, axis=1) > 0
        if out.shape[1] == vecs.shape[1]:
            out[term_ids[known]] = vecs[known]
        return out


# ---------------------------------------------------------------- TaxoGen baseline

@dataclass
class TaxoGenConfig:
    n_cluster: int = 5
    n_iter: int = 5
    filter_thre: float = 0.25
    n_expand: int = 100
    max_depth: int = 3
    min_docs_to_split: int = 50
    max_local_docs: int = 15000
    w2v: dict | None = None
    seed: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "TaxoGenConfig":
        return _from_dict(cls, d)


class TaxoGen:
    """Adaptive spherical clustering: cluster, hard-assign documents, keep the
    terms whose CaseOLAP score distribution is far from uniform (KL >
    filter_thre), re-cluster the kept terms; filtered-out terms stay at the
    parent as general terms. Children are re-embedded with word2vec trained on
    the documents of the n_expand terms closest to the cluster centre.
    Differences from the original code: scikit-learn k-means on unit vectors
    replaces the unmaintained `spherecluster`; gensim replaces the C word2vec
    binary; local training documents are capped at max_local_docs."""

    def __init__(self, cfg: TaxoGenConfig, corpus: Corpus, encoder: Word2VecEncoder,
                 log: Callable[[str], None] = print):
        self.cfg, self.corpus, self.encoder, self.log = cfg, corpus, encoder, log
        self._idf_all = np.log(1.0 + corpus.n_docs / np.maximum(corpus.df, 1))

    def build(self) -> Taxonomy:
        corpus = self.corpus
        emb = self.encoder.encode_terms(corpus.terms)
        pool = np.where(np.linalg.norm(emb, axis=1) > 0)[0]
        tax = Taxonomy("taxogen", {**asdict(self.cfg), "encoder": self.encoder.name})
        all_docs = np.arange(corpus.n_docs)
        tax.root.terms = [corpus.terms[i] for i in rank_terms(corpus, pool, all_docs)]
        tax.root.n_docs = corpus.n_docs
        self._recur(tax, ROOT, pool, emb, all_docs, level=0)
        problems = _invalid_issues(tax)
        if problems:
            raise RuntimeError(f"invalid taxonomy: {problems[:5]}")
        self.log(f"[taxogen] done: {tax.stats()}")
        return tax

    def _assign_docs(self, docs: np.ndarray, seeds: np.ndarray, labels: np.ndarray, k: int):
        """Each document goes to the cluster whose terms carry the most idf mass in it."""
        sub = self.corpus.doc_term[docs][:, seeds].astype(np.float64)
        onehot = np.zeros((len(seeds), k))
        onehot[np.arange(len(seeds)), labels] = self._idf_all[seeds]
        mass = sub @ onehot
        best = np.asarray(mass.argmax(axis=1)).ravel()
        has = np.asarray(mass.max(axis=1)).ravel() > 0
        return [docs[has & (best == c)] for c in range(k)]

    def _recur(self, tax: Taxonomy, node_id: str, pool: np.ndarray, emb: np.ndarray,
               docs: np.ndarray, level: int) -> None:
        c = self.cfg
        node = tax.nodes[node_id]
        node.general_terms = list(node.terms)
        min_terms = c.n_cluster * 3
        if level >= c.max_depth or len(docs) < c.min_docs_to_split or len(pool) < min_terms:
            return
        seeds = np.sort(pool)
        result = None
        for it in range(c.n_iter):
            if len(seeds) < min_terms:
                break
            labels = spherical_kmeans(emb[seeds], c.n_cluster, seed=c.seed)
            k = int(labels.max()) + 1
            cells = self._assign_docs(docs, seeds, labels, k)
            if sum(len(d) > 0 for d in cells) < 2:
                break
            tf, df = cell_statistics(self.corpus, cells, seeds)
            scores = caseolap_scores(tf, df)
            result = (seeds, labels, cells, scores)
            kept = seeds[kl_representativeness(scores) > c.filter_thre]
            self.log(f"[taxogen] {node_id} iter={it} seeds={len(seeds)} kept={len(kept)}")
            if len(kept) < min_terms:
                break
            seeds = np.sort(kept)
        if result is None:
            return
        seeds, labels, cells, scores = result
        children = []
        for cid in range(int(labels.max()) + 1):
            members = np.where(labels == cid)[0]
            if len(members) == 0 or len(cells[cid]) == 0:
                continue
            order = members[np.lexsort((seeds[members], -scores[cid, members]))]
            children.append((seeds[order], cells[cid]))
        if len(children) < 2:
            return
        placed = set(np.concatenate([t for t, _ in children]).tolist())
        node.general_terms = [t for t in node.terms if self.corpus.term_index[t] not in placed]
        for term_ids, cell_docs in children:
            child = tax.add_child(node_id, [self.corpus.terms[i] for i in term_ids], int(len(cell_docs)))
            child_emb = emb
            if level + 1 < c.max_depth and len(cell_docs) >= c.min_docs_to_split:
                child_emb = self._local_embedding(term_ids, pool, emb)
                valid = term_ids[np.linalg.norm(child_emb[term_ids], axis=1) > 0]
            else:
                valid = term_ids
            self._recur(tax, child.id, valid, child_emb, cell_docs, level + 1)

    def _local_embedding(self, term_ids: np.ndarray, pool: np.ndarray, emb: np.ndarray) -> np.ndarray:
        c, corpus = self.cfg, self.corpus
        center = _mean_vector(emb, term_ids)
        center_term = term_ids[np.argmax(emb[term_ids] @ center)]
        sims = emb[pool] @ emb[center_term]
        expanded = pool[np.argsort(-sims, kind="stable")[:c.n_expand]]
        docs = corpus.docs_with_any(expanded)
        if len(docs) > c.max_local_docs:
            rng = random.Random(c.seed)
            docs = np.array(sorted(rng.sample(docs.tolist(), c.max_local_docs)))
        local = Word2VecEncoder.train(corpus.get_docs(docs), seed=c.seed,
                                      **{"min_count": 3, **(c.w2v or {})})
        # the local space may have its own dimension (the original code trains 50-d vectors)
        out = np.zeros((len(emb), local.kv.vector_size), dtype=emb.dtype)
        out[term_ids] = local.encode_terms([corpus.terms[i] for i in term_ids])
        return out


# ---------------------------------------------------------------- BERTopic (optional)

def build_bertopic(corpus: Corpus, encoder: Encoder, top_n: int = 30, seed: int = 0,
                   max_docs: int | None = 100000, **kwargs) -> Taxonomy:
    """Flat BERTopic topics as a depth-1 taxonomy. Documents are embedded with
    the same encoder as the other methods, and topic words are counted over
    the shared candidate vocabulary (phrases included), so every method
    chooses among the same terms. At most max_docs documents (a seeded
    sample) are clustered, which keeps UMAP within a 16 GB machine."""
    try:
        from bertopic import BERTopic
        from sklearn.feature_extraction.text import CountVectorizer
        from umap import UMAP
    except Exception as e:  # missing package, or an installed one that fails to import
        raise ImportError(f"BERTopic baseline unavailable ({type(e).__name__}: {e}); "
                          "install with `pip install bertopic -c requirements.txt`") from e
    rows = np.arange(corpus.n_docs)
    if max_docs and corpus.n_docs > max_docs:
        rows = np.sort(np.random.default_rng(seed).choice(rows, max_docs, replace=False))
    docs = corpus.get_docs(rows)
    embeddings = encoder.encode([" ".join(t.replace("_", " ") for t in d) for d in docs])
    # only terms that occur in the sample: a zero document frequency makes
    # BERTopic's c-TF-IDF divide by zero
    present = np.asarray(corpus.doc_term[rows].sum(axis=0)).ravel() > 0
    vocabulary = [t for t, ok in zip(corpus.terms, present) if ok]
    vectorizer = CountVectorizer(vocabulary=vocabulary, tokenizer=str.split, lowercase=False,
                                 token_pattern=None)
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=seed)
    # language="english" makes BERTopic strip every non-alphanumeric character
    # (so 'long_sleeve' becomes 'longsleeve' and matches no vocabulary term);
    # the language only picks a default embedding model, unused here since
    # embeddings are passed in.
    model = BERTopic(language=kwargs.pop("language", "multilingual"), umap_model=umap_model,
                     vectorizer_model=vectorizer, top_n_words=top_n, **kwargs)
    model.fit([" ".join(d) for d in docs], embeddings=embeddings)
    tax = Taxonomy("bertopic", {"top_n": top_n, "seed": seed, "max_docs": max_docs,
                                "encoder": encoder.name, **kwargs})
    tax.root.terms = list(corpus.terms)
    tax.root.n_docs = corpus.n_docs
    placed = set()
    for topic_id in sorted(t for t in model.get_topics() if t != -1):
        words = [w.replace(" ", "_") for w, _ in model.get_topic(topic_id)]
        terms = [w for w in words if w in corpus.term_index and w not in placed]
        if terms:
            placed.update(terms)
            tax.add_child(ROOT, terms, int(len(corpus.docs_with_any(corpus.ids(terms)))))
    tax.root.general_terms = [t for t in corpus.terms if t not in placed]
    return tax


def rerank(tax: Taxonomy, corpus: Corpus) -> tuple[Taxonomy, dict]:
    """Re-rank every node's terms with the shared CaseOLAP ranker so the terms
    shown to annotators are chosen the same way for every method. Terms outside
    the corpus vocabulary are dropped and counted."""
    out = Taxonomy.from_dict(tax.to_dict())
    dropped = 0
    docs_cache: dict[str, np.ndarray] = {}

    def node_docs(node_id):
        if node_id not in docs_cache:
            docs_cache[node_id] = corpus.docs_with_any(corpus.ids(out.nodes[node_id].terms))
        return docs_cache[node_id]

    for node in list(out.non_root()):
        ids = corpus.ids(node.terms)
        dropped += len(node.terms) - len(ids)
        sib_docs = [node_docs(s.id) for s in out.siblings(node.id)]
        node.terms = [corpus.terms[i] for i in rank_terms(corpus, ids, node_docs(node.id), sib_docs)]
    return out, {"terms_dropped": dropped}
