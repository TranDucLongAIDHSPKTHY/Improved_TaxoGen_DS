"""Interchangeable building blocks for taxonomy construction:

- encoders   (term -> unit vector): SBERT, Word2Vec, char n-gram, fixed lookup
- clusterers (vectors -> labels):   HDBSCAN, spherical k-means
- ranking    (CaseOLAP popularity x distinctiveness). Embedding-agnostic, so
             it can rank the terms of ANY method's nodes the same way.
- relation scoring (Eq. 1): S_rel = w * cos + (1 - w) * BM25_hat
"""
from __future__ import annotations

import hashlib
import math
import re
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from taxorec.common import ensure_dir
from taxorec.data import Corpus


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def term_to_text(term: str) -> str:
    return term.replace("_", " ")


# ---------------------------------------------------------------- encoders

class Encoder:
    name = "encoder"

    def encode(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def encode_terms(self, terms: list[str]) -> np.ndarray:
        return self.encode([term_to_text(t) for t in terms])


class SBERTEncoder(Encoder):
    """Sentence-BERT; `revision` pins the exact model commit on the HuggingFace
    hub so a later update of the model cannot silently change the results."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 revision: str | None = None, batch_size: int = 256, device: str | None = None,
                 cache_dir=None):
        self.model_name, self.revision = model_name, revision
        self.batch_size, self.device = batch_size, device or "cpu"
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.name = f"sbert:{model_name}@{revision or 'latest'}"
        self._model = None

    def encode(self, texts: list[str], cache: bool = True) -> np.ndarray:
        """`cache=False` for large one-off lists (document embeddings have their own cache)."""
        cache_on, cache = cache, None
        if self.cache_dir and cache_on:
            key = hashlib.sha256((self.name + "\n" + "\n".join(texts)).encode("utf-8")).hexdigest()[:20]
            cache = ensure_dir(self.cache_dir) / f"sbert-{key}.npy"
            if cache.exists():
                return np.load(cache)
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device, revision=self.revision)
        vecs = self._model.encode(texts, batch_size=self.batch_size, show_progress_bar=len(texts) > 5000,
                                  normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        if cache is not None:
            np.save(cache, vecs)
        return vecs


def stable_hash(text: str) -> int:
    """Process-independent hash for gensim (Python's str hash is salted per process)."""
    return zlib.crc32(text.encode("utf-8"))


class Word2VecEncoder(Encoder):
    """Phrase vectors from word2vec trained on the phrase-merged corpus; terms
    missing from the vocabulary get a zero vector (callers treat zero rows
    as 'no representation')."""

    def __init__(self, keyed_vectors):
        self.kv = keyed_vectors
        self.name = "word2vec"

    @classmethod
    def train(cls, docs: Iterable[list[str]], vector_size: int = 100, window: int = 5,
              negative: int = 10, min_count: int = 10, sg: int = 1, epochs: int = 5,
              sample: float = 1e-3, workers: int = 1, seed: int = 0) -> "Word2VecEncoder":
        from gensim.models import Word2Vec
        model = Word2Vec(sentences=docs, vector_size=vector_size, window=window, negative=negative,
                         min_count=min_count, sg=sg, epochs=epochs, sample=sample, workers=workers,
                         seed=seed, hashfxn=stable_hash)
        return cls(model.wv)

    @classmethod
    def from_text(cls, path) -> "Word2VecEncoder":
        from gensim.models import KeyedVectors
        return cls(KeyedVectors.load_word2vec_format(str(path), binary=False))

    def has(self, term: str) -> bool:
        return term in self.kv.key_to_index

    def encode_terms(self, terms: list[str]) -> np.ndarray:
        out = np.zeros((len(terms), self.kv.vector_size), dtype=np.float32)
        for i, t in enumerate(terms):
            if t in self.kv.key_to_index:
                out[i] = self.kv[t]
        return l2_normalize(out)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.encode_terms([t.replace(" ", "_") for t in texts])


class CharNgramEncoder(Encoder):
    """Hashed character n-grams with a fixed random projection: a cheap
    surface-form encoder (no semantics), useful as an ablation and in tests."""

    def __init__(self, dim: int = 128, ngram_range=(3, 4), seed: int = 0):
        from sklearn.feature_extraction.text import HashingVectorizer
        from sklearn.random_projection import SparseRandomProjection
        import scipy.sparse as sp
        n_features = 2 ** 16
        self.vectorizer = HashingVectorizer(analyzer="char_wb", ngram_range=tuple(ngram_range),
                                            n_features=n_features, alternate_sign=False, norm="l2")
        self.projection = SparseRandomProjection(n_components=dim, dense_output=True, random_state=seed)
        self.projection.fit(sp.csr_matrix((1, n_features)))
        self.name = f"charngram:{dim}"

    def encode(self, texts: list[str]) -> np.ndarray:
        return l2_normalize(self.projection.transform(self.vectorizer.transform(texts)))


class KeyedEncoder(Encoder):
    """Fixed term -> vector lookup (precomputed vectors, tests)."""

    def __init__(self, mapping: dict[str, np.ndarray], name: str = "keyed"):
        self.mapping = mapping
        self.dim = len(next(iter(mapping.values())))
        self.name = name

    def encode_terms(self, terms: list[str]) -> np.ndarray:
        zero = np.zeros(self.dim, dtype=np.float32)
        return l2_normalize(np.stack([self.mapping.get(t, zero) for t in terms]))

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.encode_terms([t.replace(" ", "_") for t in texts])


class ContextEncoder(Encoder):
    """Context-grounded term vectors: the mean `base` (e.g. SBERT) embedding of
    up to `docs_per_term` documents containing the term, sampled with `seed`.
    Embedding a term on its own groups words of the same kind (praise,
    colours, prices); averaging over the documents it occurs in groups terms
    by the topics they are used in. Document embeddings are cached by
    document-store id, so dev/held-out subsets and seeds share them."""

    def __init__(self, base: Encoder, corpus: Corpus, docs_per_term: int = 30, seed: int = 0,
                 cache_dir=None, drop_df: float | None = None):
        """drop_df: words in more than this fraction of the documents (great,
        good, love, ...) are left out of the text a document is embedded from,
        so documents group by what they are about rather than by sentiment."""
        self.base, self.corpus, self.docs_per_term, self.seed = base, corpus, docs_per_term, seed
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.drop_df = drop_df
        self.name = f"ctx({base.name}):m{docs_per_term}" + (f":drop{drop_df}" if drop_df else "")
        self._dropped: set[str] | None = None

    def _doc_text(self, tokens: list[str]) -> str:
        if self.drop_df and self._dropped is None:
            self._dropped = {t for t, df in zip(self.corpus.terms, self.corpus.df)
                             if df > self.drop_df * self.corpus.n_docs}
        words = [t for t in tokens if t not in self._dropped] if self.drop_df else tokens
        return " ".join(words).replace("_", " ")

    def _doc_embeddings(self, store_ids: np.ndarray) -> np.ndarray:
        """Embeddings of sorted unique document-store ids, computed once and cached."""
        ids, vecs = np.zeros(0, dtype=np.int64), None
        if self.cache_dir:
            # the store's line offsets identify its content, so a rebuilt corpus
            # (where an id means another document) never reads a stale cache
            h = hashlib.sha256(self.base.name.encode("utf-8"))
            h.update(self.corpus.docstore().offsets.tobytes())
            if self.drop_df:   # the embedded text differs; the dropped words depend on the corpus df
                h.update(f"drop{self.drop_df}".encode("utf-8"))
                h.update(np.ascontiguousarray(self.corpus.df).tobytes())
            tag = h.hexdigest()[:16]
            ids_path = self.cache_dir / f"docemb-{tag}-ids.npy"
            vecs_path = self.cache_dir / f"docemb-{tag}-vecs.npy"
            if ids_path.exists() and vecs_path.exists():
                ids, vecs = np.load(ids_path), np.load(vecs_path)
        missing = np.setdiff1d(store_ids, ids)
        if len(missing):
            texts = [self._doc_text(d) for d in self.corpus.docstore().get(missing)]
            new = (self.base.encode(texts, cache=False) if isinstance(self.base, SBERTEncoder)
                   else self.base.encode(texts)).astype(np.float16)
            ids = np.concatenate([ids, missing])
            vecs = new if vecs is None else np.concatenate([vecs, new])
            order = np.argsort(ids, kind="stable")
            ids, vecs = ids[order], vecs[order]
            if self.cache_dir:
                ensure_dir(self.cache_dir)
                np.save(vecs_path, vecs)
                np.save(ids_path, ids)
        return vecs[np.searchsorted(ids, store_ids)].astype(np.float32)

    def encode_docs(self, rows) -> np.ndarray:
        """`base` embeddings of rows of this corpus (or subset), through the cache."""
        rows = np.asarray(rows, dtype=np.int64)
        store = self.corpus.doc_map[rows] if self.corpus.doc_map is not None else rows
        uniq, inverse = np.unique(store, return_inverse=True)
        return self._doc_embeddings(uniq)[inverse]

    def encode_terms(self, terms: list[str]) -> np.ndarray:
        c, td = self.corpus, self.corpus.term_doc
        rng = np.random.default_rng(self.seed)
        picks = []
        for t in terms:
            j = c.term_index.get(t)
            rows = td.indices[td.indptr[j]:td.indptr[j + 1]] if j is not None else np.zeros(0, dtype=np.int64)
            if len(rows) > self.docs_per_term:
                rows = rng.choice(rows, self.docs_per_term, replace=False)
            rows = np.sort(rows).astype(np.int64)
            picks.append(c.doc_map[rows] if c.doc_map is not None else rows)
        uniq = np.unique(np.concatenate(picks)) if picks else np.zeros(0, dtype=np.int64)
        if not len(uniq):
            return np.zeros((len(terms), 1), dtype=np.float32)
        emb = self._doc_embeddings(uniq)
        out = np.zeros((len(terms), emb.shape[1]), dtype=np.float32)
        for i, p in enumerate(picks):
            if len(p):
                out[i] = emb[np.searchsorted(uniq, p)].mean(axis=0)
        return l2_normalize(out)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.encode_terms([t.replace(" ", "_") for t in texts])


def make_encoder(cfg: dict, corpus: Corpus | None = None, seed: int = 0) -> Encoder:
    cfg = dict(cfg)
    kind = cfg.pop("type")
    if kind == "sbert":
        return SBERTEncoder(**cfg)
    if kind == "charngram":
        return CharNgramEncoder(**cfg)
    if kind == "word2vec":
        pretrained = cfg.pop("pretrained", None)
        if pretrained:
            return Word2VecEncoder.from_text(pretrained)
        if corpus is None:
            raise ValueError("training word2vec needs a corpus with a document store")
        return Word2VecEncoder.train(corpus.documents(), **{"seed": seed, **cfg})
    raise ValueError(f"unknown encoder type {kind!r}")


# ---------------------------------------------------------------- clustering

def adaptive_min_cluster_size(n: int, divisor: float, floor: int, ceiling: int) -> int:
    """HDBSCAN's min_cluster_size scaled with the pool, so the root yields broad
    groups rather than hundreds of micro-clusters."""
    return int(min(ceiling, max(floor, n / divisor)))


def spherical_kmeans(x: np.ndarray, k: int, seed: int = 0, n_init: int = 10,
                     normalize: bool = True) -> np.ndarray:
    """k-means on unit vectors (squared Euclidean is monotone in cosine there);
    normalize=False clusters the points as given (e.g. UMAP coordinates)."""
    from sklearn.cluster import KMeans
    k = min(k, len(x))
    return KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit_predict(
        l2_normalize(x) if normalize else x)


def hdbscan_labels(x: np.ndarray, min_cluster_size: int, min_samples: int | None = None,
                   selection: str = "leaf", normalize: bool = True) -> np.ndarray:
    """Density clustering; -1 marks noise. On unit vectors Euclidean distance
    is monotone in cosine distance; normalize=False for UMAP coordinates."""
    from sklearn.cluster import HDBSCAN
    if len(x) < 2 * min_cluster_size:
        return np.full(len(x), -1)
    return HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                   cluster_selection_method=selection, metric="euclidean",
                   copy=True).fit_predict(l2_normalize(x) if normalize else x)


def umap_reduce(x: np.ndarray, dim: int = 5, n_neighbors: int = 15, seed: int = 0) -> np.ndarray:
    """UMAP (cosine) to a few dimensions before clustering. In the raw 384-d
    SBERT space distances concentrate and HDBSCAN marks every DBLP term as
    noise; BERTopic reduces with the same settings. A fixed random_state makes
    UMAP single-threaded and deterministic."""
    if len(x) <= dim + 2:
        return x
    from umap import UMAP
    return UMAP(n_neighbors=min(n_neighbors, len(x) - 1), n_components=dim, min_dist=0.0,
                metric="cosine", random_state=seed).fit_transform(x).astype(np.float32)


def recenter(x: np.ndarray) -> np.ndarray:
    """Local re-contextualization: remove the node's shared direction so that
    clustering at depth > 0 separates subtopics instead of re-finding the
    parent topic (the role local embeddings play in TaxoGen)."""
    return l2_normalize(x - x.mean(axis=0, keepdims=True))


# ---------------------------------------------------------------- CaseOLAP ranking

def cell_statistics(corpus: Corpus, cells: list[np.ndarray], term_ids) -> tuple[np.ndarray, np.ndarray]:
    term_ids = np.asarray(term_ids, dtype=np.int64)
    tf = np.zeros((len(cells), len(term_ids)))
    df = np.zeros((len(cells), len(term_ids)))
    for c, docs in enumerate(cells):
        if len(docs) == 0:
            continue
        sub = corpus.doc_term[np.asarray(docs, dtype=np.int64)][:, term_ids]
        tf[c] = np.asarray(sub.sum(axis=0)).ravel()
        df[c] = np.diff(sub.tocsc().indptr)
    return tf, df


def caseolap_scores(tf: np.ndarray, df: np.ndarray, k1: float = 1.2, b: float = 0.5,
                    multiplier: float = 3.0) -> np.ndarray:
    """popularity * distinctiveness per (cell, term), after TaxoGen's
    caseslim.py (NOINT variant, the one its pipeline actually uses)."""
    present = tf > 0
    cell_len = tf.sum(axis=1, keepdims=True)
    avgdl = cell_len.mean() or 1.0
    norm_tf = np.where(present, tf * (k1 + 1) / (tf + k1 * (1 - b + b * cell_len / avgdl) + 1e-12), 0.0)
    max_df = df.max(axis=1, keepdims=True)
    df_factor = np.where(max_df > 0, np.log2(1 + df) / np.log2(1 + np.maximum(max_df, 1)), 0.0)
    s = norm_tf * df_factor * multiplier
    e = np.where(present, np.exp(s), 0.0)
    distinct = e / (1.0 + e.sum(axis=0, keepdims=True))
    return np.where(present, distinct * np.log2(1 + df), 0.0)


def kl_representativeness(scores: np.ndarray) -> np.ndarray:
    """KL divergence of each term's score distribution over cells from uniform."""
    n_cells = scores.shape[0]
    totals = scores.sum(axis=0)
    p = np.divide(scores, totals, out=np.zeros_like(scores), where=totals > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl = np.where(p > 0, p * np.log(p * n_cells), 0.0).sum(axis=0)
    return np.where(totals > 0, kl, 0.0)


def rank_terms(corpus: Corpus, term_ids, node_docs, sibling_docs: Iterable = ()) -> np.ndarray:
    """Rank a node's terms by CaseOLAP score against its siblings; ties by id."""
    term_ids = np.asarray(term_ids, dtype=np.int64)
    if len(term_ids) == 0:
        return term_ids
    cells = [np.asarray(node_docs)] + [np.asarray(s) for s in sibling_docs]
    tf, df = cell_statistics(corpus, cells, term_ids)
    score = caseolap_scores(tf, df)[0]
    return term_ids[np.lexsort((term_ids, -score))]


# ---------------------------------------------------------------- relation scoring

_SPLIT = re.compile(r"[_\s]+")


def profile_tokens(terms: list[str], k: int) -> list[str]:
    return [w for t in terms[:k] for w in _SPLIT.split(t) if w]


def bm25_scores(query: list[str], docs: list[list[str]], k1: float = 1.2, b: float = 0.75) -> np.ndarray:
    n = len(docs)
    df = Counter(w for d in docs for w in set(d))
    avgdl = (sum(len(d) for d in docs) / n) if n else 0.0
    out = np.zeros(n)
    for i, d in enumerate(docs):
        tf = Counter(d)
        dl = len(d)
        for w in set(query):
            if tf[w] == 0:
                continue
            idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
            out[i] += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * dl / (avgdl or 1)))
    return out


@dataclass
class RelationScorer:
    """Eq. (1). BM25_hat(p, c) = BM25(q_p, d_c) / BM25(q_p, d_p), clipped to
    [0, 1]: lexical evidence relative to the parent's own profile, so the
    threshold tau has an absolute meaning (sibling-wise min-max scaling would
    always give one child 0 and another 1). IDF comes from the parent and
    its candidate children. mode: none | cos | bm25 | bm25cos."""
    mode: str = "bm25cos"
    w: float = 0.5
    tau: float = 0.3
    profile_k: int = 10

    def score(self, parent_terms: list[str], parent_vec: np.ndarray,
              children_terms: list[list[str]], children_vecs: list[np.ndarray]) -> list[dict]:
        if self.mode == "none":
            return [{"s_rel": None, "accept": True} for _ in children_terms]
        weight = {"cos": 1.0, "bm25": 0.0, "bm25cos": self.w}.get(self.mode)
        if weight is None:
            raise ValueError(f"unknown relation mode {self.mode!r}")
        query = profile_tokens(parent_terms, self.profile_k)
        docs = [query] + [profile_tokens(c, self.profile_k) for c in children_terms]
        bm = bm25_scores(query, docs)
        bm_hat = np.clip(bm[1:] / bm[0], 0, 1) if bm[0] > 0 else np.zeros(len(children_terms))
        out = []
        pn = np.linalg.norm(parent_vec)
        for cv, bh in zip(children_vecs, bm_hat):
            cn = np.linalg.norm(cv)
            cos = float(max(0.0, np.dot(parent_vec, cv) / (pn * cn))) if pn > 0 and cn > 0 else 0.0
            s = weight * cos + (1 - weight) * float(bh)
            out.append({"cos": cos, "bm25_hat": float(bh), "s_rel": s, "accept": s >= self.tau})
        return out
