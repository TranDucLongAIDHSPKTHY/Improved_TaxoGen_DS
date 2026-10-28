"""Datasets: download, tokenization, phrase mining, the candidate-term Corpus,
and loaders for Amazon Reviews 2023 and the TaxoGen DBLP release.

Item text for recommendation is built from product METADATA only. Reviews
are never folded into item representations, because a review written by a
test user about a held-out item would otherwise leak into the item vector.
"""
from __future__ import annotations

import html
import re
import shutil
import urllib.request
from array import array
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np
import pandas as pd
import scipy.sparse as sp

from taxorec.common import ensure_dir, iter_jsonl, open_text, read_json, write_json

AMAZON_2023_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw"

# The DBLP release of TaxoGen (Zhang et al., KDD 2018), linked from the readme
# of github.com/franticnerd/taxogen. The hashes are those of the copy used to
# develop this code, so a download that differs is rejected.
TAXOGEN_DBLP = {
    "source": "https://drive.google.com/file/d/1GbxKrxrmFrKt5vgDHP1xe1Qr_rfvR1jh",
    "url": ("https://drive.usercontent.google.com/download?id=1GbxKrxrmFrKt5vgDHP1xe1Qr_rfvR1jh"
            "&export=download&confirm=t"),
    "zip_name": "dblp_taxogen_original.zip",
    "zip_sha256": "5ffc7f5c62a37dff51a13456326b08656dc357c32160d84b0d71168c5bf4cae0",
    "files": {
        "input/papers.txt": "79d35f5722012be1cef0e7b6a41937f200ddf6facc0292ef0476575517664726",
        "input/keywords.txt": "91e7c9256dc4f3b8300f746a5fe1ab6fb227b7de82d81ed84fcfdf1ab6ca82c8",
        "input/embeddings.txt": "33fbe839be500bbf965019ac25d5a59e4f05d6c00c2b9b28272e08b2b55a9a9d",
    },
}


# ---------------------------------------------------------------- download

def sha256_of(path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest) -> Path:
    """Download to `dest` through a .part file, so an interrupted download
    never leaves a truncated file under the final name."""
    dest = Path(dest)
    ensure_dir(dest.parent)
    print(f"[download] {url}")
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "taxorec"})
    with urllib.request.urlopen(req) as resp, open(tmp, "wb") as f:
        shutil.copyfileobj(resp, f, length=1 << 20)
    tmp.replace(dest)
    return dest


def download_amazon(category: str, out_dir) -> dict[str, Path]:
    """Download the review and metadata files of one Amazon Reviews 2023 category."""
    out_dir = ensure_dir(out_dir)
    targets = {
        "reviews": (f"{AMAZON_2023_URL}/review_categories/{category}.jsonl.gz",
                    out_dir / f"{category}.jsonl.gz"),
        "meta": (f"{AMAZON_2023_URL}/meta_categories/meta_{category}.jsonl.gz",
                 out_dir / f"meta_{category}.jsonl.gz"),
    }
    for url, dest in targets.values():
        if dest.exists():
            print(f"[download] exists, skipping: {dest}")
            continue
        fetch(url, dest)
    return {k: dest for k, (_, dest) in targets.items()}


def download_dblp(input_dir, spec: dict = TAXOGEN_DBLP) -> dict[str, Path]:
    """Download the TaxoGen DBLP release, check its sha256, and extract the
    three files the pipeline uses (papers, keywords, global embeddings) into
    `input_dir`. Files already present with the right hash are kept."""
    import zipfile
    input_dir = ensure_dir(input_dir)
    targets = {member: input_dir / Path(member).name for member in spec["files"]}
    if all(p.exists() and sha256_of(p) == spec["files"][m] for m, p in targets.items()):
        print(f"[download] DBLP files present and verified in {input_dir}")
        return {Path(m).stem: p for m, p in targets.items()}
    zip_path = input_dir.parent / spec["zip_name"]
    if not zip_path.exists():
        fetch(spec["url"], zip_path)
    got = sha256_of(zip_path)
    if got != spec["zip_sha256"]:
        bad = zip_path.with_name(zip_path.name + ".bad")
        zip_path.replace(bad)
        raise ValueError(f"{zip_path.name}: sha256 {got} does not match the expected "
                         f"{spec['zip_sha256']} (moved to {bad.name}); source: {spec['source']}")
    with zipfile.ZipFile(zip_path) as z:
        for member, dest in targets.items():
            print(f"[download] extracting {member}")
            tmp = dest.with_name(dest.name + ".part")
            with z.open(member) as src, open(tmp, "wb") as f:
                shutil.copyfileobj(src, f, length=1 << 20)
            if sha256_of(tmp) != spec["files"][member]:
                raise ValueError(f"{member}: sha256 mismatch after extraction")
            tmp.replace(dest)
    return {Path(m).stem: p for m, p in targets.items()}


# ---------------------------------------------------------------- text

_TOKEN_RE = re.compile(r"[a-z][a-z0-9]+")
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def clean_text(text) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    if isinstance(text, (list, tuple)):
        text = " ".join(clean_text(t) for t in text)
    text = html.unescape(_TAG_RE.sub(" ", str(text)))
    return _WS_RE.sub(" ", text).strip()


def mostly_ascii(text: str, threshold: float = 0.9) -> bool:
    letters = [c for c in text if c.isalpha()]
    return not letters or sum(c.isascii() for c in letters) / len(letters) >= threshold


_EN_WORDS = frozenset("i the and is it to of for this that was with my in on but not they are you have "
                      "very so be as at if all one these were would".split())
_ES_WORDS = frozenset("el la los las que es muy para pero con por una un se lo mi y del al como "
                      "esta este bien talla".split())


def looks_english(text: str, min_ratio: float = 0.08) -> bool:
    """Function-word test: unaccented Spanish passes mostly_ascii, so Spanish
    reviews are caught here (enough English function words, more English
    than Spanish ones)."""
    words = re.findall(r"[a-z]+", text.lower())
    if not words:
        return False
    en = sum(w in _EN_WORDS for w in words)
    es = sum(w in _ES_WORDS for w in words)
    return en / len(words) >= min_ratio and en > es


class SimpleTokenizer:
    """Lowercase regex tokenizer with English stop words; dependency-free."""

    def __init__(self, min_len: int = 2):
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        self.stop = set(ENGLISH_STOP_WORDS)
        self.min_len = min_len

    def __call__(self, text: str) -> list[str]:
        return [t for t in _TOKEN_RE.findall(text.lower())
                if len(t) >= self.min_len and t not in self.stop]

    def tokenize_many(self, texts: Iterable[str]) -> Iterator[list[str]]:
        for text in texts:
            yield self(text)


class SpacyTokenizer:
    """Lemmatizing tokenizer. Keeps nouns, proper nouns and adjectives (the
    adjectives survive so modifiers such as 'leather' in 'leather jacket'
    can form phrases)."""

    def __init__(self, model: str = "en_core_web_sm", keep_pos=("NOUN", "PROPN", "ADJ"),
                 n_process: int = 1, batch_size: int = 256):
        self.model, self.keep_pos = model, set(keep_pos or ())
        self.n_process, self.batch_size = n_process, batch_size
        self._nlp = None

    @property
    def nlp(self):
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load(self.model, disable=["parser", "ner"])
        return self._nlp

    def tokenize_many(self, texts: Iterable[str]) -> Iterator[list[str]]:
        for doc in self.nlp.pipe(texts, batch_size=self.batch_size, n_process=self.n_process):
            yield [t.lemma_.lower() for t in doc
                   if t.is_alpha and not t.is_stop and len(t.lemma_) > 1
                   and (not self.keep_pos or t.pos_ in self.keep_pos)]

    def __call__(self, text: str) -> list[str]:
        return next(iter(self.tokenize_many([text])))


def make_tokenizer(cfg: dict | None):
    cfg = dict(cfg or {})
    kind = cfg.pop("type", "simple")
    if kind == "simple":
        return SimpleTokenizer(**cfg)
    if kind == "spacy":
        return SpacyTokenizer(**cfg)
    raise ValueError(f"unknown tokenizer type {kind!r}")


def mine_phrases(docs_factory: Callable[[], Iterable[list[str]]], min_count: int = 20,
                 threshold: float = 10.0, passes: int = 1) -> set[str]:
    """Collocations found by gensim Phrases, streaming over `docs_factory()`
    (one pass per round); `passes=2` also finds trigrams."""
    from gensim.models.phrases import Phrases

    phrases: set[str] = set()
    for _ in range(passes):
        merger = PhraseMerger(phrases)
        model = Phrases((merger(d) for d in docs_factory()), min_count=min_count,
                        threshold=threshold, delimiter="_")
        phrases |= set(model.export_phrases().keys())
    return phrases


class PhraseMerger:
    """Greedy longest-match merging of known phrases ('long sleeve dress' ->
    'long_sleeve_dress'). Deterministic, so the same merging is applied to
    the taxonomy corpus and later to product text."""

    def __init__(self, phrases: Iterable[str]):
        self.table: set[tuple[str, ...]] = set()
        self.max_n = 1
        for p in phrases:
            parts = tuple(x for x in p.split("_") if x)
            if len(parts) > 1:
                self.table.add(parts)
                self.max_n = max(self.max_n, len(parts))

    def __call__(self, tokens: list[str]) -> list[str]:
        out, i = [], 0
        while i < len(tokens):
            for n in range(min(self.max_n, len(tokens) - i), 1, -1):
                if tuple(tokens[i:i + n]) in self.table:
                    out.append("_".join(tokens[i:i + n]))
                    i += n
                    break
            else:
                out.append(tokens[i])
                i += 1
        return out


# ---------------------------------------------------------------- corpus

class DocStore:
    """Tokenized documents, one per line, with random access by doc id."""

    def __init__(self, path):
        self.path = Path(path)
        offsets_path = self.path.with_name(self.path.name + ".offsets.npy")
        if offsets_path.exists() and offsets_path.stat().st_mtime >= self.path.stat().st_mtime:
            self.offsets = np.load(offsets_path)
        else:
            offs = array("q")
            with open(self.path, "rb") as f:
                pos = 0
                for line in f:
                    offs.append(pos)
                    pos += len(line)
            self.offsets = np.frombuffer(offs, dtype=np.int64).copy()
            np.save(offsets_path, self.offsets)

    def __len__(self) -> int:
        return len(self.offsets)

    def __iter__(self) -> Iterator[list[str]]:
        with open_text(self.path) as f:
            for line in f:
                yield line.split()

    def get(self, doc_ids) -> list[list[str]]:
        out = []
        with open(self.path, "rb") as f:
            for d in sorted(int(x) for x in doc_ids):
                f.seek(self.offsets[d])
                out.append(f.readline().decode("utf-8").split())
        return out

    @staticmethod
    def write(path, docs: Iterable[list[str]]) -> "DocStore":
        path = Path(path)
        ensure_dir(path.parent)
        with open_text(path, "wt") as f:
            for tokens in docs:
                f.write(" ".join(tokens) + "\n")
        return DocStore(path)


@dataclass
class Corpus:
    """Candidate vocabulary plus a (documents x terms) count matrix."""
    terms: list[str]
    doc_term: sp.csr_matrix
    name: str = ""
    docstore_path: str | None = None
    term_index: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self):
        self.term_index = {t: i for i, t in enumerate(self.terms)}
        self._term_doc = None
        self._df = None
        self.doc_map: np.ndarray | None = None   # row -> document id in the store (for subsets)

    @property
    def n_docs(self) -> int:
        return self.doc_term.shape[0]

    @property
    def n_terms(self) -> int:
        return len(self.terms)

    @property
    def term_doc(self) -> sp.csc_matrix:
        if self._term_doc is None:
            self._term_doc = self.doc_term.tocsc()
        return self._term_doc

    @property
    def df(self) -> np.ndarray:
        if self._df is None:
            self._df = np.diff(self.term_doc.indptr)
        return self._df

    def ids(self, terms: Iterable[str]) -> np.ndarray:
        return np.array([self.term_index[t] for t in terms if t in self.term_index], dtype=np.int64)

    def docs_with_any(self, term_ids) -> np.ndarray:
        term_ids = np.asarray(term_ids, dtype=np.int64)
        if len(term_ids) == 0:
            return np.zeros(0, dtype=np.int64)
        return np.unique(self.term_doc[:, term_ids].indices).astype(np.int64)

    def docstore(self) -> DocStore:
        if not self.docstore_path:
            raise ValueError("this corpus was built without a document store")
        return DocStore(self.docstore_path)

    def get_docs(self, rows) -> list[list[str]]:
        """Token lists of the given rows (row ids of this corpus or subset)."""
        rows = np.asarray(rows, dtype=np.int64)
        return self.docstore().get(self.doc_map[rows] if self.doc_map is not None else rows)

    def documents(self) -> Iterable[list[str]]:
        """Re-iterable token lists of all rows (word2vec makes several passes)."""
        if self.doc_map is None:
            return self.docstore()
        return self.get_docs(np.arange(self.n_docs))

    def subset(self, rows) -> "Corpus":
        """Same vocabulary, a subset of the documents (e.g. a dev/held-out split)."""
        rows = np.asarray(rows, dtype=np.int64)
        sub = Corpus(self.terms, self.doc_term[rows], name=self.name, docstore_path=self.docstore_path)
        sub.doc_map = self.doc_map[rows] if self.doc_map is not None else rows
        return sub

    @classmethod
    def build(cls, docs_factory: Callable[[], Iterable[list[str]]], min_df: int = 20,
              max_df_ratio: float = 0.35, allowed_terms: Iterable[str] | None = None,
              name: str = "", docstore_path=None) -> "Corpus":
        """Two passes over `docs_factory()`: document frequencies, then counts.
        A term is a candidate if min_df <= df <= max_df_ratio * n_docs (and,
        when given, it belongs to `allowed_terms`)."""
        allowed = set(allowed_terms) if allowed_terms is not None else None
        df: Counter = Counter()
        n_docs = 0
        for tokens in docs_factory():
            n_docs += 1
            df.update(set(tokens) if allowed is None else set(tokens) & allowed)
        max_df = max_df_ratio * n_docs
        terms = sorted(t for t, c in df.items() if min_df <= c <= max_df)
        index = {t: i for i, t in enumerate(terms)}
        indptr, indices, data = array("q", [0]), array("i"), array("i")
        for tokens in docs_factory():
            counts = Counter(index[t] for t in tokens if t in index)
            for j in sorted(counts):
                indices.append(j)
                data.append(counts[j])
            indptr.append(len(indices))
        mat = sp.csr_matrix((np.frombuffer(data, dtype=np.int32),
                             np.frombuffer(indices, dtype=np.int32),
                             np.frombuffer(indptr, dtype=np.int64)),
                            shape=(n_docs, len(terms)))
        return cls(terms, mat, name=name, docstore_path=str(docstore_path) if docstore_path else None)

    def save(self, out_dir) -> Path:
        out_dir = ensure_dir(out_dir)
        sp.save_npz(out_dir / "doc_term.npz", self.doc_term)
        write_json(out_dir / "corpus.json", {"name": self.name, "terms": self.terms,
                                             "docstore_path": self.docstore_path,
                                             "n_docs": self.n_docs})
        return out_dir

    @classmethod
    def load(cls, in_dir) -> "Corpus":
        in_dir = Path(in_dir)
        meta = read_json(in_dir / "corpus.json")
        return cls(meta["terms"], sp.load_npz(in_dir / "doc_term.npz").tocsr(),
                   name=meta.get("name", ""), docstore_path=meta.get("docstore_path"))


def term_counts(token_docs: Iterable[list[str]], vocab: list[str]) -> sp.csr_matrix:
    """Count vocabulary terms (phrases included) in already-tokenized texts."""
    merger = PhraseMerger(vocab)
    index = {t: i for i, t in enumerate(vocab)}
    rows, cols, vals = [], [], []
    n_docs = 0
    for tokens in token_docs:
        for j, c in Counter(index[t] for t in merger(tokens) if t in index).items():
            rows.append(n_docs)
            cols.append(j)
            vals.append(c)
        n_docs += 1
    return sp.csr_matrix((vals, (rows, cols)), shape=(n_docs, len(vocab)), dtype=np.float32)


# ---------------------------------------------------------------- Amazon Reviews 2023

def load_interactions(path, limit: int | None = None) -> pd.DataFrame:
    """(user, item, rating, timestamp) for every review, without the text,
    so millions of reviews fit in memory."""
    users, items, ratings, stamps = [], [], [], []
    for d in iter_jsonl(path, limit=limit):
        item = d.get("parent_asin") or d.get("asin")
        if d.get("user_id") is None or item is None:
            continue
        users.append(d["user_id"])
        items.append(item)
        ratings.append(float(d.get("rating") or 0))
        stamps.append(int(d.get("timestamp") or 0))
    return pd.DataFrame({"user_id": users, "item_id": items,
                         "rating": np.array(ratings, dtype=np.float32),
                         "timestamp": np.array(stamps, dtype=np.int64)})


def iter_review_texts(path, verified_only: bool = True, min_words: int = 30, ascii_only: bool = True,
                      exclude_pairs: set[tuple[str, str]] | None = None,
                      sample_fraction: float | None = None, seed: int = 0,
                      english_only: bool = False) -> Iterator[dict]:
    """Stream the review texts used as the taxonomy corpus. `exclude_pairs`
    holds the (user_id, item_id) interactions held out for recommendation
    evaluation, so the taxonomy is never induced from test-time text.
    `sample_fraction` keeps a deterministic hash-based subsample."""
    import zlib
    for d in iter_jsonl(path):
        if verified_only and not d.get("verified_purchase"):
            continue
        user, item = d.get("user_id"), d.get("parent_asin") or d.get("asin")
        if exclude_pairs and (user, item) in exclude_pairs:
            continue
        if sample_fraction is not None and \
                zlib.crc32(f"{seed}:{user}:{item}".encode()) % 100000 >= sample_fraction * 100000:
            continue
        text = clean_text(f"{d.get('title') or ''} {d.get('text') or ''}")
        if len(text.split()) < min_words or (ascii_only and not mostly_ascii(text)) or \
                (english_only and not looks_english(text)):
            continue
        yield {"user_id": user, "item_id": item, "timestamp": int(d.get("timestamp") or 0), "text": text}


def load_meta(path, item_ids=None, limit: int | None = None) -> pd.DataFrame:
    """Product metadata, optionally only for `item_ids` (saves memory)."""
    keep = set(item_ids) if item_ids is not None else None
    rows = []
    for d in iter_jsonl(path, limit=limit):
        if keep is not None and d.get("parent_asin") not in keep:
            continue
        rows.append({
            "item_id": d.get("parent_asin"),
            "title": clean_text(d.get("title")),
            "store": clean_text(d.get("store")),
            "features": clean_text(d.get("features")),
            "description": clean_text(d.get("description")),
            "categories": list(d.get("categories") or []),
            "main_category": clean_text(d.get("main_category")),
        })
    df = pd.DataFrame(rows).dropna(subset=["item_id"])
    return df.drop_duplicates("item_id").reset_index(drop=True)


def item_texts(meta: pd.DataFrame, fields=("title", "store", "features", "description")) -> pd.Series:
    """Item text from metadata fields only (see module docstring)."""
    text = meta[list(fields)].fillna("").astype(str).agg(" ".join, axis=1).map(clean_text)
    return pd.Series(text.values, index=meta["item_id"].values)


def gold_category_paths(meta: pd.DataFrame, drop_top: bool = True) -> dict[str, tuple[str, ...]]:
    """Amazon category paths as a silver-standard hierarchy. The first level is
    usually the store-wide department (e.g. 'Clothing, Shoes & Jewelry') and
    carries no information, so it is dropped by default."""
    out = {}
    for item, cats in zip(meta["item_id"], meta["categories"]):
        cats = [c for c in (cats or []) if c]
        if drop_top:
            cats = cats[1:]
        if cats:
            out[item] = tuple(cats)
    return out


# ---------------------------------------------------------------- DBLP (TaxoGen release)

def dblp_corpus(papers_path, keywords_path, name: str = "dblp") -> Corpus:
    """The DBLP titles distributed with TaxoGen (already phrase-merged), with
    the release's own candidate keywords as the vocabulary."""
    with open_text(keywords_path) as f:
        keywords = [line.strip().split("\t")[0] for line in f if line.strip()]
    store = DocStore(papers_path)
    return Corpus.build(lambda: iter(store), min_df=1, max_df_ratio=1.0,
                        allowed_terms=keywords, name=name, docstore_path=papers_path)
