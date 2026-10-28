import numpy as np
import pytest

from taxorec.taxonomy.builders import BuilderConfig, TaxoGen, TaxoGenConfig, TaxonomyBuilder, rerank
from taxorec.taxonomy.components import (CharNgramEncoder, ContextEncoder, RelationScorer, Word2VecEncoder,
                                         caseolap_scores, hdbscan_labels, kl_representativeness, rank_terms,
                                         umap_reduce)
from taxorec.taxonomy.evaluate import (assign_paths, evaluate_external, evaluate_intrinsic, npmi,
                                       pairwise_prf, taxonomy_vocab)
from taxorec.taxonomy.tree import ROOT, Taxonomy
from taxorec.data import Corpus, DocStore, term_counts


def topic_of(term):
    return term.split("_")[0]


# ---------------------------------------------------------------- tree

def test_validate_flags_child_identical_to_parent():
    tax = Taxonomy()
    a = tax.add_child(ROOT, ["x", "y"])
    tax.add_child(a.id, ["y", "x"])
    assert any(sev == "error" and "same term set" in msg for sev, msg in tax.validate())
    assert tax.stats()["n_degenerate_edges"] == 1


def test_prune_keeps_ancestors_and_roundtrip(tmp_path):
    tax = Taxonomy("m")
    a = tax.add_child(ROOT, ["a"], n_docs=10)
    b = tax.add_child(ROOT, ["b"], n_docs=5)
    tax.add_child(a.id, ["a1"], n_docs=9)
    tax.add_child(b.id, ["b1"], n_docs=1)
    small = tax.prune_to_size(3)
    assert small.size == 3 and all(n.parent in small.nodes for n in small.non_root())
    tax.save(tmp_path / "t.json")
    again = Taxonomy.load(tmp_path / "t.json")
    assert again.to_dict() == tax.to_dict()
    assert [c.id for c in again.children(ROOT)] == [a.id, b.id]


def test_import_legacy_txt(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text("x\tx,y\nx/z\tz,w\nq\tq\n", encoding="utf-8")
    tax = Taxonomy.from_txt(p)
    assert tax.size == 3 and tax.children(tax.children(ROOT)[0].id)[0].terms == ["z", "w"]


# ---------------------------------------------------------------- components

def test_caseolap_prefers_distinctive_terms():
    tf = np.array([[10.0, 10.0], [0.0, 10.0]])   # term 0 only in cell 0, term 1 everywhere
    df = np.array([[5.0, 5.0], [0.0, 5.0]])
    s = caseolap_scores(tf, df)
    assert s[0, 0] > s[0, 1]
    kl = kl_representativeness(s)
    assert kl[0] > kl[1]


def test_relation_scorer_is_absolute_not_minmax():
    scorer = RelationScorer("bm25", tau=0.3)
    v = np.ones(4)
    out = scorer.score(["shoe_running", "shoe_boot"], v,
                       [["shoe_running_trail"], ["shoe_running_mesh"], ["cake_recipe"]], [v, v, v])
    assert out[0]["bm25_hat"] == pytest.approx(out[1]["bm25_hat"])   # equal evidence, equal score
    assert out[0]["bm25_hat"] < 1.0                                  # no child is forced to 1
    assert out[2]["bm25_hat"] == 0 and not out[2]["accept"]


def test_hdbscan_small_pool_is_all_noise():
    assert (hdbscan_labels(np.random.default_rng(0).random((10, 4)), min_cluster_size=8) == -1).all()


# ---------------------------------------------------------------- builder

def _builder(corpus, encoder, **kw):
    cfg = dict(clusterer="kmeans", k=3, relation="none", local="none", max_depth=2,
               min_terms_to_split=6, min_docs_to_split=10, min_child_terms=2)
    cfg.update(kw)
    return TaxonomyBuilder(BuilderConfig.from_dict(cfg), corpus, encoder, log=lambda m: None)


def test_kmeans_builder_recovers_topics(synthetic_corpus, synthetic_encoder):
    tax = _builder(synthetic_corpus, synthetic_encoder).build()
    level1 = tax.children(ROOT)
    assert len(level1) == 3
    assert all(len({topic_of(t) for t in n.terms}) == 1 for n in level1)   # pure topics
    assert all(len(tax.children(n.id)) == 3 for n in level1)               # subtopics found
    assert tax.stats()["n_degenerate_edges"] == 0


def test_hdbscan_builder_never_copies_parent(synthetic_corpus, synthetic_encoder):
    tax = _builder(synthetic_corpus, synthetic_encoder, clusterer="hdbscan", hdb_divisor=4,
                   hdb_floor=3, hdb_min_samples=2, relation="bm25cos", tau=0.05, local="recenter").build()
    assert tax.size > 0
    assert not [m for s, m in tax.validate() if s == "error"]
    for node in tax.non_root():
        assert len({topic_of(t) for t in node.terms}) == 1
    for node in tax.nodes.values():
        kids = tax.children(node.id)
        if kids:
            placed = set().union(*(k.terms for k in kids))
            assert set(node.general_terms) == set(node.terms) - placed   # noise kept at the parent


def test_single_dense_group_becomes_leaf(synthetic_corpus, synthetic_encoder):
    # regression test for the old bug: one cluster == the whole pool used to
    # be emitted as a child identical to its parent
    tax = _builder(synthetic_corpus, synthetic_encoder, clusterer="hdbscan", hdb_divisor=1,
                   hdb_floor=200, hdb_ceiling=200).build()
    assert tax.size == 0 and tax.root.general_terms == tax.root.terms


def test_relation_threshold_rejects_children(synthetic_corpus, synthetic_encoder):
    tax = _builder(synthetic_corpus, synthetic_encoder, relation="cos", tau=1.01,
                   score_root_children=True).build()
    assert tax.size == 0


def test_builder_is_deterministic(synthetic_corpus, synthetic_encoder):
    a = _builder(synthetic_corpus, synthetic_encoder).build().to_dict()
    b = _builder(synthetic_corpus, synthetic_encoder).build().to_dict()
    assert a == b


class _CountingEncoder(CharNgramEncoder):
    calls = 0

    def encode(self, texts):
        _CountingEncoder.calls += len(texts)
        return super().encode(texts)


def test_context_encoder_groups_cooccurring_terms(tmp_path):
    docs = [["red", "rose", "garden"], ["red", "rose", "bloom"], ["blue", "sky", "cloud"],
            ["blue", "sky", "rain"]] * 3
    store = DocStore.write(tmp_path / "docs.txt", docs)
    corpus = Corpus.build(lambda: iter(store), min_df=2, max_df_ratio=1.0, docstore_path=tmp_path / "docs.txt")
    enc = ContextEncoder(_CountingEncoder(dim=32), corpus, docs_per_term=50, cache_dir=tmp_path / "cache")
    v = dict(zip(corpus.terms, enc.encode_terms(corpus.terms)))
    assert np.allclose(v["red"], v["rose"]) and np.allclose(v["blue"], v["sky"])   # same documents
    assert float(v["red"] @ v["blue"]) < float(v["red"] @ v["rose"])
    n_docs_embedded = _CountingEncoder.calls
    assert n_docs_embedded == len(docs)
    again = ContextEncoder(_CountingEncoder(dim=32), corpus.subset(np.arange(6)), docs_per_term=50,
                           cache_dir=tmp_path / "cache").encode_terms(["red"])
    assert _CountingEncoder.calls == n_docs_embedded             # subset reuses the cached embeddings
    assert np.allclose(again[0], ContextEncoder(CharNgramEncoder(dim=32), corpus.subset(np.arange(6)),
                                                docs_per_term=50).encode_terms(["red"])[0], atol=1e-3)
    # a rebuilt store (ids now mean other documents) must not hit the old cache
    store2 = DocStore.write(tmp_path / "docs2.txt", [["red", "rose", "petal"]] + docs)
    corpus2 = Corpus.build(lambda: iter(store2), min_df=2, max_df_ratio=1.0, docstore_path=tmp_path / "docs2.txt")
    ContextEncoder(_CountingEncoder(dim=32), corpus2, docs_per_term=50, cache_dir=tmp_path / "cache").encode_terms(["red"])
    assert _CountingEncoder.calls > n_docs_embedded


def test_document_split_recovers_topics(synthetic_corpus):
    # documents of one topic share words, so document clusters are topics and
    # each topic's terms are most distinctive in its own cluster
    enc = ContextEncoder(CharNgramEncoder(dim=64), synthetic_corpus, docs_per_term=20)
    cfg = dict(clusterer="kmeans", k=3, split="docs", split_min_kl=0.1, relation="none", local="none",
               max_depth=1, min_terms_to_split=6, min_docs_to_split=10, min_child_terms=2)
    tax = TaxonomyBuilder(BuilderConfig.from_dict(cfg), synthetic_corpus, enc, log=lambda m: None).build()
    kids = tax.children(ROOT)
    assert len(kids) == 3
    assert all(len({topic_of(t) for t in n.terms}) == 1 for n in kids)
    assert sorted(n.n_docs for n in kids) and sum(n.n_docs for n in kids) <= synthetic_corpus.n_docs
    again = TaxonomyBuilder(BuilderConfig.from_dict(cfg), synthetic_corpus, enc, log=lambda m: None).build()
    assert again.to_dict() == tax.to_dict()


def test_document_split_keeps_spread_terms_at_parent(tmp_path):
    # 'good' is in every document: frequent, but spread evenly over the
    # children, so it describes the parent; 'petal' (2 documents) is too rare
    # for the df bar and joins its topic by similarity
    topics = {"flower": ["rose", "tulip", "garden", "bloom", "seed", "soil"],
              "sky": ["cloud", "rain", "storm", "wind", "sun", "fog"],
              "car": ["engine", "wheel", "brake", "tire", "gear", "fuel"]}
    rng = np.random.default_rng(0)
    docs = [list(rng.choice(words, 4, replace=False)) + ["good"] for words in topics.values() for _ in range(30)]
    docs[0].append("petal")
    docs[1].append("petal")
    store = DocStore.write(tmp_path / "docs.txt", docs)
    corpus = Corpus.build(lambda: iter(store), min_df=2, max_df_ratio=1.0, docstore_path=tmp_path / "docs.txt")
    enc = ContextEncoder(CharNgramEncoder(dim=64), corpus, docs_per_term=20)
    cfg = dict(clusterer="kmeans", k=3, split="docs", split_min_kl=0.1, split_min_df=5, split_attach_min_sim=-1.0,
               center=True, relation="none", local="none", max_depth=1, min_terms_to_split=6,
               min_docs_to_split=10, min_child_terms=2)
    tax = TaxonomyBuilder(BuilderConfig.from_dict(cfg), corpus, enc, log=lambda m: None).build()
    kids = tax.children(ROOT)
    assert len(kids) == 3
    assert "good" in tax.root.general_terms and not any("good" in k.terms for k in kids)
    assert "petal" in next(k for k in kids if "rose" in k.terms).terms


def test_centering_lowers_parent_child_cosine(synthetic_corpus):
    enc = ContextEncoder(CharNgramEncoder(dim=64), synthetic_corpus, docs_per_term=20)
    cfg = dict(clusterer="kmeans", k=3, split="docs", split_min_kl=0.1, relation="cos", tau=-1.0,
               score_root_children=True, local="none", max_depth=1, min_terms_to_split=6,
               min_docs_to_split=10, min_child_terms=2)
    cos = {}
    for center in (False, True):
        tax = TaxonomyBuilder(BuilderConfig.from_dict({**cfg, "center": center}), synthetic_corpus, enc,
                              log=lambda m: None).build()
        cos[center] = np.mean([n.scores["cos"] for n in tax.children(ROOT)])
    assert cos[True] < cos[False]


def test_document_split_needs_context_encoder(synthetic_corpus, synthetic_encoder):
    with pytest.raises(ValueError, match="context encoder"):
        _builder(synthetic_corpus, synthetic_encoder, split="docs").build()


def test_umap_reduce_is_deterministic():
    x = np.random.default_rng(0).standard_normal((60, 16)).astype(np.float32)
    a, b = umap_reduce(x, dim=5, seed=3), umap_reduce(x, dim=5, seed=3)
    assert a.shape == (60, 5) and np.array_equal(a, b)
    assert umap_reduce(x[:6], dim=5).shape == (6, 16)   # too few points: returned unchanged


def test_umap_builder_valid_and_deterministic(synthetic_corpus, synthetic_encoder):
    # UMAP does not keep between-cluster distances on 72 points, so only
    # validity and determinism are checked here, not topic purity
    for clusterer in ("hdbscan", "kmeans"):
        kw = dict(clusterer=clusterer, reduce="umap", hdb_divisor=4, hdb_floor=3,
                  hdb_min_samples=2, relation="bm25cos", tau=0.05, local="recenter")
        tax = _builder(synthetic_corpus, synthetic_encoder, **kw).build()
        assert not [m for s, m in tax.validate() if s == "error"]
        assert tax.to_dict() == _builder(synthetic_corpus, synthetic_encoder, **kw).build().to_dict()


def test_taxogen_baseline_runs(synthetic_corpus):
    enc = Word2VecEncoder.train(synthetic_corpus.docstore(), vector_size=16, min_count=2, epochs=20, seed=0)
    cfg = TaxoGenConfig(n_cluster=3, n_iter=2, filter_thre=0.0, n_expand=20, max_depth=2,
                        min_docs_to_split=20, w2v={"vector_size": 16, "min_count": 2, "epochs": 10})
    tax = TaxoGen(cfg, synthetic_corpus, enc, log=lambda m: None).build()
    assert 2 <= len(tax.children(ROOT)) <= 3
    assert tax.stats()["n_degenerate_edges"] == 0
    again = TaxoGen(cfg, synthetic_corpus, enc, log=lambda m: None).build()
    assert again.to_dict() == tax.to_dict()
    # local vectors may have their own dimension, as in the original code (50-d local, 100-d global)
    cfg.w2v = {"vector_size": 8, "min_count": 2, "epochs": 10, "sg": 0, "sample": 1e-4}
    assert TaxoGen(cfg, synthetic_corpus, enc, log=lambda m: None).build().size >= 2


# ---------------------------------------------------------------- evaluation

def test_rerank_and_intrinsic(synthetic_corpus, synthetic_encoder):
    tax = _builder(synthetic_corpus, synthetic_encoder).build()
    shuffled = Taxonomy.from_dict(tax.to_dict())
    for n in shuffled.non_root():
        n.terms = sorted(n.terms, reverse=True)
    again, info = rerank(shuffled, synthetic_corpus)
    assert info["terms_dropped"] == 0
    c = synthetic_corpus
    for n in tax.non_root():
        expected = rank_terms(c, c.ids(n.terms), c.docs_with_any(c.ids(n.terms)),
                              [c.docs_with_any(c.ids(s.terms)) for s in tax.siblings(n.id)])
        assert again.nodes[n.id].terms == [c.terms[i] for i in expected]
    res = evaluate_intrinsic(synthetic_corpus, tax)
    assert res["structure"]["n_nodes"] == tax.size
    assert 0 < res["subsumption_mean"] <= 1 and res["term_coverage"] > 0.9


def test_npmi_coherent_beats_random(synthetic_corpus, hierarchy):
    terms, _ = hierarchy
    coherent = synthetic_corpus.ids(terms[("shoe", "alpha")][:5])
    mixed = synthetic_corpus.ids([terms[(t, s)][0] for t, s in [("shoe", "alpha"), ("bag", "beta"),
                                                                ("dress", "gamma"), ("bag", "alpha")]])
    assert npmi(synthetic_corpus, coherent) > npmi(synthetic_corpus, mixed)


def test_pairwise_prf_known_values():
    assert pairwise_prf(["a", "a", "b", "b"], ["x", "x", "y", "y"])["f1"] == pytest.approx(1.0)
    res = pairwise_prf(["a", "a", "a", "a"], ["x", "x", "y", "y"])
    assert res["precision"] == pytest.approx(2 / 6) and res["recall"] == pytest.approx(1.0)


def test_assign_paths_and_external(synthetic_corpus, synthetic_encoder, hierarchy):
    tax = _builder(synthetic_corpus, synthetic_encoder).build()
    terms, _ = hierarchy
    items, gold = [], []
    for (t, s), names in terms.items():
        for k in range(3):
            items.append(names[k:k + 3])
            gold.append((t, s))
    vocab = taxonomy_vocab(tax)
    paths = assign_paths(tax, term_counts(items, vocab), vocab)
    assert all(len(p) == 2 for p in paths)
    ext = evaluate_external(paths, gold, depths=(1, 2))
    assert ext["depth1"]["f1"] == pytest.approx(1.0) and ext["depth2"]["f1"] > 0.9
