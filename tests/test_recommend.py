import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from taxorec.recommend import (BPRMF, INF_RANK, ContentKNN, Hybrid, ItemKNN, Popularity, Split, TaxonomyKNN,
                               coverage_mask, evaluate, kcore, leave_last_out, param_grid, simplex_grid,
                               summarize_ranks, target_ranks, tune_and_test)


def interactions(rows):
    return pd.DataFrame(rows, columns=["user_id", "item_id", "timestamp"]).assign(rating=5.0)


def test_kcore_is_iterative():
    df = interactions([("u1", "a", 1), ("u1", "b", 2), ("u2", "a", 1), ("u2", "b", 2), ("u3", "c", 1)])
    out = kcore(df, 2)
    assert set(out["user_id"]) == {"u1", "u2"} and set(out["item_id"]) == {"a", "b"}


def test_kcore_users_only():
    df = interactions([("u1", "a", 1), ("u1", "b", 2), ("u1", "c", 3), ("u2", "a", 1), ("u2", "d", 2)])
    out = kcore(df, 3, k_item=1)
    assert set(out["user_id"]) == {"u1"} and set(out["item_id"]) == {"a", "b", "c"}   # items seen once stay


def test_leave_last_out_split(tmp_path):
    df = interactions([("u1", "a", 1), ("u1", "b", 2), ("u1", "c", 3), ("u1", "d", 4),
                       ("u2", "a", 5), ("u2", "c", 1)])
    s = leave_last_out(df, min_len=3)
    u1 = list(s.users).index("u1")
    items = list(s.items)
    assert s.test_item[u1] == items.index("d") and s.valid_item[u1] == items.index("c")
    assert sorted(s.train[u1].indices.tolist()) == [items.index("a"), items.index("b")]
    u2 = list(s.users).index("u2")
    assert s.test_item[u2] == -1 and s.train[u2].nnz == 2          # too short: all train
    hist = s.history(include_valid=True)
    assert hist[u1, items.index("c")] == 1 and hist[u1, items.index("d")] == 0
    s.save(tmp_path / "s.npz")
    t = Split.load(tmp_path / "s.npz")
    assert (t.train != s.train).nnz == 0 and t.test_item.tolist() == s.test_item.tolist()
    assert set(s.heldout_pairs()["role"]) == {"valid", "test"}


def test_target_ranks_masks_history_and_breaks_ties_pessimistically():
    scores = np.array([[0.9, 0.5, 0.5, 0.1, 0.0]])
    hist = sp.csr_matrix(([1.0], ([0], [0])), shape=(1, 5))
    assert target_ranks(scores, np.array([2]), hist)[0] == 1    # item 0 masked; tie -> after item 1
    assert target_ranks(scores, np.array([1]), hist)[0] == 1
    assert target_ranks(scores, np.array([3]), hist)[0] == 2
    assert target_ranks(scores, np.array([0]), hist)[0] == INF_RANK   # target in history
    # a sparse model that cannot score the target gets no credit from tie order
    flat = np.zeros((1, 1000))
    flat[0, 5] = 1.0
    assert target_ranks(flat, np.array([0]), sp.csr_matrix((1, 1000)))[0] == 999   # last place


def test_negative_scores_still_rank():
    # regression: cosine / factorization scores can be negative and must rank normally
    scores = np.array([[-0.2, -0.5, -0.9]])
    assert target_ranks(scores, np.array([0]), sp.csr_matrix((1, 3)))[0] == 0


def test_coverage_is_sign_agnostic():
    hist = sp.csr_matrix((2, 30))
    sparse_cf = np.zeros((1, 30))
    sparse_cf[0, :3] = [1, 2, 3]
    dense_cos = np.random.default_rng(0).standard_normal((1, 30))
    assert not coverage_mask(sparse_cf, hist[:1], 10)[0]
    assert coverage_mask(dense_cos, hist[:1], 10)[0]


def test_summarize_ranks():
    m = summarize_ranks(np.array([0, 2, INF_RANK]), ks=(1, 5))
    assert m["recall@1"] == pytest.approx(1 / 3) and m["recall@5"] == pytest.approx(2 / 3)
    assert m["ndcg@5"] == pytest.approx((1 + 1 / np.log2(4)) / 3)


@pytest.fixture
def small():
    rng = np.random.default_rng(0)
    x = sp.csr_matrix((rng.random((30, 20)) < 0.2).astype(np.float32))
    emb = rng.standard_normal((20, 8)).astype(np.float32)
    return x, emb


def test_itemknn_matches_bruteforce(small):
    x, _ = small
    m = ItemKNN().fit(x)
    d = x.toarray()
    n = d.sum(0)
    sim = d.T @ d / np.sqrt(np.outer(n, n) + 1e-12)
    np.fill_diagonal(sim, 0)
    sim[np.outer(n, n) == 0] = 0
    assert np.allclose(m.score(np.arange(30)), d @ sim, atol=1e-5)


def test_content_knn_matches_bruteforce(small):
    x, emb = small
    e = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    m = ContentKNN(emb).fit(x)
    assert np.allclose(m.score(np.arange(5)), x.toarray()[:5] @ (e @ e.T), atol=1e-5)


def test_taxonomy_knn_is_common_prefix_length():
    paths = [("a", "a/1"), ("a", "a/2"), ("b",), ()]
    x = sp.csr_matrix(([1.0], ([0], [0])), shape=(1, 4))
    m = TaxonomyKNN(paths).fit(x)
    assert np.allclose(m.score(np.array([0])), [[1.0, 0.5, 0.0, 0.0]])


def test_hybrid_single_component_preserves_order(small):
    x, emb = small
    sem = ContentKNN(emb).fit(x).score(np.arange(30))
    hyb = Hybrid({"sem": ContentKNN(emb)}, {"sem": 1.0}).fit(x).score(np.arange(30))
    pos = sem.max(axis=1) > 0
    assert (np.argsort(-sem[pos], axis=1)[:, :5] == np.argsort(-hyb[pos], axis=1)[:, :5]).all()
    with pytest.raises(ValueError):
        Hybrid({"sem": ContentKNN(emb)}, {"cf": 1.0})


def test_hybrid_weights_change_the_ranking(small):
    # regression: per-user max-normalization with clipping at 0 erased the
    # order of negative semantic scores, so the fused ranking ignored weights
    x, emb = small
    paths = [(f"g{i % 4}",) for i in range(20)]
    def top(w):
        h = Hybrid({"sem": ContentKNN(emb), "tax": TaxonomyKNN(paths)}, {"sem": 1 - w, "tax": w}).fit(x)
        return np.argsort(-h.score(np.arange(30)), axis=1, kind="stable")[:, :5]
    assert (top(0.2) != top(0.8)).any()
    s = Hybrid({"sem": ContentKNN(emb), "tax": TaxonomyKNN(paths)}, {"sem": 0.5, "tax": 0.5}).fit(x)
    scores = s.score(np.arange(30))
    assert scores.min() >= 0 and scores.max() <= 1 + 1e-6


def test_grids():
    assert param_grid({"a": [1, 2], "b": [3]}) == [{"a": 1, "b": 3}, {"a": 2, "b": 3}]
    g = simplex_grid(["x", "y", "z"], 0.25)
    assert all(abs(sum(w.values()) - 1) < 1e-9 and min(w.values()) > 0 for w in g) and len(g) == 3


def _clustered_split(seed=0, n_users=200, n_items=60):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        group = u % 3
        items = rng.choice(np.arange(group * 20, group * 20 + 20), size=6, replace=False)
        rows += [(f"u{u}", f"i{i:02d}", t) for t, i in enumerate(items)]
    return leave_last_out(interactions(rows))


def test_bpr_learns_structure():
    s = _clustered_split()
    m = BPRMF(dim=8, lr=0.1, epochs=20, seed=0).fit(s.history(True))
    assert m.losses[-1] < m.losses[0]
    users, targets = s.eval_users("test")
    assert evaluate(m, users, targets)["summary"]["recall@10"] > \
        evaluate(Popularity().fit(s.history(True)), users, targets)["summary"]["recall@10"]


def test_tuning_selects_on_validation_and_tests_once():
    s = _clustered_split()
    seen = []

    def factory(p):
        seen.append(p)
        return ItemKNN(**p)

    res = tune_and_test(factory, param_grid({"shrink": [0.0, 5.0]}), s, log=lambda m: None)
    assert len(seen) == 3                                   # two validation fits + one final refit
    assert res["best_params"] in ({"shrink": 0.0}, {"shrink": 5.0})
    assert res["test"]["n_users"] == int((s.test_item >= 0).sum())
    assert res["test"]["coverage"] > 0.5
