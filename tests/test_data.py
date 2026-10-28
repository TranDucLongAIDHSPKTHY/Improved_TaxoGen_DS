from taxorec import data as D
from taxorec.toy import make_toy_data


def test_phrase_merger_longest_match():
    merge = D.PhraseMerger(["long_sleeve", "long_sleeve_dress", "summer_dress"])
    assert merge("a long sleeve dress for summer dress".split()) == \
        ["a", "long_sleeve_dress", "for", "summer_dress"]


def test_corpus_build_filters_and_indexes(tmp_path):
    docs = [["a", "b", "x"], ["a", "c"], ["a", "b"], ["a", "b", "rare"]]
    store = D.DocStore.write(tmp_path / "d.txt", docs)
    corpus = D.Corpus.build(lambda: iter(store), min_df=2, max_df_ratio=0.8)
    assert corpus.terms == ["b"]                 # 'a' is in 4/4 docs > 0.8; c, x, rare below min_df
    corpus = D.Corpus.build(lambda: iter(store), min_df=2, max_df_ratio=1.0)
    assert corpus.terms == ["a", "b"]
    assert corpus.docs_with_any(corpus.ids(["b"])).tolist() == [0, 2, 3]
    assert corpus.df.tolist() == [4, 3]
    corpus.save(tmp_path / "c")
    loaded = D.Corpus.load(tmp_path / "c")
    assert loaded.terms == corpus.terms and (loaded.doc_term != corpus.doc_term).nnz == 0
    assert store.get([3, 1]) == [["a", "c"], ["a", "b", "rare"]]


def test_term_counts_uses_phrases():
    m = D.term_counts([["running", "shoe", "shoe"], []], ["running_shoe", "shoe"])
    assert m.shape == (2, 2)
    assert m.toarray().tolist() == [[1, 1], [0, 0]]


def test_toy_loaders_and_item_text_has_no_reviews(tmp_path):
    paths = make_toy_data(tmp_path, n_items_per_type=3, n_users=20)
    inter = D.load_interactions(paths["reviews"])
    assert list(inter.columns) == ["user_id", "item_id", "rating", "timestamp"]
    meta = D.load_meta(paths["meta"], item_ids=inter["item_id"].unique()[:5])
    assert len(meta) == 5 and len(D.load_meta(paths["meta"])) == 27
    texts = D.item_texts(D.load_meta(paths["meta"]))
    # item text is metadata only: no review sentence may appear in it
    sample_review = next(D.iter_review_texts(paths["reviews"], min_words=1))["text"]
    assert all(sample_review not in t for t in texts)
    gold = D.gold_category_paths(meta)
    assert all(len(p) == 2 for p in gold.values())


def test_review_stream_excludes_heldout_pairs_and_samples(tmp_path):
    from taxorec.common import write_jsonl
    rows = [{"user_id": u, "parent_asin": i, "timestamp": t, "verified_purchase": True,
             "title": "t", "text": "word " * 40} for u, i, t in [("u1", "i1", 1), ("u1", "i2", 2), ("u2", "i1", 3)]]
    write_jsonl(tmp_path / "r.jsonl", rows)
    docs = list(D.iter_review_texts(tmp_path / "r.jsonl", min_words=10, exclude_pairs={("u1", "i2")}))
    assert [(d["user_id"], d["item_id"]) for d in docs] == [("u1", "i1"), ("u2", "i1")]
    half = [d["item_id"] for d in D.iter_review_texts(tmp_path / "r.jsonl", min_words=10, sample_fraction=0.5)]
    assert half == [d["item_id"] for d in D.iter_review_texts(tmp_path / "r.jsonl", min_words=10,
                                                             sample_fraction=0.5)]   # deterministic


def test_corpus_subset_maps_back_to_documents(tmp_path):
    docs = [["a", "b"], ["b", "c"], ["a", "c"], ["c", "d"]]
    store = D.DocStore.write(tmp_path / "d.txt", docs)
    corpus = D.Corpus.build(lambda: iter(store), min_df=1, max_df_ratio=1.0, docstore_path=tmp_path / "d.txt")
    sub = corpus.subset([1, 3])
    assert sub.n_docs == 2 and sub.terms == corpus.terms
    assert sub.get_docs([1]) == [["c", "d"]]
    assert list(sub.subset([1]).documents()) == [["c", "d"]]
    assert list(corpus.documents()) == docs


def test_clean_text_strips_html():
    assert D.clean_text("Great<br />fit &amp; color") == "Great fit & color"
    assert D.mostly_ascii("plain english") and not D.mostly_ascii("très très éé ç")


def test_looks_english_rejects_unaccented_spanish():
    assert D.looks_english("I love this dress and the fit is great, but it runs a bit small for me")
    assert not D.looks_english("Me encanta el vestido, la talla es perfecta y la tela es muy suave para el verano")
    assert not D.looks_english("")


def _fake_release(tmp_path):
    """A small zip shaped like the TaxoGen DBLP release (incl. macOS junk)."""
    import zipfile
    contents = {"input/papers.txt": b"a b\nc d\n", "input/keywords.txt": b"a\nc\n",
                "input/embeddings.txt": b"2 1\na 0.1\nc 0.2\n", "input/index.txt": b"unused\n"}
    zip_path = tmp_path / "src" / "release.zip"
    zip_path.parent.mkdir()
    with zipfile.ZipFile(zip_path, "w") as z:
        for name, data in contents.items():
            z.writestr(name, data)
        z.writestr("__MACOSX/input/._papers.txt", b"junk")
    import hashlib
    spec = {"source": "test", "url": zip_path.as_uri(), "zip_name": "release.zip",
            "zip_sha256": D.sha256_of(zip_path),
            "files": {m: hashlib.sha256(contents[m]).hexdigest()
                      for m in ("input/papers.txt", "input/keywords.txt", "input/embeddings.txt")}}
    return spec


def test_download_dblp_verifies_and_extracts(tmp_path):
    spec = _fake_release(tmp_path)
    input_dir = tmp_path / "dblp_raw" / "input"
    out = D.download_dblp(input_dir, spec)
    assert sorted(p.name for p in input_dir.iterdir()) == ["embeddings.txt", "keywords.txt", "papers.txt"]
    assert out["papers"].read_bytes() == b"a b\nc d\n"
    assert (tmp_path / "dblp_raw" / "release.zip").exists()
    (tmp_path / "dblp_raw" / "release.zip").unlink()
    D.download_dblp(input_dir, spec)          # files already verified: nothing is downloaded again
    assert not (tmp_path / "dblp_raw" / "release.zip").exists()


def test_download_dblp_rejects_a_different_file(tmp_path):
    import pytest
    spec = {**_fake_release(tmp_path), "zip_sha256": "0" * 64}
    with pytest.raises(ValueError, match="does not match"):
        D.download_dblp(tmp_path / "dblp_raw" / "input", spec)
    assert (tmp_path / "dblp_raw" / "release.zip.bad").exists()
    assert not (tmp_path / "dblp_raw" / "input" / "papers.txt").exists()
