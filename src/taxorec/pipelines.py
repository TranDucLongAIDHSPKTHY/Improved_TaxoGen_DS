"""One function per CLI command. Each reads the dataset config, writes its
artifacts to fixed locations under `paths.out_dir` (see Layout), and records
a run (config, environment, input fingerprints, log) under
`<out_dir>/runs/<command>/`."""
from __future__ import annotations

import copy
import glob
import hashlib
import itertools
import json
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from taxorec import data as D
from taxorec.common import Run, ensure_dir, git_info, read_json, read_jsonl, write_json, write_jsonl
from taxorec.human_eval import (SampleConfig, analyze, build_batch, file_manifest, import_form_export,
                                load_key, load_responses, primary_holm, sample_items, sha256_file,
                                validate_responses, write_batch)
from taxorec.recommend import (BPRMF, ContentKNN, Hybrid, ItemKNN, Popularity, Split, TaxonomyKNN,
                               leave_last_out, param_grid, per_user_metric, prepare_interactions,
                               simplex_grid, tune_and_test)
from taxorec.stats import holm, paired_bootstrap
from taxorec.taxonomy.builders import (BuilderConfig, TaxoGen, TaxoGenConfig, TaxonomyBuilder,
                                       build_bertopic, rerank)
from taxorec.taxonomy.components import ContextEncoder, Encoder, SBERTEncoder, Word2VecEncoder, make_encoder
from taxorec.taxonomy.evaluate import assign_paths, evaluate_external, evaluate_intrinsic, taxonomy_vocab
from taxorec.taxonomy.tree import Taxonomy


class Layout:
    def __init__(self, cfg: dict):
        self.out = Path(cfg["paths"]["out_dir"])
        self.corpus = self.out / "corpus"
        self.raw_tokens = self.corpus / "raw_tokens.txt"
        self.docs = self.corpus / "docs.txt"
        self.rec = self.out / "rec"
        self.split = self.rec / "split.npz"
        self.heldout = self.rec / "heldout_pairs.csv"
        self.items = self.rec / "items.jsonl"
        self.item_emb = self.rec / "item_emb.npy"
        self.taxonomies = self.out / "taxonomies"
        self.tuning = self.out / "tuning"
        self.eval = self.out / "eval"
        self.human = self.out / "human_eval"
        self.frozen_pointer = self.human / "FROZEN.json"
        self.results = self.out / "results"
        self.cache = self.out / "cache"
        self.runs = self.out / "runs"
        self.archive = self.out / "archive"

    def taxonomy(self, name: str, seed: int | None = None) -> Path:
        """Main taxonomy (built with the config seed), or one of the extra seeds."""
        if seed is None:
            return self.taxonomies / f"{name}.json"
        return self.taxonomies / "seeds" / f"{name}@seed{seed}.json"


def _run(cfg: dict, command: str, inputs=()) -> Run:
    return Run(Layout(cfg).runs, command, cfg, inputs=inputs, seed=cfg.get("seed", 0))


def _seeds(cfg: dict) -> list[int]:
    return list(cfg.get("seeds") or [cfg.get("seed", 0)])


def get_encoder(cfg: dict, name: str, corpus: D.Corpus | None = None, seed: int | None = None) -> Encoder:
    """Named encoders from `encoders:`; trained word2vec models are cached per
    seed and per corpus size (a dev subset gets its own model)."""
    seed = cfg.get("seed", 0) if seed is None else seed
    spec = copy.deepcopy(cfg["encoders"][name])
    layout = Layout(cfg)
    if spec["type"] == "sbert":
        spec.setdefault("cache_dir", str(layout.cache))
        return SBERTEncoder(**{k: v for k, v in spec.items() if k != "type"})
    if spec["type"] == "context":
        if corpus is None:
            raise ValueError("a context encoder needs the corpus")
        return ContextEncoder(get_encoder(cfg, spec["base"], corpus, seed), corpus,
                              spec.get("docs_per_term", 30), seed=seed, cache_dir=layout.cache,
                              drop_df=spec.get("drop_df"))
    if spec["type"] == "word2vec" and not spec.get("pretrained"):
        # keyed by the exact documents, so a rebuilt corpus or another subset never reuses a model
        h = hashlib.sha256(corpus.docstore().offsets.tobytes())
        if corpus.doc_map is not None:
            h.update(np.ascontiguousarray(corpus.doc_map, dtype=np.int64).tobytes())
        path = layout.cache / f"{name}-seed{seed}-n{corpus.n_docs}-{h.hexdigest()[:12]}.kv"
        if path.exists():
            from gensim.models import KeyedVectors
            return Word2VecEncoder(KeyedVectors.load(str(path)))
        enc = make_encoder(spec, corpus, seed)
        ensure_dir(path.parent)
        enc.kv.save(str(path))
        return enc
    return make_encoder(spec, corpus, seed)


def _apply_dotted(spec: dict, key: str, value) -> None:
    parts = key.split(".")
    node = spec
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def _flat_metrics(res: dict) -> dict:
    return {"n_nodes": res["structure"]["n_nodes"], "npmi": res["npmi_mean"],
            "subsumption": res["subsumption_mean"], "sibling_overlap": res["sibling_overlap_mean"],
            "term_coverage": res["term_coverage"], "doc_coverage": res["doc_coverage"]}


def _aggregate(rows: list[dict], keys: list[str], metrics: list[str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = []
    for key, grp in groups.items():
        row = dict(zip(keys, key))
        row["n_seeds"] = len(grp)
        for m in metrics:
            vals = np.array([g[m] for g in grp if g.get(m) is not None], dtype=float)
            vals = vals[np.isfinite(vals)]
            row[m] = float(vals.mean()) if len(vals) else float("nan")
            row[f"{m}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        out.append(row)
    return out


# ---------------------------------------------------------------- data

def download(cfg: dict) -> None:
    if cfg["corpus"].get("source") == "dblp":
        papers = Path(cfg["corpus"]["papers"])
        if papers.name != "papers.txt" or Path(cfg["corpus"]["keywords"]).parent != papers.parent:
            raise ValueError("download expects corpus.papers/keywords to be <dir>/papers.txt, <dir>/keywords.txt")
        D.download_dblp(papers.parent)
        return
    D.download_amazon(cfg["amazon"]["category"], Path(cfg["amazon"]["reviews"]).parent)


def make_toy(cfg: dict) -> None:
    from taxorec.toy import make_toy_data
    # the toy data is a fixed test fixture, independent of the experiment seed
    paths = make_toy_data(Path(cfg["amazon"]["reviews"]).parent, seed=cfg.get("toy_data_seed", 0))
    print(f"[toy] wrote {paths['reviews']} and {paths['meta']}")


def prepare_rec(cfg: dict) -> dict:
    a, r, L = cfg["amazon"], cfg["rec"], Layout(cfg)
    run = _run(cfg, "prepare-rec", [a["reviews"], a["meta"]])
    raw = D.load_interactions(a["reviews"])
    meta = D.load_meta(a["meta"], item_ids=raw["item_id"].unique())
    texts = D.item_texts(meta, r["item_fields"])
    texts = texts[texts.str.len() >= r.get("min_item_text_len", 10)]
    core = r.get("kcore", 5)
    k_user, k_item = (core["user"], core.get("item")) if isinstance(core, dict) else (core, None)
    inter = prepare_interactions(raw, valid_items=texts.index, min_rating=r.get("min_rating"),
                                 k=k_user, k_item=k_item)
    del raw
    split = leave_last_out(inter, min_len=r.get("min_len", 3))
    ensure_dir(L.rec)
    split.save(L.split)
    split.heldout_pairs().to_csv(L.heldout, index=False)
    cats = dict(zip(meta["item_id"], meta["categories"]))
    write_jsonl(L.items, [{"item_id": i, "text": texts[i], "categories": list(cats.get(i) or [])}
                          for i in split.items])
    emb = get_encoder(cfg, r["item_encoder"]).encode([texts[i] for i in split.items])
    np.save(L.item_emb, emb)
    stats = {"n_users": split.n_users, "n_items": split.n_items, "n_interactions": int(len(inter)),
             "density": len(inter) / (split.n_users * split.n_items),
             "n_eval_users": int((split.test_item >= 0).sum()),
             "median_history": float(inter.groupby("user_id").size().median())}
    run.log(f"[prepare-rec] {stats}")
    write_json(L.rec / "stats.json", stats)
    run.finish(stats)
    return stats


def prepare_corpus(cfg: dict) -> dict:
    c, L = cfg["corpus"], Layout(cfg)
    if c["source"] == "dblp":
        run = _run(cfg, "prepare-corpus", [c["papers"], c["keywords"]])
        corpus = D.dblp_corpus(c["papers"], c["keywords"], name=cfg["dataset"])
    else:
        a = cfg["amazon"]
        exclude = None
        if c.get("exclude_heldout", True):
            if not L.heldout.exists():
                raise FileNotFoundError(f"{L.heldout} missing: run prepare-rec first so held-out "
                                        "interactions can be excluded from the taxonomy corpus")
            held = pd.read_csv(L.heldout, dtype=str)
            exclude = set(zip(held["user_id"], held["item_id"]))
        run = _run(cfg, "prepare-corpus", [a["reviews"], L.heldout if exclude else None])
        records = D.iter_review_texts(a["reviews"], verified_only=c.get("verified_only", True),
                                      min_words=c.get("min_words", 30), ascii_only=c.get("ascii_only", True),
                                      exclude_pairs=exclude, sample_fraction=c.get("sample_fraction"),
                                      seed=cfg.get("seed", 0), english_only=c.get("english_only", False))
        meta_rows = []

        def texts():
            for rec in records:
                meta_rows.append((rec["user_id"], rec["item_id"], rec["timestamp"]))
                yield rec["text"]

        # stream: tokenize to disk, mine phrases over the file, write merged docs
        raw = D.DocStore.write(L.raw_tokens, D.make_tokenizer(c.get("tokenizer")).tokenize_many(texts()))
        run.log(f"[prepare-corpus] {len(raw)} review documents tokenized")
        phrases = D.mine_phrases(lambda: iter(raw), **c["phrases"]) if c.get("phrases") else set()
        merger = D.PhraseMerger(phrases)
        kept = []

        def merged():
            for i, tokens in enumerate(raw):
                m = merger(tokens)
                if len(m) >= c.get("min_doc_tokens", 5):
                    kept.append(i)
                    yield m

        store = D.DocStore.write(L.docs, merged())
        pd.DataFrame([meta_rows[i] for i in kept], columns=["user_id", "item_id", "timestamp"]) \
            .to_csv(L.corpus / "docs_meta.csv", index=False)
        write_json(L.corpus / "phrases.json", sorted(phrases))
        corpus = D.Corpus.build(lambda: iter(store), min_df=c.get("min_df", 20),
                                max_df_ratio=c.get("max_df_ratio", 0.35), name=cfg["dataset"],
                                docstore_path=L.docs)
    corpus.save(L.corpus)
    stats = {"n_docs": corpus.n_docs, "n_terms": corpus.n_terms,
             "multiword_terms": sum("_" in t for t in corpus.terms)}
    run.log(f"[prepare-corpus] {stats}")
    write_json(L.corpus / "stats.json", stats)
    run.finish(stats)
    return stats


# ---------------------------------------------------------------- taxonomy

def _method_spec(cfg: dict, name: str, log=print) -> dict:
    """The method's config, with parameters chosen by tune-taxonomy applied
    (unless taxonomy.use_tuned is false)."""
    spec = copy.deepcopy(cfg["taxonomy"]["methods"][name])
    chosen_path = Layout(cfg).tuning / "chosen.json"
    if cfg["taxonomy"].get("use_tuned", True) and chosen_path.exists():
        chosen = read_json(chosen_path).get(name)
        if chosen:
            for key, value in chosen.items():
                _apply_dotted(spec, key, value)
            log(f"[taxonomy] {name}: using tuned parameters {chosen} from {chosen_path}")
    return spec


def _build(cfg: dict, spec: dict, corpus: D.Corpus, name: str, log, seed: int | None = None) -> Taxonomy:
    seed = cfg.get("seed", 0) if seed is None else seed
    kind = spec["type"]
    if kind == "builder":
        enc = get_encoder(cfg, spec["encoder"], corpus, seed)
        return TaxonomyBuilder(BuilderConfig.from_dict({**spec.get("builder", {}), "seed": seed}),
                               corpus, enc, method=name, log=log).build()
    if kind == "taxogen":
        enc = get_encoder(cfg, spec["encoder"], corpus, seed)
        if not isinstance(enc, Word2VecEncoder):
            raise ValueError("TaxoGen needs a word2vec encoder")
        return TaxoGen(TaxoGenConfig.from_dict({**spec.get("taxogen", {}), "seed": seed}),
                       corpus, enc, log=log).build()
    if kind == "bertopic":
        enc = get_encoder(cfg, spec.get("encoder", "sbert"), corpus, seed)
        return build_bertopic(corpus, enc, seed=seed, **spec.get("bertopic", {}))
    if kind == "import":
        return Taxonomy.from_txt(spec["path"], method=name)
    raise ValueError(f"unknown taxonomy method type {kind!r}")


def corpus_splits(cfg: dict, corpus: D.Corpus) -> dict[str, D.Corpus]:
    """Fixed document split for the taxonomy experiments (main seed): `train`
    builds the tuning grid, `val` scores it and every development decision,
    `test` is used only for the reported automatic metrics; final taxonomies
    are built on `build` = train + val."""
    s = cfg["taxonomy"]["split"]
    perm = np.random.default_rng(cfg.get("seed", 0)).permutation(corpus.n_docs)
    n_train, n_val = int(s["train"] * corpus.n_docs), int(s["val"] * corpus.n_docs)
    rows = {"train": perm[:n_train], "val": perm[n_train:n_train + n_val],
            "test": perm[n_train + n_val:], "build": perm[:n_train + n_val]}
    return {k: corpus.subset(np.sort(v)) for k, v in rows.items()}


def _open_test(cfg: dict, step: str, force: bool = False) -> None:
    """The test documents are scored once per step, after the method is fixed:
    a second opening would let test results steer the method. Each opening is
    recorded (time, commit) in eval/test_access.json; --force only for debugging."""
    if not cfg["evaluation"].get("test_once", True):
        return
    L = Layout(cfg)
    path = L.eval / "test_access.json"
    log = read_json(path) if path.exists() else {}
    if step in log and not force:
        raise RuntimeError(f"{step} already scored the test documents ({log[step]['time']}, commit "
                           f"{log[step]['git'].get('commit')}); test results are read once. Use --force "
                           "only for debugging, never for numbers that go into the paper")
    log.setdefault("history", []).append({"step": step, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                           "forced": bool(force and step in log)})
    log[step] = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "git": git_info(Path(__file__).resolve().parent)}
    write_json(path, log)


def build_taxonomy(cfg: dict, methods: list[str] | None = None, all_seeds: bool = False) -> dict:
    """Main taxonomies (config seed) on the train + val documents; with
    all_seeds, one per seed in `seeds` for the mean +- std of the automatic metrics."""
    L = Layout(cfg)
    methods = methods or list(cfg["taxonomy"]["methods"])
    corpus = corpus_splits(cfg, D.Corpus.load(L.corpus))["build"]
    out = {}
    for name in methods:
        for seed in (_seeds(cfg) if all_seeds else [None]):
            tag = f"build-taxonomy-{name}" + (f"-seed{seed}" if seed is not None else "")
            run = _run(cfg, tag, [L.corpus / "doc_term.npz", L.tuning / "chosen.json"])
            spec = _method_spec(cfg, name, run.log)
            try:
                tax = _build(cfg, spec, corpus, name, run.log, seed)
            except ImportError as e:
                if not spec.get("optional"):
                    raise
                run.log(f"[taxonomy] skipping optional method {name}: {e}")
                run.finish({"skipped": str(e)})
                break
            tax.save(L.taxonomy(name, seed))
            if seed is None:
                tax.save_txt(L.taxonomies / f"{name}.txt")
            out[tag] = tax.stats()
            run.finish(out[tag])
    return out


def select_params(rows: list[dict], rule: dict) -> tuple[dict, list[dict]]:
    """Pre-registered selection rule: among configurations with enough term
    coverage and nodes, the lowest mean rank over NPMI (higher is better),
    subsumption (higher) and sibling overlap (lower); ties -> grid order."""
    eligible = [r for r in rows if r["term_coverage"] >= rule.get("min_term_coverage", 0.3)
                and r["n_nodes"] >= rule.get("min_nodes", 10)]
    if not eligible:
        raise ValueError("no configuration satisfies the coverage/size constraints; widen the grid")

    def col(name, higher_better):
        v = np.array([r[name] for r in eligible], dtype=float)
        v = np.where(np.isfinite(v), v, -np.inf if higher_better else np.inf)
        return rankdata(-v if higher_better else v, method="average")

    mean_rank = (col("npmi", True) + col("subsumption", True) + col("sibling_overlap", False)) / 3
    for r, mr in zip(eligible, mean_rank):
        r["mean_rank"] = float(mr)
    return eligible[int(np.argmin(mean_rank))], eligible


def tune_taxonomy(cfg: dict, methods: list[str] | None = None) -> dict:
    """Choose construction parameters: build on the train documents, score on
    the disjoint val documents, before any human judgment. The test documents
    are never used here."""
    L, t = Layout(cfg), cfg["taxonomy"]["tuning"]
    splits = corpus_splits(cfg, D.Corpus.load(L.corpus))
    dev, held = splits["train"], splits["val"]
    chosen_path = L.tuning / "chosen.json"
    chosen = read_json(chosen_path) if chosen_path.exists() else {}
    for name in methods or list(t["grids"]):
        run = _run(cfg, f"tune-taxonomy-{name}", [L.corpus / "doc_term.npz"])
        base = cfg["taxonomy"]["methods"][name]
        rows = []
        for params in param_grid(t["grids"][name]):
            spec = copy.deepcopy(base)
            for key, value in params.items():
                _apply_dotted(spec, key, value)
            tax = _build(cfg, spec, dev, name, lambda m: None)
            res = evaluate_intrinsic(held, tax, cfg["evaluation"].get("top_k", 10))
            rows.append({"params": params, **_flat_metrics(res)})
            run.log(f"[tune] {name} {params}: {rows[-1]}")
        record = {"rule": {k: v for k, v in t.items() if k != "grids"},
                  "split": cfg["taxonomy"]["split"], "n_train_docs": dev.n_docs, "n_val_docs": held.n_docs,
                  "rows": rows, "chosen": None}
        try:
            best, _ = select_params(rows, t)
        except ValueError as e:
            write_json(L.tuning / f"{name}.json", record)
            run.finish({"error": str(e)})
            raise ValueError(f"tune-taxonomy {name}: {e} (every configuration is listed in "
                             f"{L.tuning / (name + '.json')})") from e
        chosen[name] = best["params"]
        write_json(L.tuning / f"{name}.json", {**record, "chosen": best})
        run.finish({"chosen": best})
    write_json(chosen_path, chosen)
    return chosen


def _get_dotted(spec: dict, key: str):
    node = spec
    for p in key.split("."):
        if not isinstance(node, dict) or p not in node:
            return None
        node = node[p]
    return node


def sensitivity(cfg: dict) -> list[dict]:
    """One-factor-at-a-time sensitivity of the tuned base method: each listed
    parameter (a method field such as builder.tau, or an encoder field such
    as encoders.sbert_ctx.drop_df) takes each of its values while every other
    parameter stays at the tuned configuration. Built on the train documents
    and scored on val, like tuning; the test documents are not read."""
    L, s = Layout(cfg), cfg["taxonomy"]["sensitivity"]
    splits = corpus_splits(cfg, D.Corpus.load(L.corpus))
    train, val = splits["train"], splits["val"]
    run = _run(cfg, "sensitivity", [L.corpus / "doc_term.npz", L.tuning / "chosen.json"])
    base = _method_spec(cfg, s["base"], run.log)
    rows = []
    for key, values in s["params"].items():
        is_encoder = key.startswith("encoders.")
        center = _get_dotted(cfg if is_encoder else base, key)
        for value in values:
            spec, run_cfg = copy.deepcopy(base), copy.deepcopy(cfg)
            _apply_dotted(run_cfg if is_encoder else spec, key, value)
            tax = _build(run_cfg, spec, train, s["base"], lambda m: None)
            res = evaluate_intrinsic(val, tax, cfg["evaluation"].get("top_k", 10))
            rows.append({"param": key, "value": value, "is_tuned": value == center,
                         **_flat_metrics(res), "npmi_by_level": res["npmi_by_level"]})
            run.log(f"[sensitivity] {key}={value}: {rows[-1]}")
    write_json(L.eval / "sensitivity.json", rows)
    run.finish({"n_runs": len(rows)})
    return rows


def ablation(cfg: dict, force: bool = False) -> list[dict]:
    """Every combination of the listed factors (`encoder` or any builder
    field, e.g. split x clusterer x relation) of the tuned base builder, plus
    the single-factor variants in `extra` (e.g. no UMAP), over all seeds,
    built on train + val and scored on the test documents."""
    L = Layout(cfg)
    grid = cfg["taxonomy"]["ablation"]
    _open_test(cfg, "ablation", force)
    splits = corpus_splits(cfg, D.Corpus.load(L.corpus))
    corpus, test = splits["build"], splits["test"]
    run = _run(cfg, "ablation", [L.corpus / "doc_term.npz", L.tuning / "chosen.json"])
    base = _method_spec(cfg, grid["base"], run.log)
    axes = {k: v for k, v in grid.items() if k not in ("base", "seeds", "extra")}
    extras = grid.get("extra") or []
    keys = list(axes) + [k for e in extras for k in e if k not in axes]
    keys = list(dict.fromkeys(keys))
    defaults = BuilderConfig.from_dict(base.get("builder", {}))

    def base_value(k):
        return base["encoder"] if k == "encoder" else getattr(defaults, k)

    combos = [dict(zip(axes, values)) for values in itertools.product(*axes.values())]
    combos += [dict(e) for e in extras]
    combos = [{k: combo.get(k, base_value(k)) for k in keys} for combo in combos]
    rows = []
    for combo, seed in itertools.product(combos, grid.get("seeds") or _seeds(cfg)):
        name = "abl_" + "_".join(str(combo[k]) for k in keys)
        spec = copy.deepcopy(base)
        spec["encoder"] = combo["encoder"] if "encoder" in combo else spec["encoder"]
        spec.setdefault("builder", {}).update({k: v for k, v in combo.items() if k != "encoder"})
        tax = _build(cfg, spec, corpus, name, lambda m: None, seed)
        tax.save(L.taxonomies / "ablation" / f"{name}@seed{seed}.json")
        res = _flat_metrics(evaluate_intrinsic(test, tax, cfg["evaluation"].get("top_k", 10)))
        rows.append({**combo, "seed": seed, **res})
        run.log(f"[ablation] {name} seed={seed}: {res}")
    summary = _aggregate(rows, keys, ["n_nodes", "npmi", "subsumption", "sibling_overlap", "term_coverage"])
    write_json(L.eval / "ablation_keys.json", keys)
    write_json(L.eval / "ablation.json", rows)
    write_json(L.eval / "ablation_summary.json", summary)
    run.finish({"n_runs": len(rows)})
    return summary


def _item_tokens(cfg: dict, items: list[dict]) -> list[list[str]]:
    return list(D.make_tokenizer(cfg["corpus"].get("tokenizer")).tokenize_many([it["text"] for it in items]))


def taxonomy_names(L: "Layout") -> list[str]:
    """Main taxonomies on disk; fingerprint() leaves <name>.json.sha256.json sidecars next to them."""
    return sorted(p.stem for p in L.taxonomies.glob("*.json") if not p.name.endswith(".sha256.json"))


def evaluate_taxonomies(cfg: dict, names: list[str] | None = None, force: bool = False) -> dict:
    """Automatic metrics on the test documents, which no build or tuning step reads."""
    _open_test(cfg, "eval-taxonomy", force)
    L, ev = Layout(cfg), cfg["evaluation"]
    corpus = corpus_splits(cfg, D.Corpus.load(L.corpus))["test"]
    names = names or taxonomy_names(L)
    run = _run(cfg, "eval-taxonomy", [L.taxonomy(n) for n in names])
    depths = ev.get("depths", [1, 2, 3])
    gold = None
    if ev.get("gold"):
        items = read_jsonl(L.items)
        keep = [it for it in items if len(it["categories"]) > 1]
        gold = (_item_tokens(cfg, keep), [tuple(it["categories"][1:]) for it in keep])

    def evaluate_one(tax: Taxonomy) -> tuple[dict, dict]:
        res = {"intrinsic": evaluate_intrinsic(corpus, tax, ev.get("top_k", 10))}
        flat = _flat_metrics(res["intrinsic"])
        if gold:
            vocab = taxonomy_vocab(tax)
            paths = assign_paths(tax, D.term_counts(gold[0], vocab), vocab, ev.get("assign_top_k", 20))
            res["external"] = evaluate_external(paths, gold[1], depths)
            for d in depths:
                ext = res["external"].get(f"depth{d}", {})
                flat[f"pairF1@{d}"], flat[f"nmi@{d}"] = ext.get("f1"), ext.get("nmi")
        return res, flat

    matched = [m for m in ev.get("size_matched_methods") or [] if m in names]
    n_match = min((Taxonomy.load(L.taxonomy(m)).size for m in matched), default=None)
    summary = {}
    for name in names:
        tax = Taxonomy.load(L.taxonomy(name))
        res, flat = evaluate_one(tax)
        if n_match and name in matched:
            # the main comparison: every method pruned to the smallest one's size
            mres, mflat = evaluate_one(tax.prune_to_size(n_match))
            res["size_matched"] = {"n_nodes": n_match, **mres}
            flat.update({f"{k}@matched_size": v for k, v in mflat.items()})
        seed_rows = [{"seed": int(p.stem.split("@seed")[1]), **evaluate_one(Taxonomy.load(p))[1]}
                     for p in sorted((L.taxonomies / "seeds").glob(f"{name}@seed*.json"))
                     if not p.name.endswith(".sha256.json")]
        if seed_rows:
            metrics = [k for k in seed_rows[0] if k != "seed"]
            res["over_seeds"] = {"rows": seed_rows, "summary": _aggregate(
                [{**r, "method": name} for r in seed_rows], ["method"], metrics)[0]}
        write_json(L.eval / f"{name}.json", res)
        summary[name] = dict(flat)
        if seed_rows:
            summary[name].update({k: v for k, v in res["over_seeds"]["summary"].items()
                                  if k.endswith("_std") or k == "n_seeds"})
        run.log(f"[eval] {name}: {summary[name]}")
    write_json(L.eval / "summary.json", summary)
    run.finish(summary)
    return summary


# ---------------------------------------------------------------- human evaluation

def make_human_eval(cfg: dict, force: bool = False) -> Path:
    L, h = Layout(cfg), cfg["human_eval"]
    if L.frozen_pointer.exists() and not force:
        raise RuntimeError(f"the annotation batch is frozen ({L.frozen_pointer}); a new batch would no "
                           "longer match the answers already collected")
    corpus = corpus_splits(cfg, D.Corpus.load(L.corpus))["build"]   # the documents the taxonomies come from
    run = _run(cfg, "make-human-eval", [L.taxonomy(m) for m in h["methods"]])
    loaded = {m: Taxonomy.load(L.taxonomy(m)) for m in h["methods"]}
    if h.get("size_matched", True):
        # items are sampled from taxonomies of the same size (the larger ones
        # pruned to the smallest), so neither method is judged on a sample of
        # its coarse edges only
        n = min(t.size for t in loaded.values())
        loaded = {m: t.prune_to_size(n) for m, t in loaded.items()}
        run.log(f"[human-eval] taxonomies pruned to {n} nodes")
    taxes = {}
    for m, tax in loaded.items():
        taxes[m], info = rerank(tax, corpus)
        run.log(f"[human-eval] {m}: {info}")
    samples = sample_items(taxes, SampleConfig(h.get("n_relation", 60), h.get("n_intrusion", 50),
                                               h.get("display_k", 5), cfg.get("seed", 0)))
    run.log(f"[human-eval] per-level quotas: {samples['quotas']}")
    out = write_batch(build_batch(samples, cfg["dataset"], seed=cfg.get("seed", 0)), L.human / "batch")
    run.finish({"batch_dir": str(out), "quotas": samples["quotas"]})
    return out


def freeze(cfg: dict) -> Path:
    """Snapshot the batch, the taxonomies it was sampled from and the config
    before any sheet is sent out; later analysis checks the key against it."""
    L, h = Layout(cfg), cfg["human_eval"]
    key = L.human / "batch" / "KEY_DO_NOT_SHARE.json"
    if not key.exists():
        raise FileNotFoundError("run make-human-eval first")
    if L.frozen_pointer.exists():
        raise RuntimeError(f"already frozen: {read_json(L.frozen_pointer)['frozen_dir']}")
    dest = ensure_dir(L.human / "frozen" / time.strftime("%Y%m%d-%H%M%S"))
    for p in (L.human / "batch").iterdir():
        shutil.copy2(p, dest / p.name)
    for m in h["methods"]:
        shutil.copy2(L.taxonomy(m), dest / f"taxonomy_{m}.json")
    write_json(dest / "config.json", cfg)
    files = sorted(p for p in dest.iterdir() if p.name != "manifest.json")
    git = git_info(Path(__file__).resolve().parent)
    write_json(dest / "manifest.json", {"git": git, "files": file_manifest(files)})
    write_json(L.frozen_pointer, {"frozen_dir": str(dest), "key_sha256": sha256_file(key), "git": git,
                                  "created": time.strftime("%Y-%m-%d %H:%M:%S")})
    if not git.get("commit") or git.get("dirty"):
        print("[freeze] WARNING: code is not committed; commit before sending sheets so the frozen "
              "batch can be tied to an exact code version")
    print(f"[freeze] {dest}")
    return dest


def import_forms(cfg: dict, files: list[str]) -> list[Path]:
    """Keep each Google Forms export unchanged under responses_raw/ and write
    the long-format answers analyze-human-eval reads under responses/."""
    L = Layout(cfg)
    raw_dir = ensure_dir(L.human / "responses_raw")
    written = []
    for f in files:
        src = Path(f)
        dest = raw_dir / src.name
        if not dest.exists():
            shutil.copy2(src, dest)
        written += list(import_form_export(dest, L.human / "responses").values())
    write_json(L.human / "responses_raw" / "manifest.json",
               file_manifest(p for p in raw_dir.iterdir() if p.name != "manifest.json"))
    for p in written:
        print(f"[import-forms] {p}")
    return written


def analyze_human_eval(cfg: dict, force: bool = False) -> dict:
    L, h = Layout(cfg), cfg["human_eval"]
    key_path = L.human / "batch" / "KEY_DO_NOT_SHARE.json"
    if L.frozen_pointer.exists():
        if sha256_file(key_path) != read_json(L.frozen_pointer)["key_sha256"]:
            raise RuntimeError("the answer key differs from the frozen one; the batch was regenerated")
    elif not force:
        raise RuntimeError("the batch was never frozen (python -m taxorec freeze) before collecting answers")
    key = load_key(key_path)
    results = {}
    for task in ("relation", "intrusion"):
        pattern = (h.get("responses") or {}).get(task) or str(L.human / "responses" / f"*_{task}.csv")
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"[human-eval] no {task} responses matched {pattern!r}")
            continue
        run = _run(cfg, f"analyze-human-eval-{task}", files)
        run.save_json("manifest.json", file_manifest(files))
        long = load_responses(files, task)
        issues = validate_responses(long, task, [it["item_id"] for it in key[task]])
        for msg in issues:
            run.log(f"[human-eval] {msg}")
        res = analyze(long, task, key, n_boot=h.get("n_boot", 5000), seed=cfg.get("seed", 0),
                      force=force or h.get("force", False))
        res["validation_issues"] = issues
        write_json(L.human / f"results_{task}.json", res)
        results[task] = res
        run.finish({m: v["pooled_accuracy"] for m, v in res["methods"].items()})
    return results


def human_eval_holm(cfgs: list[dict]) -> list[dict]:
    """Holm over the whole primary family at once (every dataset with human
    evaluation x both tasks), after analyze-human-eval has run on each. A
    missing result is an error: adjusting a partial family would understate
    the correction."""
    results = {}
    for cfg in cfgs:
        L = Layout(cfg)
        for task in ("relation", "intrusion"):
            p = L.human / f"results_{task}.json"
            if not p.exists():
                raise FileNotFoundError(f"{p} is missing; run analyze-human-eval for {cfg['dataset']} first")
            results[f"{cfg['dataset']}/{task}"] = read_json(p)
    rows = primary_holm(results)
    for cfg in cfgs:
        write_json(Layout(cfg).human / "primary_holm.json",
                   {"family": sorted(results), "alpha": 0.05, "comparisons": rows})
    for r in rows:
        print(f"[human-eval-holm] {r['result']} {r['comparison']}: diff={r['diff']:.4f} "
              f"p={r['p_value']:.4f} p_holm={r['p_holm']:.4f} -> {r['conclusion']}")
    return rows


# ---------------------------------------------------------------- recommendation

def item_paths(cfg: dict, items: list[dict], names: list[str]) -> dict[str, list[tuple]]:
    L, top_k = Layout(cfg), cfg["rec"].get("taxonomy_top_k", 20)
    tokens = None
    out = {}
    for name in names:
        if name == "gold":
            out[name] = [tuple(it["categories"][1:]) for it in items]
            continue
        tokens = tokens if tokens is not None else _item_tokens(cfg, items)
        tax = Taxonomy.load(L.taxonomy(name))
        vocab = taxonomy_vocab(tax)
        out[name] = assign_paths(tax, D.term_counts(tokens, vocab), vocab, top_k)
    return out


def run_rec(cfg: dict, only: list[str] | None = None) -> dict:
    L, r = Layout(cfg), cfg["rec"]
    run = _run(cfg, "run-rec", [L.split, L.item_emb] + [L.taxonomy(t) for t in r.get("taxonomies", [])
                                                         if t != "gold"])
    split = Split.load(L.split)
    items = read_jsonl(L.items)
    emb = np.load(L.item_emb)
    paths = item_paths(cfg, items, r.get("taxonomies", []))
    for name, p in paths.items():
        run.log(f"[rec] taxonomy {name}: {np.mean([bool(x) for x in p]):.3f} of items assigned a path")

    def component(name: str):
        if name == "sem":
            return ContentKNN(emb)
        if name == "cf":
            return ItemKNN(**r.get("cf_params", {}))
        if name.startswith("tax:"):
            return TaxonomyKNN(paths[name[4:]])
        raise ValueError(f"unknown component {name!r}")

    seed = cfg.get("seed", 0)
    ks = tuple(r.get("ks", [5, 10, 20]))
    results = {}
    for spec in r["models"]:
        if only and spec["name"] not in only:
            continue
        kind = spec["type"]
        if kind == "popularity":
            factory, grid = (lambda p: Popularity()), [{}]
        elif kind == "itemknn":
            factory, grid = (lambda p: ItemKNN(**p)), param_grid(spec.get("grid", {}))
        elif kind == "bpr":
            factory, grid = (lambda p: BPRMF(seed=seed, **p)), param_grid(spec.get("grid", {}))
        elif kind == "hybrid":
            comps = spec["components"]
            grid = simplex_grid(comps, spec.get("weight_step", 0.1))
            factory = (lambda comps: (lambda p: Hybrid({c: component(c) for c in comps}, p)))(comps)
        else:
            raise ValueError(f"unknown model type {kind!r}")
        run.log(f"[rec] {spec['name']}: {len(grid)} configurations")
        res = tune_and_test(factory, grid, split, r.get("metric", "ndcg@10"), ks,
                            max_valid_users=r.get("max_valid_users"), seed=seed, log=run.log)
        ensure_dir(L.results)
        np.savez_compressed(L.results / f"ranks_{spec['name']}.npz",
                            users=res["test_users"], ranks=res["test_ranks"])
        results[spec["name"]] = {k: res[k] for k in ("best_params", "valid", "test", "trials")}
        run.log(f"[rec] {spec['name']} test: {res['test']}")

    ref = r.get("reference")
    if ref and ref in results:
        ref_ranks = np.load(L.results / f"ranks_{ref}.npz")["ranks"]
        tests = []
        for name in results:
            if name == ref:
                continue
            ranks = np.load(L.results / f"ranks_{name}.npz")["ranks"]
            for metric in r.get("significance_metrics", ["ndcg@10", "recall@10"]):
                tests.append({"model": name, "reference": ref, "metric": metric,
                              **paired_bootstrap(per_user_metric(ranks, metric), per_user_metric(ref_ranks, metric),
                                                 n_boot=r.get("n_boot", 10000), seed=seed)})
        for t, adj in zip(tests, holm([t["p_value"] for t in tests])):
            t["p_holm"] = float(adj)
        write_json(L.results / "rec_significance.json", tests)
    write_json(L.results / "rec_summary.json", {n: {"test": v["test"], "best_params": v["best_params"]}
                                                for n, v in results.items()})
    write_json(L.results / "rec_full.json", results)
    run.finish({n: v["test"] for n, v in results.items()})
    return results


# ---------------------------------------------------------------- report & archive

def _md_table(rows: list[dict], cols: list[str]) -> str:
    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else ("" if v is None else str(v))
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(r.get(c)) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def report(cfg: dict) -> Path:
    """Collect every result file of the dataset into one markdown report."""
    L = Layout(cfg)
    parts = [f"# Results: {cfg['dataset']}\n", "Generated by `python -m taxorec report`; "
             "every number below is read from files under " f"`{L.out}`.\n"]
    if (L.corpus / "stats.json").exists():
        parts += ["## Corpus\n", _md_table([read_json(L.corpus / "stats.json")],
                                          ["n_docs", "n_terms", "multiword_terms"]), ""]
    if (L.tuning / "chosen.json").exists():
        rows = [{"method": m, "chosen": str(p)} for m, p in read_json(L.tuning / "chosen.json").items()]
        parts += ["## Tuned construction parameters (dev split)\n", _md_table(rows, ["method", "chosen"]), ""]
    if (L.eval / "summary.json").exists():
        summ = read_json(L.eval / "summary.json")
        rows = [{"method": k, **v} for k, v in summ.items()]
        cols = ["method"] + sorted({c for r in rows for c in r if c != "method"})
        parts += ["## Taxonomy quality (automatic; *_std over seeds)\n", _md_table(rows, cols), ""]
    if (L.eval / "ablation_summary.json").exists():
        rows = read_json(L.eval / "ablation_summary.json")
        keys = read_json(L.eval / "ablation_keys.json") if (L.eval / "ablation_keys.json").exists() \
            else ["encoder", "clusterer", "relation"]
        parts += ["## Ablation (mean over seeds)\n",
                  _md_table(rows, keys + ["n_seeds", "n_nodes", "npmi", "npmi_std",
                                   "subsumption", "sibling_overlap", "term_coverage"]), ""]
    for task in ("relation", "intrusion"):
        p = L.human / f"results_{task}.json"
        if p.exists():
            res = read_json(p)
            rows = [{"method": m, **{k: v for k, v in s.items() if not isinstance(v, (dict, list))}}
                    for m, s in res["methods"].items()]
            parts += [f"## Human evaluation: {task}\n", _md_table(rows, ["method", "n_items", "pooled_accuracy",
                                                                         "majority_accuracy"]),
                      f"\nKrippendorff alpha: {res.get('krippendorff_alpha')} ({res.get('krippendorff_alpha_band')}); "
                      f"Fleiss kappa: {res.get('fleiss_kappa')} ({res.get('fleiss_kappa_band')})\n",
                      "\n".join(f"- {k}: diff={v['diff']:.4f} CI=[{v['ci_low']:.4f}, {v['ci_high']:.4f}] "
                                f"p={v['p_value']:.4f}; annotators favouring each: "
                                f"{v.get('annotators', {}).get('favour_a')}/{v.get('annotators', {}).get('favour_b')} "
                                f"(sign test p={v.get('annotators', {}).get('sign_test_p')})"
                                for k, v in res["comparisons"].items()), ""]
    if (L.human / "primary_holm.json").exists():
        fam = read_json(L.human / "primary_holm.json")
        parts += [f"## Human evaluation: primary family (Holm over {len(fam['comparisons'])})\n",
                  _md_table(fam["comparisons"], ["result", "comparison", "diff", "ci_low", "ci_high",
                                                 "p_value", "p_holm", "conclusion"]), ""]
    if (L.results / "rec_summary.json").exists():
        summ = read_json(L.results / "rec_summary.json")
        rows = [{"model": m, **v["test"]} for m, v in summ.items()]
        cols = ["model"] + [c for c in rows[0] if c != "model"]
        parts += ["## Recommendation (test split)\n", _md_table(rows, cols), ""]
    if (L.results / "rec_significance.json").exists():
        rows = read_json(L.results / "rec_significance.json")
        parts += ["## Paired bootstrap vs reference (Holm-adjusted)\n",
                  _md_table(rows, ["model", "reference", "metric", "diff", "ci_low", "ci_high", "p_holm"]), ""]
    path = L.out / "REPORT.md"
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"[report] {path}")
    return path


ARCHIVE_PATTERNS = [
    "REPORT.md", "corpus/stats.json", "corpus/phrases.json", "corpus/corpus.json",
    "rec/stats.json", "rec/split.npz", "rec/heldout_pairs.csv",
    "taxonomies/*.json", "taxonomies/*.txt", "taxonomies/seeds/*.json", "tuning/*.json",
    "eval/*.json", "results/*.json", "results/*.npz",
    "human_eval/FROZEN.json", "human_eval/batch/*", "human_eval/frozen/*/*",
    "human_eval/responses/*", "human_eval/responses_raw/*", "human_eval/results_*.json",
    "runs/*/*/config.yaml", "runs/*/*/env.json", "runs/*/*/inputs.json", "runs/*/*/status.json",
    "runs/*/*/log.txt", "runs/*/*/manifest.json",
]


def archive(cfg: dict) -> Path:
    """Zip everything needed to check the paper's numbers (no review text, no
    cached models) with a sha256 manifest, ready for Zenodo. Annotator ids
    must already be pseudonymous codes."""
    L = Layout(cfg)
    files = sorted({p for pat in ARCHIVE_PATTERNS for p in L.out.glob(pat) if p.is_file()})
    dest = ensure_dir(L.archive) / f"{cfg['dataset']}-{time.strftime('%Y%m%d-%H%M%S')}.zip"
    manifest = [{"path": p.relative_to(L.out).as_posix(), "size": p.stat().st_size, "sha256": sha256_file(p)}
                for p in files]
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(L.out).as_posix())
        z.writestr("MANIFEST.json", json.dumps({"dataset": cfg["dataset"], "git": git_info(),
                                                "files": manifest}, indent=2))
    print(f"[archive] {dest} ({len(files)} files)")
    return dest
