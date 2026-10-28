"""Human evaluation of taxonomies (relation accuracy, term intrusion).

1. sample_items      level-matched samples: the same number of edges/nodes per
                     method AND per depth, so a small, shallow taxonomy is not
                     compared with a deep one on easier items.
2. build_batch       blinded, shuffled annotator sheets + a private key file.
3. load_responses    long-format answers (annotator_id, item_id, answer,
                     submitted_at) + a sha256 manifest of the raw exports.
4. integrity_report  plausibility checks run BEFORE any analysis; critical
                     flags (e.g. every annotator giving the same number of
                     'yes' answers) stop the analysis.
5. analyze           pooled/majority accuracy, agreement, and a two-way
                     (item x annotator) bootstrap for method differences.
"""
from __future__ import annotations

import hashlib
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from taxorec.common import ensure_dir, read_json, write_json
from taxorec.stats import (alpha_label, fleiss_kappa, holm, kappa_label, krippendorff_alpha_nominal,
                           sign_test, two_way_bootstrap_diff, wilson_ci)
from taxorec.taxonomy.tree import Taxonomy

RELATION_CHECKS = [
    ("running shoe, trail shoe, sneaker, hiking boot, sandal",
     "footwear, shoe, boot, heel, slipper", "yes"),
    ("violin, cello, viola, double bass, harp",
     "kitchen, oven, refrigerator, microwave, toaster", "no"),
    ("python, java, javascript, rust, golang",
     "programming language, compiler, syntax, interpreter, source code", "yes"),
    ("tulip, rose, orchid, daisy, lily",
     "volcano, earthquake, tsunami, landslide, hurricane", "no"),
]
INTRUSION_CHECKS = [
    ["apple", "banana", "orange", "grape", "mango", "bicycle"],
    ["car", "truck", "motorcycle", "bus", "van", "spaghetti"],
    ["doctor", "nurse", "surgeon", "hospital", "pharmacist", "guitar"],
]


# ---------------------------------------------------------------- sampling

@dataclass
class SampleConfig:
    n_relation: int = 60
    n_intrusion: int = 50
    display_k: int = 5
    seed: int = 0


def level_quotas(available: list[Counter], n: int) -> dict[int, int]:
    """Per-depth quotas common to all methods: allocated in proportion to the
    smallest per-level availability and capped by it."""
    levels = sorted(set.intersection(*(set(c) for c in available))) if available else []
    cap = {lvl: min(c[lvl] for c in available) for lvl in levels}
    total = sum(cap.values())
    if total <= n:
        return cap
    raw = {lvl: n * cap[lvl] / total for lvl in levels}
    quota = {lvl: int(raw[lvl]) for lvl in levels}
    for lvl in sorted(levels, key=lambda lv: (-(raw[lv] - quota[lv]), lv))[: n - sum(quota.values())]:
        quota[lvl] += 1
    return quota


def _judgeable_edges(tax: Taxonomy, k: int):
    return [(p, c) for p, c in tax.edges(include_root=False) if len(p.terms) >= 1 and len(c.terms) >= 1]


def _intrusion_candidates(tax: Taxonomy, k: int):
    out = []
    for node in tax.non_root():
        if len(node.terms) < k:
            continue
        own = set(node.terms)
        pool = sorted({t for s in tax.siblings(node.id) for t in s.terms[:k] if t not in own})
        if pool:
            out.append((node, pool))
    return out


def sample_items(taxonomies: dict[str, Taxonomy], cfg: SampleConfig) -> dict:
    rng = random.Random(cfg.seed)
    k = cfg.display_k
    edges = {m: _judgeable_edges(t, k) for m, t in taxonomies.items()}
    rel_quota = level_quotas([Counter(c.level for _, c in e) for e in edges.values()], cfg.n_relation)
    cands = {m: _intrusion_candidates(t, k) for m, t in taxonomies.items()}
    int_quota = level_quotas([Counter(n.level for n, _ in c) for c in cands.values()], cfg.n_intrusion)

    relation, intrusion = {}, {}
    for method in sorted(taxonomies):
        rel = []
        for lvl, q in rel_quota.items():
            pool = sorted([e for e in edges[method] if e[1].level == lvl], key=lambda e: e[1].id)
            for parent, child in rng.sample(pool, q):
                # the parent is described by its best terms not already shown for the
                # child, so the two lists can never be trivially identical
                shown = set(child.terms[:k])
                parent_terms = [t for t in parent.terms if t not in shown][:k]
                rel.append({"method": method, "node_id": child.id, "level": lvl,
                            "child_terms": child.terms[:k], "parent_terms": parent_terms})
        relation[method] = rel
        intr = []
        for lvl, q in int_quota.items():
            pool = sorted([c for c in cands[method] if c[0].level == lvl], key=lambda c: c[0].id)
            for node, intruders in rng.sample(pool, q):
                shown = list(node.terms[:k])
                intruder = rng.choice(intruders)
                pos = rng.randrange(k + 1)
                shown.insert(pos, intruder)
                intr.append({"method": method, "node_id": node.id, "level": lvl,
                             "terms": shown, "intruder": intruder, "intruder_index": pos})
        intrusion[method] = intr
    return {"relation": relation, "intrusion": intrusion,
            "quotas": {"relation": rel_quota, "intrusion": int_quota}}


# ---------------------------------------------------------------- batch

def _pretty(terms) -> str:
    return ", ".join(t.replace("_", " ") for t in terms)


def build_batch(samples: dict, dataset: str, seed: int = 0, with_checks: bool = True) -> dict:
    rng = random.Random(f"{dataset}:{seed}")
    methods = sorted(samples["relation"])
    codes = [f"S{i + 1}" for i in range(len(methods))]
    rng.shuffle(codes)
    code_of = dict(zip(methods, codes))

    rel = [{**it, "code": code_of[m], "is_check": False, "expected": None}
           for m in methods for it in samples["relation"][m]]
    if with_checks:
        rel += [{"method": None, "code": None, "node_id": None, "level": None, "is_check": True,
                 "child_terms": c.split(", "), "parent_terms": p.split(", "), "expected": a}
                for c, p, a in RELATION_CHECKS]
    rng.shuffle(rel)
    for i, it in enumerate(rel):
        it["item_id"] = f"R{i + 1:03d}"

    intr = [{**it, "code": code_of[m], "is_check": False} for m in methods for it in samples["intrusion"][m]]
    if with_checks:
        for terms in INTRUSION_CHECKS:
            shown = list(terms)
            rng.shuffle(shown)
            intr.append({"method": None, "code": None, "node_id": None, "level": None, "is_check": True,
                         "terms": shown, "intruder": terms[-1], "intruder_index": shown.index(terms[-1])})
    rng.shuffle(intr)
    for i, it in enumerate(intr):
        it["item_id"] = f"I{i + 1:03d}"
    return {"dataset": dataset, "codes": code_of, "relation": rel, "intrusion": intr}


INSTRUCTIONS = """# Hướng dẫn chấm / Annotation instructions ({dataset})

Làm độc lập, trong một buổi. Không trao đổi về các câu hỏi với người chấm khác.
Bạn không cần biết các cụm từ được tạo ra bằng phương pháp nào.
Work alone, in one sitting. Do not discuss items with other annotators.
You do not need to know how the terms were produced.

## Phần 1 / Part 1: quan hệ cha-con / parent-child relation
Mỗi câu có một nhóm từ CON (child) và một nhóm từ CHA (parent).
Chọn `yes` nếu nhóm con mô tả một chủ đề hẹp hơn, thuộc về chủ đề của nhóm cha;
ngược lại chọn `no`.
Each item shows a CHILD term group and a PARENT term group. Answer `yes` if the
child group describes a more specific topic that belongs under the parent
topic; otherwise `no`.

## Phần 2 / Part 2: từ lạc đề / term intrusion
Mỗi câu có sáu cụm từ: năm cụm cùng một chủ đề, một cụm không cùng chủ đề.
Chọn số thứ tự (1-6) của cụm từ lạc đề.
Each item shows six terms. Five share one topic; one does not. Answer with the
number (1-6) of the term that does not belong.

Nếu làm trên file CSV / If you use the CSV sheets: chỉ điền cột `answer`, đặt
tên file `<mã người chấm>_relation.csv` và `<mã người chấm>_intrusion.csv`.
"""

SECTION_TITLES = {
    "relation": "Phần 1: Quan hệ cha-con / Part 1: Parent-child relations",
    "intrusion": "Phần 2: Từ lạc đề / Part 2: Term intrusion",
}


def forms_rows(batch: dict) -> list[dict]:
    """One row per question for tools/google_forms/create_form.gs. Titles start
    with the item id, which is how exports are mapped back to items."""
    rows = []
    for it in batch["relation"]:
        rows.append({"task": "relation", "item_id": it["item_id"], "section_title": SECTION_TITLES["relation"],
                     "title": f"{it['item_id']}. CON / CHILD: {_pretty(it['child_terms'])}",
                     "help": f"CHA / PARENT: {_pretty(it['parent_terms'])}  —  Nhóm con có thuộc chủ đề "
                             f"của nhóm cha không? / Does the child belong under the parent?",
                     "choices": "yes|no"})
    for it in batch["intrusion"]:
        rows.append({"task": "intrusion", "item_id": it["item_id"], "section_title": SECTION_TITLES["intrusion"],
                     "title": f"{it['item_id']}. Cụm từ nào lạc đề? / Which term does not belong?",
                     "help": "",
                     "choices": "|".join(f"{j + 1}. {t.replace('_', ' ')}" for j, t in enumerate(it["terms"]))})
    return rows


def write_batch(batch: dict, out_dir) -> Path:
    out_dir = ensure_dir(out_dir)
    pd.DataFrame([{"item_id": it["item_id"], "child_terms": _pretty(it["child_terms"]),
                   "parent_terms": _pretty(it["parent_terms"]), "answer": ""}
                  for it in batch["relation"]]).to_csv(out_dir / "relation_sheet.csv", index=False)
    pd.DataFrame([{"item_id": it["item_id"],
                   **{f"term_{j + 1}": t.replace("_", " ") for j, t in enumerate(it["terms"])},
                   "answer": ""} for it in batch["intrusion"]]).to_csv(out_dir / "intrusion_sheet.csv", index=False)
    pd.DataFrame(forms_rows(batch)).to_csv(out_dir / f"forms_{batch['dataset']}.csv", index=False)
    (out_dir / "INSTRUCTIONS.md").write_text(INSTRUCTIONS.format(dataset=batch["dataset"]), encoding="utf-8")
    write_json(out_dir / "KEY_DO_NOT_SHARE.json", batch)
    return out_dir


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def form_timestamp(raw: str) -> str:
    """Google Forms writes timestamps in the form owner's locale, e.g.
    '2026/10/27 1:29:59 CH GMT+7' (Vietnamese SA = AM, CH = PM). Returns
    ISO 8601, or '' when the text is not a timestamp."""
    import re
    m = re.match(r"\s*(\d{4})/(\d{1,2})/(\d{1,2})\s+(\d{1,2}):(\d{2}):(\d{2})\s*(SA|CH|AM|PM)?"
                 r"\s*(?:GMT([+-]\d{1,2})(?::?(\d{2}))?)?\s*$", raw or "", re.IGNORECASE)
    if not m:
        return ""
    y, mo, d, h, mi, s, half, tz_h, tz_m = m.groups()
    h = int(h)
    if half:
        h = h % 12 + (12 if half.upper() in ("CH", "PM") else 0)
    out = f"{int(y):04d}-{int(mo):02d}-{int(d):02d}T{h:02d}:{int(mi):02d}:{int(s):02d}"
    return out + (f"{int(tz_h):+03d}:{int(tz_m or 0):02d}" if tz_h else "")


def import_form_export(path, out_dir, prefix: str | None = None) -> dict[str, Path]:
    """Convert a Google Forms response export (one row per annotator, one
    column per question) into long CSVs (annotator_id,item_id,answer,submitted_at)
    per task. The raw export itself must be kept unchanged next to them.
    Rows of people who declined consent (the form ends there) are dropped."""
    import re
    df = pd.read_csv(path, dtype=str).fillna("")
    cols = list(df.columns)
    ann_col = next((c for c in cols if "annotator" in c.lower() or "mã người chấm" in c.lower()), None)
    ts_col = next((c for c in cols if c.lower() in ("timestamp", "dấu thời gian")), None)
    consent_col = next((c for c in cols if "consent" in c.lower() or "đồng ý tham gia" in c.lower()), None)
    if ann_col is None:
        raise ValueError("no annotator-id column found in the export")
    if consent_col is not None:
        declined = df[consent_col].str.strip().str.lower().str.startswith(("không", "i do not", "no"))
        df = df[~declined]
    df = df[df[ann_col].str.strip() != ""]
    item_cols = {c: m.group(1) for c in cols if (m := re.match(r"\s*([RI]\d{3})\b", c))}
    rows = {"relation": [], "intrusion": []}
    for _, r in df.iterrows():
        submitted = form_timestamp(r[ts_col]) if ts_col else ""
        for col, item_id in item_cols.items():
            raw = r[col].strip()
            if item_id.startswith("R"):
                low = raw.lower()
                ans = "yes" if low.startswith(("yes", "có")) else "no" if low.startswith(("no", "không")) else low
                rows["relation"].append((r[ann_col].strip(), item_id, ans, submitted))
            else:
                m = re.match(r"\s*(\d)", raw)
                rows["intrusion"].append((r[ann_col].strip(), item_id, m.group(1) if m else raw, submitted))
    out_dir = ensure_dir(out_dir)
    prefix = prefix or f"form-{Path(path).stem}"
    written = {}
    for task, data in rows.items():
        if data:
            p = out_dir / f"{prefix}_{task}.csv"
            pd.DataFrame(data, columns=["annotator_id", "item_id", "answer", "submitted_at"]).to_csv(p, index=False)
            written[task] = p
    return written


# ---------------------------------------------------------------- responses

def file_manifest(paths) -> list[dict]:
    out = []
    for p in sorted(Path(x) for x in paths):
        out.append({"file": p.name, "size": p.stat().st_size,
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    return out


def load_responses(paths, task: str) -> pd.DataFrame:
    """Accepts long CSVs (annotator_id,item_id,answer[,submitted_at]) or filled
    sheets named '<annotator>_<task>.csv'."""
    frames = []
    for p in sorted(Path(x) for x in paths):
        df = pd.read_csv(p, dtype=str).fillna("")
        if "annotator_id" not in df.columns:
            df = df[["item_id", "answer"]].assign(annotator_id=p.stem.rsplit("_", 1)[0])
        frames.append(df)
    long = pd.concat(frames, ignore_index=True)
    long["answer"] = long["answer"].str.strip().str.lower()
    long["task"] = task
    return long


def validate_responses(long: pd.DataFrame, task: str, item_ids: list[str]) -> list[str]:
    valid = {"yes", "no"} if task == "relation" else {str(i) for i in range(1, 7)}
    issues = []
    expected = set(item_ids)
    for ann, grp in long.groupby("annotator_id"):
        missing = expected - set(grp["item_id"])
        if missing:
            issues.append(f"{ann}: {len(missing)} items unanswered")
        bad = grp[~grp["answer"].isin(valid)]
        if len(bad):
            issues.append(f"{ann}: {len(bad)} invalid answers, e.g. {bad['answer'].iloc[0]!r}")
        dup = grp["item_id"].duplicated().sum()
        if dup:
            issues.append(f"{ann}: {dup} duplicated items")
    return issues


# ---------------------------------------------------------------- integrity

def integrity_report(long: pd.DataFrame, task: str, key: dict, min_annotators: int = 8) -> dict:
    """Checks that genuine independent annotators pass and fabricated or
    copied data typically fail. Under independence (given each item's answer
    distribution), the number of times annotator a gives category c has
    variance sum_i p_ic (1 - p_ic); real annotators, who also differ in
    leniency, show at least that much spread. A ratio far below 1 means the
    marginals were constrained, which no real annotation process does."""
    items = key["relation"] if task == "relation" else key["intrusion"]
    checks = {it["item_id"]: it for it in items if it["is_check"]}
    real_ids = [it["item_id"] for it in items if not it["is_check"]]
    wide = long.pivot_table(index="annotator_id", columns="item_id", values="answer", aggfunc="first")
    n_ann = len(wide)
    flags, report = [], {"n_annotators": n_ann, "n_items": len(real_ids)}

    real = wide.reindex(columns=real_ids)
    cats = sorted(set(real.stack().unique()))

    # identical answer vectors: compare the observed number of identical pairs
    # with its expectation under independence, sum over pairs of
    # prod_i sum_c p_ic^2 (high-agreement items make chance matches likely)
    vectors = real.fillna("").apply(tuple, axis=1)
    dup_groups = [list(g.index) for _, g in vectors.groupby(vectors) if len(g) > 1]
    observed_pairs = sum(len(g) * (len(g) - 1) // 2 for g in dup_groups)
    match_prob = float(np.prod([sum(float((real[i] == c).mean()) ** 2 for c in cats) for i in real_ids]))
    expected_pairs = n_ann * (n_ann - 1) / 2 * match_prob
    report["identical_answer_vectors"] = dup_groups
    report["identical_pairs_expected"] = expected_pairs
    if observed_pairs and observed_pairs > 10 * expected_pairs + 1:
        flags.append(("critical", f"{observed_pairs} pairs of annotators gave identical answers "
                                  f"(about {expected_pairs:.3g} expected by chance)"))
    elif observed_pairs:
        flags.append(("warning", f"{observed_pairs} pairs of annotators gave identical answers"))
    ratios = {}
    for c in cats:
        hits = (real == c)
        per_ann = hits.sum(axis=1).astype(float)
        p = hits.mean(axis=0)
        expected = float((p * (1 - p)).sum())
        ratios[c] = float(per_ann.var(ddof=1) / expected) if expected > 0 and n_ann > 1 else float("nan")
    report["marginal_dispersion_ratio"] = ratios
    count_vectors = {tuple(int((real.loc[a] == c).sum()) for c in cats) for a in real.index}
    report["distinct_count_vectors"] = len(count_vectors)
    finite = [v for v in ratios.values() if np.isfinite(v)]
    if n_ann >= min_annotators and finite and float(np.mean(finite)) < 0.25:
        flags.append(("critical", f"answer counts per annotator are implausibly uniform "
                                  f"(dispersion ratio {np.mean(finite):.3f}, expected >= ~1)"))
    if n_ann >= 5 and len(count_vectors) == 1:
        flags.append(("critical", "every annotator has exactly the same answer counts"))

    # pre-registered rule (DE_CUONG_DANH_GIA.md 6.4): two or more wrong attention
    # checks exclude the annotator from this task; one slip does not
    failed = {}
    for ann in wide.index:
        if not checks:
            break
        correct = 0
        for item_id, it in checks.items():
            ans = wide.loc[ann].get(item_id)
            truth = it["expected"] if task == "relation" else str(it["intruder_index"] + 1)
            correct += int(ans == truth)
        if len(checks) - correct >= 2:
            failed[ann] = correct / len(checks)
    report["failed_attention"] = failed

    straight = [a for a in real.index if real.loc[a].value_counts(normalize=True).iloc[0] >= 0.95]
    report["straight_liners"] = straight
    if straight:
        flags.append(("warning", f"{len(straight)} annotators gave (nearly) the same answer to everything"))

    if "submitted_at" in long.columns:
        ts = pd.to_datetime(long.groupby("annotator_id")["submitted_at"].first(), errors="coerce",
                            format="ISO8601", utc=True).dropna()
        if len(ts) >= 5:
            gaps = ts.sort_values().diff().dt.total_seconds().dropna()
            close = float((gaps < 2).mean())
            report["submissions_within_2s"] = close
            if close > 0.5:
                flags.append(("critical", "most submissions arrived within 2 seconds of each other"))
    report["flags"] = flags
    report["ok"] = not any(sev == "critical" for sev, _ in flags)
    return report


# ---------------------------------------------------------------- analysis

def correctness_table(long: pd.DataFrame, task: str, key: dict, exclude=()) -> pd.DataFrame:
    items = {it["item_id"]: it for it in key[task] if not it["is_check"]}
    df = long[long["item_id"].isin(items) & ~long["annotator_id"].isin(exclude)].copy()
    df["method"] = df["item_id"].map(lambda i: items[i]["method"])
    df["level"] = df["item_id"].map(lambda i: items[i]["level"])
    if task == "relation":
        df["correct"] = (df["answer"] == "yes").astype(int)
    else:
        truth = {i: str(it["intruder_index"] + 1) for i, it in items.items()}
        df["correct"] = (df["answer"] == df["item_id"].map(truth)).astype(int)
    return df[["annotator_id", "item_id", "method", "level", "correct"]]


def _matrix(df: pd.DataFrame, annotators: list[str]) -> np.ndarray:
    return df.pivot_table(index="item_id", columns="annotator_id", values="correct",
                          aggfunc="first").reindex(columns=annotators).to_numpy(dtype=float)


def analyze(long: pd.DataFrame, task: str, key: dict, n_boot: int = 5000, seed: int = 0,
            force: bool = False) -> dict:
    integrity = integrity_report(long, task, key)
    if not integrity["ok"] and not force:
        raise ValueError(f"integrity check failed, refusing to analyze: {integrity['flags']}")
    table = correctness_table(long, task, key, exclude=integrity["failed_attention"])
    annotators = sorted(table["annotator_id"].unique())
    out = {"task": task, "integrity": integrity, "n_annotators": len(annotators), "methods": {}}
    for method, grp in table.groupby("method"):
        k, n = int(grp["correct"].sum()), len(grp)
        per_item = grp.groupby("item_id")["correct"].mean()
        out["methods"][method] = {
            "n_items": int(grp["item_id"].nunique()), "n_votes": n,
            "pooled_accuracy": k / n, "wilson_ci_descriptive": wilson_ci(k, n),
            "majority_accuracy": float((per_item > 0.5).mean()),
            "by_level": {int(lvl): float(g["correct"].mean()) for lvl, g in grp.groupby("level")},
        }
    counts = table.groupby("item_id")["correct"].agg(["sum", "count"])
    # agreement on the binary correctness of each judgment, with the bands
    # fixed in docs/human_eval/DE_CUONG_DANH_GIA.md before the forms went out
    if counts["count"].nunique() == 1 and len(counts):
        out["fleiss_kappa"] = fleiss_kappa(np.column_stack([counts["sum"], counts["count"] - counts["sum"]]))
        out["fleiss_kappa_band"] = kappa_label(out["fleiss_kappa"])
    wide = table.pivot_table(index="annotator_id", columns="item_id", values="correct", aggfunc="first")
    out["krippendorff_alpha"] = krippendorff_alpha_nominal(wide.to_numpy(dtype=float))
    out["krippendorff_alpha_band"] = alpha_label(out["krippendorff_alpha"])
    per_annotator = table.groupby(["annotator_id", "method"])["correct"].mean().unstack()
    methods = sorted(out["methods"])
    out["comparisons"] = {}
    for i, a in enumerate(methods):
        for b in methods[i + 1:]:
            res = two_way_bootstrap_diff(
                _matrix(table[table["method"] == a], annotators),
                _matrix(table[table["method"] == b], annotators), n_boot=n_boot, seed=seed)
            # secondary: how many annotators judged a better than b
            d = (per_annotator[a] - per_annotator[b]).dropna()
            wins, losses = int((d > 0).sum()), int((d < 0).sum())
            res["annotators"] = {"favour_a": wins, "favour_b": losses, "ties": int((d == 0).sum()),
                                 "sign_test_p": sign_test(wins, losses)}
            out["comparisons"][f"{a} vs {b}"] = res
    return out


def primary_holm(results: dict[str, dict], alpha: float = 0.05) -> list[dict]:
    """The pre-registered primary family (DE_CUONG_DANH_GIA.md 6.1): every
    method comparison of every dataset and task, Holm-adjusted together.
    `results` maps a label such as 'dblp/relation' to the output of `analyze`.
    A comparison counts as 'a better' only with a positive difference and an
    adjusted p below alpha; anything else is 'not established', never 'equal'."""
    rows = [{"result": label, "comparison": name,
             **{k: v[k] for k in ("acc_a", "acc_b", "diff", "ci_low", "ci_high", "p_value")},
             "integrity_flags": res["integrity"]["flags"]}
            for label, res in results.items() for name, v in res["comparisons"].items()]
    for row, adj in zip(rows, holm([r["p_value"] for r in rows]) if rows else []):
        row["p_holm"] = float(adj)
        row["conclusion"] = "a better" if row["diff"] > 0 and adj < alpha else "not established"
    return rows


def load_key(path) -> dict:
    return read_json(path)
