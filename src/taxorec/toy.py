"""A small synthetic dataset in the Amazon Reviews 2023 file format, with a
known product hierarchy (department > product type) used as gold categories.
It exists only to smoke-test the pipeline end to end in about a minute
(`python -m taxorec make-toy-data`, then the toy config). It is not, and
must never be reported as, experimental data."""
from __future__ import annotations

import gzip
import json
import random
from pathlib import Path

from taxorec.common import ensure_dir

TREE = {
    "footwear": {
        "running shoe": ["mesh", "cushion", "jogging", "marathon", "trail", "breathable", "lightweight", "grip"],
        "leather boot": ["ankle", "waterproof", "lace", "heel", "winter", "suede", "zipper", "rugged"],
        "beach sandal": ["flip", "flop", "strap", "slide", "summer", "pool", "foam", "toe"],
    },
    "bag": {
        "school backpack": ["laptop", "compartment", "padded", "student", "zipper", "pocket", "travel", "strap"],
        "leather handbag": ["tote", "shoulder", "purse", "clutch", "crossbody", "designer", "gold", "chain"],
        "slim wallet": ["card", "holder", "rfid", "bifold", "coin", "cash", "billfold", "minimalist"],
    },
    "dress": {
        "evening dress": ["gown", "formal", "sequin", "prom", "cocktail", "satin", "elegant", "maxi"],
        "summer dress": ["floral", "sundress", "cotton", "casual", "beach", "sleeveless", "linen", "breezy"],
        "wedding dress": ["bridal", "lace", "veil", "train", "ivory", "tulle", "bride", "ceremony"],
    },
}
DEPARTMENT_WORDS = {"footwear": ["shoe", "sole", "insole", "fit"],
                    "bag": ["bag", "carry", "storage", "handle"],
                    "dress": ["dress", "fabric", "waist", "hem"]}
GENERIC = ["great", "love", "quality", "price", "nice", "recommend", "perfect", "happy", "size",
           "color", "product", "order", "arrived", "time", "good", "really", "value", "buy"]


def make_toy_data(out_dir, n_items_per_type: int = 30, n_users: int = 600, seed: int = 0) -> dict[str, Path]:
    rng = random.Random(seed)
    out_dir = ensure_dir(out_dir)
    items = []
    for dept, types in TREE.items():
        for ptype, attrs in types.items():
            for k in range(n_items_per_type):
                item_id = f"T{len(items):05d}"
                words = rng.sample(attrs, 3)
                items.append({"item_id": item_id, "dept": dept, "ptype": ptype, "attrs": words,
                              "title": f"{' '.join(words[:2])} {ptype} {rng.choice(DEPARTMENT_WORDS[dept])}"})
    by_type: dict[str, list[dict]] = {}
    for it in items:
        by_type.setdefault(it["ptype"], []).append(it)

    meta_path = out_dir / "meta_Toy.jsonl.gz"
    with gzip.open(meta_path, "wt", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps({
                "parent_asin": it["item_id"], "title": it["title"], "store": f"store{rng.randrange(20)}",
                "features": [f"{w} {it['ptype']}" for w in it["attrs"]],
                "description": [f"A {it['attrs'][2]} {it['ptype']} for every day."],
                "categories": ["Toy Store", it["dept"].title(), it["ptype"].title()],
                "main_category": "Toy",
            }) + "\n")

    reviews_path = out_dir / "Toy.jsonl.gz"
    types = list(by_type)
    with gzip.open(reviews_path, "wt", encoding="utf-8") as f:
        for u in range(n_users):
            home = rng.choice(types)
            dept = next(d for d, ts in TREE.items() if home in ts)
            t = 1_600_000_000_000 + rng.randrange(10 ** 9)
            for _ in range(rng.randint(4, 10)):
                r = rng.random()
                if r < 0.6:
                    pool = by_type[home]
                elif r < 0.85:
                    pool = by_type[rng.choice(list(TREE[dept]))]
                else:
                    pool = by_type[rng.choice(types)]
                it = rng.choice(pool)
                t += rng.randrange(10 ** 6, 10 ** 8)
                attrs = TREE[it["dept"]][it["ptype"]]
                body = (rng.sample(GENERIC, 8) + rng.sample(attrs, 4) + it["ptype"].split() * 2
                        + rng.sample(DEPARTMENT_WORDS[it["dept"]], 2) + rng.sample(GENERIC, 8)
                        + rng.sample(attrs, 3) + it["ptype"].split())
                f.write(json.dumps({
                    "user_id": f"U{u:05d}", "parent_asin": it["item_id"], "asin": it["item_id"],
                    "rating": float(rng.choice([3, 4, 5, 5, 5])), "timestamp": t,
                    "verified_purchase": rng.random() < 0.9, "helpful_vote": 0,
                    "title": f"{rng.choice(GENERIC)} {it['ptype']}", "text": " ".join(body),
                }) + "\n")
    return {"reviews": reviews_path, "meta": meta_path}
