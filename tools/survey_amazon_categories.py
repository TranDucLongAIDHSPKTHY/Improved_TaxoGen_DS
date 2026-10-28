"""Survey Amazon Reviews 2023 categories for a fourth dataset with a gold
category tree, applying the rules fixed in docs/CHON_BO_DU_LIEU_THU_4.md.

Rule D is a fixed list; rule S reads only the Content-Length of each review
file (HTTP HEAD); rule G streams the metadata file of each category that
passes D and S and reads only its `categories` field. Nothing is saved but
the survey table, and no method is run.

    python tools/survey_amazon_categories.py
"""
from __future__ import annotations

import gzip
import io
import json
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime

from taxorec.common import ensure_dir, write_json
from taxorec.data import AMAZON_2023_URL

CATEGORIES = [
    "All_Beauty", "Amazon_Fashion", "Appliances", "Arts_Crafts_and_Sewing", "Automotive",
    "Baby_Products", "Beauty_and_Personal_Care", "Books", "CDs_and_Vinyl",
    "Cell_Phones_and_Accessories", "Clothing_Shoes_and_Jewelry", "Digital_Music", "Electronics",
    "Gift_Cards", "Grocery_and_Gourmet_Food", "Handmade_Products", "Health_and_Household",
    "Health_and_Personal_Care", "Home_and_Kitchen", "Industrial_and_Scientific", "Kindle_Store",
    "Magazine_Subscriptions", "Movies_and_TV", "Musical_Instruments", "Office_Products",
    "Patio_Lawn_and_Garden", "Pet_Supplies", "Software", "Sports_and_Outdoors",
    "Subscription_Boxes", "Tools_and_Home_Improvement", "Toys_and_Games", "Unknown", "Video_Games",
]
EXCLUDED = {
    "Amazon_Fashion": "already used", "Handmade_Products": "already used",
    "Clothing_Shoes_and_Jewelry": "same domain as Fashion",
    "Arts_Crafts_and_Sewing": "same domain as Handmade",
    **{c: "content/digital or not a product" for c in (
        "Books", "Kindle_Store", "Movies_and_TV", "CDs_and_Vinyl", "Digital_Music", "Video_Games",
        "Software", "Magazine_Subscriptions", "Gift_Cards", "Subscription_Boxes", "Unknown")},
}
SIZE_MIN, SIZE_MAX = 77_803_288, 292_425_439     # Handmade and Fashion review files (bytes)
G1, G2, G3_SHARE, G3_RANGE, G4 = 0.80, 0.80, 0.01, (5, 50), 0.50


def _open(url: str, method: str = "GET"):
    return urllib.request.urlopen(urllib.request.Request(url, method=method,
                                                         headers={"User-Agent": "taxorec"}))


def review_size(category: str) -> int | None:
    try:
        with _open(f"{AMAZON_2023_URL}/review_categories/{category}.jsonl.gz", "HEAD") as r:
            return int(r.headers["Content-Length"])
    except urllib.error.HTTPError:
        return None


def tree_stats(category: str) -> dict:
    """Gold-tree statistics, with the department level dropped as evaluation does."""
    n = usable = deep = 0
    top = Counter()
    with _open(f"{AMAZON_2023_URL}/meta_categories/meta_{category}.jsonl.gz") as r, \
            io.TextIOWrapper(gzip.GzipFile(fileobj=r), encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            n += 1
            cats = [c for c in (json.loads(line).get("categories") or []) if c][1:]
            if cats:
                usable += 1
                deep += len(cats) >= 2
                top[cats[0]] += 1
    big = [c for c, k in top.items() if k >= G3_SHARE * usable] if usable else []
    return {"n_items": n,
            "g1_usable_share": usable / n if n else 0.0,
            "g2_depth2_share": deep / usable if usable else 0.0,
            "g3_n_level1_ge_1pct": len(big),
            "g4_largest_level1_share": max(top.values()) / usable if usable else 1.0,
            "largest_level1": top.most_common(1)[0][0] if top else None}


def main() -> None:
    rows = []
    for c in CATEGORIES:
        row = {"category": c, "review_bytes": review_size(c)}
        if c in EXCLUDED:
            row["result"] = f"D: {EXCLUDED[c]}"
        elif row["review_bytes"] is None:
            row["result"] = "S: review file not found"
        elif not SIZE_MIN <= row["review_bytes"] <= SIZE_MAX:
            row["result"] = "S: size out of range"
        else:
            print(f"[survey] reading categories of {c} ...", flush=True)
            row.update(tree_stats(c))
            fails = [name for name, ok in (
                ("G1", row["g1_usable_share"] >= G1), ("G2", row["g2_depth2_share"] >= G2),
                ("G3", G3_RANGE[0] <= row["g3_n_level1_ge_1pct"] <= G3_RANGE[1]),
                ("G4", row["g4_largest_level1_share"] <= G4)) if not ok]
            row["result"] = "pass" if not fails else "G: fails " + ",".join(fails)
        print(f"[survey] {c}: {row['result']}", flush=True)
        rows.append(row)
    passed = [r for r in rows if r["result"] == "pass"]
    chosen = max(passed, key=lambda r: r["review_bytes"])["category"] if passed else None
    out = ensure_dir("outputs/dataset_survey") / f"survey_{datetime.now():%Y%m%d-%H%M%S}.json"
    write_json(out, {"rules": "docs/CHON_BO_DU_LIEU_THU_4.md", "chosen": chosen, "rows": rows})
    print(f"[survey] chosen: {chosen}; table: {out}")


if __name__ == "__main__":
    main()
