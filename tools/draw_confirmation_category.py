"""Draw the e-commerce confirmation category by the rule fixed in
docs/CHON_BO_XAC_NHAN.md (section 2): rules D and S on the Amazon Reviews 2023
review-file sizes, then one draw with numpy.random.default_rng(20261028).

    python tools/draw_confirmation_category.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SURVEY = Path("outputs/dataset_survey/survey_20261028-080610.json")
EXCLUDED = {
    "Amazon_Fashion", "Handmade_Products",                       # already used
    "Clothing_Shoes_and_Jewelry", "Arts_Crafts_and_Sewing",      # same domain as a used one
    "Books", "Kindle_Store", "Movies_and_TV", "CDs_and_Vinyl", "Digital_Music", "Video_Games",
    "Software", "Magazine_Subscriptions", "Gift_Cards", "Subscription_Boxes", "Unknown",
}
LOW, HIGH = 292_425_439, 1_000_000_000      # Amazon Fashion's review file .. 1 GB
SEED = 20261028


def eligible(rows: list[dict]) -> list[str]:
    return sorted(r["category"] for r in rows
                  if r["category"] not in EXCLUDED and LOW <= r["review_bytes"] <= HIGH)


def draw(names: list[str], seed: int = SEED) -> str:
    return str(np.random.default_rng(seed).choice(names))


def main() -> None:
    names = eligible(json.loads(SURVEY.read_text(encoding="utf-8"))["rows"])
    print(json.dumps({"eligible": names, "seed": SEED, "chosen": draw(names)}, indent=2))


if __name__ == "__main__":
    main()
