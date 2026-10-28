import importlib.util
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "tools" / "draw_confirmation_category.py"


def _tool():
    spec = importlib.util.spec_from_file_location("draw_confirmation_category", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_eligible_applies_domain_and_size_rules_and_draw_is_fixed():
    tool = _tool()
    rows = [{"category": "Toys", "review_bytes": 500_000_000},
            {"category": "Books", "review_bytes": 500_000_000},          # excluded domain
            {"category": "Tiny", "review_bytes": 1_000},                  # below Fashion
            {"category": "Huge", "review_bytes": 5_000_000_000},          # above 1 GB
            {"category": "Appliances", "review_bytes": tool.LOW}]
    names = tool.eligible(rows)
    assert names == ["Appliances", "Toys"]
    assert tool.draw(names) == tool.draw(names) in names
