import importlib.util
from pathlib import Path

import pytest

from taxorec.taxonomy.tree import ROOT, Taxonomy

TOOL = Path(__file__).resolve().parents[1] / "tools" / "taxogen_fidelity.py"


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("taxogen_fidelity", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_convert_reads_compress_output_and_node_documents(tool, tmp_path):
    root = tmp_path / "our-l3-0.15"
    for folder, n in [("ml", 3), ("ml/deep_learning", 2), ("db", 1)]:
        (root / folder).mkdir(parents=True, exist_ok=True)
        (root / folder / "doc_ids.txt").write_text("".join(f"{i}\n" for i in range(n)))
    compressed = tmp_path / "taxonomy.txt"
    compressed.write_text("*/ml\tml,learning\n*/db\tdb,query\n*/ml/deep_learning\tdeep_learning,cnn\n",
                          encoding="utf-8")
    tax = tool.convert(root, compressed)
    assert tax.size == 3 and tax.depth == 2
    ml = next(n for n in tax.children(ROOT) if n.terms[0] == "ml")
    assert ml.n_docs == 3 and [c.terms for c in tax.children(ml.id)] == [["deep_learning", "cnn"]]


def test_level1_overlap(tool):
    a, b = Taxonomy("a"), Taxonomy("b")
    a.add_child(ROOT, ["x", "y"])
    b.add_child(ROOT, ["x", "z"])
    b.add_child(ROOT, ["x", "y"])
    assert tool.level1_overlap(a, b) == 1.0 and tool.level1_overlap(b, a) == pytest.approx((1 / 3 + 1) / 2)
