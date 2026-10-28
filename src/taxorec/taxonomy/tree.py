"""Taxonomy tree shared by every construction method, evaluator and recommender."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

from taxorec.common import open_text, read_json, write_json

ROOT = "root"


@dataclass
class Node:
    id: str
    parent: str | None
    level: int
    terms: list[str] = field(default_factory=list)          # ranked; the node's whole term pool
    general_terms: list[str] = field(default_factory=list)  # pool terms not passed to any child
    n_docs: int = 0
    scores: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.terms[0] if self.terms else self.id


class Taxonomy:
    def __init__(self, method: str = "", params: dict | None = None):
        self.method = method
        self.params = dict(params or {})
        self.nodes: dict[str, Node] = {ROOT: Node(ROOT, None, 0)}
        self._children: dict[str, list[str]] = defaultdict(list)

    # ---------------------------------------------------------------- structure
    @property
    def root(self) -> Node:
        return self.nodes[ROOT]

    def add_child(self, parent_id: str, terms: list[str], n_docs: int = 0,
                  scores: dict | None = None, node_id: str | None = None) -> Node:
        parent = self.nodes[parent_id]
        node_id = node_id or f"{parent_id}/{len(self._children[parent_id])}"
        if node_id in self.nodes:
            raise ValueError(f"duplicate node id {node_id}")
        node = Node(node_id, parent_id, parent.level + 1, list(terms), [], n_docs, dict(scores or {}))
        self.nodes[node_id] = node
        self._children[parent_id].append(node_id)
        return node

    def children(self, node_id: str) -> list[Node]:
        return [self.nodes[c] for c in self._children.get(node_id, [])]

    def parent(self, node_id: str) -> Node | None:
        p = self.nodes[node_id].parent
        return self.nodes[p] if p is not None else None

    def ancestors(self, node_id: str) -> list[Node]:
        out, node = [], self.parent(node_id)
        while node is not None and node.id != ROOT:
            out.append(node)
            node = self.parent(node.id)
        return out[::-1]

    def non_root(self) -> Iterator[Node]:
        return (n for n in self.nodes.values() if n.id != ROOT)

    def edges(self, include_root: bool = False) -> list[tuple[Node, Node]]:
        return [(self.nodes[n.parent], n) for n in self.non_root()
                if include_root or n.parent != ROOT]

    def siblings(self, node_id: str) -> list[Node]:
        node = self.nodes[node_id]
        return [c for c in self.children(node.parent) if c.id != node_id] if node.parent else []

    @property
    def depth(self) -> int:
        return max((n.level for n in self.nodes.values()), default=0)

    @property
    def size(self) -> int:
        return len(self.nodes) - 1

    def path_labels(self, node_id: str) -> list[str]:
        return [a.label for a in self.ancestors(node_id)] + [self.nodes[node_id].label]

    # ---------------------------------------------------------------- checks
    def validate(self) -> list[tuple[str, str]]:
        """(severity, message) pairs. 'error' issues make a taxonomy unusable:
        empty nodes, and children whose term set equals their parent's (a
        degenerate edge that annotators trivially accept)."""
        issues = []
        for node in self.non_root():
            if not node.terms:
                issues.append(("error", f"{node.id}: empty node"))
                continue
            parent = self.nodes[node.parent]
            if parent.terms and set(node.terms) == set(parent.terms):
                issues.append(("error", f"{node.id}: same term set as its parent {parent.id}"))
        for node_id in list(self._children):
            kids = self.children(node_id)
            if len(kids) == 1:
                issues.append(("warning", f"{node_id}: only one child"))
            seen: dict[str, str] = {}
            for kid in kids:
                for t in kid.terms:
                    if t in seen and seen[t] != kid.id:
                        issues.append(("warning", f"{node_id}: term {t!r} in siblings "
                                                  f"{seen[t]} and {kid.id}"))
                        break
                    seen[t] = kid.id
        return issues

    def stats(self) -> dict:
        per_level = defaultdict(int)
        for n in self.non_root():
            per_level[n.level] += 1
        internal = [nid for nid, kids in self._children.items() if kids]
        placed = set().union(*(n.terms for n in self.non_root())) if self.size else set()
        return {
            "n_nodes": self.size,
            "depth": self.depth,
            "nodes_per_level": dict(sorted(per_level.items())),
            "mean_branching": (sum(len(self._children[i]) for i in internal) / len(internal)
                               if internal else 0.0),
            "n_leaves": sum(1 for n in self.non_root() if not self._children.get(n.id)),
            "n_terms_placed": len(placed),
            "n_degenerate_edges": sum(1 for sev, msg in self.validate()
                                      if sev == "error" and "same term set" in msg),
        }

    def prune_to_size(self, n_nodes: int) -> "Taxonomy":
        """Keep the n_nodes highest-priority nodes (shallower first, then more
        documents), never keeping a node without its parent. Used to compare
        methods at matched taxonomy size."""
        order = sorted(self.non_root(), key=lambda n: (n.level, -n.n_docs, n.id))
        keep = {ROOT}
        for node in order:
            if len(keep) - 1 >= n_nodes:
                break
            if node.parent in keep:
                keep.add(node.id)
        out = Taxonomy(self.method, {**self.params, "pruned_to": n_nodes})
        out.nodes[ROOT] = Node(**asdict(self.root))
        for node in sorted(self.non_root(), key=lambda n: (n.level, n.id)):
            if node.id in keep:
                copy = out.add_child(node.parent, node.terms, node.n_docs, node.scores, node_id=node.id)
                copy.general_terms = list(node.general_terms)
        return out

    # ---------------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        return {"method": self.method, "params": self.params,
                "nodes": [asdict(n) for n in sorted(self.nodes.values(), key=lambda n: (n.level, n.id))]}

    @classmethod
    def from_dict(cls, d: dict) -> "Taxonomy":
        tax = cls(d.get("method", ""), d.get("params"))
        for nd in d["nodes"]:
            node = Node(**nd)
            if node.id == ROOT:
                tax.nodes[ROOT] = node
            else:
                tax.nodes[node.id] = node
                tax._children[node.parent].append(node.id)
        return tax

    def save(self, path) -> Path:
        return write_json(path, self.to_dict())

    @classmethod
    def load(cls, path) -> "Taxonomy":
        return cls.from_dict(read_json(path))

    def save_txt(self, path, top_k: int = 10) -> None:
        """Human-readable dump: label path, then the top terms."""
        with open_text(path, "wt") as f:
            for node in sorted(self.non_root(), key=lambda n: n.id):
                f.write(" / ".join(self.path_labels(node.id)) + "\t" + ", ".join(node.terms[:top_k]) + "\n")

    @classmethod
    def from_txt(cls, path, method: str = "imported") -> "Taxonomy":
        """Import the legacy 'a/b/c<TAB>term,term,...' format."""
        tax = cls(method, {"source": str(path)})
        by_path = {"": ROOT}
        with open_text(path) as f:
            rows = [line.rstrip("\n").split("\t") for line in f if "\t" in line]
        for path_str, terms in sorted(rows, key=lambda r: r[0].count("/")):
            parent_path = path_str.rsplit("/", 1)[0] if "/" in path_str else ""
            node = tax.add_child(by_path[parent_path], [t for t in terms.split(",") if t])
            by_path[path_str] = node.id
        return tax
