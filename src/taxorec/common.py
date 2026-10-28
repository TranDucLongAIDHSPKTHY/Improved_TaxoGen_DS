"""Shared helpers: UTF-8/gzip I/O, YAML configs with `_base_` inheritance and
`a.b=value` overrides, and run bookkeeping. Every experiment writes its
resolved config, environment, input fingerprints, log and results into its
own run directory, so each number in the paper traces back to one run."""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import yaml


# ---------------------------------------------------------------- I/O

def open_text(path, mode: str = "rt"):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8")
    newline = "\n" if ("w" in mode or "a" in mode) else None
    return open(path, mode, encoding="utf-8", newline=newline)


def ensure_dir(path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def iter_jsonl(path, limit: int | None = None) -> Iterator[dict]:
    with open_text(path) as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if line:
                yield json.loads(line)


def _json_default(obj: Any):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def _atomic_write_text(path, text: str) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)
    return path


def write_json(path, obj: Any) -> Path:
    return _atomic_write_text(path, json.dumps(obj, indent=2, ensure_ascii=False,
                                               default=_json_default) + "\n")


def read_json(path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path, rows: Iterable[dict]) -> Path:
    lines = (json.dumps(r, ensure_ascii=False, default=_json_default) for r in rows)
    return _atomic_write_text(path, "".join(line + "\n" for line in lines))


def read_jsonl(path) -> list[dict]:
    return list(iter_jsonl(path))


# ---------------------------------------------------------------- configs

def deep_merge(base: dict, override: dict) -> dict:
    """Recursive merge; a mapping containing `_replace_: true` replaces the
    base mapping instead of being merged into it."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and value.get("_replace_"):
            out[key] = {k: copy.deepcopy(v) for k, v in value.items() if k != "_replace_"}
        elif isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def apply_override(cfg: dict, expr: str) -> None:
    key, sep, raw = expr.partition("=")
    if not sep or not key:
        raise ValueError(f"override must look like 'a.b=value', got {expr!r}")
    parts = key.strip().split(".")
    node = cfg
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ValueError(f"cannot override {key!r}: {part!r} is not a mapping")
    node[parts[-1]] = yaml.safe_load(raw)


def load_config(path, overrides: Iterable[str] = ()) -> dict:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    bases = cfg.pop("_base_", [])
    if isinstance(bases, str):
        bases = [bases]
    merged: dict = {}
    for base in bases:
        merged = deep_merge(merged, load_config(path.parent / base))
    cfg = deep_merge(merged, cfg)
    for expr in overrides:
        apply_override(cfg, expr)
    return cfg


def config_hash(cfg: Any, n: int = 10) -> str:
    blob = json.dumps(cfg, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:n]


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


# ---------------------------------------------------------------- runs

PACKAGES = ["numpy", "scipy", "scikit-learn", "pandas", "pyyaml", "sentence-transformers",
            "gensim", "spacy", "torch", "statsmodels", "faiss-cpu", "bertopic"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def package_versions(names: Iterable[str] = PACKAGES) -> dict[str, str | None]:
    out = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def hardware_info() -> dict[str, Any]:
    """CPU model, logical CPUs and RAM: results must come from one machine,
    since BLAS code paths (AVX2 vs AVX-512, thread count) can differ across CPUs."""
    cpu = platform.processor() or None
    ram_gb = None
    try:
        if sys.platform == "win32":
            import ctypes
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()

            class _Mem(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            mem = _Mem()
            mem.dwLength = ctypes.sizeof(_Mem)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
            ram_gb = round(mem.ullTotalPhys / 2 ** 30, 1)
        elif Path("/proc/cpuinfo").exists():
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
            ram_gb = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30, 1)
    except (OSError, ValueError, AttributeError):
        pass
    return {"cpu": cpu, "logical_cpus": os.cpu_count(), "ram_gb": ram_gb}


def git_info(path=".") -> dict[str, Any]:
    def _git(*args):
        res = subprocess.run(["git", *args], cwd=path, capture_output=True, text=True)
        return res.stdout.strip() if res.returncode == 0 else None

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def fingerprint(path, use_cache: bool = True) -> dict[str, Any]:
    """sha256 of a file, cached in a sidecar keyed by size and mtime."""
    path = Path(path)
    st = path.stat()
    sidecar = path.with_name(path.name + ".sha256.json")
    if use_cache and sidecar.exists():
        try:
            cached = read_json(sidecar)
            if cached["size"] == st.st_size and cached["mtime"] == st.st_mtime:
                return cached
        except (OSError, KeyError, ValueError):
            pass
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    info = {"path": str(path), "size": st.st_size, "mtime": st.st_mtime, "sha256": h.hexdigest()}
    if use_cache:
        try:
            write_json(sidecar, info)
        except OSError:
            pass
    return info


class Run:
    def __init__(self, out_root, name: str, config: dict, inputs: Iterable = (),
                 seed: int | None = None, fingerprint_inputs: bool = True):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.config = config
        self.dir = ensure_dir(Path(out_root) / name / f"{stamp}-{config_hash(config)}")
        self.started = time.time()
        if seed is not None:
            set_seed(seed)
        with open(self.dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
        write_json(self.dir / "env.json", {
            "python": sys.version,
            "platform": platform.platform(),
            "hardware": hardware_info(),
            "argv": sys.argv,
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "seed": seed,
            "git": git_info(Path(__file__).resolve().parent),
            "packages": package_versions(),
        })
        inputs = [Path(p) for p in inputs if p]
        write_json(self.dir / "inputs.json", [
            fingerprint(p) if fingerprint_inputs and p.is_file() else {"path": str(p), "exists": p.exists()}
            for p in inputs
        ])
        self.logger = logging.getLogger(f"taxorec.run.{name}.{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        fmt = logging.Formatter("%(asctime)s %(message)s")
        for handler in (logging.FileHandler(self.dir / "log.txt", encoding="utf-8"),
                        logging.StreamHandler(sys.stdout)):
            handler.setFormatter(fmt)
            self.logger.addHandler(handler)
        self.log(f"run dir: {self.dir}")

    def log(self, msg: str) -> None:
        self.logger.info(msg)

    def path(self, rel: str) -> Path:
        p = self.dir / rel
        ensure_dir(p.parent)
        return p

    def save_json(self, rel: str, obj: Any) -> Path:
        return write_json(self.path(rel), obj)

    def finish(self, summary: dict | None = None) -> None:
        write_json(self.dir / "status.json", {"elapsed_sec": time.time() - self.started,
                                              "summary": summary or {}})
        for handler in list(self.logger.handlers):
            handler.close()
            self.logger.removeHandler(handler)
