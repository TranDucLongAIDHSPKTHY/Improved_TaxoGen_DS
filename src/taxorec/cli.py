"""Command line: `python -m taxorec <command> --config configs/<dataset>.yaml`.

Order for an Amazon dataset:
  download -> prepare-rec -> prepare-corpus -> tune-taxonomy -> build-taxonomy
  -> build-taxonomy --all-seeds -> ablation -> eval-taxonomy -> make-human-eval
  -> freeze -> (send sheets, collect answers) -> import-forms -> analyze-human-eval
  -> run-rec -> report -> archive
`all` runs every step that needs no human input (up to make-human-eval, plus
run-rec and report).
Once analyze-human-eval has run on every dataset with human evaluation,
`human-eval-holm --config A --config B` adjusts the primary family together.
"""
from __future__ import annotations

import argparse

from taxorec import pipelines as P
from taxorec.common import load_config

STEPS = {
    "download": lambda cfg, a: P.download(cfg),
    "make-toy-data": lambda cfg, a: P.make_toy(cfg),
    "prepare-rec": lambda cfg, a: P.prepare_rec(cfg),
    "prepare-corpus": lambda cfg, a: P.prepare_corpus(cfg),
    "tune-taxonomy": lambda cfg, a: P.tune_taxonomy(cfg, a.method),
    "build-taxonomy": lambda cfg, a: P.build_taxonomy(cfg, a.method, all_seeds=a.all_seeds),
    "sensitivity": lambda cfg, a: P.sensitivity(cfg),
    "ablation": lambda cfg, a: P.ablation(cfg, force=a.force),
    "eval-taxonomy": lambda cfg, a: P.evaluate_taxonomies(cfg, a.method, force=a.force),
    "make-human-eval": lambda cfg, a: P.make_human_eval(cfg, force=a.force),
    "freeze": lambda cfg, a: P.freeze(cfg),
    "import-forms": lambda cfg, a: P.import_forms(cfg, a.file),
    "analyze-human-eval": lambda cfg, a: P.analyze_human_eval(cfg, force=a.force),
    "run-rec": lambda cfg, a: P.run_rec(cfg, a.model),
    "report": lambda cfg, a: P.report(cfg),
    "archive": lambda cfg, a: P.archive(cfg),
}


def run_all(cfg: dict) -> None:
    has_rec = "rec" in cfg
    if has_rec:
        P.prepare_rec(cfg)
    P.prepare_corpus(cfg)
    if cfg["taxonomy"].get("tuning"):
        P.tune_taxonomy(cfg)
    P.build_taxonomy(cfg)
    P.build_taxonomy(cfg, all_seeds=True)
    if cfg["taxonomy"].get("ablation"):
        P.ablation(cfg)
    P.evaluate_taxonomies(cfg)
    if cfg.get("human_eval", {}).get("enabled", True):
        P.make_human_eval(cfg)
    if has_rec:
        P.run_rec(cfg)
    P.report(cfg)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="taxorec", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=sorted(STEPS) + ["all", "human-eval-holm"])
    parser.add_argument("--config", action="append", required=True,
                        help="dataset config, e.g. configs/amazon_fashion.yaml "
                             "(human-eval-holm: one per dataset with human evaluation)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, e.g. --set taxonomy.methods.improved.builder.tau=0.4")
    parser.add_argument("--method", action="append", help="tune/build/eval-taxonomy: only these methods")
    parser.add_argument("--all-seeds", action="store_true",
                        help="build-taxonomy: one taxonomy per seed in `seeds` (for mean +- std)")
    parser.add_argument("--model", action="append", help="run-rec: only these models")
    parser.add_argument("--file", action="append", default=[], help="import-forms: Google Forms CSV export(s)")
    parser.add_argument("--force", action="store_true",
                        help="override a safety check (frozen batch / failed integrity check); "
                             "never for results that go into the paper")
    args = parser.parse_args(argv)
    cfgs = [load_config(c, args.set) for c in args.config]
    if args.command == "human-eval-holm":
        P.human_eval_holm(cfgs)
        return
    if len(cfgs) > 1:
        parser.error("only human-eval-holm takes several --config")
    cfg = cfgs[0]
    if args.command == "all":
        run_all(cfg)
    else:
        STEPS[args.command](cfg, args)


if __name__ == "__main__":
    main()
