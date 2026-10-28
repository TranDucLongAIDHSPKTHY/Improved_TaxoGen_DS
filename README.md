# TaxoGen-Improved: taxonomy construction and taxonomy-guided recommendation

Reproducible experiments comparing TaxoGen (Zhang et al., KDD 2018) with an
SBERT document-clustering (UMAP + HDBSCAN) + BM25-cosine variant, evaluated directly (automatic metrics,
human judgments) and downstream (product recommendation on Amazon Reviews 2023).

Every number in the paper is produced by one command below and written to
`outputs/<dataset>/`, together with the exact config, package versions, git
commit and sha256 of every input file (`outputs/<dataset>/runs/`).

## Quick start (1 minute, no downloads)

```bash
python -m venv .venv && .venv\Scripts\activate        # Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel     # old pip cannot install pyproject-only packages
pip install -r requirements-lock.txt && pip install -e . # exact verified environment (incl. spaCy model, BERTopic)
# or, direct dependencies only:
#   pip install -r requirements.txt && pip install -e . && python -m spacy download en_core_web_sm
#   pip install bertopic -c requirements.txt             # optional baseline; -c keeps numpy/torch pinned
python -m pytest                                         # 74 tests, ~3 min

python -m taxorec make-toy-data --config configs/toy.yaml
python -m taxorec all --config configs/toy.yaml          # whole pipeline on synthetic data
# -> outputs/toy/REPORT.md
```

The toy dataset only shows that the pipeline runs; its numbers mean nothing.
Set `PYTHONHASHSEED=0` for bit-identical word2vec results across processes.

## Real experiments

| dataset | config | data |
|---|---|---|
| Amazon Fashion | `configs/amazon_fashion.yaml` | `python -m taxorec download --config configs/amazon_fashion.yaml` |
| Handmade Products | `configs/handmade_products.yaml` | `python -m taxorec download --config configs/handmade_products.yaml` |
| DBLP (TaxoGen release) | `configs/dblp.yaml` | `python -m taxorec download --config configs/dblp.yaml` (the [TaxoGen](https://github.com/franticnerd/taxogen) release from Google Drive, ~380 MB, sha256-verified) |

```bash
C=configs/amazon_fashion.yaml
python -m taxorec prepare-rec     --config $C   # split (train / valid / test), item embeddings
python -m taxorec prepare-corpus  --config $C   # review corpus WITHOUT held-out interactions
python -m taxorec tune-taxonomy   --config $C   # parameters built on train, chosen on val
python -m taxorec sensitivity     --config $C   # one parameter at a time around the tuned one (val)
python -m taxorec build-taxonomy  --config $C   # main taxonomies (seed 42)
python -m taxorec build-taxonomy  --config $C --all-seeds   # seeds 42, 0, 1, 2, 3, for mean +- std
python -m taxorec ablation        --config $C   # split (docs|terms) x clusterer x relation scorer (+ variants) x seeds
python -m taxorec eval-taxonomy   --config $C   # automatic metrics on test, at matched size and full size (+ gold)
python -m taxorec make-human-eval --config $C   # blinded sheets + Google Forms input
python -m taxorec freeze          --config $C   # snapshot the batch BEFORE sending it out
python -m taxorec import-forms    --config $C --file export.csv   # keep raw export, convert
python -m taxorec analyze-human-eval --config $C
python -m taxorec run-rec         --config $C   # tuned on validation, tested once
python -m taxorec report          --config $C   # outputs/<dataset>/REPORT.md
python -m taxorec archive         --config $C   # zip + sha256 manifest for Zenodo
```

The annotation forms are created with `tools/google_forms/create_form.gs`;
participant documents are in `docs/human_eval/`.

Any config value can be overridden from the command line, e.g.
`--set taxonomy.methods.improved.builder.tau=0.4`; the override is recorded in the run directory.

## Layout

```
configs/                 one YAML per dataset (+ shared _base_*.yaml)
src/taxorec/
  cli.py                 the `python -m taxorec <command>` entry point
  pipelines.py           one function per command; fixed output locations
  data.py                download, tokenization, phrases, Corpus, Amazon/DBLP loaders
  taxonomy/
    tree.py              Taxonomy data structure (+ validation, pruning, I/O)
    components.py        encoders, clustering, CaseOLAP ranking, BM25-cosine scoring
    builders.py          TaxoGen-Improved / ablation builder, TaxoGen, BERTopic
    evaluate.py          automatic metrics and item-to-path assignment
  human_eval.py          sampling, blinded sheets, integrity checks, statistics
  recommend.py           split, models, full-ranking metrics, tuning
  stats.py               bootstrap, Holm, Wilson, Fleiss, Krippendorff
  toy.py                 synthetic smoke-test data
tests/                   pytest suite (runs on synthetic data)
tools/google_forms/      Apps Script that builds the annotation form
docs/                    step-by-step guide, decisions, participant documents (Vietnamese)
```

## Protocol choices that matter

* Same corpus and candidate vocabulary for every taxonomy method on a dataset.
* Terms shown to annotators are ranked by one shared, embedding-agnostic
  ranker (CaseOLAP), sampled with equal per-depth quotas for every method.
* Documents are split once into train (20%), val (30%) and test (50%):
  construction parameters are built on train and chosen on val with a
  pre-registered rule, before any human judgment; final taxonomies are built
  on train + val; the test documents are read once, by eval-taxonomy and the
  ablation (a second reading is refused and every reading is logged). Methods
  are compared at matched size (every taxonomy pruned to the smallest one's
  size, shallower and larger nodes first), for the automatic metrics and for
  the human-evaluation sample, with full-size numbers reported too; word2vec runs
  single-threaded (bit-identical), the SBERT model is pinned to a commit, and
  automatic metrics are reported as mean +- std over 5 seeds.
* A taxonomy with a child identical to its parent is rejected at build time.
* The annotation batch is frozen (sha256) before sheets are sent; analysis
  refuses to run on a batch that differs from the frozen one.
* Annotator answers pass an integrity check before any statistics are computed;
  method differences use a two-way (item x annotator) bootstrap.
* Item representations use product metadata only (no review text), and reviews
  of validation/test interactions are excluded from the taxonomy corpus.
* Interactions: users need >= 3 (train/valid/test), items have no minimum;
  5-core leaves nothing in Amazon_Fashion or Handmade_Products (2023 release).
* Recommendation hyper-parameters and fusion weights are tuned on the
  validation item only; the test item is scored once, over the full catalog,
  with pessimistic tie-breaking.
