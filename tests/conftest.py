import random

import numpy as np
import pytest

from taxorec.data import Corpus, DocStore
from taxorec.taxonomy.components import KeyedEncoder

TOPICS = ["shoe", "bag", "dress"]
SUBTOPICS = ["alpha", "beta", "gamma"]
N_TERMS = 8


def term_name(topic, sub, j):
    return f"{topic}_{sub}_{j}"


@pytest.fixture(scope="session")
def hierarchy():
    """3 topics x 3 subtopics x 8 terms, documents drawn mostly from one subtopic."""
    terms = {(t, s): [term_name(t, s, j) for j in range(N_TERMS)] for t in TOPICS for s in SUBTOPICS}
    rng = random.Random(0)
    docs = []
    for _ in range(900):
        t, s = rng.choice(TOPICS), rng.choice(SUBTOPICS)
        doc = rng.sample(terms[(t, s)], 4)
        doc += rng.sample(terms[(t, rng.choice(SUBTOPICS))], 1)
        doc += ["filler", "words", "here"]
        rng.shuffle(doc)
        docs.append(doc)
    return terms, docs


@pytest.fixture(scope="session")
def synthetic_corpus(hierarchy, tmp_path_factory):
    _, docs = hierarchy
    path = tmp_path_factory.mktemp("corpus") / "docs.txt"
    store = DocStore.write(path, docs)
    return Corpus.build(lambda: iter(store), min_df=2, max_df_ratio=0.5, docstore_path=path)


@pytest.fixture(scope="session")
def synthetic_encoder(hierarchy):
    terms, _ = hierarchy
    rng = np.random.default_rng(0)
    dim = 48
    topic_vec = {t: rng.standard_normal(dim) for t in TOPICS}
    sub_vec = {(t, s): rng.standard_normal(dim) for t in TOPICS for s in SUBTOPICS}
    mapping = {}
    for (t, s), names in terms.items():
        for name in names:
            mapping[name] = (3.0 * topic_vec[t] + 1.2 * sub_vec[(t, s)]
                             + 0.15 * rng.standard_normal(dim)).astype(np.float32)
    return KeyedEncoder(mapping)
