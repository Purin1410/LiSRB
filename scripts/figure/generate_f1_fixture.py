#!/usr/bin/env python3
"""Generate the code-backed relation example used by the LiSRB F1 figure.

This script imports the production ``TreeRelationBuilder``. It deliberately has no
second implementation of the parser or relation matrix. The small amount of code
below only decodes the selected pair into human-readable fields for the figure.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

TYPE_NAMES = ("ROOT", "SUP", "SUB", "NUM", "DEN")
TOKENS = (
    "\\frac", "{", "b", "^", "{", "2", "x", "}", "}",
    "{", "b", "^", "{", "y", "}", "}",
)
DISPLAY_TOKEN_INDICES = (0, 2, 5, 6, 10, 13)
PAIR = (2, 10)  # numerator b -> denominator b
NUM_BUCKETS = 16
MODE = "full"
REL_SET = "core"
PAD_ID = 0


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_tree_bias_module(repo_root: Path):
    module_path = repo_root / "models" / "transformer" / "tree_bias.py"
    spec = importlib.util.spec_from_file_location("lisrb_tree_bias", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _dictionary_path(repo_root: Path) -> Path:
    return repo_root.parent / "crohme_data" / "data" / "dictionary.txt"


def _load_vocabulary(path: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    word2idx = {"<pad>": 0, "<sos>": 1, "<eos>": 2}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw:
            word2idx[raw] = len(word2idx)
    return word2idx, {idx: token for token, idx in word2idx.items()}


def _lcp_length(left: Sequence[int], right: Sequence[int]) -> int:
    length = 0
    for a, b in zip(left, right):
        if a != b:
            break
        length += 1
    return length


def _pair_fields(
    relation_ids: Any,
    paths: Sequence[Sequence[int]],
    i: int,
    j: int,
    num_buckets: int,
) -> Dict[str, Any]:
    path_i = tuple(paths[i])
    path_j = tuple(paths[j])
    lcp = _lcp_length(path_i, path_j)
    distance = len(path_i) + len(path_j) - 2 * lcp
    bucket = min(max(distance, 0), num_buckets - 1)
    type_i = path_i[lcp] if lcp < len(path_i) else 0
    type_j = path_j[lcp] if lcp < len(path_j) else 0
    return {
        "i": i,
        "j": j,
        "token_i": TOKENS[i],
        "token_j": TOKENS[j],
        "lcp_path": [TYPE_NAMES[k] for k in path_i[:lcp]] or ["ROOT"],
        "distance": distance,
        "distance_bucket": bucket,
        "ordered_type_pair": [TYPE_NAMES[type_i], TYPE_NAMES[type_j]],
        "r_ij": int(relation_ids[i, j].item()),
        "r_ji": int(relation_ids[j, i].item()),
    }


def build_fixture(repo_root: Path | None = None) -> Dict[str, Any]:
    repo_root = repo_root or _repo_root()
    tree_bias = _load_tree_bias_module(repo_root)
    word2idx, id2tok = _load_vocabulary(_dictionary_path(repo_root))

    missing = [token for token in TOKENS if token not in word2idx]
    if missing:
        raise ValueError(f"CROHME dictionary lacks F1 tokens: {missing}")

    token_ids = [word2idx[token] for token in TOKENS]
    builder = tree_bias.TreeRelationBuilder(
        id2tok=id2tok,
        pad_id=PAD_ID,
        num_buckets=NUM_BUCKETS,
        mode=MODE,
        rel_set=REL_SET,
    )
    tensor = tree_bias.torch.tensor(token_ids, dtype=tree_bias.torch.long)
    paths = builder._paths_for_seq_ids(tensor)
    relation_ids = builder.build(tensor.unsqueeze(0))[0]
    pair = _pair_fields(relation_ids, paths, PAIR[0], PAIR[1], NUM_BUCKETS)

    if pair["r_ij"] == pair["r_ji"]:
        raise AssertionError("The displayed F1 pair must encode an ordered relation")
    if pair["ordered_type_pair"] != ["NUM", "DEN"] or pair["distance"] != 2:
        raise AssertionError(f"Unexpected displayed relation: {pair}")

    fixture: Dict[str, Any] = {
        "schema_version": 1,
        "expression_latex": "\\frac{b^{2x}}{b^y}",
        "source_boundary": {
            "training": "teacher-forced target tokens",
            "inference": "current generated hypothesis prefix",
        },
        "tokens": [
            {
                "index": index,
                "token": token,
                "token_id": token_ids[index],
                "context_path": [TYPE_NAMES[k] for k in paths[index]],
                "displayed": index in DISPLAY_TOKEN_INDICES,
            }
            for index, token in enumerate(TOKENS)
        ],
        "display_pair": pair,
        "config": {
            "num_buckets": NUM_BUCKETS,
            "mode": MODE,
            "rel_set": builder.rel_set,
            "num_relations": builder.num_relations,
            "type_order": list(TYPE_NAMES),
            "relation_formula": "bucket * 25 + type_i * 5 + type_j",
        },
        "implementation": {
            "builder": "LiSRB/models/transformer/tree_bias.py::TreeRelationBuilder",
            "generated_by": "LiSRB/scripts/figure/generate_f1_fixture.py",
        },
    }
    canonical = json.dumps(fixture, sort_keys=True, separators=(",", ":")).encode()
    fixture["content_sha256"] = hashlib.sha256(canonical).hexdigest()
    return fixture


def write_fixture(output: Path, check: bool = False) -> None:
    rendered = json.dumps(build_fixture(), indent=2, ensure_ascii=False) + "\n"
    if check:
        if not output.exists() or output.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"Fixture is missing or stale: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")


def main() -> None:
    default_output = (
        _repo_root().parent
        / "LiSRB_Paper_Agent_Kit_v1"
        / "04_figures"
        / "F1"
        / "data"
        / "relation_fixture.json"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    write_fixture(args.output.resolve(), check=args.check)


if __name__ == "__main__":
    main()
