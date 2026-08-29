#!/usr/bin/env python
"""Rebuild submission.csv from an existing candidates.csv — no GPU, no model.

This reproduces the *deterministic half* of the pipeline.  Generation uses
temperature sampling and therefore cannot be reproduced bit-for-bit across
different hardware, but aggregation is pure arithmetic over the parsed answers
already recorded in ``candidates.csv``.

Running this on the committed artifacts must yield a file identical to the
submitted ``submission.csv``:

    python src/aggregate_only.py \
        --candidates artifacts/candidates.csv \
        --test       artifacts/test_ids.csv \
        --out        /tmp/rebuilt.csv
    diff /tmp/rebuilt.csv artifacts/submission.csv   # expected: no output

``candidates.csv`` contains one row per (problem, prompt, sample) with the
integer parsed out of that sample's generated text.  No answer key, no external
data — the vote is computed from those parsed values alone.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_inference import write_submission  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, help="candidates.csv from a run")
    ap.add_argument("--test", required=True,
                    help="CSV with the id column in submission order")
    ap.add_argument("--out", default="submission_rebuilt.csv")
    args = ap.parse_args()

    cands = pd.read_csv(args.candidates)
    cands["id"] = cands["id"].astype(str)
    test = pd.read_csv(args.test)
    test["id"] = test["id"].astype(str)

    print(f"candidates : {len(cands)} rows / {cands['id'].nunique()} problems")
    print(f"prompts    : {sorted(cands['prompt'].unique())}")
    print(f"test order : {len(test)} ids")
    write_submission(cands, test, args.out)


if __name__ == "__main__":
    main()
