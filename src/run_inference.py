#!/usr/bin/env python
"""End-to-end inference for the 2026 Deep Learning Challenge.

Reads a test CSV with columns ``id`` and ``question`` and writes
``submission.csv`` with columns ``id`` and ``answer`` (integer, exact match).

Pipeline
--------
1. Stage 1 - sample ``--k1`` completions per problem with prompt A.
2. Problems where every parsed answer agrees are considered settled and are
   excluded from further generation (adaptive self-consistency).
3. Stage 2 - for the remaining problems, sample ``--k2`` completions with each
   of the extra prompts (default: B, C, C2).
4. Aggregate with *prompt-averaged voting*: within each prompt the vote share of
   every candidate answer is computed, then the shares are averaged across
   prompts with equal weight.  The answer with the highest average share wins.

The submission file is rewritten after every stage, so an interrupted run still
leaves a complete, valid submission on disk.  Generated candidates are appended
to ``<work>/candidates.csv``; re-running the script skips (problem, prompt)
pairs that are already present, so the job is resumable.

Only the base model produces answers.  There is no lookup table, no external
dataset of answers, and no network access at inference time.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mathx  # noqa: E402
import genlib  # noqa: E402
import prompts  # noqa: E402


def log(msg: str) -> None:
    print(msg, flush=True)


def set_prompt(name: str) -> None:
    """genlib.build_prompt reads these two attributes at call time."""
    system, user = prompts.REGISTRY[name]
    mathx.PROMPT_SYSTEM = system
    mathx.PROMPT_USER = user


def prompt_seed(base: int, name: str) -> int:
    """A distinct RNG stream per prompt.

    Prompt diversity only helps if the prompts sample independently, so every
    registered prompt gets its own seed block.  Deriving the offset from the
    first character gives ``C`` and ``C2`` the same stream; the position in the
    sorted registry keeps them apart.  The block size (1e6) is far larger than
    any chunk offset, so the streams cannot overlap.
    """
    order = sorted(prompts.REGISTRY)
    index = order.index(name) if name in order else len(order)
    return base + 1_000_000 * (index + 1)


def load_model(model_dir: str, dtype: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    kwargs = dict(
        attn_implementation="sdpa",
        device_map={"": 0},
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch_dtype, **kwargs)
    except TypeError:  # transformers < 5 spelling
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch_dtype, **kwargs)

    model.eval()
    model.config.use_cache = True
    model.generation_config.pad_token_id = tok.pad_token_id
    for key in ("temperature", "top_p", "top_k"):
        if hasattr(model.generation_config, key):
            setattr(model.generation_config, key, None)
    return model, tok


def attach_adapter(model, adapter_dir: str):
    """Optional LoRA adapter.  Unused in the submitted configuration."""
    from peft import PeftModel

    log(f"  attaching LoRA adapter: {adapter_dir}")
    return PeftModel.from_pretrained(model, adapter_dir, torch_dtype=torch.float16)


def generate_stage(model, tok, frame, prompt_name, k, args, out_csv, seed_base):
    """Sample k completions per row of `frame` with one prompt; append to out_csv."""
    if len(frame) == 0:
        return
    set_prompt(prompt_name)
    total = len(frame)
    t0 = time.time()
    for start in range(0, total, args.chunk):
        chunk = frame.iloc[start : start + args.chunk].reset_index(drop=True)
        questions, meta = [], []
        for _, row in chunk.iterrows():
            for j in range(k):
                questions.append(row["question"])
                meta.append((str(row["id"]), j))
        res = genlib.generate_rounds(
            model,
            tok,
            questions,
            rounds=tuple(args.rounds),
            batch_size=args.batch_size,
            token_budget=args.token_budget,
            max_input=args.max_input,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=seed_base + start,
            verbose=False,
        )
        rows = []
        for (pid, j), text, ntok in zip(meta, res["texts"], res["ntok"]):
            value, method = mathx.extract(text, "lenient")
            rows.append(
                dict(
                    id=pid,
                    prompt=prompt_name,
                    sample_idx=j,
                    pred=(int(value) if value is not None else None),
                    method=method,
                    end_kind=mathx.end_kind(text),
                    ntok=ntok,
                )
            )
        pd.DataFrame(rows).to_csv(
            out_csv, mode="a", header=not os.path.exists(out_csv), index=False
        )
        done = start + len(chunk)
        elapsed = (time.time() - t0) / 60
        log(
            f"    [{prompt_name}] {done:5d}/{total} | {elapsed:6.1f} min "
            f"| eta {elapsed / done * (total - done):6.1f} min"
        )


def prompt_averaged_vote(group: pd.DataFrame):
    """Equal weight per prompt; within a prompt, weight by vote share."""
    scores = defaultdict(float)
    names = sorted(group["prompt"].unique())
    for name in names:
        preds = [p for p in group.loc[group["prompt"] == name, "pred"] if pd.notna(p)]
        if not preds:
            continue
        for value, count in Counter(preds).items():
            scores[value] += count / len(preds) / len(names)
    if not scores:
        return None
    return max(scores, key=scores.get)


def write_submission(candidates: pd.DataFrame, test: pd.DataFrame, path: str) -> int:
    picks = {pid: prompt_averaged_vote(g) for pid, g in candidates.groupby("id")}
    valid = [int(v) for v in picks.values() if v is not None]
    fallback = Counter(valid).most_common(1)[0][0] if valid else 0
    answers, n_fallback = [], 0
    for pid in test["id"]:
        value = picks.get(pid)
        if value is None or abs(int(value)) > 10**12:
            answers.append(fallback)
            n_fallback += 1
        else:
            answers.append(int(value))
    sub = pd.DataFrame({"id": test["id"].astype(str), "answer": answers})
    sub["answer"] = sub["answer"].astype(int)
    sub.to_csv(path, index=False)
    log(f"  wrote {path}  rows={len(sub)}  fallback={n_fallback}")
    return n_fallback


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True, help="CSV with columns id,question")
    ap.add_argument("--model-dir", required=True, help="local Qwen2.5-3B-Instruct directory")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--work", default="./work")
    ap.add_argument("--adapter-dir", default=None, help="optional LoRA adapter (unused by default)")
    ap.add_argument("--k1", type=int, default=4, help="samples per problem, stage 1 (prompt A)")
    ap.add_argument("--k2", type=int, default=6, help="samples per problem per extra prompt")
    ap.add_argument("--extra-prompts", default="B,C,C2")
    ap.add_argument("--rounds", type=int, nargs="+", default=[512, 512, 512])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--token-budget", type=int, default=45000)
    ap.add_argument("--max-input", type=int, default=1792)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stage", default="all", choices=["all", "1", "2"])
    args = ap.parse_args()

    os.makedirs(args.work, exist_ok=True)
    cand_csv = os.path.join(args.work, "candidates.csv")

    test = pd.read_csv(args.test)
    assert {"id", "question"} <= set(test.columns), f"unexpected columns: {list(test.columns)}"
    test["id"] = test["id"].astype(str)
    log(f"[data] {len(test)} problems from {args.test}")

    log("[model] loading")
    model, tok = load_model(args.model_dir, args.dtype)
    if args.adapter_dir:
        model = attach_adapter(model, args.adapter_dir)
    log(f"  VRAM {torch.cuda.memory_allocated() / 2**30:.2f} GB")

    def load_candidates() -> pd.DataFrame:
        if not os.path.exists(cand_csv):
            return pd.DataFrame(columns=["id", "prompt", "sample_idx", "pred", "method",
                                         "end_kind", "ntok"])
        df = pd.read_csv(cand_csv)
        df["id"] = df["id"].astype(str)
        return df

    # ---------------- stage 1: prompt A over every problem ----------------
    if args.stage in ("all", "1"):
        done = set(load_candidates().query("prompt == 'A'")["id"])
        todo = test[~test["id"].isin(done)].reset_index(drop=True)
        log(f"[stage 1] prompt A x{args.k1} on {len(todo)} problems "
            f"({len(done)} already done)")
        generate_stage(model, tok, todo, "A", args.k1, args, cand_csv, args.seed)
        write_submission(load_candidates(), test, args.out)

    # ---------------- stage 2: extra prompts on unsettled problems --------
    if args.stage in ("all", "2"):
        cands = load_candidates()
        settled = set()
        for pid, g in cands[cands["prompt"] == "A"].groupby("id"):
            preds = [p for p in g["pred"] if pd.notna(p)]
            if preds and len(set(preds)) == 1:
                settled.add(pid)
        log(f"[stage 2] settled by unanimity: {len(settled)} / {len(test)}")
        hard = test[~test["id"].isin(settled)].reset_index(drop=True)
        for name in [p.strip() for p in args.extra_prompts.split(",") if p.strip()]:
            done = set(cands.query("prompt == @name")["id"])
            todo = hard[~hard["id"].isin(done)].reset_index(drop=True)
            log(f"  prompt {name} x{args.k2} on {len(todo)} problems")
            generate_stage(model, tok, todo, name, args.k2, args, cand_csv,
                           prompt_seed(args.seed, name))
            cands = load_candidates()
            write_submission(cands, test, args.out)

    cands = load_candidates()
    log(f"[done] candidates={len(cands)} problems={cands['id'].nunique()}")
    per_prompt = cands.groupby("prompt").agg(
        n=("pred", "size"),
        parsed=("pred", lambda s: float(s.notna().mean())),
        clean=("end_kind", lambda s: float((s == "clean_final").mean())),
    )
    log(per_prompt.to_string())
    json.dump(
        {"n_problems": int(cands["id"].nunique()), "n_candidates": int(len(cands)),
         "config": vars(args)},
        open(os.path.join(args.work, "run_report.json"), "w"),
        indent=2, default=str,
    )


if __name__ == "__main__":
    main()
