# -*- coding: utf-8 -*-
import argparse
import csv
import json
import os
from typing import List, Optional, Tuple

import numpy as np
from transformers import AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare EA vs Baseline decoding speed by model_id list (no states)."
    )
    p.add_argument("--prefix", type=str, required=True, help="Project root, e.g., /your_path")
    p.add_argument("--bench-name", type=str, required=True, help="Benchmark name, e.g., gsm8k")
    p.add_argument(
        "--model_ids",
        type=str,
        required=True,
        help="Comma-separated model_ids (exactly those used in step 4).",
    )

    p.add_argument("--temperature", type=str, default="1.0")

    p.add_argument(
        "--ea_file_pattern",
        type=str,
        help="EA jsonl pattern with one {} for model_id. "
             "Default: {prefix}/EAGLE/{bench}/{model_id}-temperature-{temp}.jsonl",
    )
    p.add_argument(
        "--base_file_pattern",
        type=str,
        help="Baseline jsonl pattern with one {} for model_id. "
             "Default: {prefix}/EAGLE/{bench}/{model_id}-temperature-baseline-{temp}.jsonl",
    )

    p.add_argument(
        "--tokenizer_path",
        type=str,
        help="Default: {prefix}/data/model_weight/llama/llama-3-8b-instruct",
    )
    p.add_argument(
        "--csv_path",
        type=str,
        help="Default: {prefix}/EAGLE/result/summary_temperature_{temp}.csv",
    )
    return p.parse_args()


def derive_defaults(prefix: str, bench: str, temp: str) -> Tuple[str, str, str, str]:
    pr = prefix.rstrip("/")
    ea_pat = f"{pr}/EAGLE/{bench}/{{}}-temperature-{temp}.jsonl"
    base_pat = f"{pr}/EAGLE/{bench}/{{}}-temperature-baseline-{temp}.jsonl"
    tok = f"{pr}/data/model_weight/llama/llama-3-8b-instruct"
    csvp = f"{pr}/EAGLE/result/summary_temperature_{temp}.csv"
    return ea_pat, base_pat, tok, csvp


def compute_ea_stats(ea_file: str):
    with open(ea_file, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    speeds, new_tokens, idxs = [], [], []
    for dp in rows:
        if "choices" not in dp:
            continue
        for ch in dp["choices"]:
            tokens = sum(ch.get("new_tokens", []))
            time_s = sum(ch.get("wall_time", []))
            if time_s > 0:
                speeds.append(tokens / time_s)
            new_tokens.append(tokens)
            idxs.append(sum(ch.get("idxs", [])))
    return speeds, new_tokens, idxs


def compute_base_stats(base_file: str, tokenizer):
    with open(base_file, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    speeds0, total_token, total_time = [], 0, 0.0
    for dp in rows:
        turns = dp["choices"][0]["turns"]
        tokens = sum(max(0, len(tokenizer(t).input_ids) - 1) for t in turns)
        time_s = float(sum(dp["choices"][0]["wall_time"]))
        if time_s > 0:
            speeds0.append(tokens / time_s)
        total_token += tokens
        total_time += time_s
    return speeds0, total_token, total_time


def save_csv(csv_path: str, model_ids: List[str],
             ratios: List[Optional[float]], token_idx_ratios: List[Optional[float]]):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Model ID", "Speed Ratio (EA/Base)", "New Tokens / Idxs"])
        for mid, r, t in zip(model_ids, ratios, token_idx_ratios):
            w.writerow([
                mid,
                f"{r:.8f}" if r is not None else "N/A",
                f"{t:.8f}" if t is not None else "N/A",
            ])


def main():
    args = parse_args()
    ea_pat_def, base_pat_def, tok_def, csv_def = derive_defaults(
        args.prefix, args.bench_name, args.temperature
    )
    ea_pat = args.ea_file_pattern or ea_pat_def
    base_pat = args.base_file_pattern or base_pat_def
    tok_path = args.tokenizer_path or tok_def
    csv_path = args.csv_path or csv_def

    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    model_ids = [m.strip() for m in args.model_ids.split(",") if m.strip()]

    ratios = [None] * len(model_ids)
    token_idx_ratios = [None] * len(model_ids)

    for i, mid in enumerate(model_ids):
        ea_file = ea_pat.format(mid)
        base_file = base_pat.format(mid)

        if not os.path.exists(ea_file):
            print(f"[miss] EA:   {ea_file}")
            continue
        if not os.path.exists(base_file):
            print(f"[miss] BASE: {base_file}")
            continue

        sp_ea, new_tokens, idxs = compute_ea_stats(ea_file)
        if not sp_ea:
            print(f"[warn] EA speeds empty: {ea_file}")
            continue
        sp_b, _, _ = compute_base_stats(base_file, tokenizer)
        if not sp_b:
            print(f"[warn] BASE speeds empty: {base_file}")
            continue

        mean_ea = float(np.mean(sp_ea))
        mean_b = float(np.mean(sp_b))
        ratios[i] = (mean_ea / mean_b) if mean_b > 0 else None
        token_idx_ratios[i] = (np.sum(new_tokens) / np.sum(idxs)) if np.sum(idxs) else None

    save_csv(csv_path, model_ids, ratios, token_idx_ratios)
    print(f"[done] CSV: {csv_path}")


if __name__ == "__main__":
    main()
