#!/usr/bin/env python3
"""
Redrob Intelligent Candidate Discovery & Ranking - Stage 1 (rule-based, CPU-only)

Usage:
    python rank.py --candidates candidates.jsonl --out outputs/submission.csv
    python rank.py --candidates candidates.jsonl.gz --out outputs/submission.csv

Design:
    Pass 1 streams every candidate once, computes a lightweight
    (candidate_id, score, subscores, trap_flags) record, and discards the
    heavy JSON. This keeps memory flat regardless of dataset size.

    After pass 1, all candidates are sorted by score (desc), tie-broken by
    candidate_id (asc), and the top 100 IDs are selected.

    Pass 2 streams the file a second time and, for only those 100 IDs,
    re-extracts full features so the reasoner can write factual,
    candidate-specific reasoning text.

No network access and no ML models are used - this is pure rule-based
scoring, so a 100k-row file comfortably finishes well inside 5 minutes on
CPU.
"""

import argparse
import csv
import sys
import time
from typing import Any, Dict, List, Tuple

from src.data_loader import get_candidate_id, stream_candidates
from src.feature_extractor import extract_features
from src.reasoner import generate_reasoning
from src.scorer import score_candidate
from src.utils import sort_key_for_ranking

TOP_N = 100


def pass_one_score_all(path: str) -> List[Tuple[str, float, Dict[str, float], List[str]]]:
    """Stream every candidate once, returning lightweight scored records."""
    results = []
    count = 0
    for candidate in stream_candidates(path):
        cid = get_candidate_id(candidate)
        if cid is None:
            continue
        features = extract_features(candidate)
        score, subscores = score_candidate(features)
        results.append((cid, score, subscores, features["trap_flags"]))
        count += 1
        if count % 20000 == 0:
            print(f"[rank] scored {count} candidates so far...", file=sys.stderr)
    print(f"[rank] finished scoring {count} candidates.", file=sys.stderr)
    return results


def pass_two_collect_top(path: str, top_ids: set) -> Dict[str, Dict[str, Any]]:
    """Second streaming pass: re-extract full features only for the
    candidates that made the top-N cut, so reasoning text can cite facts."""
    collected: Dict[str, Dict[str, Any]] = {}
    for candidate in stream_candidates(path):
        cid = get_candidate_id(candidate)
        if cid is None or cid not in top_ids:
            continue
        collected[cid] = extract_features(candidate)
        if len(collected) == len(top_ids):
            break
    return collected


def build_top_100(all_scores: List[Tuple[str, float, Dict[str, float], List[str]]]):
    ranked = sorted(
        all_scores,
        key=lambda r: (-round(float(r[1]), 4), str(r[0]))
    )
    return ranked[:TOP_N]


def enforce_monotonic_scores(rows: List[Dict[str, Any]]) -> None:
    """Guarantee non-increasing score across ranks 1..100, even in the rare
    case of float rounding causing an out-of-order pair after tie-breaking."""
    for i in range(1, len(rows)):
        if rows[i]["score"] > rows[i - 1]["score"]:
            rows[i]["score"] = rows[i - 1]["score"]


def write_csv(rows: List[Dict[str, Any]], out_path: str) -> None:
    fieldnames = ["candidate_id", "rank", "score", "reasoning"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Redrob Stage 1 candidate ranker")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl or candidates.jsonl.gz")
    parser.add_argument("--out", required=True, help="Path to write the output submission.csv")
    args = parser.parse_args()

    start = time.time()

    print("[rank] pass 1: scoring all candidates...", file=sys.stderr)
    all_scores = pass_one_score_all(args.candidates)

    if not all_scores:
        print("[rank] ERROR: no valid candidates found in input file.", file=sys.stderr)
        sys.exit(1)

    top_100 = build_top_100(all_scores)
    top_ids = {cid for cid, _, _, _ in top_100}

    print(f"[rank] pass 2: re-fetching detail for top {len(top_ids)} candidates...", file=sys.stderr)
    detail_by_id = pass_two_collect_top(args.candidates, top_ids)

    rows = []
    for rank_idx, (cid, score, subscores, trap_flags) in enumerate(top_100, start=1):
        features = detail_by_id.get(cid)
        if features is None:
            # Shouldn't happen, but fail safe with a minimal reasoning string
            reasoning = "Candidate selected based on overall composite score."
        else:
            reasoning = generate_reasoning(features, rank_idx)
        rows.append({
            "candidate_id": cid,
            "rank": rank_idx,
            "score": round(score, 4),
            "reasoning": reasoning,
        })

    enforce_monotonic_scores(rows)

    write_csv(rows, args.out)

    elapsed = time.time() - start
    print(f"[rank] wrote {len(rows)} rows to {args.out} in {elapsed:.1f}s", file=sys.stderr)

    if elapsed > 300:
        print("[rank] WARNING: exceeded the 5 minute budget.", file=sys.stderr)


if __name__ == "__main__":
    main()