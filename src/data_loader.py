"""Streaming reader for candidates.jsonl / candidates.jsonl.gz.

Design goal: never materialize the full candidate set in memory. Callers
(rank.py) stream through the file, compute a lightweight score per record,
and discard the heavy record. A second, separate stream pass is used later
to re-fetch full detail only for the ~100 candidates that make the cut.
"""

import gzip
import json
import sys
from typing import Any, Dict, Iterator, Optional


def open_candidates_file(path: str):
    """Open a candidates file for text reading, transparently handling gzip
    based on the file extension."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def stream_candidates(path: str) -> Iterator[Dict[str, Any]]:
    """Yield one candidate dict at a time. Malformed JSON lines are skipped
    (with a warning to stderr) rather than crashing the whole run - a
    hackathon dataset with a few bad rows shouldn't sink the submission."""
    with open_candidates_file(path) as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[data_loader] skipping malformed line {line_num}: {e}", file=sys.stderr)
                continue
            if not isinstance(record, dict):
                print(f"[data_loader] skipping non-object line {line_num}", file=sys.stderr)
                continue
            yield record


def get_candidate_id(candidate: Dict[str, Any]) -> Optional[str]:
    """candidate_id lives under profile.candidate_id per the schema, but we
    fall back to a top-level candidate_id if a record is shaped differently."""
    profile = candidate.get("profile") or {}
    cid = profile.get("candidate_id") or candidate.get("candidate_id")
    return str(cid) if cid is not None else None