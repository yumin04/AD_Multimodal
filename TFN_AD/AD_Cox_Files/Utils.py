from __future__ import annotations
from typing import List, Dict, Any
import pickle
import numpy as np

import os, json

# --------------------------
# Save/Load utilities
# --------------------------
def save_sequences(path: str, seqs: List["PatientSeq"]) -> None:
    with open(path, "wb") as f:
        pickle.dump(seqs, f)

def load_sequences(path: str) -> List["PatientSeq"]:
    with open(path, "rb") as f:
        return pickle.load(f)

# --------------------------
# Sanity checks
# --------------------------
def sanity_check_sequences(seqs: List["PatientSeq"]) -> Dict[str, Any]:
    """
    Basic report: checks time monotonicity, sequence lengths, event rate, and counts violations.
    """
    report = {
        "num_patients": len(seqs),
        "avg_T": None,
        "min_T": None,
        "max_T": None,
        "event_rate": None,
        "violations": 0
    }
    if not seqs:
        return report

    lengths = [len(s.times) for s in seqs]
    report["avg_T"] = float(np.mean(lengths))
    report["min_T"] = int(np.min(lengths))
    report["max_T"] = int(np.max(lengths))
    report["event_rate"] = float(np.mean([s.event for s in seqs]))

    v = 0
    for s in seqs:
        if (np.diff(s.times) < -1e-8).any():
            v += 1
    report["violations"] = v
    return report

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def assert_disjoint(a, b, c) -> None:
    A, B, C = set(a), set(b), set(c)
    assert A.isdisjoint(B) and A.isdisjoint(C) and B.isdisjoint(C), "Splits are not disjoint!"
