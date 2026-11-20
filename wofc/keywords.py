"""Baseline 1: keyword rules, written from the words that appear in the training notes.

This is what a reliability engineer writes first: a dictionary per label. It is exact on wording it knows and
silent on wording it does not, which is why it reports "no match" instead of guessing.
"""
from __future__ import annotations

import re

COMPONENT_KEYWORDS = {
    "bearing": ["bearing", "brg"],
    "belt": ["belt"],
    "motor": ["motor", "mtr"],
    "hydraulic hose": ["hose"],
    "seal": ["seal"],
    "sensor": ["sensor", "prox", "photo eye"],
    "gearbox": ["gearbox", "gear box", "reducer"],
    "coupling": ["coupling", "cplg"],
}
FAILURE_KEYWORDS = {
    "wear": ["worn", "wear"],
    "leak": ["leak", "dripping"],
    "overheating": ["hot", "overheat"],
    "misalignment": ["misalign", "alignment", "tracking"],
    "breakage": ["broken", "snapped", "cracked"],
    "electrical": ["tripped", "shorted", "no power"],
    "contamination": ["contaminat", "dirty", "debris"],
    "loose": ["loose", "slack"],
}


def _first(text: str, table: dict[str, list[str]]) -> str | None:
    """The label whose keyword appears earliest in the note; None when nothing matches."""
    best, pos = None, len(text) + 1
    for label, words in table.items():
        for w in words:
            m = re.search(r"\b" + re.escape(w), text)
            if m and m.start() < pos:
                best, pos = label, m.start()
    return best


def classify(text: str) -> dict:
    t = text.lower()
    return {"component": _first(t, COMPONENT_KEYWORDS), "failure": _first(t, FAILURE_KEYWORDS)}
