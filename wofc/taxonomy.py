"""The label set: which component failed, and how. Only the listed pairs exist (a sensor does not leak)."""

COMPONENTS = ("bearing", "belt", "motor", "hydraulic hose", "seal", "sensor", "gearbox", "coupling")
FAILURES = ("wear", "leak", "overheating", "misalignment", "breakage", "electrical", "contamination", "loose")

VALID = {
    "bearing": ("wear", "overheating", "contamination", "misalignment"),
    "belt": ("wear", "misalignment", "breakage", "loose"),
    "motor": ("overheating", "electrical"),
    "hydraulic hose": ("leak", "wear", "breakage"),
    "seal": ("leak", "wear"),
    "sensor": ("electrical", "contamination", "misalignment"),
    "gearbox": ("leak", "overheating", "wear", "contamination"),
    "coupling": ("misalignment", "wear", "breakage", "loose"),
}
PAIRS = [(c, f) for c in COMPONENTS for f in VALID[c]]


def is_valid(component: str, failure: str) -> bool:
    return failure in VALID.get(component, ())
