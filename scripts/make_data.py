"""Generates the made-up labeled work orders in data/.

  train.csv           240 notes written with the "seen" vocabulary
  test.csv            120 notes: 60 more with seen wording and 60 with wording that never appears in train
                      ("weeping" for a leak, "pillow block" for a bearing), the way a new site writes differently

Labels come from construction, so they are exact. Run it again and you get the same files (fixed seed).
"""
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wofc.taxonomy import PAIRS  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "data"

COMPONENT_WORDS = {
    "seen": {
        "bearing": ["bearing", "brg", "bearings", "drive end bearing"],
        "belt": ["belt", "drive belt", "v-belt"],
        "motor": ["motor", "mtr", "drive motor"],
        "hydraulic hose": ["hyd hose", "hydraulic hose", "hose"],
        "seal": ["seal", "mech seal", "shaft seal"],
        "sensor": ["sensor", "prox", "photo eye"],
        "gearbox": ["gearbox", "gear box", "reducer"],
        "coupling": ["coupling", "cplg", "jaw coupling"],
    },
    "new": {
        "bearing": ["pillow block", "roller race"],
        "belt": ["timing strap", "poly-v"],
        "motor": ["servo", "prime mover"],
        "hydraulic hose": ["hyd line", "pressure line"],
        "seal": ["o-ring", "packing gland"],
        "sensor": ["transducer", "limit switch"],
        "gearbox": ["gear case", "gear drive"],
        "coupling": ["spider insert", "shaft connector"],
    },
}
FAILURE_WORDS = {
    "seen": {
        "wear": ["worn", "worn out", "showing wear"],
        "leak": ["leaking", "dripping", "leaking oil"],
        "overheating": ["running hot", "overheating", "hot to touch"],
        "misalignment": ["misaligned", "out of alignment", "tracking off"],
        "breakage": ["broken", "snapped", "cracked"],
        "electrical": ["tripped", "shorted", "no power"],
        "contamination": ["contaminated", "dirty", "full of debris"],
        "loose": ["loose", "slack", "came loose"],
    },
    "new": {
        "wear": ["chewed up", "glazed", "pitted"],
        "leak": ["weeping", "seeping", "puddling"],
        "overheating": ["smoking", "discolored from heat", "reading high on IR gun"],
        "misalignment": ["walking off", "crooked", "high runout"],
        "breakage": ["fractured", "split", "in pieces"],
        "electrical": ["kicked the breaker", "lost a phase", "failed megger test"],
        "contamination": ["gunked up", "sludged", "full of grit"],
        "loose": ["backed out", "wobbling", "has play"],
    },
}
ASSETS = ["CV-3", "CV-5", "P-101", "PU-7", "GB-12", "F-2", "M-4", "EX-1", "PR-9", "CT-1"]
ACTIONS = ["replaced", "repl", "adjusted", "tightened", "cleaned", "reset, monitoring", "parts ordered",
           "tagged out", "wo to follow", "will chk next shift", "temp fix, needs PM"]
TEMPLATES = [
    "{comp} on {asset} {fail}, {action}",
    "{asset} {comp} {fail}. {action}",
    "found {comp} {fail} on {asset}, {action}",
    "{asset}: {comp} {fail} - {action}",
    "ops reported {asset} down, {comp} {fail}, {action}",
]


def typo(word: str, rng: random.Random) -> str:
    if len(word) < 5:
        return word
    i = rng.randrange(1, len(word) - 2)
    return word[:i] + word[i + 1] + word[i] + word[i + 2:]


def note(component: str, failure: str, vocab: str, rng: random.Random) -> str:
    text = rng.choice(TEMPLATES).format(comp=rng.choice(COMPONENT_WORDS[vocab][component]),
                                        fail=rng.choice(FAILURE_WORDS[vocab][failure]),
                                        asset=rng.choice(ASSETS), action=rng.choice(ACTIONS))
    if rng.random() < 0.15:  # a typo in one of the action words, never in the label words
        words = text.split()
        j = len(words) - 1
        words[j] = typo(words[j], rng)
        text = " ".join(words)
    return text if rng.random() < 0.6 else text.upper() if rng.random() < 0.2 else text.capitalize()


def rows(n: int, vocab: str, rng: random.Random, start: int, split: str) -> list[dict]:
    out = []
    for i in range(n):
        component, failure = PAIRS[i % len(PAIRS)] if i < len(PAIRS) else rng.choice(PAIRS)
        out.append({"id": f"WO{start + i}", "text": note(component, failure, vocab, rng), "component": component,
                    "failure": failure, "wording": vocab, "split": split})
    rng.shuffle(out)
    return out


def main() -> None:
    rng = random.Random(20261120)
    OUT.mkdir(exist_ok=True)
    train = rows(240, "seen", rng, 10000, "train")
    test = rows(60, "seen", rng, 20000, "test") + rows(60, "new", rng, 30000, "test")
    for name, data in (("train.csv", train), ("test.csv", test)):
        with open(OUT / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
        print(f"data/{name}: {len(data)} rows")


if __name__ == "__main__":
    main()
