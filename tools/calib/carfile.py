"""Read and write calib/car.json (the per-car calibration settings) in a compact, hand-editable layout."""

import json, re, shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "pipeline/calib/car.example.json"


def path(rs):
    return Path(rs) / "calib/car.json"


def load(rs, create=False):
    p = path(rs)
    if not p.exists():
        if not create:
            return {}
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(EXAMPLE, p)
        print(f"Created {p} from {EXAMPLE.relative_to(REPO)}")
    return json.loads(p.read_text())


def dumps(car):
    txt = json.dumps(car, indent=2)
    num = r"-?[\d.]+(?:e-?\d+)?"
    # Put each point, each short number list and each polygon on one line.
    txt = re.sub(r"\[\s+((?:%s,\s+)*%s)\s+\]" % (num, num), lambda m: "[" + re.sub(r",\s+", ", ", m.group(1)) + "]", txt)
    pt = r"\[%s(?:, %s)*\]" % (num, num)
    txt = re.sub(r"\[\s+(%s(?:,\s+%s)*)\s+\]" % (pt, pt), lambda m: "[" + re.sub(r",\s+\[", ", [", m.group(1)) + "]", txt)
    return txt + "\n"


def save(rs, car):
    p = path(rs)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(dumps(car))
    return p
