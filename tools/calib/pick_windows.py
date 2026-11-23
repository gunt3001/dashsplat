#!/usr/bin/env python3
"""Plot the calibration drive and suggest the six time windows pipeline/calib uses.

Calibration does not use the whole drive. features.py takes keyframes from six windows:
five to fit the rig, then one held-out stretch that scale.py reconstructs blind to check the
metric scale. This script reads data/drive/timeline.npz (written by ingest), suggests:

  * 5 fit windows of 6-12 s, all moving, at least two of them through a turn,
  * 1 held-out window of about 15 m of straight driving, ideally pulling away from a stop,

and draws speed, turn rate and the GPS track with the windows shaded, so you can check them.

Usage (from the repository root, after ingest):

    python tools/calib/pick_windows.py            # plot + print suggestion -> work/calib/windows.png
    python tools/calib/pick_windows.py --write    # also store the suggestion in calib/car.json
    python tools/calib/pick_windows.py --show     # plot whatever calib/car.json already has

Each window is [start_s, end_s, spacing_m]: one keyframe every spacing_m metres travelled.
"""

import argparse, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import carfile

MOVING = 1.5  # m/s; every sample in a window must be at least this fast
MAX_FIT_M = 80  # longer fit windows only add keyframes (the author's were 10-90 m, ~30 keyframes each)
GAP = 3.0  # s between windows
trapz = getattr(np, "trapezoid", None) or np.trapz  # renamed in NumPy 2


def windows_moving(t, v, a, b):
    s = (t >= a) & (t < b)
    return s.any() and v[s].min() >= MOVING


def stats(t, v, yaw, a, b):
    s = (t >= a) & (t < b)
    dist = float(trapz(v[s], t[s]))
    turn = float(np.degrees(abs(yaw[s][-1] - yaw[s][0])))
    return dist, turn


def overlaps(a, b, taken, gap=GAP):
    return any(a < y + gap and b > x - gap for x, y, *_ in taken)


def suggest(t, v, yaw):
    T = t[-1]
    # Held-out: prefer a departure (speed rising through 0.5 m/s), else any straight stretch.
    held = None
    starts = t[1:][(v[:-1] < 0.5) & (v[1:] >= 0.5)]
    for t0 in starts:
        for L in np.arange(3.5, 6.01, 0.25):
            a, b = t0 + 0.3, t0 + 0.3 + L
            if b > T:
                break
            d, turn = stats(t, v, yaw, a, b)
            if d >= 14 and turn < 10:
                held = (a, b, "departure")
                break
        if held:
            break
    if held is None:
        for a in np.arange(0, T - 4, 0.5):
            for L in (3.5, 4.0, 5.0, 6.0):
                b = a + L
                if b <= T and windows_moving(t, v, a, b):
                    d, turn = stats(t, v, yaw, a, b)
                    if 12 <= d <= 22 and turn < 8:
                        held = (a, b, "straight")
                        break
            if held:
                break
    if held is None:
        sys.exit("Could not find a held-out stretch (~15 m straight). Is the drive long enough and moving?")

    # Fit windows: every moving 6-12 s candidate, scored; two turns first, then spread out.
    cands = []
    for a in np.arange(0, T - 6, 0.5):
        for L in (6, 8, 10, 12):
            b = a + L
            if b <= T and windows_moving(t, v, a, b):
                d, turn = stats(t, v, yaw, a, b)
                if d <= MAX_FIT_M:
                    cands.append((a, b, d, turn))
    taken = [(held[0], held[1])]
    fit = []
    for c in sorted([c for c in cands if c[3] >= 15], key=lambda c: -c[3]):
        if len([f for f in fit if f[3] >= 15]) >= 2:
            break
        if not overlaps(c[0], c[1], taken):
            fit.append(c); taken.append(c)
    if len([f for f in fit if f[3] >= 15]) < 2:
        print("WARNING: fewer than two windows with a turn of 15 deg or more; calibration of the car's "
              "axes will be weak. Drive a route with a few 90 deg turns.")
    rest = [c for c in cands if not overlaps(c[0], c[1], taken)]
    while len(fit) < 5 and rest:
        # farthest (in time) from what is already chosen, longer windows preferred
        far = lambda c: min(abs((c[0] + c[1]) / 2 - (x + y) / 2) for x, y, *_ in taken) + c[1] - c[0]
        c = max(rest, key=far)
        fit.append(c); taken.append(c)
        rest = [r for r in rest if not overlaps(r[0], r[1], taken)]
    if len(fit) < 5:
        sys.exit(f"Only found {len(fit)} fit windows; the drive needs more moving time.")
    fit.sort()
    out = []
    for a, b, d, turn in fit:
        out.append([round(float(a), 1), round(float(b), 1), round(float(np.clip(d / 30, 0.6, 2.5)), 2)])
    d, _ = stats(t, v, yaw, held[0], held[1])
    out.append([round(float(held[0]), 1), round(float(held[1]), 1), round(float(np.clip(d / 18, 0.6, 1.5)), 2)])
    return out, held[2]


def plot(t, v, yaw, tl, windows, dest):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rate = np.degrees(np.gradient(yaw, t))
    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1])
    ax1, ax2, ax3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])
    ax1.plot(t, v * 3.6, lw=0.8, color="#333")
    ax1.set_ylabel("speed (km/h)")
    ax2.plot(t, np.clip(rate, -40, 40), lw=0.8, color="#333")
    ax2.set_ylabel("turn rate (deg/s)")
    ax2.set_xlabel("drive time (s)")
    x, y = tl["x"], tl["y"]
    ax3.plot(x, y, lw=0.8, color="#bbb")
    ax3.set_aspect("equal")
    ax3.set_title("dead-reckoned track (m)")
    for i, (a, b, sp) in enumerate(windows):
        held = i == len(windows) - 1
        col = "#d62728" if held else "#1f77b4"
        lab = "held-out" if held else f"fit {i + 1}"
        for ax in (ax1, ax2):
            ax.axvspan(a, b, color=col, alpha=0.25)
        ax1.text((a + b) / 2, ax1.get_ylim()[1] * 0.95, lab, ha="center", va="top", fontsize=8, color=col)
        s = (t >= a) & (t < b)
        ax3.plot(x[s], y[s], lw=3, color=col)
        ax3.annotate(lab, (x[s][0], y[s][0]), fontsize=8, color=col)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=110)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rs", default=os.getenv("RS", "."), help="installation root (default: $RS or .)")
    ap.add_argument("--write", action="store_true", help="store the suggestion in calib/car.json")
    ap.add_argument("--show", action="store_true", help="plot the windows already in calib/car.json")
    a = ap.parse_args()
    rs = Path(a.rs).resolve()
    tl = np.load(rs / "data/drive/timeline.npz")
    t, v, yaw = tl["t"], tl["speed"], np.unwrap(tl["yaw"])
    moving = float(np.sum(np.diff(t)[v[1:] > 0.5]))
    print(f"Drive: {t[-1]:.0f} s, {moving:.0f} s moving, {float(trapz(v, t)):.0f} m")

    if a.show:
        windows = carfile.load(rs).get("windows")
        if not windows:
            sys.exit("calib/car.json has no windows yet; run without --show first")
    else:
        windows, kind = suggest(t, v, yaw)
        print(f"Suggested windows ([start_s, end_s, spacing_m]; last = held-out {kind}):")
    for i, w in enumerate(windows):
        d, turn = stats(t, v, yaw, w[0], w[1])
        tag = "held-out" if i == len(windows) - 1 else f"fit {i + 1}  "
        print(f"  {tag}  {w}   {d:5.0f} m, turn {turn:4.0f} deg, ~{int(d / w[2])} keyframes")
    dest = rs / "work/calib/windows.png"
    plot(t, v, yaw, tl, windows, dest)
    print(f"Plot: {dest}")
    if a.write and not a.show:
        car = carfile.load(rs, create=True)
        car["windows"] = windows
        print(f"Saved windows to {carfile.save(rs, car)}")


if __name__ == "__main__":
    main()
