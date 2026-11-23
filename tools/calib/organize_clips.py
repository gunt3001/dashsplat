#!/usr/bin/env python3
"""Find one continuous drive in a TeslaCam folder and lay it out the way ingest expects.

Tesla writes one file per camera per minute, named like 2026-11-20_10-15-30-front.mp4, under
TeslaCam/RecentClips, SavedClips/<event>/ and SentryClips/<event>/. pipeline/ingest/ingest.py
wants each minute in its own folder:

    <dest>/clips/<minute>/<minute>-<camera>.mp4

Usage (from the repository root):

    python tools/calib/organize_clips.py /Volumes/TESLA/TeslaCam            # list the drives found
    python tools/calib/organize_clips.py /Volumes/TESLA/TeslaCam --drive 3  # lay out drive 3 in data/clips
    python tools/calib/organize_clips.py SRC --from 2026-11-20_10-15-30 --to 2026-11-20_10-24-31

Files are symlinked by default, so keep the source mounted (or pass --copy). Before linking, the
first minute is checked: it must open, have all six cameras at the HW4 sizes, and carry Tesla's
SEI telemetry (an encrypted or pre-2025.44.25 clip fails this check).
"""

import argparse, datetime as dt, os, re, shutil, sys
from collections import defaultdict
from pathlib import Path

CAMS = ["front", "back", "left_repeater", "right_repeater", "left_pillar", "right_pillar"]
HW4 = {"front": (2896, 1876), **{c: (1448, 938) for c in CAMS[1:]}}
NAME = re.compile(r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})-(" + "|".join(CAMS) + r")\.mp4$")
GAP_S = 90  # a minute that starts more than this after the previous one starts a new drive


def scan(src):
    minutes = defaultdict(dict)
    for p in Path(src).rglob("*.mp4"):
        m = NAME.match(p.name)
        if not m or p.name.startswith("._"):
            continue
        stamp, cam = m.groups()
        # The same minute can be in RecentClips and a saved event; keep the larger copy.
        old = minutes[stamp].get(cam)
        if old is None or p.stat().st_size > old.stat().st_size:
            minutes[stamp][cam] = p
    complete = {s: c for s, c in minutes.items() if len(c) == 6}
    partial = sorted(s for s, c in minutes.items() if len(c) != 6)
    return complete, partial


def when(stamp):
    return dt.datetime.strptime(stamp, "%Y-%m-%d_%H-%M-%S")


def drives(stamps):
    out = []
    for s in sorted(stamps):
        if out and (when(s) - when(out[-1][-1])).total_seconds() <= GAP_S:
            out[-1].append(s)
        else:
            out.append([s])
    return out


def check(files):
    """Open the first minute's six files; return a list of problems (empty means fine)."""
    import av

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline/ingest"))
    from ingest import parse_packet

    problems = []
    for cam in CAMS:
        try:
            c = av.open(str(files[cam]))
        except Exception as e:  # encrypted clips usually fail right here
            problems.append(f"{cam}: cannot open ({e}); is the clip encrypted?")
            continue
        st = c.streams.video[0]
        size = (st.width, st.height)
        if size != HW4[cam]:
            problems.append(f"{cam}: {size[0]}x{size[1]}, expected HW4 {HW4[cam][0]}x{HW4[cam][1]}")
        seen = with_sei = 0
        for pkt in c.demux(st):
            if not pkt.size:
                continue
            sei, has_slice = parse_packet(bytes(pkt))
            if has_slice:
                seen += 1
                with_sei += bool(sei and "speed" in sei)
            if seen >= 90:
                break
        c.close()
        if with_sei == 0:
            problems.append(f"{cam}: no telemetry SEI in the first {seen} frames")
        else:
            print(f"  {cam:15s} {size[0]}x{size[1]}  telemetry in {with_sei}/{seen} frames")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="TeslaCam folder (or any folder containing the .mp4 files)")
    ap.add_argument("--dest", default="data", help="ingest root; clips go in DEST/clips (default: data)")
    ap.add_argument("--drive", type=int, help="number of the drive to lay out, from the listing")
    ap.add_argument("--from", dest="start", help="first minute to include, e.g. 2026-11-20_10-15-30")
    ap.add_argument("--to", dest="end", help="last minute to include")
    ap.add_argument("--copy", action="store_true", help="copy files instead of symlinking")
    ap.add_argument("--force", action="store_true", help="replace an existing DEST/clips")
    ap.add_argument("--skip-check", action="store_true", help="lay out the files even if the check fails")
    a = ap.parse_args()

    complete, partial = scan(a.src)
    if not complete:
        sys.exit(f"No complete six-camera minutes found under {a.src}")
    found = drives(complete)
    print(f"{len(complete)} complete minutes in {len(found)} drives"
          + (f" ({len(partial)} minutes missing a camera were skipped)" if partial else ""))
    for i, d in enumerate(found, 1):
        print(f"  [{i}] {d[0]} to {d[-1]}  ~{len(d)} min")

    if a.drive is None and a.start is None:
        print("\nPick one with --drive N (or --from/--to). For calibration, a 5-10 minute daytime drive with turns.")
        return
    if a.drive is not None:
        if not 1 <= a.drive <= len(found):
            sys.exit(f"--drive must be between 1 and {len(found)}")
        chosen = found[a.drive - 1]
    else:
        chosen = [s for s in sorted(complete) if s >= a.start and (a.end is None or s <= a.end)]
        if not chosen:
            sys.exit("No minutes in that range")
        if len(drives(chosen)) > 1:
            sys.exit("That range has a gap of more than 90 s; ingest needs one continuous drive")

    print(f"\nChecking the first minute, {chosen[0]}:")
    problems = check(complete[chosen[0]])
    if problems:
        print("\n".join("  PROBLEM " + p for p in problems))
        if not a.skip_check:
            sys.exit("Fix these before ingesting (see docs/calibrating-your-car.md, step 1).")
        print("  --skip-check given: laying out the files anyway")

    clips = Path(a.dest) / "clips"
    if clips.exists() and any(clips.iterdir()):
        if not a.force:
            sys.exit(f"{clips} already has clips; pass --force to replace them")
        shutil.rmtree(clips)
    for stamp in chosen:
        d = clips / stamp
        d.mkdir(parents=True, exist_ok=True)
        for cam, f in complete[stamp].items():
            out = d / f"{stamp}-{cam}.mp4"
            shutil.copy2(f, out) if a.copy else out.symlink_to(f.resolve())
    print(f"\n{'Copied' if a.copy else 'Linked'} {len(chosen)} minutes into {clips}. Next:")
    print(f"  RS=$PWD .venv/bin/python pipeline/ingest/ingest.py {a.dest}")


if __name__ == "__main__":
    main()
