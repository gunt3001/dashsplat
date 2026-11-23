#!/usr/bin/env python3
"""Make the images you need to draw the body masks, and check the masks you drew.

Each camera sees parts of the car (bodywork, the rear camera housing, trim, dark lens
borders). Calibration must ignore those pixels, because they move with the car and would
fool the feature matcher. This script takes the per-pixel median of ~60 frames spread over
the moving parts of the drive: the world blurs away and the car's own body stays sharp.

For each camera it writes, under work/calib/mask-sheets/:

  median-<cam>.png    the median image at 600x389, the coordinate space car.json masks use
  overlay-<cam>.png   the same, enlarged 2x with a labelled 50 px grid, and the current mask
                      in red (from calib/car.json, or the upstream Model 3 defaults)
  all.png             the six overlays on one sheet

Usage (from the repository root, after ingest):

    python tools/calib/mask_sheets.py
    python tools/calib/mask_sheets.py --frames 100     # more frames, cleaner median

Edit "masks" in calib/car.json and re-run to check. Points are [x, y] on the 600x389 grid,
which is the same for every camera, whatever its real resolution.
"""

import argparse, json, os, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
SIZE = (600, 389)
UP = 2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rs", default=os.getenv("RS", "."), help="installation root (default: $RS or .)")
    ap.add_argument("--frames", type=int, default=60, help="frames per camera in the median (default 60)")
    a = ap.parse_args()
    rs = Path(a.rs).resolve()
    os.environ["RS"] = str(rs)  # features.py reads RS and calib/car.json at import
    sys.path.insert(0, str(REPO / "pipeline/calib"))
    import cv2 as cv
    import features

    data = rs / "data/drive"
    out = rs / "work/calib/mask-sheets"
    out.mkdir(parents=True, exist_ok=True)
    tl = np.load(data / "timeline.npz")
    moving = tl["t"][tl["speed"] > 3.0]
    if len(moving) < a.frames:
        sys.exit("Not enough moving footage (speed > 3 m/s) in data/drive")
    times = moving[np.linspace(0, len(moving) - 1, a.frames).astype(int)]
    custom = set(features.car("masks", {}))
    print(f"Masks from {features.CAR_FILE if custom else 'upstream Model 3 defaults'}"
          + (f" (cameras without an entry use the defaults: {sorted(set(features.CAMS) - custom)})" if custom and custom != set(features.CAMS) else ""))

    sheets = []
    for cam in features.CAMS:
        rows = json.load(open(data / "cams" / f"{cam}.json"))
        ct = np.array([r["t"] for r in rows])
        stack = []
        for t in times:
            n = rows[int(np.argmin(abs(ct - t)))]["n"]
            im = cv.imread(str(data / "frames" / cam / f"{n:06d}.jpg"))
            if im is not None:
                stack.append(cv.resize(im, SIZE, interpolation=cv.INTER_AREA))
        med = np.median(np.stack(stack), axis=0).astype(np.uint8)
        cv.imwrite(str(out / f"median-{cam}.png"), med)

        big = cv.resize(med, (SIZE[0] * UP, SIZE[1] * UP), interpolation=cv.INTER_CUBIC)
        mask = features.mask_image(cam, *big.shape[1::-1])
        red = big.copy()
        red[mask == 0] = (0.45 * red[mask == 0] + 0.55 * np.array([40, 40, 230])).astype(np.uint8)
        edges = cv.Canny(mask, 50, 150)
        red[edges > 0] = (0, 0, 255)
        for x in range(0, SIZE[0] + 1, 50):
            cv.line(red, (x * UP, 0), (x * UP, red.shape[0]), (255, 255, 0), 1)
            cv.putText(red, str(x), (x * UP + 2, 14), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        for y in range(0, SIZE[1] + 1, 50):
            cv.line(red, (0, y * UP), (red.shape[1], y * UP), (255, 255, 0), 1)
            cv.putText(red, str(y), (2, y * UP - 3), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        cv.putText(red, cam, (10, red.shape[0] - 12), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv.imwrite(str(out / f"overlay-{cam}.png"), red)
        sheets.append(red)
        print(f"  {cam:15s} median of {len(stack)} frames")

    grid = np.vstack([np.hstack(sheets[i:i + 2]) for i in range(0, 6, 2)])
    cv.imwrite(str(out / "all.png"), grid)
    print(f"Wrote {out}/ (open all.png first)")


if __name__ == "__main__":
    main()
