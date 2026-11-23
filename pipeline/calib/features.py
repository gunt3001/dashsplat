"""Select synchronized keyframes, apply measured body masks, and extract/match RootSIFT."""

import os, sys, json, itertools
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2 as cv
import numpy as np

cv.setNumThreads(1)
RS = Path(os.getenv("RS", ".")).resolve()
DATA = RS / "data/drive"
W = RS / "work/calib"
OUT = RS / "calib"
CAMS = ["front", "back", "left_pillar", "right_pillar", "left_repeater", "right_repeater"]

# Per-car, per-drive settings. Defaults below are the author's Model 3 drive; a new car
# overrides any of them in $RS/calib/car.json (template: pipeline/calib/car.example.json).
CAR_FILE = Path(os.getenv("CALIB_CAR", str(OUT / "car.json")))
CAR = json.load(open(CAR_FILE)) if CAR_FILE.exists() else {}


def car(key, default):
    return CAR.get(key, default)


# World-valid polygons in the 600x389 temporal median contact sheets. At the back
# both boundaries are car/camera housing, not the limits of a rectangular sensor.
# car.json "masks": {cam: {"keep": [[x, y], ...], "remove": [[[x, y], ...], ...]}}, same
# 600x389 coordinates. "keep" (optional) masks everything outside one polygon; each
# "remove" polygon is masked out. A camera missing from car.json uses the defaults.
def mask_image(cam, w, h, spec=None):
    m = np.full((h, w), 255, np.uint8)
    poly = lambda ps, color: cv.fillPoly(
        m, [np.rint(np.array(ps) * [w / 600, h / 389]).astype(np.int32)], color
    )
    spec = car("masks", {}).get(cam) if spec is None else spec
    if spec is not None:
        if spec.get("keep"):
            m[:] = 0
            poly(spec["keep"], 255)
        for ps in spec.get("remove", []):
            poly(ps, 0)
    else:
        if cam == "back":
            m[:] = 0
            poly(
                [
                    (4, 177),
                    (12, 146),
                    (28, 119),
                    (50, 91),
                    (80, 63),
                    (115, 40),
                    (154, 21),
                    (204, 7),
                    (234, 3),
                    (360, 3),
                    (400, 10),
                    (443, 27),
                    (486, 50),
                    (524, 78),
                    (551, 106),
                    (574, 138),
                    (590, 177),
                    (595, 208),
                    (595, 245),
                    (578, 270),
                    (546, 288),
                    (512, 301),
                    (460, 312),
                    (404, 322),
                    (347, 326),
                    (285, 328),
                    (230, 324),
                    (175, 316),
                    (125, 305),
                    (76, 293),
                    (43, 281),
                    (20, 267),
                    (5, 247),
                ],
                255,
            )
        elif cam == "left_repeater":
            poly(
                [
                    (0, 0),
                    (151, 0),
                    (136, 8),
                    (46, 7),
                    (8, 44),
                    (19, 84),
                    (37, 150),
                    (39, 240),
                    (52, 316),
                    (68, 389),
                    (0, 389),
                ],
                0,
            )
        elif cam == "right_repeater":
            poly(
                [
                    (600, 0),
                    (447, 0),
                    (460, 17),
                    (479, 24),
                    (545, 14),
                    (569, 12),
                    (594, 39),
                    (594, 70),
                    (569, 148),
                    (568, 237),
                    (549, 389),
                    (600, 389),
                ],
                0,
            )
        elif cam == "left_pillar":
            poly([(600, 328), (595, 351), (580, 373), (556, 389), (600, 389)], 0)
        elif cam == "right_pillar":
            poly([(0, 325), (8, 350), (21, 370), (44, 389), (0, 389)], 0)
        elif cam == "front":
            poly(
                [(0, 380), (115, 384), (260, 387), (440, 385), (600, 378), (600, 389), (0, 389)], 0
            )
    m[:3] = 0
    m[-3:] = 0
    m[:, :3] = 0
    m[:, -3:] = 0
    return m


def masks():
    meta = json.load(open(DATA / "meta.json"))
    (OUT / "masks").mkdir(exist_ok=True, parents=True)
    for cam in CAMS:
        w, h = meta["sizes"][cam]
        cv.imwrite(str(OUT / "masks" / f"{cam}.png"), mask_image(cam, w, h))


def select():
    tl = np.load(DATA / "timeline.npz")
    rows = [json.load(open(DATA / "cams" / f"{c}.json")) for c in CAMS]
    ts = [np.array([r["t"] for r in rr]) for rr in rows]
    frames = []
    # car.json "windows": [[start_s, end_s, spacing_m], ...]; the LAST one is the held-out
    # blind scale check (seg 5 below: solve.py fits seg < 5, scale.py uses seg == 5).
    segs = car("windows", None) or [
        (0, 4, 0.7),
        (82, 94, 0.85),
        (95, 104, 1.05),
        (104.1, 111.4, 1.15),
        (318.5, 325, 1.7),
        (231.6, 235.4, 0.85),
    ]
    if len(segs) != 6:
        sys.exit(f"{CAR_FILE}: 'windows' needs exactly 6 entries (5 fit windows, then 1 held-out), got {len(segs)}")
    for s, (a, b, spacing) in enumerate(segs):
        last = None
        for i in np.flatnonzero((tl["t"] >= a) & (tl["t"] < b) & (tl["speed"] > 0.5)):
            xy = np.array([tl["x"][i], tl["y"][i]])
            t = tl["t"][i]
            if last is not None and np.linalg.norm(xy - last) < spacing:
                continue
            inds = [np.argmin(abs(tt - t)) for tt in ts]
            offsets = [float(tt[j] - t) for tt, j in zip(ts, inds)]
            if max(abs(x) for x in offsets) > 0.02:
                continue
            frames.append(
                dict(
                    t=float(t),
                    seg=s,
                    xy=xy.tolist(),
                    yaw=float(tl["yaw"][i]),
                    ns=[r[j]["n"] for r, j in zip(rows, inds)],
                    dt=offsets,
                )
            )
            last = xy
    W.mkdir(exist_ok=True, parents=True)
    json.dump(frames, open(W / "frames.json", "w"))
    return frames


def extract(item):
    k, c, n = item
    dest = W / "features" / f"{k:03d}_{c}.npz"
    if dest.exists():
        return
    im = cv.imread(str(DATA / "frames" / CAMS[c] / f"{n:06d}.jpg"), 0)
    mask = cv.imread(str(OUT / "masks" / f"{CAMS[c]}.png"), 0)
    scale = 0.72 if c == 0 else 1.0
    im = cv.resize(im, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA) if scale != 1 else im
    mask = cv.resize(mask, (im.shape[1], im.shape[0]), interpolation=cv.INTER_NEAREST)
    sift = cv.SIFT_create(
        nfeatures=3000, contrastThreshold=0.016, edgeThreshold=12, enable_precise_upscale=True
    )
    kp, ds = sift.detectAndCompute(im, mask)
    p = (np.array([k.pt for k in kp]) + 0.5) / scale  # COLMAP pixel centres
    uniq = np.sort(np.unique(np.round(p * 2).astype(int), return_index=True, axis=0)[1])
    ds = ds[uniq]
    ds = np.rint(255 * np.sqrt(ds / (ds.sum(1, keepdims=True) + 1e-9))).astype(np.uint8)
    np.savez(dest, p=p[uniq].astype("f"), d=ds)


def pairs(fr):
    # Orientations of the eight candidate adjacent/forward-facing lens pairs.
    adjacent = [(0, 2), (0, 3), (2, 4), (3, 5), (1, 4), (1, 5), (1, 2), (1, 3), (2, 3), (4, 5)]
    out = set()
    for k, f in enumerate(fr):
        for step in [1, 2, 3, 5, 8]:
            j = k + step
            if j >= len(fr) or fr[j]["seg"] != f["seg"]:
                continue
            for c in range(6):
                out.add((k * 6 + c, j * 6 + c))
        for l, r in adjacent:
            for step in [-10, -5, -2, 0, 2, 5, 10]:
                j = k + step
                if not 0 <= j < len(fr) or fr[j]["seg"] != f["seg"]:
                    continue
                out.add(tuple(sorted((k * 6 + l, j * 6 + r))))
    return sorted(out)


def match(fr, split=False):
    # Process one target at a time so each FLANN index is built only once.
    allpairs = pairs(fr)
    dest = W / ("matches-split" if split else "matches")
    if split:
        needed = set()
        for parity in (0, 1):
            ix = [k for k in range(parity, len(fr), 2) if fr[k]["seg"] < 5]
            ff = [fr[k] for k in ix]
            needed.update((ix[i // 6] * 6 + i % 6, ix[j // 6] * 6 + j % 6) for i, j in pairs(ff))
        allpairs = sorted(needed.difference(allpairs))
    groups = {}
    for i, j in allpairs:
        groups.setdefault(j, []).append(i)
    F = []
    for k in range(len(fr)):
        for c in range(6):
            f = np.load(W / "features" / f"{k:03d}_{c}.npz")
            F.append((f["p"], f["d"].astype("f")))

    def group(item):
        j, ii = item
        fp = dest / f"{j:05d}.npz"
        if fp.exists():
            return
        idx = cv.flann_Index(F[j][1], dict(algorithm=1, trees=3))
        dic = {}
        for i in ii:
            matches, dist = idx.knnSearch(F[i][1], 2, params={"checks": 30})
            ids = np.flatnonzero(dist[:, 0] < (0.75 if i % 6 == j % 6 else 0.77) ** 2 * dist[:, 1])
            v = matches[ids, 0]
            if len(ids) < 9:
                continue
            # Local epipolar test on raw coordinates is intentionally permissive for fisheye; true geometry is tested after unprojection.
            _, mask = cv.findFundamentalMat(
                F[i][0][ids], F[j][0][v], cv.USAC_MAGSAC, 3.5, 0.997, 600
            )
            if mask is not None and mask.sum() >= 9:
                good = mask.ravel().astype(bool)
                dic[str(i)] = np.column_stack((ids[good], v[good])).astype(np.int32)
        np.savez(fp, **dic)

    dest.mkdir(exist_ok=True)
    list(ThreadPoolExecutor(14).map(group, groups.items()))


if __name__ == "__main__":
    if "--splits-only" in sys.argv:
        match(json.load(open(W / "frames.json")), True)
        sys.exit(0)
    masks()
    fr = select()
    print("keyframes per window", np.bincount([r["seg"] for r in fr]), flush=True)
    (W / "features").mkdir(exist_ok=True)
    list(
        ThreadPoolExecutor(12).map(
            extract, [(k, c, n) for k, f in enumerate(fr) for c, n in enumerate(f["ns"])]
        )
    )
    print("features complete", flush=True)
    match(fr)
    match(fr, True)
    print("matching complete", flush=True)
