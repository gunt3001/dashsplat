"""Export exactly the PLAN.md contract, figures, uncertainty and evidence report."""

from align import *
from scipy.optimize import brentq
from collections import Counter


def bundle(s):
    axes, centre, g = align(s)
    R = axes.T @ B @ Rot.from_rotvec(s.cam[:, 8:11]).as_matrix()
    pos = (s.cam[:, 11:14] - centre) @ axes
    ori = Rot.from_matrix(R @ B0.T).as_euler("ZYX", degrees=True)
    ori[:, 1] *= -1
    uv, z = s.project(s.obs, s.points[s.pi])
    err = np.linalg.norm(uv - s.pixels[s.obs], axis=1)
    c = s.obs // 3100 % 6
    fov = []
    full = []
    mask = []
    for i in range(6):
        w, h = s.sizes[i]
        p = s.cam[i]
        m = cv.imread(str(OUT / "masks" / f"{CAMS[i]}.png"), 0)
        mask.append(m)
        x = np.flatnonzero(m[int(p[3])])
        y = np.flatnonzero(m[:, int(p[2])])
        inv = lambda r: np.rad2deg(
            brentq(lambda t: t * (1 + t * t * (p[4] + t * t * p[5])) - r, 0, 1.85)
        )
        angles = lambda a, b, d: inv(abs(a - p[2 + d]) / p[d]) + inv(abs(b - p[2 + d]) / p[d])
        fov.append([angles(x[0] + 0.5, x[-1] + 0.5, 0), angles(y[0] + 0.5, y[-1] + 0.5, 1)])
        full.append([angles(0.5, w - 0.5, 0), angles(0.5, h - 0.5, 1)])
    return dict(
        R=R, pos=pos, ori=ori, g=g, err=err, c=c, fov=np.array(fov), full=np.array(full), mask=mask
    )


def transfers(s):
    # Withhold every target-camera observation when triangulating X. Never
    # threshold the target reprojection error, either for selection or stats.
    n = s.obs
    O, r = s.rays(n)
    M = np.eye(3)[None] - r[:, :, None] * r[:, None, :]
    b = np.einsum("nij,nj->ni", M, O)
    cuts = np.r_[0, np.flatnonzero(np.diff(s.pi)) + 1, len(n)]
    items = []
    for a, e in zip(cuts[:-1], cuts[1:]):
        ns = n[a:e]
        cs = ns // 3100 % 6
        ks = ns // 3100 // 6
        if cs.min() == cs.max():
            continue
        for target, k, ct in zip(ns, ks, cs):
            for src in np.unique(cs):
                if src == ct:
                    continue
                before = np.flatnonzero((cs == src) & (ks < k))
                after = np.flatnonzero((cs == src) & (ks > k))
                si = before if len(before) >= 2 else after
                if len(si) < 2:
                    continue
                si += a
                A = M[si].sum(0)
                if np.linalg.eigvalsh(A)[0] / len(si) < 0.00013:
                    continue
                X = np.linalg.solve(A, b[si].sum(0))
                rd = np.einsum("ij,ij->i", X - O[si], r[si])
                sr = (
                    np.linalg.norm(np.cross(X - O[si], r[si]), axis=1)
                    / np.maximum(rd, 0.01)
                    * s.cam[src, 0]
                )
                if np.min(rd) < 0.3 or np.max(rd) > 175 or np.sqrt(np.mean(sr**2)) > 2.5:
                    continue
                near = si[np.argmin(abs(n[si] // 3100 // 6 - k))]
                items.append([target, n[near], *X])
    data = np.array(items)
    tar = data[:, 0].astype(int)
    src = data[:, 1].astype(int)
    uv, z = s.project(tar, data[:, 2:])
    ok = z > 0.03
    return tar[ok], src[ok], uv[ok]


def debug(s, d):
    debug = OUT / "debug"
    debug.mkdir(exist_ok=True)
    tar, src, uv = transfers(s)
    err = np.linalg.norm(s.pixels[tar] - uv, axis=1)
    c = tar // 3100 % 6
    stats = []
    smallr = []
    smallo = []
    for i, name in enumerate(CAMS):
        k = int(np.argmin([abs(x["t"] - 85.5) for x in s.fr]))
        im = cv.imread(str(DATA / "frames" / name / f"{s.fr[k]['ns'][i]:06d}.jpg"))
        w, h = s.sizes[i]
        p = s.cam[i]
        KK = np.array([[p[0], 0, p[2] - 0.5], [0, p[1], p[3] - 0.5], [0, 0, 1]])
        f = p[0] * (1 if i == 0 else 0.92)
        nn = np.array([[f, 0, w / 2 - 0.5], [0, f, h / 2 - 0.5], [0, 0, 1]])
        mx, my = cv.fisheye.initUndistortRectifyMap(
            KK, p[4:8], np.eye(3), nn, (int(w), int(h)), cv.CV_32FC1
        )
        im[d["mask"][i] == 0] = 0
        rect = cv.remap(im, mx, my, cv.INTER_LINEAR)
        cv.imwrite(str(debug / f"{name}-rectified.jpg"), rect)
        smallr.append(cv.resize(rect, (420, 272)))
        use = np.flatnonzero(c == i)
        ts = tar // 3100
        ss = src // 3100
        pair = Counter(zip(ts[use].tolist(), ss[use].tolist())).most_common(1)[0][0]
        ix = np.flatnonzero((ts == pair[0]) & (ss == pair[1]))
        kt = pair[0] // 6
        ks, ci = divmod(pair[1], 6)
        sn = CAMS[ci]
        load = lambda cam, frame: cv.imread(
            str(DATA / "frames" / CAMS[cam] / f"{s.fr[frame]['ns'][cam]:06d}.jpg")
        )
        left = load(ci, ks)
        right = load(i, kt)
        wh = (900, 583)
        left = cv.resize(left, wh)
        right = cv.resize(right, wh)
        pix = lambda v, cam: tuple(np.rint((v - 0.5) * np.array(wh) / s.sizes[cam]).astype(int))
        ix = ix[np.linspace(0, len(ix) - 1, min(40, len(ix)), dtype=int)]
        for j, id in enumerate(ix):
            p1 = pix(s.pixels[src[id]], ci)
            p2 = pix(s.pixels[tar[id]], i)
            pp = pix(uv[id], i)
            cv.circle(left, p1, 5, (0, 215, 255), 1, cv.LINE_AA)
            cv.circle(right, p2, 6, (0, 215, 255), 2, cv.LINE_AA)
            cv.drawMarker(right, pp, (240, 65, 245), cv.MARKER_CROSS, 9, 2, cv.LINE_AA)
            cv.line(right, p2, pp, (240, 65, 245), 1, cv.LINE_AA)
            if j < 12:
                for image, pt in [(left, p1), (right, pp)]:
                    cv.putText(
                        image,
                        str(j + 1),
                        (pt[0] + 6, pt[1] - 4),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.43,
                        (0, 215, 255),
                        1,
                        cv.LINE_AA,
                    )
        overlay = np.vstack((np.zeros((44, 1800, 3), np.uint8), np.hstack((left, right))))
        cv.putText(
            overlay,
            f"{sn}  {s.fr[ks]['t']:.2f}s   source rays only",
            (10, 29),
            0,
            0.72,
            (240, 240, 240),
            2,
            cv.LINE_AA,
        )
        cv.putText(
            overlay,
            f"{name}  {s.fr[kt]['t']:.2f}s   gold: measured / magenta: projected",
            (910, 29),
            0,
            0.65,
            (240, 240, 240),
            2,
            cv.LINE_AA,
        )
        cv.imwrite(str(debug / f"{name}-transfer.jpg"), overlay)
        smallo.append(cv.resize(overlay, (860, 300)))
        stats.append(
            [
                len(use),
                *np.percentile(err[use], [50, 90]),
                float(np.median(err[ix])),
                float(np.max(err[ix])),
                len(ix),
            ]
        )
    cv.imwrite(
        str(debug / "rectified-contact.jpg"),
        np.vstack([np.hstack(smallr[k : k + 2]) for k in range(0, 6, 2)]),
    )
    cv.imwrite(str(debug / "transfer-contact.jpg"), np.vstack(smallo))
    return stats


def main():
    s = Scene().load()
    d = bundle(s)
    A = Scene("A").load()
    Bsplit = Scene("B").load()
    a, b = bundle(A), bundle(Bsplit)
    stats = debug(s, d)
    rig = {"cameras": {}}
    per = []
    for i, n in enumerate(CAMS):
        p = s.cam[i]
        T = np.eye(4)
        T[:3, :3] = d["R"][i]
        T[:3, 3] = d["pos"][i]
        w, h = s.sizes[i]
        rig["cameras"][n] = dict(
            model="OPENCV_FISHEYE",
            width=int(w),
            height=int(h),
            params=p[:8].tolist(),
            T_ego_cam=T.tolist(),
            time_offset_s=float(p[14]),
            rs_readout_s=float(p[15]),
            mask=f"masks/{n}.png",
        )
        e = d["err"][d["c"] == i]
        dp = A.cam[i, :8] - Bsplit.cam[i, :8]
        per.append(
            dict(
                rms=np.sqrt(np.mean(e**2)),
                obs=len(e),
                points=len(np.unique(s.pi[d["c"] == i])),
                p95=np.percentile(e, 95),
                focal=100 * np.max(abs(dp[:2]) / p[:2]),
                pp=max(abs(dp[2:4])),
                dist=max(abs(dp[4:8])),
                pos=100 * np.linalg.norm(a["pos"][i] - b["pos"][i]),
                rot=np.rad2deg(Rot.from_matrix(a["R"][i].T @ b["R"][i]).magnitude()),
                time=max(abs(A.cam[i, 14] - Bsplit.cam[i, 14]), 0) * 1000,
                rs=abs(A.cam[i, 15] - Bsplit.cam[i, 15]) * 1000,
            )
        )
    json.dump(rig, open(OUT / "rig.json", "w"), indent=2)
    lines = [
        "# Six-camera rig calibration — single HW4 drive",
        "",
        "**Result:** six footage-fitted lenses and poses. Subpixel RMS is an inlier fit statistic. Absolute axle position and independent metric scale are less certain; see below.",
        "",
        "## Method",
        "",
        "RootSIFT matches, robust pair filtering, conflict-free temporal tracks and later cross-camera joins. Joint CPU Ceres sparse-Schur BA fits per-frame 6-DoF ego corrections, metric landmarks, lens focal lengths/centres, two equidistant distortion terms (one for front), camera poses, effective exposure offsets and rolling shutter. k3/k4 are fixed at zero: high-order terms cannot be reliably separated from this drive. Pixel centres follow COLMAP: the top-left centre is (0.5, 0.5). World is ENU; camera axes right/down/forward; the exported transform maps camera to ego. No learned models or factory rig parameters. Physical mount locations and the published front focal length are initial/soft priors, not ground truth.",
        "",
        "The fit uses "
        + str(int(s.valid.sum()))
        + " synchronized rig frames from five windows (0–4, 82–94, 95–104, 104.1–111.4, 318.5–325 s). Speed-integrated displacement fixes metric scale. Heading and pose soft priors plus a no-lateral-slip and along-track increment residual define the vehicle frame. Dark borders and bodywork are manual polygons from the temporal median of moving footage. Triangulation requires ≥3 consistent observations and rejects behind-camera points. Cauchy BA is followed by 4 px and 3 px geometric outlier gates; the final timing fit does not rethreshold the observations. RMS is sqrt(mean(dx²+dy²)) in native-resolution pixels.",
        "",
        "## Rig and final bundle",
        "",
        "HFOV/VFOV first pair is the usable span along the principal axes after the body mask; next is the model span to the sensor edges. Yaw positive left, pitch positive up, roll about the forward optical direction after yaw/pitch. Positions in metres, ground origin at rear axle.",
        "",
        "|Camera|Usable H × V (°)|Sensor H × V (°)|x, y, z (m)|yaw, pitch, roll (°)|RMS / p95 (px)|inlier obs / tracks|offset / readout (ms)|",
        "|---|---:|---:|---|---|---:|---:|---:|",
    ]
    fmt = lambda v, n: ", ".join(f"{x:.{n}f}" for x in v)
    for i, n in enumerate(CAMS):
        v = per[i]
        lines.append(
            f"|{n}|{fmt(d['fov'][i], 1)}|{fmt(d['full'][i], 1)}|{fmt(d['pos'][i], 3)}|{fmt(d['ori'][i], 2)}|{v['rms']:.3f} / {v['p95']:.2f}|{v['obs']} / {v['points']}|{s.cam[i, 14] * 1000:.2f} / {s.cam[i, 15] * 1000:.2f}|"
        )
    lines += [
        "",
        "Offsets are relative to the front clock (front fixed at zero). Effective readout is top-to-bottom, referenced at the middle row. For row v, use t + offset + (v/height − 0.5) × readout. The timing model linearizes world velocity and yaw rate over an exposure; no gyro or hardware timing measurement exists. Rolling shutter and clock offsets can trade off with lens parameters and pose.",
        "",
        "## Gauge and symmetry",
        "",
    ]
    g = d["g"]
    lines += [
        f"The rear tyre corridor photometric road-plane sweep accepted {g['road_pairs']}/{g['road_total']} frame pairs. Its median height in the preliminary frame was {g['road']:.3f} m; 16th–84th percentiles {fmt(g['road_range'], 3)} m. Grades and suspension motion limit absolute height. Yaw comes from robust mid-arc travel direction; lateral zero is the mean of the two left/right mount midpoints. No mirror rotations or pair-specific coordinate equality were imposed.",
        "",
        f"The no-slip rear-axle fit alone puts the back lens at x={g['rear_x_unconstrained']:.3f} m; the car’s rear extent bound is −{car('rear_overhang_m', 0.977):.3f} m. The delivered common longitudinal origin is the nearest feasible one under that hard body-envelope bound. The {100 * abs(g['x_axle_raw'] - g['x_axle_used']):.1f} cm change applies to all six; it is not a measurement of the rear lens recess. Trust relative baselines more than absolute x. The lateral residual at the unconstrained solution is {g['lateral_rms'] * 1000:.1f} mm per increment (already informed by the no-slip prior).",
        "",
    ]
    pairs = []
    E = np.diag([1, -1, 1])
    C = np.diag([-1, 1, 1])
    for l, r in [(2, 3), (4, 5)]:
        trans = d["pos"][l] - E @ d["pos"][r]
        ang = np.rad2deg(Rot.from_matrix(d["R"][l].T @ E @ d["R"][r] @ C).magnitude())
        pairs.append((100 * np.linalg.norm(trans), ang))
        lines.append(
            f"- {CAMS[l].split('_')[1]} mirror residual: {100 * np.linalg.norm(trans):.1f} cm ({fmt(100 * trans, 1)} cm XYZ), {ang:.2f}° SO(3)."
        )
    lines += [
        "",
        "## Independent frame splits",
        "",
        "Alternating keyframes from each route window, no shared images; features, matches, track assembly, triangulation, BA, timing and ground alignment are fit separately from identical starting priors. The scenes are shared, so this is same-drive repeatability, not between-drive accuracy. Distortion differences are absolute coefficient units (percentage is meaningless near zero). The body bound anchors back x in each run.",
        "",
        "|Camera|max Δf/f|Δ principal point (max axis px)|max Δk|Δ position (cm)|Δ orientation (°)|Δ offset / readout (ms)|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for i, n in enumerate(CAMS):
        x = per[i]
        lines.append(
            f"|{n}|{x['focal']:.3f}%|{x['pp']:.2f}|{x['dist']:.4f}|{x['pos']:.2f}|{x['rot']:.3f}|{x['time']:.2f} / {x['rs']:.2f}|"
        )
    sc = json.load(open(W / "scale-check.json"))
    v = sc["visual"]
    te = sc["telemetry"]
    lines += [
        "",
        "## Held-out scale",
        "",
        f"{v[0]['start']:.3f}–{v[0]['end']:.3f} s, excluded from both rig and ground fits. Freeze the rig and solve 18 held-out poses and landmarks, with one fixed gauge pose, no velocity/odometric measurements, and only visual velocities for exposure compensation. Arbitrary forward spacings of 0.60 and 1.30 m lead to {v[0]['distance']:.4f} and {v[1]['distance']:.4f} m, with {v[0]['cross_points']} / {v[1]['cross_points']} multi-camera landmarks. Integrated telemetry: {te:.4f} m. Difference **{(v[0]['distance'] / te - 1) * 100:+.2f}%**. This misses the ±1% goal; the rig was not rescaled to the holdout. This short departure is not a whole-drive metric accuracy claim.",
        "",
        "## Cross-camera checks",
        "",
        "Triangulate from ≥2 images in a single other lens strictly before or strictly after the target frame. Exclude all target-lens observations from point fitting; require source-only angular baseline, positive range and source RMS <2.5 px. No target residual gate. Poses/intrinsics were fit on this scene, so this is leave-target-lens-out point transfer, not an unseen-scene test. Gold circles mark measured features; magenta crosses mark predictions. Each figure uses the image pair with the most eligible transfers, independent of its errors. Large tails arise from narrow source triangulation angles, viewpoint changes and mismatches; the transfer statistic is stricter than fitted-point RMS.",
        "",
        "|Target|transfers|median / p90 (px)|shown count / median / max (px)|",
        "|---|---:|---:|---:|",
    ]
    for i, n in enumerate(CAMS):
        k, med, p90, sm, mx, count = stats[i]
        lines.append(f"|{n}|{k}|{med:.2f} / {p90:.2f}|{count} / {sm:.2f} / {mx:.2f}|")
    lines += [
        "",
        "The rear is retained. Its masked rim and bumper are not extrapolated into the rectified view. Confidence in the usable optical core is much higher than in the extreme rim or its absolute longitudinal mount. Rectified images use a forward pinhole crop, not a full-width fill of a nearly 180° fisheye. Inspection of the rectified drive-through images showed straight masonry joints, posts and lane edges. All six paired overlays place projections on the same identifiable landmarks (traffic signal supports, façade edges, signs and trees); the front has the largest transfer errors.",
        "",
    ]
    (OUT / "report.md").write_text(chr(10).join(lines))
    print(chr(10).join(lines), flush=True)


if __name__ == "__main__":
    main()
