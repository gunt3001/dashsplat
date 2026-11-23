"""Joint metric rig BA, independent keyframe splits, and blind visual scale run."""

from features import *
from scipy.spatial.transform import Rotation as Rot
from collections import defaultdict
import subprocess, argparse

B0 = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0.0]])
# Starting guesses (priors) per camera, in CAMS order: front, back, left/right pillar,
# left/right repeater. Defaults are the author's Model 3; override in car.json
# ("yaw_deg", "pitch_deg", "mounts_m", "focal_px"). Mount positions are metres from the
# centre of the rear axle at ground level: x forward, y left, z up.
YAW = car("yaw_deg", [0, 180, 64, -64, 148, -148])
PITCH = car("pitch_deg", [-3, -32, -5, -5, -5, -5])
POS = [
    car("mounts_m", {}).get(c, d)
    for c, d in zip(
        CAMS,
        [
            [1.94, 0, 1.31],
            [-0.89, 0, 0.82],
            [1.12, 0.85, 1.15],
            [1.12, -0.85, 1.15],
            [2.53, 0.88, 0.79],
            [2.53, -0.88, 0.79],
        ],
    )
]
FOCAL = car("focal_px", [3639, 460, 810, 810, 820, 820])
B = Rot.from_euler("ZY", np.column_stack((YAW, -np.array(PITCH))), degrees=True).as_matrix() @ B0


class Scene:
    def __init__(self, split="all"):
        self.split = split
        self.w = W / split
        self.w.mkdir(exist_ok=True, parents=True)
        fr = json.load(open(W / "frames.json"))
        self.fr = fr
        self.nf = len(fr)
        self.sizes = np.array([json.load(open(DATA / "meta.json"))["sizes"][c] for c in CAMS])
        self.dt = np.array([r["dt"] for r in fr])
        tl = np.load(DATA / "timeline.npz")
        t = tl["t"]
        yaw = np.unwrap(tl["yaw"])
        ts = np.array([r["t"] for r in fr])
        v = np.interp(ts, t, tl["speed"])
        ya = np.interp(ts, t, yaw)
        omega = (np.interp(ts + 0.07, t, yaw) - np.interp(ts - 0.07, t, yaw)) / 0.14
        self.base = np.column_stack(
            (
                [r["xy"][0] for r in fr],
                [r["xy"][1] for r in fr],
                ya,
                v * np.cos(ya),
                v * np.sin(ya),
                omega,
            )
        )
        self.dtfr = np.zeros((len(fr), 6))
        f = np.array(FOCAL, float)
        self.prior = np.column_stack(
            (f, f, self.sizes * 0.5, np.zeros((6, 7)), POS, np.zeros((6, 2)))
        )
        self.cam = self.prior.copy()
        self.sig = np.broadcast_to(
            np.array(
                [
                    1000,
                    1000,
                    35,
                    35,
                    0.25,
                    0.08,
                    0.01,
                    0.01,
                    0.25,
                    0.25,
                    0.35,
                    0.25,
                    0.12,
                    0.15,
                    0.01,
                    0.008,
                ]
            ),
            (6, 16),
        ).copy()
        self.sig[:, :2] = f[:, None] * 0.35
        self.pixels = []
        for k in range(len(fr)):
            for c in range(6):
                self.pixels.append(np.load(W / "features" / f"{k:03d}_{c}.npz")["p"])
        self.pixels = np.vstack([np.pad(p, ((0, 3100 - len(p)), (0, 0))) for p in self.pixels])
        self.valid = np.array(
            [
                (r["seg"] == 5 if split == "scale" else r["seg"] < 5)
                and (split not in ["A", "B"] or (i % 2 == int(split == "B")))
                for i, r in enumerate(fr)
            ]
        )

    def poses(self):
        tau = self.dt + self.cam[None, :, 14]
        y = self.base[:, None, 2] + tau * self.base[:, None, 5]
        R = (
            Rot.from_euler("z", y.reshape(-1, 1)).as_matrix().reshape((-1, 6, 3, 3))
            @ Rot.from_rotvec(self.dtfr[:, :3]).as_matrix()[:, None]
        )
        t = np.pad(self.base[:, :2], ((0, 0), (0, 1))) + self.dtfr[:, 3:]
        t = t[:, None] + np.pad(tau[:, :, None] * self.base[:, None, 3:5], ((0, 0), (0, 0), (0, 1)))
        return (R @ (B @ Rot.from_rotvec(self.cam[:, 8:11]).as_matrix())[None]).reshape(
            (-1, 3, 3)
        ), (t + np.einsum("kcij,cj->kci", R, self.cam[:, 11:14])).reshape((-1, 3))

    def rayposes(self, nodes):
        im = nodes // 3100
        k = im // 6
        c = im % 6
        row = (self.pixels[nodes, 1] / self.sizes[c, 1] - 0.5) * self.cam[c, 15]
        R, T = self.poses()
        rr = R[im]
        tt = T[im]
        angle = row * self.base[k, 5]
        co = np.cos(angle)
        si = np.sin(angle)
        r = rr.copy()
        r[:, 0] = co[:, None] * rr[:, 0] - si[:, None] * rr[:, 1]
        r[:, 1] = si[:, None] * rr[:, 0] + co[:, None] * rr[:, 1]
        origin = (
            self.base[k, :2]
            + self.dtfr[k, 3:5]
            + (self.dt[k, c] + self.cam[c, 14])[:, None] * self.base[k, 3:5]
        )
        x, y = (tt[:, :2] - origin).T
        t = tt.copy()
        t[:, 0] += row * self.base[k, 3] + (co - 1) * x - si * y
        t[:, 1] += row * self.base[k, 4] + si * x + (co - 1) * y
        return r, t

    def rays(self, nodes):
        im = nodes // 3100
        c = im % 6
        p = self.cam[c]
        xy = (self.pixels[nodes] - p[:, 2:4]) / p[:, :2]
        r = np.linalg.norm(xy, axis=1)
        theta = r.copy()
        for _ in range(7):
            t = theta**2
            theta -= np.clip(
                (theta * (1 + t * (p[:, 4] + t * p[:, 5])) - r)
                / np.maximum(0.2, 1 + t * (3 * p[:, 4] + 5 * t * p[:, 5])),
                -0.15,
                0.15,
            )
        dir = np.column_stack((xy * (np.sin(theta) / np.maximum(r, 1e-10))[:, None], np.cos(theta)))
        rots, cent = self.rayposes(nodes)
        return cent, np.einsum("nij,nj->ni", rots, dir)

    def project(self, nodes, X):
        im = nodes // 3100
        c = im % 6
        R, T = self.rayposes(nodes)
        d = np.einsum("nji,nj->ni", R, X - T)
        rho = np.linalg.norm(d[:, :2], axis=1)
        th = np.arctan2(rho, d[:, 2])
        p = self.cam[c]
        t = th**2
        uv = (
            p[:, :2]
            * d[:, :2]
            * (th * (1 + t * (p[:, 4] + t * p[:, 5])) / np.maximum(rho, 1e-12))[:, None]
            + p[:, 2:4]
        )
        return uv, d[:, 2]

    def tracks(self, cross):
        # Greedy track union with explicit one-keypoint-per-image conflict rejection.
        # Temporal pairs first; cross-lens links must pass the current metric ray test.
        pairs = []
        files = list((W / "matches").glob("*.npz"))
        if self.split in ["A", "B"]:
            files += list((W / "matches-split").glob("*.npz"))
        for file in sorted(files):
            j = int(file.stem)
            if not self.valid[j // 6]:
                continue
            with np.load(file) as d:
                for key in d.files:
                    i = int(key)
                    if not self.valid[i // 6] or (not cross and i % 6 != j % 6):
                        continue
                    a = d[key]
                    pairs.append((i, j, a))
        pairs.sort(
            key=lambda p: (p[0] % 6 != p[1] % 6, abs(p[0] // 6 - p[1] // 6), -len(p[2]), p[0])
        )
        parent = {}
        views = {}

        def root(i):
            while i in parent:
                i = parent[i]
            return i

        for i, j, a in pairs:
            u, v = (a[:, 0] + 3100 * i).astype(int), (a[:, 1] + 3100 * j).astype(int)
            if i % 6 != j % 6:
                C, s = self.rays(u)
                D, t = self.rays(v)
                N = np.cross(s, t)
                nc = np.linalg.norm(N, axis=1)
                ep = np.abs(np.sum((D - C) * N, axis=1)) / np.maximum(
                    np.linalg.norm(np.cross(D - C, s), axis=1), 1e-9
                )
                valid = ep * np.min(self.cam[[i % 6, j % 6], 0]) < 10
                u, v = u[valid], v[valid]
            for un, vn in zip(u.tolist(), v.tolist()):
                x = root(un)
                y = root(vn)
                if x == y:
                    continue
                sx = views.get(x)
                sy = views.get(y)
                if sx is None:
                    sx = {i}
                if sy is None:
                    sy = {j}
                if not sx.isdisjoint(sy):
                    continue
                if len(sx) < len(sy):
                    x, y = y, x
                    sx, sy = sy, sx
                parent[y] = x
                sx.update(sy)
                views[x] = sx
                views.pop(y, None)
        good = {x: [x] for x, s in views.items() if len(s) >= 3}
        for n in parent:
            r = root(n)
            if r in good:
                good[r].append(n)
        tracks = [sorted(x) for x in good.values()]
        print(
            "raw tracks",
            len(tracks),
            "cross",
            sum(len(set(v // 3100 % 6 for v in x)) > 1 for x in tracks),
            flush=True,
        )
        return tracks

    def initialize(self, tracks, limit, cut):
        # Select tracks with informative angular baselines, but maintain coverage over each frame/camera.
        rng = np.random.default_rng(546)
        tr = []
        for x in tracks:
            if len(x) > 18:
                x = [x[i] for i in np.linspace(0, len(x) - 1, 18, dtype=int)]
            tr.append(x)
        ns = np.array([len(x) for x in tr])
        n = np.concatenate(tr)
        ti = np.repeat(np.arange(len(tr)), ns)
        O, r = self.rays(n)
        M = np.eye(3)[None] - r[:, :, None] * r[:, None, :]
        b = np.einsum("nij,nj->ni", M, O)
        ix = np.r_[0, ns.cumsum()[:-1]]
        MA = np.add.reduceat(M, ix)
        sol = np.linalg.solve(MA + 1e-9 * np.eye(3), np.add.reduceat(b, ix)[:, :, None])[:, :, 0]
        uv, z = self.project(n, sol[ti])
        err = np.linalg.norm(uv - self.pixels[n], axis=1)
        mu = np.add.reduceat(np.minimum(err, 100), ix) / ns
        dst = np.linalg.norm(sol - O[ix], axis=1)
        g = (mu < cut) & (dst > 0.6) & (dst < 330) & (np.minimum.reduceat(z, ix) > 0.02)
        first = n[ix] // 3100
        cross = np.minimum.reduceat(n // 3100 % 6, ix) != np.maximum.reduceat(n // 3100 % 6, ix)
        score = np.log(ns) * 2 + 2 * cross - np.log1p(mu / cut) + rng.uniform(0, 2, len(tr))
        counts = defaultdict(int)
        accepted = []
        for j in np.argsort(-score):
            if not g[j]:
                continue
            key = (
                first[j],
                int(self.pixels[n[ix[j]], 0] / (self.sizes[first[j] % 6, 0] / 4)),
                int(self.pixels[n[ix[j]], 1] / (self.sizes[first[j] % 6, 1] / 3)),
            )
            if counts[key] >= limit:
                continue
            counts[key] += 1
            accepted.append(j)
        aa = np.sort(accepted)
        self.points = sol[aa]
        self.obs = np.concatenate([tr[j] for j in aa])
        self.pi = np.repeat(np.arange(len(aa)), ns[aa])
        print("initial points", len(aa), "observations", len(self.obs), flush=True)

    def prune(self, cut):
        ob = self.obs
        pi = self.pi
        uv, z = self.project(ob, self.points[pi])
        err = np.linalg.norm(uv - self.pixels[ob], axis=1)
        ok = (err < cut) & (z > 0)
        ns = np.bincount(pi[ok], minlength=len(self.points))
        ok &= ns[pi] >= 3
        keep = ns >= 3
        self.points = self.points[keep]
        self.obs = ob[ok]
        self.pi = np.cumsum(keep)[pi[ok]] - 1

    def ba(self, it, loss, mode=0):
        nc = 6
        nf = self.nf
        fname = self.w / "ba.txt"
        out = self.w / "solution.bin"
        sig = np.broadcast_to([0.035, 0.035, 0.013, 0.28, 0.28, 0.45], (nf, 6)).copy()
        if mode == 2:
            sig[:] = 1e6
            sig[np.flatnonzero(self.valid)[0]] = (
                1e-8  # hold one pose as gauge only; no odometric scale/pose prior
            )
        edges = []
        ix = np.flatnonzero(self.valid)
        for a, b in zip(ix[:-1], ix[1:]):
            if mode != 1 or self.fr[a]["seg"] != self.fr[b]["seg"]:
                continue
            dx, dy = self.base[b, :2] - self.base[a, :2]
            yaw = (self.base[a, 2] + self.base[b, 2]) / 2
            dd = np.hypot(dx, dy)
            edges.append([a, b, dx, dy, yaw, dd, 0.018 + 0.016 * dd, 0.018 + 0.008 * dd])
        with open(fname, "w") as f:
            print(
                6,
                nf,
                len(self.points),
                len(self.obs),
                it,
                loss,
                mode,
                int(os.getenv("BA_THREADS", "14")),
                len(edges),
                file=f,
            )
            np.savetxt(
                f,
                np.column_stack(
                    (self.cam, B.reshape((6, 9)), self.prior, self.sig, self.sizes[:, 1])
                ),
                fmt="%.12g",
            )
            np.savetxt(f, np.column_stack((self.base, self.dtfr, sig)), fmt="%.12g")
            np.savetxt(f, np.array(edges).reshape((-1, 8)), fmt=["%d", "%d"] + ["%.12g"] * 6)
            np.savetxt(f, self.points, fmt="%.12g")
            im = self.obs // 3100
            k = im // 6
            c = im % 6
            np.savetxt(
                f,
                np.column_stack((k, c, self.pi, self.dt[k, c], self.pixels[self.obs])),
                fmt=["%d", "%d", "%d", "%.10g", "%.8f", "%.8f"],
            )
        subprocess.run([W / "build/rig_ba", fname, out], check=True)
        x = np.fromfile(out)
        self.cam = x[:96].reshape((6, 16))
        self.dtfr = x[96 : 96 + nf * 6].reshape((-1, 6))
        self.points = x[96 + nf * 6 :].reshape((-1, 3))
        self.status()

    def status(self):
        uv, z = self.project(self.obs, self.points[self.pi])
        err = np.linalg.norm(uv - self.pixels[self.obs], axis=1)
        c = self.obs // 3100 % 6
        print(
            "RMS px",
            np.round([np.sqrt(np.mean(err[c == i] ** 2)) for i in range(6)], 2),
            "fx",
            np.round(self.cam[:, 0], 1),
            flush=True,
        )

    def save(self):
        np.savez(
            self.w / "result.npz",
            cam=self.cam,
            frame=self.dtfr,
            points=self.points,
            obs=self.obs,
            pi=self.pi,
        )

    def load(self):
        d = np.load(self.w / "result.npz")
        self.cam, self.dtfr, self.points, self.obs, self.pi = [
            d[k] for k in ["cam", "frame", "points", "obs", "pi"]
        ]
        return self

    def run(self):
        mode = 2 if self.split == "scale" else 0
        if mode:
            self.cam = np.load(W / "all/result.npz")["cam"].copy()
        seed = self.tracks(False)
        self.initialize(seed, 2, 30)
        self.ba(35, 5, mode)
        self.prune(9)
        self.ba(25, 2, mode)
        self.save()
        # After each pass, reconsider all matches using the improved unprojection/rig.
        tr = self.tracks(True)
        self.initialize(tr, 5, 9)
        self.ba(30, 2, mode)
        self.prune(4)
        self.ba(22, 1.6, mode)
        self.prune(3)
        self.ba(18, 1.6, mode)
        self.save()
        self.timed() if not mode else None

    def timed(self):
        self.ba(35, 1.6, 1)
        self.save()


if __name__ == "__main__":
    s = Scene(sys.argv[1] if len(sys.argv) > 1 else "all")
    if "--timed" in sys.argv:
        s.load().timed()
    else:
        s.run()
