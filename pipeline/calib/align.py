"""Fix the remaining vehicle-coordinate gauge using the motion, road and pair centres."""

from solve import *
from scipy.optimize import least_squares


def align(s):
    Y = Rot.from_euler("z", s.base[:, 2:3]).as_matrix()
    ego = Y @ Rot.from_rotvec(s.dtfr[:, :3]).as_matrix()
    t = np.pad(s.base[:, :2], ((0, 0), (0, 1))) + s.dtfr[:, 3:]
    gr = json.load(open(s.w / "ground.json"))
    height = gr["height"]
    # Mid-arc lateral slip ≈ dx * delta_heading - dyaw * forward distance.
    # Find the reference that has zero lateral motion through a turn; select
    # the yaw of +x from straight-line motion in the same robust fit.
    ix = np.flatnonzero(s.valid)
    a = ix[:-1]
    b = ix[1:]
    same = np.array([s.fr[j]["seg"] == s.fr[i]["seg"] for i, j in zip(a, b)])
    a, b = a[same], b[same]
    yaw = np.unwrap(np.arctan2(ego[:, 1, 0], ego[:, 0, 0]))
    h = (yaw[a] + yaw[b]) / 2
    dd = t[b, :2] - t[a, :2]
    lateral = -np.sin(h) * dd[:, 0] + np.cos(h) * dd[:, 1]
    forward = np.cos(h) * dd[:, 0] + np.sin(h) * dd[:, 1]
    dyaw = yaw[b] - yaw[a]
    fit = least_squares(
        lambda p: lateral + p[0] * 2 * np.sin(dyaw / 2) - p[1] * forward,
        [0, 0],
        loss="soft_l1",
        f_scale=0.018,
    ).x
    # Road fixes height, not mount roll/pitch: a narrow rear strip has poor
    # roll observability and individual roads can bank. Keep the measured
    # up from the motion plane. Do not impose mirror rotations or baselines.
    # A gauge change does not move a single camera relative to the other five.
    # The rear glass cannot lie behind the rearmost bumper plane. If the visual
    # no-slip optimum violates that hard physical bound, use the closest feasible
    # origin, and report the conflicting unconstrained estimate separately.
    axes = Rot.from_euler("z", fit[1]).as_matrix()
    p = s.cam[:, 11:14]
    pp = p @ axes
    cy = np.mean(pp[2:, 1])
    raw = pp[1, 0] - fit[0]
    # car.json "rear_overhang_m": rear axle centre to the rearmost point of the bumper (the back
    # camera cannot sit behind it), measured horizontally on the car.
    cx = min(fit[0], pp[1, 0] + car("rear_overhang_m", 0.977))
    centre = np.array([cx, cy, height]) @ axes.T
    return (
        axes,
        centre,
        dict(
            road=gr["height"],
            road_range=gr["quantiles"],
            road_pairs=gr["accepted"],
            road_total=gr["total"],
            x_axle_raw=float(fit[0]),
            x_axle_used=float(cx),
            rear_x_unconstrained=float(raw),
            yaw_deg=float(np.rad2deg(fit[1])),
            lateral_rms=float(
                np.sqrt(np.mean((lateral + fit[0] * 2 * np.sin(dyaw / 2) - fit[1] * forward) ** 2))
            ),
        ),
    )
