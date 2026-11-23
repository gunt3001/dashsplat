# Calibrating your own Tesla

Before dashsplat can rebuild a drive, it needs to know exactly how your car's six cameras are set up: where
each one sits on the car, which way it points, how its lens bends the picture, and how their clocks line up.
That description is the **rig**, stored in `calib/rig.json` and `calib/masks/`. This repository doesn't ship
one, and the rig the author made fits only their Model 3. This guide makes one for your car.

You do this **once per car**. After that, every drive reuses the same rig.

**Before you start**

- **Time:** an afternoon. Most of it is the drive, measuring the car and drawing the masks. The computer
  part runs on its own.
- **Car:** HW4, six cameras, firmware 2025.44.25 or later (so the video carries telemetry). Written for a
  **Model Y**. The upstream code was tuned on a Model 3.
- **Computer:** a Mac (Apple Silicon) or Linux. Calibration uses only the CPU, so no GPU is needed.
- **Disk:** about 25 GB free for a 6-minute calibration drive. Ingest writes every frame as a JPEG.
- **Tools:** a tape measure, a piece of string with a small weight (a plumb line), and a helper if you have one.

**How it works.** You record a normal drive. A script finds the same street details (corners, signs,
lane marks) in all six cameras over time. It then solves for the camera positions and lenses that make those
details line up in 3D, using the car's own speed and heading as a guide. You provide three things:

1. **Time windows:** which stretches of the drive to use.
2. **Camera positions:** rough measurements, so the solver starts in the right place.
3. **Body masks:** which parts of each picture are your own car, so the solver ignores them.

All three go into one file: `calib/car.json`.

All commands below run from the **repository root**. Everything they write (`data/`, `calib/`, `work/`) is
ignored by git.

---

## Step 0: Set up the computer (once)

On a Mac, with [Homebrew](https://brew.sh):

```sh
brew install python@3.12 ceres-solver cmake ffmpeg
cd /path/to/dashsplat
python3.12 -m venv .venv
.venv/bin/pip install numpy scipy opencv-python av matplotlib
```

On Linux, install `libceres-dev cmake g++ ffmpeg python3-venv` with your package manager. Then create the
same `.venv`.

Check it worked:

```sh
.venv/bin/python -c "import numpy, scipy, cv2, av, matplotlib; print('ok')"
```

---

## Step 1: Record a calibration drive

Drive for **5 to 10 minutes** with dashcam recording on. A good calibration drive has:

- **Daylight** and a dry road. No night and no rain.
- **Lots of turns.** At least three proper 90° turns (street corners, roundabouts). Turns are what let the
  solver find the car's axes and the rear axle.
- **Things to see:** buildings, poles, signs and parked cars. A suburban or town route is ideal. An empty
  motorway or a field is the worst case.
- **Moving most of the time,** at normal city speed (20–60 km/h).
- **At least one start from a full stop** on a straight road, like pulling away from a stop sign. The last
  window uses it as a blind check.
- A clean windscreen and camera covers, especially the repeaters and the rear camera.

Afterwards, copy the `TeslaCam` folder from the car's USB drive **onto your computer's own disk**. Reading
thousands of frames from a USB stick is slow.

> **Encrypted clips.** Since firmware 2026.20, Tesla encrypts dashcam clips by default in most markets.
> If step 2 reports that it can't open the files, turn encryption off for new recordings in the car
> (or decrypt them at dashcam.tesla.com) and copy them again.

---

## Step 2: Pick the drive and lay out the files

Tesla saves one file per camera per minute. Ingest wants one folder per minute.
`tools/calib/organize_clips.py` handles that.

List the drives it finds (minutes less than 90 s apart count as one drive):

```sh
.venv/bin/python tools/calib/organize_clips.py ~/TeslaCam
```

```text
42 complete minutes in 5 drives
  [1] 2026-11-20_08-02-11 to 2026-11-20_08-09-12  ~8 min
  [2] 2026-11-20_17-30-40 to 2026-11-20_17-36-41  ~7 min
  ...
```

Pick your calibration drive by number:

```sh
.venv/bin/python tools/calib/organize_clips.py ~/TeslaCam --drive 2
```

The script first checks the first minute. It looks for all six cameras at the HW4 sizes (front 2896×1876,
the others 1448×938) and for telemetry in the frames. If everything passes it prints one line per camera,
like `front 2896x1876 telemetry in 90/90 frames`, then links the files into `data/clips/`.

| If it says | It means | Do this |
|---|---|---|
| `cannot open … encrypted?` | The clips are encrypted | See the note in step 1 |
| `no telemetry SEI` | Firmware too old, or the files were re-encoded | Update the car and record again. Always use the original files |
| `… expected HW4 …` | Not HW4 sizes | The pipeline only knows HW4. Stop here and ask |
| `already has clips` | You ran it before | Add `--force` to replace them |

Options: `--from 2026-11-20_17-30-40 --to 2026-11-20_17-34-41` to choose your own range of minutes, and
`--copy` to copy the files instead of linking them.

---

## Step 3: Ingest

```sh
RS=$PWD .venv/bin/python pipeline/ingest/ingest.py data
```

This decodes every frame of all six cameras into `data/drive/frames/` and reads the telemetry. It takes a while
and uses a lot of disk. At the end it prints a summary:

```text
fps 36.0 duration 412.3 distance 3120.5 DR end ... GPS end ...
```

Check that `duration` (seconds) and `distance` (metres) roughly match the drive you remember. Once the
frames are decoded, ingest saves `data/drive/raw_tables.json`, and later runs reuse it. If you change the clips
in `data/clips`, add `--force` so it decodes again.

---

## Step 4: Choose the time windows

Calibration uses six short stretches of the drive:

- **Five fit windows** to solve the rig. They should include turns.
- **One held-out window, always last.** The solver never sees it while fitting. Afterwards it rebuilds that
  stretch without using the car's speedometer, and checks whether the distance comes out right. That shows
  whether the rig's sense of scale is correct.

Let the helper suggest them:

```sh
.venv/bin/python tools/calib/pick_windows.py --write
```

```text
Suggested windows ([start_s, end_s, spacing_m]; last = held-out departure):
  fit 1    [55.5, 63.5, 2.39]      72 m, turn   52 deg, ~30 keyframes
  ...
  held-out  [10.4, 13.9, 1.5]      30 m, turn    0 deg, ~19 keyframes
Plot: work/calib/windows.png
Saved windows to calib/car.json
```

`--write` creates `calib/car.json` (a copy of `pipeline/calib/car.example.json`) the first time. Then it saves
the windows into it.

Open `work/calib/windows.png`. Blue bands are fit windows and the red one is held-out, shown on the speed
graph, the turn-rate graph and the map. Check:

- At least two blue windows sit on turns (spikes in the turn-rate graph).
- No window is in a car park, a tunnel, or behind a lorry blocking the view (you know the route; the
  script doesn't).

To change a window, edit it in `calib/car.json`. Each one is `[start_s, end_s, spacing_m]`. Times are seconds
from the start of the drive. `spacing_m` means "take a keyframe every this many metres". Aim for about 30
keyframes per fit window (`distance ÷ 30`). Keep exactly six windows, with the held-out one last. Run
`pick_windows.py --show` to redraw the plot from your edited file.

---

## Step 5: Measure where the cameras are

The solver only needs rough starting positions. It finds the exact ones itself, but the closer you start,
the more reliably it finds them. **Within ±5 cm is plenty.**

**Reference point:** the centre of the rear axle, at ground level. Park on flat, level ground with the
wheels straight. Measure every camera as three numbers in metres:

- **x:** how far **in front of** the rear wheel centre (hub). Negative if behind it.
- **y:** how far **left** of the car's centreline. Negative for the right side; 0 for front and back.
- **z:** height **above the ground**.

Where the six cameras are on a Model Y:

| Camera (`car.json` name) | Where it is |
|---|---|
| `front` | Top centre of the windscreen, in the housing behind the rear-view mirror. Measure to the middle of the housing |
| `back` | Above the rear number plate, on the tailgate |
| `left_pillar`, `right_pillar` | In the B-pillars, the posts between the front and rear doors, about shoulder height, facing forward and out |
| `left_repeater`, `right_repeater` | In the Tesla "T" badges on the front wings, just behind the front wheels, looking backwards |

**How to measure x without crawling under the car:**

1. Hang the plumb line from the camera so the weight just touches the ground. Mark that spot with chalk or tape.
2. Do the same from the centre of the rear wheel hub on the same side.
3. Measure the distance between the two marks along the length of the car. That's x.
4. For the front and back cameras, use the hub on either side. The point under the camera is on the centreline.
5. For y, measure the mark's distance from the centreline. Half the distance between the two side marks
   works, since the cameras are mirror images.
6. For z, measure straight down from the camera to the ground.

Measure one more distance: from the rear wheel hub to the **rearmost point of the rear bumper**, along the
car. That's `rear_overhang_m`. It stops the solver from putting the rear camera behind the car.

Put your numbers into `calib/car.json`, replacing the Model 3 values:

```json
"car": "Model Y HW4",
"mounts_m": {
  "front": [x, 0, z],
  "back": [-x, 0, z],
  "left_pillar": [x, y, z],
  "right_pillar": [x, -y, z],
  "left_repeater": [x, y, z],
  "right_repeater": [x, -y, z]
},
"rear_overhang_m": 1.0,
```

For a sense of scale, the Model 3 values were: front `[1.94, 0, 1.31]`, back `[-0.89, 0, 0.82]`, pillars
`[1.12, ±0.85, 1.15]`, repeaters `[2.53, ±0.88, 0.79]`, overhang `0.977`. A Model Y is about 18 cm taller,
so your heights should come out higher.

Leave `yaw_deg`, `pitch_deg` and `focal_px` alone. They are loose guesses at which way each camera points
and how much it zooms. HW4 uses the same cameras across models, and the solver corrects them.

---

## Step 6: Draw the body masks

Some cameras see part of your own car: the bonnet edge, the camera housing around the rear lens, the side of the
car in the repeaters, door trim in the pillars. Those pixels move with the car, so they would confuse the
solver. Masks tell it to ignore them.

Make the images:

```sh
.venv/bin/python tools/calib/mask_sheets.py
```

This takes the per-pixel median of about 60 frames from the moving parts of the drive, so **the world blurs
away and only your car stays sharp**. It writes to `work/calib/mask-sheets/`:

- `all.png`: all six cameras at a glance. Start here.
- `overlay-<camera>.png`: one camera, enlarged, with a **labelled grid every 50 units**. The current mask is
  shaded **red**. That starts as the Model 3 masks, which will be wrong in places for a Model Y.
- `median-<camera>.png`: the plain median image.

### The coordinate grid

Every camera uses the same **600 × 389** grid whatever its real resolution: x from 0 (left) to 600 (right),
y from 0 (top) to 389 (bottom). The grid labels on the overlays are in these units.

### Writing a mask

Each camera's entry in `calib/car.json` under `"masks"` has two parts:

- `"remove"`: a list of shapes to **ignore**. Each shape is a list of corner points `[x, y]`, in order
  around the outline (clockwise or anticlockwise, never zig-zag).
- `"keep"` (optional): **one** shape to keep, with everything outside it ignored. Use it when the useful
  view is an island in the middle of the picture, like the Model 3's rear camera, ringed by its housing.

Example, ignoring a strip of bodywork along the bottom of the front camera:

```json
"front": {
  "remove": [
    [[0, 372], [150, 378], [300, 381], [450, 378], [600, 370], [600, 389], [0, 389]]
  ]
}
```

To edit a mask:

1. Open `overlay-<camera>.png`. Find the sharp, unmoving parts of your car.
2. Read the corners of each part off the grid, **erring towards covering a bit too much**. Losing a few
   pixels of road costs nothing, while letting a strip of bodywork through does harm.
3. Write the points into `calib/car.json` and run `mask_sheets.py` again. Repeat until the red shading covers
   your car and nothing else.

What to cover: bonnet and body panels, mirror and camera housings, the dark round edge of a wide lens, and
any glare that sits still. What to leave visible: road, kerbs, other cars, buildings and sky.

> **Tip:** you can ask Claude Code to draft the polygons. It can open `all.png`, propose points for each
> camera, run `mask_sheets.py` and show you the result to approve.

---

## Step 7: Run the calibration

```sh
RS=$PWD BA_THREADS=12 bash pipeline/calib/run.sh
```

Set `BA_THREADS` to your number of performance cores (M4 Max: 10 or 12). The script runs these stages in order:

1. `features.py`: writes the masks to `calib/masks/`, picks keyframes from your windows (it prints
   `keyframes per window [..]`), then finds and matches features.
2. `solve.py all`, `solve.py A`, `solve.py B`: solves the rig with all keyframes, then twice more with half
   the keyframes each, as a consistency check.
3. `ground.py`, then `scale.py`: road height, then the blind held-out scale check.
4. `deliver.py`: writes `calib/rig.json`, `calib/report.md` and the overlays in `calib/debug/`.

It can take a while. If it stops partway, fix the cause and run the same command again. Matched features are
cached in `work/calib/`. **If you change the windows or masks, delete `work/calib/features`,
`work/calib/matches` and `work/calib/matches-split` first**, so they're rebuilt.

---

## Step 8: Check the result

Open `calib/report.md`. Compare it with the author's Model 3 rig (`pipeline/calib/REPORT.md`):

| Where in `report.md` | Good | Author's result | If it's worse |
|---|---|---|---|
| **Rig and final bundle**, column `RMS / p95 (px)` | RMS under ~1.5 px for every camera | 0.81–0.97 px | Usually a mask is missing a bit of car, or that camera's windows have little to see |
| Same table, `inlier obs / tracks` | Thousands per camera | 12,000–25,000 obs | Very low for one camera: its mask is too big or its view is blocked |
| Same table, `x, y, z (m)` | Within ~20 cm of your tape measurements, left/right mirror-symmetric | Up to 22 cm from the author's guesses | Far off, or a side's y has the wrong sign: check `mounts_m` |
| **Independent frame splits**, `Δ position (cm)` | A few cm | 1.4–2.8 cm | Too few keyframes or turns. Use longer windows or add turns |
| **Held-out scale**, the final `Difference` | Within ±3 % (the goal is ±1 %) | +2.14 % | The held-out window is poor (turning, blocked view) or the fit windows lack turns |
| Cross-camera transfer table, `median` | A few px (the front camera is always worst) | 0.7–2.8 px | Check the overlays in `calib/debug/*-transfer.jpg` |

The overlays in `calib/debug/` are the most direct check. In the `*-transfer.jpg` images, points
projected from one camera should land on the same street details (poles, sign edges, window corners) in
the next camera. The `*-rectified.jpg` images should show straight lines (kerbs, building edges) as straight.

`keyframes per window` showing a 0 means a window had no moments where all six cameras captured a frame
within 20 ms while moving. Move or lengthen that window.

---

## Step 9: Use the rig

Keep `calib/rig.json`, `calib/masks/` and `calib/car.json` safe; that's your car's calibration. Then run the
pipeline on any drive:

```sh
.venv/bin/python pipeline/run/drive.py run /path/to/clips --name my-drive --rig stored --window 60 90
```

Use `--rig stored` at first. The runner's default, `--rig verify`, re-checks your rig against thresholds
measured on the author's car (`pipeline/run/rig-baseline.json`), which don't apply to a Model Y.
`--window 60 90` limits the first run to 30 seconds.

---

## Reference: `calib/car.json`

| Key | Meaning | Set by |
|---|---|---|
| `windows` | Six `[start_s, end_s, spacing_m]`. The last is held-out | `pick_windows.py --write`, then by hand |
| `mounts_m` | Each camera's `[x, y, z]` in metres from the rear axle centre at ground level (x forward, y left, z up) | Your tape measurements |
| `rear_overhang_m` | Rear axle centre to the rearmost point of the bumper, metres | Your tape measurement |
| `masks` | Per camera: `remove` (list of polygons) and optional `keep` (one polygon), on the 600×389 grid | By hand, checked with `mask_sheets.py` |
| `yaw_deg`, `pitch_deg`, `focal_px` | Starting guesses for direction and zoom, in the order front, back, left/right pillar, left/right repeater | Leave as is |

Any key you leave out falls back to the author's Model 3 values. A different file can be used with
`CALIB_CAR=/path/to/car.json`. The scripts that read it are `pipeline/calib/features.py` (windows, masks),
`solve.py` (mounts and starting guesses), `align.py` and `deliver.py` (rear overhang).
