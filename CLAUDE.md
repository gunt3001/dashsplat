# CLAUDE.md

Notes for working in this fork of dashsplat: a pipeline that rebuilds a 3D scene (Gaussian splats, a road texture
atlas and tracked vehicles) plus a generated third-person chase video from the six dashcam cameras of a Tesla. The
upstream author built it in about a day, on one drive from one car (Model 3, HW4), on a Mac Studio (M5 Ultra, 96 GB).
Start with `README.md`, `PLAN.md` (the data contract) and `pipeline/run/README.md` (the one-command runner).

## Goal of this fork

Run the pipeline on the owner's own Tesla footage on their own hardware. The main machine is NixOS with an
**RTX 4080 (16 GB VRAM), 32 GB RAM, 32 threads**, with Docker but no NVIDIA container runtime configured yet. A Mac
is the alternative.

## Verdict: CUDA or Mac?

**Both work in principle. The Mac path gives better results today; the CUDA path is easier to set up and is the
realistic choice on the 4080, with caveats.**

| | Mac (Apple Silicon) | Linux + CUDA (this machine) |
|---|---|---|
| Status upstream | Primary platform; every published result came from it | A port that "ran end to end on one rented L40S; not at parity with the Mac result" (`pipeline/cuda/README.md`) |
| Setup | No installer or requirements file; environments were built by hand stage by stage | One script: `pipeline/cuda/setup-drive.sh` (assumes Ubuntu-based NGC image, apt, root) |
| Splat fit | MetalSplat Metal kernels | MetalSplat's pure-PyTorch *reference* path (`pipeline/stitch/fit_raster.py`), slower; the test scene had 390k object splats vs 1.44M on Mac |
| Vehicle classifier | Apple Vision via `swiftc` | SigLIP 2 (`pipeline/movers/classifier.py`); produced 69 tracks vs 36, many under 1 s (noisier) |
| Measured speed | Faster (Studio chase: 102 s per generated second) | L40S chase: 168 s per generated second |
| Hardware needed | ~14 GB GPU memory for Difix fp16; author had 96 GB unified. Realistically 32 GB+ unified, 64 GB+ is safer | Tested only on 48 GB cards (L40S, A40) |

So: if the Mac is a high-memory M-series Pro/Max/Ultra (32 GB+, ideally 64 GB+), it is the platform the code was
actually tuned on and will give the better scene. On a smaller Mac, use the 4080. Either way the **calibration rig
is the real blocker, not the GPU** (see below).

## The blockers, in priority order

1. **No calibrated rig is included.** The runner needs `calib/rig.json` and `calib/masks/` for *your* car.
   `pipeline/calib` solves one from footage. Upstream it was hard-wired to the author's Model 3 and drive (masks,
   time windows, mount priors, rear overhang); this fork moved those into `calib/car.json`, and
   [docs/calibrating-your-car.md](docs/calibrating-your-car.md) walks through filling it in. `run.sh` still
   defaults `RS=$HOME/dev/reality-splat`, so always pass `RS=$PWD`. `pipeline/run/rig-baseline.json`
   holds rig-check thresholds measured on the author's car; `drive.py --rig refit` requires ≥120 s of motion and two
   ≥12° turns. Calibration is CPU-only (Ceres), so it runs the same on either platform.
2. **Footage requirements** (`README.md`, "What it does not do"):
   - Firmware 2025.44.25 or later, so the clips carry SEI telemetry (speed, heading, GPS…). `pipeline/ingest/ingest.py`
     parses it per access unit; without it nothing downstream works.
   - Since 2026.20 Tesla encrypts dashcam clips by default in most markets: decrypt in the car or at
     dashcam.tesla.com first.
   - Six cameras at HW4 sizes (front 2896x1876, others 1448x938). HW3 (four cameras), Cybertruck (seven), night and
     rain are untested. Check your car's hardware version before investing time.
   - One continuous drive, car moving most of the time.
3. **16 GB VRAM on the 4080 is tight for the chase-video stage.** Difix-ref at the runner's canvas (1472x960,
   `PAD` in `pipeline/genfill/clip.py`) peaked at **15.03 GB reserved** in fp16 on the A40
   (`pipeline/cuda/README.md`); the desktop already uses ~1.2 GB. Likely OOM unless you run headless or with
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. At 1024x576 it was 7.9 GB, but `PAD` is tied to the chase
   camera geometry, so lowering it is a code change to verify, not a flag. fp32 (~28 GB) will not fit. Use
   `--clip none` to skip generated video entirely and still get the 3D scene.
   Other GPU stages fit: Mask2Former ~5 GB, DA3 ~6 GB, MapAnything ~9 GB, loaded one at a time.
4. **32 GB system RAM** is a third of the author's machine. Per-stage RSS was mostly not logged; an older splat fit
   used 5.6–8 GB RSS. Start with `--window T0 T1` on a 30 s stretch before a whole drive.
5. **Disk:** ~15 GB working files per 30 s window plus ~12 GB of weights. This machine has ~620 GB free.

## Owner's setup (decided)

- Car: **HW4 Model Y**, recent firmware (telemetry expected). Not the Model 3 the code was tuned on.
- Plan: run the full pipeline on an **M4 Max MacBook Pro, 36 GB unified memory** (Metal splat fit and Apple Vision,
  room for Difix's ~14 GB). The NixOS/RTX 4080 box is the fallback.

## Calibrating a new car (pipeline/calib)

**Step-by-step guide: [docs/calibrating-your-car.md](docs/calibrating-your-car.md).** Keep it in sync with
the code below and with the owner's Notion copy ("Tesla Calibration for Dashsplat").

The runner never calibrates from scratch: `pipeline/run/calibration.py` always starts from a stored
`calib/rig.json` + `calib/masks/` (`--rig stored|verify|refit`). The first rig comes from `pipeline/calib`,
run by hand (`RS=$PWD bash pipeline/calib/run.sh`). It is CPU-only (OpenCV RootSIFT + Ceres BA).

- **Per-car settings live in `calib/car.json`** (gitignored; template `pipeline/calib/car.example.json`, which
  holds the author's Model 3 values; `CALIB_CAR` overrides the path). Keys: `windows`, `masks`, `mounts_m`,
  `rear_overhang_m`, `yaw_deg`, `pitch_deg`, `focal_px`. Read by `features.py` (`car()`, `mask_image()`),
  `solve.py` (priors), `align.py` and `deliver.py` (overhang). A missing key falls back to the original
  hard-coded Model 3 value; with no `car.json` the outputs are byte-identical to upstream.
- `windows` must have exactly 6 entries: `seg < 5` fits the rig, `seg == 5` is the blind scale holdout
  (`solve.py`, `features.match`).
- Mask polygons are in a 600x389 space for every camera; `keep` = one polygon kept, `remove` = polygons masked.
- `rear_overhang_m` is rear axle centre to the rearmost bumper point; `align.py` won't place the back lens
  behind it (the author's solve ended exactly on that bound).
- Feature caches in `work/calib/{features,matches,matches-split}` are indexed by keyframe and must be deleted
  when windows or masks change.
- Helpers in `tools/calib/`: `organize_clips.py` (TeslaCam folder → `data/clips/<minute>/`, checks HW4
  sizes, encryption and SEI telemetry), `pick_windows.py` (suggest/plot windows, `--write` into car.json),
  `mask_sheets.py` (temporal-median images + mask overlays in `work/calib/mask-sheets/`), `carfile.py`
  (compact car.json I/O).
- Ingest: `pipeline/ingest/ingest.py data` reads `data/clips/<clip>/<clip>-<cam>.mp4`, writes `data/drive/`
  (every frame as JPEG, ~2 GB per 30 s).
- Use `--rig stored` on the first pipeline runs: `pipeline/run/rig-baseline.json` thresholds are the author's car.

## Running on this machine (CUDA)

The setup script targets `nvcr.io/nvidia/pytorch:25.03-py3` as root with apt, and keeps the image's bundled CUDA
torch. On NixOS, run it in that container rather than natively:

1. Enable GPU passthrough for Docker in the NixOS config, then rebuild:
   `hardware.nvidia-container-toolkit.enable = true;` (currently `/etc/cdi` is empty and Docker has only `runc`).
2. Start the container with the repo mounted:
   `docker run --rm -it --device nvidia.com/gpu=all --shm-size=16g -v $PWD:/work -w /work nvcr.io/nvidia/pytorch:25.03-py3`
3. Inside it: put the rig in `calib/` and clips in `clips/`, then
   ```sh
   bash pipeline/cuda/setup-drive.sh
   export RS_INSTALL=$PWD HF_HOME=$PWD/third_party/hf TORCH_HOME=$PWD/third_party/torch DIFIX_RENDER_ONLY_DECODE=1
   .venv/bin/python pipeline/run/drive.py run clips --name my-drive                         # preview tier
   .venv/bin/python pipeline/run/drive.py run clips --name my-drive --quality full --clip none
   .venv/bin/python pipeline/run/drive.py status --name my-drive
   ```
   Re-running the same command resumes. On Linux the runner starts its own private viewer on 127.0.0.1 to check
   output (`pipeline/run/drive.py`, near line 782).
4. View: `bun deploy/serve.ts viewer/dist scene 8090 127.0.0.1` and open `http://127.0.0.1:8090/?scene=my-drive`.

Files written by root in the container will be root-owned on the host; `chown` afterwards or pass `--user`.

## Running on a Mac

No installer. Follow `README.md` "On a Mac": Python 3.12 + PyTorch (MPS), `brew install ceres-solver cmake ffmpeg`,
Xcode command-line tools (Apple Vision via Swift), Bun. Use `pipeline/cuda/setup-drive.sh` as the checklist of
packages, clones (Depth-Anything-3, metalsplat, Difix3D at pinned SHAs) and builds, translating apt to Homebrew.
`pipeline/genfill/setup.sh` sets up the separate Difix environment (genfill pins diffusers 0.25.1, transformers
4.38.0, PEFT 0.9.0; keep it isolated from the depth env).

## Code facts worth knowing

- Device selection is mostly `cuda → mps → cpu` (`pipeline/depth/models.py:19`, `pipeline/masks/seg.py:13`,
  `pipeline/genfill/difix.py:29`, `pipeline/stitch/model.py` `gpu()`).
- **Hard-coded `device="mps"` remains** in `pipeline/stitch/model.py` (~lines 80–86) and
  `pipeline/stitch/v2field.py` (~lines 126–127, 301–306), and `pipeline/stitch/perceptual.py`. The CUDA run worked,
  so these are presumably off the runner's path, but they are the first suspects for an MPS error on Linux.
- Platform switches use `sys.platform` for `.dylib` vs `.so` builds and `cp --reflink=auto` vs `cp -c`
  (`pipeline/run/stages.py`, `pipeline/run/transactions.py`). `setup-drive.sh` still names Linux outputs `.dylib`;
  that is intentional (loaders look for that name).
- The runner's full-quality fit is `stitch.v4fit` (`pipeline/run/full.py:370`).
- Many stage scripts outside `pipeline/run` default to `RS=~/dev/reality-splat`. Set `RS=$PWD` when running a stage
  by hand. The runner sets it for the stages it calls.
- The viewer's dev server reads `scenes/`; the pipeline publishes to `scene/`.
- Ignore the author-only scripts: `pipeline/run/install.sh`, `pipeline/cuda/run_pod.py`, `*_worker.sh`, `pack_*.py`,
  `drive_receipts.py`, `verify_drive.py`, `window_clips.py`, `deploy/deploy.sh`, `film/studio.sh`. Stage READMEs
  mention hosts `anomastudio`, `basebook`, `ferry` and `10.0.50.51`; those are the author's machines.

## Licences

Code is Apache-2.0, but several default models are **non-commercial only** (DA3-LARGE, MapAnything, Mask2Former
Cityscapes weights, NVIDIA Difix-ref). Personal use of your own footage is fine; do not build anything commercial on
these weights. See the README's licence table and `THIRD_PARTY.md`.
