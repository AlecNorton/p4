# P4 quick reference (Turing)

## 1. One-time setup

```bash
# Turing login node. Work in scratch.
mkdir -p ~/scratch/rbe595 && cd ~/scratch/rbe595

# conda env (Python 3.10)
module load miniconda3
conda create -n aerial_robotics python=3.10 -y
conda activate aerial_robotics            # every new shell: module load miniconda3 && conda activate aerial_robotics

# packages P4 needs
module load cuda12.6/toolkit/12.6.3       # gsplat compiles against it
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
pip install gsplat==1.5.3 ninja numpy==2.2.6 opencv-python matplotlib scipy safetensors huggingface_hub

# simulator library (dev branch; main cannot load the pack)
git clone -b dev https://github.com/pearwpi/splat_hitl.git
pip install -e splat_hitl

# our repo: scene, tools, SEA-RAFT code + weights all included
git clone git@github.com:AlecNorton/p4.git
cd p4
```

- `p4_contract.json` (480 x 320) is committed. Do **not** run `tools/make_contract.py` without `--width 480`.
- `tools/check_setup.py` step 7 imports `policy_template.py`, which has syntax errors: expect it to fail there.
- First GPU run compiles gsplat (a few minutes, once).

## 2. Run the simulation

```bash
cd ~/scratch/rbe595/p4
sbatch Code/run_sim.sbatch                                          # starter-layout version
POLICY=test_policy_classes:ModeGapFlyer sbatch Code/run_sim.sbatch  # mode-class version
sbatch Code/run_sim.sbatch --max-s 60                               # extra args go to Wrapper.py
squeue -u $USER; tail -f slurm-p4_sim-<jobid>.out
```

- Results: `runs/<jobid>/` -> `run_log.txt`, `run.json`, `run.png`, `fpv/`, `FPVVideo.mp4`, `debug/` (rho, MAD, gap mask: `.npz`, `.png` panel, `_mask.png` full-res binary mask, `_frames.json`).
- Don't run two jobs at once from the same folder: they share `fpv/`.
- Interactive GPU shell: `srun -p academic -A rbe595 --gres=gpu:1 -c 4 --mem=16G -t 1:00:00 --pty bash`, then
  `python Code/Wrapper.py --pack . --worker-python $(which python) --policy <module:Class>`.

## 3. Policy code: two versions, same behaviour

Both files have the same perception and sway; pick whichever is comfortable and fill that one.

| | `Code/test_policy_template.py` : `GapFlyer` | `Code/test_policy_classes.py` : `ModeGapFlyer` |
|---|---|---|
| Layout | starter file, TODOs filled; if-chain on `self.mode` in `act()` | P3 style: one class per mode |
| Mode state | attributes on the policy (`self.sway`, `self.move_hold`) | inside each mode class |
| Mode start | by hand (`if ... is None:` first-tick checks) | `start(r)`, called automatically on switch |
| Switch | set `self.mode = "..."` (+ `self.ctrl.reset()`) | `update()` returns the next mode's name |
| New mode | another `if` block in `act()` | new class + add it to `self.modes` in `reset()` |
| Repeat sway | `self.sway = None; self.mode = "search"` | `return ..., "search"` |

Shared helpers live in `test_policy_template.py` (the class version imports them): `SeaRaftFlow`, `rho_from_flow`, `RhoStack`, `fit_wall`, `segment`, geometry helpers, parameters (`SWAY_*`, `TOP_FRAC`, ...).

### Mode-class ideology (`test_policy_classes.py`)
- `start(r)`: runs once when the mode is entered (save hold point, plan trajectory, reset counters).
- `update(r)`: runs every tick (15 Hz) and returns `(v_world, yaw_rate, next_mode_or_None)`.
- `r` is the policy: `r.state`, `r.rgb`, `r.velocity_to(target, state, v_ff)`, `r.det`, `r.log()`, `r.locate()`.
- `act()` = `self.cur.update(self)`; a returned name -> `switch()` -> new mode's `start()`.

## 4. Status: what is done, what to fill

| Part | Status |
|---|---|
| `search` | **done**: sway (right / up, quintic legs, 2 s each) -> SEA-RAFT flow on 20 kept frames -> median inverse depth -> wall plane -> gap mask in `det["mask"]` (also `rho`, `wall`, `labels`) |
| `locate(det, state)` | **### For AN to fill #####**: gap mask -> 3D world target (wall plane depth: `det["wall"]`) |
| `move` | **### For AN to fill #####**: placeholder hovers. Next: pick the gap, align, go through, stop |
| Controller | `Code/control.py` (`PositionController`, `Quintic`) via `velocity_to` |

- Rules: no yaw, nothing hard-coded, never read `scene/` files or gate fields in the policy.
- Pixels below 0.3 m world height (floor, `MIN_HEIGHT_M`) are left out of RANSAC and the hull.
- Gap mask cleaned with an opening (erode then dilate, `OPEN_PX = 5`): sim gives 1 clean instance, the big hole.
