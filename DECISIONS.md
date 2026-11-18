# P4 decisions

Spec: https://rbe549.github.io/rbe595/fall2026/proj/p4/ — fly through the largest gap of an unknown window, RGB only.

## Environment
| Item | Decision |
|---|---|
| Machine | Turing (WPI). GPU jobs via sbatch: `-p academic --account rbe595` |
| Python | conda env `aerial_robotics` (`~/.conda/envs/aerial_robotics`, Py 3.10, torch 2.6.0+cu126, gsplat, onnxruntime, opencv, splat_hitl on `dev`) |
| Blender | 3.6.23 LTS (closest to `Window.blend`'s 3.0), portable at `/scratch/vchezhian/blender/blender-3.6.23-linux-x64/`, headless Cycles on a GPU node |
| Lab | Policy may use CUDA (container has GPU access) |

## Camera / contract
- FOV fixed at 60.66° (horizontal). Resolution **480 × 320** (`p4_contract.json`, RGB only, velocity body_flu, 15 Hz).
- fx = fy = 240 / tan(30.33°) = **410.3 px**; principal point assumed at image centre.
- No yaw for the whole project (yaw rate always 0).

## Perception
- **Optical flow: NeuFlow v2**, pretrained "mixed" weights, no training. Cited in code and report.
- No fallback method. If results are bad after the full pipeline runs in simulation, consider SEA-RAFT.
- **Motion: vertical sway** (up–down). Thrust only, so no tilt and no rotational flow; state gives no roll/pitch to derotate with.
- Depth from flow with the known baseline b (sim or Vicon): **Z = f·|b| / |dv|**. No flow-direction / residual checks for now.
- Segment near (panel) vs far (background through the gap). Gap = largest far region enclosed by panel.

## Iterations
1. **Iteration 1:** single frame pair → depth → gap mask → rough window position → IoU vs ground truth. No multi-frame association, no fallbacks or safety checks.
2. **Iteration 2:** associate / fuse multiple frames over time.
- Deferred until iteration 1 + IoU verification: number of sways (once vs far+near), commit behaviour (blind vs tracked), gap ranking (inscribed circle vs area), safety margins.

## Blender evaluation (iteration 1)
- Keep `Window.blend` unchanged; all overrides from `Blender/render_pairs.py` at render time.
- Keep the scene's own camera: 39.6° FOV, 1920 × 1080, 50 mm lens → **fx = 2666.7 px**. Do not rescale the scene.
- Window PNG: **black pixel = gap (opening)**; alpha ignored.
- Test set: 4 window shapes × 4 panel textures (Granite background), baselines **0.1:0.1:0.6 m** (camera moved straight up).
- GT gap mask = inside the panel's outer square AND not panel in the ID-mask pass (pass index 2).
- NeuFlow input padded 1080 → 1088 (multiple of 16), output cropped back.
- Outputs per case: RGB, flow, depth, predicted mask, GT, overlay; plus an IoU table.

## Code layout
| Path | What |
|---|---|
| `Code/test_policy_template.py` | All our policy code; run with `Wrapper.py --policy test_policy_template:<Class>` |
| `Code/neuflow/` | NeuFlow v2 (third party, cited) + weights |
| `Blender/render_pairs.py` | Blender render script (submitted with `Blender/`) |
| `eval/` | Evaluation scripts (iteration 1: `iter1_blender.py`) |

## Git
- `main` fast-forwarded to `AN` (3b1edbf). AN's bugs in `Code/policy_template.py` and `README.md` left as is (flagged). Not pushed.

---

## Updates 2026-11-17/18 (these replace earlier entries where they conflict)

### Blender findings
- NeuFlow v2 fails when the panel shift is near half its 1/16-scale cell (8, 24, 40 px...); confirmed by an 8 px pre-shift test. More `iters_s16` made it worse. Highlight in the report.
- Hull = convex hull over **all** inlier patches >= 1% of the image (a gap crossing the frame splits the panel).
- Gap = RANSAC outliers **behind** the plane inside the hull (no separate gap threshold).

### Flow model
- **SEA-RAFT-S** (`Tartan-C-T-TSKH-spring540x960-S`, `spring-S` config, BSD-3) replaces NeuFlow v2. Blender mean IoU 0.98-0.99 across baselines, sway stack 0.976.
- Flight: SEA-RAFT at **480 x 320, scale 0**. The ImageNet ResNet18 download at model build is bypassed (weights come from the checkpoint).

### Policy: the sway action (`Code/test_policy_template.py`, class `SwayGapFlyer`)
- P3-style modes (`start`/`update` -> velocity, yaw rate, next mode). Sway is one repeatable mode.
- Settle **0.5 s** hover, then the reference frame (RGB + pose).
- 4 legs, each a `control.Quintic`, **T = 2 s**: right out, right back, up out, up back. Right = body -y.
- Amplitude and frame range are parameters: `SWAY_AMP_M = 0.20`, `SWAY_MIN_M = 0.07`.
- Frames: per leg 5 stratified-random target offsets in [0.07, 0.20] m; return legs draw **new** random targets. A frame is kept on the first tick the **measured** offset along the leg axis crosses the next target.
- Each kept frame stores its own measured translation (same pose as its render); rho uses the full measured translation:
  rho = (F.g)/|g|^2, g = (-f t_x + u' t_z, -f t_y + v' t_z), camera frame.
- Tracking: `control.PositionController` (P3 gains), feed-forward from the quintic.
- After the 4 legs: median rho (+ MAD) -> wall -> gap mask; then switch to `Hold` (placeholder for the next action).
- **RANSAC input downsampled in flight**: 2x2 max-pool of rho, then top 50% (8x fewer points). Max-pool chosen by the user; **flag if it fails** (max keeps outlier pixels). Hull and gap mask stay at full resolution.
- Debug output per sway: `.npz` (ref RGB, rho, MAD, masks, frame log) + PNG + full-res binary mask PNG.
- **Floor mask**: pixels with world height < `MIN_HEIGHT_M = 0.3` m (from median rho; rho <= 0 left alone) are left out of RANSAC and of the hull's inlier patches. Fixed the floor band stretching the hull (sim: gap 31.4% -> 3.1% of image, floor check no longer triggered).
- **Mask clean-up**: opening (erode then dilate) of the gap mask, 5 px ellipse (`OPEN_PX`). Sim: 14 pieces -> 1 (the big hole), 2.8% of image.
- Two policy layouts, same behaviour: `test_policy_template.py:GapFlyer` (starter if-chain) and `test_policy_classes.py:ModeGapFlyer` (P3 mode classes, imports the shared helpers). Collaborators fill whichever they prefer.
- Gap choice, standoff, commit and failure handling: not yet decided.
