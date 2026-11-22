# V2, V3 and V5: what changed

**Files:** `Code/test_policy_template_v2.py`, `_v3.py` and `_v5.py` (there is no v4). V2 and V3 started as copies of `test_policy_template.py` (unchanged); V5 = V3 + derotation. Class: `GapFlyer`.

**What happens now (all three)**
1. Modes: `search -> calculate -> move -> align -> commit -> done`. `calculate` is new.
2. `search`: same sway as before, but it only stores images and how far the drone had moved for each; no optical flow while moving. The legs now also hold the reference heading (same P yaw controller as `align`).
3. `calculate`: hover at the reference position and do one job per tick: flow (one image per tick, 19 in the runs below), median, RANSAC, segment, locate. 23 ticks, about 1.5 s.
4. Why: before, one tick did `find_gap` + `locate` (about 206 ms with rendering). The worst is now 123 ms; over 100 ms counts as late, and 3 late in a row make the drone hold.

**Differences**

5. V2: `move` and `commit` use PID to a fixed point, as before. It must get within 1 cm (`move`) and 1 mm (`commit`), so it could get stuck on a real drone.
6. V3: `move` and `commit` follow a quintic path (0.8 m/s cap). The first tick plans it while hovering; PID only corrects the tracking error. Arrived = within 4 cm, or 0.5 s after the planned time.
7. V5: V3 plus flow derotation. Each stored image keeps its measured yaw relative to the reference, and that rotation is taken out of the flow before the depth calculation. Roll and pitch are not corrected.
8. Why V5: yaw error looks like depth error. Fake wall at 2.80 m with a 1.1 deg/s heading drift: V3 measured 3.26 m, V5 2.80 m (perfect flow and yaw measurement assumed).

**Changes to Alec's code (only in these copies)**

9. The target and yaw lines after `find_gap` moved into `calculate`. `locate`, `align` and `furthest.py` are unchanged.
10. `locate` and the wall normal now use the reference image's pose, not the current one, so drift after the reference image no longer shifts the target.
11. V3 and V5 only: the `move` and `commit` blocks were edited for the quintic. `find_gap` is split up, so the commented-out `scan_again` needs updating if it is used again.

**Simulation (one run each of the current code, no collisions)**

12. Gate passed at: original 14.60 s, V2 16.13 s, V3 17.87 s, V5 17.87 s. Closest approach was 182 mm in all four. V3 and V5 are slower because the quintic starts and stops smoothly.
13. The sim drone follows commands perfectly and never drifts in yaw, so drift, heading hold and derotation are untested there. Rendering alone took about 50-65 ms per tick on an A30 (the tick timer is 66 ms): check the lab GPU.

**Run it** (from `p4/`, one job at a time)

14. `POLICY=test_policy_template_v5:GapFlyer sbatch Code/run_sim.sbatch` (or `_v2`, `_v3`).
15. Results go to `runs/<jobid>/`. It is git-ignored, so the runs above (V2: 2579373, V3: 2579374, V5: 2579375) are only on Turing.
