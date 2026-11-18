#!/usr/bin/env python3
"""P4 policy, mode-class version (P3 style). Same perception and sway as
test_policy_template.py (imported from there), different state machine layout:

    each mode is a class:  start(r)   once, when the mode is entered
                           update(r)  every tick -> (world velocity, yaw rate, next mode or None)
    r is the policy (gives state, rgb, controller, perception helpers).
    act() calls self.cur.update(self); a returned name switches mode (switch() calls its start()).

    python3 Code/Wrapper.py --pack . --policy test_policy_classes:ModeGapFlyer --worker-python PYTHON
    tools/fly.py ... --policy test_policy_classes:ModeGapFlyer        # on the drone

Modes:
    search  sway, collect frames, find the gap (done)   -> "move"
    move    ### For AN to fill#####                      (placeholder: hover)
To repeat the sway from any mode: return ..., "search" (start() clears the stack).
"""
import atexit
import json
import os
import time

import cv2
import numpy as np
import torch

from splat_hitl.commands import Action
from splat_hitl.policy import Policy

from control import PositionController, Quintic
from fpv import FpvRecorder
from test_policy_template import (DEBUG_DIR, IMG_H, IMG_W, SETTLE_S, SWAY_AMP_M, SWAY_BINS,
                                  SWAY_MIN_M, SWAY_SEED, SWAY_T_S, RhoStack, SeaRaftFlow,
                                  body_to_world, camera_matrix, debug_panel, fit_wall, low_mask,
                                  rho_from_flow, segment, world_to_body, world_to_cam)


# --------------------------------------------------------------------------- #
# The modes. update() returns (world velocity, yaw rate, next mode name or None).
# --------------------------------------------------------------------------- #
class Search:
    """Sway: settle, reference frame, 4 quintic legs keeping frames on measured target
    crossings (new random targets per leg); then find the gap."""
    name = "search"

    def start(self, r):
        s = r.state
        self.t0 = s.t_s
        self.hold = np.array(s.position_m, float)
        self.phase = "settle"
        self.ref_rgb = self.ref_p = self.ref_yaw = None
        self.legs, self.i, self.frames = [], 0, []
        self.rng = np.random.default_rng(SWAY_SEED + r.sway_count)
        r.sway_count += 1
        r.stack.clear()
        r.ctrl.reset()

    def _targets(self):
        edges = np.linspace(SWAY_MIN_M, SWAY_AMP_M, SWAY_BINS + 1)
        return [float(self.rng.uniform(lo, hi)) for lo, hi in zip(edges[:-1], edges[1:])]

    def _make_legs(self, t_now):
        right = body_to_world([0.0, -1.0, 0.0], self.ref_yaw)
        up = np.array([0.0, 0.0, 1.0])
        v_gain = 1.875 / SWAY_T_S                      # Quintic: T = 1.875 / v_gain for a short move
        for axis_name, axis in (("right", right), ("up", up)):
            end = self.ref_p + SWAY_AMP_M * axis
            for direction, (a, b) in (("out", (self.ref_p, end)), ("back", (end, self.ref_p))):
                tg = sorted(self._targets(), reverse=(direction == "back"))   # new per leg
                self.legs.append({"axis_name": axis_name, "axis": axis, "dir": direction,
                                  "traj": Quintic(a, b, v_gain=v_gain), "targets": tg, "k": 0})
        self.legs[0]["t0"] = t_now

    def _keep_frame(self, r, leg, offset):
        """One kept frame: flow reference -> now, rho with this frame's own measured translation."""
        s = r.state
        t_world = np.array(s.position_m, float) - self.ref_p
        t_cam = world_to_cam(t_world, self.ref_yaw)
        t0 = time.time()
        fl = r.flow(self.ref_rgb, r.rgb)
        r.stack.add(rho_from_flow(fl, t_cam, r.f, r.up_t, r.vp_t))
        ms = 1000.0 * (time.time() - t0)
        self.frames.append({"leg": self.i, "axis": leg["axis_name"], "dir": leg["dir"],
                            "target_m": round(leg["targets"][leg["k"]], 4),
                            "offset_axis_m": round(float(offset), 4),
                            "t_world_m": np.round(t_world, 4).tolist(),
                            "t_cam_m": np.round(t_cam, 4).tolist(),
                            "t_s": round(s.t_s, 3), "flow_ms": round(ms, 1)})
        r.log("frame %d: leg %d %s %s, target %.3f m, measured %.3f m, flow %.0f ms"
              % (len(self.frames), self.i, leg["axis_name"], leg["dir"], leg["targets"][leg["k"]],
                 offset, ms))

    def update(self, r):
        s = r.state
        if self.phase == "settle":
            v = r.velocity_to(self.hold, s)
            if s.t_s - self.t0 >= SETTLE_S:
                self.ref_rgb = np.array(r.rgb, np.float32, copy=True)
                self.ref_p = np.array(s.position_m, float)
                self.ref_yaw = float(s.yaw_rad)
                self._make_legs(s.t_s)
                self.phase = "legs"
                r.log("reference frame at %s" % np.round(self.ref_p, 3))
            return v, 0.0, None

        leg = self.legs[self.i]
        t = s.t_s - leg["t0"]
        p_d, v_d = leg["traj"].sample(t)
        v = r.velocity_to(p_d, s, v_d)
        offset = float(np.dot(np.array(s.position_m, float) - self.ref_p, leg["axis"]))

        def crossed(k):
            tgt = leg["targets"][k]
            return offset >= tgt if leg["dir"] == "out" else offset <= tgt

        if leg["k"] < len(leg["targets"]) and crossed(leg["k"]):
            self._keep_frame(r, leg, offset)              # one frame per tick, with its own pose
            leg["k"] += 1
            while leg["k"] < len(leg["targets"]) and crossed(leg["k"]):
                r.log("target %.3f m skipped: crossed on the same tick" % leg["targets"][leg["k"]])
                leg["k"] += 1
        if not leg["traj"].done(t, s.position_m):
            return v, 0.0, None
        self.i += 1
        r.ctrl.reset()
        if self.i < len(self.legs):
            self.legs[self.i]["t0"] = s.t_s
            return v, 0.0, None

        r.det = r.find_gap(self)                          # all legs done
        if r.det is not None:
            w = r.det["wall"]
            r.log("gap: %d frames, wall Z %.2f m, normal %.1f deg off axis, floor rejected %d, "
                  "gap %.1f%% of image, %d instances, %.0f ms"
                  % (len(self.frames), w["wall_z_m"], w["normal_vs_axis_deg"], w["floor_rejected"],
                     100.0 * r.det["mask"].mean(), r.det["n_instances"], r.det["ms"]))
        return v, 0.0, "move"


class Move:
    """After the gap is found. r.det holds the gap mask (and rho, wall, labels);
    r.locate(r.det, r.state) is where the 3D target should come from."""
    name = "move"

    def start(self, r):
        ### For AN to fill#####
        self.p = np.array(r.state.position_m, float)       # placeholder: hold here

    def update(self, r):
        ### For AN to fill#####
        return r.velocity_to(self.p, r.state), 0.0, None   # placeholder: hover


# --------------------------------------------------------------------------- #
# The policy
# --------------------------------------------------------------------------- #
class ModeGapFlyer(Policy):
    """Mode-class state machine + SEA-RAFT sway perception + P3 controller. RGB in, velocity out."""

    name = "p4_mode_gap_flyer"
    sensor_fingerprint = None      # set to the contract's sensor fingerprint once fixed

    def __init__(self, record_fpv=True):
        self.flow = SeaRaftFlow(IMG_H, IMG_W)              # loaded and warmed up once, here
        K = camera_matrix(IMG_W, IMG_H)
        self.f, self.cx, self.cy = float(K[0, 0]), float(K[0, 2]), float(K[1, 2])
        vv, uu = np.mgrid[0:IMG_H, 0:IMG_W].astype(np.float64)
        self.up, self.vp = uu + 0.5 - self.cx, vv + 0.5 - self.cy
        self.up_t = torch.from_numpy(self.up).float().to(self.flow.dev)
        self.vp_t = torch.from_numpy(self.vp).float().to(self.flow.dev)
        self.fpv = FpvRecorder() if record_fpv else None
        self.debug = []
        atexit.register(self.save_debug)
        self.reset()

    def reset(self):
        self.mode = "search"
        self.ctrl = PositionController(max_speed=1.0, max_climb=0.6)     # P3 limits
        self.modes = {m.name: m for m in (Search(), Move())}             # add new modes here
        self.cur = None
        self.state, self.rgb = None, None
        self.stack = RhoStack()
        self.sway_count = 0
        self.det = None

    # -- helpers the modes use ---------------------------------------------------
    def velocity_to(self, target_world, state, velocity_ff=(0.0, 0.0, 0.0)):
        return self.ctrl.step(state.position_m, target_world, velocity_ff)

    def find_gap(self, search):
        """Median rho over the stack -> wall -> gap mask, or None."""
        if len(self.stack) == 0:
            return None
        t0 = time.time()
        rho, mad = self.stack.median_mad()
        low = low_mask(rho, self.vp, self.f, search.ref_p[2])          # floor: out of RANSAC and hull
        wall = fit_wall(rho, self.f, self.cx, self.cy, np.random.default_rng(SWAY_SEED), keep=~low)
        gap, hull, inl, labels, n_inst = segment(rho, wall, self.up, self.vp, keep=~low)
        det = {"mask": gap, "rho": rho, "mad": mad, "hull": hull, "inliers": inl, "low": low,
               "labels": labels, "n_instances": n_inst, "wall": wall,
               "ref_rgb": search.ref_rgb, "ref_position_m": search.ref_p,
               "ref_yaw_rad": search.ref_yaw, "frames": search.frames,
               "ms": 1000.0 * (time.time() - t0)}
        self.debug.append(det)
        return det

    def locate(self, det, state):
        """Detection -> where the gap's centre is in the world, or None."""
        ### For AN to fill#####
        return None

    def log(self, msg):
        print("%6.2f s  [%s] %s" % (self.state.t_s, self.mode, msg))

    def switch(self, name):
        if self.cur is not None:
            self.log("-> %s" % name)
        self.mode = name
        self.cur = self.modes[name]
        self.ctrl.reset()
        self.cur.start(self)

    # -- the loop ----------------------------------------------------------------
    def act(self, obs, state) -> Action:
        if obs.rgb.shape[:2] != (IMG_H, IMG_W):
            raise RuntimeError("camera is %dx%d; this policy is built for %dx%d (p4_contract.json)"
                               % (obs.rgb.shape[1], obs.rgb.shape[0], IMG_W, IMG_H))
        self.state, self.rgb = state, obs.rgb
        if self.cur is None:
            self.switch("search")
        v_world, yaw_rate, nxt = self.cur.update(self)
        if nxt is not None:
            self.switch(nxt)
        if self.fpv is not None:
            self.fpv.add(obs.rgb, mask=None if self.det is None else self.det["mask"], mode=self.mode)
        return Action("velocity", "body_flu", world_to_body(v_world, state.yaw_rad),
                      yaw_rate_rad_s=float(yaw_rate))

    # -- debug output, written at exit -------------------------------------------
    def save_debug(self):
        if not self.debug:
            return
        os.makedirs(DEBUG_DIR, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for i, d in enumerate(self.debug):
            base = os.path.join(DEBUG_DIR, "%s_classes_sway%d" % (stamp, i))
            w = d["wall"]
            np.savez_compressed(base + ".npz", ref_rgb=d["ref_rgb"], rho=d["rho"], mad=d["mad"],
                                gap=d["mask"], hull=d["hull"], inliers=d["inliers"], labels=d["labels"],
                                low=d["low"],
                                ref_position_m=d["ref_position_m"], ref_yaw_rad=d["ref_yaw_rad"],
                                plane=w["plane"], thr=w["thr"])
            with open(base + "_frames.json", "w") as fh:
                json.dump({"frames": d["frames"], "wall_z_m": w["wall_z_m"],
                           "normal_vs_axis_deg": w["normal_vs_axis_deg"],
                           "floor_rejected": w["floor_rejected"], "n_instances": d["n_instances"],
                           "find_gap_ms": d["ms"]}, fh, indent=1)
            cv2.imwrite(base + ".png", debug_panel(d))
            cv2.imwrite(base + "_mask.png", d["mask"].astype(np.uint8) * 255)   # full res, binary
            print("debug: %s.npz / .png / _mask.png / _frames.json" % base)
        self.debug = []
