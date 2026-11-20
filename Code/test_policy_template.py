#!/usr/bin/env python3
"""Your P4 policy: find the gap in the window panel, from the camera's RGB
images alone, and fly through it.

    python3 Code/Wrapper.py --pack . --policy test_policy_template:GapFlyer --worker-python PYTHON
    tools/fly.py ... --policy test_policy_template:GapFlyer          # on the drone

act(obs, state) is called 15 times a second with

    obs.rgb          (H, W, 3) float32 in [0, 1]
    obs.depth_m      None: P4 has no depth. Wrapper.py and tools/fly.py take
                     it out before your policy sees the frame
    state.t_s        seconds since the run started
    state.position_m, state.velocity_ms
                     scene metres and m/s (our scenes are stored in Vicon metres)
    state.yaw_rad    heading, counter-clockwise from +x

and returns Action("velocity", "body_flu", [vx, vy, vz], yaw_rate_rad_s=w):
m/s in the body frame (x forward, y left, z up) and a counter-clockwise yaw
rate in rad/s. The camera sits at the drone's centre, looks along body +x and
turns with it.

Rendering, your perception and act() share one 66 ms frame on the lab PC.
Where the gap is must come from the images: do not read the scene's files,
state.gate_index, state.gate_distance_m or state.gate_bearing_rad, and do not
hard-code where the window or the gap is.

Our approach (this file is the starter's policy_template.py with the TODOs filled):
  search  sway: settle 0.5 s, keep a reference frame, then four quintic legs
          (right out, right back, up out, up back). On each leg a frame is kept
          when the MEASURED offset along the leg first crosses one of 5
          stratified-random targets in [SWAY_MIN_M, SWAY_AMP_M]; each leg draws
          new targets. SEA-RAFT gives the flow reference -> frame, turned into
          inverse depth rho with that frame's own measured translation, and
          stacked. Then find_gap(): median rho -> wall plane -> gap mask.
  move    placeholder, for AN.

Third-party code: SEA-RAFT (Wang, Lipson, Deng, ECCV 2024, BSD-3), in Code/searaft.
"""
import atexit
import json
import math
import os
import sys
import time
from argparse import Namespace

import cv2
import numpy as np
import torch
from furthest import furthest

from splat_hitl.commands import Action
from splat_hitl.policy import Policy

from control import PositionController, Quintic
from fpv import FpvRecorder

HERE = os.path.dirname(os.path.abspath(__file__))
FOV_X_DEG = 60.656255852364644         # fixed: the drone camera's field of view

# --- ours ------------------------------------------------------------------------
SEARAFT_DIR = os.path.join(HERE, "searaft")
IMG_W, IMG_H = 480, 320                 # p4_contract.json
# sway (the search motion)
SETTLE_S = 0.5                          # hover before the reference frame
SWAY_AMP_M = 0.20                       # leg length
SWAY_MIN_M = 0.07                       # smallest offset a frame is kept at
SWAY_T_S = 2.0                          # duration of one leg (0 -> A or A -> 0)
SWAY_BINS = 5                           # frames per leg
SWAY_SEED = 0
# wall / gap
TOP_FRAC = 0.5                          # RANSAC on the top 50% of the pooled rho
INLIER_REL = 0.05                       # |rho - plane| < 5% of the median selected rho
RANSAC_ITERS = 500
POOL = 2                                # 2x2 max-pool of rho, for RANSAC only
FLOOR_DEG = 45.0                        # plane normal within this of world up = floor
MIN_PATCH = 0.01                        # inlier patches kept for the hull (fraction of image)
MIN_HEIGHT_M = 0.3                      # pixels whose world height is below this (floor) are left out
                                        # of RANSAC and of the hull's inlier patches
OPEN_PX = 5                             # gap mask clean-up: erode then dilate (opening), ellipse kernel;
                                        # removes slivers/specks thinner than about this many pixels
DEBUG_DIR = os.path.join("fpv", "debug")


# --------------------------------------------------------------------------- #
# Geometry helpers (starter)
# --------------------------------------------------------------------------- #
def camera_matrix(width, height, fov_x_deg=FOV_X_DEG):
    """Pinhole intrinsics of the drone camera: square pixels, centre at the middle."""
    f = 0.5 * width / math.tan(math.radians(fov_x_deg) / 2.0)
    return np.array([[f, 0.0, width / 2.0],
                     [0.0, f, height / 2.0],
                     [0.0, 0.0, 1.0]])


def pixel_ray(u, v, K):
    """Pixel (u, v) -> unit ray in the camera frame (x right, y down, z forward).

    Pixel (i, j) covers [i, i + 1) x [j, j + 1): its centre is (i + 0.5, j + 0.5).
    """
    d = np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])
    return d / np.linalg.norm(d)


def camera_to_world_dir(d_cam, yaw_rad):
    """A direction in the camera frame -> the world. Uses the yaw only; roll
    and pitch are a few degrees."""
    d = np.asarray(d_cam, float)
    body = np.array([d[2], -d[0], -d[1]])                 # x forward, y left, z up
    return body_to_world(body, yaw_rad)

def pixel_to_camera(u, v, depth_m, K):
    """Pixel (u, v) at depth z -> camera frame (x right, y down, z forward).

    Pixel (i, j) covers [i, i + 1) x [j, j + 1): its centre is (i + 0.5, j + 0.5).
    """
    u, v, z = (np.asarray(a, float) for a in (u, v, depth_m))
    return np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], axis=-1)


def camera_to_world(p_cam, state):
    """Camera frame -> world. Uses the yaw only; roll and pitch are a few degrees."""
    p = np.asarray(p_cam, float)
    body = np.stack([p[..., 2], -p[..., 0], -p[..., 1]], axis=-1)        # FLU
    c, s = math.cos(state.yaw_rad), math.sin(state.yaw_rad)
    world = np.stack([c * body[..., 0] - s * body[..., 1],
                      s * body[..., 0] + c * body[..., 1], body[..., 2]], axis=-1)
    return np.asarray(state.position_m, float) + world

def triangulate(p1, d1, p2, d2):
    """Where two rays p + t d (world frame, t > 0) pass closest.

    Returns (point, miss_m): the midpoint of the closest approach and how far
    apart the rays pass there. (None, inf) when the rays are nearly parallel --
    the camera has not moved far enough across them to tell how far away the
    point is -- or when they meet behind the camera.
    """
    p1, d1, p2, d2 = (np.asarray(a, float) for a in (p1, d1, p2, d2))
    w = p1 - p2
    a, b, c = d1 @ d1, d1 @ d2, d2 @ d2
    d, e = d1 @ w, d2 @ w
    den = a * c - b * b
    if den < 1e-9:
        return None, float("inf")
    t1, t2 = (b * e - c * d) / den, (a * e - b * d) / den
    if t1 <= 0.0 or t2 <= 0.0:
        return None, float("inf")
    q1, q2 = p1 + t1 * d1, p2 + t2 * d2
    return (q1 + q2) / 2.0, float(np.linalg.norm(q1 - q2))


def body_to_world(v_body, yaw_rad):
    """Body (x forward, y left, z up) -> world."""
    v = np.asarray(v_body, float)
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])


def world_to_body(v_world, yaw_rad):
    """World velocity -> body (x forward, y left, z up)."""
    v = np.asarray(v_world, float)
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    return np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1], v[2]])


def wrap_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def world_to_cam(v_world, yaw_rad):
    """(ours) World vector -> camera frame (x right, y down, z forward). Yaw only."""
    b = world_to_body(v_world, yaw_rad)
    return np.array([-b[1], -b[2], b[0]])


# --------------------------------------------------------------------------- #
# Perception helpers (ours)
# --------------------------------------------------------------------------- #
class SeaRaftFlow:
    """SEA-RAFT-S at full resolution (scale 0). __call__(rgb0, rgb1) -> (2, H, W) torch flow
    on rgb0's pixel grid, in pixels, on the model's device."""

    def __init__(self, height, width, cfg="spring-S", model="Tartan-C-T-TSKH-spring540x960-S"):
        sys.path.insert(0, os.path.join(SEARAFT_DIR, "core"))
        from raft import RAFT
        import torchvision.models as tvm
        with open(os.path.join(SEARAFT_DIR, "config", "eval", cfg + ".json")) as fh:
            args = Namespace(**json.load(fh))
        args.scale = 0                                  # run at 480 x 320 as is
        self.iters = args.iters
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # SEA-RAFT builds its encoders from ImageNet ResNet18 weights (a download);
        # the checkpoint overwrites them, so build them untrained: no internet needed.
        orig = tvm.resnet18
        tvm.resnet18 = lambda *a, **k: orig(weights=None)
        try:
            m = RAFT.from_pretrained(os.path.join(SEARAFT_DIR, "models", model), args=args)
        finally:
            tvm.resnet18 = orig
        self.model = m.to(self.dev).eval()
        blank = np.zeros((height, width, 3), np.float32)
        for _ in range(3):                              # warm-up here, not in flight
            self(blank, blank)
        print("SEA-RAFT-S on %s, %dx%d, %d iters" % (self.dev, width, height, self.iters))

    def _tensor(self, rgb01):
        t = torch.from_numpy(np.ascontiguousarray(rgb01, dtype=np.float32))
        return t.permute(2, 0, 1)[None].to(self.dev) * 255.0

    @torch.no_grad()
    def __call__(self, rgb0, rgb1):
        out = self.model(self._tensor(rgb0), self._tensor(rgb1), iters=self.iters, test_mode=True)
        return out["flow"][-1][0]


def rho_from_flow(flow, t_cam, f, up, vp):
    """Inverse depth (1/m) from flow and the camera's translation (camera frame, metres).

    flow = rho * g,  g = (-f tx + u' tz, -f ty + v' tz);  rho = (flow . g) / |g|^2.
    up, vp: centred pixel coordinates (torch, H x W). Returns float16 torch (H, W).
    """
    tx, ty, tz = (float(x) for x in t_cam)
    gx = -f * tx + up * tz
    gy = -f * ty + vp * tz
    rho = (flow[0] * gx + flow[1] * gy) / (gx * gx + gy * gy + 1e-12)
    return rho.half()


class RhoStack:
    """Per-pixel stack of rho maps on the model's device (float16)."""

    def __init__(self):
        self.maps = []

    def clear(self):
        self.maps = []

    def add(self, rho):
        self.maps.append(rho)

    def __len__(self):
        return len(self.maps)

    def median_mad(self):
        st = torch.stack(self.maps).float()
        med = st.median(dim=0).values
        mad = (st - med).abs().median(dim=0).values
        return med.cpu().numpy(), mad.cpu().numpy()


UP_CAM = np.array([0.0, -1.0, 0.0])              # world up in the camera frame (no roll)
AXIS_CAM = np.array([0.0, 0.0, 1.0])


def _angle_deg(a, b):
    return math.degrees(math.acos(min(1.0, abs(float(np.dot(a, b))))))


def ransac_plane(u, v, r, thr, iters, rng):
    """Plane r = A u + B v + C by RANSAC (|residual| < thr), refitted on its inliers."""
    X = np.stack([u, v, np.ones_like(u)], 1)
    best, best_n = None, -1
    for _ in range(iters):
        idx = rng.choice(len(r), 3, replace=False)
        try:
            p = np.linalg.solve(X[idx], r[idx])
        except np.linalg.LinAlgError:
            continue
        n = int((np.abs(X @ p - r) < thr).sum())
        if n > best_n:
            best, best_n = p, n
    inl = np.abs(X @ best - r) < thr
    p, *_ = np.linalg.lstsq(X[inl], r[inl], rcond=None)
    return p


def plane_normal_cam(p, f):
    """rho = A u' + B v' + C -> unit normal (camera frame), pointing back at the camera."""
    n = np.array([p[0] * f, p[1] * f, p[2]])
    n /= np.linalg.norm(n)
    return n if n[2] < 0 else -n


def low_mask(rho, vp, f, ref_z):
    """Pixels whose world height is below MIN_HEIGHT_M (the floor). Height from the median
    rho: Z = 1/rho, camera y = Z v'/f (down), world z = ref_z - y (no roll). rho <= 0 is
    left alone (no depth; it is excluded later anyway)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        z_w = ref_z - (vp / f) / rho
    return (rho > 0) & (z_w < MIN_HEIGHT_M)


def fit_wall(rho, f, cx, cy, rng, keep=None):
    """Wall plane from rho: 2x2 max-pool, top TOP_FRAC, RANSAC; a floor is dropped once.
    keep (H, W) bool: pixels allowed in (False = left out, e.g. below MIN_HEIGHT_M).
    Max-pool chosen on purpose (it keeps outlier pixels): flag it if it fails."""
    h, w = rho.shape
    hp, wp = h // POOL, w // POOL
    src = rho if keep is None else np.where(keep, rho, -np.inf)       # left-out pixels never win the max
    pooled = src[:hp * POOL, :wp * POOL].reshape(hp, POOL, wp, POOL).max(axis=(1, 3))
    vs, us = np.mgrid[0:hp, 0:wp]
    u = (us * POOL + POOL / 2.0 - cx).ravel()          # centre of each pooled cell, centred coords
    v = (vs * POOL + POOL / 2.0 - cy).ravel()
    r = pooled.ravel().astype(np.float64)
    alive = np.isfinite(r)
    info = {"floor_rejected": 0}
    for attempt in range(2):
        cand = np.where(alive)[0]
        cut = np.quantile(r[cand], 1.0 - TOP_FRAC)
        sel = cand[r[cand] >= cut]
        thr = INLIER_REL * float(np.median(r[sel]))
        p = ransac_plane(u[sel], v[sel], r[sel], thr, RANSAC_ITERS, rng)
        n = plane_normal_cam(p, f)
        if attempt == 0 and _angle_deg(n, UP_CAM) < FLOOR_DEG:
            info["floor_rejected"] = 1
            alive &= ~(np.abs(u * p[0] + v * p[1] + p[2] - r) < thr)
            continue
        break
    info.update(plane=p, thr=thr, normal_cam=n, normal_vs_axis_deg=_angle_deg(n, AXIS_CAM),
                wall_z_m=(1.0 / p[2]) if p[2] > 0 else float("nan"))
    return info


def segment(rho, wall, up, vp, keep=None):
    """Full resolution: hull of all inlier patches >= MIN_PATCH; gap = outliers behind the
    plane inside it. keep (H, W) bool: pixels allowed as inliers (False = left out).
    Returns gap, hull, inliers (bool), instance labels, instance count."""
    h, w = rho.shape
    A, B, C = wall["plane"]
    plane = A * up + B * vp + C
    inl = np.abs(rho - plane) < wall["thr"]
    if keep is not None:
        inl &= keep
    inl = inl.astype(np.uint8)
    hull = np.zeros((h, w), np.uint8)
    nlab, lab, stats, _ = cv2.connectedComponentsWithStats(inl, connectivity=8)
    keep = [i for i in range(1, nlab) if stats[i, cv2.CC_STAT_AREA] >= MIN_PATCH * h * w]
    if keep:
        cnts, _ = cv2.findContours(np.isin(lab, keep).astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
        cv2.fillPoly(hull, [cv2.convexHull(np.vstack(cnts))], 1)
    gap = ((hull > 0) & (rho < plane - wall["thr"])).astype(np.uint8)
    if OPEN_PX > 1:                                     # clean-up: erode, then dilate back
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_PX, OPEN_PX))
        gap = cv2.dilate(cv2.erode(gap, k), k)
    gap = gap > 0
    n_inst, labels = cv2.connectedComponents(gap.astype(np.uint8), connectivity=8)
    return gap, hull > 0, inl > 0, labels, n_inst - 1


# --------------------------------------------------------------------------- #
# The policy (starter class, TODOs filled)
# --------------------------------------------------------------------------- #
class GapFlyer(Policy):
    """Gap finder + state machine + your P2b controller. RGB in, velocity out."""
    
    name = "p4_gap_flyer"
    # Once your perception is fixed at one resolution, set this to your contract's
    # PolicyContract.load("p4_contract.json").observation.sensor.fingerprint()
    # and the runtime will refuse to fly it through any other camera.
    sensor_fingerprint = None

    def __init__(self, record_fpv=True):
        ######## TODO - LOAD WHAT YOUR PERCEPTION NEEDS ########
        self.flow = SeaRaftFlow(IMG_H, IMG_W)          # loaded and warmed up once, here
        self.K = camera_matrix(IMG_W, IMG_H)
        self.f, self.cx, self.cy = float(self.K[0, 0]), float(self.K[0, 2]), float(self.K[1, 2])
        vv, uu = np.mgrid[0:IMG_H, 0:IMG_W].astype(np.float64)
        self.up, self.vp = uu + 0.5 - self.cx, vv + 0.5 - self.cy      # pixel centres, centred
        self.up_t = torch.from_numpy(self.up).float().to(self.flow.dev)
        self.vp_t = torch.from_numpy(self.vp).float().to(self.flow.dev)
        self.debug = []
        atexit.register(self.save_debug)
        ######## END TODO ########
        self.fpv = FpvRecorder() if record_fpv else None
        self.target = None
        self.safety_dist = .2
        self.reset()

    def reset(self):
        """Called once before each run. Reset the state machine here."""
        self.mode = "search"
        self.prev = None            # (rgb, position_m, yaw_rad) of the last frame
        self.gap = None             # your estimate of the gap's centre, world metres
        # ours
        self.ctrl = PositionController(max_speed=1.0, max_climb=0.6)     # P3 limits
        self.stack = RhoStack()
        self.sway = None            # the running sway (search motion); None = start a new one
        self.sway_count = 0
        self.det = None             # find_gap's result
        self.move_hold = None

    def log(self, state, msg):
        print("%6.2f s  [%s] %s" % (state.t_s, self.mode, msg))

    # -- perception ----------------------------------------------------------
    def add_frame(self, rgb, state, leg, offset_axis):
        """(ours) One kept sway frame: flow reference -> rgb, rho with this frame's own
        measured translation, onto the stack. The pose is the one this frame was rendered at."""
        sw = self.sway
        t_world = np.array(state.position_m, float) - sw["ref_p"]
        t_cam = world_to_cam(t_world, sw["ref_yaw"])
        t0 = time.time()
        fl = self.flow(sw["ref_rgb"], rgb)
        self.stack.add(rho_from_flow(fl, t_cam, self.f, self.up_t, self.vp_t))
        ms = 1000.0 * (time.time() - t0)
        sw["frames"].append({"leg": sw["i"], "axis": leg["axis_name"], "dir": leg["dir"],
                             "target_m": round(leg["targets"][leg["k"]], 4),
                             "offset_axis_m": round(float(offset_axis), 4),
                             "t_world_m": np.round(t_world, 4).tolist(),
                             "t_cam_m": np.round(t_cam, 4).tolist(),
                             "t_s": round(state.t_s, 3), "flow_ms": round(ms, 1)})
        self.log(state, "frame %d: leg %d %s %s, target %.3f m, measured %.3f m, flow %.0f ms"
                 % (len(sw["frames"]), sw["i"], leg["axis_name"], leg["dir"],
                    leg["targets"][leg["k"]], offset_axis, ms))

    def find_gap(self):
        """The gap, from the stacked frames of the last sway, or None.

        Returns a dict with "mask" (H, W) bool -- the FPV recorder draws it -- and
        rho, mad, wall, labels, n_instances, the reference frame and the frame log.
        """
        ######## TODO - FIND THE GAP ########
        if len(self.stack) == 0:
            return None
        sw = self.sway
        t0 = time.time()
        rho, mad = self.stack.median_mad()
        low = low_mask(rho, self.vp, self.f, sw["ref_p"][2])           # floor: out of RANSAC and hull
        wall = fit_wall(rho, self.f, self.cx, self.cy, np.random.default_rng(SWAY_SEED), keep=~low)
        gap, hull, inl, labels, n_inst = segment(rho, wall, self.up, self.vp, keep=~low)
        det = {"mask": gap, "rho": rho, "mad": mad, "hull": hull, "inliers": inl, "low": low,
               "labels": labels, "n_instances": n_inst, "wall": wall,
               "ref_rgb": sw["ref_rgb"], "ref_position_m": sw["ref_p"],
               "ref_yaw_rad": sw["ref_yaw"], "frames": sw["frames"],
               "ms": 1000.0 * (time.time() - t0)}
        self.debug.append(det)
        return det
        ######## END TODO ########

    def locate(self, det, state, scan = 0):
        """Detection -> where the gap's centre is in the world, or None."""
        ######## TODO - WHERE IS THE GAP ########
        try:
            w = self.det["wall"]
            depth = w["wall_z_m"]
            mask = det["mask"]
            pt = furthest(mask, scan)
            cam_points = pixel_to_camera(pt[0], pt[1], depth, self.K)
            world_point = camera_to_world(cam_points, state)
            return world_point
        except:
            return None

    # -- control -------------------------------------------------------------
    def velocity_to(self, target_world, state, velocity_ff=(0.0, 0.0, 0.0)):
        """Your P2b position loop: a world target -> a world velocity in m/s."""
        return self.ctrl.step(state.position_m, target_world, velocity_ff)

    # -- sway: the search motion (ours) ------------------------------------------
    def _targets(self, rng):
        edges = np.linspace(SWAY_MIN_M, SWAY_AMP_M, SWAY_BINS + 1)
        return [float(rng.uniform(lo, hi)) for lo, hi in zip(edges[:-1], edges[1:])]

    def start_sway(self, state):
        """A new sway from here. Repeatable: clears the stack."""
        self.sway = {"phase": "settle", "t0": state.t_s, "hold": np.array(state.position_m, float),
                     "rng": np.random.default_rng(SWAY_SEED + self.sway_count),
                     "legs": [], "i": 0, "frames": []}
        self.sway_count += 1
        self.stack.clear()
        self.ctrl.reset()

    def _make_legs(self, t_now):
        sw = self.sway
        p, yaw = sw["ref_p"], sw["ref_yaw"]
        right = body_to_world([0.0, -1.0, 0.0], yaw)
        up = np.array([0.0, 0.0, 1.0])
        v_gain = 1.875 / SWAY_T_S                      # Quintic: T = 1.875 / v_gain for a short move
        for axis_name, axis in (("right", right), ("up", up)):
            end = p + SWAY_AMP_M * axis
            for direction, (a, b) in (("out", (p, end)), ("back", (end, p))):
                tg = sorted(self._targets(sw["rng"]), reverse=(direction == "back"))   # new per leg
                sw["legs"].append({"axis_name": axis_name, "axis": axis, "dir": direction,
                                   "traj": Quintic(a, b, v_gain=v_gain), "targets": tg, "k": 0})
        sw["legs"][0]["t0"] = t_now

    def sway_step(self, rgb, state):
        """One tick of the sway. Returns (world velocity, finished)."""
        sw = self.sway
        if sw["phase"] == "settle":
            v = self.velocity_to(sw["hold"], state)
            if state.t_s - sw["t0"] >= SETTLE_S:
                sw["ref_rgb"] = np.array(rgb, np.float32, copy=True)
                sw["ref_p"] = np.array(state.position_m, float)
                sw["ref_yaw"] = float(state.yaw_rad)
                self._make_legs(state.t_s)
                sw["phase"] = "legs"
                self.log(state, "reference frame at %s" % np.round(sw["ref_p"], 3))
            return v, False

        leg = sw["legs"][sw["i"]]
        t = state.t_s - leg["t0"]
        p_d, v_d = leg["traj"].sample(t)
        v = self.velocity_to(p_d, state, v_d)
        offset = float(np.dot(np.array(state.position_m, float) - sw["ref_p"], leg["axis"]))

        def crossed(k):
            tgt = leg["targets"][k]
            return offset >= tgt if leg["dir"] == "out" else offset <= tgt

        if leg["k"] < len(leg["targets"]) and crossed(leg["k"]):
            self.add_frame(rgb, state, leg, offset)       # one frame per tick, with its own pose
            leg["k"] += 1
            while leg["k"] < len(leg["targets"]) and crossed(leg["k"]):
                self.log(state, "target %.3f m skipped: crossed on the same tick" % leg["targets"][leg["k"]])
                leg["k"] += 1
        if leg["traj"].done(t, state.position_m):
            sw["i"] += 1
            self.ctrl.reset()
            if sw["i"] == len(sw["legs"]):
                return v, True
            sw["legs"][sw["i"]]["t0"] = state.t_s
        return v, False

    # -- the state machine -----------------------------------------------------
    def act(self, obs, state) -> Action:
        if obs.rgb.shape[:2] != (IMG_H, IMG_W):
            raise RuntimeError("camera is %dx%d; this policy is built for %dx%d (p4_contract.json)"
                               % (obs.rgb.shape[1], obs.rgb.shape[0], IMG_W, IMG_H))

        v_world, yaw_rate = np.zeros(3), 0.0
        ######## TODO - THE STATE MACHINE ########
        # search  sway, collect frames, then find the gap (repeat: self.sway = None, mode "search")
        # move    after the gap is found
        if self.mode == "search":
            if self.sway is None:
                self.start_sway(state)
            v_world, finished = self.sway_step(obs.rgb, state)
            if finished:
                self.det = self.find_gap()
                if self.det is not None:
                    w = self.det["wall"]
                    self.log(state, "gap: %d frames, wall Z %.2f m, normal %.1f deg off axis, floor "
                             "rejected %d, gap %.1f%% of image, %d instances, %.0f ms"
                             % (len(self.det["frames"]), w["wall_z_m"], w["normal_vs_axis_deg"],
                                w["floor_rejected"], 100.0 * self.det["mask"].mean(),
                                self.det["n_instances"], self.det["ms"]))
                self.mode = "move"
                self.ctrl.reset()
                self.log(state, "-> move")
                #Assign target based on self.det
                #Move to a point self.safety_dist away from target based on normal. 
                self.target = self.locate(self.det, state)
                #self.log(state, f"Target before safety_dist: {self.target}")
                normal_vec = self.det["wall"]["normal_cam"]
                #self.log(state, f"Normal Vec: {camera_to_world_dir(normal_vec, state.yaw_rad)}")
                normal_vec = camera_to_world_dir(normal_vec, state.yaw_rad)
                self.target = np.add(np.multiply(normal_vec, self.safety_dist), self.target)
                self.log(state, f"Target: {self.target}")

        if self.mode == "move":
            ### For AN to fill#####
            if(self.target is not None):
                v_world = self.velocity_to(self.target, state)
                pos_diff = np.linalg.norm(np.subtract(state.position_m, self.target))
                if(pos_diff <= 1e-3):
                    self.mode = "align"
                    self.ctrl.reset()
                    self.log(state, "->align")
                    self.move_hold = np.array(state.position_m, float)

            else:
                v_world = self.velocity_to(self.move_hold, state)

        if(self.mode == "align"):
            des_angle = self.det["wall"]["normal_vs_axis"]
            self.log(state, f"Desired angle: {des_angle}")
            diff =  wrap_pi(des_angle - state.yaw_rad)
            if(abs(diff) < 1e-3):
                self.log(state, "mode -> scan again")
                self.mode = "scan_again"
                yaw_rate = 0
            else:
                yaw_rate = .01
                self.log(state, f"Rotating... current yaw: {state.yaw_rad}")
            
        if(self.mode == "scan_again"):
            normal_vec = self.det["wall"]["normal_cam"]
            if self.sway is None:
                self.start_sway(state)
            v_world, finished = self.sway_step(obs.rgb, state)
            if finished:
                self.det = self.find_gap()
                if self.det is not None:
                    w = self.det["wall"]
                    self.log(state, "gap: %d frames, wall Z %.2f m, normal %.1f deg off axis, floor "
                                "rejected %d, gap %.1f%% of image, %d instances, %.0f ms"
                                % (len(self.det["frames"]), w["wall_z_m"], w["normal_vs_axis_deg"],
                                w["floor_rejected"], 100.0 * self.det["mask"].mean(),
                                self.det["n_instances"], self.det["ms"]))
                self.mode = "commit"
                self.ctrl.reset()
                self.log(state, "-> move")
                #Assign target based on self.det
                #Move to a point self.safety_dist through  target based on normal. 
                self.target = self.locate(self.det, state, scan=1)
                #self.log(state, f"Target before safety_dist: {self.target}")
                normal_vec = self.det["wall"]["normal_cam"]
                #self.log(state, f"Normal Vec: {camera_to_world_dir(normal_vec, state.yaw_rad)}")
                normal_vec = camera_to_world_dir(-normal_vec, state.yaw_rad)
                self.target = np.add(np.multiply(normal_vec, .5), self.target)
                self.log(state, f"Target: {self.target}")
                
        if(self.mode=="commit"):
            if(self.target is not None):
                v_world = self.velocity_to(self.target, state)
                pos_diff = np.linalg.norm(np.subtract(state.position_m, self.target))
                if(pos_diff <= 1e-3):
                    self.mode = "done"
                    self.log(state, "mode -> done!")
                    self.ctrl.reset()
                    self.move_hold = np.array(state.position_m, float)
            else:
                v_world = self.velocity_to(self.move_hold, state)

        if(self.mode == "done"):
            self.move_hold = np.array(state.position_m, float)
            v_world = self.velocity_to(self.move_hold, state)

        
        if self.fpv is not None:
            self.fpv.add(obs.rgb, mask=None if self.det is None else self.det["mask"], mode=self.mode)
        self.prev = (obs.rgb, np.array(state.position_m, float), float(state.yaw_rad))
        return Action("velocity", "body_flu", world_to_body(v_world, state.yaw_rad),
                      yaw_rate_rad_s=float(yaw_rate))

    # -- debug output (ours), written at exit, not in the loop --------------------
    def save_debug(self):
        if not self.debug:
            return
        os.makedirs(DEBUG_DIR, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for i, d in enumerate(self.debug):
            base = os.path.join(DEBUG_DIR, "%s_sway%d" % (stamp, i))
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


def debug_panel(d):
    """(ours) ref RGB | median rho | MAD | gap overlay (hull yellow), side by side."""
    def heat(x, lo, hi):
        x8 = np.clip((x - lo) / max(hi - lo, 1e-12) * 255, 0, 255).astype(np.uint8)
        return cv2.applyColorMap(x8, cv2.COLORMAP_TURBO)
    rgb = cv2.cvtColor((np.clip(d["ref_rgb"], 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    lo, hi = np.nanpercentile(d["rho"], [1, 99])
    ov = rgb.copy()
    g = d["mask"]
    ov[g] = (0.5 * ov[g] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
    cnts, _ = cv2.findContours(d["hull"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ov, cnts, -1, (0, 255, 255), 1)
    return np.hstack([rgb, heat(d["rho"], lo, hi), heat(d["mad"], 0, np.nanpercentile(d["mad"], 99)), ov])


class HoverHere(Policy):
    """Zero velocity, camera recorded. Fly it first each session to check the loop."""

    name = "p4_hover"
    sensor_fingerprint = None

    def __init__(self):
        self.fpv = FpvRecorder()

    def act(self, obs, state) -> Action:
        self.fpv.add(obs.rgb)
        return Action("velocity", "body_flu", np.zeros(3))
