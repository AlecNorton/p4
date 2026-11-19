#!/usr/bin/env python3
"""P4 entry point: flies your policy in the simulator, through the same loop
the drone flies with.

    python3 Code/Wrapper.py --worker-port 7790        # the lab PC's render server
    python3 Code/Wrapper.py --worker-python PYTHON    # a GPU on this machine (Turing)
    python3 Code/Wrapper.py --cpu                     # no GPU: seconds per frame
    python3 Code/Wrapper.py --cpu --scene ../p4_starter/scene --policy policy_template:GapFlyer

The simulated drone follows each command exactly: no lag, no drag, no wind.
Your policy gets what it gets in flight from tools/fly.py: the RGB image with
no depth, and the state without the gate fields. The run is scored the same
way: through the gap or not, a virtual collision within --clearance-m of the
scene, and the time. Writes run.json, run.png (if matplotlib is installed) and
your policy's FPV frames.

The starter pack is --pack, else $P4_PACK, else ../../p4_starter from this file.
Use relative paths: we run this on our machine.
"""
import argparse
import importlib
import json
import math
import os
import sys
import time
from dataclasses import replace

import numpy as np

from splat_hitl.bundle import SceneBundle
from splat_hitl.collision import CollisionMonitor
from splat_hitl.contract import PolicyContract
from splat_hitl.frames import matrix_to_quat, rpy_to_matrix
from splat_hitl.policy import Policy
from splat_hitl.renderer import SplatWorkerClient, worker_command
from splat_hitl.runtime import (PoseSample, PoseSource, Runtime, RuntimeConfig,
                                TransformedPoseSource)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PACK = os.path.normpath(os.path.join(HERE, "..", "..", "p4_starter"))


def _is_pack(d):
    return bool(d) and os.path.isdir(os.path.join(d, "scene"))


def resolve_pack(arg):
    for value, source in ((arg, "--pack"), (os.environ.get("P4_PACK"), "$P4_PACK")):
        if value is not None:
            if _is_pack(value):
                return os.path.abspath(value)
            sys.exit("%s = %s is not a starter pack: no scene/ inside it." % (source, value))
    if _is_pack(DEFAULT_PACK):
        return DEFAULT_PACK
    sys.exit("no starter pack at %s. Pass --pack or set P4_PACK." % DEFAULT_PACK)


class SimDrone(PoseSource):
    """The drone, in Vicon metres. Follows each Hover command exactly."""

    def __init__(self, position_m, yaw_rad):
        self.p = np.array(position_m, float).reshape(3)
        self.yaw = float(yaw_rad)
        self.t = 0.0

    def latest(self):
        return PoseSample(self.t, self.p.copy(), matrix_to_quat(rpy_to_matrix(0.0, 0.0, self.yaw)))

    def step(self, cmd, dt):
        if cmd is not None:
            c, s = math.cos(self.yaw), math.sin(self.yaw)
            self.p[0] += (c * cmd.vx - s * cmd.vy) * dt       # body -> world
            self.p[1] += (s * cmd.vx + c * cmd.vy) * dt
            self.p[2] = cmd.z_distance                         # an absolute height
            self.yaw = (self.yaw + cmd.yaw_rate * dt + math.pi) % (2 * math.pi) - math.pi
        self.t += dt


class CameraOnly(Policy):
    """Your policy, handed the RGB image without depth and the state without
    the gate fields: what tools/fly.py hands it on the drone."""

    def __init__(self, inner):
        self.inner = inner
        self.name = inner.name
        self.sensor_fingerprint = inner.sensor_fingerprint
        if hasattr(inner, "contract_fingerprint"):
            self.contract_fingerprint = inner.contract_fingerprint

    def reset(self):
        self.inner.reset()

    def act(self, obs, state):
        return self.inner.act(replace(obs, depth_m=None),
                              replace(state, gate_index=0, gate_distance_m=None,
                                      gate_bearing_rad=None))


def load_policy(spec):
    """'module:Class' -> an instance. The module is looked up next to this file first."""
    if ":" not in spec:
        sys.exit("--policy wants module:Class, e.g. policy_template:GapFlyer")
    mod, cls = spec.split(":", 1)
    sys.path.insert(0, HERE)
    return getattr(importlib.import_module(mod), cls)()


def make_renderer(a, sensor, bundle, pack):
    if a.worker_port is not None or a.worker_python is not None:
        r = (SplatWorkerClient(sensor, port=a.worker_port) if a.worker_port is not None
             else SplatWorkerClient(sensor, worker_command(bundle, python=a.worker_python)))
        want = bundle.transform().metres_per_unit
        if abs(r.scale_to_metres - want) > 1e-4 * want:          # as ros_node checks
            r.close()
            sys.exit("the renderer is drawing another scene (%s), not %s. Start the "
                     "render server on this scene." % (r.scene or "unknown", bundle.name))
        return r
    if a.cpu:
        sys.path.insert(0, os.path.join(pack, "tools"))
        from cpu_renderer import CpuSplatRenderer
        print("CPU renderer: several seconds per frame, so a run takes a while.")
        return CpuSplatRenderer(sensor, bundle)
    sys.exit("pick a renderer: --worker-port 7790 (lab PC), --worker-python PYTHON "
             "(a GPU here) or --cpu (slow). See install.md.")


def fly(bundle, contract, renderer, policy, start_m, start_yaw_rad,
        clearance_m=0.05, max_s=120.0, verbose=True):
    """One run in the simulator. Returns the Runtime and a per-tick log."""
    tf = bundle.transform()
    drone = SimDrone(start_m, start_yaw_rad)
    source = TransformedPoseSource(drone, tf)
    cfg = RuntimeConfig(hold_altitude_m=float(start_m[2]), contract=contract,
                        tick_budget_s=1e6,           # timing is reported, not enforced
                        max_duration_s=float(max_s))
    rt = Runtime(source, renderer, CameraOnly(policy), bundle.gates(),
                 CollisionMonitor(bundle.esdf(), clearance_m), cfg)
    dt = contract.control.dt_s
    log = []
    try:
        while True:
            r = rt.step(now=drone.t)
            s = source.latest()
            log.append({"t": round(drone.t, 4), "state": r.state,
                        "vicon_m": drone.p.round(4).tolist(),
                        "scene_m": s.position_m.round(4).tolist(),
                        "yaw_deg": round(math.degrees(drone.yaw), 2),
                        "mode": getattr(policy, "mode", None),
                        "cmd": None if r.command is None else
                        [round(r.command.vx, 4), round(r.command.vy, 4),
                         round(r.command.yaw_rate, 4), round(r.command.z_distance, 4)],
                        "policy_ms": round(r.policy_s * 1000.0, 2),
                        "render_ms": round(r.render_s * 1000.0, 2),
                        "events": list(r.events)})
            if verbose:
                for e in r.events:
                    print("%6.2f s  %s" % (drone.t, e))
                if rt.steps % 15 == 0 and not r.finished:
                    print("%6.2f s  at (%.2f, %.2f, %.2f) m, yaw %4.0f deg, %s"
                          % (drone.t, drone.p[0], drone.p[1], drone.p[2],
                             math.degrees(drone.yaw), getattr(policy, "mode", "")))
            if r.finished:
                break
            drone.step(r.command, dt)
    except KeyboardInterrupt:
        rt.stop("operator_stop")
        print("\nstopped")
    return rt, log


def through_at(log):
    """Simulated time the drone went through the gap, or None."""
    return next((r["t"] for r in log if any("PASSED" in e for e in r["events"])), None)


def report(rt, log):
    print("\n" + rt.summary())
    t_gap = through_at(log)
    print("  the gap     : %s" % ("flown through at %.2f s" % t_gap if t_gap is not None
                                  else "not flown through"))
    if log:
        print("  time        : %.2f s simulated" % log[-1]["t"])
    pol = np.array([r.policy_s for r in rt.history if r.policy_s > 0]) * 1000.0
    ren = np.array([r.render_s for r in rt.history if r.render_s > 0]) * 1000.0
    if len(pol):
        print("  your policy : %.0f ms mean, %.0f ms worst, per frame" % (pol.mean(), pol.max()))
    if len(ren):
        print("  rendering   : %.0f ms mean, per frame" % ren.mean())
    print("  (on the lab PC, rendering and your policy share one 66 ms frame)")


def plot(log, gates, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        return None
    P = np.array([r["scene_m"] for r in log])
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection = '3d')
    ax.plot(P[:, 0], P[:, 1], P[:, 2], "-", color="tab:blue", lw=1.5, label="drone")
    ax.plot(P[0, 0], P[0, 1], P[0, 2], "o", color="tab:blue")
    for g in gates.gates:
        r = g.right * g.width_m / 2.0
        u = g.up * g.height_m / 2.0
        a, b = g.centre - r, g.centre + r
        a, b, c, d = a+u, b+u, b-u, a-u #Left upper corner, right upper corner, right lower corner, left lower corner 
        verts = list_of_vertices(a, b, c, d)
        print(f"Verts: {verts}")
        poly = Poly3DCollection(verts, alpha=.3)
        poly.set_facecolor((1, 0, 0))
        poly.set_edgecolor('black')
        ax.add_collection(poly)
        #ax.annotate(g.name, g.centre[:2], textcoords="offset points", xytext=(6, 6))
    ax.set_xlim(-1, 5)
    ax.set_ylim(-2, 2)
    ax.set_zlim(0, 3)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")

    ax.set_ylabel("y (m)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path

def list_of_vertices(a, b, c, d):
    face1 = [a, b, c, d, a]
    verts = []
    verts.append(face1)
    return np.asarray(verts)

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", default=None, help="path to p4_starter")
    ap.add_argument("--scene", default=None, help="scene folder; default <pack>/scene")
    ap.add_argument("--contract", default=None,
                    help="default <pack>/p4_contract.json, from tools/make_contract.py")
    ap.add_argument("--policy", default="policy_template:GapFlyer", help="module:Class")
    ap.add_argument("--pair", default=None,
                    help="start pair in the scene's task.json; default the first")
    ap.add_argument("--clearance-m", type=float, default=0.05)
    ap.add_argument("--max-s", type=float, default=120.0, help="simulated seconds")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--worker-port", type=int, default=None,
                   help="render server on this machine (the lab PC: 7790)")
    g.add_argument("--worker-python", default=None,
                   help="a python with torch and gsplat: starts a GPU worker here")
    g.add_argument("--cpu", action="store_true", help="render with numpy (slow)")
    ap.add_argument("--log", default="run.json")
    a = ap.parse_args(argv)

    pack = resolve_pack(a.pack)
    scene = os.path.abspath(a.scene or os.path.join(pack, "scene"))
    contract_path = a.contract or os.path.join(pack, "p4_contract.json")
    if not os.path.exists(contract_path):
        sys.exit("no %s: run python3 tools/make_contract.py in the starter pack." % contract_path)
    contract = PolicyContract.load(contract_path)
    if contract.observation.channels != "rgb":
        sys.exit("%s hands the policy %s; P4 is RGB only. Write it again with "
                 "python3 tools/make_contract.py." % (contract_path, contract.observation.channels))
    bundle = SceneBundle.load(scene)
    task = bundle.task()
    pair = (task["pairs"][0] if a.pair is None
            else next((p for p in task["pairs"] if p["name"] == a.pair), None))
    if pair is None:
        sys.exit("no pair %r in %s/task.json. Have: %s"
                 % (a.pair, scene, ", ".join(p["name"] for p in task["pairs"])))
    start = np.array(pair["start"], float)
    yaw0 = math.radians(float(pair.get("start_yaw_deg", 0.0)))

    print("scene    : %s" % scene)
    print("contract : %s" % contract)
    print("start    : %s m, yaw %.0f deg (Vicon)" % (start, math.degrees(yaw0)))

    policy = load_policy(a.policy)
    renderer = make_renderer(a, contract.observation.sensor, bundle, pack)
    t0 = time.time()
    try:
        rt, log = fly(bundle, contract, renderer, policy, start, yaw0,
                      a.clearance_m, a.max_s)
    finally:
        renderer.close()
    report(rt, log)
    print("  wall clock  : %.0f s" % (time.time() - t0))

    with open(a.log, "w") as fh:
        json.dump({"scene": bundle.name, "pair": pair["name"], "policy": a.policy,
                   "outcome": rt.reason or rt.state,
                   "through_gap": through_at(log) is not None,
                   "time_s": log[-1]["t"] if log else 0.0,
                   "ticks": log}, fh)
    print("wrote %s" % a.log)
    png = plot(log, bundle.gates(), os.path.splitext(a.log)[0] + ".png")
    if png:
        print("wrote %s" % png)
    fpv = getattr(policy, "fpv", None)
    if fpv is not None:
        fpv.save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
