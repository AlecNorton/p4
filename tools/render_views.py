#!/usr/bin/env python3
"""Renders the drone camera's view at poses you pick, to test your gap finder
on the splat. These images are for testing only: never train on them.

    python3 tools/render_views.py --cpu                          # six views of the window
    python3 tools/render_views.py --cpu --pose 1.0 0.2 0.8 -20   # x y z in Vicon metres, yaw in degrees
    python3 tools/render_views.py --worker-port 7790 --out views

Writes view_NN.png per view, RGB at your contract's resolution: what your
policy gets, since P4 has no depth. The camera is level, as on the drone.
"""
import argparse
import math
import os
import sys

import numpy as np

from splat_hitl.bundle import SceneBundle
from splat_hitl.contract import PolicyContract
from splat_hitl.frames import matrix_to_rpy, rpy_to_matrix
from splat_hitl.renderer import SplatWorkerClient, worker_command

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)

#: x, y, z (Vicon metres), yaw (deg): the start, and on the way to the window
COURSE_VIEWS = [
    (0.0, 0.0, 0.70, 0.0),            # the start
    (0.0, 0.5, 0.90, -10.0),          # the start, 0.5 m to the left
    (0.0, -0.5, 0.90, 10.0),          # the start, 0.5 m to the right
    (1.2, 0.0, 1.00, 0.0),            # part of the way
    (1.8, 0.3, 0.80, -5.0),           # closer, low and to the left
    (1.8, -0.4, 1.30, 5.0),           # closer, high and to the right
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default=os.path.join(PACK, "scene"))
    ap.add_argument("--contract", default=os.path.join(PACK, "p4_contract.json"))
    ap.add_argument("--pose", nargs=4, type=float, action="append",
                    metavar=("X", "Y", "Z", "YAW_DEG"), help="repeat for more views")
    ap.add_argument("--out", default="views")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--worker-port", type=int, default=None)
    g.add_argument("--worker-python", default=None)
    g.add_argument("--cpu", action="store_true")
    a = ap.parse_args(argv)

    if not os.path.exists(a.contract):
        sys.exit("no %s: run python3 tools/make_contract.py" % a.contract)
    sensor = PolicyContract.load(a.contract).observation.sensor
    b = SceneBundle.load(a.scene)
    tf = b.transform()
    if a.worker_port is not None:
        r = SplatWorkerClient(sensor, port=a.worker_port)
    elif a.worker_python is not None:
        r = SplatWorkerClient(sensor, worker_command(b, python=a.worker_python))
    elif a.cpu:
        sys.path.insert(0, HERE)
        from cpu_renderer import CpuSplatRenderer
        r = CpuSplatRenderer(sensor, b)
    else:
        sys.exit("pick a renderer: --worker-port 7790, --worker-python PYTHON or --cpu")
    sys.path.insert(0, os.path.join(PACK, "Code"))
    from fpv import write_png

    os.makedirs(a.out, exist_ok=True)
    try:
        for i, (x, y, z, yaw_deg) in enumerate(a.pose or COURSE_VIEWS):
            p = tf.point_to_splat(np.array([x, y, z], float)).reshape(3) * tf.metres_per_unit
            rpy = matrix_to_rpy(tf.rotation_to_splat(rpy_to_matrix(0.0, 0.0, math.radians(yaw_deg))))
            o = r.render(p, rpy)
            stem = os.path.join(a.out, "view_%02d" % i)
            write_png(stem + ".png", (np.clip(o.rgb, 0, 1) * 255 + 0.5).astype(np.uint8))
            print("%s  (%.2f, %.2f, %.2f) m, yaw %.0f deg, %.1f s"
                  % (stem, x, y, z, yaw_deg, o.render_s))
    finally:
        r.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
