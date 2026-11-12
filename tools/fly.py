#!/usr/bin/env python3
"""ros_node for P4: your policy gets the RGB image and no depth.

    PYTHONPATH=Code:$PYTHONPATH python3 ../p4_starter/tools/fly.py \\
        --topic /vicon/<object>/<object> ... --policy policy_template:GapFlyer

Every flag is ros_node's and means the same (python3 -m splat_hitl.ros_node -h
lists them). This does not change splat_hitl: it starts ros_node with your
policy wrapped so that it gets what it gets in Wrapper.py -- obs.depth_m is
None and the state has no gate fields -- and it refuses a contract that would
hand your policy depth through obs.policy_input. Run it where you would run
ros_node: in the lab container.
"""
import argparse
import sys
from dataclasses import replace

import splat_hitl.ros_node as ros_node
from splat_hitl.contract import PolicyContract
from splat_hitl.policy import Policy


class CameraOnly(Policy):
    """Your policy, handed the RGB image without depth and the state without
    the gate fields."""

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


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if "-h" in argv or "--help" in argv:
        print(__doc__)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--contract", default=None)
    a, _ = ap.parse_known_args(argv)
    if a.contract is not None:
        channels = PolicyContract.load(a.contract).observation.channels
        if channels != "rgb":
            sys.exit("%s hands the policy %s; P4 is RGB only. Write it again with "
                     "python3 tools/make_contract.py." % (a.contract, channels))

    # ros_node loads the policy through this name when it starts, so the policy
    # it flies is the wrapped one.
    load = ros_node.load_policy
    ros_node.load_policy = lambda spec: CameraOnly(load(spec))
    return ros_node.main(argv)


if __name__ == "__main__":
    sys.exit(main())
