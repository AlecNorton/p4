#!/usr/bin/env python3
"""Your P4 policy: find the gap in the window panel, from the camera's RGB
images alone, and fly through it. Fill in the TODOs.

    python3 Code/Wrapper.py                                    # in the simulator
    tools/fly.py ... --policy policy_template:GapFlyer          # on the drone

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

You know how the camera moved between two frames, from state.position_m. A
point's motion in the image then tells you how far away it is: near things
move more than far ones. Through the gap you see the far wall, which barely
moves.

Rendering, your perception and act() share one 66 ms frame on the lab PC's
CPU. Where the gap is must come from the images: do not read the scene's
files, state.gate_index, state.gate_distance_m or state.gate_bearing_rad, and
do not hard-code where the window or the gap is.
"""
import math
import os

import numpy as np

from splat_hitl.commands import Action
from splat_hitl.policy import Policy

from fpv import FpvRecorder

HERE = os.path.dirname(os.path.abspath(__file__))
FOV_X_DEG = 60.656255852364644         # fixed: the drone camera's field of view


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

class WorldRay():
    def __init__(self, origin, direction):
        self.origin = np.asarray(origin, float)
        self.direction = np.asarray(direction, float)
    
    def __eq__(self, other):
        if isinstance(other, WorldRay):
            if(other.origin == self.origin and other.direction = self.direction):
                return True
        return False
    
    def dist(self, other):
        if isinstance(other, WorldRay):
            return np.linalg.norm(np.subtract(self.origin, other.origin))
        return None


class GapFlyer(Policy):
    """Gap finder + state machine + your P2b controller. RGB in, velocity out."""

    name = "p4_gap_flyer"
    # Once your perception is fixed at one resolution, set this to your contract's
    # PolicyContract.load("p4_contract.json").observation.sensor.fingerprint()
    # and the runtime will refuse to fly it through any other camera.
    sensor_fingerprint = None

    def __init__(self, record_fpv=True):
        ######## TODO - LOAD WHAT YOUR PERCEPTION NEEDS ########
        # A network's weights, say: keep them next to this file and load them
        # by a path built from HERE. Load once, here: never inside act().
        self.net = None
        ######## END TODO ########
        self.fpv = FpvRecorder() if record_fpv else None
        self.K = None
        #A list of all rays up to length self.rays_length for triangulation. 
        self.rays = []
        self.rays_length = 50
        self.reset()
        #How confident we want to be of gap before aligning ourselves with it. 
        self.confidence_threshold = .9

    def reset(self):
        """Called once before each run. Reset the state machine here."""
        self.mode = "search"
        self.prev = None            # (rgb, position_m, yaw_rad) of the last frame
        self.gap = None             # your estimate of the gap's centre, world metres

    # -- perception ----------------------------------------------------------
    def find_gap(self, rgb, prev):
        """The gap in this image, or None.

        prev is (rgb, position_m, yaw_rad) of the previous frame, or None on the
        first. Return a dict, e.g. {"mask": (H, W) bool, "centre": (u, v)}; the
        FPV recorder draws "mask" and "centre".
        """
        ######## TODO - FIND THE GAP ########
        # The panel is the nearest thing in front of you, and through the gap
        # you see the far wall. As the camera moves, near things move across
        # the image more than far ones, so the gap is the region inside the
        # panel whose image motion is small. Of several such regions, the
        # largest is the one to fly through.
        return None
        ######## END TODO ########

    def locate(self, det, state):
        """Detection -> where the gap's centre is in the world, or None."""
        ######## TODO - WHERE IS THE GAP ########
        # The gap's centre lies on the ray through det["centre"] (pixel_ray,
        # then camera_to_world_dir with state.yaw_rad), starting at
        # state.position_m. Two such rays from places far enough apart cross
        # at it (triangulate): keep the rays, not just the last one.
        pix_ray = self.pixel_ray(det["centre"][0], det["centre"][1], self.K)
        world_ray = self.camera_to_world_dir(pix_ray, state.yaw_rad)
        norm_world_ray = np.divide(world_ray, np.linalg.norm(world_ray))
        self.rays.append(WorldRay(state.position_m, world_ray))
        if(len(self.rays) > self.rays_length):
            self.rays.pop(0)
        one_ray, second_ray = self.get_distanced_rays()
        if(one_ray is None):
            return None
        midpoint, missed_dist = triangulate(one_ray.origin, one_ray.direction, second_ray.origin, second_ray.direction)
        return (midpoint, missed_dist)
        ######## END TODO ########

    def get_distanced_rays(self):
        max_dist = -inf
        chosen_ray_one = None
        chosen_ray_second = None
        for one_ray in self.rays:
            for second_ray in self.rays:
                if (one_ray == second_ray):
                    continue
                else:
                    dist = one_ray.dist(second_ray)
                    if(dist >= max_dist):
                        max_dist = dist
                        chosen_ray_one = one_ray
                        chosen_ray_second = second_ray
        return chosen_ray_one, chosen_ray_second

        


    # -- control -------------------------------------------------------------
    def velocity_to(self, target_world, state):
        """Your P2b position loop: a world target -> a world velocity in m/s."""
        ######## TODO - YOUR P2B CONTROLLER ########
        return np.zeros(3)
        ######## END TODO ########

    def refine(self, midpoint, dist):
        ##TODO:
        confidence = 0.0
        if self.gap is None:
            #First ray cast i.e. first estimate of gap orientation, i.e. best estimate. 
            self.gap = midpoint
        else:
            #TODO: Implement some type of refinement (filtering), such as (1-weight)*self.gap + (weight*midpoint) / 2, where weight is a float value 0-1 of how much we trust new data
            

        return confidence


    # -- the state machine -----------------------------------------------------
    def act(self, obs, state) -> Action:
        if self.K is None:
            h, w = obs.rgb.shape[:2]
            self.K = camera_matrix(w, h)

        det = self.find_gap(obs.rgb, self.prev)
        seen = None if det is None else self.locate(det, state)

        v_world, yaw_rate = np.zeros(3), 0.0
        confidence = refine(seen[0], seen[1])
        
        ######## TODO - THE STATE MACHINE ########
        # Keep self.gap on the gap you are flying to: refine it from `seen`
        # while you approach, and stop once you are committed.
        # search    no gap yet: move so the image moves -- the slow sideways
        #           sweep below -- until find_gap and locate agree on one
        # move_to     move to a point in front of the gap, on its axis
        #turn_to
        # approach  fly at the gap, keeping it in the middle of the image
        # through   commit: carry on straight until you are past the panel --
        #           the gap fills and then leaves the image before you get there
        # done      past the panel: hover
        # Change self.mode when a step is done.
        if self.mode == "search":
            #Searching up and down.
            v_world = body_to_world([0.0, 0.15 * math.cos(0.8 * state.t_s), 0.0],
                                    state.yaw_rad)
            if(seen is not None):
                if(confidence > self.confidence_threshold):
                    self.mode=="move_to"
                    #TODO: Choose target point aligned with axis
                    '''
                    target_point = self.gap - (self.target_axis)*self.safety_dist
                    v_world = self.velocity_to(target_point, state)
                    '''
        if self.mode == "move_to":
            v_world = self.velocity_to(target_point, state)
            #We've neared the point enough where the drone has stopped OR the drone's state_position is near the target_point
            if(np.linalg.norm(np.subtract(state.position_m, target_point)) < 1e-3):
                v_world = np.zeros(3)
                self.mode == "turn_to"
                #TODO: set desired yaw
                desired_yaw = None
        if self.mode == "turn_to":
            #TODO: Need to change the yaw so that the drone is facing the correct way, by setting desired state.yaw_rad
            if(abs(state.yaw_rad - desired_yaw) <= 1e-3):
                self.mode == "approach"
        if self.mode == "approach":
            #Approach up to 10 cm, keeping gap in the middle of the image.
            self.target_point = self.gap - (self.target_axis)*.01
            v_world = self.velocity_to(self.target_point, state)
            if(np.linalg.norm(np.subtract(state.position_m, target_point)) < 1e-3):
                v_world = np.zeros(3)
                self.mode == "commit"
                #Choose a target point 1 meter in front of drone, which should be facing the gap, .1 m away from it and fly cleanly through.
                self.target_point = self.state_position + body_to_world([1, 0, 0], state.yaw_rad)

        if self.mode == "commit":
            #Choose a target point 1 meter in front of drone, which should be facing the gap, .1 m away from it and fly cleanly through.
            v_world = velocity_to(self.target_point, state)
            if (np.linalg.norm(np.subtract(state.position_m, target_point)) < 1e-3):
                v_world = np.zeros(3)
                self.mode == "done"
        
        if self.mode == "done":
            print("All done!")
            

        


        ######## END TODO ########

        if self.fpv is not None:
            centre = None if det is None else det.get("centre")
            self.fpv.add(obs.rgb, mask=None if det is None else det.get("mask"),
                         points=None if centre is None else [centre], mode=self.mode)
        self.prev = (obs.rgb, np.array(state.position_m, float), float(state.yaw_rad))
        return Action("velocity", "body_flu", world_to_body(v_world, state.yaw_rad),
                      yaw_rate_rad_s=float(yaw_rate))


class HoverHere(Policy):
    """Zero velocity, camera recorded. Fly it first each session to check the loop."""

    name = "p4_hover"
    sensor_fingerprint = None

    def __init__(self):
        self.fpv = FpvRecorder()

    def act(self, obs, state) -> Action:
        self.fpv.add(obs.rgb)
        return Action("velocity", "body_flu", np.zeros(3))
