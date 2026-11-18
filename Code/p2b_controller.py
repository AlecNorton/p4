"""The position controller: position error in, velocity command out.

In Project 2a this file held a cascade:

    position PID -> velocity PID -> attitude -> body rates -> motor mixing

On the Crazyflie the firmware does everything after the first stage, so only
that first stage is left: three PIDs (x, y, z), nine gains. The `pid` class is
the one from 2a, with the time step passed in on every call instead of fixed.
"""
import math

import numpy as np


class pid:
    def __init__(self, kp, ki, kd, filter_tau, dt, minVal=-1.0, maxVal=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.filter_tau = filter_tau    # low-pass on the derivative, seconds
        self.dt = dt                    # default time step, seconds
        self.minVal = minVal
        self.maxVal = maxVal
        self.reset()

    def reset(self):
        self.prev_err = None
        self.prev_filter_val = 0.0
        self.prev_integral = 0.0

    def step(self, dsOrErr, current_state=None, dt=None):
        dt = self.dt if dt is None or dt <= 0 else dt

        # Error
        err = dsOrErr if current_state is None else dsOrErr - current_state

        # Derivative of the error, low-pass filtered. No derivative on the
        # very first call: there is no previous error to difference against.
        err_der = 0.0 if self.prev_err is None else (err - self.prev_err) / dt
        alpha = min(1.0, dt / self.filter_tau)
        err_der_filtered = err_der * alpha + self.prev_filter_val * (1 - alpha)

        # Integral, clamped so it cannot wind up
        err_integral = float(np.clip(err * dt + self.prev_integral,
                                     self.minVal, self.maxVal))

        out = self.kp * err + self.kd * err_der_filtered + self.ki * err_integral
        if math.isnan(out) or math.isinf(out):
            raise Exception('PID output is not finite (err=%r)' % (err,))

        self.prev_err = err
        self.prev_filter_val = err_der_filtered
        self.prev_integral = err_integral
        return float(np.clip(out, self.minVal, self.maxVal))


class PositionController:
    """v_cmd = feed-forward velocity + PID(target position - position).

    All vectors are in one world frame (whichever the caller uses; the policy
    uses scene metres). The output is limited to `max_speed` in the horizontal
    plane and `max_climb` vertically.
    """

    ###########################################################################
    # THE NINE GAINS. kp is 1/seconds: a 0.1 m error asks for kp*0.1 m/s.
    #            kp    ki    kd
    GAINS = {'x': (1.5, 0.0, 0.05),
             'y': (1.5, 0.0, 0.05),
             'z': (1.5, 0.0, 0.05)}
    ###########################################################################

    def __init__(self, dt=1.0 / 15.0, max_speed=0.6, max_climb=0.4,
                 filter_tau=0.15, gains=None):
        g = dict(self.GAINS if gains is None else gains)
        self.max_speed = float(max_speed)
        self.max_climb = float(max_climb)
        # Each PID is allowed the full speed; the sum is limited in step().
        self.x_pid = pid(*g['x'], filter_tau, dt, -max_speed, max_speed)
        self.y_pid = pid(*g['y'], filter_tau, dt, -max_speed, max_speed)
        self.z_pid = pid(*g['z'], filter_tau, dt, -max_climb, max_climb)

    def reset(self):
        for c in (self.x_pid, self.y_pid, self.z_pid):
            c.reset()

    def step(self, position, target, velocity_ff=(0.0, 0.0, 0.0), dt=None):
        """One control tick. Returns the world-frame velocity command."""
        p = np.asarray(position, dtype=float)
        w = np.asarray(target, dtype=float)
        ff = np.asarray(velocity_ff, dtype=float)
        v = np.array([ff[0] + self.x_pid.step(w[0], p[0], dt),
                      ff[1] + self.y_pid.step(w[1], p[1], dt),
                      ff[2] + self.z_pid.step(w[2], p[2], dt)])
        # Limit the horizontal SPEED (the norm), not each axis, so the
        # direction of the command is kept.
        h = float(math.hypot(v[0], v[1]))
        if h > self.max_speed:
            v[:2] *= self.max_speed / h
        v[2] = float(np.clip(v[2], -self.max_climb, self.max_climb))
        return v
