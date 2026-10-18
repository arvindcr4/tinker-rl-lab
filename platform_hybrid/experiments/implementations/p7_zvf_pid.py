"""Toy group-size PID; bounded controller, no remote logging or model claims."""

import math


class ZVF_PIDController:
    def __init__(self, target_zvf=0.5, kp=1.0, ki=0.1, kd=0.05, initial_g=8, min_g=2, max_g=16):
        if not all(math.isfinite(v) for v in (target_zvf, kp, ki, kd, initial_g, min_g, max_g)):
            raise ValueError("Finite controller parameters required")
        if not 0 <= target_zvf <= 1 or not 1 <= min_g <= initial_g <= max_g or min(kp, ki, kd) < 0:
            raise ValueError("Invalid PID target, gains or group bounds")
        self.target, self.kp, self.ki, self.kd = target_zvf, kp, ki, kd
        self.integral = self.prev_error = 0.0
        self.base_g = float(initial_g)
        self.current_g, self.min_g, self.max_g = float(initial_g), min_g, max_g

    def step(self, current_zvf):
        if not math.isfinite(current_zvf) or not 0 <= current_zvf <= 1:
            raise ValueError("ZVF must be in [0, 1]")
        error = current_zvf - self.target
        candidate_integral = self.integral + error
        output = (
            self.kp * error + self.ki * candidate_integral + self.kd * (error - self.prev_error)
        )
        candidate = self.base_g + output
        bounded = max(self.min_g, min(self.max_g, candidate))
        if (
            candidate == bounded
            or (candidate > self.max_g and error < 0)
            or (candidate < self.min_g and error > 0)
        ):
            self.integral = candidate_integral
        self.prev_error, self.current_g = error, bounded
        return max(self.min_g, min(self.max_g, math.ceil(bounded)))
