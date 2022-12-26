"""Defines rocket booster in Box2d."""
import math

import numpy
import numpy as np

from Box2D import b2Vec2
from Box2D.Box2D import b2World

from src.utils import Config

from .body import Booster2D
from .model import ModelLoader


class Booster(Booster2D):
    """Booster class.

    Inherits from Booster2D class. Contains booster's logic.
    """

    def __init__(self, world: b2World, config: Config) -> None:
        """Initializes Booster class."""
        super().__init__(world=world, config=config)

        # Install the booster's brain.
        self.model = ModelLoader(config=config)()

        # Forces predicted by neural network.
        self.predictions = np.zeros(shape=(self.num_dims * self.num_engines))
        self.max_force_main = config.env.booster.engine.main.max_force
        self.max_angle_main = config.env.booster.engine.main.max_angle
        self.max_force_cold_gas = config.env.booster.engine.cold_gas.max_force
        self.max_angle_cold_gas = config.env.booster.engine.cold_gas.max_angle

        # Observed state, also input data for neural network.
        self.state = None

        # Booster's reward (or fitness score)
        self.rewards = []
        self.distance_x_old = float("inf")
        self.distance_y_old = float("inf")

    def comp_reward(self) -> None:
        """Computes reward for current simulation step.

        Accumulates defined rewards for booster.

        """
        if self.body.active:

            # alpha = 0.6
            # beta = 0.001

            # Distance to landing pad.
            pos_x, pos_y = self.body.position
            pos_y -= 0.5 * self.hull.height - self.legs.y_ground + self.eta
            pos_pad = self.config.env.landing_pad.position
            distance_x = (pos_pad.x - pos_x) ** 2
            distance_y = (pos_pad.y - pos_y) ** 2
            distance = (distance_x + distance_y) ** 0.5

            # Velocity.
            # vel = self.body.linearVelocity
            # velocity = (vel.x**2 + vel.y**2) ** 0.5

            reward = 0.0

            # Reward agent if distance to landing pad gets smaller.
            if distance_x <= self.distance_x_old:
                self.distance_x_old = distance_x
                if distance_y <= self.distance_y_old:
                    self.distance_y_old = distance_y
                    # Soft XNOR coupling:
                    # r_distance = 1.0 / (1.0 + alpha * distance)
                    # r_velocity = 1.0 / (1.0 + beta * velocity)
                    # reward += r_distance * r_velocity + (1.0 - r_distance) * (1.0 - r_velocity)
                    # reward += 1.0
                    # Simple:
                    reward += 100.0 / (1.0 + distance)
            else:
                reward -= 0.01

            if self._detected_escape():
                reward -= 10.0

            if self._detected_stress():
                reward -= 10.0

            if self._detected_impact():
                reward -= 10.0

            if self._detected_landing():
                reward += 10.0

            self.rewards.append(reward)

    def _is_outside_domain(self) -> bool:
        """Checks if center of mass of booster is outside domain."""

        # Get domain boundaries
        x_min = self.config.env.domain.x_min
        x_max = self.config.env.domain.x_max
        y_min = self.config.env.domain.y_min
        y_max = self.config.env.domain.y_max

        # Compare booster position to all four domain boundaries
        pos_x, pos_y = self.body.position

        if pos_x < x_min:
            return True
        elif pos_y < y_min:
            return True
        elif pos_x > x_max:
            return True
        elif pos_y > y_max:
            return True

        return False

    def _detected_escape(self) -> bool:
        """Detects when booster escapes from the domain.

        Deactivates booster if center of gravity crosses the
        domain boundary.
        """
        if self.body.active:
            if self._is_outside_domain():
                self.body.active = False
                self.predictions.fill(0.0)
                return True

        return False

    def _detected_impact(self) -> bool:
        """Detects impact with landing pad.

        Deactivates booster in case of impact.

        An impact has occurred when a booster is one
        length unit above the ground at a higher than
        defined velocity.
        """
        if self.body.active:
            distance_threshold = 2.0
            pad = self.config.env.landing_pad.position
            pos_y = self.body.position.y
            pos_y -= 0.5 * self.hull.height - self.legs.y_ground + self.eta
            if (pos_y - pad.y) < distance_threshold:
                vel_x, vel_y = self.body.linearVelocity
                v_max_x = self.config.env.landing.v_max.x
                v_max_y = self.config.env.landing.v_max.y
                if (vel_y < v_max_y) or (abs(vel_x) > v_max_x):
                    self.body.active = False
                    self.predictions.fill(0.0)
                    return True

        return False

    def _detected_stress(self) -> bool:
        """Detects high stress on vehicle.

        Deactivates booster if stress limits are being exceeded.
        """
        if self.body.active:
            stress = self.config.env.booster.stress

            max_angle = self._deg_to_rad(stress.max_angle)
            max_angular_velocity = self._deg_to_rad(stress.max_angular_velocity)

            if abs(self.body.transform.angle) > max_angle:
                self.body.active = False
                self.predictions.fill(0.0)
                return True
            elif abs(self.body.angularVelocity) > max_angular_velocity:
                self.body.active = False
                self.predictions.fill(0.0)
                return True

        return False

    def _detected_landing(self) -> bool:
        """Detects successful landing of booster.

        Deactivates booster after successful landing.
        A successful landing is defined by a maximal
        distance within the engines can be turned off.
        """
        if self.body.active:
            # A landing is successful if booster is
            # within 20 meters or less from the center.
            dist_x_max = 20.0

            # Turn of engines if booster is less
            # than 0.5 meters above the ground.
            dist_y_max = 0.5

            pos_x, pos_y = self.body.position
            pos_y -= 0.5 * self.hull.height - self.legs.y_ground + self.eta

            pos_pad = self.config.env.landing_pad.position

            dist_x = abs(pos_pad.x - pos_x)
            dist_y = abs(pos_pad.y - pos_y)

            if (dist_x < dist_x_max) and (dist_y < dist_y_max):
                self.body.active = False
                self.predictions.fill(0.0)
                return True

        return False

    def comp_action(self) -> None:
        """Computes actions from observed sate."""
        if self.body.active:
            # Pre-processing.
            # state = self._pre_process(self.state)
            state = self.state

            # Predict actions.
            pred = self.model.predict(state)

            # Data post-processing
            self.predictions = self._post_process(pred)

    def fetch_state(self):
        """Fetches state from booster that is fed into neural network."""
        if self.body.active:
            self.state = np.array(
                (
                    self.body.position.x,
                    self.body.position.y,
                    self.body.linearVelocity.x,
                    self.body.linearVelocity.y,
                    self.body.transform.angle,
                    self.body.angularVelocity,
                )
            )

    def _pre_process(self, data: numpy.ndarray) -> numpy.ndarray:
        """Applies pre-processing to fetched data.

        Normalizes fetched data.

        Args:
            data: Array holding fetched data.

        Returns:
            Array with normalized data.
        """
        pos_x, pos_y = data[0], data[1]
        vel_x, vel_y = data[2], data[3]
        angle = data[4]
        angular_vel = data[5]

        # Position
        pos_x_min, pos_x_max = -100.0, 100.0
        pos_y_min, pos_y_max = -5.0, 500.0
        pos_x = 2.0 * (pos_x - pos_x_min) / (pos_x_max - pos_x_min) - 1.0
        pos_y = 2.0 * (pos_y - pos_y_min) / (pos_y_max - pos_y_min) - 1.0
        data[0], data[1] = pos_x, pos_y

        # Velocity
        vel_x_min, vel_x_max = -100.0, 100.0
        vel_y_min, vel_y_max = -100.0, 100.0
        vel_x = 2.0 * (vel_x - vel_x_min) / (vel_x_max - vel_x_min) - 1.0
        vel_y = 2.0 * (vel_y - vel_y_min) / (vel_y_max - vel_y_min) - 1.0
        data[2], data[3] = vel_x, vel_y

        # Angle
        angle_min, angle_max = -0.5 * math.pi, 0.5 * math.pi  # [pi rad]
        angle = 2.0 * (angle - angle_min) / (angle_max - angle_min) - 1.0
        data[4] = angle

        # Angular velocity
        angular_vel_min, angular_vel_max = -0.5 * math.pi, 0.5 * math.pi  # [pi rad / s]
        angular_vel = (
            2.0 * (angular_vel - angular_vel_min) / (angular_vel_max - angular_vel_min)
            - 1.0
        )
        data[5] = angular_vel

        return data

    def _post_process(self, pred: numpy.ndarray) -> tuple:
        """Applies post-processing to raw network output.

        Network predicts for each engine level of thrust and angle. Predictions are
        between [0, 1]. From these predictions as well as from the maximum power of
        the engines and gimbal angle, the force components f_x and f_y are computed
        for each engine. Thus,

        pred = [
            p_main,
            phi_main,
            p_left,
            phi_left,
            p_right,
            phi_right
        ]

        is transformed to

        predictions = [
            f_x_main,
            f_y_main,
            f_x_left,
            f_y_left,
            f_x_right,
            f_y_right
        ]

        Args:
            pred: Raw network predictions.

        Returns:
            Array of force components f_x and f_y for each engine.
        """
        p_main, phi_main, p_left, phi_left, p_right, phi_right = pred

        # Main engine
        phi = 2.0 * phi_main - 1.0  # [-1, 1]
        arg = phi * self.max_angle_main * math.pi / 180.0
        force = p_main * self.max_force_main
        f_main_x = math.sin(arg) * force
        f_main_y = math.cos(arg) * force

        # Left engine
        phi = 2.0 * phi_left - 1.0  # [-1, 1]
        arg = phi * self.max_angle_cold_gas * math.pi / 180.0
        force = p_left * self.max_force_cold_gas
        f_left_x = math.cos(arg) * force
        f_left_y = math.sin(arg) * force

        # Right engine
        phi = 2.0 * phi_right - 1.0  # [-1, 1]
        arg = phi * self.max_angle_cold_gas * math.pi / 180.0
        force = p_right * self.max_force_cold_gas
        f_right_x = math.cos(arg) * force
        f_right_y = math.sin(arg) * force

        return np.array((f_main_x, f_main_y, f_left_x, f_left_y, f_right_x, f_right_y))

    def apply_action(self) -> None:
        """Applies force to engines predicted by neural network."""
        if self.body.active:

            (
                f_main_x,
                f_main_y,
                f_left_x,
                f_left_y,
                f_right_x,
                f_right_y,
            ) = self.predictions

            # Main engine x-component
            f = self.body.GetWorldVector(localVector=b2Vec2(f_main_x, f_main_y))
            p = self.body.GetWorldPoint(
                localPoint=b2Vec2(0.0, -(0.5 * self.hull.height + self.engines.height))
            )
            self.body.ApplyForce(f, p, True)

            # Left cold gas thruster
            f = self.body.GetWorldVector(localVector=b2Vec2(f_left_x, f_left_y))
            p = self.body.GetWorldPoint(
                localPoint=b2Vec2(-0.5 * self.hull.width, 0.5 * self.hull.height)
            )
            self.body.ApplyForce(f, p, True)

            # Right cold gas thruster
            f = self.body.GetWorldVector(localVector=b2Vec2(-f_right_x, f_right_y))
            p = self.body.GetWorldPoint(
                localPoint=b2Vec2(0.5 * self.hull.width, 0.5 * self.hull.height)
            )
            self.body.ApplyForce(f, p, True)

    def __repr__(self) -> str:
        """Prints booster data."""
        info = (
            f"{self.body.transform = }\n"
            f"{self.body.position = }\n"
            f"{self.body.angle = }\n"
            f"{self.body.localCenter = }\n"
            f"{self.body.worldCenter = }\n"
            f"{self.body.massData = }\n"
            f"{self.body.linearVelocity = }\n"
            f"{self.body.angularVelocity = }\n"
        )
        return info
