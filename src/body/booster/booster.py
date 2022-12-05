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
    """Booster class

    Inherits from Booster2D class. Contains booster's logic.
    """

    num_engines = 3
    num_dims = 2

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

        # Input data for neural network
        self.data = []

        # Booster's reward (or fitness score)
        self.reward = 0.0
        self.score_old = 0.0

        self.engine_running = True

    def comp_reward(self) -> None:
        """Computes reward for current simulation step.

        Accumulates defined rewards for booster.

        TODO: Install different reward functions depending on optimization method.
        """
        if self.body.active:

            vel = self.body.linearVelocity

            # Reward only if sink rate is negative.
            if vel.y <= 0.0:

                pos_x, pos_y = self.body.position
                pos_pad = self.config.env.landing_pad.position
                eta = 1.0 / 60.0
                pos_y -= 0.5 * self.hull.height - self.legs.y_ground + eta

                distance = ((pos_pad.x - pos_x) ** 2 + (pos_pad.y - pos_y) ** 2) ** 0.5
                velocity = (vel.x**2 + vel.y**2) ** 0.5

                alpha = 0.1
                beta = 0.01

                # Reward proximity to landing pad
                reward_pos = 1.0 / (1.0 + alpha * distance)

                # Reward soft touchdown
                reward_vel = 1.0 # / (1.0 + beta * velocity)

                # Only final reward at end of epoch.
                score = reward_pos * reward_vel
                if score > self.score_old:
                    self.reward = 1.0
                else:
                    self.reward = 0.0
                self.score_old = score

                ###
                # TODO: Only for RL required.
                # Add reward to most resent memory
                self.model.memory[-1][-1] = self.reward
                ###

            else:
                # TODO: Move this to boundary conditions.
                self.body.active = False
                self.predictions.fill(0.0)

    def detect_landing(self) -> None:
        """Detects successful landing of booster.

        Turns off engines after successful landing. Turning off engines while
        staying active, the booster can still get rewards.
        """
        # Define the maximum distance in x and y direction
        # within engine can be turned off (safely).
        dist_x_max = 5.0
        dist_y_max = 1.0

        pos_x, pos_y = self.body.position
        eta = 1.0 / 60.0
        pos_y -= 0.5 * self.hull.height - self.legs.y_ground + eta

        pos_pad = self.config.env.landing_pad.position

        dist_x = abs(pos_pad.x - pos_x)
        dist_y = abs(pos_pad.y - pos_y)

        if (dist_x < dist_x_max) and (dist_y < dist_y_max):
            self.engine_running = False

    def detect_excess_stress(self) -> None:
        """Detects excess stress on booster.

        Deactivates booster if stress limits are being exceeded.
        """
        stress = self.config.env.booster.stress

        max_angle = self._deg_to_rad(stress.max_angle)
        max_angular_velocity = self._deg_to_rad(stress.max_angular_velocity)

        # Angle
        if abs(self.body.transform.angle) > max_angle:
            self.body.active = False
            self.predictions.fill(0.0)
            return
        # Angular velocity
        elif abs(self.body.angularVelocity) > max_angular_velocity:
            self.body.active = False
            self.predictions.fill(0.0)
            return

    def comp_action(self) -> None:
        """Computes next section of actions applied to engines.
        
        Next steps of action are computed by feeding obstacle data coming
        from ray casting to the drone's neural network which then returns
        a set of actions (forces) to be applied to the drone's engines.
        """
        if self.body.active:

            if self.engine_running:

                # Pre-processing
                # data = self._pre_process(self.data)
                data = self.data  # state

                # Raw network predictions
                pred = self.model(data)  # returns the action

                # Data post-processing
                self.predictions = self._post_process(pred)

            else:
                self.predictions.fill(0.0)

    def fetch_state(self):
        """Fetches data (or state) from booster that is fed into neural network."""
        self.data = np.array(  # TODO: Change to state
            (
                self.body.position.x,
                self.body.position.y,
                self.body.linearVelocity.x,
                self.body.linearVelocity.y,
                self.body.angularVelocity,  # TODO: Change with angle.
                self.body.transform.angle,
            )
        )

    def _pre_process(self, data: numpy.ndarray) -> numpy.ndarray:
        """Applies pre-processing to fetched data.

        TODO: Use online method to compute running mean and standard deviation
              for data normalization.

        Normalizes fetched data.

        Args:
            data: Array holding fetched data.

        Returns:
            Array with normalized data.
        """
        pos_x, pos_y = data[0], data[1]
        vel_x, vel_y = data[2], data[3]
        angular_vel = data[4]
        angle = data[5]

        # Position
        pos_x_min, pos_x_max = -200.0, 200.0
        pos_y_min, pos_y_max = -5.0, 300.0
        pos_x = 2.0 * (pos_x - pos_x_min) / (pos_x_max - pos_x_min) - 1.0
        pos_y = 2.0 * (pos_y - pos_y_min) / (pos_y_max - pos_y_min) - 1.0
        data[0], data[1] = pos_x, pos_y

        # Velocity
        vel_x_min, vel_x_max = -50.0, 50.0
        vel_y_min, vel_y_max = 0.0, 100.0
        vel_x = 2.0 * (vel_x - vel_x_min) / (vel_x_max - vel_x_min) - 1.0
        vel_y = 2.0 * (vel_y - vel_y_min) / (vel_y_max - vel_y_min) - 1.0
        data[2], data[3] = vel_x, vel_y

        # Angular velocity
        angular_vel_min, angular_vel_max = -0.5 * math.pi, 0.5 * math.pi  # [pi rad / s]
        angular_vel = (
            2.0 * (angular_vel - angular_vel_min) / (angular_vel_max - angular_vel_min)
            - 1.0
        )
        data[4] = angular_vel

        # Angle
        angle_min, angle_max = -0.5 * math.pi, 0.5 * math.pi  # [pi rad]
        angle = 2.0 * (angle - angle_min) / (angle_max - angle_min) - 1.0
        data[5] = angle

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

    def _print_booster_info(self) -> None:
        """Prints booster data."""
        print(f"{self.body.transform = }")
        print(f"{self.body.position = }")
        print(f"{self.body.angle = }")
        print(f"{self.body.localCenter = }")
        print(f"{self.body.worldCenter = }")
        print(f"{self.body.massData = }")
        print(f"{self.body.linearVelocity = }")
        print(f"{self.body.angularVelocity = }\n")
