"""Environment class.

The Environment class holds the worlds objects that 
can interact with each other. These are the booster
and the landing pad. 

The Environment class also wraps the Framework class 
that calls the physics engine and rendering engines.
"""
import copy
import math
import random
import numpy as np

from Box2D.Box2D import b2Vec2

from src.utils.config import Config
from src.framework import Framework
from src.body.booster import Booster
from src.body.pad import LandingPad


class Environment(Framework):
    """Environment holding all world bodies.

    This class holds the world's bodies as well as methods
    to control the drones.

    Attributes:
        boosters:
    """

    def __init__(self, config: Config) -> None:
        """Initializes environment class."""
        super().__init__(config=config)

        self.world.gravity = b2Vec2(config.env.gravity.x, config.env.gravity.y)

        self.landing_pad = LandingPad(world=self.world, config=config)

        self.num_boosters = config.optimizer.num_boosters
        self.boosters = [
            Booster(world=self.world, config=config) for _ in range(self.num_boosters)
        ]

        # Add reference of boosters to world class for easier rendering handling.
        self.world.boosters = self.boosters

        # Index of current fittest agent
        self.idx_best = 0

    def _is_outside(self, booster: Booster) -> bool:
        """Checks if center of mass of booster is outside domain."""

        # Get domain boundaries
        x_min = self.config.env.domain.x_min
        x_max = self.config.env.domain.x_max
        y_min = self.config.env.domain.y_min
        y_max = self.config.env.domain.y_max

        # Compare booster position to all four domain boundaries
        pos_x, pos_y = booster.body.position

        # Option 1
        if pos_x < x_min:
            return True
        elif pos_y < y_min:
            return True
        elif pos_x > x_max:
            return True
        elif pos_y > y_max:
            return True
        return False

        # Option 2
        # if (pos_x < x_min) or (pos_y < y_min) or (pos_x > x_max) or (pos_y > y_max):
        #     return True
        # return False

        # Option 3
        # if (x_min < pos_x < x_max) and (y_min < pos_y < y_max):
        #     return False
        # return True

    def detect_escape(self) -> None:
        """Detects when booster escapes from the domain.

        Deactivates booster if center of gravity crosses the
        domain boundary.
        """
        for booster in self.boosters:
            if booster.body.active:
                if self._is_outside(booster):
                    booster.body.active = False
                    booster.predictions = np.zeros(shape=(self.num_dims * self.num_engines))


    def reset(self, add_noise: bool = True) -> None:
        """Resets boosters in environment.

        Resets kinematic variables as well as score
        and activity state. If enabled, adds noise to 
        kinematic variables.

        Args:
            use_noise: If true, adds noise to kinematic variables.
        """
        for booster in self.boosters:

            # Kinematic variables
            position = copy.copy(booster.init_position)
            linear_velocity = copy.copy(booster.init_linear_velocity)
            angular_velocity = copy.copy(booster.init_angular_velocity)
            angle = copy.copy(booster.init_angle)

            if add_noise:
                noise = self.config.env.booster.noise

                # Position
                noise_x = random.gauss(mu=0.0, sigma=noise.position.x)
                noise_y = random.gauss(mu=0.0, sigma=noise.position.y)
                position += (noise_x, noise_y)

                # Linear velocity
                noise_x = random.gauss(mu=0.0, sigma=noise.linear_velocity.x)
                noise_y = random.gauss(mu=0.0, sigma=noise.linear_velocity.y)
                linear_velocity += (noise_x, noise_y)

                deg_to_rad = math.pi / 180.0

                # Angular velocity
                noise_angular_velocity = random.gauss(mu=0.0, sigma=noise.angular_velocity)
                angular_velocity += deg_to_rad * noise_angular_velocity

                # Angle
                noise_angle = random.gauss(mu=0.0, sigma=noise.angle)
                angle += deg_to_rad * noise_angle

            booster.body.position = position
            booster.body.linearVelocity = linear_velocity
            booster.body.angularVelocity = angular_velocity
            booster.body.angle = angle

            # Reset fitness score for next generation.
            booster.score = 0.0

            # Reactivate booster after collision in last generation.
            booster.body.active = True

    def detect_impact(self) -> None:
        """Calls impact detection method of each booster."""
        for booster in self.boosters:
            booster.detect_impact()

    def detect_stress(self) -> None:
        """Calls stress detection method of each booster."""
        for booster in self.boosters:
            booster.detect_excess_stress()

    # def detect_escape(self) -> None:
    #     """Calls escape detection method of each booster."""
    #     for booster in self.boosters:
    #         booster.detect_escape()

    def fetch_data(self) -> None:
        """Fetches data for neural network of booster"""
        for booster in self.boosters:
            booster.fetch_data()

    def comp_score(self) -> None:
        """Computes fitness score of each boosters."""
        for booster in self.boosters:
            booster.comp_score()

    def comp_action(self) -> None:
        """Computes next set of actions."""
        for booster in self.boosters:
            booster.comp_action()

    def apply_action(self) -> None:
        """Applies action coming from neural network to all boosters."""
        for booster in self.boosters:
            booster.apply_action()

    def is_active(self) -> bool:
        """Checks if at least one booster is active."""
        for booster in self.boosters:
            if booster.body.active:
                return False
        return True

    def select(self) -> float:
        """Selects best agent for reproduction."""
        scores = [booster.score for booster in self.boosters]
        self.idx_best = np.argmax(scores)
        return scores[self.idx_best]

    def mutate(self) -> None:
        """Mutates network parameters of each booster."""
        # Get network of fittest booster to reproduce.
        model = self.boosters[self.idx_best].model

        # Pass best model to other boosters and mutate their weights.
        for booster in self.boosters:
            booster.mutate(model)