"""Environment class.

The Environment class holds the worlds objects that 
can interact with each other. These are the booster
and the landing pad. 

The Environment class also wraps the Framework class 
that calls the physics engine and rendering engines.
"""
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
        landingpad: Domain walls.
        boosters:
    """

    def __init__(self, config: Config) -> None:
        """Initializes environment class."""
        super().__init__(config=config)

        self.world.gravity = b2Vec2(config.env.gravity.x, config.env.gravity.y)

        LandingPad(self.world)

        self.num_boosters = config.optimizer.num_boosters
        self.boosters = [
            Booster(world=self.world, config=config) for _ in range(self.num_boosters)
        ]

        # Add reference of boosters to world class for easier rendering handling.
        self.world.boosters = self.boosters

        # Index of current fittest agent
        self.idx_best = 0

    def reset(self) -> None:
        """Resets boosters' position and velocity.

        Uses the boosters' initial position and velocity
        and adds some noise.
        """
        for booster in self.boosters:

            booster.body.position = booster.init_position
            booster.body.linearVelocity = booster.init_linear_velocity
            booster.body.angularVelocity = booster.init_angular_velocity
            booster.body.angle = booster.init_angle

            # Reset fitness score for next generation.
            booster.score = 0.0

            # Reactivate booster after collision in last generation.
            booster.body.active = True

    def impact_detection(self) -> None:
        """Calls impact detection method of each booster."""
        for booster in self.boosters:
            booster.detect_impact()

    def detect_escape(self) -> None:
        """Calls escape detection method of each booster."""
        for booster in self.boosters:
            booster.detect_escape()

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
