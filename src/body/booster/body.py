"""The booster's body."""
import math

from Box2D import (
    b2Vec2,
    b2World,
)

from src.utils import Config

from .engine import Engines
from .hull import Hull
from .landing_legs import LandingLegs


class Booster2D:
    """Rocket booster class.

    Rocket booster consists of three main parts:
        - Low density hull
        - Medium density landing legs
        - High density engines

    Look also here to avoid collision of objects:
    https://github.com/pybox2d/pybox2d/blob/master/library/Box2D/examples/collision_filtering.py

    Attributes:
        world:
        config:
        init_position:
        init_linear_velocity:
        init_angular_velocity:
        init_angle:
        random_rotation:    # noise_rotation
        random_translation:
        body:
    """
    num_engines = 3
    num_dims = 2

    def __init__(self, world: b2World, config: Config) -> None:
        """Initializes Booster2D class."""

        self.config = config
        self.world = world

        init = config.env.booster.init

        self.init_position = b2Vec2(
            init.position.x,
            init.position.y,
        )
        self.init_linear_velocity = b2Vec2(
            init.linear_velocity.x,
            init.linear_velocity.y,
        )

        self.init_angular_velocity = self._deg_to_rad(init.angular_velocity)
        self.init_angle = self._deg_to_rad(init.angle)

        self.fixed_rotation = config.env.booster.fixed_rotation

        self.body = self.world.CreateDynamicBody(
            bullet=True,
            allowSleep=False,
            position=self.init_position,
            linearVelocity=self.init_linear_velocity,
            angularVelocity=self.init_angular_velocity,
            angle=self.init_angle,
            fixedRotation=self.fixed_rotation,
        )

        self.hull = Hull(body=self.body, config=config)

        self.engines = Engines(
            body=self.body,
            hull=self.hull,
            config=config,
        )

        self.legs = LandingLegs(
            body=self.body,
            hull=self.hull,
            config=config,
        )

        # Compute contact sphere for impact detection
        eta = 2.0 * (1.0 / 60.0)  # step_size = 0.0167 # ~ 1.0 / 60.0
        a = self.legs.x_ground_high + 0.5 * self.hull.width
        b = 0.5 * self.hull.height - self.legs.y_ground
        self.contact_threshold = (a**2 + b**2) ** 0.5 + eta

    @staticmethod
    def _deg_to_rad(deg: float) -> float:
        """Converts from degrees to radians.

        Args:
            float: Number of degrees.

        Returns:
            Radians.
        """
        rad = deg * math.pi / 180.0
        return rad
