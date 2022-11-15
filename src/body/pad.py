"""Defines landing pad."""
from Box2D import b2EdgeShape
from Box2D.Box2D import b2World, b2Vec2

from src.utils.config import Config


class LandingPad:
    """Class for landing pad.

    Describes landing pad at position (x, y).

    Attributes:
        body: Static body object.
    """

    landing_pad_diameter = 86   # [m]

    def __init__(self, world: b2World, config: Config) -> None:
        """Initializes landing pad."""

        # Center of landing pad, 'GPS' coordinates
        position = config.env.landing_pad.position

        vertices = [
            (-0.5 * self.landing_pad_diameter, 0.0),
            (0.5 * self.landing_pad_diameter, 0.0),
        ]

        self.body = world.CreateStaticBody(
            position=b2Vec2(position.x, position.y),
            shapes=b2EdgeShape(vertices=vertices),
        )
