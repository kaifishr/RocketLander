"""Defines landing pad."""
from Box2D import b2EdgeShape
from Box2D.Box2D import b2World, b2Vec2


class LandingPad:
    """Class for landing pad.

    Attributes:
        world:
    """
    pos_x = 0
    pos_y = 0
    landing_pad_diameter = 86

    def __init__(self, world: b2World) -> None:
        """Initializes landing pad."""

        vertices = [(-0.5 * self.landing_pad_diameter, 0), (0.5 * self.landing_pad_diameter, 0)]

        self.body = world.CreateStaticBody(
            position=b2Vec2(self.pos_x, self.pos_y),
            shapes=b2EdgeShape(vertices=vertices),
        )
