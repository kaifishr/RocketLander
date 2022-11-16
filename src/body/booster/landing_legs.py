"""The booster's landing legs.
"""
from Box2D import (
    b2Body,
    b2Filter, 
    b2FixtureDef, 
    b2PolygonShape, 
    b2Vec2, 
)

from src.utils import Config

from .hull import Hull


class LandingLegs:
    """Landing legs class.

    Adds landing legs to booster.
    """
    # Mass of four landing legs 
    num_legs = 4
    mass_leg = 600

    # Define coordinates of polygon in shape of landing leg
    y_hull_low = 2.0   # [m]
    y_hull_high = 4.0  # [m]
    y_ground = -2.5  # [m]
    x_ground_low = 6.5  # [m]
    x_ground_high = 7.0  # [m]

    # Area of legs
    # NOTE: We assume that the mass of the four landing legs
    # is concentrated in two legs in a two-dimensional world.
    area_0 = 0.5 * x_ground_high * (y_hull_high - y_ground)
    area_1 = 0.5 * x_ground_low * (y_hull_low - y_ground)
    area = 2 * (area_0 - area_1)

    mass = num_legs * mass_leg  # [kg]
    density = mass / area  # [kg / m^2]

    def __init__(self, body: b2Body, hull: Hull, config: Config) -> None:
        """Initializes engines class."""
        self.body = body
        self.hull = hull
        self.friction = config.env.friction
        self._add_legs()

    def _add_legs(self) -> None:
        """Adds landing legs to booster."""

        # Center coordinates
        center = b2Vec2(-0.5 * self.hull.width, -0.5 * self.hull.height)

        # Left landing leg
        r_0 = center + b2Vec2(0.0, self.y_hull_low)
        r_1 = center + b2Vec2(0.0, self.y_hull_high)
        r_2 = center + b2Vec2(-self.x_ground_low, self.y_ground)
        r_3 = center + b2Vec2(-self.x_ground_high, self.y_ground)

        left_leg_polygon = b2PolygonShape(vertices=(r_0, r_1, r_2, r_3))

        left_leg_fixture = b2FixtureDef(
            shape=left_leg_polygon,
            density=self.density,
            friction=self.friction,
            filter=b2Filter(groupIndex=-1),  # no interactions with other boosters
        )

        left_leg_fixture = self.body.CreateFixture(left_leg_fixture)

        # Center coordinates
        center = b2Vec2(0.5 * self.hull.width, -0.5 * self.hull.height)

        # Right landing leg
        r_0 = center + b2Vec2(0.0, self.y_hull_low)
        r_1 = center + b2Vec2(0.0, self.y_hull_high)
        r_2 = center + b2Vec2(self.x_ground_low, self.y_ground)
        r_3 = center + b2Vec2(self.x_ground_high, self.y_ground)

        right_leg_polygon = b2PolygonShape(vertices=(r_0, r_1, r_2, r_3))

        right_leg_fixture = b2FixtureDef(
            shape=right_leg_polygon,
            density=self.density,
            friction=self.friction,
            filter=b2Filter(groupIndex=-1), # no interactions with other boosters
        )

        _ = self.body.CreateFixture(right_leg_fixture)