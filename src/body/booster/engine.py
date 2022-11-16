"""The booster's engines."""
from Box2D import (
    b2Body,
    b2Filter,
    b2FixtureDef, 
    b2PolygonShape, 
    b2Vec2,
)

from src.utils import Config

from .hull import Hull


class Engines:
    """Engines class.

    Defines booster's engines.
    """
    # Mass of nine Merlin engines
    num_engines = 9
    height = 0.8  # [m]
    width_hull = 3.7  # [m]
    width_min = 0.5  # [m]
    width_max = 1.0  # [m]

    mass = 9 * 470  # [kg]
    density = mass / (width_hull * height)  # [kg / m^2]

    def __init__(self, body: b2Body, hull: Hull, config: Config) -> None:
        """Initializes engines class."""
        self.body = body
        self.hull = hull
        self.friction = config.env.friction
        self._add_engines()

    def _engine_nozzle(self, mount_point: b2Vec2):
        """Adds three engines to booster."""
        r_0 = mount_point + b2Vec2(0.5 * self.width_min, 0.0)
        r_1 = mount_point + b2Vec2(-0.5 * self.width_min, 0.0)
        r_2 = mount_point + b2Vec2(-0.5 * self.width_max, -self.height)
        r_3 = mount_point + b2Vec2(0.5 * self.width_max, -self.height)
        return r_0, r_1, r_2, r_3

    def _add_engines(self) -> None:
        """Adds engines to booster."""

        mount_points = [
            b2Vec2(0.0, -0.5 * self.hull.height),
            b2Vec2(-(1.0 / 3.0) * self.hull.width, -0.5 * self.hull.height),
            b2Vec2((1.0 / 3.0) * self.hull.width, -0.5 * self.hull.height),
        ]

        for mount_point in mount_points:

            engine_polygon = b2PolygonShape(
                vertices=self._engine_nozzle(mount_point=mount_point)
            )

            engine_fixture_def = b2FixtureDef(
                shape=engine_polygon,
                density=self.density,
                friction=self.friction,
                filter=b2Filter(groupIndex=-1),  # negative groups never collide
            )

            _ = self.body.CreateFixture(engine_fixture_def)
