"""The booster's hull."""
from Box2D import (
    b2Body,
    b2Filter,
    b2FixtureDef,
    b2PolygonShape,
)

from src.utils import Config


class Hull:
    """Booster hull."""

    # Mass of hull is the booster's dry mass (~25600 kg),
    # plus the mass for fuel (~3000 kg),
    # minus the mass of nine Merlin engines (~470 kg / engine).

    dry_mass = 25600  # [kg]
    fuel = 3000  # [kg]
    num_engines = 9  # [kg]
    mass_engine = 470  # [kg]
    height = 46.0  # [m], Approx. hull's length minus length of Merlin engines.
    width = 3.7  # [m]

    mass = dry_mass + fuel - num_engines * mass_engine  # [kg]
    density = mass / (height * width)

    vertices = [(0.5, 0.5), (-0.5, 0.5), (-0.5, -0.5), (0.5, -0.5)]

    def __init__(self, body: b2Body, config: Config) -> None:
        """Initializes hull class."""
        self.friction = config.env.friction
        self._add_hull(body=body)

    def _add_hull(self, body) -> None:
        """Adds hull to body."""
        self.vertices = [(self.width * x, self.height * y) for (x, y) in self.vertices]
        hull_fixture_def = b2FixtureDef(
            shape=b2PolygonShape(vertices=self.vertices),
            density=self.density,
            friction=self.friction,
            filter=b2Filter(groupIndex=-1),  # negative groups never collide
        )
        _ = body.CreateFixture(hull_fixture_def)
