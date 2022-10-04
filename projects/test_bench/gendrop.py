#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import numpy as np
import pygame

from Box2D.examples.framework import (Framework, main)
from Box2D import (b2CircleShape, b2EdgeShape, b2FixtureDef, b2PolygonShape)
# from simple_framework import SimpleFramework


class Bullet(Framework):
# class Bullet(SimpleFramework):

    name = "Test"
    description = 'Test description'

    def __init__(self):
        super(Bullet, self).__init__()

        # Bullet
        self.init_position = (0.0, 30.0)
        self.init_speed = (0.0, -40.0)
        self.density = 2.0

        self.bullet = self.world.CreateDynamicBody(
            bullet=True,
            position=self.init_position,
            linearVelocity=self.init_speed,
        )

        # vertices_prototype = np.random.random(size=(16, 2))
        vertices_prototype = [(-0.5, -0.5), (-0.5, -0.25),  (-0.5, 0.0),  (-0.5, 0.25),
                              (-0.5, 0.5),  (-0.25, 0.5),  (0.0, 0.5),  (0.25, 0.5),
                              (0.5, 0.5),  (0.5, 0.25),  (0.5, 0.0),  (0.5, -0.25),
                              (0.5, -0.5),  (0.25, -0.5),  (0.0, -0.5),  (-0.25, -0.5)]

        scale = 4.0
        self.vertices = [tuple(scale * num for num in item) for item in vertices_prototype]
        self.fixture_def = b2FixtureDef(shape=b2PolygonShape(vertices=self.vertices), density=self.density)
        self.bullet_fixture = self.bullet.CreateFixture(self.fixture_def)

        # Circles
        self.circle_radius = 0.5
        self.circle_friction = 1.0
        self.circle_density = 1.0
        self.circles = list()
        self.position = list()
        self._generate_bodies()

        # World
        self.world.gravity = (0.0, 0.0)
        boundaries = self.world.CreateStaticBody(
            shapes=[
                b2EdgeShape(vertices=[(-10, 0), (10, 0)]),
                b2EdgeShape(vertices=[(-10, 0), (-10, 20)]),
                b2EdgeShape(vertices=[(10, 0), (10, 20)]),
            ]
        )

        # Optimization
        self.population_size = 4
        self.max_steps = 100
        self.fitness = list()
        self.individuals = list()

        self._generate_population()
        self._mutate()

    def _reset_objects(self):
        self._reset_bullet()
        self._reset_circles()

    def _reset_bullet(self):
        self.bullet.DestroyFixture(self.bullet_fixture)
        self.bullet.CreateFixture(self.fixture_def)
        self.bullet.transform = [self.init_position, 0]
        self.bullet.linearVelocity = self.init_speed
        self.bullet.angularVelocity = 0.0

    def _reset_circles(self):
        for circle, (x, y) in zip(self.circles, self.position):
            circle.transform = [(x, y), 0.0]
            circle.linearVelocity = (0.0, 0.0)
            circle.angularVelocity = 0.0

    def _generate_bodies(self):
        r = self.circle_radius
        cols = 20
        rows = 20

        for j in range(cols):
            for i in range(rows):
                x = -10.0 + r * (2.0 * j + 1.0)
                y = r * (2.0 * i + 1.0)
                self._create_circle((x, y))
                self.position.append((x, y))

    def _create_circle(self, pos):
        fixture = b2FixtureDef(shape=b2CircleShape(radius=self.circle_radius, pos=(0, 0)), 
                               density=self.circle_density, 
                               friction=self.circle_friction)

        self.circles.append(self.world.CreateDynamicBody(position=pos, fixtures=fixture))

    def _generate_population(self):
        for _ in range(self.population_size):
            self.individuals.append(self.vertices)

    def _mutate(self):
        buffer = list()
        for vertices in self.individuals:
            buffer.append([tuple((num + 10*np.random.uniform(-1, 1)) for num in item) for item in vertices])
        self.individuals = buffer

    def _set_fixture_def(self):
        self.fixture_def = b2FixtureDef(shape=b2PolygonShape(vertices=self.vertices), density=self.density)

    def Step(self, settings):
        t0 = time.time()
        super(Bullet, self).Step(settings)
        # print(self.bullet.position)
        # for vertices in self.individuals:
        if (self.stepCount % self.max_steps) == 0:
            print(self.bullet.position)
            self._reset_objects()
            self._mutate()
            print(time.time() - t0)


if __name__ == "__main__":
    main(Bullet)
