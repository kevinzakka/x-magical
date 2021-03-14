import pymunk as pm

from xmagical import render as r
from xmagical.style import COLORS_RGB, darken_rgb

from .base import NonHolonomicEmbodiment

# pytype: disable=attribute-error


class StickEmbodiment(NonHolonomicEmbodiment):
    def __init__(self, width: float, height: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.width = width * self.radius
        self.height = height * self.radius

        self.robot_color = COLORS_RGB["grey"]
        self.eye_txty = (0.3 * self.radius, 0.0 * self.radius)

    def _setup_body(self):
        inertia = pm.moment_for_box(self.mass, (self.width, self.height))
        self.body = pm.Body(self.mass, inertia)

    def _setup_shape(self):
        friction = 0.5
        robot_group = 1
        vertices = r.make_rect(self.width, self.height, True).initial_pts
        self.shape = pm.Poly(self.body, vertices)
        self.shape.filter = pm.ShapeFilter(group=robot_group)
        self.shape.friction = friction

    def _setup_graphic(self):
        graphics_body = r.make_rect(self.width, self.height, True)
        dark_robot_color = darken_rgb(self.robot_color)
        graphics_body.color = self.robot_color
        graphics_body.outline_color = dark_robot_color
        self.graphic_bodies.append(graphics_body)


class ShortstickEmbodiment(StickEmbodiment):
    def __init__(self, *args, **kwargs):
        super().__init__(width=1.5, height=0.5, *args, **kwargs)


class MediumstickEmbodiment(StickEmbodiment):
    def __init__(self, *args, **kwargs):
        super().__init__(width=3.0, height=0.5, *args, **kwargs)


class LongstickEmbodiment(StickEmbodiment):
    def __init__(self, *args, **kwargs):
        super().__init__(width=8.0, height=0.5, *args, **kwargs)
