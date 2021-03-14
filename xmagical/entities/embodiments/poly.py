import math

import pymunk as pm
import pymunk.autogeometry as autogeom

from xmagical import geom as gtools
from xmagical import render as r
from xmagical.style import COLORS_RGB, SHAPE_LINE_THICKNESS, darken_rgb

from .base import NonHolonomicEmbodiment

# pytype: disable=attribute-error


class PolygonEmbodiment(NonHolonomicEmbodiment):
    def __init__(self, num_sides: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.num_sides = num_sides
        self.side_len = gtools.regular_poly_circ_rad_to_side_length(
            self.num_sides, math.sqrt(math.pi) * 0.6 * self.radius
        )

        self.robot_color = COLORS_RGB["grey"]
        self.eye_txty = (0.3 * self.radius, 0.0 * self.radius)

    def _setup_body(self):
        self.poly_verts = gtools.compute_regular_poly_verts(
            self.num_sides, self.side_len
        )
        inertia = pm.moment_for_poly(self.mass, self.poly_verts, (0, 0), 0)
        self.body = pm.Body(self.mass, inertia)

    def _setup_shape(self):
        friction = 0.5
        robot_group = 1
        self.shape = pm.Poly(self.body, self.poly_verts)
        self.shape.filter = pm.ShapeFilter(group=robot_group)
        self.shape.friction = friction

    def _setup_graphic(self):
        graphics_body = r.Poly(self.poly_verts, outline=True)
        dark_robot_color = darken_rgb(self.robot_color)
        graphics_body.color = self.robot_color
        graphics_body.outline_color = dark_robot_color
        self.graphic_bodies.append(graphics_body)
