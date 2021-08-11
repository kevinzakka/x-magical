import math

import pymunk as pm
import pymunk.autogeometry as autogeom

from xmagical import geom as gtools
from xmagical import render as r
from xmagical.style import COLORS_RGB, SHAPE_LINE_THICKNESS, darken_rgb

from .base import NonHolonomicEmbodiment

# pytype: disable=attribute-error


class StarEmbodiment(NonHolonomicEmbodiment):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.side_len = math.sqrt(math.pi) * 0.6 * self.radius
        self.star_npoints = 5
        self.star_out_rad = 1.3 * self.side_len
        self.star_in_rad = 0.5 * self.star_out_rad

        self.robot_color = COLORS_RGB["grey"]
        self.eye_txty = (0.3 * self.radius, 0.0 * self.radius)

    def _setup_body(self):
        star_verts = gtools.compute_star_verts(
            self.star_npoints, self.star_out_rad, self.star_in_rad
        )
        self.cvx_parts = autogeom.convex_decomposition(star_verts + star_verts[:1], 0)
        star_hull = autogeom.to_convex_hull(star_verts, 1e-5)
        inertia = pm.moment_for_poly(self.mass, star_hull, (0, 0), 0)
        self.body = pm.Body(self.mass, inertia)

    def _setup_shape(self):
        friction = 0.5
        self.shape = []
        star_group = self.generate_group_id()
        for convex_part in self.cvx_parts:
            shape = pm.Poly(self.body, convex_part)
            shape.filter = pm.ShapeFilter(group=star_group)
            shape.friction = friction
            self.shape.append(shape)
            del shape

    def _setup_graphic(self):
        star_short_verts = gtools.compute_star_verts(
            self.star_npoints,
            self.star_out_rad - SHAPE_LINE_THICKNESS,
            self.star_in_rad - SHAPE_LINE_THICKNESS,
        )
        short_convex_parts = autogeom.convex_decomposition(
            star_short_verts + star_short_verts[:1], 0
        )
        geoms = []
        for part in short_convex_parts:
            geoms.append(r.Poly(part, outline=False))
        geoms_outer = []
        for part in self.cvx_parts:
            geoms_outer.append(r.Poly(part, outline=False))
        dark_robot_color = darken_rgb(self.robot_color)
        for g in geoms_outer:
            g.color = dark_robot_color
        for g in geoms:
            g.color = self.robot_color
        self.graphic_bodies.extend([*geoms_outer, *geoms])
