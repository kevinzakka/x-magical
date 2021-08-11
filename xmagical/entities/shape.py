"""Pushable shapes in the environment."""

import enum
import math
from typing import Any, List, Tuple

import numpy as np
import pymunk as pm
import pymunk.autogeometry as autogeom

from xmagical import geom as gtools
from xmagical import render as r
from xmagical.style import COLORS_RGB, SHAPE_LINE_THICKNESS, darken_rgb

from .base import Entity

# pytype: disable=attribute-error


class ShapeType(str, enum.Enum):
    TRIANGLE = "triangle"
    SQUARE = "square"
    PENTAGON = "pentagon"
    # hexagon is somewhat hard to distinguish from pentagon, and octagon is
    # very hard to distinguish from circle at low resolutions
    HEXAGON = "hexagon"
    OCTAGON = "octagon"
    CIRCLE = "circle"
    STAR = "star"


class ShapeColor(str, enum.Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    YELLOW = "yellow"


# limited set of types and colors to use for random generation
# (WARNING: not all benchmarks use the two arrays below! Some have used their
# own arrays so that changes to the base SHAPE_TYPES array don't break the
# benchmark's default shape layout.)
SHAPE_TYPES = np.asarray(
    [ShapeType.SQUARE, ShapeType.PENTAGON, ShapeType.STAR, ShapeType.CIRCLE],
    dtype="object",
)
SHAPE_COLORS = np.asarray(
    [ShapeColor.RED, ShapeColor.GREEN, ShapeColor.BLUE, ShapeColor.YELLOW],
    dtype="object",
)
POLY_TO_FACTOR_SIDE_PARAMS = {
    ShapeType.TRIANGLE: (0.8, 3),
    ShapeType.PENTAGON: (1.0, 5),
    ShapeType.HEXAGON: (1.0, 6),
    ShapeType.OCTAGON: (1.0, 8),
}


class Shape(Entity):
    """A shape that can be pushed around."""

    def __init__(
        self,
        shape_type: ShapeType,
        color_name: ShapeColor,
        shape_size: float,
        init_pos: Tuple[float, float],
        init_angle: float,
        mass: float = 0.5,
    ):
        self.shape_type = shape_type
        # This "size" can be interpreted in different ways depending on the
        # shape type, but area of shape should increase quadratically in this
        # number regardless of shape type.
        self.shape_size = shape_size
        self.color_name = color_name
        self.color = COLORS_RGB[self.color_name]
        self.init_pos = init_pos
        self.init_angle = init_angle
        self.mass = mass

        self.shape_body = None

    def reconstruct_signature(self):
        cls = type(self)
        kwargs = dict(
            shape_type=self.shape_type,
            color_name=self.color_name,
            shape_size=self.shape_size,
            init_pos=self.shape_body.position,
            init_angle=self.shape_body.angle,
            mass=self.mass,
        )
        return cls, kwargs

    # ===================================== #
    # Shape construction methods.
    # ===================================== #
    def _make_square(self, side_len: float) -> List[pm.shapes.Poly]:
        # Body.
        self.shape_body = body = pm.Body()
        body.position = self.init_pos
        body.angle = self.init_angle
        self.add_to_space(body)
        # Shape.
        shape = pm.Poly.create_box(
            body,
            (side_len, side_len),
            0.01 * side_len,  # Slightly bevelled corners.
        )
        shape.mass = self.mass
        shapes = [shape]
        del shape
        return shapes

    def _make_circle(self) -> List[pm.shapes.Poly]:
        # Body.
        inertia = pm.moment_for_circle(self.mass, 0, self.shape_size, (0, 0))
        self.shape_body = body = pm.Body(self.mass, inertia)
        body.position = self.init_pos
        body.angle = self.init_angle
        self.add_to_space(body)
        # Shape.
        shape = pm.Circle(body, self.shape_size, (0, 0))
        shapes = [shape]
        del shape
        return shapes

    def _make_star(
        self,
        star_npoints: int,
        star_out_rad: float,
        star_in_rad: float,
    ) -> Tuple[List[pm.shapes.Poly], List[List[pm.Vec2d]]]:
        # Body.
        star_verts = gtools.compute_star_verts(star_npoints, star_out_rad, star_in_rad)
        # Create an exact convex decomposition.
        convex_parts = autogeom.convex_decomposition(star_verts + star_verts[:1], 0)
        star_hull = autogeom.to_convex_hull(star_verts, 1e-5)
        star_inertia = pm.moment_for_poly(self.mass, star_hull, (0, 0), 0)
        self.shape_body = body = pm.Body(self.mass, star_inertia)
        body.position = self.init_pos
        body.angle = self.init_angle
        self.add_to_space(body)
        # Shape.
        shapes = []
        star_group = self.generate_group_id()
        for convex_part in convex_parts:
            shape = pm.Poly(body, convex_part)
            # Avoid self-intersection with a shape filter.
            shape.filter = pm.ShapeFilter(group=star_group)
            shapes.append(shape)
            del shape
        return shapes, convex_parts

    def _make_regular_polygon(
        self,
        num_sides: int,
        side_len: float,
    ) -> Tuple[List[pm.shapes.Poly], List[Tuple[float, float]]]:
        # Body.
        poly_verts = gtools.compute_regular_poly_verts(num_sides, side_len)
        inertia = pm.moment_for_poly(self.mass, poly_verts, (0, 0), 0)
        self.shape_body = body = pm.Body(self.mass, inertia)
        body.position = self.init_pos
        body.angle = self.init_angle
        self.add_to_space(body)
        # Shape.
        shape = pm.Poly(body, poly_verts)
        shapes = [shape]
        del shape
        return shapes, poly_verts

    def setup(self, *args, **kwargs) -> None:
        super().setup(*args, **kwargs)

        # Physics. This joint setup was taken form tank.py in the pymunk
        # examples.
        if self.shape_type == ShapeType.SQUARE:
            side_len = math.sqrt(math.pi) * self.shape_size
            shapes = self._make_square(side_len)
        elif self.shape_type == ShapeType.CIRCLE:
            shapes = self._make_circle()
        elif self.shape_type == ShapeType.STAR:
            star_npoints = 5
            star_out_rad = 1.3 * self.shape_size
            star_in_rad = 0.5 * star_out_rad
            shapes, convex_parts = self._make_star(
                star_npoints, star_out_rad, star_in_rad
            )
        else:
            # These are free-form shapes b/c no helpers exist in Pymunk.
            try:
                factor, num_sides = POLY_TO_FACTOR_SIDE_PARAMS[self.shape_type]
            except KeyError:
                raise NotImplementedError("haven't implemented", self.shape_type)
            side_len = factor * gtools.regular_poly_circ_rad_to_side_length(
                num_sides, self.shape_size
            )
            shapes, poly_verts = self._make_regular_polygon(num_sides, side_len)

        for shape in shapes:
            shape.friction = 0.5
            self.add_to_space(shape)

        trans_joint = pm.PivotJoint(
            self.space.static_body, self.shape_body, (0, 0), (0, 0)
        )
        trans_joint.max_bias = 0
        trans_joint.max_force = self.phys_vars.shape_trans_joint_max_force
        self.add_to_space(trans_joint)
        rot_joint = pm.GearJoint(self.space.static_body, self.shape_body, 0.0, 1.0)
        rot_joint.max_bias = 0
        rot_joint.max_force = self.phys_vars.shape_rot_joint_max_force
        self.add_to_space(rot_joint)

        # Graphics.
        geoms_outer = []
        if self.shape_type == ShapeType.SQUARE:
            geoms = [r.make_square(side_len, outline=True)]
        elif self.shape_type == ShapeType.CIRCLE:
            geoms = [r.make_circle(self.shape_size, 100, True)]
        elif self.shape_type == ShapeType.STAR:
            star_short_verts = gtools.compute_star_verts(
                star_npoints,
                star_out_rad - SHAPE_LINE_THICKNESS,
                star_in_rad - SHAPE_LINE_THICKNESS,
            )
            short_convex_parts = autogeom.convex_decomposition(
                star_short_verts + star_short_verts[:1], 0
            )
            geoms = []
            for part in short_convex_parts:
                geoms.append(r.Poly(part, outline=False))
            geoms_outer = []
            for part in convex_parts:
                geoms_outer.append(r.Poly(part, outline=False))
        elif (
            self.shape_type == ShapeType.OCTAGON
            or self.shape_type == ShapeType.HEXAGON
            or self.shape_type == ShapeType.PENTAGON
            or self.shape_type == ShapeType.TRIANGLE
        ):
            geoms = [r.Poly(poly_verts, outline=True)]
        else:
            raise NotImplementedError("haven't implemented", self.shape_type)

        if self.shape_type == ShapeType.STAR:
            for g in geoms_outer:
                g.color = darken_rgb(self.color)
            for g in geoms:
                g.color = self.color
        else:
            for g in geoms:
                g.color = self.color
                g.outline_color = darken_rgb(self.color)

        self.shape_xform = r.Transform()
        shape_compound = r.Compound(geoms_outer + geoms)
        shape_compound.add_transform(self.shape_xform)
        self.viewer.add_geom(shape_compound)

    def pre_draw(self) -> None:
        self.shape_xform.reset(
            translation=self.shape_body.position, rotation=self.shape_body.angle
        )
