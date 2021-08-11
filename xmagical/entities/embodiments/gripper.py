import math
from typing import Tuple, Union

import numpy as np
import pymunk as pm

from xmagical import geom as gtools
from xmagical import render as r
from xmagical.style import COLORS_RGB, ROBOT_LINE_THICKNESS, darken_rgb, lighten_rgb

from .base import NonHolonomicEmbodiment

# pytype: disable=attribute-error


def make_finger_vertices(upper_arm_len, forearm_len, thickness, side_sign):
    """Make annoying finger polygons coordinates. Corresponding composite shape
    will have origin at root of upper arm, with upper arm oriented straight
    upwards and forearm above it."""
    up_shift = upper_arm_len / 2
    upper_arm_vertices = gtools.rect_verts(thickness, upper_arm_len)
    forearm_vertices = gtools.rect_verts(thickness, forearm_len)
    # Now rotate upper arm into place & then move it to correct position.
    upper_start = pm.vec2d.Vec2d(side_sign * thickness / 2, upper_arm_len / 2)
    forearm_offset_unrot = pm.vec2d.Vec2d(-side_sign * thickness / 2, forearm_len / 2)
    rot_angle = side_sign * math.pi / 8
    forearm_trans = upper_start + forearm_offset_unrot.rotated(rot_angle)
    forearm_trans.y += up_shift
    forearm_vertices_trans = [
        v.rotated(rot_angle) + forearm_trans for v in forearm_vertices
    ]
    for v in upper_arm_vertices:
        v.y += up_shift
    upper_arm_verts_final = [(v.x, v.y) for v in upper_arm_vertices]
    forearm_verts_final = [(v.x, v.y) for v in forearm_vertices_trans]
    return upper_arm_verts_final, forearm_verts_final


class NonHolonomicGripperEmbodiment(NonHolonomicEmbodiment):
    """A non-holonomic embodiment with fingers that open and close."""

    DOF = 3  # Degrees of freedom.

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Max angle from vertical on inner side and outer.
        self.finger_rot_limit_outer = math.pi / 8
        self.finger_rot_limit_inner = 0.0

        self.finger_thickness = 0.25 * self.radius
        self.finger_upper_length = 1.1 * self.radius
        self.finger_lower_length = 0.7 * self.radius

        self.robot_color = COLORS_RGB["grey"]

        self.eye_txty = (0.4 * self.radius, 0.3 * self.radius)

    def _setup_body(self):
        inertia = pm.moment_for_circle(self.mass, 0, self.radius, (0, 0))
        self.body = pm.Body(self.mass, inertia)

    def _setup_extra_bodies(self):
        super()._setup_extra_bodies()

        # Finger bodies/controls (annoying).
        self.finger_bodies = []
        self.finger_motors = []
        self.finger_vertices = []
        self.finger_inner_vertices = []
        self.target_finger_angle = 0.0
        for finger_side in [-1, 1]:
            # Basic finger body.
            finger_verts = make_finger_vertices(
                upper_arm_len=self.finger_upper_length,
                forearm_len=self.finger_lower_length,
                thickness=self.finger_thickness,
                side_sign=finger_side,
            )
            self.finger_vertices.append(finger_verts)
            finger_inner_verts = make_finger_vertices(
                upper_arm_len=self.finger_upper_length - ROBOT_LINE_THICKNESS * 2,
                forearm_len=self.finger_lower_length - ROBOT_LINE_THICKNESS * 2,
                thickness=self.finger_thickness - ROBOT_LINE_THICKNESS * 2,
                side_sign=finger_side,
            )
            finger_inner_verts = [
                [(x, y + ROBOT_LINE_THICKNESS) for x, y in box]
                for box in finger_inner_verts
            ]
            self.finger_inner_vertices.append(finger_inner_verts)

            # These are movement limits; they are useful below, but also
            # necessary to make initial positioning work.
            if finger_side < 0:
                lower_rot_lim = -self.finger_rot_limit_inner
                upper_rot_lim = self.finger_rot_limit_outer
            if finger_side > 0:
                lower_rot_lim = -self.finger_rot_limit_outer
                upper_rot_lim = self.finger_rot_limit_inner
            finger_mass = self.mass / 8
            finger_inertia = pm.moment_for_poly(finger_mass, sum(finger_verts, []))
            finger_body = pm.Body(finger_mass, finger_inertia)
            if finger_side < 0:
                delta_finger_angle = upper_rot_lim
                finger_body.angle = self.init_angle + delta_finger_angle
            else:
                delta_finger_angle = lower_rot_lim
                finger_body.angle = self.init_angle + delta_finger_angle

            # Position of finger relative to body.
            finger_rel_pos = (
                finger_side * self.radius * 0.45,
                self.radius * 0.1,
            )
            finger_rel_pos_rot = gtools.rotate_vec(finger_rel_pos, self.init_angle)
            finger_body.position = gtools.add_vecs(
                self.body.position, finger_rel_pos_rot
            )
            self.add_to_space(finger_body)
            self.finger_bodies.append(finger_body)

            # Pivot joint to keep it in place (it will rotate around this
            # point).
            finger_piv = pm.PivotJoint(self.body, finger_body, finger_body.position)
            finger_piv.error_bias = 0.0
            self.add_to_space(finger_piv)

            # Rotary limit joint to stop it from getting too far out of line.
            finger_limit = pm.RotaryLimitJoint(
                self.body, finger_body, lower_rot_lim, upper_rot_lim
            )
            finger_limit.error_bias = 0.0
            self.add_to_space(finger_limit)

            # Motor to move the fingers around (very limited in power so as not
            # to conflict with rotary limit joint).
            finger_motor = pm.SimpleMotor(self.body, finger_body, 0.0)
            finger_motor.rate = 0.0
            finger_motor.max_bias = 0.0
            finger_motor.max_force = self.phys_vars.robot_finger_max_force
            self.add_to_space(finger_motor)
            self.finger_motors.append(finger_motor)

    def _setup_shape(self):
        friction = 0.5
        robot_group = 1
        self.shape = pm.Circle(self.body, self.radius, (0, 0))
        self.shape.filter = pm.ShapeFilter(group=robot_group)
        self.shape.friction = friction

    def _setup_extra_shapes(self):
        robot_group = 1
        finger_friction = 5.0  # Grippy fingers.
        self.finger_shapes = []
        for finger_body, finger_verts, finger_side in zip(
            self.finger_bodies, self.finger_vertices, [-1, 1]
        ):
            finger_subshapes = []
            for finger_subverts in finger_verts:
                finger_subshape = pm.Poly(finger_body, finger_subverts)
                finger_subshape.filter = pm.ShapeFilter(group=robot_group)
                finger_subshape.friction = finger_friction
                finger_subshapes.append(finger_subshape)
            self.finger_shapes.append(finger_subshapes)
            self.extra_shapes.extend(finger_subshapes)

    def _setup_graphic(self):
        graphics_body = r.make_circle(self.radius, 100, True)
        dark_robot_color = darken_rgb(self.robot_color)
        graphics_body.color = self.robot_color
        graphics_body.outline_color = dark_robot_color
        self.graphic_bodies.append(graphics_body)

    def _setup_extra_graphics(self):
        super()._setup_extra_graphics()

        light_robot_color = lighten_rgb(self.robot_color, 4)
        self.finger_xforms = []
        finger_outer_geoms = []
        finger_inner_geoms = []
        for finger_outer_subshapes, finger_inner_verts, finger_side in zip(
            self.finger_shapes, self.finger_inner_vertices, [-1, 1]
        ):
            finger_xform = r.Transform()
            self.finger_xforms.append(finger_xform)
            for finger_subshape in finger_outer_subshapes:
                vertices = [(v.x, v.y) for v in finger_subshape.get_vertices()]
                finger_outer_geom = r.Poly(vertices, False)
                finger_outer_geom.color = self.robot_color
                finger_outer_geom.add_transform(finger_xform)
                finger_outer_geoms.append(finger_outer_geom)
            for vertices in finger_inner_verts:
                finger_inner_geom = r.Poly(vertices, False)
                finger_inner_geom.color = light_robot_color
                finger_inner_geom.add_transform(finger_xform)
                finger_inner_geoms.append(finger_inner_geom)
        for geom in finger_outer_geoms:
            self.viewer.add_geom(geom)
        for geom in finger_inner_geoms:
            self.viewer.add_geom(geom)

    def set_action(
        self,
        action: Union[np.ndarray, Tuple[float, float, float]],
    ) -> None:
        assert len(action) == NonHolonomicGripperEmbodiment.DOF

        super().set_action(action[:2])

        self.target_finger_angle = np.clip(
            self.finger_rot_limit_outer - action[2],
            0,
            self.finger_rot_limit_outer,
        )

    def update(self, dt: float) -> None:
        super().update(dt)

        for finger_body, finger_motor, finger_side in zip(
            self.finger_bodies, self.finger_motors, [-1, 1]
        ):
            rel_angle = finger_body.angle - self.body.angle
            # For the left finger, the target angle is measured
            # counterclockwise; for the right, it's measured clockwise
            # (chipmunk is always counterclockwise).
            angle_error = rel_angle + finger_side * self.target_finger_angle
            target_rate = max(-1, min(1, angle_error * 10))
            if abs(target_rate) < 1e-4:
                target_rate = 0.0
            finger_motor.rate = target_rate

    def pre_draw(self) -> None:
        super().pre_draw()

        for finger_xform, finger_body in zip(self.finger_xforms, self.finger_bodies):
            finger_xform.reset(
                translation=finger_body.position, rotation=finger_body.angle
            )

    @property
    def finger_width(self):
        diff_angle = self.finger_bodies[0].angle - self.finger_bodies[1].angle
        diff_angle = np.degrees(diff_angle) / 45  # [0, 1]
        return np.clip(diff_angle, 0.0, 1.0)
