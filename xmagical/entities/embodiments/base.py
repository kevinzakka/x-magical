import abc
from typing import Tuple, Union

import numpy as np
import pymunk as pm

from xmagical import render as r
from xmagical.entities.base import Entity

# pytype: disable=attribute-error


class Embodiment(Entity, abc.ABC):
    """Base abstraction for robotic embodiments."""

    def __init__(
        self,
        radius: float,
        init_pos,
        init_angle: float,
        mass: float = 1.0,
    ) -> None:
        self.radius = radius
        self.init_pos = init_pos
        self.init_angle = init_angle
        self.mass = mass

        # These need to be constructed in the _setup methods below.
        self.control_body = None
        self.body = None
        self.shape = None
        self.graphic_bodies = []
        self.xform = None

        # These do not necessarily need to be constructed.
        self.extra_bodies = []
        self.extra_shapes = []
        self.extra_graphic_bodies = []

    @abc.abstractclassmethod
    def _setup_body(self):
        pass

    def _setup_extra_bodies(self):
        pass

    @abc.abstractclassmethod
    def _setup_control_body(self):
        pass

    @abc.abstractclassmethod
    def _setup_shape(self):
        pass

    def _setup_extra_shapes(self):
        pass

    @abc.abstractclassmethod
    def _setup_graphic(self):
        pass

    def _setup_extra_graphics(self):
        pass

    @abc.abstractclassmethod
    def set_action(self, move_action: np.ndarray) -> None:
        pass

    @abc.abstractclassmethod
    def pre_draw(self) -> None:
        pass

    @abc.abstractclassmethod
    def update(self, dt: float) -> None:
        pass

    def setup(self, *args, **kwargs):
        super().setup(*args, **kwargs)

        self._setup_body()
        assert self.body is not None
        self.body.position = self.init_pos
        self.body.angle = self.init_angle
        self.add_to_space(self.body)

        # Main body position and angle should be set before calling setup on the
        # extra bodies since they might depend on those values.
        self._setup_extra_bodies()
        self.add_to_space(*self.extra_bodies)

        self._setup_control_body()
        assert self.control_body is not None

        self._setup_shape()
        self._setup_extra_shapes()
        assert self.shape is not None
        self.add_to_space(self.shape)
        self.add_to_space(self.extra_shapes)

        self._setup_graphic()
        self._setup_extra_graphics()
        assert self.graphic_bodies

        self.xform = r.Transform()
        self.robot_compound = r.Compound(
            [*self.graphic_bodies, *self.extra_graphic_bodies]
        )
        self.robot_compound.add_transform(self.xform)
        self.viewer.add_geom(self.robot_compound)


class NonHolonomicEmbodiment(Embodiment):
    """A embodiment with 2 degrees of freedom: velocity and turning angle."""

    DOF = 2  # Degrees of freedom.

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.rel_turn_angle = 0.0
        self.target_speed = 0.0
        self._speed_limit = 4.0 * self.radius
        self._angle_limit = 1.5

        self.eye_txty = (0.4 * self.radius, 0.0 * self.radius)

    def reconstruct_signature(self):
        kwargs = dict(
            radius=self.radius,
            init_pos=self.body.position,
            init_angle=self.body.angle,
            mass=self.mass,
        )
        return type(self), kwargs

    def _setup_control_body(self):
        self.control_body = control_body = pm.Body(body_type=pm.Body.KINEMATIC)
        control_body.position = self.init_pos
        control_body.angle = self.init_angle
        self.add_to_space(control_body)
        pos_control_joint = pm.PivotJoint(control_body, self.body, (0, 0), (0, 0))
        pos_control_joint.max_bias = 0
        pos_control_joint.max_force = self.phys_vars.robot_pos_joint_max_force
        self.add_to_space(pos_control_joint)
        rot_control_joint = pm.GearJoint(control_body, self.body, 0.0, 1.0)
        rot_control_joint.error_bias = 0.0
        rot_control_joint.max_bias = 2.5
        rot_control_joint.max_force = self.phys_vars.robot_rot_joint_max_force
        self.add_to_space(rot_control_joint)

    def _setup_extra_bodies(self):
        # Googly eye control bodies & joints.
        self.pupil_bodies = []
        for _ in range(2):
            eye_mass = self.mass / 10
            eye_radius = self.radius
            eye_inertia = pm.moment_for_circle(eye_mass, 0, eye_radius, (0, 0))
            eye_body = pm.Body(eye_mass, eye_inertia)
            eye_body.angle = self.init_angle
            eye_joint = pm.DampedRotarySpring(self.body, eye_body, 0, 0.1, 3e-3)
            eye_joint.max_bias = 3.0
            eye_joint.max_force = 0.001
            self.pupil_bodies.append(eye_body)
            self.add_to_space(eye_joint)
        self.extra_bodies.extend(self.pupil_bodies)

    def _setup_extra_graphics(self):
        self.eye_shapes = []
        self.pupil_transforms = []
        for x_sign in [-1, 1]:
            eye = r.make_circle(0.2 * self.radius, 100, outline=False)
            eye.color = (1.0, 1.0, 1.0)  # White color.
            eye_base_transform = r.Transform(
                translation=(x_sign * self.eye_txty[0], self.eye_txty[1])
            )
            eye.add_transform(eye_base_transform)
            pupil = r.make_circle(0.12 * self.radius, 100, outline=False)
            pupil.color = (0.1, 0.1, 0.1)  # Black color.
            pupil_transform = r.Transform()
            pupil.add_transform(r.Transform(translation=(0, self.radius * 0.07)))
            pupil.add_transform(pupil_transform)
            pupil.add_transform(eye_base_transform)
            self.pupil_transforms.append(pupil_transform)
            self.eye_shapes.extend([eye, pupil])
        self.extra_graphic_bodies.extend(self.eye_shapes)

    def set_action(
        self,
        action: Union[np.ndarray, Tuple[float, float]],
    ) -> None:
        assert len(action) == NonHolonomicEmbodiment.DOF
        self.target_speed = np.clip(action[0], -self._speed_limit, self._speed_limit)
        self.rel_turn_angle = np.clip(action[1], -self._angle_limit, self._angle_limit)

    def update(self, dt: float) -> None:
        del dt
        self.control_body.angle = self.body.angle + self.rel_turn_angle
        x_vel_vector = pm.vec2d.Vec2d(0.0, self.target_speed)
        vel_vector = self.body.rotation_vector.cpvrotate(x_vel_vector)
        self.control_body.velocity = vel_vector

    def pre_draw(self) -> None:
        self.xform.reset(translation=self.body.position, rotation=self.body.angle)
        for pupil_xform, pupil_body in zip(self.pupil_transforms, self.pupil_bodies):
            pupil_xform.reset(rotation=pupil_body.angle - self.body.angle)
