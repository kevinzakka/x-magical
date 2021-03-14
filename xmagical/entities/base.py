import abc
import weakref

import pymunk as pm

# pytype: disable=attribute-error


class Entity(abc.ABC):
    """Any logical thing that can be displayed and/or interact via physics."""

    @abc.abstractmethod
    def setup(self, viewer, space, phys_vars):
        """Set up entity graphics/physics and a pm.Space. Only called once."""
        self.shapes = []
        self.bodies = []
        self.viewer = weakref.proxy(viewer)
        self.space = weakref.proxy(space)
        self.phys_vars = weakref.proxy(phys_vars)

    def update(self, dt):
        """Do a logic/physics update at some fixed time interval."""
        pass

    def pre_draw(self):
        """Do a graphics state update.

        Used to, e.g., update state of internal `Geom`s. This doesn't have to be
        done at every physics time step.
        """
        pass

    def reconstruct_signature(self):
        """Produce signature necessary to reconstruct this entity in its
        current pose. This is useful for creating new scenarios out of existing
        world states.

        Returns: tuple of (cls, kwargs), where:
            cls (type): the class that should be used to construct the
                instance.
            kwargs (dict): keyword arguments that should be passed to the
                constructor for cls."""
        raise NotImplementedError(
            f"no .reconstruct_signature() implementation for object "
            f"'{self}' of type '{type(self)}'"
        )

    def generate_group_id(self):
        """Generate a new, unique group ID. Intended to be called from
        `.setup()` when creating `ShapeFilter`s."""
        if not hasattr(self.space, "_group_ctr"):
            self.space._group_ctr = 999
        self.space._group_ctr += 1
        return self.space._group_ctr

    @staticmethod
    def format_reconstruct_signature(cls, kwargs):
        """String-format a reconstruction signature. Makes things as
        easy to cut-and-paste as possible."""
        prefix = "    "
        kwargs_sig_parts = []
        for k, v in sorted(kwargs.items()):
            v_str = str(v)
            if isinstance(v, pm.vec2d.Vec2d):
                v_str = "(%.5g, %.5g)" % (v.x, v.y)
            elif isinstance(v, float):
                v_str = "%.5g" % v
            part = f"{prefix}{k}={v_str}"
            kwargs_sig_parts.append(part)
        kwargs_sig = ",\n".join(kwargs_sig_parts)
        result = f"{cls.__name__}(\n{kwargs_sig})"
        return result

    def add_to_space(self, *objects):
        """For adding a body or shape to the Pymunk 'space'. Keeps track of
        shapes/bodies so they can be used later. Should be called instead of
        space.add()."""

        def _add(obj):
            self.space.add(obj)
            if isinstance(obj, pm.Body):
                self.bodies.append(obj)
            elif isinstance(obj, pm.Shape):
                self.shapes.append(obj)
            elif isinstance(obj, pm.Constraint):
                pass
            else:
                raise TypeError(
                    f"don't know how to handle object '{obj}' of type "
                    f"'{type(obj)}' in class '{type(self)}'"
                )

        for obj in objects:
            if not obj:
                continue
            if isinstance(obj, list):
                for o in obj:
                    _add(o)
            else:
                _add(obj)
