import pymunk as pm

from xmagical import render as r
from xmagical.style import COLORS_RGB, lighten_rgb

from .base import Entity

# pytype: disable=attribute-error


class GoalRegion(Entity):
    """A goal region that the robot should push certain shapes into.

    It's up to the caller to figure out exactly which shapes & call methods for
    collision checking/scoring.
    """

    def __init__(self, x, y, h, w, color_name, dashed=True):
        self.x = x
        self.y = y
        assert h > 0, w > 0
        self.h = h
        self.w = w
        self.color_name = color_name
        self.base_color = COLORS_RGB[color_name]
        self.dashed = dashed

        self.goal_body = None

    def reconstruct_signature(self):
        kwargs = dict(
            x=self.goal_body.position[0] - self.w / 2,
            y=self.goal_body.position[1] + self.h / 2,
            h=self.h,
            w=self.w,
            color_name=self.color_name,
        )
        return type(self), kwargs

    def setup(self, *args, **kwargs):
        super().setup(*args, **kwargs)

        # the space only needs a sensor body
        self.goal_body = pm.Body(body_type=pm.Body.STATIC)
        self.goal_shape = pm.Poly.create_box(self.goal_body, (self.w, self.h))
        self.goal_shape.sensor = True
        self.goal_body.position = (self.x + self.w / 2, self.y - self.h / 2)
        self.add_to_space(self.goal_body, self.goal_shape)

        # Graphics.
        outer_color = self.base_color
        inner_color = lighten_rgb(self.base_color, times=2)
        inner_rect = r.make_rect(self.w, self.h, True, dashed=self.dashed)
        inner_rect.color = inner_color
        inner_rect.outline_color = outer_color
        self.goal_xform = r.Transform()
        inner_rect.add_transform(self.goal_xform)
        self.viewer.add_geom(inner_rect)

    def get_overlapping_ents(self, ent_index, contained=False, com_overlap=False):
        """Get all entities overlapping this region.

        Args:
            ent_index (EntityIndex): index of entities to query over.
            contained (bool): set this to True to only return entities that are
                fully contained in the regions. Otherwise, if this is False,
                all entities that overlap the region at all will be returned.

        Returns:
            ents ([Entity]): list of entities intersecting the current one."""

        # first look up all overlapping shapes
        shape_results = self.space.shape_query(self.goal_shape)
        overlap_shapes = {r.shape for r in shape_results}

        # if necessary, do total containment check on shapes
        if contained:
            # This does a containment check based *only* on axis-aligned
            # bounding boxes. This is valid if our goal region is an
            # axis-aligned bounding box, but could lead to false positives if
            # the goal region were a different shape, or if it was rotated.
            goal_bb = self.goal_shape.bb
            overlap_shapes = {s for s in overlap_shapes if goal_bb.contains(s.bb)}
        if com_overlap:
            goal_bb = self.goal_shape.bb
            overlap_shapes = {
                s for s in overlap_shapes if goal_bb.contains_vect(s.body.position)
            }

        # now look up all indexed entities that own at least one overlapping
        # shape
        relevant_ents = set()
        for shape in overlap_shapes:
            try:
                ent = ent_index.entity_for(shape)
            except KeyError:
                # shape not in index
                continue
            relevant_ents.add(ent)

        # if necessary, filter the entities so that only those with *all*
        # shapes within the region (or with COMs of all bodies in the region)
        # are included
        if contained or com_overlap:
            new_relevant_ents = set()
            for relevant_ent in relevant_ents:
                shapes = set(ent_index.shapes_for(relevant_ent))
                if shapes <= overlap_shapes:
                    new_relevant_ents.add(relevant_ent)
            relevant_ents = new_relevant_ents

        return relevant_ents

    def pre_draw(self):
        self.goal_xform.reset(
            translation=self.goal_body.position, rotation=self.goal_body.angle
        )
