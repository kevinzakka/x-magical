"""Entity helper methods."""

# pytype: disable=attribute-error


class EntityIndex:
    def __init__(self, entities):
        """Build a reverse index mapping shapes to entities. Assumes that all
        the shapes which make up an entity are stored as attributes on the
        shape, or are attached to bodies which are stored as attributes on the
        shape.

        Args:
            entities ([Entity]): list of entities to process.

        Returns:
            ent_index (dict): dictionary mapping shapes to entities."""
        self._shape_to_ent = dict()
        self._ent_to_shapes = dict()
        for entity in entities:
            shapes = entity.shapes
            self._ent_to_shapes[entity] = shapes
            for shape in shapes:
                assert shape not in self._shape_to_ent, (
                    f"shape {shape} appears in {entity} and "
                    f"{self._shape_to_ent[shape]}"
                )
                self._shape_to_ent[shape] = entity

    def entity_for(self, shape):
        """Look up the entity associated with a particular shape. Raises
        `KeyError` if no known entity owns the shape."""
        return self._shape_to_ent[shape]

    def shapes_for(self, ent):
        """Return a set of shapes associated with the given entity. Raises
        `KeyError` if the given entity is not in the index."""
        return self._ent_to_shapes[ent]
