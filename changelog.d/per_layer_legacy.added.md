Added :class:`~.PerLayerLegacySolve`, a new analysis stage that generalizes
:class:`~.LegacySolve` to gate sets with multiple unitary layers by independently solving
the noise model for each gate layer and combining the per-layer results into a single
:class:`~.ModelData`.
