"""Shared source-connectivity classification for 3D clearance and 2D layering."""

import json
from pathlib import Path

import numpy as np


def policy():
    config = json.loads(
        (Path(__file__).resolve().parents[1] / "config/hair_clearance.json").read_text()
    )
    if config["world_rear_axis"] != [0, 1, 0]:
        raise ValueError("This tested POPY policy requires world +Y rear")
    return config


def strand_sides(positions, triangles, lower_z=1.50):
    """Return a constant front(-1)/rear(+1) side for each connected hair strand.

    World +Y is the configured model's rear direction. This policy keeps a
    strand on one side of the outfit while the head turns; it is a collision
    routing decision, not an inference about arbitrary model coordinates.
    """
    positions, triangles = np.asarray(positions), np.asarray(triangles)
    parent = list(range(len(positions)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b, c in triangles:
        a = root(int(a))
        for other in (b, c):
            b = root(int(other))
            if a != b:
                parent[b] = a
    labels = np.array([root(i) for i in range(len(parent))])
    sides = np.ones(len(parent), np.int8)
    for label in np.unique(labels):
        ids = np.flatnonzero(labels == label)
        lower = ids[positions[ids, 2] < lower_z]
        mean_y = positions[lower if len(lower) else ids, 1].mean()
        sides[ids] = 1 if mean_y >= 0 else -1
    return sides
