"""Read OBJ geometry without splitting vertices at UV or normal seams."""

from collections.abc import Iterable
from pathlib import Path

import numpy as np


def read_obj_geometry(
    lines: Iterable[str], mesh_path: Path, *, triangulate: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    vertices = []
    faces = []
    for line in lines:
        parts = line.split("#", 1)[0].split()
        if not parts:
            continue
        if parts[0] == "v":
            vertices.append([float(value) for value in parts[1:4]])
        elif parts[0] == "f":
            if len(parts) < 4 or (not triangulate and len(parts) != 4):
                raise ValueError(f"Expected triangular OBJ faces in {mesh_path}.")
            indices = [int(corner.split("/")[0]) for corner in parts[1:]]
            if any(index == 0 for index in indices):
                raise ValueError(f"OBJ vertex indices cannot be zero in {mesh_path}.")
            if any(index < 0 and -index > len(vertices) for index in indices):
                raise ValueError(f"OBJ vertex index is out of range in {mesh_path}.")
            polygon = [index - 1 if index > 0 else len(vertices) + index for index in indices]
            faces.extend([polygon[0], polygon[i], polygon[i + 1]] for i in range(1, len(polygon) - 1))
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError(f"Expected finite 3D vertices in {mesh_path}.")
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError(f"Expected valid triangular vertex indices in {mesh_path}.")
    return vertices, faces
