from os import path
import os
from pathlib import Path

from meshlib import mrmeshpy as mm


def print_mesh_topology(mesh_path: str | Path, b_check = False, *, mesh: mm.Mesh | None = None) -> int:
    """Print vertex, face, edge counts and the Euler characteristic."""
    if mesh is None:
        mesh = mm.loadMesh(str(mesh_path))
    topology = mesh.topology
    vertices = topology.numValidVerts()
    faces = topology.numValidFaces()
    edges = topology.computeNotLoneUndirectedEdges()
    euler_characteristic = vertices + faces - edges

    non_manifold_edges = len(mm.findMultipleEdges(topology))
    mesh_copy = mm.Mesh(mesh)
    duplications = mm.std_vector_MeshBuilder_VertDuplication()
    mm.duplicateMultiHoleVertices(mesh_copy, 1, duplications)
    non_manifold_vertices = len({dup.srcVert.get() for dup in duplications})

    print(
        f"Topology for {Path(mesh_path).name}: "
        f"vertices={vertices}, faces={faces}, edges={edges}, \n"
        f"Euler characteristic (V + F - E)={euler_characteristic}\n"
        f"non-manifold edges={non_manifold_edges}, "
        f"non-manifold vertices={non_manifold_vertices}, "
    )
    if b_check:
        if euler_characteristic != 2:
            print(f"Warning: Euler characteristic is not 2 for {Path(mesh_path).name}")
        if non_manifold_edges > 0:
            print(f"Warning: Non-manifold edges found in {Path(mesh_path).name}")
        if non_manifold_vertices > 0:
            print(f"Warning: Non-manifold vertices found in {Path(mesh_path).name}")
    return euler_characteristic
