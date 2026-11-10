from pathlib import Path

from meshlib import mrmeshpy as mm


def print_mesh_topology(mesh_path: str | Path) -> int:
    """Print vertex, face, edge counts and the Euler characteristic."""
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
    return euler_characteristic