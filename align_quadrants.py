"""
Align two meshes by aligning patches, then smoothing the seams.
To create the patches:
- The first patch is the top cap. It is defined by a ring of vertices around the shoulder of the top of the sphere.
 it has a center point. The center points of the two meshes should be aligned. The top seam is parameterized by
 edge lengths around the seam.
 - the second patch is the bottom cap. It is defined by a ring of vertices around the shoulder of the bottom of the sphere. It also has a center point, and the bottom seam is parameterized by edge lengths around the seam.
. It also has a center point. 
- the third and fourth patches are made by dividing the midsection of the sphere along a vertical seam that connects
the starting points of the top and bottom seams. The other half of the midsection is defined similarly, but from the mid point
of the top and bottom sides. 
For the top and bottom caps first map the boundary to a circle in the plane, with the center point of the origin.
Do this for both of the meshes. This defines a correspondence between the vertices in the top cap of the two meshes.
Do the same for the bottom cap.
Do the same thing for the two side patches, except map them to a rectangle in the plane, with the vertical seam along one edge.

When reading in the meshes read in just the vertices and the faces, ignoring any duplicate vertices or other attributes.
"""


import argparse
import argparse
from os import path
import json
from pathlib import Path

from meshlib import mrmeshnumpy as mn
from meshlib import mrmeshpy as mm
import numpy as np
from seam_topology import SeamTopology
from uv_texture import _geodesic
from MeshLabCleanup import _read_mesh_no_textures
from remove_calyx import _get_vertex_star


def _define_quadrant(mesh: mm.Mesh, edge_adjacency, seam_vertices: list[int], center_vertex: int):
    """ Using the graph adjacency do a breadth first search starting from the center vertex to define each quadrant patch. 
    The search should stop when it reaches the seam vertices. Return the list of vertices that belong to the each quadrant patch. """
    from collections import deque
    
    visited = set()
    queue = deque([center_vertex])
    cap_vertices = []

    while queue:
        v = queue.popleft()
        if v in visited:
            continue
        visited.add(v)
        if v in seam_vertices:
            continue
        cap_vertices.append(v)
        for neighbor, _ in edge_adjacency[v]:
            if neighbor not in visited:
                queue.append(neighbor)

    return cap_vertices


def _place_in_circle(mesh: mm.Mesh, edge_adjacency, boundary_vs: list[int], inside_vs: list[int], center: int):
    """ Set up a least squares problem to place the inside vertices in a circle defined by the boundary vertices and the center. 
    Use the edge lengths as weights in the least squares problem to maintain the relative distances between vertices. 
    Place the boundary vertices on the circle based on their original edge lengths.
    Place the center vertex at the origin."""

    map_vs = {center: 0}
    map_back_vs = {0: center}
    for i, v in enumerate(boundary_vs):
        map_vs[v] = i + 1
        map_back_vs[i + 1] = v
    for i, v in enumerate(inside_vs):
        map_vs[v] = i + 1 + len(boundary_vs)
        map_back_vs[i + 1 + len(boundary_vs)] = v

    # The A matrix and b vector for the least squares problem will be set up here. 
    n_constraints = len(boundary_vs) + len(inside_vs) + 1  # +1 for the center vertex
    a_mat = np.zeros((n_constraints, n_constraints))
    b_vec = np.zeros((n_constraints, 2))

    # For now, we just place the boundary vertices on the circle and the center at the origin.
    # Compute the total perimeter length of the boundary
    verts = mn.getNumpyVerts(mesh)

    total_length = 0.0
    edge_lengths = []
    for i in range(len(boundary_vs)):
        v1 = boundary_vs[i]
        v2 = boundary_vs[(i + 1) % len(boundary_vs)]
        edge_length = np.linalg.norm(verts[v1] - verts[v2])
        total_length += edge_length
        edge_lengths.append(edge_length)

    # Place the center vertex at the origin
    a_mat[0, 0] = 1.0
    b_vec[0] = np.array([0.0, 0.0])

    # Place the boundary vertices on the circle based on their original edge lengths
    import math
    angle = 0.0
    for i in range(len(boundary_vs)):
        a_mat[i + 1, i + 1] = 1.0
        b_vec[i + 1] = np.array([math.cos(angle), math.sin(angle)])
        angle += 2 * math.pi * edge_lengths[i] / total_length

    # Now place the inside the vertices as 1/n * their neighbors
    for i, v in enumerate(inside_vs):
        a_mat[i + 1 + len(boundary_vs), i + 1 + len(boundary_vs)] = 1.0
        neighbors = [n for n, _ in edge_adjacency[v]]
        n_neighbors = len(neighbors)
        perc = 1.0 / n_neighbors
        for neighbor in neighbors:
            map_v = map_vs.get(neighbor, None)
            if map_v is not None:
                a_mat[i + 1 + len(boundary_vs), map_v] = -perc

    x = np.linalg.solve(a_mat, b_vec)
    return x, map_back_vs


def _place_in_square(mesh: mm.Mesh, edge_adjacency, boundary_seam_vs: list[list[int]], inside_vs: list[int], map_vs: dict):
    # The A matrix and b vector for the least squares problem will be set up here. 
    
    # Make a flat list of all the boundary vs
    boundary_vs = [v for sublist in boundary_seam_vs for v in sublist]
    map_vs = {}
    map_back_vs = {}
    count = 0
    for v in boundary_vs:
        map_vs[v] = count
        map_back_vs[count] = v
        count += 1
    for v in inside_vs:
        if v not in map_vs:
            map_vs[v] = len(map_vs)
            map_back_vs[len(map_vs) - 1] = v

    n_constraints = len(boundary_vs) + len(inside_vs)
    a_mat = np.zeros((n_constraints, n_constraints))
    b_vec = np.zeros((n_constraints, 2))

    # Each seam gets placed on a consecutive side of the square by edge length
    sides_x = [1.0, -1.0, -1.0, 1.0]
    sides_y = [1.0, 1.0, -1.0, -1.0]

    # Place the seams on the four sides of the square
    for indx, seam in enumerate(boundary_seam_vs):
        side_x_start = sides_x[indx]
        side_y_start = sides_y[indx]
        side_x_end = sides_x[(indx + 1) % 4]
        side_y_end = sides_y[(indx + 1) % 4]
        # Sum up the edge lengths along the seam to determine placement on the side of the square
        edge_lengths = []
        for i in range(len(seam) - 1):
            v1 = seam[i]
            v2 = seam[i + 1]
            edge_lengths.append(np.linalg.norm(mesh.vertices[v1] - mesh.vertices[v2]))
        total_length = sum(edge_lengths)
        t = 0.0
        for vi, v in enumerate(seam):
            row = map_vs[v]
            a_mat[row, row] = 1.0
            b_vec[row] = np.array([side_x_start + (side_x_end - side_x_start) * t,
                                   side_y_start + (side_y_end - side_y_start) * t])
            t += (edge_lengths[vi - 1] / total_length if vi > 0 else 0.0)

    # Now place the inside the vertices as 1/n * their neighbors
    for i, v in enumerate(inside_vs):
        row = map_vs[v]
        a_mat[row, row] = 1.0
        neighbors = [n for n, _ in edge_adjacency[v]]
        n_neighbors = len(neighbors)
        perc = 1.0 / n_neighbors
        for neighbor in neighbors:
            map_v = map_vs.get(neighbor, None)
            if map_v is not None:
                a_mat[row, map_v] = -perc

    x = np.linalg.solve(a_mat, b_vec)
    return x, map_back_vs


def _create_adjacency_structure(mesh: mm.Mesh) -> list[list[tuple[int, float]]]:
    """ Create an edge adjacency structure as a list of neighbors for each vertex. """
    adjacency = [[] for _ in range(mesh.topology.numValidVerts())]

    verts = mn.getNumpyVerts(mesh)

    for v in range(0, mesh.topology.numValidVerts()):
        v_neighs = _get_vertex_star(mesh.topology, mm.VertId(v))
        for u in v_neighs:
            if u < v:
                continue

            np.linalg.norm(verts[u] - verts[v])
            length = np.linalg.norm(verts[u] - verts[v])
            weight = float(length)

            adjacency[u].append((v, weight))
            adjacency[v].append((u, weight))
    return adjacency


def _define_second_side_seam(mesh: mm.Mesh, seam_vertices: list[int], start_vertex: int, end_vertex: int):
    """ Using the graph adjacency do a breadth first search starting from the start vertex to define the second side seam patch. 
    The search should continue until it reaches the end vertex or a seam vertex, but should not cross a seam vertex. Return the list of vertices that belong to the second side seam patch. """
    edge_adjacency = _create_adjacency_structure(mesh)
    seam_vertices = set(seam_vertices)
    if start_vertex in seam_vertices:
        seam_vertices.remove(start_vertex)
    if end_vertex in seam_vertices:
        seam_vertices.remove(end_vertex)
    get_seam = _geodesic(adjacency=edge_adjacency, v1=start_vertex, v2=end_vertex, no_cross_vs=seam_vertices)
    
    return edge_adjacency, get_seam[0]


def _project_patches(mesh: mm.Mesh, seam_topology: SeamTopology):
    """ Define the new seam, then project each of the four patches onto the plane. """
    # Define the new seam by connecting the top and bottom seams at the mid point
    top_seam_vs = seam_topology.seams["top"].vertex_ids
    bottom_seam_vs = seam_topology.seams["bottom"].vertex_ids
    side_seam_vs = seam_topology.seams["side"].vertex_ids
    split_top_seam = len(top_seam_vs) // 2
    mid_top_seam = top_seam_vs[split_top_seam]
    split_bottom_seam = len(bottom_seam_vs) // 2
    mid_bottom_seam = bottom_seam_vs[split_bottom_seam]
    # All of the seam vertices in a set
    all_seam_vs = set(top_seam_vs[0:split_top_seam-1]) | set(top_seam_vs[split_top_seam+1:]) | \
                  set(bottom_seam_vs[0:split_bottom_seam-1]) | set(bottom_seam_vs[split_bottom_seam+1:]) | set(side_seam_vs)

    edge_adjacency, second_side_seam_vs = _define_second_side_seam(mesh=mesh, seam_vertices=all_seam_vs, start_vertex=mid_top_seam, end_vertex=mid_bottom_seam)


    # Top cap
    top_cap = _define_quadrant(mesh, edge_adjacency, top_seam_vs, seam_topology.top_vertex)
    bottom_cap = _define_quadrant(mesh, edge_adjacency, bottom_seam_vs, seam_topology.bottom_vertex)
    all_patch_vs = set(top_cap) | set(bottom_cap)
    all_not_free_vs = all_seam_vs | all_patch_vs

    # One side patch should be in the positive z direction
    # Take all vertices that are not in all_seam_vs or the top/bottom caps and find the ones with the
    #. minimum/maximum z coordinate to determine which side patch is in the positive z direction
    verts = mn.getNumpyVerts(mesh)
    z_mid = np.mean(verts[:, 2])
    zs_verts = np.zeros((verts.shape[0],)) + z_mid
    zs_verts = verts[:, 2]
    zs_verts[list(all_not_free_vs)] = z_mid

    min_z_vertex = np.argmin(zs_verts)
    max_z_vertex = np.argmax(zs_verts)

    side_patch = _define_quadrant(mesh, edge_adjacency, side_seam_vs, min_z_vertex)
    second_side_patch = _define_quadrant(mesh, edge_adjacency, second_side_seam_vs, max_z_vertex)

    # Now project the top and bottom caps to circles and the side patches to squares
    top_circle = _place_in_circle(mesh=mesh, edge_adjacency=edge_adjacency, boundary_vs=top_seam_vs, inside_vs=top_cap, center=seam_topology.top_vertex)
    bottom_circle = _place_in_circle(mesh=mesh, edge_adjacency=edge_adjacency, boundary_vs=bottom_seam_vs, inside_vs=bottom_cap, center=seam_topology.bottom_vertex)
    first_square = [side_seam_vs, bottom_seam_vs[0:split_bottom_seam], list(reversed(second_side_seam_vs)), list(reversed(top_seam_vs[split_top_seam:]))]
    square1 = _place_in_square(mesh=mesh, edge_adjacency=edge_adjacency, boundary_vs=first_square, inside_vs=side_patch)
    second_square = [second_side_seam_vs, list(reversed(bottom_seam_vs[split_bottom_seam:])), list(reversed(side_seam_vs)), top_seam_vs[0:split_top_seam]]
    square2 = _place_in_square(mesh=mesh, edge_adjacency=edge_adjacency, boundary_vs=second_square, inside_vs=second_side_patch, center=max_z_vertex)

    return top_circle, bottom_circle, square1, square2


def _place_vertices(mesh: mm.Mesh, 
                    mesh_patches: list[list[int]],
                    canonical_mesh: mm.Mesh, 
                    canonical_patches: list[list[int]]):
    # For each patch, figure out which vertex the canonical mesh vertx is closest to in the
    #. plane and set it to that vertex. The mesh is dense enough that there's no point in doing
    # barycentric projection would be overkill.
    for mesh_patch, canonical_patch in zip(mesh_patches, canonical_patches):
        locs_mesh, map_vs_mesh = mesh_patch[0], mesh_patch[1]
        locs_canonical_mesh, map_vs_canonical = canonical_patch[0], canonical_patch[1]
        for indx, loc in enumerate(locs_canonical_mesh):
            # Figure out which of the locs_mesh the loc is closest to
            x_dist = locs_mesh[0] - loc[0]
            y_dist = locs_mesh[1] - loc[1]
            dist = x_dist**2 + y_dist**2
            which_v = np.argmin(dist)
            which_vid_on_mesh = map_vs_mesh[which_v]
            which_vid_on_canonical = map_vs_canonical[indx]
            canonical_mesh.points[which_vid_on_canonical] = mesh.points[which_vid_on_mesh]  


def align_meshes(input_path: Path, canonical_path: Path, output_path: Path):
    """ Read in the meshes (no uv coordinates) and the seam topologies then patch each"""
    mesh = _read_mesh_no_textures(input_path)
    canonical_mesh = _read_mesh_no_textures(canonical_path)
    # Read the seam topologies for both meshes
    seam_path = input_path.with_name(input_path.stem + "_seam_topology.json")
    with seam_path.open(encoding="utf-8") as stream:
        input_topology = SeamTopology.from_dict(json.load(stream))
    canonical_seam_path = canonical_path.with_name(canonical_path.stem + "_seam_topology.json")
    with canonical_seam_path.open(encoding="utf-8") as stream:
        canonical_topology = SeamTopology.from_dict(json.load(stream))

    mesh_projected = _project_patches(mesh, input_topology)
    canonical_mesh_projected = _project_patches(canonical_mesh, canonical_topology)

    for indx in range(4):
        _place_vertices(mesh, mesh_projected[indx], canonical_mesh, canonical_mesh_projected[indx])

    mm.save_mesh(output_path, canonical_mesh)


def main() -> None:
    root = Path("/Users/cindygrimm/PycharmProjects/data/ScannedModels")
    scan = root / "ProsserEnvyApples"
    parser = argparse.ArgumentParser(description="Align a canonical apple mesh to a normalized scan.")
    parser.add_argument("input", nargs="?", type=Path, default=scan / "CalyxRemoved/Envy_001_FullRes.obj")
    parser.add_argument("canonical", nargs="?", type=Path, default=root / "Canonical/apple_fullres.obj")
    parser.add_argument("output", nargs="?", type=Path, default=scan / "AlignedMeshes/Envy_001_FullRes.obj")
    args = parser.parse_args()
    align_meshes(args.input, args.canonical, args.output)


if __name__ == "__main__":
    main()