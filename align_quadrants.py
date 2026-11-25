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

from scipy.sparse import csc_array
from scipy.sparse.linalg import spsolve


from meshlib import mrmeshnumpy as mn
from meshlib import mrmeshpy as mm
import numpy as np
from seam_topology import SeamTopology
from uv_texture import _geodesic
from obj_geometry import create_adjacency_structure, read_mesh_no_textures
from projection_routines import place_in_circle, place_in_square


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


def _define_second_side_seam(mesh: mm.Mesh, seam_vertices: list[int], start_vertex: int, end_vertex: int):
    """ Using the graph adjacency do a breadth first search starting from the start vertex to define the second side seam patch. 
    The search should continue until it reaches the end vertex or a seam vertex, but should not cross a seam vertex. Return the list of vertices that belong to the second side seam patch. """
    edge_adjacency = create_adjacency_structure(mesh)
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
    top_circle = place_in_circle(mesh=mesh, edge_adjacency=edge_adjacency, boundary_vs=top_seam_vs, inside_vs=top_cap, center=seam_topology.top_vertex)
    bottom_circle = place_in_circle(mesh=mesh, edge_adjacency=edge_adjacency, boundary_vs=bottom_seam_vs, inside_vs=bottom_cap, center=seam_topology.bottom_vertex)
    first_square = [side_seam_vs, bottom_seam_vs[0:split_bottom_seam], list(reversed(second_side_seam_vs)), list(reversed(top_seam_vs[split_top_seam:]))]
    square1 = place_in_square(mesh=mesh, edge_adjacency=edge_adjacency, boundary_vs=first_square, inside_vs=side_patch)
    second_square = [second_side_seam_vs, list(reversed(bottom_seam_vs[split_bottom_seam:])), list(reversed(side_seam_vs)), top_seam_vs[0:split_top_seam]]
    square2 = place_in_square(mesh=mesh, edge_adjacency=edge_adjacency, boundary_vs=second_square, inside_vs=second_side_patch, center=max_z_vertex)

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
    mesh = read_mesh_no_textures(input_path)
    canonical_mesh = read_mesh_no_textures(canonical_path)
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