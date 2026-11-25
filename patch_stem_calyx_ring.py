"""Patch a clean ring into the stem and mark out the calyx region.."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from meshlib import mrmeshnumpy as mn
from meshlib import mrmeshpy as mm
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve

import numpy as np
from scipy.sparse import csr_matrix
from seam_topology import SeamInfo, SeamTopology
from obj_geometry import get_vertex_star, create_adjacency_structure, bfs_graph, read_obj, write_obj
from filename_fixing import write_material_file


def _is_y_extremum(
    neighbors: list[set[int]], y: np.ndarray, vertex: int, find_min: bool, eps: float, min_height: float
) -> tuple[bool, float, list[int]]:
    """Breadth-first search outward from vertex, one ring at a time, over welded ids.

    The vertex is a local minimum (or maximum) in y if no vertex reached before the
    neighborhood has risen (fallen) by min_height is lower (higher) than it by more than eps."""
    sign = 1.0 if find_min else -1.0
    visited = {vertex}
    frontier = [vertex]
    max_rise = 0.0
    b_was_extremum = False
    while frontier:
        next_frontier = []
        # For the current frontier, track the maximum rise and whether any neighbor violates the extremum condition
        max_rise_frontier = 0.0
        b_found_neighbor_not_extremum = False
        for current in frontier:
            for neighbor in neighbors[current]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                rise = sign * (y[neighbor] - y[vertex])
                if rise < -eps:
                    b_found_neighbor_not_extremum = True
                    break
                next_frontier.append(neighbor)
                max_rise_frontier = max(max_rise_frontier, rise)
        if b_found_neighbor_not_extremum:
            break

        # Made it through one loop of neighbors, so this is an extreme
        max_rise = max(max_rise, max_rise_frontier)
        b_was_extremum = True
        frontier = next_frontier
    return b_was_extremum, max_rise, list(visited)


def _high_curvature_vertices(
    mesh: mm.Mesh, 
    vertices: np.ndarray,
    center: np.ndarray,
    height: float
) -> list[tuple[int, float, bool]]:
    """Identify vertices that are local minima or maxima in the y direction. For each vertex, grow two neighborhoods,
    one that grows upwards in the y direction and one that grows downwards. Identify vertices that are local extrema within these neighborhoods.
    The total neighborhood height around the vertex should be at least 0.1 units."""

    n_vertices = vertices.shape[0]

    # Build the nearest neighbor structure for the vertices
    vert_adj = [set() for _ in range(n_vertices)]
    vs_in_half = np.flatnonzero(np.abs(vertices[:, 1] - center[1]) <= height)
    for vi in vs_in_half:
        vert_idx = mm.VertId(vi)
        for nbr_idx in get_vertex_star(mesh.topology, vert_idx):
            vert_adj[vi].add(int(nbr_idx))

    print(f"Mesh y bounds: center={center[1]}, height={height}")
    # For each vertex in the given part of the mesh, make a list of its neighbors that are in the positive y direction and another list
    # of the neighbors in the negative y direction
    v_extrema = []
    eps = 0.01
    calyx_len = 1

    all_max = set()
    all_min = set()
    for vi in np.flatnonzero(np.abs(vertices[:, 1] - center[1]) <= height):
        vi = int(vi)
        is_extremum, val, visited = _is_y_extremum(vert_adj, vertices[:, 1], vi, find_min=True, eps=eps, min_height=calyx_len)
        if is_extremum:
            if val > calyx_len:
                v_extrema.append((vi, val, True))
                all_min.update(visited)
        else:
            is_extremum, val, visited = _is_y_extremum(vert_adj, vertices[:, 1], vi, find_min=False, eps=eps, min_height=calyx_len)
            if is_extremum:
                if val > calyx_len:
                    v_extrema.append((vi, val, False))
                    all_max.update(visited)

    print(f"Filtered positive end vertices: {len([v for v in v_extrema if v[2]])}")
    print(f"Filtered negative end vertices: {len([v for v in v_extrema if not v[2]])}")
    print(f"All positive end vertices (including neighbors): {len(all_min)}")
    print(f"All negative end vertices (including neighbors): {len(all_max)}")   
    return v_extrema

  
def _bottom_center(vertices: np.ndarray, 
                   v_extrema: list[tuple[int, float, bool]], 
                   center: np.ndarray) -> tuple[float, float, float]:
    """Estimate the calyx center and diameter to both the ring enclosing the calyx and the
       ring for the shoulder at the bottom of the apple"""
    if len(vertices) == 0 or not np.isfinite(vertices).all():
        raise ValueError("Input mesh must have finite vertices.")

    # First step is to group the min and max values by whether or not they
    #. are in the calyx or the divot right around the calyx or the shoulder at the bottom of the apple
    # The rise for the vertices in the calyx (the second value of v_min) should be a lot
    # bigger (more like 5 mm than 1 mm)
    # Find the maximum rise among the min vertices to identify the calyx region
    max_rise_min = max((val for vi, val, is_min in v_extrema if is_min), default=0.0)
    max_rise_max = max((val for vi, val, is_min in v_extrema if not is_min), default=0.0)

    # Mark as inside just those vertices that have big extrema and are within a distance 10 of the 
    #. center in x,y
    inside_calyx = []
    for vi, val, is_min in v_extrema:
        vertex = vertices[vi]
        radius = np.hypot(vertex[0] - center[0], vertex[2] - center[2])
        height = vertex[1] - center[1]
        if radius <= 10:
            if is_min and val > 0.25 * max_rise_min:
                print(f"Vertex {vi} is inside calyx with radius {radius} value {val} height {height}")
                inside_calyx.append(vi)
            elif not is_min and val > 0.25 * max_rise_max:
                print(f"Vertex {vi} is inside calyx with radius {radius} value {val} height {height}")
                inside_calyx.append(vi) 

    # Calculate the radius of the calyx region based on the inside vertices

    # Estimate the center of the calyx and its diameter based on the inside vertices
    inside_vertices = vertices[inside_calyx]
    if len(inside_vertices) == 0:
        raise ValueError("No vertices found inside the calyx region.")  
    # Find the distance between the vertices and the center along the x and z axes
    distances = np.hypot(inside_vertices[:, 0] - center[0], inside_vertices[:, 2] - center[2])
    calyx_radius = 0.9 * distances.max() + 0.1 * distances.mean()

    # The bottom circle of the apple that passes through the shoulder region
    # should be defined by a rough circle centered in the calyx region.
    all_calyx_bottom = [vi for vi, _, _ in v_extrema if vi not in inside_calyx]

    # Now do the same thing for the vertices that are in v_min and v_max, but not inside the calyx
    shoulder_radii = np.hypot(vertices[all_calyx_bottom, 0] - center[0], vertices[all_calyx_bottom, 2] - center[2])
    shoulder_radius = 0.9 * shoulder_radii.max() + 0.1 * shoulder_radii.mean()
    if shoulder_radius == 0.0:
        raise ValueError("Calyx diameter is zero, invalid calyx region.")

    # max distance of all extrema from the center in y
    max_y_dist = max(abs(vertices[vi, 1] - center[1]) for vi, _, _ in v_extrema)
    return calyx_radius, shoulder_radius, max_y_dist


def _top_center(vertices: np.ndarray, 
                v_extrema: list[tuple[int, float, bool]], 
                center: np.ndarray) -> tuple[float, float, float]:
    """Estimate the best center for the stem and a shoulder ring radius"""
    if len(vertices) == 0 or not np.isfinite(vertices).all():
        raise ValueError("Input mesh must have finite vertices.")

    # First step is find the minima that is closest to the center.
    local_minima_vs = [vi for vi, _, is_min in v_extrema if is_min]
    if not local_minima_vs:
        raise ValueError("No local minima found in the mesh.")
    distances = np.hypot(vertices[local_minima_vs, 0] - center[0], vertices[local_minima_vs, 2] - center[2])
    closest_min_index = np.argmin(distances)
    height = vertices[local_minima_vs[closest_min_index], 1]

    # Take all the vertices that are at height and within 3 of the center in x-z plane and slighly above and take their median
    vs_at_height = np.flatnonzero(np.abs(vertices[:, 1] - height) <= 2.0)  # 0.1 is a small tolerance
    vs_in_xy = np.flatnonzero(np.hypot(vertices[:, 0] - center[0], vertices[:, 2] - center[2]) <= 4)  # within 3 units in x-z plane
    vs_in_stem_region = np.intersect1d(vs_at_height, vs_in_xy)
    if vs_in_stem_region.size == 0:
        print("Warning: No vertices found in the stem region.")
        center_in_stem_region = center
    else:
        center_in_stem_region = np.median(vertices[vs_in_stem_region], axis=0)

    # The top circle of the apple passes through the shoulder region
    # should be defined by a rough circle centered in the calyx region.
    local_maxima_vs = [vi for vi, _, is_min in v_extrema if not is_min]

    # Now find the shoulder radii based on the local maxima that are not inside the calyx
    if len(local_maxima_vs) > 0:
        shoulder_radii = np.hypot(vertices[local_maxima_vs, 0] - center[0], vertices[local_maxima_vs, 2] - center[2])
        shoulder_radius = 0.9 * shoulder_radii.max() + 0.1 * shoulder_radii.mean()
    else:
        y_max = np.max(vertices[:, 1])
        y_top_vs = vertices[:, 1] > 0.95 * y_max
        shoulder_radii = np.hypot(vertices[y_top_vs, 0] - center[0], vertices[y_top_vs, 2] - center[2])
        shoulder_radius = 0.9 * shoulder_radii.max() + 0.1 * shoulder_radii.mean()
    if shoulder_radius == 0.0:
        raise ValueError("Shoulder radius is zero, invalid shoulder region.")

    # max distance of all extrema from the center in y
    max_y_dist = max(abs(vertices[vi, 1] - center[1]) for vi, _, _ in v_extrema)
    return center_in_stem_region, shoulder_radius, max_y_dist, closest_min_index, vs_in_stem_region


def _inside_ring(verts: np.ndarray, 
                 faces: np.ndarray, 
                 center: np.ndarray, 
                 radius: float, 
                 height: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Boolean mask of vertices connected to the center through vertices inside the x-z circle.
    Returns: 
    - inside: Boolean mask of vertices inside the ring and connected to the center vertex.
    - crossing_edges: Edges that cross from the inside to the outside
    - crossing ts: the t value that indicates where along the edge the crossing occurs.
    - crossing faces: faces that have at least one vertex inside and one vertex outside the ring.
    """
    dists = np.hypot(verts[:, 0] - center[0], verts[:, 2] - center[2]) 
    inside_disk = dists <= radius
    right_height = np.abs(verts[:, 1] - center[1]) <= height
    inside = inside_disk & right_height
    print(f"Inside disk: {np.count_nonzero(inside_disk)}, Right height: {np.count_nonzero(right_height)}, Inside: {np.count_nonzero(inside)}")
    if not np.count_nonzero(inside) > 0:
        raise ValueError("No mesh vertex lies inside the calyx circle.")

    # find the vertex closest to the center that is inside the ring
    dists[~inside] += 30.0
    v_id_start = dists.argmin()
    if not inside[v_id_start]:
        raise ValueError("The closest vertex to the center is not inside the ring.")

    # Do breadth-first search to find all vertices connected to the starting vertex within the ring
    all_edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges_inside = all_edges[inside[all_edges[:, 0]] & inside[all_edges[:, 1]]]
    graph = coo_matrix((np.ones(len(edges_inside)), (edges_inside[:, 0], edges_inside[:, 1])), shape=(len(verts),) * 2)
    _, label = connected_components(graph, directed=False)
    inside = inside & (label == label[v_id_start])

    # All the edges that cross from inside to outside the ring
    ec_condition = (inside[all_edges[:, 0]] & ~inside[all_edges[:, 1]]) | (inside[all_edges[:, 1]] & ~inside[all_edges[:, 0]])
    edges_crossing = all_edges[ec_condition]

    # Remove duplicate crossing edges by sorting the edges by their vertex indices and using np.unique
    edges_crossing = np.array([sorted(edge) for edge in edges_crossing])
    edges_crossing = np.unique(edges_crossing, axis=0)

    # Compute the t values for the crossing edges
    crossing_ts = []
    for edge in edges_crossing:
        v0, v1 = verts[edge[0]], verts[edge[1]]
        d0 = np.hypot(v0[0] - center[0], v0[2] - center[2]) - radius
        d1 = np.hypot(v1[0] - center[0], v1[2] - center[2]) - radius
        if np.isclose(d0, d1):
            t = 0.5
        else:
            t = d0 / (d0 - d1)
        crossing_ts.append(t)
    crossing_ts = np.array(crossing_ts)

    # Convert inside to a list of vertex indices and face indices
    inside_vs = np.where(inside)[0]
    faces_crossing = faces[np.any(inside[faces], axis=1) & np.any(~inside[faces], axis=1)]
    return inside_vs, edges_crossing, crossing_ts, faces_crossing


def _mapped_vertex_id(vertex_map: mm.VertMap, vertex_id: int) -> int:
    mapped_id = vertex_map[mm.VertId(vertex_id)].get()
    if mapped_id < 0:
        raise ValueError(f"Seam vertex {vertex_id} was removed while packing the mesh.")
    return mapped_id


def _pack_with_seams(mesh: mm.Mesh) -> mm.VertMap:

    face_map = mm.FaceMap()
    vertex_map = mm.VertMap()
    mesh.pack(face_map, vertex_map)
    mesh.invalidateCaches()
    return vertex_map


def _split_boundary_edges(mesh: mm.Mesh,
                          seam_topology: SeamTopology,
                          vertices: np.ndarray, 
                          crossing_edges: np.ndarray, 
                          crossing_ts: np.ndarray) -> tuple[int, ...]:
    """Split the boundary edges of the mesh at the specified parametric positions.

    Args:
        mesh: The input mesh.
        crossing_edges: Array of edges that cross the boundary.
        crossing_ts: Array of parametric positions along each edge where the split should occur.

    Returns:
        Array of new vertex indices created by splitting the edges.
    """
    new_vertex_ids = []

    # First get a unique set of edges with their locations and thetas
    unique_undirected_edges: dict[int, tuple[tuple[int, int], mm.Vector3f, float]] = {}
    for e, t in zip(crossing_edges, crossing_ts):
        v0, v1 = mm.VertId(e[0]), mm.VertId(e[1])
        edge_id = mesh.topology.findEdge(v0, v1)
        edge_id_sym = mesh.topology.findEdge(v1, v0)
        if not mesh.topology.hasEdge(edge_id) and not mesh.topology.hasEdge(edge_id_sym):
            continue
        if t <= 0.0 or t >= 1.0:
            print(f"Bad t {t}, setting to small value in 0, 1 range")
            t = max(min(t, 1.0 - 1e-2), 1e-2)
        new_vert_loc = (1 - t) * vertices[e[0]] + t * vertices[e[1]]
        new_vert_loc = mm.Vector3f(new_vert_loc[0], new_vert_loc[1], new_vert_loc[2])
        if edge_id.get() in unique_undirected_edges or edge_id_sym.get() in unique_undirected_edges:
            continue
        unique_undirected_edges[edge_id.get()] = (e, new_vert_loc)

    # Calculate the new locations and their theta values
    print(f"Splitting edges")
    for undir_id, (edge, new_vert_loc) in unique_undirected_edges.items():
        """Split the edge and the adjacent faces at the specified parametric position."""
        new_edge_id = mesh.splitEdge(mm.EdgeId(undir_id), new_vert_loc)
        new_vert = mesh.topology.dest(new_edge_id)
        new_vertex_ids.append(new_vert.get())
        mesh.points[mm.VertId(new_vert.get())] = new_vert_loc

    vertex_map = _pack_with_seams(mesh)
    new_vertex_ids = [_mapped_vertex_id(vertex_map, v) for v in new_vertex_ids]

    return new_vertex_ids


def _color_mesh_by_vertex_type(mesh: mm.Mesh, verts_to_color: list[tuple[list[int], mm.Color]], output_path: Path):
    vertex_colors = mm.VertColors()
    vertex_colors.resize(mesh.points.size(), mm.Color(255, 255, 255, 255))
    for vert_ids, color in verts_to_color:
        for vi in vert_ids:
            vertex_colors[mm.VertId(vi)] = color

    save_settings = mm.SaveSettings()
    save_settings.colors = vertex_colors
    mm.saveMesh(mesh, str(output_path), save_settings)


def _add_calyx_ring(mesh: mm.Mesh, seam_topology: SeamTopology, center: np.ndarray,
                    height: float, output_path: Path,
                    b_do_debug: bool = False
                    ) -> tuple[np.ndarray, float]:
    """Add a calyx ring to the mesh based on the bottom center and high curvature vertices."""
    verts = mn.getNumpyVerts(mesh)

    # Identify min and max extremum in the bottom of the mesh
    v_extrema = _high_curvature_vertices(mesh=mesh, vertices=verts, center=center, height=height)
    if b_do_debug:
        v_max = [vi for vi, _, _ in v_extrema if _]
        v_min = [vi for vi, _, _ in v_extrema if not _]
        vs_color = [(v_max, mm.Color(255, 0, 0, 255)), (v_min, mm.Color(0, 255, 0, 255))]
        curv_path = output_path.with_name(output_path.stem + "_curvature" + ".ply")
        _color_mesh_by_vertex_type(mesh, vs_color, curv_path)

    # Find a radius that encloses the calyx and another for the bumpy bit on the bottom of the mesh
    calyx_radius, shoulder_radius, max_y_dist = _bottom_center(vertices=verts, v_extrema=v_extrema, center=center)

    # Fudge some values to get the outer calyx radius
    outer_calyx = 0.8 * calyx_radius + 0.2 * shoulder_radius
    if b_do_debug:
        # All the vertices
        v_shoulder = []
        v_calyx = []
        v_bottom = []
        vs_in_half = np.flatnonzero(np.abs(verts[:, 1] - center[1]) <= height)
        for vi in vs_in_half:
            if np.abs(verts[vi, 1] - center[1]) > max_y_dist * 1.1:
                v_bottom.append(vi)
            xz_dist = np.hypot(verts[vi, 0] - center[0], verts[vi, 2] - center[2])    
            if xz_dist < outer_calyx:
                v_calyx.append(vi)
            elif shoulder_radius * 0.8 <= xz_dist < shoulder_radius * 1.25:
                v_shoulder.append(vi)   
            else:
                v_bottom.append(vi)

        split_path = output_path.with_name(output_path.stem + "_split" + ".ply")
        vs_color = [(v_calyx, mm.Color(255, 0, 0, 255)), (v_shoulder, mm.Color(0, 255, 0, 255)), (v_bottom, mm.Color(0, 0, 255, 255))]
        _color_mesh_by_vertex_type(mesh, vs_color, split_path)

    # Now actually find the vertices that cross the calyx boundary
    faces = mn.getNumpyFaces(mesh.topology)
    in_calyx, crossing_edges, crossing_ts, crossing_faces = _inside_ring(verts=verts, 
                                                                         faces=faces, 
                                                                         center=center, 
                                                                         radius=outer_calyx, height=max_y_dist * 1.1)

    if b_do_debug:
        vs_color = [(in_calyx, mm.Color(0, 255, 0, 255))]
        edge_vs = []
        for e in crossing_edges:
            edge_vs.append(e[0])
            edge_vs.append(e[1])

        ring_path = output_path.with_name(output_path.stem + "_ring" + ".ply")
        vs_color = [(v_calyx, mm.Color(255, 0, 0, 255)), (edge_vs, mm.Color(0, 255, 0, 255))]
        _color_mesh_by_vertex_type(mesh, vs_color, ring_path)

    if not crossing_faces.any():
        raise ValueError("No calyx faces found.")

    # Split the boundary edges of the calyx region to prepare for cutting out calyx and or patching
    ring_vs = _split_boundary_edges(
        mesh=mesh,
        seam_topology=seam_topology,
        vertices=verts, 
        crossing_edges=crossing_edges, 
        crossing_ts=crossing_ts
        )
    verts = mn.getNumpyVerts(mesh)
    ring_locs = verts[list(ring_vs)]

    if b_do_debug:
        ring_split_path = output_path.with_name(output_path.stem + "_ring_split" + ".ply")
        vs_color = [(ring_vs, mm.Color(255, 0, 0, 255))]
        _color_mesh_by_vertex_type(mesh, vs_color, ring_split_path)

    return ring_locs, shoulder_radius


def _sort_ring_vertices(topo, boundary_vs: list[int]) -> list[int]:
    """ Start with one vertex in the boundary list and sort the boundary vertices in order around the loop """
    if not boundary_vs:
        return []

    sorted_vs = [boundary_vs[0]]
    while len(sorted_vs) < len(boundary_vs):
        last_v = sorted_vs[-1]
        adj_vs = get_vertex_star(topo=topo, v=mm.VertId(last_v))
        b_found = False
        for adj_v in adj_vs:
            if adj_v in boundary_vs and adj_v not in sorted_vs:
                e_forward = topo.findEdge(mm.VertId(last_v), mm.VertId(adj_v))
                if topo.left(e_forward).valid():
                    continue
                else:
                    if b_found:
                        raise ValueError(f"Multiple valid next vertices found for vertex {last_v}")
                    b_found = True
                    sorted_vs.append(adj_v)
        if not b_found:
            raise ValueError(f"Cannot find next vertex in boundary loop for vertex {last_v}")

    if not len(sorted_vs) == len(boundary_vs):
        raise ValueError("Boundary loop is incomplete.")
    return sorted_vs


def _patch_stem(mesh: mm.Mesh, seam_topology: SeamTopology, center: np.ndarray,
                height: float, output_path: Path,
                b_do_debug: bool = False,
                ) -> tuple[np.ndarray, float]:
    """ Find a small 2mm ring around the lowest point in the top, cut it out and replace with a flatish patch with a center point"""
    verts = mn.getNumpyVerts(mesh)

    # Identify min and max extremum at the top of the mesh
    v_extrema = _high_curvature_vertices(mesh=mesh, vertices=verts, center=center, height=height)
    if b_do_debug:
        v_max = [vi for vi, _, _ in v_extrema if _]
        v_min = [vi for vi, _, _ in v_extrema if not _]
        vs_color = [(v_max, mm.Color(255, 0, 0, 255)), (v_min, mm.Color(0, 255, 0, 255))]
        curv_path = output_path.with_name(output_path.stem + "_top_curvature" + ".ply")
        _color_mesh_by_vertex_type(mesh, vs_color, curv_path)

    # Find a radius that encloses the calyx and another for the bumpy bit on the bottom of the mesh
    center_pt, shoulder_radius, max_y_dist, closest_min_index, marked_stem = _top_center(vertices=verts, v_extrema=v_extrema, center=center)

    if b_do_debug:
        vs_color = [(marked_stem, mm.Color(0, 255, 0, 255)), ([closest_min_index], mm.Color(255, 0, 0, 255))]
        curv_path = output_path.with_name(output_path.stem + "_top_stem" + ".ply")
        _color_mesh_by_vertex_type(mesh, vs_color, curv_path)

    # Now actually find the vertices that cross the stem boundary
    faces = mn.getNumpyFaces(mesh.topology)
    stem_radius = 1.5  # Two millimeters
    in_cstem, crossing_edges, crossing_ts, crossing_faces = _inside_ring(verts=verts, 
                                                                         faces=faces, 
                                                                         center=center_pt, 
                                                                         radius=stem_radius, height=max_y_dist * 1.1)

    if b_do_debug:
        edge_vs = []
        for e in crossing_edges:
            edge_vs.append(e[0])
            edge_vs.append(e[1])

        ring_path = output_path.with_name(output_path.stem + "_top_ring" + ".ply")
        vs_color = [(edge_vs, mm.Color(0, 255, 0, 255)), (in_cstem, mm.Color(255, 0, 0, 255))]
        _color_mesh_by_vertex_type(mesh, vs_color, ring_path)

    if not crossing_faces.any():
        raise ValueError("No stem faces found.")

    # Split the boundary edges of the calyx region to prepare for cutting out stem and patching
    ring_vs = _split_boundary_edges(
        mesh=mesh,
        seam_topology=seam_topology,
        vertices=verts, 
        crossing_edges=crossing_edges, 
        crossing_ts=crossing_ts
        )

    if b_do_debug:
        ring_split_path = output_path.with_name(output_path.stem + "_stem_split" + ".ply")
        vs_color = [(ring_vs, mm.Color(255, 0, 0, 255))]
        _color_mesh_by_vertex_type(mesh, vs_color, ring_split_path)

    # Cut out the stem region and prepare for patching
    verts_to_delete: mm.VertBitSet = mm.VertBitSet()
    verts_to_delete.resize(mesh.topology.vertSize())

    # Mark vertices for deletion
    for vid in in_cstem:
        v: mm.VertId = mm.VertId(vid)
        if mesh.topology.hasVert(v):
            verts_to_delete.set(v, True)
    faces_to_delete: mm.FaceBitSet = mm.getIncidentFaces(topology=mesh.topology, verts=verts_to_delete)
    mesh.deleteFaces(fs=faces_to_delete)
    vertex_map = _pack_with_seams(mesh)

    verts = mn.getNumpyVerts(mesh)
    ring_vs = [_mapped_vertex_id(vertex_map, v) for v in ring_vs]
    ring_locs = verts[ring_vs, :]

    if b_do_debug:
        mesh_with_hole_path = output_path.with_name(output_path.stem + "_stem_hole" + ".ply")
        vs_color = [(ring_vs, mm.Color(255, 0, 0, 255))]
        _color_mesh_by_vertex_type(mesh, vs_color, mesh_with_hole_path)

    center = np.mean(verts[ring_vs, :], axis=0)
    center_pt_loc = mm.Vector3f(center[0], center[1], center[2])
    center_vert: mm.VertId = mesh.addPoint(pos=center_pt_loc)
    seam_topology.top_vertex = int(center_vert)

    new_faces = []
    ring_vs = _sort_ring_vertices(mesh.topology, ring_vs)
    for indx in range(len(ring_vs)):
        v1: mm.VertId = mm.VertId(ring_vs[indx])
        v2: mm.VertId = mm.VertId(ring_vs[(indx + 1) % len(ring_vs)])
        new_face = mm.ThreeVertIds([center_vert, v1, v2])
        new_faces.append(new_face)
    new_faces_tri = mm.Triangulation(
        mm.std_vector_std_array_Id_VertTag_3ul(new_faces)
    )

    n_faces_befpre = mesh.topology.numValidFaces()
    settings: mm.MeshBuilder.BuildSettings = mm.MeshBuilder.BuildSettings()
    settings.shiftFaceId = mesh.topology.numValidFaces()

    # 4. Add the triangle to mesh.topology and invalidate caches
    mm.MeshBuilder.addTriangles(
        mesh.topology,
        new_faces_tri,
        settings,
    )
    n_faces_after = mesh.topology.numValidFaces()
    if n_faces_after - n_faces_befpre != len(new_faces):
        raise ValueError("Failed to add all new faces to the mesh.")
    vertex_map = _pack_with_seams(mesh)
    ring_vs = [_mapped_vertex_id(vertex_map, v) for v in ring_vs]
    center_vert = mm.VertId(_mapped_vertex_id(vertex_map, center_vert.get()))
    mesh.invalidateCaches()

    if b_do_debug:
        vs_color = [(ring_vs, mm.Color(0, 255, 0, 255)), ([center_vert.get()], mm.Color(255, 0, 0, 255))]
        patch_path = output_path.with_name(output_path.stem + "_top_patch" + ".ply")
        _color_mesh_by_vertex_type(mesh, vs_color, patch_path)

    return ring_locs, shoulder_radius


def patch_stem_and_ring_calyx(
    mesh: mm.Mesh, seam_topology: SeamTopology, output_path: Path, b_do_debug: bool = False
) -> mm.Mesh:
    height = 0.15 * (seam_topology.ymax - seam_topology.ymin)
    top_center = np.asarray(seam_topology.top_point)
    bottom_center = np.asarray(seam_topology.bottom_point)

    calyx_v_locs, bottom_shoulder_radius = _add_calyx_ring(
        mesh=mesh, seam_topology=seam_topology, center=bottom_center, height=height, output_path=output_path, b_do_debug=b_do_debug
    )

    stem_v_locs, top_shoulder_radius = _patch_stem(
        mesh=mesh, seam_topology=seam_topology, center=top_center, height=height, output_path=output_path, b_do_debug=b_do_debug
    )
    seam_topology.top_shoulder_radius = float(top_shoulder_radius)
    seam_topology.bottom_shoulder_radius = float(bottom_shoulder_radius)

    verts = mn.getNumpyVerts(mesh)
    # Find the closest vertex index
    calyx_v_id = []
    calyx_v_loc = []
    for loc in calyx_v_locs:
        dists = np.linalg.norm(verts - loc, axis=1)
        closest_idx = np.argmin(dists)
        calyx_v_id.append(int(closest_idx))
        calyx_v_loc.append([float(loc[0]), float(loc[1]), float(loc[2])])
    calyx_seam = SeamInfo(name="calyx", 
                          vertex_ids=calyx_v_id, 
                          vertex_locations=calyx_v_loc, 
                          center=seam_topology.bottom_point, 
                          radius=float(bottom_shoulder_radius))

    seam_topology.seams["calyx"] = calyx_seam

    # Now do the same for the stem
    stem_v_id = []
    stem_v_loc = []
    for loc in stem_v_locs:
        dists = np.linalg.norm(verts - loc, axis=1)
        closest_idx = np.argmin(dists)
        stem_v_id.append(int(closest_idx))
        stem_v_loc.append([float(loc[0]), float(loc[1]), float(loc[2])])
    stem_seam = SeamInfo(name="stem", 
                         vertex_ids=stem_v_id, 
                         vertex_locations=stem_v_loc, 
                         center=[float(top_center[0]), float(top_center[1]), float(top_center[2])], 
                         radius=float(top_shoulder_radius))

    seam_topology.seams["stem"] = stem_seam
    return mesh


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, default=None, help="Directory the paths are relative to")
    parser.add_argument("--input-path", type=Path, required=True, help="Apple scan mesh")
    parser.add_argument("--output-path", type=Path, required=True, help="Patched mesh to write")
    parser
    args = parser.parse_args()
    input_path, output_path = args.input_path, args.output_path
    if args.root_dir is not None:
        input_path, output_path = args.root_dir / input_path, args.root_dir / output_path

    seam_topology_path = input_path.with_name(input_path.stem + "_seam_topology.json")
    with open(seam_topology_path, "r") as handle:
        seam_topology = SeamTopology.from_dict(json.load(handle))

    mesh=mm.loadMesh(str(input_path))
    # Patch the stem connection with a ring of vertices and add a split ring around the calyx region
    # Also computes the size of the ring for the bottom part of the apple and center points
    patched_mesh = patch_stem_and_ring_calyx(
        mesh=mesh, seam_topology=seam_topology, output_path=output_path, b_do_debug=False
    )


if __name__ == "__main__":
    main()
