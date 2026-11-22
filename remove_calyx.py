"""Split a textured OBJ at its saved calyx ring and grid-patch the remaining apple."""

import argparse
import json
import math
import shlex
import shutil
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
from normalize_uv_projection import _calyx_uv_candidates, _read_obj, _read_texture, _vertex_geometry
from uv_texture import _find_centers


def _get_vertex_star(topo, v):
  if not topo.hasVert(v):
    return []

  # Get the first outgoing half-edge from vertex v
  start_edge = topo.edgeWithOrg(v)
  if not start_edge.valid():
    return []  # Isolated vertex with no connected faces

  neighbors = []
  curr_edge = start_edge

  # Circulate through all outgoing half-edges around v
  while True:
    dest_vert = topo.dest(curr_edge)
    neighbors.append(dest_vert.get())
    # In MeshLib, next() rotates to the next outgoing edge around the same origin vertex
    curr_edge = topo.next(curr_edge)
    if curr_edge == start_edge:
      break

  return neighbors


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
        for nbr_idx in _get_vertex_star(mesh.topology, vert_idx):
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


def _pack_with_seams(mesh: mm.Mesh, seam_topology: SeamTopology) -> mm.VertMap:
    face_map = mm.FaceMap()
    vertex_map = mm.VertMap()
    mesh.pack(face_map, vertex_map)
    mesh.invalidateCaches()

    seam_topology.renumber_seams(vertex_map=vertex_map)
    return vertex_map


def _store_ring(mesh: mm.Mesh, seam_topology: SeamTopology, name: str,
                ring_vs: list[int], center: np.ndarray, radius: float) -> None:
    vertices = mn.getNumpyVerts(mesh)
    seam_topology.add_seam(name, SeamInfo(
        name=name,
        vertex_ids=list(ring_vs),
        vertex_locations=vertices[ring_vs].tolist(),
        uv_coords=[],
        center=tuple(float(value) for value in center),
        radius=float(radius),
    ))


def _split_boundary_edges(mesh: mm.Mesh,
                          seam_topology: SeamTopology,
                          center: np.ndarray,
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
    for undir_id, (edge, new_vert_loc) in unique_undirected_edges.items():
        """Split the edge and the adjacent faces at the specified parametric position."""
        new_edge_id = mesh.splitEdge(mm.EdgeId(undir_id), new_vert_loc)
        new_vert = mesh.topology.dest(new_edge_id)
        new_vertex_ids.append(new_vert.get())
        mesh.points[mm.VertId(new_vert.get())] = new_vert_loc
        print(f"Split edge {undir_id} at new vertex {new_vert.get()}, loc{new_vert_loc}")

    vertex_map = _pack_with_seams(mesh, seam_topology)
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
                    ) -> tuple[tuple[tuple[int, ...], tuple[float, ...]], float, float]:
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
        center=center,
        vertices=verts, 
        crossing_edges=crossing_edges, 
        crossing_ts=crossing_ts
        )
    _store_ring(mesh, seam_topology, "calyx", list(ring_vs), center, outer_calyx)

    if b_do_debug:
        ring_split_path = output_path.with_name(output_path.stem + "_ring_split" + ".ply")
        vs_color = [(ring_vs, mm.Color(255, 0, 0, 255))]
        _color_mesh_by_vertex_type(mesh, vs_color, ring_split_path)

    return ring_vs, outer_calyx, shoulder_radius


def _sort_ring_vertices(topo, boundary_vs: list[int]) -> list[int]:
    """ Start with one vertex in the boundary list and sort the boundary vertices in order around the loop """
    if not boundary_vs:
        return []

    sorted_vs = [boundary_vs[0]]
    while len(sorted_vs) < len(boundary_vs):
        last_v = sorted_vs[-1]
        adj_vs = _get_vertex_star(topo=topo, v=mm.VertId(last_v))
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
                ) -> tuple[list[int], np.ndarray, float, float]:
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
        center=center,
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
    vertex_map = _pack_with_seams(mesh, seam_topology)

    verts = mn.getNumpyVerts(mesh)
    ring_vs = [_mapped_vertex_id(vertex_map, v) for v in ring_vs]

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
    vertex_map = _pack_with_seams(mesh, seam_topology)
    ring_vs = [_mapped_vertex_id(vertex_map, v) for v in ring_vs]
    center_vert = mm.VertId(_mapped_vertex_id(vertex_map, center_vert.get()))
    mesh.invalidateCaches()
    _store_ring(mesh, seam_topology, "stem", ring_vs, center_pt_loc, stem_radius)

    if b_do_debug:
        vs_color = [(ring_vs, mm.Color(0, 255, 0, 255)), ([center_vert.get()], mm.Color(255, 0, 0, 255))]
        patch_path = output_path.with_name(output_path.stem + "_top_patch" + ".ply")
        _color_mesh_by_vertex_type(mesh, vs_color, patch_path)

    return ring_vs, center, shoulder_radius, max_y_dist


def patch_stem_and_ring_calyx(
    mesh: mm.Mesh, seam_topology: SeamTopology, output_path: Path, b_do_debug: bool = False
) -> mm.Mesh:
    height = 0.15 * (seam_topology.ymax - seam_topology.ymin)
    top_center = np.asarray(seam_topology.top_point)
    bottom_center = np.asarray(seam_topology.bottom_point)
    calyx_ring_vs, calyx_radius, bottom_shoulder_radius = _add_calyx_ring(
        mesh=mesh, seam_topology=seam_topology, center=bottom_center, height=height, output_path=output_path, b_do_debug=b_do_debug
    )

    stem_ring_vs, top_center_loc, top_shoulder_radius, max_y_dist_top = _patch_stem(
        mesh=mesh, seam_topology=seam_topology, center=top_center, height=height, output_path=output_path, b_do_debug=b_do_debug
    )
    seam_topology.top_shoulder_radius = float(top_shoulder_radius)
    seam_topology.bottom_shoulder_radius = float(bottom_shoulder_radius)
    return mesh


def check_calyx_and_stem(input_path: Path, output_path: Path) -> Path:
    """Cut the calyx out of the mesh at input_path, patch the hole, and write the result to output_path."""
    input_path, output_path = Path(input_path), Path(output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input mesh not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    mesh: mm.Mesh = mm.loadMesh(str(input_path))
    vertices = mn.getNumpyVerts(mesh)
    top_point, bottom_point = _find_centers(input_path=input_path, vertices=vertices)
    seam_topology = SeamTopology(
        ymin=float(np.min(vertices[:, 1])),
        ymax=float(np.max(vertices[:, 1])),
        top_point=tuple(top_point),
        bottom_point=tuple(bottom_point),
        seams={},
    )
    mesh_patched = patch_stem_and_ring_calyx(
        mesh=mesh, seam_topology=seam_topology, output_path=output_path, b_do_debug=True
    )


def _calyx_faces(vertices, faces, seam, bottom_point):
    """Find the component inside the ring without relying on seam vertex order."""
    ids = seam.vertex_ids
    if len(ids) < 3 or len(set(ids)) != len(ids) or any(
        not isinstance(v, int) or v < 0 or v >= len(vertices) for v in ids
    ):
        raise ValueError("The calyx seam must contain at least three distinct valid vertex IDs.")
    locations = np.asarray(seam.vertex_locations, dtype=float)
    if locations.shape != (len(ids), 3) or not np.allclose(
        vertices[ids], locations, rtol=1e-5, atol=1e-5
    ):
        raise ValueError("Calyx seam vertex locations do not match the OBJ.")
    ring_set = set(ids)
    edges = {}
    for face_id, face in enumerate(faces):
        for a, b in zip(face, np.roll(face, -1)):
            edges.setdefault(tuple(sorted((int(a), int(b)))), []).append(face_id)
    ring_edges = {edge for edge in edges if edge[0] in ring_set and edge[1] in ring_set}
    neighbors = {v: [] for v in ids}
    for a, b in ring_edges:
        neighbors[a].append(b)
        neighbors[b].append(a)
        if len(edges[(a, b)]) != 2:
            raise ValueError("Every calyx ring edge must have two adjacent faces.")
    if any(len(n) != 2 for n in neighbors.values()):
        raise ValueError("Calyx seam vertices must form one simple mesh-edge ring.")
    ring = [ids[0]]
    while True:
        choices = [v for v in neighbors[ring[-1]] if len(ring) == 1 or v != ring[-2]]
        following = choices[0]
        if following == ring[0]:
            break
        if following in ring:
            raise ValueError("Calyx seam contains multiple or intersecting rings.")
        ring.append(following)
    if len(ring) != len(ids):
        raise ValueError("Calyx seam contains multiple rings.")

    adjacency = [[] for _ in faces]
    for edge, incident in edges.items():
        if len(incident) > 2:
            raise ValueError(f"Non-manifold OBJ edge {edge}.")
        if edge not in ring_edges and len(incident) == 2:
            a, b = incident
            adjacency[a].append(b)
            adjacency[b].append(a)
    center = np.asarray(bottom_point, dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("The seam topology must contain a finite bottom center.")
    candidates = np.array([v for v in np.unique(faces) if v not in ring_set])
    if not len(candidates):
        raise ValueError("No vertices are available inside the calyx ring.")
    polygon = vertices[ring][:, [0, 2]]
    max_distance = float(np.max(np.linalg.norm(vertices[ring] - center, axis=1)))
    center_vertex = None
    for candidate in candidates[np.argsort(np.linalg.norm(vertices[candidates] - center, axis=1))]:
        if np.linalg.norm(vertices[candidate] - center) > max_distance + 1e-8:
            break
        x, z = vertices[candidate, [0, 2]]
        is_inside = False
        for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
            if (a[1] > z) != (b[1] > z) and x < (b[0] - a[0]) * (z - a[1]) / (b[1] - a[1]) + a[0]:
                is_inside = not is_inside
        if is_inside:
            center_vertex = int(candidate)
            break
    if center_vertex is None:
        raise ValueError("No interior calyx vertex was found within the projected ring.")
    seed = int(np.flatnonzero(np.any(faces == center_vertex, axis=1))[0])
    inside = {seed}
    stack = [seed]
    while stack:
        for neighbor in adjacency[stack.pop()]:
            if neighbor not in inside:
                inside.add(neighbor)
                stack.append(neighbor)
    if any(sum(face in inside for face in edges[edge]) != 1 for edge in ring_edges):
        raise ValueError("Calyx ring does not separate the mesh into an inside and outside.")
    # Match the patch winding to the removed faces.
    for face_id in edges[tuple(sorted((ring[0], ring[1])))]:
        if face_id in inside:
            face = list(faces[face_id])
            if face[(face.index(ring[0]) + 1) % 3] != ring[1]:
                ring = [ring[0], *reversed(ring[1:])]
            break
    return ring, np.array(sorted(inside), dtype=int), center_vertex


def _cross(a, b, c):
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _delaunay_patch(points, triangles, sign, tolerance, scale):
    """Improve interior edges while keeping the calyx boundary fixed."""
    triangles = np.asarray(triangles, dtype=int).copy()
    while True:
        edges = {}
        for index, face in enumerate(triangles):
            for a, b in zip(face, np.roll(face, -1)):
                edges.setdefault(tuple(sorted((int(a), int(b)))), []).append(index)
        changed = False
        for incident in edges.values():
            if len(incident) != 2:
                continue
            first, second = incident
            face = triangles[first]
            other = triangles[second]
            shared = set(face) & set(other)
            for i in range(3):
                a, b, c = map(int, (face[i], face[(i + 1) % 3], face[(i + 2) % 3]))
                if a in shared and b in shared:
                    break
            d = next(int(v) for v in other if v not in shared)
            replacement = np.array([[c, a, d], [c, d, b]])
            if any(sign * _cross(*points[tri]) <= tolerance for tri in replacement):
                continue
            pa, pb, pc = points[[a, b, c]] - points[d]
            determinant = (np.dot(pa, pa) * (pb[0] * pc[1] - pb[1] * pc[0])
                           - np.dot(pb, pb) * (pa[0] * pc[1] - pa[1] * pc[0])
                           + np.dot(pc, pc) * (pa[0] * pb[1] - pa[1] * pb[0]))
            if sign * determinant <= scale ** 4 * 1e-12:
                continue
            triangles[first], triangles[second] = replacement
            changed = True
            break
        if not changed:
            return triangles


def _unfold_patch_uvs(points, uvs, triangles, geometry_sign, tolerance):
    """Flip interior edges where doing so reduces UV inversions without folding the geometry."""
    triangles = triangles.copy()
    uv_sign = np.sign(sum(_cross(*uvs[face]) for face in triangles))
    if not uv_sign:
        raise ValueError("The calyx ring has zero UV area.")
    uv_tolerance = 1e-14
    while True:
        areas = np.array([uv_sign * _cross(*uvs[face]) for face in triangles])
        bad = np.flatnonzero(areas <= uv_tolerance)
        if not len(bad):
            return triangles
        edge_faces = {}
        for index, face in enumerate(triangles):
            for a, b in zip(face, np.roll(face, -1)):
                edge_faces.setdefault(tuple(sorted((int(a), int(b)))), []).append(index)
        best = None
        for index in bad:
            face = triangles[index]
            for i in range(3):
                a, b, c = map(int, (face[i], face[(i + 1) % 3], face[(i + 2) % 3]))
                neighbors = edge_faces[(min(a, b), max(a, b))]
                if len(neighbors) != 2:
                    continue
                other = neighbors[0] if neighbors[1] == index else neighbors[1]
                d = next(int(v) for v in triangles[other] if v != a and v != b)
                replacement = np.array([[c, a, d], [c, d, b]])
                if any(geometry_sign * _cross(*points[tri]) <= tolerance for tri in replacement):
                    continue
                new_areas = np.array([uv_sign * _cross(*uvs[tri]) for tri in replacement])
                old_areas = areas[[index, other]]
                old_score = (int(np.sum(old_areas <= uv_tolerance)),
                             float(np.maximum(uv_tolerance - old_areas, 0).sum()))
                new_score = (int(np.sum(new_areas <= uv_tolerance)),
                             float(np.maximum(uv_tolerance - new_areas, 0).sum()))
                improvement = (old_score[0] - new_score[0], old_score[1] - new_score[1])
                if improvement > (0, 1e-18) and (best is None or improvement > best[0]):
                    best = (improvement, index, other, replacement)
        if best is None:
            raise ValueError("The calyx center and ring UVs produce a degenerate or inverted grid patch.")
        _, index, other, replacement = best
        triangles[index], triangles[other] = replacement


def _grid_patch(vertices, faces, uv_faces, uvs, ring, inside, center_vertex, center):
    """Triangulate the ring in XZ, inserting a Cartesian grid and the center."""
    boundary = vertices[ring][:, [0, 2]]
    scale = float(np.max(np.ptp(boundary, axis=0)))
    if scale <= 0:
        raise ValueError("Calyx ring has no XZ extent.")
    tolerance = scale * scale * 1e-10
    area = sum(_cross(np.zeros(2), boundary[i], boundary[(i + 1) % len(ring)])
               for i in range(len(ring)))
    if abs(area) <= tolerance:
        raise ValueError("Calyx ring has zero projected area.")
    sign = 1 if area > 0 else -1
    points = list(boundary)
    remaining = list(range(len(ring)))
    triangles = []
    while len(remaining) > 3:
        for i, current in enumerate(remaining):
            a, b, c = remaining[i - 1], current, remaining[(i + 1) % len(remaining)]
            if sign * _cross(points[a], points[b], points[c]) <= tolerance:
                continue
            if any(
                min(sign * _cross(points[a], points[b], points[p]),
                    sign * _cross(points[b], points[c], points[p]),
                    sign * _cross(points[c], points[a], points[p])) >= -tolerance
                for p in remaining if p not in (a, b, c)
            ):
                continue
            triangles.append((a, b, c))
            remaining.pop(i)
            break
        else:
            raise ValueError("Calyx ring is self-intersecting or cannot be triangulated in XZ.")
    triangles.append(tuple(remaining))

    center = np.asarray(center, dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("Calyx ring must have a finite 3D center.")
    grid_size = max(3, len(ring) // 20)
    spacing = 2 * float(np.max(np.abs(boundary - center[[0, 2]]))) / grid_size
    if spacing <= scale * 1e-8:
        raise ValueError("Calyx ring is too small for a grid.")
    def insert(point, required=False):
        matches = []
        for index, (a, b, c) in enumerate(triangles):
            crosses = np.array([sign * _cross(points[a], points[b], point),
                                sign * _cross(points[b], points[c], point),
                                sign * _cross(points[c], points[a], point)])
            if np.min(crosses) >= -tolerance:
                matches.append((index, (a, b, c), crosses))
        if not matches or any(
            np.linalg.norm(point - points[v]) <= scale * 1e-9
            for _, triangle, _ in matches for v in triangle
        ):
            if required:
                raise ValueError("Calyx center must be strictly inside the projected ring.")
            return
        new_index = len(points)
        replacement = []
        for _, triangle, crosses in matches:
            for edge in range(3):
                if crosses[edge] > tolerance:
                    replacement.append((triangle[edge], triangle[(edge + 1) % 3], new_index))
        if not replacement or (required and len(matches) == 1 and np.min(matches[0][2]) <= tolerance):
            raise ValueError("Calyx center lies on the boundary of the projected ring.")
        for index, _, _ in reversed(matches):
            triangles.pop(index)
        points.append(point)
        triangles.extend(replacement)
        return new_index

    center_v_id = insert(center[[0, 2]], required=True)
    edge_vectors = np.roll(boundary, -1, axis=0) - boundary
    edge_lengths = np.sum(edge_vectors * edge_vectors, axis=1)
    if np.any(edge_lengths <= tolerance):
        raise ValueError("Calyx ring has coincident projected vertices.")
    grid_extent = math.ceil(grid_size / 2)
    for x in range(-grid_extent, grid_extent + 1):
        for z in range(-grid_extent, grid_extent + 1):
            if x or z:
                point = center[[0, 2]] + spacing * np.array([x, z])
                t = np.clip(np.sum((point - boundary) * edge_vectors, axis=1) / edge_lengths, 0, 1)
                distance = np.min(np.linalg.norm(point - boundary - t[:, None] * edge_vectors, axis=1))
                if distance > 0.2 * spacing:
                    insert(point)
    points = np.asarray(points)
    triangles = _delaunay_patch(points, triangles, sign, tolerance, scale)

    def unique_uv(vertex):
        values = uvs[uv_faces[inside][faces[inside] == vertex]]
        if not len(values) or not np.allclose(values, values[0], rtol=0, atol=2e-6):
            raise ValueError(f"Calyx vertex {vertex} must have one UV coordinate on the calyx.")
        return values[0]

    ring_uvs = np.array([unique_uv(v) for v in ring])
    center_uv = unique_uv(center_vertex)
    # Harmonic interpolation preserves the boundary UVs and the original center UV.
    fixed = np.vstack((np.column_stack((vertices[ring, 1], ring_uvs)),
                       [float(np.mean(vertices[ring, 1])), *center_uv]))
    adjacency = [set() for _ in points]
    for a, b, c in triangles:
        for v, w in ((a, b), (b, c), (c, a)):
            adjacency[v].add(w)
            adjacency[w].add(v)
    n_fixed = len(fixed)
    rows, cols, data = [], [], []
    rhs = np.zeros((len(points) - n_fixed, 3))
    for vertex in range(n_fixed, len(points)):
        row = vertex - n_fixed
        rows.append(row)
        cols.append(row)
        data.append(len(adjacency[vertex]))
        for neighbor in adjacency[vertex]:
            if neighbor < n_fixed:
                rhs[row] += fixed[neighbor]
            else:
                rows.append(row)
                cols.append(neighbor - n_fixed)
                data.append(-1)
    if len(rhs):
        values = np.vstack((fixed, spsolve(coo_matrix((data, (rows, cols)),
                                                     shape=(len(rhs), len(rhs))).tocsr(), rhs)))
    else:
        values = fixed
    patch_vertices = np.column_stack((points[:, 0], values[:, 0], points[:, 1]))
    # Intersect each center-to-grid ray with the ring so the offset vanishes at its boundary.
    for index in range(len(ring), len(points)):
        direction = points[index] - center[[0, 2]]
        if index == len(ring):
            patch_vertices[index, 1] += 2
            continue
        relative = boundary - center[[0, 2]]
        denominator = direction[0] * edge_vectors[:, 1] - direction[1] * edge_vectors[:, 0]
        valid = np.abs(denominator) > scale * scale * 1e-12
        distances = np.full(len(boundary), np.inf)
        segment_positions = np.full(len(boundary), np.inf)
        distances[valid] = (relative[valid, 0] * edge_vectors[valid, 1]
                            - relative[valid, 1] * edge_vectors[valid, 0]) / denominator[valid]
        segment_positions[valid] = (relative[valid, 0] * direction[1]
                                    - relative[valid, 1] * direction[0]) / denominator[valid]
        hits = distances[(distances >= 1 - 1e-9) & (segment_positions >= -1e-9)
                         & (segment_positions <= 1 + 1e-9)]
        if not len(hits):
            raise ValueError("Cannot project a calyx grid vertex to the boundary.")
        radius_ratio = min(1.0, 1.0 / float(np.min(hits)))
        patch_vertices[index, 1] += 2 * (1 - radius_ratio * radius_ratio)
    triangles = np.asarray(triangles, dtype=int)
    patch_uvs = values[:, 1:]
    triangles = _unfold_patch_uvs(points, patch_uvs, triangles, sign, tolerance)
    return patch_vertices, patch_uvs, triangles, center_v_id


def _write_textured_obj(path, vertices, faces, uvs, uv_faces, material_name):
    used_vertices = np.unique(faces)
    used_uvs = np.unique(uv_faces)
    vertex_map = np.full(len(vertices), -1, dtype=int)
    uv_map = np.full(len(uvs), -1, dtype=int)
    vertex_map[used_vertices] = np.arange(1, len(used_vertices) + 1)
    uv_map[used_uvs] = np.arange(1, len(used_uvs) + 1)
    with path.open("w", encoding="utf-8", newline="\n") as out:
        out.write(f"mtllib {shlex.quote(path.with_suffix('.mtl').name)}\n")
        for point in vertices[used_vertices]:
            out.write("v " + " ".join(f"{value:.12g}" for value in point) + "\n")
        for uv in uvs[used_uvs]:
            out.write(f"vt {uv[0]:.12g} {uv[1]:.12g}\n")
        out.write(f"usemtl {material_name}\n")
        for face, uv_face in zip(faces, uv_faces):
            out.write("f " + " ".join(f"{vertex_map[v]}/{uv_map[t]}"
                                        for v, t in zip(face, uv_face)) + "\n")


def remove_calyx(input_path: Path, output_path: Path, seam_path: Path | None = None) -> Path:
    """Write the patched OBJ and a separate textured ``<output>_calyx.obj``."""

    # First run patch_stem_and_ring_calyx to add the calyx ring and stem
    input_path, output_path = Path(input_path), Path(output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input mesh not found: {input_path}")
    if input_path.suffix.lower() != ".obj" or output_path.suffix.lower() != ".obj":
        raise ValueError("Input and output meshes must be OBJ files.")
    seam_path = Path(seam_path) if seam_path is not None else input_path.with_name(
        input_path.stem + "_seam_topology.json"
    )
    # This input mesh and topology should have the stem ring and the calys ring
    # Do this so it doesn't duplicate vertices
    lines, uvs, uv_faces, material_path = _read_obj(input_path)
    vertices, faces = _vertex_geometry(lines, uv_faces, input_path)
    texture_path = _read_texture(material_path)
    if np.any(uvs < 0) or np.any(uvs > 1):
        print(f"WARNING Min and max UV coordinates: {uvs.min()}, {uvs.max()} clipped")
        uvs = np.clip(uvs, 0, 1)
    seam_file = seam_path / (input_path.stem + "_seam_topology.json")
    with seam_file.open(encoding="utf-8") as stream:
        topology = SeamTopology.from_dict(json.load(stream))
    if "calyx" not in topology.seams:
        raise ValueError("Seam topology must contain a calyx ring.")
    seam = topology.seams["calyx"]
    _calyx_uv_candidates(lines, uv_faces, uvs, seam, input_path)
    ring, inside, center_vertex = _calyx_faces(vertices, faces, seam, topology.bottom_point)
    seam_entries = {vertex: (location, uv) for vertex, location, uv in
                    zip(seam.vertex_ids, seam.vertex_locations, seam.uv_coords)}
    seam.vertex_ids = ring
    seam.vertex_locations = [seam_entries[vertex][0] for vertex in ring]
    seam.uv_coords = [seam_entries[vertex][1] for vertex in ring]
    patch_vertices, patch_uvs, patch_faces, center_vid = _grid_patch(
        vertices, faces, uv_faces, uvs, ring, inside, center_vertex, seam.center
    )
    topology.bottom_vertex = center_vid
    
    saved_uvs = dict(zip(seam.vertex_ids, seam.uv_coords))
    for vertex, uv in zip(ring, patch_uvs):
        if not np.any(np.linalg.norm(np.asarray(saved_uvs[vertex]) - uv, axis=1) <= 2e-6):
            raise ValueError(f"Saved calyx UV for vertex {vertex} does not match the calyx side of the OBJ.")
    material_lines = material_path.read_text(encoding="utf-8").splitlines()
    material_names = [shlex.split(line, comments=True)[1] for line in material_lines
                      if shlex.split(line, comments=True)[:1] == ["newmtl"]]
    material_name = material_names[0]
    used_materials = {shlex.split(line, comments=True)[1] for line in lines
                      if shlex.split(line, comments=True)[:1] == ["usemtl"]}
    if used_materials != {material_name}:
        raise ValueError("All OBJ faces must use the single diffuse material.")

    calyx_sub_path = output_path.parent / "Calyx"
    calyx_path = calyx_sub_path.with_name(output_path.stem + "_calyx.obj")
    output_seams = calyx_sub_path.with_name(output_path.stem + "_seam_topology.json")
    body_texture = calyx_sub_path.with_suffix(texture_path.suffix)
    calyx_texture = calyx_sub_path.with_suffix(".png")
    outputs = [calyx_sub_path, calyx_sub_path.with_suffix(".mtl"), body_texture,
               calyx_path, calyx_path.with_suffix(".mtl"), calyx_texture, output_seams]
    sources = {p.resolve() for p in (input_path, seam_path, material_path, texture_path)}
    if len({p.resolve() for p in outputs}) != len(outputs) or any(p.resolve() in sources for p in outputs):
        raise ValueError("Output files must be distinct and must not overwrite input files.")

    calyx_uv_faces = uv_faces[inside]
    selected_uvs = uvs[np.unique(calyx_uv_faces)]
    lower = selected_uvs.min(axis=0)
    upper = selected_uvs.max(axis=0)
    extent = float(np.max(upper - lower))
    if extent <= 0:
        raise ValueError("Calyx UVs have zero extent.")
    # Eight output pixels of padding, with a uniform scale to avoid distortion.
    side = extent * 256 / 240
    origin = (lower + upper - side) / 2
    cropped_uvs = (uvs - origin) / side
    with Image.open(texture_path) as image:
        width, height = image.size
        transform = (side * width / 256, 0, origin[0] * width,
                     0, side * height / 256, (1 - origin[1] - side) * height)
        mode = "RGBA" if "A" in image.getbands() or "transparency" in image.info else "RGB"
        cropped = image.convert(mode).transform((256, 256), Image.Transform.AFFINE, transform,
                                                resample=Image.Resampling.BICUBIC)

    keep = np.ones(len(faces), dtype=bool)
    keep[inside] = False
    patch_ids = np.concatenate((np.asarray(ring), np.arange(len(vertices), len(vertices) + len(patch_vertices) - len(ring))))
    all_vertices = np.vstack((vertices, patch_vertices[len(ring):]))
    all_uvs = np.vstack((uvs, patch_uvs))
    body_faces = np.vstack((faces[keep], patch_ids[patch_faces]))
    body_uv_faces = np.vstack((uv_faces[keep], len(uvs) + patch_faces))
    used_vertices = np.unique(body_faces)
    vertex_map = {int(vertex): index for index, vertex in enumerate(used_vertices)}
    for saved_seam in topology.seams.values():
        if any(vertex not in vertex_map for vertex in saved_seam.vertex_ids):
            raise ValueError(f"The {saved_seam.name} seam contains vertices removed with the calyx.")
    for boundary in topology.boundary_paths:
        if any(vertex not in vertex_map for vertex in boundary["vertex_ids"]):
            raise ValueError("A boundary path contains vertices removed with the calyx.")
    topology.renumber_seams(vertex_map)
    topology.bottom_point = patch_vertices[len(ring)].tolist()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(texture_path, body_texture)
    cropped.save(calyx_texture)
    for path, texture in ((output_path, body_texture), (calyx_path, calyx_texture)):
        with path.with_suffix(".mtl").open("w", encoding="utf-8", newline="\n") as stream:
            for line in material_lines:
                parts = shlex.split(line, comments=True)
                stream.write(f"map_Kd {shlex.quote(texture.name)}\n" if parts[:1] == ["map_Kd"] else line + "\n")
    _write_textured_obj(output_path, all_vertices, body_faces, all_uvs, body_uv_faces, material_name)
    _write_textured_obj(calyx_path, vertices, faces[inside], cropped_uvs, calyx_uv_faces, material_name)

    with output_seams.open("w", encoding="utf-8") as stream:
        json.dump(topology.to_dict(), stream)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, default=None, help="Directory the paths are relative to")
    parser.add_argument("--input-path", type=Path, required=True, help="Apple scan mesh")
    parser.add_argument("--output-path", type=Path, required=True, help="Patched mesh to write")
    parser.add_argument("--seam-path", type=Path, help="Calyx seam topology JSON (default: companion JSON)")
    args = parser.parse_args()
    input_path, output_path = args.input_path, args.output_path
    if args.root_dir is not None:
        input_path, output_path = args.root_dir / input_path, args.root_dir / output_path
    seam_path = args.seam_path
    if seam_path is not None and args.root_dir is not None:
        seam_path = args.root_dir / seam_path
    print(f"Wrote {remove_calyx(input_path, output_path, seam_path)}")


if __name__ == "__main__":
    main()
