"""Remove an apple calyx, close the opening, and export the removed region (MeshLib).

The calyx is at the bottom (min y) of the apple and is roughly 10-20 units across. The bottom center is
estimated from the lowest vertices near the horizontal center. The calyx region is every face connected to
that center whose vertices lie within a circle in the x-z plane. The region is cut out, the opening is
patched with a smooth surface (edge lengths about the mesh average), and the calyx is saved separately.

Outputs: the patched mesh (output_path), <stem>_calyx.ply and <stem>_calyx_seam.csv next to it.
The mesh is used as loaded; vertices are not welded."""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from meshlib import mrmeshnumpy as mn
from meshlib import mrmeshpy as mm
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


import numpy as np
import pymeshlab
from scipy.sparse import csr_matrix


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
) -> bool:
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
    return b_was_extremum, max_rise


def _high_curvature_vertices(
    mesh: mm.Mesh, 
    vertices: np.ndarray,
    output_path: Path
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Identify vertices that are local minima or maxima in the y direction. For each vertex, grow two neighborhoods,
    one that grows upwards in the y direction and one that grows downwards. Identify vertices that are local extrema within these neighborhoods.
    The total neighborhood height around the vertex should be at least 0.1 units."""

    # y bounds
    y_min = np.min(vertices[:, 1])
    y_max = np.max(vertices[:, 1]) 
    n_vertices = vertices.shape[0] 
    y_clip = y_min + 0.15 * (y_max - y_min)

    vertex_colors = mm.VertColors()
    vertex_colors.resize(mesh.points.size(), mm.Color(155, 135, 125, 255))

    # Build the nearest neighbor structure for the vertices
    vert_adj = [set() for _ in range(n_vertices)]
    for vi in np.flatnonzero(vertices[:, 1] <= y_clip):
        vert_idx = mm.VertId(vi)
        for nbr_idx in _get_vertex_star(mesh.topology, vert_idx):
            vert_adj[vi].add(int(nbr_idx))

    print(f"Mesh y bounds: min={y_min}, max={y_max}")
    # For each vertex in the bottom part of the mesh, make a list of its neighbors that are in the positive y direction and another list
    # of the neighbors in the negative y direction
    v_max = []
    v_min = []
    eps = 0.01
    calyx_len = 3

    for vi in np.flatnonzero(vertices[:, 1] <= y_clip):
        vi = int(vi)
        vertex_colors[mm.VertId(vi)] = mm.Color(255, 255, 255, 255)
        is_extremum, val = _is_y_extremum(vert_adj, vertices[:, 1], vi, find_min=True, eps=eps, min_height=calyx_len)
        if is_extremum:
            v_min.append((vi, val))
            vertex_colors[mm.VertId(vi)] = mm.Color(255, 255, 0, 255)
        else:
            is_extremum, val = _is_y_extremum(vert_adj, vertices[:, 1], vi, find_min=False, eps=eps, min_height=calyx_len)
            if is_extremum:
                v_max.append((vi, val))
                vertex_colors[mm.VertId(vi)] = mm.Color(255, 0, 255, 255)

    print(f"Filtered positive end vertices: {len(v_min)}")
    print(f"Filtered negative end vertices: {len(v_max)}")
    curv_path = output_path.with_name(output_path.stem + "_curvature" + ".ply")

    # Write out a copy of the mesh with vertex colors indicating extremum vertices
    # Save the mesh with vertex colors (use a format that supports colors, like .ply or .obj)
    save_settings = mm.SaveSettings()
    save_settings.colors = vertex_colors
    mm.saveMesh(mesh, str(curv_path), save_settings)
    return v_min, v_max


def _bottom_center(vertices: np.ndarray) -> np.ndarray:
    """Estimate the calyx center from the lowest vertices near the horizontal center."""
    if len(vertices) == 0 or not np.isfinite(vertices).all():
        raise ValueError("Input mesh must have finite vertices.")
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    middle = (low + high) / 2.0
    diameter = float(max(high[0] - low[0], high[2] - low[2]))
    near_axis = np.hypot(vertices[:, 0] - middle[0], vertices[:, 2] - middle[2]) <= 0.25 * diameter
    lower = vertices[:, 1] <= low[1] + 0.2 * (high[1] - low[1])
    candidates = vertices[near_axis & lower]
    if len(candidates) == 0:
        raise ValueError("No vertices found near the bottom center.")
    return np.array([candidates[:, 0].mean(), candidates[:, 1].min(), candidates[:, 2].mean()])


def _calyx_vertices(verts: np.ndarray, faces: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    """Boolean mask of vertices connected to the center through vertices inside the x-z circle."""
    inside = np.hypot(verts[:, 0] - center[0], verts[:, 2] - center[2]) <= radius
    inside &= verts[:, 1] <= center[1] + 2.0 * radius
    seed = int(np.argmin(np.linalg.norm(verts - center, axis=1) + np.where(inside, 0.0, np.inf)))
    if not inside[seed]:
        raise ValueError("No mesh vertex lies inside the calyx circle.")
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = edges[inside[edges[:, 0]] & inside[edges[:, 1]]]
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(verts),) * 2)
    _, label = connected_components(graph, directed=False)
    return inside & (label == label[seed])


def remove_calyx(input_path: Path, output_path: Path) -> Path:
    """Cut the calyx out of the mesh at input_path, patch the hole, and write the result to output_path."""
    input_path, output_path = Path(input_path), Path(output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input mesh not found: {input_path}")

    mesh: mm.Mesh = mm.loadMesh(str(input_path))
    verts = mn.getNumpyVerts(mesh)
    faces = mn.getNumpyFaces(mesh.topology)
    if len(faces) == 0:
        raise ValueError(f"Input mesh has no faces: {input_path}")

    extrema_min, extrema_max = _high_curvature_vertices(mesh=mesh, vertices=verts, output_path=output_path)

    diameter = float(np.ptp(verts[:, [0, 2]], axis=0).max())
    radius = max(min(15.0, 0.15 * diameter), 3.0 * mesh.averageEdgeLength())
    center = _bottom_center(verts)
    in_calyx = _calyx_vertices(verts, faces, center, radius)

    face_mask = in_calyx[faces].all(axis=1)
    if not face_mask.any():
        raise ValueError("No calyx faces found.")
    if face_mask.mean() > 0.5:
        raise ValueError("Calyx region covers most of the mesh; check the input orientation.")
    region = mn.faceBitSetFromBools(face_mask)

    calyx = mm.Mesh(mesh)
    calyx.topology.deleteFaces(mn.faceBitSetFromBools(~face_mask))
    calyx.pack()

    patched = mm.Mesh(mesh)
    patched.topology.deleteFaces(region)

    # Only fill holes created by the cut; the scan may have other boundaries (such as UV chart edges).
    seam_points = []
    settings = mm.FillHoleNicelySettings()
    settings.triangulateOnly = False
    settings.subdivideSettings.maxEdgeLen = 1.5 * float(mesh.averageEdgeLength())
    settings.subdivideSettings.maxEdgeSplits = 1_000_000
    settings.smoothCurvature = True
    for edge in patched.topology.findHoleRepresentiveEdges():
        point = patched.orgPnt(edge)
        if math.hypot(point.x - center[0], point.z - center[2]) > radius * 1.5:
            continue
        ring = patched.topology.getLeftRing(edge)
        seam_points.extend((p.x, p.y, p.z) for p in (patched.orgPnt(e) for e in ring))
        mm.fillHoleNicely(patched, edge, settings)
    if not seam_points:
        raise ValueError("Cutting the calyx did not leave an opening to patch.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stem = output_path.with_suffix("")
    mm.saveMesh(patched, str(output_path))
    mm.saveMesh(calyx, str(stem) + "_calyx.ply")
    with open(str(stem) + "_calyx_seam.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "y", "z"])
        writer.writerows(seam_points)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, default=None, help="Directory the paths are relative to")
    parser.add_argument("--input-path", type=Path, required=True, help="Apple scan mesh")
    parser.add_argument("--output-path", type=Path, required=True, help="Patched mesh to write")
    args = parser.parse_args()
    input_path, output_path = args.input_path, args.output_path
    if args.root_dir is not None:
        input_path, output_path = args.root_dir / input_path, args.root_dir / output_path
    remove_calyx(input_path, output_path)
    print(f"Wrote {remove_calyx(input_path, output_path)}")


if __name__ == "__main__":
    main()
