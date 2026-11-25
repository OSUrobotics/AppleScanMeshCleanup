"""Extend normalized texture charts with pixels from the opposite side of their seams."""

import argparse
from collections import Counter, defaultdict
import heapq
import json
import os
import shlex
import shutil
import tempfile
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from obj_geometry import (
    bfs_graph, compute_edge_locs_circle, compute_edge_locs_line, read_obj, read_texture,
)
from seam_topology import SeamTopology
from normalize_uv_projection import _warp_texture_face
from uv_texture import _CAP_RADIUS, _CAP_CENTERS, _UV_PADDING


_RINGS = 10
_MAX_TOP_SEAM_UV_EDGE = 0.02


def _mesh_adjacency(vertices, face_triangles):
    vertex_faces = [[] for _ in vertices]
    neighbors = [set() for _ in vertices]
    for face_id, face in enumerate(face_triangles):
        for i, vertex in enumerate(face):
            vertex = int(vertex)
            vertex_faces[vertex].append(face_id)
            neighbors[vertex].update((int(face[(i + 1) % 3]), int(face[(i + 2) % 3])))
    adjacency = [
        [(neighbor, float(np.linalg.norm(vertices[vertex] - vertices[neighbor])))
         for neighbor in adjacent]
        for vertex, adjacent in enumerate(neighbors)
    ]
    return vertex_faces, adjacency


def _seam_corner_faces(vertex_faces, seam_ids):
    side_faces = {face for vertex in seam_ids["side"] for face in vertex_faces[vertex]}
    cap_faces = {
        face for name in ("top", "bottom") for vertex in seam_ids[name] for face in vertex_faces[vertex]
    }
    return side_faces & cap_faces


def _expand_ring_faces(face_triangles, edge_adj, start_vertices, n, allowed_faces=None):
    """Return faces within n vertex rings of a seam, restricted to one chart."""
    allowed = None if allowed_faces is None else set(allowed_faces)
    expanded = set()
    frontier = set(start_vertices)
    for _ in range(n):
        new_faces = {
            face for vertex in frontier for face in edge_adj[vertex]
            if face not in expanded and (allowed is None or face in allowed)
        }
        expanded.update(new_faces)
        frontier = {int(vertex) for face in new_faces for vertex in face_triangles[face]}
        if not frontier:
            break
    return expanded


def _boundary_edges(face_triangles, faces):
    counts = Counter(
        tuple(sorted((int(face[i]), int(face[(i + 1) % 3]))))
        for face_id in faces for face in (face_triangles[face_id],) for i in range(3)
    )
    return {edge for edge, count in counts.items() if count == 1}


def _fill_patch_holes(face_triangles, vertex_faces, faces, grown_faces):
    """Include enclosed faces that a vertex-ring expansion left behind."""
    remaining = grown_faces - faces
    grown_boundary = _boundary_edges(face_triangles, grown_faces)
    while remaining:
        seed = remaining.pop()
        component = {seed}
        frontier = [seed]
        reaches_boundary = False
        while frontier:
            face = face_triangles[frontier.pop()]
            for i in range(3):
                a, b = int(face[i]), int(face[(i + 1) % 3])
                if tuple(sorted((a, b))) in grown_boundary:
                    reaches_boundary = True
                adjacent = set(vertex_faces[a]).intersection(vertex_faces[b], remaining)
                remaining.difference_update(adjacent)
                component.update(adjacent)
                frontier.extend(adjacent)
        if not reaches_boundary:
            faces.update(component)
    return faces


def _shortest_cut(adjacency, start, targets, forbidden):
    distances = {start: 0.0}
    previous = {}
    queue = [(0.0, start)]
    while queue:
        distance, vertex = heapq.heappop(queue)
        if distance != distances[vertex]:
            continue
        if vertex in targets:
            path = [vertex]
            while path[-1] != start:
                path.append(previous[path[-1]])
            return list(reversed(path))
        for neighbor, length in adjacency[vertex]:
            if neighbor in forbidden:
                continue
            new_distance = distance + length
            if new_distance < distances.get(neighbor, float("inf")):
                distances[neighbor] = new_distance
                previous[neighbor] = vertex
                heapq.heappush(queue, (new_distance, neighbor))
    raise ValueError("Cannot connect the seam to the outer ring.")


def _get_outer_boundary(face_triangles, edge_adj, start_vertices, n, *,
                        allowed_faces, vertices, adjacency):
    """Find an ordered outer ring and a shortest cut from the seam to it."""
    seam = list(start_vertices)
    if len(seam) > 1 and seam[0] == seam[-1]:
        seam.pop()
    faces = _expand_ring_faces(face_triangles, edge_adj, seam, n, allowed_faces)
    if not faces:
        raise ValueError("No faces found next to the seam.")
    grown_faces = _expand_ring_faces(face_triangles, edge_adj, seam, n + 1, allowed_faces)
    faces = _fill_patch_holes(face_triangles, edge_adj, faces, grown_faces)
    boundary = _boundary_edges(face_triangles, faces)
    seam_edges = {
        tuple(sorted((seam[i], seam[(i + 1) % len(seam)])))
        for i in range(len(seam))
    }
    if not seam_edges.issubset(boundary):
        raise ValueError("The seam is not a closed boundary of the selected faces.")
    outer_edges = boundary - seam_edges
    neighbors = defaultdict(list)
    for a, b in outer_edges:
        neighbors[a].append(b)
        neighbors[b].append(a)
    if not neighbors or any(len(adjacent) != 2 for adjacent in neighbors.values()):
        raise ValueError("The expanded seam does not have a single closed outer boundary.")
    patch_vertices = {int(v) for face in faces for v in face_triangles[face]}
    patch_edges = {
        tuple(sorted((int(face[i]), int(face[(i + 1) % 3]))))
        for face_id in faces for face in (face_triangles[face_id],) for i in range(3)
    }
    patch_adjacency = [
        [(neighbor, length) for neighbor, length in adjacent
         if tuple(sorted((vertex, neighbor))) in patch_edges]
        if vertex in patch_vertices else []
        for vertex, adjacent in enumerate(adjacency)
    ]
    cut = _shortest_cut(patch_adjacency, seam[0], set(neighbors), set(seam[1:]))
    start = cut[-1]
    first, second = neighbors[start]
    directed_boundary = {
        (int(face[i]), int(face[(i + 1) % 3]))
        for face_id in faces for face in (face_triangles[face_id],) for i in range(3)
        if tuple(sorted((int(face[i]), int(face[(i + 1) % 3])))) in boundary
    }
    seam_forward = (seam[0], seam[1]) in directed_boundary
    if ((start, first) in directed_boundary) == seam_forward:
        first = second
    outer = [start, first]
    while True:
        choices = [vertex for vertex in neighbors[outer[-1]] if vertex != outer[-2]]
        if len(choices) != 1:
            raise ValueError("The outer boundary is not a simple ring.")
        if choices[0] == start:
            break
        if choices[0] in outer:
            raise ValueError("The outer boundary is not a simple ring.")
        outer.append(choices[0])
    if len(outer) != len(neighbors):
        raise ValueError("The expanded seam has multiple outer boundaries.")
    return faces, outer, cut


def _solve_patch(face_nodes, fixed):
    neighbors = defaultdict(set)
    for face in face_nodes:
        for i, node in enumerate(face):
            neighbors[node].update((face[(i + 1) % 3], face[(i + 2) % 3]))
    interior = sorted(set(neighbors) - set(fixed))
    result = dict(fixed)
    if not interior:
        return result
    indices = {node: i for i, node in enumerate(interior)}
    rows, cols, values = [], [], []
    rhs = np.zeros((len(interior), 2))
    for node, row in indices.items():
        rows.append(row)
        cols.append(row)
        values.append(len(neighbors[node]))
        for neighbor in neighbors[node]:
            if neighbor in indices:
                rows.append(row)
                cols.append(indices[neighbor])
                values.append(-1)
            else:
                rhs[row] += fixed[neighbor]
    matrix = coo_matrix((values, (rows, cols)), shape=(len(interior), len(interior))).tocsr()
    solution = spsolve(matrix, rhs)
    if not np.isfinite(solution).all():
        raise ValueError("Could not project the seam patch.")
    result.update({node: solution[row] for node, row in indices.items()})
    return result


def _split_patch(face_triangles, faces, cut):
    """Split the patch at a cut so its closed boundaries can map to lines."""
    face_ids = sorted(faces)
    parent = list(range(3 * len(face_ids)))

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a, b):
        parent[find(a)] = find(b)

    cut_edges = {tuple(sorted(pair)) for pair in zip(cut, cut[1:])}
    edge_corners = defaultdict(list)
    for index, face_id in enumerate(face_ids):
        face = face_triangles[face_id]
        for i in range(3):
            a, b = int(face[i]), int(face[(i + 1) % 3])
            edge_corners[tuple(sorted((a, b)))].append({a: 3 * index + i, b: 3 * index + (i + 1) % 3})
    for edge, sides in edge_corners.items():
        if edge in cut_edges:
            continue
        for side in sides[1:]:
            for vertex in edge:
                union(sides[0][vertex], side[vertex])
    nodes = []
    node_vertices = {}
    for index, face_id in enumerate(face_ids):
        face_nodes = []
        for i, vertex in enumerate(face_triangles[face_id]):
            node = find(3 * index + i)
            node_vertices[node] = int(vertex)
            face_nodes.append(node)
        nodes.append(face_nodes)
    return face_ids, nodes, node_vertices


def _line_patch(vertices, face_triangles, faces, seam, outer, cut, start, end, offset):
    face_ids, face_nodes, node_vertices = _split_patch(face_triangles, faces, cut)
    seam_locs = compute_edge_locs_line(vertices, seam, start, end) + [end]
    outer_locs = compute_edge_locs_line(vertices, outer + [outer[0]], start + offset, end + offset)
    fixed_by_vertex = {v: loc for v, loc in zip(seam[1:], seam_locs[1:-1])}
    fixed_by_vertex.update({v: loc for v, loc in zip(outer[1:], outer_locs[1:-1])})
    node_neighbors = defaultdict(set)
    for nodes in face_nodes:
        for node in nodes:
            node_neighbors[node].update(node_vertices[other] for other in nodes if other != node)
    fixed = {}
    for node, vertex in node_vertices.items():
        if vertex in fixed_by_vertex:
            fixed[node] = fixed_by_vertex[vertex]
        elif vertex in (seam[0], outer[0]):
            ring = seam if vertex == seam[0] else outer
            locs = seam_locs if vertex == seam[0] else outer_locs
            first = ring[1] in node_neighbors[node]
            last = ring[-1] in node_neighbors[node]
            if first == last:
                raise ValueError("Cannot distinguish the two sides of the seam cut.")
            fixed[node] = locs[0] if first else locs[-1]
    projected = _solve_patch(face_nodes, fixed)
    return face_ids, np.asarray([[projected[node] for node in face] for face in face_nodes])


def _circle_patch(vertices, face_triangles, faces, seam, outer, seam_uvs, center):
    face_ids = sorted(faces)
    face_nodes = [[int(v) for v in face_triangles[face]] for face in face_ids]
    fixed = dict(zip(seam, seam_uvs))
    outer_locs = np.asarray(compute_edge_locs_circle(vertices, outer, _CAP_RADIUS * 1.1)) + center
    fixed.update(zip(outer, outer_locs))
    projected = _solve_patch(face_nodes, fixed)
    return face_ids, np.asarray([[projected[node] for node in face] for face in face_nodes])


def _side_seam_line(seam, side_faces, face_triangles, face_uv_triangles, uvs, vertices):
    seam_set = set(seam)
    uv_by_vertex = defaultdict(list)
    for face in side_faces:
        for vertex, uv in zip(face_triangles[face], face_uv_triangles[face]):
            if vertex in seam_set:
                uv_by_vertex[int(vertex)].append(uvs[uv])
    lengths = np.linalg.norm(np.diff(vertices[seam], axis=0), axis=1)
    if lengths.sum() == 0:
        raise ValueError("The side seam has zero length.")
    fractions = np.r_[0.0, np.cumsum(lengths)] / lengths.sum()
    # Corner faces can use the other corner copy; interior seam UVs determine the line.
    points = []
    positions = []
    for index, vertex in enumerate(seam[1:-1], 1):
        if vertex in uv_by_vertex:
            points.append(np.mean(uv_by_vertex[vertex], axis=0))
            positions.append(fractions[index])
    if len(points) < 2:
        raise ValueError("Need at least two interior UVs on each side seam.")
    coefficients, _, _, _ = np.linalg.lstsq(
        np.column_stack((np.ones(len(positions)), positions)), np.asarray(points), rcond=None
    )
    return coefficients[0], coefficients[0] + coefficients[1]


def _side_patch(face_triangles, face_uv_triangles, uvs, vertex_faces, seed_faces,
                allowed_faces, seam, source_line, target_line):
    faces = set(seed_faces)
    frontier = set(seed_faces)
    for _ in range(2 * (_RINGS - 1)):
        vertices = {int(v) for face in frontier for v in face_triangles[face]} - set(seam)
        new = {face for vertex in vertices for face in vertex_faces[vertex]
               if face in allowed_faces and face not in faces}
        faces.update(new)
        frontier = new
    source_start, source_end = source_line
    target_start, target_end = target_line
    source_direction = source_end - source_start
    target_direction = target_end - target_start
    source_length = np.linalg.norm(source_direction)
    target_length = np.linalg.norm(target_direction)
    if source_length == 0 or target_length == 0:
        raise ValueError("A side seam has zero UV length.")
    source_tangent = source_direction / source_length
    target_tangent = target_direction / target_length
    source_normal = np.array([-source_tangent[1], source_tangent[0]])
    target_normal = np.array([-target_tangent[1], target_tangent[0]])
    chart_center = np.array([0.5, 0.5])
    if np.dot(chart_center - (source_start + source_end) / 2, source_normal) < 0:
        source_normal = -source_normal
    if np.dot(chart_center - (target_start + target_end) / 2, target_normal) > 0:
        target_normal = -target_normal
    face_ids = sorted(faces)
    source_uv = uvs[face_uv_triangles[face_ids]]
    delta = source_uv - source_start
    projected = (target_start + (delta @ source_tangent)[..., None] * target_tangent * (target_length / source_length)
                 + (delta @ source_normal)[..., None] * target_normal)
    return face_ids, projected


def _copy_projected_faces(source, output, occupied, source_pixels, face_uv_triangles,
                          face_ids, projected_uvs, scratch, scratch_covered, excluded_faces):
    height, width = source.shape[:2]
    b_bad_length = width * 0.2
    for face_id, projected in zip(face_ids, projected_uvs):
        if face_id in excluded_faces:
            continue
        target_pixels = projected * [width - 1, height - 1]
        target_pixels[:, 1] = height - 1 - target_pixels[:, 1]
        minimum = np.maximum(np.floor(target_pixels.min(axis=0)).astype(int), 0)
        maximum = np.minimum(np.ceil(target_pixels.max(axis=0)).astype(int), [width - 1, height - 1])
        if np.any(minimum > maximum):
            continue
        region = np.s_[minimum[1]:maximum[1] + 1, minimum[0]:maximum[0] + 1]
        _warp_texture_face(np.arange(3), target_pixels, source_pixels[face_uv_triangles[face_id]],
                           source, scratch, scratch_covered, b_bad_length)
        eligible = scratch_covered[region] & ~occupied[region]
        output[region][eligible] = scratch[region][eligible]
        occupied[region] |= eligible
        scratch_covered[region] = False


def _exclude_long_top_seam_faces(face_ids, projected_uvs, excluded_faces):
    edge_lengths = np.linalg.norm(projected_uvs - np.roll(projected_uvs, -1, axis=1), axis=2)
    long_faces = {
        face for face, lengths in zip(face_ids, edge_lengths)
        if face not in excluded_faces and np.any(lengths > _MAX_TOP_SEAM_UV_EDGE)
    }
    if long_faces:
        warnings.warn(
            f"Skipping {len(long_faces)} projected top-seam faces with UV edges longer than "
            f"{_MAX_TOP_SEAM_UV_EDGE}.",
            RuntimeWarning,
            stacklevel=2,
        )
    return excluded_faces | long_faces


def _fill_in_blank_pixels_with_closest_color(output, occupied):
    pad = 10
    for indx in range(0, 5):  # Repeat the process a few times to fill in more pixels
        occupied_boundary = occupied.copy()
        shift_left = occupied[:, :-1] 
        shift_right = occupied[:, 1:]
        shift_up = occupied[:-1, :]
        shift_down = occupied[1:, :]
        occupied_boundary[:, 1:] = occupied_boundary[:, 1:] | shift_left
        occupied_boundary[:, :-1] = occupied_boundary[:, :-1] | shift_right
        occupied_boundary[1:, :] = occupied_boundary[1:, :] | shift_up
        occupied_boundary[:-1, :] = occupied_boundary[:-1, :] | shift_down
        occupied_boundary = occupied_boundary & ~occupied
        print(f"Iteration {indx+1}, occupied pixels: {np.sum(occupied)}")
        # Get the x,y values of the non zero pixels in the occupied boundary
        coords = np.argwhere(occupied_boundary)
        for pix in coords:
            y, x = pix
            if x < pad or x >= output.shape[1] - pad or y < pad or y >= output.shape[0] - pad:
                continue
            for xp in range(x-1, x+2):
                for yp in range(y-1, y+2):
                    if occupied[yp, xp]:
                        pix_color = output[yp, xp, :]
            for fy in range(y - pad, y + pad):
                for fx in range(x-pad, x+pad):
                    if not occupied[fy, fx]:
                        occupied[fy, fx] = True
                        output[fy, fx, :] = pix_color
    col = np.mean(output[occupied, :], axis=0)
    coords = np.argwhere(~occupied)
    for y, x in coords:
        output[y, x, :] = col


def _expand_pixels(mesh_path, texture_path, seam_path, mesh_data=None):
    vertices, uvs, _, _, faces, face_uvs, _, _ = (
        read_obj(mesh_path) if mesh_data is None else mesh_data
    )
    with seam_path.open(encoding="utf-8") as stream:
        topology = SeamTopology.from_dict(json.load(stream))
    seam_ids = {name: list(topology.seams[name].vertex_ids) for name in ("top", "bottom", "side")}
    vertex_faces, adjacency = _mesh_adjacency(vertices, faces)
    excluded_faces = _seam_corner_faces(vertex_faces, seam_ids)
    blocked = set().union(*map(set, seam_ids.values()))
    cap_vertices = {}
    for name in ("top", "bottom"):
        center = topology.top_vertex if name == "top" else topology.bottom_vertex
        if center < 0:
            center = int(np.argmax(vertices[:, 1]) if name == "top" else np.argmin(vertices[:, 1]))
        cap_vertices[name] = set(bfs_graph(adjacency, center, blocked)) | set(seam_ids[name])
    regions = {name: set(topology.face_lists[name]) for name in ("top", "bottom", "middle")}
    for name in ("top", "bottom"):
        if any(not set(faces[face]).issubset(cap_vertices[name]) for face in regions[name]):
            raise ValueError(f"The {name} face list disagrees with the seam-separated cap.")

    with Image.open(texture_path) as image:
        mode = "RGBA" if "A" in image.getbands() or "transparency" in image.info else "RGB"
        source = np.asarray(image.convert(mode))
    height, width = source.shape[:2]
    if min(height, width) < 2:
        raise ValueError("The texture must be at least 2 by 2 pixels.")
    source_pixels = uvs * [width - 1, height - 1]
    source_pixels[:, 1] = height - 1 - source_pixels[:, 1]
    chart_image = Image.new("1", (width, height))
    draw = ImageDraw.Draw(chart_image)
    for face in face_uvs:
        draw.polygon([tuple(source_pixels[uv]) for uv in face], fill=1)
    occupied = np.asarray(chart_image, dtype=bool).copy()
    output = source.copy()
    scratch = np.zeros_like(source)
    scratch_covered = np.zeros((height, width), dtype=bool)

    corners = np.array([[0.5, _UV_PADDING], [1 - _UV_PADDING, 0.5],
                        [0.5, 1 - _UV_PADDING], [_UV_PADDING, 0.5]])
    for cap_index, name in enumerate(("top", "bottom")):
        seam = seam_ids[name]
        cap_uvs = {}
        for vertex in seam:
            choices = {int(uv) for face in vertex_faces[vertex] if face in regions[name]
                       for v, uv in zip(faces[face], face_uvs[face]) if v == vertex}
            if len(choices) != 1:
                raise ValueError(f"Expected one {name} cap UV for seam vertex {vertex}.")
            cap_uvs[vertex] = uvs[choices.pop()]
        patch, outer, _ = _get_outer_boundary(
            faces, vertex_faces, seam, _RINGS, allowed_faces=regions["middle"],
            vertices=vertices, adjacency=adjacency)
        ids, projected = _circle_patch(vertices, faces, patch, seam, outer,
                                       [cap_uvs[v] for v in seam], _CAP_CENTERS[cap_index])
        _copy_projected_faces(source, output, occupied, source_pixels, face_uvs,
                              ids, projected, scratch, scratch_covered, excluded_faces)
        patch, outer, cut = _get_outer_boundary(
            faces, vertex_faces, seam, _RINGS, allowed_faces=regions[name],
            vertices=vertices, adjacency=adjacency)
        start, end = (corners[0], corners[1]) if name == "top" else (corners[3], corners[2])
        direction = end - start
        normal = np.array([direction[1], -direction[0]])
        normal /= np.linalg.norm(normal)
        if np.dot(normal, (start + end) / 2 - [0.5, 0.5]) < 0:
            normal = -normal
        ids, projected = _line_patch(vertices, faces, patch, seam, outer, cut, start, end, normal * 0.01)
        patch_excluded = (_exclude_long_top_seam_faces(ids, projected, excluded_faces)
                          if name == "top" else excluded_faces)
        _copy_projected_faces(source, output, occupied, source_pixels, face_uvs,
                              ids, projected, scratch, scratch_covered, patch_excluded)

    sides = [set(topology.face_lists[name]) - excluded_faces
             for name in ("left_side_seam", "right_side_seam")]
    side_lines = [_side_seam_line(seam_ids["side"], side, faces, face_uvs, uvs, vertices) for side in sides]
    for index in range(2):
        ids, projected = _side_patch(faces, face_uvs, uvs, vertex_faces, sides[index],
                                     regions["middle"] - excluded_faces, seam_ids["side"],
                                     side_lines[index], side_lines[1 - index])
        _copy_projected_faces(source, output, occupied, source_pixels, face_uvs,
                              ids, projected, scratch, scratch_covered, excluded_faces)

    _fill_in_blank_pixels_with_closest_color(output, occupied)
    return output


def flood_fill_texture_map(input_path: Path, mesh_path: Path, seam_path: Path, output_path: Path) -> Path:
    """Extend a normalized mesh's texture, optionally replacing the input texture."""
    input_path, mesh_path, seam_path, output_path = map(Path, (input_path, mesh_path, seam_path, output_path))
    if seam_path.suffix.lower() != ".json":
        raise ValueError("Expected a seam topology JSON for the normalized mesh.")
    if output_path.resolve() in {mesh_path.resolve(), seam_path.resolve()}:
        raise ValueError("The output texture must not overwrite the mesh or seam file.")
    mesh_data = read_obj(mesh_path)
    material_path = mesh_path.parent / mesh_data[-1]
    if output_path.resolve() == material_path.resolve():
        raise ValueError("The output texture must not overwrite the mesh material.")
    pixels = _expand_pixels(mesh_path, input_path, seam_path, mesh_data)
    if output_path.suffix.lower() in (".jpg", ".jpeg"):
        warnings.warn("Saving as JPEG can change existing chart pixels; use PNG to preserve them exactly.",
                      RuntimeWarning, stacklevel=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output_path.parent, suffix=output_path.suffix, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        Image.fromarray(pixels).save(temporary_path)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path


def expand_texture_map(input_path: Path, output_path: Path) -> Path:
    """Extend cap, body, and side charts across their seams without overwriting charts."""
    input_path, output_path = Path(input_path), Path(output_path)
    if output_path.suffix.lower() != ".obj":
        raise ValueError("The output mesh must be an OBJ file.")
    seam_path = input_path.with_name(input_path.stem + "_seam_topology.json")
    mesh_data = read_obj(input_path)
    material_name = mesh_data[-1]
    material_path = input_path.parent / material_name
    texture_path = read_texture(material_path)
    output_material = output_path.with_suffix(".mtl")
    output_texture = output_path.with_suffix(".jpg")
    output_seam = output_path.with_name(output_path.stem + "_seam_topology.json")
    if {p.resolve() for p in (output_path, output_material, output_texture, output_seam)} & {
        p.resolve() for p in (input_path, material_path, texture_path, seam_path)
    }:
        raise ValueError("Output files must not overwrite the input mesh, material, texture, or seams.")
    output = _expand_pixels(input_path, texture_path, seam_path, mesh_data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output).save(output_texture)
    with material_path.open(encoding="utf-8") as source_material, output_material.open("w", encoding="utf-8") as target:
        for line in source_material:
            target.write(f"map_Kd {shlex.quote(output_texture.name)}\n" if line.split()[:1] == ["map_Kd"] else line)
    with input_path.open(encoding="utf-8") as source_obj, output_path.open("w", encoding="utf-8") as target:
        for line in source_obj:
            target.write(f"mtllib {shlex.quote(output_material.name)}\n" if line.split()[:1] == ["mtllib"] else line)
    shutil.copyfile(seam_path, output_seam)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", default=Path("/Users/cindygrimm/PycharmProjects/data/ScannedModels/ProsserEnvyApples"), type=Path)
    parser.add_argument("--input-path", default=Path("NormalizedTexture/Envy_005_FullRes.obj"), type=Path)
    parser.add_argument("--output-path", default=Path("FixedTexture/Envy_005_FullRes.obj"), type=Path)
    args = parser.parse_args()
    expand_texture_map(args.root_dir / args.input_path, args.root_dir / args.output_path)


if __name__ == "__main__":
    main()
