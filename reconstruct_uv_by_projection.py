"""Transfer a textured OBJ onto a nearby mesh and reconstruct its UV seams."""

import shlex
import shutil
from pathlib import Path

import numpy as np
from meshlib import mrmeshnumpy as mn
from meshlib import mrmeshpy as mm
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from align_meshes import _SurfaceLocator, _mesh
from obj_geometry import read_texture
from seam_topology import SeamInfo, SeamTopology


_UV_TOLERANCE = 2e-6


def _edge_faces(faces):
    edges = {}
    for face_id, face in enumerate(faces):
        for corner in range(3):
            a, b = int(face[corner]), int(face[(corner + 1) % 3])
            edges.setdefault(tuple(sorted((a, b))), []).append(face_id)
    if any(len(incident) > 2 for incident in edges.values()):
        raise ValueError("UV reconstruction requires a manifold mesh.")
    return edges


def _edge_uv(face, uvs, vertex):
    return uvs[np.flatnonzero(face == vertex)[0]]


def _source_charts(vertices, faces, face_uvs, topology):
    rows, cols = [], []
    for edge, incident in _edge_faces(faces).items():
        if len(incident) != 2:
            continue
        first, second = incident
        if all(np.linalg.norm(_edge_uv(faces[first], face_uvs[first], v)
                              - _edge_uv(faces[second], face_uvs[second], v)) <= _UV_TOLERANCE
               for v in edge):
            rows.extend((first, second))
            cols.extend((second, first))
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(faces), len(faces)))
    count, components = connected_components(graph, directed=False)
    if count != 3:
        raise ValueError(f"Expected three connected source UV charts; found {count}.")
    top = np.asarray(topology.top_point, dtype=float)
    bottom = np.asarray(topology.bottom_point, dtype=float)
    axis = top - bottom
    if top.shape != (3,) or bottom.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) == 0:
        raise ValueError("Seam topology must contain distinct finite top and bottom points.")
    heights = vertices[faces].mean(axis=1) @ axis
    means = [float(np.mean(heights[components == chart])) for chart in range(3)]
    order = np.argsort(means)
    labels = np.empty(3, dtype=int)
    labels[order[2]], labels[order[0]], labels[order[1]] = 0, 1, 2
    return labels[components]


def _ordered_path(edges, *, closed, start=None):
    neighbors = {}
    for a, b in edges:
        neighbors.setdefault(a, []).append(b)
        neighbors.setdefault(b, []).append(a)
    if not neighbors:
        raise ValueError("Could not reconstruct a required UV seam.")
    ends = [vertex for vertex, adjacent in neighbors.items() if len(adjacent) == 1]
    if (closed and any(len(adjacent) != 2 for adjacent in neighbors.values())
            or not closed and (len(ends) != 2 or any(len(adjacent) not in (1, 2) for adjacent in neighbors.values()))):
        raise ValueError("Reconstructed UV seam is not a simple loop or path.")
    start = min(neighbors) if closed and start is None else (min(ends) if start is None else start)
    if start not in neighbors or (not closed and start not in ends):
        raise ValueError("UV seam endpoints do not meet the cap seams.")
    path = [start]
    previous = None
    while True:
        choices = sorted(v for v in neighbors[path[-1]] if v != previous)
        if not choices:
            break
        following = choices[0]
        if following == start and closed:
            break
        if following in path:
            raise ValueError("Reconstructed UV seam contains a cycle or branch.")
        previous, current = path[-1], following
        path.append(current)
    if len(path) != len(neighbors):
        raise ValueError("Reconstructed UV seam has disconnected components.")
    return path


def _merge_vertex_uvs(faces, face_uvs):
    by_vertex = {}
    uvs = []
    uv_faces = np.empty_like(faces)
    for face_id, face in enumerate(faces):
        for corner, vertex in enumerate(face):
            uv = face_uvs[face_id, corner]
            indices = by_vertex.setdefault(int(vertex), [])
            index = next((i for i in indices if np.linalg.norm(uvs[i] - uv) <= _UV_TOLERANCE), None)
            if index is None:
                index = len(uvs)
                uvs.append(uv.copy())
                indices.append(index)
            uv_faces[face_id, corner] = index
    return np.asarray(uvs), uv_faces


def _triangle_weights(point, triangle):
    a = triangle[1] - triangle[0]
    b = triangle[2] - triangle[0]
    delta = point - triangle[0]
    aa, ab, bb = a @ a, a @ b, b @ b
    determinant = aa * bb - ab * ab
    if determinant > 1e-20:
        s = ((delta @ a) * bb - (delta @ b) * ab) / determinant
        t = ((delta @ b) * aa - (delta @ a) * ab) / determinant
        if s >= 0 and t >= 0 and s + t <= 1:
            return np.array([1 - s - t, s, t])
    best_distance = np.inf
    best = None
    for first, second in ((0, 1), (1, 2), (2, 0)):
        edge = triangle[second] - triangle[first]
        length = edge @ edge
        fraction = float(np.clip((point - triangle[first]) @ edge / length, 0, 1)) if length > 0 else 0
        distance = np.linalg.norm(point - triangle[first] - fraction * edge)
        if distance < best_distance:
            best_distance = distance
            best = np.zeros(3)
            best[first], best[second] = 1 - fraction, fraction
    return best


def _reconstruct_seams(vertices, faces, face_uvs, charts, topology):
    edge_faces = _edge_faces(faces)
    ring_edges = [[], []]
    side_edges = []
    for edge, incident in edge_faces.items():
        if len(incident) != 2:
            continue
        a, b = incident
        pair = {int(charts[a]), int(charts[b])}
        if pair == {0, 2}:
            ring_edges[0].append(edge)
        elif pair == {1, 2}:
            ring_edges[1].append(edge)
        elif pair == {0, 1}:
            raise ValueError("Top and bottom UV charts meet without a body chart.")
        elif pair == {2} and any(
            np.linalg.norm(_edge_uv(faces[a], face_uvs[a], vertex)
                           - _edge_uv(faces[b], face_uvs[b], vertex)) > _UV_TOLERANCE
            for vertex in edge
        ):
            side_edges.append(edge)

    side = _ordered_path(side_edges, closed=False)
    top_vertices = set(v for edge in ring_edges[0] for v in edge)
    bottom_vertices = set(v for edge in ring_edges[1] for v in edge)
    if side[0] in bottom_vertices and side[-1] in top_vertices:
        side.reverse()
    if side[0] not in top_vertices or side[-1] not in bottom_vertices:
        raise ValueError("The reconstructed side seam must connect the top and bottom seams.")
    rings = [_ordered_path(ring_edges[i], closed=True, start=side[0 if i == 0 else -1])
             for i in range(2)]

    copies = [[] for _ in vertices]
    chart_copies = [[[] for _ in range(3)] for _ in vertices]
    for face, face_uv, chart in zip(faces, face_uvs, charts):
        for vertex, uv in zip(face, face_uv):
            for collection in (copies[vertex], chart_copies[vertex][chart]):
                if not any(np.linalg.norm(uv - other) <= _UV_TOLERANCE for other in collection):
                    collection.append(uv)

    def chart_uv(vertex, chart):
        values = chart_copies[vertex][chart]
        if len(values) != 1:
            raise ValueError(f"UV seam vertex {vertex} has ambiguous chart coordinates.")
        return values[0].tolist()

    # Follow the same body chart copy along each side of the cut.
    first_edge = tuple(sorted(side[:2]))
    first_faces = edge_faces[first_edge]
    side_paths = []
    for first_face in first_faces:
        coords = [_edge_uv(faces[first_face], face_uvs[first_face], side[0])]
        previous_uv = _edge_uv(faces[first_face], face_uvs[first_face], side[1])
        coords.append(previous_uv)
        for a, b in zip(side[1:-1], side[2:]):
            incident = edge_faces[tuple(sorted((a, b)))]
            candidates = [face for face in incident
                          if np.linalg.norm(_edge_uv(faces[face], face_uvs[face], a) - previous_uv)
                          <= _UV_TOLERANCE]
            if len(candidates) != 1:
                raise ValueError("Cannot follow both sides of the reconstructed side seam.")
            previous_uv = _edge_uv(faces[candidates[0]], face_uvs[candidates[0]], b)
            coords.append(previous_uv)
        side_paths.append([uv.tolist() for uv in coords])

    new_seams = {}
    axis = np.asarray(topology.top_point, dtype=float) - np.asarray(topology.bottom_point, dtype=float)
    axis /= np.linalg.norm(axis)
    for i, name in enumerate(("top", "bottom")):
        center = np.asarray(topology.top_point if i == 0 else topology.bottom_point, dtype=float)
        offsets = vertices[rings[i]] - center
        radius = float(np.mean(np.linalg.norm(offsets - np.outer(offsets @ axis, axis), axis=1)))
        new_seams[name] = SeamInfo(name, rings[i], vertices[rings[i]].tolist(),
                                   [[uv.tolist() for uv in copies[v]] for v in rings[i]],
                                   tuple(center), radius)
    new_seams["side"] = SeamInfo("side", side, vertices[side].tolist(),
                                  [[uv.tolist() for uv in copies[v]] for v in side], None, None)
    for name in ("calyx", "stem"):
        if name in topology.seams:
            seam = topology.seams[name]
            if any(v < 0 or v >= len(vertices) or not copies[v] for v in seam.vertex_ids):
                raise ValueError(f"{name} seam contains invalid or unreferenced target vertices.")
            seam.vertex_locations = vertices[seam.vertex_ids].tolist()
            seam.uv_coords = [[uv.tolist() for uv in copies[v]] for v in seam.vertex_ids]

    top, bottom = rings
    records = [
        (top[1:], [chart_uv(v, 0) for v in top[1:]]),
        (bottom[1:], [chart_uv(v, 1) for v in bottom[1:]]),
        (top[1:], [chart_uv(v, 2) for v in top[1:]]),
        (side, side_paths[0]),
        (list(reversed(bottom[1:])), [chart_uv(v, 2) for v in reversed(bottom[1:])]),
        (list(reversed(side)), list(reversed(side_paths[1]))),
    ]
    topology.set_boundary_paths(records)
    topology.seams.update(new_seams)


def reconstruct_uv_by_projection(source_mesh: Path, target_mesh: mm.Mesh,
                                 seam_topology: SeamTopology, output_path: Path) -> Path:
    """Project source OBJ UVs onto target faces, approximating seams along target edges."""
    source_mesh, output_path = Path(source_mesh), Path(output_path)
    if source_mesh.suffix.lower() != ".obj" or output_path.suffix.lower() != ".obj":
        raise ValueError("Source and output meshes must be OBJ files.")
    lines, source_vertices, source_faces, uvs, uv_faces, material = _mesh(source_mesh)
    texture = _read_texture(material)
    output_material = output_path.with_suffix(".mtl")
    output_texture = output_path.with_suffix(texture.suffix)
    sources = {path.resolve() for path in (source_mesh, material, texture)}
    outputs = [output_path, output_material, output_texture]
    if len({path.resolve() for path in outputs}) != len(outputs) or any(
        path.resolve() in sources for path in outputs
    ):
        raise ValueError("Output files must not overwrite the source mesh, material, or texture.")
    material_lines = material.read_text(encoding="utf-8").splitlines()
    material_names = [parts[1] for line in material_lines
                      if (parts := shlex.split(line, comments=True)) and parts[0] == "newmtl"]
    used_materials = {parts[1] for line in lines
                      if (parts := shlex.split(line, comments=True)) and parts[0] == "usemtl"}
    if len(material_names) != 1 or used_materials != set(material_names):
        raise ValueError("All source OBJ faces must use the single diffuse material.")

    vertices = mn.getNumpyVerts(target_mesh).astype(np.float64)
    faces = mn.getNumpyFaces(target_mesh.topology).astype(np.int64)
    if not len(vertices) or not len(faces) or not np.isfinite(vertices).all():
        raise ValueError("Target mesh must contain finite vertices and triangular faces.")
    source_uvs = uvs[uv_faces]
    source_charts = _source_charts(source_vertices, source_faces, source_uvs, seam_topology)
    locator = _SurfaceLocator(source_vertices, source_faces, source_charts)
    all_locator = _SurfaceLocator(source_vertices, source_faces, np.zeros(len(source_faces), dtype=int))
    charts = np.empty(len(faces), dtype=int)
    face_uvs = np.empty((len(faces), 3, 2), dtype=float)
    for index, face in enumerate(faces):
        points = vertices[face]
        center = points.mean(axis=0)
        source_face, _ = all_locator.locate(center, 0, 0)
        chart = int(source_charts[source_face])
        charts[index] = chart
        for corner, point in enumerate(points):
            # The small interior offset selects the correct copy at a UV cut.
            probe = point + 1e-3 * (center - point)
            match, _ = locator.locate(probe, chart, source_face)
            weights = _triangle_weights(point, source_vertices[source_faces[match]])
            face_uvs[index, corner] = weights @ source_uvs[match]
    output_uvs, output_uv_faces = _merge_vertex_uvs(faces, face_uvs)
    face_uvs = output_uvs[output_uv_faces]
    _reconstruct_seams(vertices, faces, face_uvs, charts, seam_topology)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(texture, output_texture)
    with output_material.open("w", encoding="utf-8", newline="\n") as stream:
        for line in material_lines:
            parts = shlex.split(line, comments=True)
            stream.write(f"map_Kd {shlex.quote(output_texture.name)}\n" if parts[:1] == ["map_Kd"] else line + "\n")
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(f"mtllib {shlex.quote(output_material.name)}\n")
        for point in vertices:
            stream.write("v " + " ".join(f"{value:.12g}" for value in point) + "\n")
        for uv in output_uvs:
            stream.write(f"vt {uv[0]:.12g} {uv[1]:.12g}\n")
        stream.write(f"usemtl {shlex.quote(material_names[0])}\n")
        for face, face_uv in zip(faces, output_uv_faces):
            stream.write("f " + " ".join(f"{vertex + 1}/{uv + 1}"
                                           for vertex, uv in zip(face, face_uv)) + "\n")
    return output_path
