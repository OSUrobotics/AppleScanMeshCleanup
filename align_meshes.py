""" Putting a mesh into correspondence with a canonical mesh
Input:
    - mesh: The input mesh to be aligned.
    - canonical_mesh: The reference canonical mesh.
    - seam files, that define the top, bottom, and side seams of both meshes
    - a text file that has the two polar endpoint locations on the original mesh

Output:
    - correspondence: A mapping from vertices of the input mesh to the canonical mesh.
    - The canonical mesh re-shaped to be the same shape as the input mesh

Objective: The top conical polar vertex should go to the point on the mesh that is closest to the point given in the endpoint file
The bottom conical polar vertex should go to the point on the mesh that is closest to the other point given in the endpoint file.
The seams should go to the corresponding seams on the canonical mesh, preserving the alignment defined by the seam files.

There are five cone peaks in the canonical mesh at the bottom (the calyx). As best as possible these should go to the points on the
bottom of the input mesh that have high curvature and stick out. 
The other objective is to minimize the distortion of the canonical mesh as it is reshaped to match the input mesh.

Run with optional positional arguments:
    python align_meshes.py [input.obj [canonical.obj [input_seam.csv
        [canonical_seam.csv [endpoints.txt [output.obj]]]]]]
The default input is in NormalizedTexture. The output retains the canonical
material and texture. The companion _correspondence.csv contains zero-based
input vertex and canonical face indices followed by three barycentric weights.
Peak selection and surface fitting are geometric approximations; they do not
guarantee a globally optimal or intersection-free deformation.
"""

import argparse
import csv
import math
import shutil
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

from normalize_uv_projection import _read_obj, _read_texture


def _mesh(path: Path):
    lines, uv, uv_faces, material = _read_obj(path)
    points = []
    faces = []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v":
            points.append([float(value) for value in parts[1:4]])
        elif parts[0] == "f":
            face = []
            for corner in parts[1:]:
                index = int(corner.split("/")[0])
                face.append(index - 1 if index > 0 else len(points) + index)
            faces.append(face)
    vertices = np.asarray(points, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError(f"Invalid vertex coordinates in {path}.")
    if triangles.shape != uv_faces.shape or triangles.min() < 0 or triangles.max() >= len(vertices):
        raise ValueError(f"Invalid face indices in {path}.")
    return lines, vertices, triangles, uv, uv_faces, material


def _seam_ids(path: Path, vertex_count: int) -> list[list[int]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = [[entry.strip() for entry in row if entry.strip()] for row in csv.reader(stream)]
    if len(rows) == 18:
        rows = [rows[i] for i in (0, 3, 6, 9, 12, 15)]
    elif len(rows) == 10:
        rows = [rows[i] for i in (0, 3, 6, 7, 8, 9)]
    else:
        raise ValueError(f"Expected 10 or 18 rows in seam CSV {path}.")
    seams = [[int(value) for value in row] for row in rows]
    for seam in seams:
        if len(seam) < 2 or min(seam) < 0 or max(seam) >= vertex_count:
            raise ValueError(f"Invalid seam vertex indices in {path}.")
    return seams


def _endpoints(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8", newline="") as stream:
        points = np.asarray(
            [[float(value.strip()) for value in row] for row in csv.reader(stream) if row],
            dtype=np.float64,
        )
    if points.shape != (2, 3) or not np.isfinite(points).all():
        raise ValueError(f"Expected top and bottom XYZ coordinates in {path}.")
    if np.linalg.norm(points[0] - points[1]) == 0:
        raise ValueError("The top and bottom endpoints must differ.")
    return points


def _chart(uv: np.ndarray) -> int:
    if np.linalg.norm(uv - (0.9, 0.1)) <= 0.101:
        return 0
    if np.linalg.norm(uv - (0.1, 0.9)) <= 0.101:
        return 1
    return 2


class _UVLocator:
    """Find a UV point on the matching chart, returning triangle and barycentric weights."""

    def __init__(self, uv: np.ndarray, uv_faces: np.ndarray):
        self.triangles = uv[uv_faces]
        self.charts = np.array([_chart(triangle.mean(axis=0)) for triangle in self.triangles])
        self.trees = {}
        self.indices = {}
        for chart in range(3):
            ids = np.flatnonzero(self.charts == chart)
            if not len(ids):
                raise ValueError(f"UV mesh has no faces for chart {chart}.")
            self.indices[chart] = ids
            self.trees[chart] = cKDTree(self.triangles[ids].mean(axis=1))

    def locate(self, point: np.ndarray, chart: int) -> tuple[int, np.ndarray]:
        indices = self.indices[chart]
        count = min(64, len(indices))
        _, nearby = self.trees[chart].query(point, k=count)
        candidates = indices[np.atleast_1d(nearby)]
        triangle = self.triangles[candidates]
        a = triangle[:, 1] - triangle[:, 0]
        b = triangle[:, 2] - triangle[:, 0]
        delta = point - triangle[:, 0]
        determinant = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        valid = np.abs(determinant) > 1e-14
        s = np.zeros(len(candidates))
        t = np.zeros(len(candidates))
        s[valid] = (delta[valid, 0] * b[valid, 1] - delta[valid, 1] * b[valid, 0]) / determinant[valid]
        t[valid] = (a[valid, 0] * delta[valid, 1] - a[valid, 1] * delta[valid, 0]) / determinant[valid]
        inside = valid & (s >= -1e-8) & (t >= -1e-8) & (s + t <= 1 + 1e-8)
        if np.any(inside):
            i = int(np.flatnonzero(inside)[0])
            return int(candidates[i]), np.array([1 - s[i] - t[i], s[i], t[i]])
        # Near chart boundaries, the polygonal circle may not contain the other mesh's boundary.
        best_distance = math.inf
        best = None
        for i, vertices in enumerate(triangle):
            for j in range(3):
                k = (j + 1) % 3
                edge = vertices[k] - vertices[j]
                length_sq = float(edge @ edge)
                if length_sq <= 1e-20:
                    continue
                fraction = float(np.clip((point - vertices[j]) @ edge / length_sq, 0, 1))
                distance = float(np.linalg.norm(point - (vertices[j] + fraction * edge)))
                if distance < best_distance:
                    weights = np.zeros(3)
                    weights[j], weights[k] = 1 - fraction, fraction
                    best_distance = distance
                    best = (int(candidates[i]), weights)
        if best is None or best_distance > 0.02:
            raise ValueError(f"No corresponding UV triangle near {point.tolist()} in chart {chart}.")
        return best


class _SurfaceLocator:
    """Project points to nearby triangles on the corresponding fitted chart."""

    def __init__(self, vertices, faces, charts):
        self.triangles = vertices[faces]
        self.indices = {}
        self.trees = {}
        for chart in range(3):
            ids = np.flatnonzero(charts == chart)
            self.indices[chart] = ids
            self.trees[chart] = cKDTree(self.triangles[ids].mean(axis=1))

    def locate(self, point, chart, uv_face):
        ids = self.indices[chart]
        _, nearby = self.trees[chart].query(point, k=min(64, len(ids)))
        candidates = np.unique(np.r_[ids[np.atleast_1d(nearby)], uv_face])
        triangles = self.triangles[candidates]
        a = triangles[:, 1] - triangles[:, 0]
        b = triangles[:, 2] - triangles[:, 0]
        delta = point - triangles[:, 0]
        aa = np.einsum("ij,ij->i", a, a)
        ab = np.einsum("ij,ij->i", a, b)
        bb = np.einsum("ij,ij->i", b, b)
        ad = np.einsum("ij,ij->i", a, delta)
        bd = np.einsum("ij,ij->i", b, delta)
        determinant = aa * bb - ab * ab
        valid = determinant > 1e-20
        s = np.zeros(len(candidates))
        t = np.zeros(len(candidates))
        s[valid] = (ad[valid] * bb[valid] - bd[valid] * ab[valid]) / determinant[valid]
        t[valid] = (bd[valid] * aa[valid] - ad[valid] * ab[valid]) / determinant[valid]
        valid &= (s >= 0) & (t >= 0) & (s + t <= 1)
        weights = np.column_stack((1 - s - t, s, t))
        projected = np.einsum("ij,ijk->ik", weights, triangles)
        distance = np.linalg.norm(projected - point, axis=1)
        distance[~valid] = np.inf
        for first, second in ((0, 1), (1, 2), (2, 0)):
            edge = triangles[:, second] - triangles[:, first]
            length = np.einsum("ij,ij->i", edge, edge)
            fraction = np.zeros(len(candidates))
            nonzero = length > 1e-20
            fraction[nonzero] = np.clip(
                np.einsum("ij,ij->i", point - triangles[:, first], edge)[nonzero] / length[nonzero], 0, 1
            )
            edge_distance = np.linalg.norm(triangles[:, first] + fraction[:, None] * edge - point, axis=1)
            better = edge_distance < distance
            weights[better] = 0
            weights[better, first] = 1 - fraction[better]
            weights[better, second] = fraction[better]
            distance[better] = edge_distance[better]
        closest = int(np.argmin(distance))
        return int(candidates[closest]), weights[closest]


def _vertex_uvs(vertices: np.ndarray, faces: np.ndarray, uv: np.ndarray, uv_faces: np.ndarray):
    per_vertex: list[list[tuple[int, np.ndarray]]] = [[] for _ in vertices]
    for face, face_uv in zip(faces, uv_faces):
        chart = _chart(uv[face_uv].mean(axis=0))
        for vertex, uv_index in zip(face, face_uv):
            if not any(existing_chart == chart for existing_chart, _ in per_vertex[vertex]):
                per_vertex[vertex].append((chart, uv[uv_index]))
    if any(not mappings for mappings in per_vertex):
        raise ValueError("Mesh contains vertices not referenced by any textured face.")
    return per_vertex


def _oriented_seams(seams: list[list[int]], points: np.ndarray, faces: np.ndarray) -> list[list[int]]:
    """Orient ring rows consistently around the top-to-bottom axis and sides downward."""
    result = [list(dict.fromkeys(row)) for row in seams]
    if (set(result[0]) != set(result[2]) or set(result[1]) != set(result[4])
            or set(result[3]) != set(result[5])):
        raise ValueError("The repeated seam rows do not describe the same mesh vertices.")
    top_center = points[result[0]].mean(axis=0)
    bottom_center = points[result[1]].mean(axis=0)
    axis = bottom_center - top_center
    axis_length = np.linalg.norm(axis)
    if axis_length <= 1e-12:
        raise ValueError("Cannot determine seam orientation: ring centers coincide.")
    axis /= axis_length
    edges = {
        tuple(sorted((int(face[i]), int(face[(i + 1) % 3]))))
        for face in faces for i in range(3)
    }

    def ring_direction(ring):
        coords = points[ring] - points[ring].mean(axis=0)
        area = np.cross(coords, np.roll(coords, -1, axis=0)).sum(axis=0)
        signed = float(area @ axis)
        if np.linalg.norm(area) <= 1e-12 or abs(signed) <= 1e-8 * np.linalg.norm(area):
            raise ValueError("Cannot determine seam orientation: ring area is ambiguous.")
        return signed

    for index in (3, 5):
        side = result[index]
        direction = float((points[side[-1]] - points[side[0]]) @ axis)
        if abs(direction) <= 1e-8 * axis_length:
            raise ValueError("Cannot determine side seam orientation.")
        if direction < 0:
            side.reverse()
    for index in (0, 1):
        ring = result[index]
        anchor = result[3][0 if index == 0 else -1]
        if anchor in ring:
            offset = ring.index(anchor)
            ring = ring[offset:] + ring[:offset]
        else:
            if any(tuple(sorted((anchor, vertex))) not in edges for vertex in (ring[0], ring[-1])):
                raise ValueError("Side seam endpoint does not connect to both ends of the ring.")
            ring = [anchor, *ring]
        if ring_direction(ring) < 0:
            ring = [ring[0], *reversed(ring[1:])]
        for first, second in zip(ring, ring[1:] + ring[:1]):
            if tuple(sorted((first, second))) not in edges:
                raise ValueError("A ring seam contains vertices that are not adjacent in the mesh.")
        result[index] = ring

    # Rows 2 and 4 repeat the ring boundaries; use the verified anchored loops.
    result[2] = result[0]
    result[4] = result[1]
    return result


def _resample_seam(
    source_ids: list[int],
    target_ids: list[int],
    source_points: np.ndarray,
    target_points: np.ndarray,
    closed: bool,
) -> tuple[np.ndarray, np.ndarray]:
    source = list(dict.fromkeys(source_ids))
    target = list(dict.fromkeys(target_ids))
    if len(source) < 2 or len(target) < 2:
        raise ValueError("A seam must contain at least two distinct vertices.")
    if closed:
        source.append(source[0])
        target.append(target[0])

    def fractions(points: np.ndarray) -> np.ndarray:
        distances = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        if distances[-1] <= 0:
            raise ValueError("A seam has zero length.")
        return distances / distances[-1]

    source_fraction = fractions(source_points[source])
    target_fraction = fractions(target_points[target])
    positions = np.column_stack([
        np.interp(source_fraction, target_fraction, target_points[target, axis])
        for axis in range(3)
    ])
    return np.asarray(source), positions


def _seam_correspondence(source_ids, target_ids, source_points, target_points, closed):
    source = list(dict.fromkeys(source_ids))
    target = list(dict.fromkeys(target_ids))
    if closed:
        source.append(source[0])
        target.append(target[0])

    def fractions(ids, points):
        length = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points[ids], axis=0), axis=1))]
        if length[-1] <= 0 or np.any(np.diff(length) <= 0):
            raise ValueError("A seam contains coincident consecutive vertices.")
        return length / length[-1]

    source_fraction = fractions(source, source_points)
    target_fraction = fractions(target, target_points)
    result = []
    for vertex, fraction in zip(source, source_fraction):
        edge = min(int(np.searchsorted(target_fraction, fraction, side="right") - 1), len(target) - 2)
        weight = (fraction - target_fraction[edge]) / (target_fraction[edge + 1] - target_fraction[edge])
        result.append((vertex, target[edge], target[edge + 1], weight))
    return result


def _seam_uv_transforms(template_seams, scan_seams, template, points, canonical_uvs, scan_uvs):
    """Orient the chart lookup to the same anchored seam directions as the fit."""
    def coordinate(mappings, chart):
        return next((coord for candidate, coord in mappings if candidate == chart), None)

    transforms = []
    for chart, center in enumerate((np.array([0.9, 0.1]), np.array([0.1, 0.9]))):
        def frame(seam, mappings):
            coords = [coordinate(mappings[vertex], chart) for vertex in seam]
            if any(coord is None for coord in coords):
                raise ValueError(f"Seam vertex is missing from UV chart {chart}.")
            radial = np.asarray(coords) - center
            area = np.sum(radial[:, 0] * np.roll(radial[:, 1], -1)
                          - radial[:, 1] * np.roll(radial[:, 0], -1))
            if abs(area) < 1e-12 or np.linalg.norm(radial[0]) < 1e-12:
                raise ValueError(f"Cannot determine UV orientation of chart {chart}.")
            return math.atan2(radial[0, 1], radial[0, 0]), np.sign(area)

        source_angle, source_sign = frame(template_seams[chart], canonical_uvs)
        target_angle, target_sign = frame(scan_seams[chart], scan_uvs)
        reflection = source_sign * target_sign
        angle = target_angle - reflection * source_angle
        rotation = np.array([[math.cos(angle), -math.sin(angle)],
                             [math.sin(angle), math.cos(angle)]])
        transforms.append((center, rotation @ np.diag([1, reflection])))

    samples = []
    for index in (0, 1):
        source, dest = template_seams[index], scan_seams[index]
        for vertex, first, second, weight in _seam_correspondence(source, dest, template, points, True):
            if vertex == source[0] or first == dest[0] or second == dest[0]:
                continue
            source_uv = coordinate(canonical_uvs[vertex], 2)
            first_uv = coordinate(scan_uvs[first], 2)
            second_uv = coordinate(scan_uvs[second], 2)
            if source_uv is not None and first_uv is not None and second_uv is not None:
                samples.append((source_uv, (1 - weight) * first_uv + weight * second_uv))
    if not samples:
        raise ValueError("Cannot determine body UV orientation from the seam vertices.")
    source_uv, target_uv = np.array(samples).transpose(1, 0, 2)
    center = np.array([0.5, 0.5])
    candidates = []
    for quarter in range(4):
        angle = quarter * math.pi / 2
        rotation = np.array([[round(math.cos(angle)), -round(math.sin(angle))],
                             [round(math.sin(angle)), round(math.cos(angle))]])
        for reflection in (1, -1):
            matrix = rotation @ np.diag([1, reflection])
            error = np.mean(np.sum(((source_uv - center) @ matrix.T + center - target_uv) ** 2, axis=1))
            candidates.append((error, matrix))
    transforms.append((center, min(candidates, key=lambda candidate: candidate[0])[1]))
    return transforms


def _match_calyx_peaks(
    template: np.ndarray,
    template_graph,
    target: np.ndarray,
    points: np.ndarray,
    faces: np.ndarray,
    uv: np.ndarray,
    uv_faces: np.ndarray,
    bottom_seam: list[int],
    endpoints: np.ndarray,
    reserved_vertices: set[int],
) -> dict[int, np.ndarray]:
    neighbors = template_graph.tocsr()
    seam_height = float(np.mean(template[bottom_seam, 1]))
    peak_candidates = []
    for vertex in np.flatnonzero(template[:, 1] < seam_height):
        adjacent = neighbors.indices[neighbors.indptr[vertex]:neighbors.indptr[vertex + 1]]
        if len(adjacent) and np.all(template[vertex, 1] < template[adjacent, 1] - 1e-10):
            peak_candidates.append(int(vertex))
    if len(peak_candidates) != 5:
        raise ValueError(f"Expected five distinct canonical calyx peaks, found {len(peak_candidates)}.")

    scan_edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    scan_edges.sort(axis=1)
    scan_edges = np.unique(scan_edges, axis=0)
    scan_graph = coo_matrix(
        (np.ones(2 * len(scan_edges)), (np.r_[scan_edges[:, 0], scan_edges[:, 1]], np.r_[scan_edges[:, 1], scan_edges[:, 0]])),
        shape=(len(points), len(points)),
    ).tocsr()
    degree = np.asarray(scan_graph.sum(axis=1)).ravel()
    if np.any(degree == 0):
        raise ValueError("The input mesh has isolated vertices.")
    axis = endpoints[1] - endpoints[0]
    axis /= np.linalg.norm(axis)
    prominence = ((points - (scan_graph @ points) / degree[:, None]) @ axis)
    bottom_faces = np.array([_chart(uv[face].mean(axis=0)) == 1 for face in uv_faces])
    bottom_vertices = np.unique(faces[bottom_faces])
    if len(bottom_vertices) < 5:
        raise ValueError("The input bottom chart does not contain enough vertices for five calyx peaks.")
    search = cKDTree(points[bottom_vertices])
    diameter = float(np.linalg.norm(np.ptp(points, axis=0)))
    if diameter <= 0:
        raise ValueError("The input mesh has zero size.")
    used = set()
    matches = {}
    for peak in peak_candidates:
        local = search.query_ball_point(target[peak], 0.1 * diameter)
        choices = np.array([
            bottom_vertices[i] for i in local
            if bottom_vertices[i] not in used and bottom_vertices[i] not in reserved_vertices
        ], dtype=int)
        if not len(choices):
            raise ValueError("Could not find five distinct bottom-chart calyx candidates.")
        distance = np.linalg.norm(points[choices] - target[peak], axis=1)
        depth = (points[choices] - endpoints[1]) @ axis
        score = 2.0 * prominence[choices] + 0.1 * depth - 0.15 * distance
        selected = int(choices[np.argmax(score)])
        used.add(selected)
        matches[peak] = points[selected]
    return matches


def align_meshes(
    input_path: Path,
    canonical_path: Path,
    input_seams_path: Path,
    canonical_seams_path: Path,
    endpoints_path: Path,
    output_path: Path,
) -> Path:
    """Fit the canonical mesh to the normalized scan and write input-to-canonical correspondence."""
    input_path, canonical_path = Path(input_path), Path(canonical_path)
    output_path = Path(output_path)
    _, points, faces, uv, uv_faces, input_material = _mesh(input_path)
    lines, template, template_faces, template_uv, template_uv_faces, material_path = _mesh(canonical_path)
    scan_seams = _seam_ids(Path(input_seams_path), len(points))
    template_seams = _seam_ids(Path(canonical_seams_path), len(template))
    scan_seams = _oriented_seams(scan_seams, points, faces)
    template_seams = _oriented_seams(template_seams, template, template_faces)
    endpoints = _endpoints(Path(endpoints_path))
    texture_path = _read_texture(material_path)
    output_material = output_path.with_suffix(".mtl")
    output_texture = output_path.with_suffix(texture_path.suffix)
    correspondence_path = output_path.with_name(output_path.stem + "_correspondence.csv")
    sources = {
        p.resolve() for p in (input_path, canonical_path, input_seams_path,
                              canonical_seams_path, endpoints_path, input_material,
                              _read_texture(input_material), material_path, texture_path)
    }
    if output_path.suffix.lower() != ".obj" or any(
        p.resolve() in sources for p in (output_path, output_material, output_texture, correspondence_path)
    ):
        raise ValueError("Output must be an OBJ and must not overwrite an input file.")

    scan_locator = _UVLocator(uv, uv_faces)
    canonical_locator = _UVLocator(template_uv, template_uv_faces)
    canonical_uvs = _vertex_uvs(template, template_faces, template_uv, template_uv_faces)
    scan_uvs = _vertex_uvs(points, faces, uv, uv_faces)
    uv_transforms = _seam_uv_transforms(
        template_seams, scan_seams, template, points, canonical_uvs, scan_uvs
    )
    target = np.empty_like(template)
    for index, mappings in enumerate(canonical_uvs):
        chart, coord = mappings[0]
        center, transform = uv_transforms[chart]
        face, weights = scan_locator.locate(center + transform @ (coord - center), chart)
        target[index] = weights @ points[faces[face]]

    constraints: dict[int, np.ndarray] = {}
    seam_pairs = []
    for seam_index in range(6):
        source = template_seams[seam_index]
        dest = scan_seams[seam_index]
        closed = seam_index in (0, 1, 2, 4)
        seam_pairs.append((source, dest, closed))
        ids, positions = _resample_seam(source, dest, template, points, closed)
        for vertex, position in zip(ids, positions):
            constraints.setdefault(int(vertex), position)

    scan_tree = cKDTree(points)
    endpoint_ids = scan_tree.query(endpoints)[1]
    def chart_pole(chart, center):
        candidates = [
            (float(np.linalg.norm(coord - center)), index)
            for index, mappings in enumerate(canonical_uvs)
            for candidate_chart, coord in mappings if candidate_chart == chart
        ]
        return min(candidates)[1]

    top_pole = chart_pole(0, np.array([0.9, 0.1]))
    bottom_pole = chart_pole(1, np.array([0.1, 0.9]))
    constraints[top_pole] = points[endpoint_ids[0]]
    constraints[bottom_pole] = points[endpoint_ids[1]]

    edges = np.concatenate((template_faces[:, [0, 1]], template_faces[:, [1, 2]], template_faces[:, [2, 0]]))
    edges.sort(axis=1)
    edges = np.unique(edges, axis=0)
    graph = coo_matrix(
        (np.ones(2 * len(edges)), (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
        shape=(len(template), len(template)),
    ).tocsr()
    peak_constraints = _match_calyx_peaks(
        template, graph, target, points, faces, uv, uv_faces, template_seams[1], endpoints,
        set(int(vertex) for vertex in endpoint_ids) | set().union(*map(set, scan_seams)),
    )
    constraints.update(peak_constraints)
    degree = np.asarray(graph.sum(axis=1)).ravel()
    laplacian = diags(degree) - graph
    # Penalize changes in local edge vectors relative to a best-fit similarity transform.
    template_center = template.mean(axis=0)
    target_center = target.mean(axis=0)
    centered_template = template - template_center
    centered_target = target - target_center
    u, singular_values, vt = np.linalg.svd(centered_template.T @ centered_target)
    signs = np.ones(3)
    signs[-1] = np.linalg.det(u @ vt)
    rotation = (u * signs) @ vt
    scale = np.dot(singular_values, signs) / np.sum(centered_template ** 2)
    reference = scale * centered_template @ rotation + target_center
    system = diags(np.maximum(degree, 1.0)) + 0.15 * laplacian
    fixed = np.array(sorted(constraints))
    free = np.setdiff1d(np.arange(len(template)), fixed)
    fitted = np.empty_like(template)
    fitted[fixed] = np.array([constraints[int(i)] for i in fixed])
    rhs = (degree[free, None] * target[free] + 0.15 * (laplacian @ reference)[free]
           - system[free][:, fixed] @ fitted[fixed])
    fitted[free] = spsolve(system[free][:, free].tocsc(), rhs)
    if not np.isfinite(fitted).all():
        raise ValueError("Could not solve the canonical mesh deformation.")

    edge_faces = {}
    vertex_faces = {}
    for face_index, face in enumerate(template_faces):
        for i in range(3):
            vertex_faces.setdefault(int(face[i]), face_index)
            edge_faces.setdefault(tuple(sorted((int(face[i]), int(face[(i + 1) % 3])))), face_index)
    seam_mapping = {}
    for source, dest, closed in seam_pairs:
        for vertex, first, second, weight in _seam_correspondence(dest, source, points, template, closed):
            edge = tuple(sorted((first, second)))
            if edge not in edge_faces:
                raise ValueError(f"Canonical seam edge {edge} is not present in the mesh.")
            face_index = edge_faces[edge]
            weights = np.zeros(3)
            face = template_faces[face_index]
            weights[np.flatnonzero(face == first)[0]] = 1 - weight
            weights[np.flatnonzero(face == second)[0]] = weight
            seam_mapping.setdefault(vertex, (face_index, weights))
    for template_vertex, scan_vertex in [
        (top_pole, int(endpoint_ids[0])), (bottom_pole, int(endpoint_ids[1])),
        *[(vertex, int(scan_tree.query(point)[1])) for vertex, point in peak_constraints.items()],
    ]:
        face_index = vertex_faces[template_vertex]
        weights = np.zeros(3)
        weights[np.flatnonzero(template_faces[face_index] == template_vertex)[0]] = 1
        seam_mapping[scan_vertex] = face_index, weights

    surface_locator = _SurfaceLocator(fitted, template_faces, canonical_locator.charts)
    correspondence = []
    for vertex, mappings in enumerate(scan_uvs):
        if vertex in seam_mapping:
            face, weights = seam_mapping[vertex]
        else:
            chart, coord = mappings[0]
            center, transform = uv_transforms[chart]
            uv_face, _ = canonical_locator.locate(center + transform.T @ (coord - center), chart)
            face, weights = surface_locator.locate(points[vertex], chart, uv_face)
        correspondence.append((vertex, face, *weights))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(texture_path, output_texture)
    with output_material.open("w", encoding="utf-8", newline="\n") as stream:
        for line in material_path.read_text(encoding="utf-8").splitlines():
            stream.write(f"map_Kd {output_texture.name}\n" if line.strip().startswith("map_Kd ") else line + "\n")
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        vertex = 0
        for line in lines:
            fields = line.split()
            if fields and fields[0] == "v":
                x, y, z = fitted[vertex]
                stream.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
                vertex += 1
            elif fields and fields[0] == "mtllib":
                stream.write(f"mtllib {output_material.name}\n")
            elif fields and fields[0] == "vn":
                continue
            elif fields and fields[0] == "f":
                corners = ["/".join(corner.split("/")[:2]) for corner in fields[1:]]
                stream.write("f " + " ".join(corners) + "\n")
            else:
                stream.write(line + "\n")
    with correspondence_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["input_vertex", "canonical_face", "weight_0", "weight_1", "weight_2"])
        writer.writerows(correspondence)
    return output_path


def main() -> None:
    root = Path("/Users/cindygrimm/PycharmProjects/data/ScannedModels")
    scan = root / "ProsserEnvyApples"
    parser = argparse.ArgumentParser(description="Align a canonical apple mesh to a normalized scan.")
    parser.add_argument("input", nargs="?", type=Path, default=scan / "NormalizedTexture/Envy1NoStemHiRes.obj")
    parser.add_argument("canonical", nargs="?", type=Path, default=root / "apple.obj")
    parser.add_argument("input_seams", nargs="?", type=Path, default=scan / "Voxelized/Envy1_seam.csv")
    parser.add_argument("canonical_seams", nargs="?", type=Path, default=root / "apple_seam.csv")
    parser.add_argument("endpoints", nargs="?", type=Path, default=scan / "Voxelized/Envy1.txt")
    parser.add_argument("output", nargs="?", type=Path, default=scan / "Aligned/Envy1Aligned.obj")
    args = parser.parse_args()
    align_meshes(args.input, args.canonical, args.input_seams, args.canonical_seams, args.endpoints, args.output)


if __name__ == "__main__":
    main()