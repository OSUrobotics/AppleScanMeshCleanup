""" Putting a mesh into correspondence with a canonical mesh
Input:
    - mesh: The input mesh to be aligned.
    - canonical_mesh: The reference canonical mesh.
    - seam topology files defining the calyx seams of both meshes
    - the input seam topology contains the two polar endpoint locations on the input mesh

Output:
    - correspondence: A mapping from vertices of the input mesh to the canonical mesh.
    - The canonical mesh re-shaped to be the same shape as the input mesh

Objective: The top and bottom polar vertices should go to the input
mesh vertices closest to the endpoints in the input seam data.
Fit the two halves separated by the midpoint along the pole axis, pinning
the calyx seam to the corresponding scan ring.

The other objective is to minimize the distortion of the canonical mesh as it is reshaped to match the input mesh.

Run with optional positional arguments:
    python align_meshes.py [input.obj [canonical.obj [output.obj]]]
The default input is in RemoveCalyx. The output retains the canonical
material and uses the scan texture resampled into canonical UV space.
The companion _correspondence.csv contains zero-based
input vertex and canonical face indices followed by three barycentric weights.
Surface fitting is a geometric approximation; it does not
guarantee a globally optimal or intersection-free deformation.
"""

import argparse
import collections
import csv
import json
import math
import warnings
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from filename_fixing import _apple_name_base
from obj_geometry import create_adjacency_structure, read_obj, read_texture
from fix_texture_maps import flood_fill_texture_map
from seam_topology import SeamTopology


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


def _seam_ids(seam_topology: SeamTopology, vertex_count: int) -> list[list[int]]:
    seams = [seam.vertex_ids for seam in seam_topology.seams.values()]
    for seam in seams:
        if len(seam) < 2 or min(seam) < 0:
            raise ValueError(f"Invalid seam vertex indices in input seam topology.")
        if max(seam) >= vertex_count:
            raise ValueError(f"Invalid seam vertex indices in input seam topology.")
    return seams


def _endpoints(path: Path | SeamTopology) -> np.ndarray:
    if isinstance(path, SeamTopology):
        try:
            points = np.asarray([path.top_point, path.bottom_point], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("Seam topology must contain two finite 3D center coordinates.") from exc
        if points.shape != (2, 3) or not np.isfinite(points).all():
            raise ValueError("Seam topology must contain two finite 3D center coordinates.")
    else:
        _, points = (_read_seam_data(path) if path.suffix.lower() == ".json"
                     else _read_legacy_seam_data(path))
    if points is None:
        raise ValueError(f"Expected top and bottom XYZ coordinates at the end of {path}.")
    if np.linalg.norm(points[0] - points[1]) == 0:
        raise ValueError("The top and bottom endpoints must differ.")
    return points


def _calyx_ids(seam_topology: SeamTopology, vertex_count: int) -> list[int] | None:
    if "calyx" not in seam_topology.seams:
        raise ValueError(f"Seam topology must contain a calyx ring.")
    ids = seam_topology.seams["calyx"].vertex_ids

    if len(ids) < 3 or min(ids) < 0 or max(ids) >= vertex_count:
        raise ValueError(f"Invalid calyx seam vertex indices.")
    if len(set(ids)) != len(ids):
        # Find duplicate vertex indices in the calyx seam
        duplicates = [item for item, count in collections.Counter(ids).items() if count > 1]
        print(f"WARNING: Duplicate vertex indices in calyx seam: {duplicates}")
        # Remove the duplicate vertex indices, keeping the original order
        ids = list(dict.fromkeys(ids))  # Remove duplicates while preserving order
    return ids


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


def _validate_normalized_seams(seams, mappings, path: Path) -> None:
    for chart, center in enumerate((np.array([0.9, 0.1]), np.array([0.1, 0.9]))):
        for vertex in seams[chart]:
            coord = next((uv for candidate, uv in mappings[vertex] if candidate == chart), None)
            if coord is None or not np.isclose(np.linalg.norm(coord - center), 0.1, atol=1e-5, rtol=0):
                raise ValueError(
                    f"Input mesh {path} does not have normalized UV seams: "
                    f"vertex {vertex} is not on the boundary of chart {chart}. "
                    "Run UV normalization and align the mesh from NormalizedTexture."
                )


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


def _half_surface(points, faces, axis):
    """Split scan triangles at the midpoint without changing the input mesh."""
    heights = points @ axis
    midpoint = (heights.min() + heights.max()) / 2
    triangles = points[faces]
    triangle_heights = heights[faces] - midpoint
    pieces = []
    labels = []
    for half in (0, 1):
        signed = triangle_heights if half == 1 else -triangle_heights
        inside = signed >= 0
        complete = triangles[np.all(inside, axis=1)]
        if len(complete):
            pieces.extend(complete)
            labels.extend([half] * len(complete))
        crossing = np.flatnonzero(np.any(inside, axis=1) & ~np.all(inside, axis=1))
        for index in crossing:
            polygon = []
            for corner in range(3):
                previous = (corner - 1) % 3
                a, b = triangles[index, previous], triangles[index, corner]
                da, db = signed[index, previous], signed[index, corner]
                if (da < 0 and db > 0) or (da > 0 and db < 0):
                    polygon.append(a + da / (da - db) * (b - a))
                if db >= 0:
                    polygon.append(b)
            for corner in range(1, len(polygon) - 1):
                pieces.append(np.array([polygon[0], polygon[corner], polygon[corner + 1]]))
                labels.append(half)
    if not pieces or set(labels) != {0, 1}:
        raise ValueError("The scan mesh must have surface triangles in both midpoint halves.")
    vertices = np.asarray(pieces).reshape(-1, 3)
    split_faces = np.arange(len(vertices)).reshape(-1, 3)
    return vertices, split_faces, _SurfaceLocator(vertices, split_faces, np.asarray(labels))


def _refine_fit(fitted, graph, surface_points, surface_faces, locator, halves, protected):
    """Average neighbors and project each free vertex onto its scan half."""
    graph = graph.tocsr()
    degree = np.asarray(graph.sum(axis=1)).ravel()
    if np.any(degree == 0):
        raise ValueError("The canonical mesh has isolated vertices.")

    refined = fitted.copy()
    indices = np.setdiff1d(np.arange(len(fitted)), list(protected))
    for _ in range(4):
        refined[indices] = (graph[indices] @ refined) / degree[indices, None]
        for vertex in indices:
            half = int(halves[vertex])
            face, weights = locator.locate(refined[vertex], half, int(locator.indices[half][0]))
            refined[vertex] = weights @ surface_points[surface_faces[face]]
    return refined


def _seam_correspondence(source_ids, target_ids, source_points, target_points, closed, offset=0.0):
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
    for vertex, fraction in zip(source[:-1] if closed else source, source_fraction):
        if closed:
            fraction = (fraction + offset) % 1.0
        edge = min(int(np.searchsorted(target_fraction, fraction, side="right") - 1), len(target) - 2)
        weight = (fraction - target_fraction[edge]) / (target_fraction[edge + 1] - target_fraction[edge])
        result.append((vertex, target[edge], target[edge + 1], weight))
    return result


def _oriented_calyx(ids, faces, points, axis):
    """Order a mesh-edge ring consistently around the pole axis."""
    neighbors = {vertex: set() for vertex in ids}
    for face in faces:
        for first, second in zip(face, np.roll(face, -1)):
            if first in neighbors and second in neighbors:
                neighbors[first].add(int(second))
                neighbors[second].add(int(first))
    if all(second in neighbors[first] for first, second in zip(ids, ids[1:] + ids[:1])):
        ring = list(ids)
        start = ring.index(min(ring))
        ring = ring[start:] + ring[:start]
    else:
        if any(len(adjacent) != 2 for adjacent in neighbors.values()):
            raise ValueError("The calyx seam must form one simple mesh-edge ring.")
        start = min(ids)
        ring = [start]
        previous, current = start, min(neighbors[start])
        while current != start:
            if current in ring:
                raise ValueError("The calyx seam must form one simple mesh-edge ring.")
            ring.append(current)
            previous, current = current, next(vertex for vertex in neighbors[current] if vertex != previous)
        if len(ring) != len(ids):
            raise ValueError("The calyx seam contains multiple rings.")
    coords = points[ring] - points[ring].mean(axis=0)
    area = np.cross(coords, np.roll(coords, -1, axis=0)).sum(axis=0)
    signed_area = float(area @ axis)
    if np.linalg.norm(area) <= 1e-12 or abs(signed_area) <= 1e-8 * np.linalg.norm(area):
        raise ValueError("Cannot determine calyx seam orientation around the pole axis.")
    if signed_area < 0:
        ring = [ring[0], *reversed(ring[1:])]
    return ring


def _calyx_offset(source, target, initial_points, target_points):
    anchor = initial_points[source[0]]
    coords = target_points[target]
    edges = np.roll(coords, -1, axis=0) - coords
    lengths_sq = np.sum(edges * edges, axis=1)
    if np.any(lengths_sq <= 1e-20):
        raise ValueError("The calyx seam contains coincident consecutive vertices.")
    fractions = np.clip(np.sum((anchor - coords) * edges, axis=1) / lengths_sq, 0, 1)
    edge = int(np.argmin(np.sum((coords + fractions[:, None] * edges - anchor) ** 2, axis=1)))
    lengths = np.sqrt(lengths_sq)
    return float((np.sum(lengths[:edge]) + fractions[edge] * lengths[edge]) / np.sum(lengths))


def _resample_texture(
    texture_path, output_texture, scan_faces, scan_uv, scan_uv_faces,
    template_faces, template_uv, template_uv_faces, correspondence, uv_transforms,
    chart_correspondences=None,
):
    """Warp scan triangles into canonical UV charts using vertex correspondences."""
    vertex_faces = [[] for _ in range(int(template_faces.max()) + 1)]
    for face_index, face in enumerate(template_faces):
        for vertex in face:
            vertex_faces[vertex].append(face_index)
    charts = np.array([_chart(template_uv[face].mean(axis=0)) for face in template_uv_faces])
    output_uv = np.empty((len(scan_faces), 3, 2), dtype=float)
    cache = {}
    for face_index, (face, uv_face) in enumerate(zip(scan_faces, scan_uv_faces)):
        chart = _chart(scan_uv[uv_face].mean(axis=0))
        center, transform = uv_transforms[chart]
        for corner, (vertex, uv_index) in enumerate(zip(face, uv_face)):
            key = (int(vertex), int(uv_index), chart)
            if key not in cache:
                _, canonical_face, *weights = correspondence[vertex]
                weights = np.asarray(weights)
                vertices = template_faces[canonical_face]
                support = vertices[weights > 1e-10]
                if not len(support):
                    raise ValueError(f"Invalid correspondence weights for scan vertex {vertex}.")
                choices = set(vertex_faces[support[0]])
                for shared_vertex in support[1:]:
                    choices.intersection_update(vertex_faces[shared_vertex])
                choices = sorted(index for index in choices if charts[index] == chart)
                if not choices:
                    alternate = (chart_correspondences or {}).get((int(vertex), chart))
                    if alternate is None:
                        raise ValueError(
                            f"Scan vertex {vertex} has no corresponding canonical face on chart {chart}."
                        )
                    canonical_face, weights = alternate
                    weights = np.asarray(weights)
                    vertices = template_faces[canonical_face]
                    choices = [canonical_face]
                expected = center + transform.T @ (scan_uv[uv_index] - center)
                candidates = []
                for candidate in choices:
                    coordinate = np.zeros(2)
                    for source_vertex, weight in zip(vertices, weights):
                        if weight > 1e-10:
                            corner_index = np.flatnonzero(template_faces[candidate] == source_vertex)[0]
                            coordinate += weight * template_uv[template_uv_faces[candidate, corner_index]]
                    candidates.append(coordinate)
                nearest = np.argmin(np.linalg.norm(np.asarray(candidates) - expected, axis=1))
                cache[key] = candidates[nearest]
            output_uv[face_index, corner] = cache[key]
    source_triangles = scan_uv[scan_uv_faces]
    source_edges = np.linalg.norm(source_triangles - np.roll(source_triangles, 1, axis=1), axis=2)
    target_edges = np.linalg.norm(output_uv - np.roll(output_uv, 1, axis=1), axis=2)
    source_scale = np.max(source_edges, axis=1)
    stretched = np.flatnonzero(
        (np.max(target_edges, axis=1) > 0.25)
        & (np.max(target_edges, axis=1) > 3 * source_scale + 0.05)
    )
    if len(stretched):
        locator = _SurfaceLocator(template_uv, template_uv_faces, charts)
        repaired = 0
        unresolved = 0
        for face_index in stretched:
            chart = _chart(source_triangles[face_index].mean(axis=0))
            center, transform = uv_transforms[chart]
            expected = (source_triangles[face_index] - center) @ transform + center
            errors = np.linalg.norm(output_uv[face_index] - expected, axis=1)
            for corner in np.flatnonzero(errors > max(0.1, 3 * source_scale[face_index])):
                chart_face = int(locator.indices[chart][0])
                match, weights = locator.locate(expected[corner], chart, chart_face)
                output_uv[face_index, corner] = weights @ template_uv[template_uv_faces[match]]
                repaired += 1
            new_edges = np.linalg.norm(
                output_uv[face_index] - np.roll(output_uv[face_index], 1, axis=0), axis=1
            )
            if np.max(new_edges) > 0.25 and np.max(new_edges) > 3 * source_scale[face_index] + 0.05:
                unresolved += 1
        if repaired:
            warnings.warn(
                f"Texture resampling used face-corner UV lookup for {repaired} "
                "corners of stretched mapped triangles.",
                UserWarning, stacklevel=2,
            )
        if unresolved:
            warnings.warn(
                f"Texture resampling still has {unresolved} unusually stretched mapped triangles.",
                UserWarning, stacklevel=2,
            )
    source_uv = scan_uv[scan_uv_faces].reshape(-1, 2)
    triangles = np.arange(len(source_uv)).reshape(-1, 3)
    with Image.open(texture_path) as image:
        width, height = image.size
    first = output_uv[:, 1] - output_uv[:, 0]
    second = output_uv[:, 2] - output_uv[:, 0]
    determinant = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    degenerate = np.count_nonzero(np.abs(determinant) * (width - 1) * (height - 1) < 1e-12)
    if degenerate:
        warnings.warn(
            f"Texture resampling skipped {degenerate} degenerate mapped triangles; "
            "the output may contain gaps.",
            UserWarning,
            stacklevel=2,
        )
    _warp_texture(texture_path, output_texture, source_uv, output_uv.reshape(-1, 2), triangles)


def align_meshes(
    input_path: Path,
    output_path: Path,
    seam_path: Path,
    canonical_path: Path,
) -> Path:
    """Fit the canonical mesh using the input topology JSON in the seam directory."""
    input_path, canonical_path = Path(input_path), Path(canonical_path)
    output_path = Path(output_path)
    canonical_seams_path = canonical_path.with_name(canonical_path.stem + "_seam_topology.json")
    if not canonical_seams_path.is_file():
        canonical_seams_path = canonical_path.with_name(canonical_path.stem + "_seam.csv")
    apple_name = _apple_name_base(input_path.stem)
    input_seams_path = Path(seam_path) / (f"{apple_name}_FullRes_seam_topology.json")
    with input_seams_path.open(encoding="utf-8") as stream:
        input_topology = SeamTopology.from_dict(json.load(stream))
    with canonical_seams_path.open(encoding="utf-8") as stream:
        canonical_topology = SeamTopology.from_dict(json.load(stream))

    # Read in like this so it doesn't duplicate vertices when aligning meshes
    _, points, faces, uv, uv_faces, input_material = _mesh(input_path)
    lines, template, template_faces, template_uv, template_uv_faces, material_path = _mesh(canonical_path)
    
    scan_calyx = _calyx_ids(input_topology, len(points))
    template_calyx = _calyx_ids(canonical_topology, len(template))
    endpoints = _endpoints(input_topology)
    texture_path = _read_texture(material_path)
    scan_texture_path = _read_texture(input_material)
    output_material = output_path.with_suffix(".mtl")
    output_texture = output_path.with_suffix(".jpg")
    correspondence_path = output_path.with_name(output_path.stem + "_correspondence.csv")
    sources = {
        p.resolve() for p in (input_path, canonical_path, input_seams_path,
                              canonical_seams_path, input_material,
                              scan_texture_path, material_path, texture_path)
    }
    if output_path.suffix.lower() != ".obj" or any(
        p.resolve() in sources for p in (output_path, output_material, output_texture, correspondence_path)
    ):
        raise ValueError("Output must be an OBJ and must not overwrite an input file.")

    endpoint_ids = cKDTree(points).query(endpoints)[1]
    top_pole, bottom_pole = map(int, cKDTree(template).query(_endpoints(canonical_topology))[1])
    canonical_axis = template[bottom_pole] - template[top_pole]
    canonical_axis_length = np.linalg.norm(canonical_axis)
    if canonical_axis_length <= 1e-12:
        raise ValueError("The canonical pole vertices must be distinct.")
    canonical_axis /= canonical_axis_length
    scan_axis = points[endpoint_ids[1]] - points[endpoint_ids[0]]
    scan_axis_length = np.linalg.norm(scan_axis)
    if scan_axis_length <= 1e-12:
        raise ValueError("The input pole vertices must be distinct.")
    scan_axis /= scan_axis_length
    rotation, _ = Rotation.align_vectors([scan_axis], [canonical_axis])
    initial = (rotation.apply(template - template[top_pole]) * (scan_axis_length / canonical_axis_length)
               + points[endpoint_ids[0]])
    calyx_pair = None
    if scan_calyx is not None and template_calyx is not None:
        template_calyx = _oriented_calyx(template_calyx, template_faces, template, canonical_axis)
        scan_calyx = _oriented_calyx(scan_calyx, faces, points, scan_axis)
        offset = _calyx_offset(template_calyx, scan_calyx, initial, points)
        calyx_pair = template_calyx, scan_calyx, offset
    else:
        warnings.warn("Calyx seams are unavailable in legacy CSV data; calyx matching is skipped.",
                      stacklevel=2)
    target = initial.copy()
    if scan_calyx is not None and int(endpoint_ids[1]) in scan_calyx:
        raise ValueError("The bottom endpoint resolves to a calyx seam vertex, not the patch center.")
    constraints = {
        top_pole: points[endpoint_ids[0]],
        bottom_pole: points[endpoint_ids[1]],
    }
    if calyx_pair is not None:
        source, dest, offset = calyx_pair
        for vertex, first, second, weight in _seam_correspondence(
            source, dest, template, points, True, offset
        ):
            if vertex in (top_pole, bottom_pole):
                raise ValueError("A canonical pole cannot also lie on the calyx seam.")
            constraints[vertex] = (1 - weight) * points[first] + weight * points[second]

    canonical_heights = template @ canonical_axis
    midpoint = (canonical_heights.min() + canonical_heights.max()) / 2
    halves = (canonical_heights >= midpoint).astype(int)
    surface_points, surface_faces, half_locator = _half_surface(points, faces, scan_axis)
    for vertex in range(len(template)):
        if vertex not in constraints:
            half = int(halves[vertex])
            face, weights = half_locator.locate(target[vertex], half, int(half_locator.indices[half][0]))
            target[vertex] = weights @ surface_points[surface_faces[face]]

    edges = np.concatenate((template_faces[:, [0, 1]], template_faces[:, [1, 2]], template_faces[:, [2, 0]]))
    edges.sort(axis=1)
    edges = np.unique(edges, axis=0)
    graph = coo_matrix(
        (np.ones(2 * len(edges)), (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
        shape=(len(template), len(template)),
    ).tocsr()
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

    fitted = _refine_fit(
        fitted, graph, surface_points, surface_faces, half_locator, halves, constraints,
    )

    edge_faces = {}
    vertex_faces = {}
    for face_index, face in enumerate(template_faces):
        for i in range(3):
            vertex_faces.setdefault(int(face[i]), face_index)
            edge_faces.setdefault(tuple(sorted((int(face[i]), int(face[(i + 1) % 3])))), face_index)
    seam_mapping = {}
    if calyx_pair is not None:
        source, dest, offset = calyx_pair
        for vertex, first, second, weight in _seam_correspondence(
            dest, source, points, template, True, -offset
        ):
            edge = tuple(sorted((first, second)))
            if edge not in edge_faces:
                raise ValueError(f"Canonical calyx seam edge {edge} is not present in the mesh.")
            face_index = edge_faces[edge]
            weights = np.zeros(3)
            face = template_faces[face_index]
            weights[np.flatnonzero(face == first)[0]] = 1 - weight
            weights[np.flatnonzero(face == second)[0]] = weight
            seam_mapping[vertex] = face_index, weights
    for template_vertex, scan_vertex in (
        (top_pole, int(endpoint_ids[0])),
        (bottom_pole, int(endpoint_ids[1])),
    ):
        face_index = vertex_faces[template_vertex]
        weights = np.zeros(3)
        weights[np.flatnonzero(template_faces[face_index] == template_vertex)[0]] = 1
        seam_mapping[scan_vertex] = face_index, weights

    surface_locator = _SurfaceLocator(fitted, template_faces, np.zeros(len(template_faces), dtype=int))
    correspondence = []
    for vertex, point in enumerate(points):
        if vertex in seam_mapping:
            face, weights = seam_mapping[vertex]
        else:
            face, weights = surface_locator.locate(point, 0, int(surface_locator.indices[0][0]))
        correspondence.append((vertex, face, *weights))

    scan_uvs = _vertex_uvs(points, faces, uv, uv_faces)
    uv_transforms = [(np.array(center), np.eye(2)) for center in
                     ((0.9, 0.1), (0.1, 0.9), (0.5, 0.5))]
    charts = np.array([_chart(template_uv[face].mean(axis=0)) for face in template_uv_faces])
    chart_locator = _SurfaceLocator(fitted, template_faces, charts)
    chart_correspondences = {}
    for vertex, mappings in enumerate(scan_uvs):
        for chart, _ in mappings:
            if not len(chart_locator.indices[chart]):
                raise ValueError(f"Canonical mesh has no faces for UV chart {chart}.")
            chart_correspondences[vertex, chart] = chart_locator.locate(
                points[vertex], chart, int(chart_locator.indices[chart][0])
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _resample_texture(
        scan_texture_path, output_texture, faces, uv, uv_faces,
        template_faces, template_uv, template_uv_faces, correspondence, uv_transforms,
        chart_correspondences,
    )
    with output_material.open("w", encoding="utf-8", newline="\n") as stream:
        for line in material_path.read_text(encoding="utf-8").splitlines():
            stream.write(f"map_Kd {output_texture.name}\n" if line.strip().startswith("map_Kd ") else line + "\n")
    uv_indices = {}
    output_uvs = []
    output_uv_faces = []
    for face, uv_face in zip(template_faces, template_uv_faces):
        mapped = []
        for vertex, uv_index in zip(face, uv_face):
            coord = template_uv[uv_index]
            key = (int(vertex), float(coord[0]), float(coord[1]))
            if key not in uv_indices:
                uv_indices[key] = len(output_uvs) + 1
                output_uvs.append(coord)
            mapped.append(uv_indices[key])
        output_uv_faces.append(mapped)
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        vertex = 0
        face_index = 0
        for line in lines:
            fields = line.split()
            if fields and fields[0] == "v":
                x, y, z = fitted[vertex]
                stream.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
                vertex += 1
            elif fields and fields[0] == "mtllib":
                stream.write(f"mtllib {output_material.name}\n")
            elif fields and fields[0] in ("vt", "vn"):
                continue
            elif fields and fields[0] == "f":
                if face_index == 0:
                    for u, v in output_uvs:
                        stream.write(f"vt {u:.9g} {v:.9g}\n")
                corners = [f"{vertex_id + 1}/{uv_id}" for vertex_id, uv_id in
                           zip(template_faces[face_index], output_uv_faces[face_index])]
                stream.write("f " + " ".join(corners) + "\n")
                face_index += 1
            else:
                stream.write(line + "\n")
    with correspondence_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["input_vertex", "canonical_face", "weight_0", "weight_1", "weight_2"])
        writer.writerows(correspondence)

    flood_fill_texture_map(input_path=output_texture, mesh_path=output_path, seam_path=canonical_seams_path, output_path=output_texture)
    return output_path


def main() -> None:
    root = Path("/Users/cindygrimm/PycharmProjects/data/ScannedModels")
    scan = root / "ProsserEnvyApples"
    parser = argparse.ArgumentParser(description="Align a canonical apple mesh to a normalized scan.")
    parser.add_argument("input", nargs="?", type=Path, default=scan / "CalyxRemoved/Envy_001_FullRes.obj")
    parser.add_argument("canonical", nargs="?", type=Path, default=root / "Canonical/apple_fullres.obj")
    parser.add_argument("output", nargs="?", type=Path, default=scan / "AlignedMeshes/Envy_001_FullRes.obj")
    args = parser.parse_args()
    align_meshes(args.input, args.output, args.input.parent, args.canonical)


if __name__ == "__main__":
    main()