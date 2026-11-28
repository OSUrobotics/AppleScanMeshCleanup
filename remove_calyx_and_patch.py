"""Split a textured OBJ at its saved calyx ring and grid-patch the remaining apple."""

import argparse
import json
import math
import shutil
from pathlib import Path
from PIL import Image


import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve

from seam_topology import SeamTopology
from obj_geometry import read_obj, read_texture, write_obj
from filename_fixing import write_material_file
from patch_stem_calyx_ring import write_mesh_with_seams
from remove_calyx import _remove_and_renumber, _calyx_faces
from uv_texture import _CAP_CENTERS, _CAP_RADIUS


def _cross(a, b, c):
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _patch_vertex_normals(vertices, faces):
    normals = np.zeros_like(vertices, dtype=float)
    for face in faces:
        a, b, c = map(int, face)
        normal = np.cross(vertices[b] - vertices[a], vertices[c] - vertices[a])
        normals[[a, b, c]] += normal
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 0
    normals[valid] /= lengths[valid, None]
    normals[~valid] = (0, -1, 0)
    return normals


def _delaunay_patch(points, triangles, sign, tolerance, scale, uv_points, uv_sign):
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
            if any(uv_sign * _cross(*uv_points[tri]) <= 1e-14 for tri in replacement):
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


def _grid_patch(vertices, faces, uv_faces, uvs, ring, inside, center_vertex, center):
    """Build the geometry patch with its new UV vertices inside the existing UV ring."""
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
    center = np.asarray(center, dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("Calyx ring must have a finite 3D center.")

    def unique_uv(vertex):
        values = uvs[uv_faces[inside][faces[inside] == vertex]]
        if not len(values) or not np.allclose(values, values[0], rtol=0, atol=2e-6):
            raise ValueError(f"Calyx vertex {vertex} must have one UV coordinate on the calyx.")
        return values[0]

    ring_uvs = np.array([unique_uv(v) for v in ring])
    center_uv = unique_uv(center_vertex)
    uv_area = sum(_cross(np.zeros(2), ring_uvs[i], ring_uvs[(i + 1) % len(ring)])
                  for i in range(len(ring)))
    if abs(uv_area) <= 1e-14:
        raise ValueError("The calyx ring has zero UV area.")
    uv_sign = 1 if uv_area > 0 else -1
    center_xz = center[[0, 2]]
    uv_scale = float(np.max(np.ptp(ring_uvs, axis=0)))
    if uv_scale <= 0:
        raise ValueError("Calyx ring has no UV extent.")
    uv_tolerance = uv_scale * uv_scale * 1e-10

    geometry_fan = np.array([
        _cross(center_xz, boundary[i], boundary[(i + 1) % len(ring)])
        for i in range(len(ring))
    ])
    uv_fan = np.array([
        _cross(center_uv, ring_uvs[i], ring_uvs[(i + 1) % len(ring)])
        for i in range(len(ring))
    ])
    use_center_fan = (
        np.all(sign * geometry_fan > tolerance)
        and np.all(uv_sign * uv_fan > 1e-14)
    )

    points = list(boundary)
    uv_points = list(ring_uvs)
    if use_center_fan:
        center_v_id = len(points)
        points.append(center_xz)
        uv_points.append(center_uv)
        triangles = [
            (center_v_id, i, (i + 1) % len(ring)) if sign > 0
            else (center_v_id, (i + 1) % len(ring), i)
            for i in range(len(ring))
        ]
    else:
        remaining = list(range(len(ring)))
        triangles = []
        while len(remaining) > 3:
            for i, current in enumerate(remaining):
                a, b, c = remaining[i - 1], current, remaining[(i + 1) % len(remaining)]
                if (sign * _cross(points[a], points[b], points[c]) <= tolerance
                        or uv_sign * _cross(uv_points[a], uv_points[b], uv_points[c]) <= uv_tolerance):
                    continue
                if any(
                    min(sign * _cross(points[a], points[b], points[p]),
                        sign * _cross(points[b], points[c], points[p]),
                        sign * _cross(points[c], points[a], points[p])) >= -tolerance
                    or min(uv_sign * _cross(uv_points[a], uv_points[b], uv_points[p]),
                           uv_sign * _cross(uv_points[b], uv_points[c], uv_points[p]),
                           uv_sign * _cross(uv_points[c], uv_points[a], uv_points[p])) >= -uv_tolerance
                    for p in remaining if p not in (a, b, c)
                ):
                    continue
                triangles.append((a, b, c))
                remaining.pop(i)
                break
            else:
                raise ValueError("Calyx ring is self-intersecting or cannot be triangulated in XZ.")
        triangles.append(tuple(remaining))

    grid_size = max(3, len(ring) // 20)
    spacing = 2 * float(np.max(np.abs(ring_uvs - center_uv))) / grid_size
    if spacing <= uv_scale * 1e-8:
        raise ValueError("Calyx UV ring is too small for a grid.")

    def barycentric(point, triangle):
        a, b, c = (uv_points[vertex] for vertex in triangle)
        denominator = _cross(a, b, c)
        if abs(denominator) <= uv_tolerance:
            return None
        return np.array([
            _cross(point, b, c),
            _cross(a, point, c),
            _cross(a, b, point),
        ]) / denominator

    def insert(point_uv, point_xz=None, required=False):
        matches = []
        for index, (a, b, c) in enumerate(triangles):
            triangle = (a, b, c)
            uv_crosses = np.array([
                uv_sign * _cross(uv_points[a], uv_points[b], point_uv),
                uv_sign * _cross(uv_points[b], uv_points[c], point_uv),
                uv_sign * _cross(uv_points[c], uv_points[a], point_uv),
            ])
            if np.min(uv_crosses) < -uv_tolerance:
                continue
            weights = barycentric(point_uv, triangle)
            if weights is None or np.min(weights) < -1e-8:
                continue
            mapped_xz = weights @ np.asarray(points)[list(triangle)]
            candidate_xz = mapped_xz if point_xz is None else point_xz
            geom_crosses = np.array([
                sign * _cross(points[a], points[b], candidate_xz),
                sign * _cross(points[b], points[c], candidate_xz),
                sign * _cross(points[c], points[a], candidate_xz),
            ])
            if np.min(geom_crosses) >= -tolerance:
                matches.append((index, triangle, uv_crosses, geom_crosses, mapped_xz))

        if (not matches or any(
                np.linalg.norm(point_uv - uv_points[v]) <= uv_scale * 1e-9
                for _, triangle, _, _, _ in matches for v in triangle
        )):
            if required:
                raise ValueError("Calyx center must be inside both projected rings.")
            return None
        new_index = len(points)
        replacement = []
        for _, triangle, uv_crosses, geom_crosses, _ in matches:
            for edge in range(3):
                if geom_crosses[edge] > tolerance and uv_crosses[edge] > uv_tolerance:
                    replacement.append((triangle[edge], triangle[(edge + 1) % 3], new_index))
        if not replacement:
            if required:
                raise ValueError("Calyx center lies on the boundary of the projected ring.")
            return None
        for index, *_ in reversed(matches):
            triangles.pop(index)
        points.append(matches[0][4] if point_xz is None else point_xz)
        uv_points.append(point_uv)
        triangles.extend(replacement)
        return new_index

    if not use_center_fan:
        center_v_id = insert(center_uv, center_xz, required=True)

    uv_edge_vectors = np.roll(ring_uvs, -1, axis=0) - ring_uvs
    uv_edge_lengths = np.sum(uv_edge_vectors * uv_edge_vectors, axis=1)
    if np.any(uv_edge_lengths <= uv_tolerance):
        raise ValueError("Calyx UV ring has coincident vertices.")
    grid_extent = math.ceil(grid_size / 2)
    for x in range(-grid_extent, grid_extent + 1):
        for z in range(-grid_extent, grid_extent + 1):
            if x or z:
                point = center_uv + spacing * np.array([x, z])
                t = np.clip(np.sum((point - ring_uvs) * uv_edge_vectors, axis=1) / uv_edge_lengths, 0, 1)
                distance = np.min(np.linalg.norm(point - ring_uvs - t[:, None] * uv_edge_vectors, axis=1))
                if distance > 0.2 * spacing:
                    insert(point)
    points = np.asarray(points)
    uv_points = np.asarray(uv_points)
    triangles = _delaunay_patch(points, triangles, sign, tolerance, scale, uv_points, uv_sign)

    # Interpolate height over the patch while retaining the source cap's UV map.
    edge_vectors = np.roll(boundary, -1, axis=0) - boundary
    fixed_y = np.concatenate((vertices[ring, 1], [float(np.mean(vertices[ring, 1]))]))
    adjacency = [set() for _ in points]
    for a, b, c in triangles:
        for v, w in ((a, b), (b, c), (c, a)):
            adjacency[v].add(w)
            adjacency[w].add(v)
    n_fixed = len(fixed_y)
    rows, cols, data = [], [], []
    rhs = np.zeros(len(points) - n_fixed)
    for vertex in range(n_fixed, len(points)):
        row = vertex - n_fixed
        rows.append(row)
        cols.append(row)
        data.append(len(adjacency[vertex]))
        for neighbor in adjacency[vertex]:
            if neighbor < n_fixed:
                rhs[row] += fixed_y[neighbor]
            else:
                rows.append(row)
                cols.append(neighbor - n_fixed)
                data.append(-1)
    if len(rhs):
        heights = np.concatenate((fixed_y, spsolve(coo_matrix((data, (rows, cols)),
                                                               shape=(len(rhs), len(rhs))).tocsr(), rhs)))
    else:
        heights = fixed_y
    patch_vertices = np.column_stack((points[:, 0], heights, points[:, 1]))
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
    return patch_vertices, uv_points, triangles, center_v_id


def remove_calyx_and_save(input_path, output_path):
    vertex_loc, vertex_uv, vert_uv_ids, vertex_normals, face_triangles, face_uv_triangles, face_norm_triangles, mtllibs = read_obj(input_path)
    seam_file = input_path.parent / (input_path.stem + "_seam_topology.json")
    with seam_file.open(encoding="utf-8") as stream:
        topology = SeamTopology.from_dict(json.load(stream))

    if "calyx" not in topology.seams or "stem" not in topology.seams:
        raise ValueError("Seam topology must contain a calyx ring and a stem ring.")

    ring, inside, _ = _calyx_faces(
        vertex_loc, face_triangles, topology.seams["calyx"], topology.bottom_point
    )
    inside_faces = np.flatnonzero(inside).tolist()
    topology.seams["calyx"].vertex_ids = ring.tolist()
    bottom_vertex = topology.bottom_vertex
    patch_vertices, patch_uvs, patch_faces, center_vid = _grid_patch(
        vertex_loc, face_triangles, face_uv_triangles, vertex_uv, ring, inside_faces,
        bottom_vertex, topology.seams["calyx"].center
    )
    new_vertex_loc, new_vertex_uv, new_vertex_norm, new_face_triangles, new_face_uv_triangles, new_face_normals = _remove_and_renumber(
        vertex_loc, vertex_uv, vertex_normals, vert_uv_ids, face_triangles,
        face_uv_triangles, face_norm_triangles, topology, inside_faces
    )

    ring_vertex_ids = topology.seams["calyx"].vertex_ids
    ring_count = len(ring_vertex_ids)
    patch_vertex_start = len(new_vertex_loc)
    patch_uv_start = len(new_vertex_uv)
    patch_normal_start = len(new_vertex_norm)
    new_vertex_loc.extend(patch_vertices[ring_count:])
    new_vertex_uv.extend(patch_uvs)
    new_vertex_norm.extend(_patch_vertex_normals(patch_vertices, patch_faces))

    vertex_map = [
        ring_vertex_ids[index] if index < ring_count else patch_vertex_start + index - ring_count
        for index in range(len(patch_vertices))
    ]
    for face in patch_faces:
        new_face_triangles.append([vertex_map[int(vertex)] for vertex in face])
        new_face_uv_triangles.append([patch_uv_start + int(vertex) for vertex in face])
        new_face_normals.append([patch_normal_start + int(vertex) for vertex in face])
    topology.bottom_vertex = vertex_map[center_vid]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    input_texture = read_texture(input_path.parent / mtllibs)
    output_texture = output_path.parent / input_texture.name
    if input_texture.resolve() != output_texture.resolve():
        shutil.copy2(input_texture, output_texture)
    output_material = output_path.with_suffix(".mtl")
    write_material_file(output_material, output_texture.name)
    write_obj(
        output_path, new_vertex_loc, new_vertex_uv, new_vertex_norm, new_face_triangles,
        new_face_uv_triangles, new_face_normals, output_path.stem, output_material.name
    )
    write_mesh_with_seams(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, default="/Users/cindygrimm/PycharmProjects/data/ScannedModels/ProsserEnvyApples/", help="Directory the paths are relative to")
    parser.add_argument("input", nargs="?", default="PaddedTexture/Envy_001_FullRes.obj", type=Path, help="Rebaked textured OBJ")
    parser.add_argument("output", nargs="?", default="CalyxRemoved/Envy_001_FullRes.obj", type=Path, help="Output OBJ path")

    args = parser.parse_args()
    input_path = args.root_dir / args.input if args.root_dir is not None else args.input
    output_path = args.root_dir / args.output if args.root_dir is not None else args.output

    print(f"Wrote {remove_calyx_and_save(input_path, output_path)}")


if __name__ == "__main__":
    main()
