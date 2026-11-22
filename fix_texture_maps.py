"""Fill white or transparent texture pixels with colors from across UV seams."""

import argparse
import os
import tempfile
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import lsqr

from normalize_uv_projection import _match_seam_uvs, _read_obj
from legacy_seams import _read_legacy_seam_data
from obj_geometry import read_obj_geometry as _mesh_geometry


_MAX_PROJECTED_EDGE_LENGTH = 0.01  # Maximum allowed projected edge length in UV space


def _seam_edges(uvs: np.ndarray, faces: np.ndarray, records):
    indices = _match_seam_uvs(uvs, [points for _, points in records])
    face_edges = {}
    for face in faces:
        for a, b, third in ((face[0], face[1], face[2]), (face[1], face[2], face[0]),
                            (face[2], face[0], face[1])):
            face_edges.setdefault(tuple(sorted((int(a), int(b)))), []).append(int(third))
    boundary_neighbors = {}
    for (a, b), thirds in face_edges.items():
        if len(thirds) == 1:
            boundary_neighbors.setdefault(a, set()).add(b)
            boundary_neighbors.setdefault(b, set()).add(a)

    all_edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    graph = coo_matrix(
        (np.ones(2 * len(all_edges)), (np.r_[all_edges[:, 0], all_edges[:, 1]],
                                       np.r_[all_edges[:, 1], all_edges[:, 0]])),
        shape=(len(uvs), len(uvs)),
    ).tocsr()
    _, labels = connected_components(graph, directed=False)

    alternatives = {}
    for (ids, _), group_indices in zip(records, indices):
        for vertex_id, uv_index in zip(ids, group_indices):
            alternatives.setdefault((vertex_id, int(labels[uv_index])), set()).add(int(uv_index))

    def find_edge(id_a, id_b, a, b):
        if len(face_edges.get(tuple(sorted((a, b))), [])) == 1:
            return a, b
        chart = int(labels[a])
        choices = [
            (u, v)
            for u in alternatives.get((id_a, chart), {a})
            for v in alternatives.get((id_b, chart), {b})
            if len(face_edges.get(tuple(sorted((u, v))), [])) == 1
        ]
        if len(choices) != 1:
            raise ValueError(f"Seam edge {id_a}-{id_b} is not a unique OBJ UV boundary edge.")
        return choices[0]

    paired = {}
    for group, ((ids, _), uv_indices) in enumerate(zip(records, indices)):
        if len(ids) < 2:
            raise ValueError(f"Seam group {group + 1} needs at least two vertices.")
        segments = [(ids[i], ids[i + 1], int(uv_indices[i]), int(uv_indices[i + 1]))
                    for i in range(len(ids) - 1)]
        if group in (0, 1, 2, 4) and ids[0] != ids[-1]:
            first, last = int(uv_indices[0]), int(uv_indices[-1])
            if len(ids) > 2 and last in boundary_neighbors.get(first, set()):
                segments.append((ids[-1], ids[0], last, first))
            else:
                # The voxelized seam exporter omits the ring's first vertex.
                known = set(map(int, uv_indices))
                candidates = [
                    boundary_neighbors.get(endpoint, set()) - known for endpoint in (first, last)
                ]
                if any(len(candidate) != 1 for candidate in candidates):
                    raise ValueError(f"Cannot reconstruct the missing vertex of seam group {group + 1}.")
                missing_id = -1 if group in (0, 2) else -2
                segments.extend([
                    (missing_id, ids[0], next(iter(candidates[0])), first),
                    (ids[-1], missing_id, last, next(iter(candidates[1]))),
                ])
        for id_a, id_b, a, b in segments:
            if id_a == id_b:
                continue
            a, b = find_edge(id_a, id_b, a, b)
            thirds = face_edges.get(tuple(sorted((a, b))), [])
            if id_a > id_b:
                a, b = b, a
            paired.setdefault(tuple(sorted((id_a, id_b))), []).append(
                ((a, b, thirds[0], int(labels[a])), group)
            )

    edges = []
    for key, entries in paired.items():
        if len(entries) != 2:
            raise ValueError(f"Seam edge {key} has {len(entries)} sides; expected two.")
        sides, groups = zip(*entries)
        if set(groups) not in ({0, 2}, {1, 4}, {3, 5}):
            raise ValueError(f"Seam edge {key} belongs to incompatible seam groups {groups}.")
        edges.extend(((sides[0], sides[1]), (sides[1], sides[0])))
    ring_faces, side_faces = set(), set()
    for entries in paired.values():
        for side, group in entries:
            face = tuple(sorted(side[:3]))
            (side_faces if group in (3, 5) else ring_faces).add(face)
    corner_faces = ring_faces & side_faces
    if not edges:
        raise ValueError("The seam data contains no paired edges.")
    return labels, edges, corner_faces


def _seam_patches(faces: np.ndarray, edges, excluded_faces=frozenset()):
    vertex_faces = {}
    boundary_faces = {}
    for face_id, face in enumerate(faces):
        for vertex in face:
            vertex_faces.setdefault(int(vertex), set()).add(face_id)
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            boundary_faces.setdefault(tuple(sorted((int(a), int(b)))), []).append(face_id)

    excluded_ids = {face_id for face_id, face in enumerate(faces)
                    if tuple(sorted(map(int, face))) in excluded_faces}
    pending = {
        index for index, (_, source) in enumerate(edges)
        if tuple(sorted(source[:3])) not in excluded_faces
    }
    seam_neighbors = {}
    for index, (destination, source) in enumerate(edges):
        for target_uv, source_uv in zip(destination[:2], source[:2]):
            seam_neighbors.setdefault((destination[3], source[3], target_uv, source_uv), []).append(index)
    while pending:
        first = min(pending)
        pending.remove(first)
        group = [first]
        initial_target, initial_source = edges[first]
        pins = dict(zip(initial_source[:2], initial_target[:2]))
        for index in group:
            destination, source = edges[index]
            for target_uv, source_uv in zip(destination[:2], source[:2]):
                key = (destination[3], source[3], target_uv, source_uv)
                for neighbor in seam_neighbors[key]:
                    if neighbor in pending:
                        target, origin = edges[neighbor]
                        if any(source_id in pins and pins[source_id] != target_id
                               for target_id, source_id in zip(target[:2], origin[:2])):
                            continue
                        pins.update(zip(origin[:2], target[:2]))
                        pending.remove(neighbor)
                        group.append(neighbor)
        selected = set()
        for index in group:
            destination, source = edges[index]
            selected.update(boundary_faces[tuple(sorted(source[:2]))])
        frontier_vertices = set(pins)
        for _ in range(8):
            incident = {face_id for vertex in frontier_vertices for face_id in vertex_faces[vertex]}
            new_faces = incident - selected - excluded_ids
            selected.update(new_faces)
            frontier_vertices = {int(vertex) for face_id in new_faces for vertex in faces[face_id]}
        yield sorted(selected), pins, edges[group[0]]


def _project_patch(vertices, vertex_faces, uv_faces, uvs, face_ids, pins, seam, uv_vertices):
    patch_uvs = sorted({int(index) for face_id in face_ids for index in uv_faces[face_id]})
    local_index = {uv: index for index, uv in enumerate(patch_uvs)}
    positions = {}
    for face_id in face_ids:
        for uv, vertex in zip(uv_faces[face_id], vertex_faces[face_id]):
            if int(uv) in positions and positions[int(uv)] != int(vertex):
                raise ValueError("A UV coordinate is shared by different mesh vertices.")
            positions[int(uv)] = int(vertex)
    destination, source = seam
    for source_uv, target_uv in pins.items():
        if uv_vertices.get(target_uv) != {positions[source_uv]}:
            raise ValueError("Paired seam UVs do not refer to the same mesh vertices.")
    source_face_id = next(face_id for face_id in face_ids if
                          source[0] in uv_faces[face_id] and source[1] in uv_faces[face_id])
    source_face = list(uv_faces[source_face_id])
    order = 1 if (source_face.index(source[1]) - source_face.index(source[0])) % 3 == 1 else -1
    a, b, third = uvs[list(destination[:3])]
    edge = b - a
    cross = edge[0] * (third[1] - a[1]) - edge[1] * (third[0] - a[0])
    if abs(cross) < 1e-14:
        print("Degenerate UV triangle detected with vertices:", a, b, third)
        return {}, 0.0
    orientation = -np.sign(cross) * order

    row_indices, column_indices, values = [], [], []
    reference = []
    for row, face_id in enumerate(face_ids):
        xyz = vertices[vertex_faces[face_id]]
        first_edge, second_edge = xyz[1] - xyz[0], xyz[2] - xyz[0]
        length = float(np.linalg.norm(first_edge))
        area = float(np.linalg.norm(np.cross(first_edge, second_edge)))
        if length <= 1e-12 or area <= 1e-12:
            raise ValueError(f"Cannot reproject degenerate mesh face {face_id}.")
        x = float(first_edge @ second_edge) / length
        y = area / length
        reference.append(np.array([[0, 0], [length, 0], [x, orientation * y]]))
        for edge_index, (a, b) in enumerate(((0, 1), (1, 2), (2, 0))):
            row_indices.extend([3 * row + edge_index] * 2)
            column_indices.extend([local_index[int(uv_faces[face_id, a])],
                                   local_index[int(uv_faces[face_id, b])]])
            values.extend([-1, 1])
    matrix = coo_matrix((values, (row_indices, column_indices)),
                        shape=(3 * len(face_ids), len(patch_uvs))).tocsr()
    pinned_indices = np.array([local_index[uv] for uv in pins], dtype=int)
    pinned_values = np.array([uvs[target] for target in pins.values()])
    free = np.setdiff1d(np.arange(len(patch_uvs)), pinned_indices)
    solution = np.zeros((len(patch_uvs), 2))
    solution[pinned_indices] = pinned_values
    initial_face = face_ids.index(source_face_id)
    ref = reference[initial_face]
    source_a, source_b = source_face.index(source[0]), source_face.index(source[1])
    ref_edge = ref[source_b] - ref[source_a]
    target_edge = uvs[destination[1]] - uvs[destination[0]]
    scale = np.linalg.norm(target_edge) / np.linalg.norm(ref_edge)
    if scale <= 1e-12:
        raise ValueError("A target seam edge has zero UV length.")
    angle = np.arctan2(target_edge[1], target_edge[0]) - np.arctan2(ref_edge[1], ref_edge[0])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    reference = [triangle * scale for triangle in reference]
    rotations = [rotation.copy() for _ in face_ids]
    # Alternate rigid triangle fits with a least-squares solve for unpinned vertices.
    for _ in range(20):
        desired = np.concatenate([
            (triangle[[1, 2, 0]] - triangle) @ rotation.T
            for triangle, rotation in zip(reference, rotations)
        ])
        if len(free):
            rhs = desired - matrix[:, pinned_indices] @ pinned_values
            for component in (0, 1):
                solved = lsqr(matrix[:, free], rhs[:, component], atol=1e-9, btol=1e-9,
                              iter_lim=max(1000, 4 * len(free)))
                if solved[1] not in (0, 1, 2):
                    raise ValueError(f"Seam neighborhood projection did not converge (LSQR status {solved[1]}).")
                solution[free, component] = solved[0]
        for index, (face_id, triangle) in enumerate(zip(face_ids, reference)):
            projected = solution[[local_index[int(uv)] for uv in uv_faces[face_id]]]
            source_edges = triangle[[1, 2, 0]] - triangle
            target_edges = projected[[1, 2, 0]] - projected
            u, _, vt = np.linalg.svd(target_edges.T @ source_edges)
            rotation = u @ vt
            if np.linalg.det(rotation) < 0:
                u[:, -1] *= -1
                rotation = u @ vt
            rotations[index] = rotation
    return {uv: solution[index] for uv, index in local_index.items()}, orientation


def _copy_projected_face(result, source, missing, occupied, target, original):
    height, width = missing.shape
    scale = np.array([width - 1, height - 1])
    target = target * scale
    original = original * scale
    target[:, 1] = height - 1 - target[:, 1]
    original[:, 1] = height - 1 - original[:, 1]
    a, b = target[1] - target[0], target[2] - target[0]
    determinant = a[0] * b[1] - a[1] * b[0]
    if abs(determinant) < 1e-12:
        raise ValueError("A reprojected seam neighborhood contains a degenerate UV triangle.")
    minimum = np.maximum(np.floor(target.min(axis=0)).astype(int), 0)
    maximum = np.minimum(np.ceil(target.max(axis=0)).astype(int), [width - 1, height - 1])
    if np.any(minimum > maximum):
        return
    x, y = np.meshgrid(np.arange(minimum[0], maximum[0] + 1),
                       np.arange(minimum[1], maximum[1] + 1))
    dx, dy = x - target[0, 0], y - target[0, 1]
    s = (dx * b[1] - dy * b[0]) / determinant
    t = (a[0] * dy - a[1] * dx) / determinant
    mask = (s >= -1e-8) & (t >= -1e-8) & (s + t <= 1 + 1e-8)
    mask &= missing[y, x] & ~occupied[y, x]
    if not np.any(mask):
        return
    source_points = original[0] + s[mask, None] * (original[1] - original[0]) + t[mask, None] * (original[2] - original[0])
    source_x = np.clip(np.rint(source_points[:, 0]).astype(int), 0, width - 1)
    source_y = np.clip(np.rint(source_points[:, 1]).astype(int), 0, height - 1)
    valid = ~missing[source_y, source_x]
    result[y[mask][valid], x[mask][valid]] = source[source_y[valid], source_x[valid]]
    occupied[y[mask][valid], x[mask][valid]] = True


def _fill_seam_pixels(result: np.ndarray, coordinates: np.ndarray, edges) -> int:
    height, width = result.shape[:2]
    seam_image = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(seam_image)
    for destination, _ in edges:
        draw.line([tuple(coordinates[index]) for index in destination[:2]], fill=3)
    seam = np.asarray(seam_image, dtype=bool)
    result[seam, 1] = 255
    missing = np.isclose(np.sum(result[:, :, 0:2], axis=2), 0)
    if result.shape[2] == 4:
        missing |= result[:, :, 3] == 0
    to_fill = seam & missing
    print(f"Pixels to fill: {np.count_nonzero(seam)} {np.count_nonzero(to_fill)}")
    y, x = np.nonzero(to_fill)
    while len(y):
        counts = np.zeros(len(y), dtype=np.int16)
        total = np.zeros((len(y), result.shape[2]), dtype=np.float64)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                ny, nx = y + dy, x + dx
                inside = (ny >= 0) & (ny < height) & (nx >= 0) & (nx < width)
                indices = np.flatnonzero(inside)
                indices = indices[~missing[ny[indices], nx[indices]]]
                total[indices] += result[ny[indices], nx[indices]]
                counts[indices] += 1
        fill = counts > 0
        if not np.any(fill):
            break
        result[y[fill], x[fill]] = np.rint(total[fill] / counts[fill, None]).astype(np.uint8)
        missing[y[fill], x[fill]] = False
        y, x = y[~fill], x[~fill]
    return len(y)


def _remainder_fill_red(result: np.ndarray):
    """ Any unfilled pixels are filled in with the mean of the surrounding pixels. """
    height, width = result.shape[:2]
    for indx in range(0, 2):
        empty = np.isclose(np.sum(result[:, :, 0:2], axis=2), 0)
        print(f"Iteration {indx}, empty pixels: {np.count_nonzero(empty)}")
        ys, xs = np.nonzero(empty)
        for x, y in zip(xs, ys):
            if x < 5 or x > width - 5 or y < 5 or y > height - 5:
                continue
            patch = result[y-5:y+5, x-5:x+5]
            empty_patch = empty[y-5:y+5, x-5:x+5]
            if not np.any(~empty_patch):
                continue
            patch_color = np.mean(patch[~empty_patch], axis=0)
            result[y, x, 0:3] = patch_color
    

def flood_fill_texture_map(
    input_path: Path, mesh_path: Path, seam_path: Path, output_path: Path
) -> Path:
    """Reproject five vertex rings across each seam, keeping seam UVs fixed.

    Copy non-white, non-transparent source pixels into missing pixels outside
    the original charts. Pixels outside the projected neighborhoods are unchanged.
    """
    input_path, mesh_path, seam_path, output_path = map(
        Path, (input_path, mesh_path, seam_path, output_path)
    )
    if output_path.resolve() in {mesh_path.resolve(), seam_path.resolve()}:
        raise ValueError("The output texture must not overwrite the mesh or seam file.")
    lines, uvs, faces, material_path = _read_obj(mesh_path)
    vertices, vertex_faces = _mesh_geometry(lines, mesh_path)
    if output_path.resolve() == material_path.resolve():
        raise ValueError("The output texture must not overwrite the mesh material.")
    if np.any(uvs < 0) or np.any(uvs > 1):
        raise ValueError("The mesh UVs must lie within the texture image.")
    records = (_read_seam_records(seam_path) if seam_path.suffix.lower() == ".json"
               else _read_legacy_seam_data(seam_path)[0])
    _, edges, excluded_faces = _seam_edges(uvs, faces, records)
    uv_vertices = {}
    for face, vertex_face in zip(faces, vertex_faces):
        for uv, vertex in zip(face, vertex_face):
            uv_vertices.setdefault(int(uv), set()).add(int(vertex))
    with Image.open(input_path) as image:
        has_alpha = "A" in image.getbands() or "transparency" in image.info
        pixels = np.array(image.convert("RGBA" if has_alpha else "RGB"))
    height, width = pixels.shape[:2]
    if width < 2 or height < 2:
        raise ValueError("The texture image must be at least 2 by 2 pixels.")
    missing = np.isclose(np.sum(pixels[:, :, 0:2], axis=2), 0)
    if has_alpha:
        missing |= pixels[:, :, 3] == 0

    coordinates = uvs * [width - 1, height - 1]
    coordinates[:, 1] = height - 1 - coordinates[:, 1]
    chart_image = Image.new("1", (width, height), 0)
    chart_draw = ImageDraw.Draw(chart_image)
    for face in faces:
        chart_draw.polygon([tuple(coordinates[index]) for index in face], fill=1)
    occupied = np.array(chart_image, dtype=bool)
    result = pixels.copy()
    skipped_faces = 0
    long_faces = 0
    for face_ids, pins, seam in _seam_patches(faces, edges, excluded_faces):
        projected, orientation = _project_patch(
            vertices, vertex_faces, faces, uvs, face_ids, pins, seam, uv_vertices
        )
        if projected == {}:
            continue
        for face_id in face_ids:
            face = faces[face_id]
            target = np.array([projected[int(index)] for index in face])
            a, b = target[1] - target[0], target[2] - target[0]
            if orientation * (a[0] * b[1] - a[1] * b[0]) <= 1e-14:
                skipped_faces += 1
                continue
            if np.any(np.linalg.norm(target - np.roll(target, -1, axis=0), axis=1)
                      > _MAX_PROJECTED_EDGE_LENGTH):
                long_faces += 1
                continue
            _copy_projected_face(result, pixels, missing, occupied, target, uvs[face].copy())
    if skipped_faces:
        warnings.warn(f"Skipped {skipped_faces} flipped or degenerate reprojected seam faces.",
                      RuntimeWarning, stacklevel=2)
    if long_faces:
        warnings.warn(f"Skipped {long_faces} reprojected seam faces with UV edges longer than "
                      f"{_MAX_PROJECTED_EDGE_LENGTH}.", RuntimeWarning, stacklevel=2)
    #unfilled_seam = _fill_seam_pixels(result, coordinates, edges)
    #if unfilled_seam:
    #    warnings.warn(f"Could not fill {unfilled_seam} seam pixels without valid neighbors.",
    #                  RuntimeWarning, stacklevel=2)

    remainder_fill = _remainder_fill_red(result)
    if remainder_fill:
        warnings.warn(f"Could not fill {remainder_fill} remaining pixels without valid neighbors.",
                      RuntimeWarning, stacklevel=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output_path.parent, suffix=output_path.suffix, delete=False) as temp:
        temporary_path = Path(temp.name)
    try:
        Image.fromarray(result).save(temporary_path)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("texture", type=Path)
    parser.add_argument("mesh", type=Path, help="Original full-resolution textured OBJ")
    parser.add_argument("seams", type=Path, help="Matching seam topology JSON or legacy seam CSV")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    flood_fill_texture_map(args.texture, args.mesh, args.seams, args.output)


if __name__ == "__main__":
    main()
