"""Normalize an OBJ's three UV charts and warp its texture to match.

The top and bottom boundaries become circles of radius 0.1 centered at
(0.9, 0.1) and (0.1, 0.9). The body boundary becomes the diamond with
corners (0.5, 0), (1, 0.5), (0.5, 1), and (0, 0.5). The seam CSV contains
six groups of three rows: vertex indices, U coordinates, and V coordinates,
for the top, bottom, and the four clockwise sides of the body.
Each circular chart's first seam vertex lies on the positive U axis.
The warped texture is padded by four pixels outside the rasterized charts.
The output OBJ has a companion <output stem>_seam.csv with normalized UVs.
"""

import argparse
import csv
import math
import shlex
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree


_SEAM_TOLERANCE = 2e-6
_TEXTURE_PADDING_PIXELS = 4
_DIAMOND_CORNERS = np.array(
    [[0.5, 0.0], [1.0, 0.5], [0.5, 1.0], [0.0, 0.5]]
)


def _read_obj(input_path: Path):
    lines = input_path.read_text(encoding="utf-8").splitlines()
    uvs = []
    faces = []
    mtllibs = []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "vt":
            if len(parts) < 3:
                raise ValueError(f"Invalid texture coordinate in {input_path}: {line}")
            uvs.append([float(parts[1]), float(parts[2])])
        elif parts[0] == "f":
            if len(parts) != 4:
                raise ValueError(f"Expected triangular OBJ faces in {input_path}.")
            face = []
            for corner in parts[1:]:
                fields = corner.split("/")
                if len(fields) < 2 or not fields[1]:
                    raise ValueError(f"Face is missing texture coordinates: {line}")
                index = int(fields[1])
                if index == 0:
                    raise ValueError("OBJ texture indices cannot be zero.")
                face.append(index - 1 if index > 0 else len(uvs) + index)
            faces.append(face)
        elif parts[0] == "mtllib":
            mtllibs.extend(shlex.split(line)[1:])

    coordinates = np.asarray(uvs, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or not np.isfinite(coordinates).all():
        raise ValueError(f"Expected finite UV coordinates in {input_path}.")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError(f"Expected triangular faces with UV coordinates in {input_path}.")
    if triangles.min() < 0 or triangles.max() >= len(coordinates):
        raise ValueError(f"Face UV index is out of range in {input_path}.")
    if len(mtllibs) != 1:
        raise ValueError(f"Expected exactly one OBJ material library in {input_path}.")
    return lines, coordinates, triangles, input_path.parent / mtllibs[0]


def _read_texture(material_path: Path) -> Path:
    textures = []
    materials = 0
    for line in material_path.read_text(encoding="utf-8").splitlines():
        parts = shlex.split(line, comments=True)
        if parts and parts[0] == "newmtl":
            materials += 1
        if parts and (parts[0].startswith("map_") or parts[0] in ("bump", "disp", "decal")) and parts[0] != "map_Kd":
            raise ValueError(f"Unsupported material texture {parts[0]} in {material_path}.")
        if parts and parts[0] == "map_Kd":
            if len(parts) != 2:
                raise ValueError(f"Unsupported map_Kd options in {material_path}: {line}")
            textures.append(material_path.parent / parts[1])
    if materials != 1 or len(textures) != 1:
        raise ValueError(f"Expected one material with one diffuse texture in {material_path}.")
    return textures[0]


def _read_seam_records(seam_path: Path) -> list[tuple[list[int], np.ndarray]]:
    with seam_path.open(encoding="utf-8", newline="") as seam_file:
        rows = [[entry.strip() for entry in row] for row in csv.reader(seam_file)]
    if len(rows) != 18:
        raise ValueError(f"Expected 18 seam CSV rows in {seam_path}, found {len(rows)}.")
    seams = []
    for start in range(0, 18, 3):
        group = [row[:-1] if row and not row[-1] else row for row in rows[start:start + 3]]
        if not group[0] or len({len(row) for row in group}) != 1:
            raise ValueError(f"Seam rows {start + 1}-{start + 3} have inconsistent lengths.")
        ids = [int(value) for value in group[0]]
        if min(ids) < 0:
            raise ValueError(f"Negative mesh vertex index in {seam_path}.")
        points = np.column_stack((
            np.asarray(group[1], dtype=np.float64),
            np.asarray(group[2], dtype=np.float64),
        ))
        if not np.isfinite(points).all():
            raise ValueError(f"Non-finite seam coordinates in {seam_path}.")
        seams.append((ids, points))
    return seams


def _read_seams(seam_path: Path) -> list[np.ndarray]:
    return [uv for _, uv in _read_seam_records(seam_path)]


def _match_seam_uvs(original_uvs: np.ndarray, seams: list[np.ndarray]) -> list[np.ndarray]:
    seam_indices = []
    uv_tree = cKDTree(original_uvs)
    for index, seam in enumerate(seams):
        distances, matches = uv_tree.query(seam)
        if np.max(distances) > _SEAM_TOLERANCE:
            raise ValueError(
                f"Seam {index + 1} does not match the OBJ UVs "
                f"(maximum error {np.max(distances):.6g}); use the matching voxelized CSV."
            )
        seam_indices.append(matches)
    return seam_indices


def _write_seam_csv(
    output_path: Path,
    records: list[tuple[list[int], np.ndarray]],
    seam_indices: list[np.ndarray],
    normalized_uvs: np.ndarray,
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        for (ids, _), indices in zip(records, seam_indices):
            coordinates = normalized_uvs[indices]
            writer.writerow(ids)
            writer.writerow([f"{u:.9g}" for u in coordinates[:, 0]])
            writer.writerow([f"{v:.9g}" for v in coordinates[:, 1]])


def _ordered_boundary(edges: np.ndarray) -> list[int]:
    neighbors: dict[int, list[int]] = {}
    for a, b in edges:
        neighbors.setdefault(int(a), []).append(int(b))
        neighbors.setdefault(int(b), []).append(int(a))
    if not neighbors or any(len(adjacent) != 2 for adjacent in neighbors.values()):
        raise ValueError("Each UV chart must have one simple closed boundary.")
    first = next(iter(neighbors))
    boundary = [first]
    visited = {first}
    previous = -1
    current = first
    while True:
        adjacent = neighbors[current]
        following = adjacent[0] if adjacent[0] != previous else adjacent[1]
        if following == first:
            break
        if following in visited:
            raise ValueError("The UV chart boundary intersects itself.")
        boundary.append(following)
        visited.add(following)
        previous, current = current, following
    if len(boundary) != len(neighbors):
        raise ValueError("A UV chart has multiple boundary loops.")
    return boundary


def _parameterize(
    original_uvs: np.ndarray,
    faces: np.ndarray,
    seams: list[np.ndarray],
) -> np.ndarray:
    all_edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    all_edges.sort(axis=1)
    edges, counts = np.unique(all_edges, axis=0, return_counts=True)
    if np.any(edges[:, 0] == edges[:, 1]) or np.any(counts > 2):
        raise ValueError("UV charts contain degenerate or non-manifold edges.")
    graph = coo_matrix(
        (np.ones(2 * len(edges)), (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
        shape=(len(original_uvs), len(original_uvs)),
    ).tocsr()
    component_count, labels = connected_components(graph, directed=False)
    if component_count != 3:
        raise ValueError(f"Expected three connected UV charts, found {component_count}.")
    seam_indices = _match_seam_uvs(original_uvs, seams)
    chart_labels = []
    for seam in seam_indices:
        unique = np.unique(labels[seam])
        if len(unique) != 1:
            raise ValueError("A seam CSV row crosses multiple UV charts.")
        chart_labels.append(int(unique[0]))
    if len(set(chart_labels[:2] + [chart_labels[2]])) != 3 or len(set(chart_labels[2:])) != 1:
        raise ValueError("The seam CSV does not identify separate top, bottom, and body charts.")

    result = original_uvs.copy()
    for chart_index, chart_label in enumerate(chart_labels[:3]):
        vertices = np.flatnonzero(labels == chart_label)
        boundary_edges = edges[(counts == 1) & (labels[edges[:, 0]] == chart_label)]
        boundary = _ordered_boundary(boundary_edges)
        boundary_set = set(boundary)
        for seam_index in ([chart_index] if chart_index < 2 else range(2, 6)):
            if not set(map(int, seam_indices[seam_index])).issubset(boundary_set):
                raise ValueError(f"Seam {seam_index + 1} is not on its OBJ chart boundary.")
        boundary_array = np.asarray(boundary)
        source = original_uvs[boundary_array]

        if chart_index < 2:
            center = np.array((0.9, 0.1) if chart_index == 0 else (0.1, 0.9))
            signed_area = np.sum(source[:, 0] * np.roll(source[:, 1], -1) - source[:, 1] * np.roll(source[:, 0], -1))
            if abs(signed_area) < 1e-14:
                raise ValueError("A circular chart has zero boundary area.")
            lengths = np.linalg.norm(np.roll(source, -1, axis=0) - source, axis=1)
            if np.any(lengths <= 0):
                raise ValueError("A circular chart has a zero-length boundary edge.")
            angles = np.r_[0.0, np.cumsum(lengths[:-1])] / lengths.sum() * (2 * math.pi)
            angles *= 1 if signed_area > 0 else -1
            anchor = boundary.index(int(seam_indices[chart_index][0]))
            angles -= angles[anchor]
            target = center + 0.1 * np.column_stack((np.cos(angles), np.sin(angles)))
        else:
            corner_indices = [
                int(np.argmin(np.linalg.norm(source - corner, axis=1)))
                for corner in _DIAMOND_CORNERS
            ]
            if len(set(corner_indices)) != 4:
                raise ValueError("Could not identify four distinct body-chart corners.")
            target = np.empty_like(source)
            order = sorted(range(4), key=lambda i: corner_indices[i])
            for start_corner, end_corner in zip(order, order[1:] + order[:1]):
                start = corner_indices[start_corner]
                end = corner_indices[end_corner]
                indices = np.array([(start + step) % len(boundary) for step in range((end - start) % len(boundary) + 1)])
                segment = source[indices]
                distances = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(segment, axis=0), axis=1))]
                if distances[-1] <= 0:
                    raise ValueError("A body seam side has zero length.")
                fraction = distances / distances[-1]
                target[indices] = (
                    (1.0 - fraction[:, None]) * _DIAMOND_CORNERS[start_corner]
                    + fraction[:, None] * _DIAMOND_CORNERS[end_corner]
                )
            if any((order[(i + 1) % 4] - order[i]) % 4 not in (1, 3) for i in range(4)):
                raise ValueError("Body-chart corners do not follow the diamond boundary.")

        result[boundary_array] = target
        interior = np.setdiff1d(vertices, boundary_array)
        if len(interior):
            degree = np.asarray(graph.sum(axis=1)).ravel()
            laplacian = diags(degree) - graph
            rhs = -(laplacian[interior][:, boundary_array] @ target)
            solution = spsolve(laplacian[interior][:, interior].tocsc(), rhs)
            if not np.isfinite(solution).all():
                raise ValueError("Could not solve the UV chart parameterization.")
            result[interior] = solution
    return result


def _warp_texture(
    texture_path: Path,
    output_texture: Path,
    original_uvs: np.ndarray,
    normalized_uvs: np.ndarray,
    faces: np.ndarray,
) -> None:
    with Image.open(texture_path) as input_image:
        source = np.asarray(input_image.convert("RGBA"))
    height, width = source.shape[:2]
    output = np.zeros_like(source)
    covered = np.zeros((height, width), dtype=bool)
    scale = np.array([width - 1, height - 1])
    source_pixels = original_uvs * scale
    target_pixels = normalized_uvs * scale
    source_pixels[:, 1] = (height - 1) - source_pixels[:, 1]
    target_pixels[:, 1] = (height - 1) - target_pixels[:, 1]
    for face in faces:
        destination = target_pixels[face]
        origin = source_pixels[face]
        a = destination[1] - destination[0]
        b = destination[2] - destination[0]
        determinant = a[0] * b[1] - a[1] * b[0]
        if abs(determinant) < 1e-12:
            continue
        minimum = np.maximum(np.floor(destination.min(axis=0)).astype(int), 0)
        maximum = np.minimum(np.ceil(destination.max(axis=0)).astype(int), [width - 1, height - 1])
        if np.any(minimum > maximum):
            continue
        x, y = np.meshgrid(
            np.arange(minimum[0], maximum[0] + 1),
            np.arange(minimum[1], maximum[1] + 1),
        )
        dx, dy = x - destination[0, 0], y - destination[0, 1]
        s = (dx * b[1] - dy * b[0]) / determinant
        t = (a[0] * dy - a[1] * dx) / determinant
        mask = (s >= -1e-8) & (t >= -1e-8) & (s + t <= 1 + 1e-8)
        if not np.any(mask):
            continue
        source_x = np.clip(origin[0, 0] + s[mask] * (origin[1, 0] - origin[0, 0]) + t[mask] * (origin[2, 0] - origin[0, 0]), 0, width - 1)
        source_y = np.clip(origin[0, 1] + s[mask] * (origin[1, 1] - origin[0, 1]) + t[mask] * (origin[2, 1] - origin[0, 1]), 0, height - 1)
        x0, y0 = source_x.astype(int), source_y.astype(int)
        x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
        fx, fy = (source_x - x0)[:, None], (source_y - y0)[:, None]
        samples = (
            source[y0, x0] * (1 - fx) * (1 - fy)
            + source[y0, x1] * fx * (1 - fy)
            + source[y1, x0] * (1 - fx) * fy
            + source[y1, x1] * fx * fy
        )
        output[y[mask], x[mask]] = np.rint(samples).astype(np.uint8)
        covered[y[mask], x[mask]] = True
    if not np.any(covered):
        raise ValueError("No texture pixels were covered by the normalized UV triangles.")
    distances, nearest = distance_transform_edt(~covered, return_indices=True)
    padding = (~covered) & (distances <= _TEXTURE_PADDING_PIXELS)
    output[padding] = output[nearest[0][padding], nearest[1][padding]]
    Image.fromarray(output).save(output_texture)


def normalize_uv_projection(input_path: Path, seam_path: Path, output_path: Path) -> Path:
    """Normalize one textured OBJ using its matching voxelized seam CSV."""
    input_path, seam_path, output_path = Path(input_path), Path(seam_path), Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("The output OBJ must differ from the input OBJ.")
    if output_path.suffix.lower() != ".obj":
        raise ValueError("The normalized mesh output must be an OBJ file.")
    lines, original_uvs, faces, material_path = _read_obj(input_path)
    texture_path = _read_texture(material_path)
    if np.any(original_uvs < -1e-8) or np.any(original_uvs > 1.0 + 1e-8):
        raise ValueError("The input UVs must lie within the texture image; tiled UVs are not supported.")
    records = _read_seam_records(seam_path)
    seams = [uv for _, uv in records]
    normalized_uvs = _parameterize(original_uvs, faces, seams)
    seam_indices = _match_seam_uvs(original_uvs, seams)
    output_texture = output_path.with_suffix(".png")
    output_material = output_path.with_suffix(".mtl")
    output_seams = output_path.with_name(output_path.stem + "_seam.csv")
    sources = {input_path.resolve(), seam_path.resolve(), material_path.resolve(), texture_path.resolve()}
    if any(path.resolve() in sources for path in (output_path, output_material, output_texture, output_seams)):
        raise ValueError("Output files must not overwrite the input mesh, seam, material, or texture.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _warp_texture(texture_path, output_texture, original_uvs, normalized_uvs, faces)
    _write_seam_csv(output_seams, records, seam_indices, normalized_uvs)
    material_lines = material_path.read_text(encoding="utf-8").splitlines()
    with output_material.open("w", encoding="utf-8", newline="\n") as material_file:
        for line in material_lines:
            material_file.write(f"map_Kd {output_texture.name}\n" if line.strip().startswith("map_Kd ") else line + "\n")
    uv_index = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as obj_file:
        for line in lines:
            parts = line.split()
            if parts and parts[0] == "vt":
                u, v = normalized_uvs[uv_index]
                extra = f" {' '.join(parts[3:])}" if len(parts) > 3 else ""
                obj_file.write(f"vt {u:.9g} {v:.9g}{extra}\n")
                uv_index += 1
            elif parts and parts[0] == "mtllib":
                obj_file.write(f"mtllib {output_material.name}\n")
            else:
                obj_file.write(line + "\n")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize an OBJ's UV charts and texture.")
    parser.add_argument("input", nargs="?", default="/Users/cindygrimm/PycharmProjects/data/ScannedModels/ProsserEnvyApples/BlenderRebaked/Envy1NoStemHiRes.obj", type=Path, help="Rebaked textured OBJ")
    parser.add_argument("seams", nargs="?", default="/Users/cindygrimm/PycharmProjects/data/ScannedModels/ProsserEnvyApples/Voxelized/Envy1_seam.csv", type=Path, help="Matching voxelized seam CSV")
    parser.add_argument("output", nargs="?", default="/Users/cindygrimm/PycharmProjects/data/ScannedModels/ProsserEnvyApples/NormalizedTexture/Envy1NoStemHiRes.obj", type=Path, help="Output OBJ path")
    args = parser.parse_args()
    normalize_uv_projection(args.input, args.seams, args.output)


if __name__ == "__main__":
    main()
