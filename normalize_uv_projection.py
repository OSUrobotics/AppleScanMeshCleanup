"""Normalize an OBJ's three UV charts and warp its texture to match.

The top and bottom boundaries become circles of radius 0.1 centered at
(0.9, 0.1) and (0.1, 0.9). The body boundary becomes the diamond with
corners (0.5, 0), (1, 0.5), (0.5, 1), and (0, 0.5). Seam topology JSON
contains six ordered boundary paths and the top and bottom center positions.
Legacy seam CSV files are also accepted as input.
Each circular chart's first seam vertex lies on the positive U axis.
The warped texture is padded by four pixels outside the rasterized charts.
The output OBJ has a companion <output stem>_seam_topology.json with normalized UVs.
"""

import argparse
import csv
import json
import math
import shlex
import warnings
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

from filename_fixing import _split_filename_into_apple_type_and_number
from seam_topology import SeamTopology


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


def _read_seam_data(
    seam_path: Path,
) -> tuple[list[tuple[list[int], np.ndarray]], np.ndarray | None]:
    if seam_path.suffix.lower() == ".json":
        with seam_path.open(encoding="utf-8") as seam_file:
            topology = SeamTopology.from_dict(json.load(seam_file))
        topology.set_boundary_paths(
            (path["vertex_ids"], path["uv_coords"]) for path in topology.boundary_paths
        )
        records = [
            (path["vertex_ids"], np.asarray(path["uv_coords"], dtype=np.float64))
            for path in topology.boundary_paths
        ]
        centers = np.asarray([topology.top_point, topology.bottom_point], dtype=np.float64)
        if centers.shape != (2, 3) or not np.isfinite(centers).all():
            raise ValueError(f"Expected two finite 3D center coordinates in {seam_path}.")
        return records, centers
    with seam_path.open(encoding="utf-8", newline="") as seam_file:
        rows = [[entry.strip() for entry in row] for row in csv.reader(seam_file)]
    if len(rows) not in (18, 20):
        raise ValueError(f"Expected 18 or 20 seam CSV rows in {seam_path}, found {len(rows)}.")
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
    centers = None
    if len(rows) == 20:
        try:
            centers = np.asarray([[float(value) for value in row] for row in rows[18:]], dtype=np.float64)
        except ValueError as exc:
            raise ValueError(f"Expected two finite 3D center coordinates in {seam_path}.") from exc
        if centers.shape != (2, 3) or not np.isfinite(centers).all():
            raise ValueError(f"Expected two finite 3D center coordinates in {seam_path}.")
    return seams, centers


def _read_seam_records(seam_path: Path) -> list[tuple[list[int], np.ndarray]]:
    return _read_seam_data(seam_path)[0]


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
                f"(maximum error {np.max(distances):.6g}); use the matching seam topology."
            )
        seam_indices.append(matches)
    return seam_indices


def _center_uv_indices(
    lines: list[str], faces: np.ndarray, centers: np.ndarray, input_path: Path
) -> list[np.ndarray]:
    vertices = []
    vertex_faces = []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v":
            vertices.append([float(value) for value in parts[1:4]])
        elif parts[0] == "f":
            indices = [int(corner.split("/")[0]) for corner in parts[1:]]
            if 0 in indices:
                raise ValueError(f"OBJ vertex indices cannot be zero in {input_path}.")
            vertex_faces.append([index - 1 if index > 0 else len(vertices) + index for index in indices])
    coordinates = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(vertex_faces, dtype=np.int64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or not np.isfinite(coordinates).all():
        raise ValueError(f"Expected finite 3D vertices in {input_path}.")
    if triangles.shape != faces.shape or triangles.min() < 0 or triangles.max() >= len(coordinates):
        raise ValueError(f"Face vertex index is out of range in {input_path}.")
    closest = cKDTree(coordinates).query(centers)[1]
    return [np.unique(faces[triangles == index]) for index in closest]


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
    center_uv_indices: list[np.ndarray],
) -> np.ndarray:
    all_edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    all_edges.sort(axis=1)
    edges, counts = np.unique(all_edges, axis=0, return_counts=True)
    invalid_edges = (edges[:, 0] == edges[:, 1]) | (counts > 2)
    if np.any(invalid_edges):
        warnings.warn(
            "UV charts contain degenerate or non-manifold edges; perturbing their UV coordinates. "
            "This does not repair the topology, and parameterization may still fail.",
            RuntimeWarning,
            stacklevel=2,
        )
        affected = np.unique(edges[invalid_edges])
        perturbed_uvs = original_uvs.copy()
        noise = np.random.default_rng(0).uniform(-1e-8, 1e-8, size=(len(affected), 2))
        perturbed_uvs[affected] = np.clip(perturbed_uvs[affected] + noise, 0.0, 1.0)
    else:
        perturbed_uvs = original_uvs
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
            raise ValueError("A seam boundary path crosses multiple UV charts.")
        chart_labels.append(int(unique[0]))
    if len(set(chart_labels[:2] + [chart_labels[2]])) != 3 or len(set(chart_labels[2:])) != 1:
        raise ValueError("The seam data does not identify separate top, bottom, and body charts.")

    result = perturbed_uvs.copy()
    for chart_index, chart_label in enumerate(chart_labels[:3]):
        vertices = np.flatnonzero(labels == chart_label)
        boundary_edges = edges[(counts == 1) & (labels[edges[:, 0]] == chart_label)]
        boundary = _ordered_boundary(boundary_edges)
        boundary_set = set(boundary)
        for seam_index in ([chart_index] if chart_index < 2 else range(2, 6)):
            if not set(map(int, seam_indices[seam_index])).issubset(boundary_set):
                raise ValueError(f"Seam {seam_index + 1} is not on its OBJ chart boundary.")
        boundary_array = np.asarray(boundary)
        source = perturbed_uvs[boundary_array]

        if chart_index < 2:
            center = np.array((0.9, 0.1) if chart_index == 0 else (0.1, 0.9))
            signed_area = np.sum(source[:, 0] * np.roll(source[:, 1], -1) - source[:, 1] * np.roll(source[:, 0], -1))
            if abs(signed_area) < 1e-14:
                raise ValueError("A circular chart has zero boundary area.")
            lengths = np.linalg.norm(np.roll(source, -1, axis=0) - source, axis=1)
            lengths[lengths == 0] = lengths.mean() * 1e-6
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

        fixed_indices = boundary_array
        fixed_targets = target
        if chart_index < 2:
            matches = center_uv_indices[chart_index]
            matches = matches[labels[matches] == chart_label]
            if len(matches) != 1:
                raise ValueError(
                    f"The closest {'top' if chart_index == 0 else 'bottom'} center vertex "
                    "must have exactly one UV coordinate on its circular chart."
                )
            if int(matches[0]) in boundary_set:
                raise ValueError("The closest center vertex must be inside its circular chart, not on its boundary.")
            fixed_indices = np.r_[boundary_array, matches]
            fixed_targets = np.vstack((target, center))
        result[fixed_indices] = fixed_targets
        interior = np.setdiff1d(vertices, fixed_indices)
        if len(interior):
            degree = np.asarray(graph.sum(axis=1)).ravel()
            laplacian = diags(degree) - graph
            rhs = -(laplacian[interior][:, fixed_indices] @ fixed_targets)
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


def _find_seam_file(input_path: Path, seam_dir: Path, *, legacy_only: bool = False) -> Path:
    if not seam_dir.is_dir():
        raise NotADirectoryError(f"Seam directory not found or not a directory: {seam_dir}")
    apple = _split_filename_into_apple_type_and_number(input_path.name)
    patterns = (("*_seam.csv", "CSV"),) if legacy_only else (
        ("*_seam_topology.json", "topology JSON"), ("*_seam.csv", "CSV")
    )
    for pattern, kind in patterns:
        matches = []
        for candidate in seam_dir.glob(pattern):
            if not candidate.is_file():
                continue
            try:
                candidate_apple = _split_filename_into_apple_type_and_number(candidate.name)
            except ValueError:
                continue
            if candidate_apple == apple:
                matches.append(candidate)
        if len(matches) > 1:
            names = ", ".join(sorted(path.name for path in matches))
            raise ValueError(f"Multiple seam {kind} files match {input_path.name} in {seam_dir}: {names}")
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No seam CSV found or topology JSON found for {input_path.name} in {seam_dir}.")


def normalize_uv_projection(input_path: Path, output_path: Path, seam_path: Path) -> Path:
    """Normalize one textured OBJ using matching seam topology in seam_path."""
    input_path, seam_dir, output_path = Path(input_path), Path(seam_path), Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("The output OBJ must differ from the input OBJ.")
    if output_path.suffix.lower() != ".obj":
        raise ValueError("The normalized mesh output must be an OBJ file.")
    seam_path = _find_seam_file(input_path, seam_dir)
    lines, original_uvs, faces, material_path = _read_obj(input_path)
    texture_path = _read_texture(material_path)
    if np.any(original_uvs < -1e-8) or np.any(original_uvs > 1.0 + 1e-8):
        raise ValueError("The input UVs must lie within the texture image; tiled UVs are not supported.")
    records, centers = _read_seam_data(seam_path)
    if centers is None:
        raise ValueError(f"Expected top and bottom 3D center rows in {seam_path}.")
    seams = [uv for _, uv in records]
    center_uv_indices = _center_uv_indices(lines, faces, centers, input_path)
    normalized_uvs = _parameterize(original_uvs, faces, seams, center_uv_indices)
    seam_indices = _match_seam_uvs(original_uvs, seams)
    output_texture = output_path.with_suffix(".png")
    output_material = output_path.with_suffix(".mtl")
    output_seams = output_path.with_name(output_path.stem + "_seam_topology.json")
    sources = {input_path.resolve(), seam_path.resolve(), material_path.resolve(), texture_path.resolve()}
    if any(path.resolve() in sources for path in (output_path, output_material, output_texture, output_seams)):
        raise ValueError("Output files must not overwrite the input mesh, seam, material, or texture.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _warp_texture(texture_path, output_texture, original_uvs, normalized_uvs, faces)
    if seam_path.suffix.lower() == ".json":
        with seam_path.open(encoding="utf-8") as stream:
            topology = SeamTopology.from_dict(json.load(stream))
        coordinates = []
        targets = []
        for seam in topology.seams.values():
            for index, copies in enumerate(seam.uv_coords):
                if copies:
                    start = len(coordinates)
                    coordinates.extend(copies)
                    targets.append((seam, index, start, len(coordinates)))
        if coordinates:
            matched = _match_seam_uvs(original_uvs, [np.asarray(coordinates, dtype=np.float64)])[0]
            for seam, index, start, end in targets:
                seam.uv_coords[index] = normalized_uvs[matched[start:end]].tolist()
    else:
        vertices = np.asarray([
            [float(value) for value in line.split()[1:4]]
            for line in lines if line.split() and line.split()[0] == "v"
        ])
        topology = SeamTopology(
            float(vertices[:, 1].min()), float(vertices[:, 1].max()),
            tuple(centers[0]), tuple(centers[1]),
        )
    topology.set_boundary_paths(
        (ids, normalized_uvs[indices]) for (ids, _), indices in zip(records, seam_indices)
    )
    with output_seams.open("w", encoding="utf-8") as stream:
        json.dump(topology.to_dict(), stream)
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
    parser.add_argument("input", nargs="?", default="/Users/cindygrimm/PycharmProjects/data/ScannedModels/ProsserEnvyApples/BlenderRebaked/Envy_002_FullRes.obj", type=Path, help="Rebaked textured OBJ")
    parser.add_argument("output", nargs="?", default="/Users/cindygrimm/PycharmProjects/data/ScannedModels/ProsserEnvyApples/NormalizedTexture/Envy_002_FullRes_seam.obj", type=Path, help="Output OBJ path")
    parser.add_argument("seams", nargs="?", default="/Users/cindygrimm/PycharmProjects/data/ScannedModels/ProsserEnvyApples/Parameterized", type=Path, help="Directory containing seam topology JSON or legacy seam CSV files")
    args = parser.parse_args()
    normalize_uv_projection(args.input, args.output, args.seams)


if __name__ == "__main__":
    main()
