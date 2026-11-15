"""Fill white or transparent texture pixels with colors from across UV seams."""

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from normalize_uv_projection import _match_seam_uvs, _read_obj, _read_seam_records


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
                (a, b, thirds[0], int(labels[a]))
            )

    edges = []
    for key, sides in paired.items():
        if len(sides) != 2:
            raise ValueError(f"Seam edge {key} has {len(sides)} sides; expected two.")
        edges.extend(((sides[0], sides[1]), (sides[1], sides[0])))
    if not edges:
        raise ValueError("The seam CSV contains no paired edges.")
    return labels, edges


def flood_fill_texture_map(
    input_path: Path, mesh_path: Path, seam_path: Path, output_path: Path
) -> Path:
    """Fill missing pixels using matching seam edges in the original textured OBJ.

    Pure white RGB and fully transparent pixels are treated as missing. Existing
    non-white, non-transparent pixels are left unchanged.
    """
    input_path, mesh_path, seam_path, output_path = map(
        Path, (input_path, mesh_path, seam_path, output_path)
    )
    if output_path.resolve() in {mesh_path.resolve(), seam_path.resolve()}:
        raise ValueError("The output texture must not overwrite the mesh or seam CSV.")
    _, uvs, faces, material_path = _read_obj(mesh_path)
    if output_path.resolve() == material_path.resolve():
        raise ValueError("The output texture must not overwrite the mesh material.")
    if np.any(uvs < 0) or np.any(uvs > 1):
        raise ValueError("The mesh UVs must lie within the texture image.")
    labels, edges = _seam_edges(uvs, faces, _read_seam_records(seam_path))
    with Image.open(input_path) as image:
        has_alpha = "A" in image.getbands() or "transparency" in image.info
        pixels = np.array(image.convert("RGBA" if has_alpha else "RGB"))
    height, width = pixels.shape[:2]
    if width < 2 or height < 2:
        raise ValueError("The texture image must be at least 2 by 2 pixels.")
    missing = np.all(pixels[:, :, :3] == 255, axis=2)
    if has_alpha:
        missing |= pixels[:, :, 3] == 0

    coordinates = uvs * [width - 1, height - 1]
    coordinates[:, 1] = height - 1 - coordinates[:, 1]
    chart_image = Image.new("I", (width, height), -1)
    chart_draw = ImageDraw.Draw(chart_image)
    for face in faces:
        chart_draw.polygon([tuple(coordinates[index]) for index in face], fill=int(labels[face[0]]))
    charts = np.asarray(chart_image)

    seam_image = Image.new("I", (width, height), 0)
    seam_draw = ImageDraw.Draw(seam_image)
    for index, (side, _) in enumerate(edges, 1):
        seam_draw.line([tuple(coordinates[side[0]]), tuple(coordinates[side[1]])], fill=index, width=1)
    seam_pixels = np.asarray(seam_image)
    if not np.any(seam_pixels):
        raise ValueError("No seam edges could be rasterized into the texture.")
    nearest_seam = distance_transform_edt(seam_pixels == 0, return_distances=False, return_indices=True)
    edge_at_pixel = seam_pixels[nearest_seam[0], nearest_seam[1]]
    del nearest_seam

    result = pixels.copy()
    for chart in sorted({opposite[3] for _, opposite in edges}):
        valid = (charts == chart) & ~missing
        if not np.any(valid):
            raise ValueError(f"UV chart {chart} contains no non-white texture pixels.")
        nearest_color = distance_transform_edt(~valid, return_distances=False, return_indices=True)
        for row in range(0, height, 256):
            ys, xs = np.nonzero(missing[row:row + 256])
            ys += row
            selected = edge_at_pixel[ys, xs]
            order = np.argsort(selected, kind="stable")
            ys, xs, selected = ys[order], xs[order], selected[order]
            edge_ids, starts = np.unique(selected, return_index=True)
            ends = np.r_[starts[1:], len(selected)]
            for edge_index, first, last in zip(edge_ids, starts, ends):
                side, opposite = edges[edge_index - 1]
                if opposite[3] != chart:
                    continue
                y, x = ys[first:last], xs[first:last]
                start, end = coordinates[[side[0], side[1]]]
                other_start, other_end, other_third = coordinates[[opposite[0], opposite[1], opposite[2]]]
                direction = end - start
                other_direction = other_end - other_start
                length_squared = float(direction @ direction)
                other_length = float(np.linalg.norm(other_direction))
                if length_squared < 1e-12 or other_length < 1e-6:
                    raise ValueError("A seam edge is too short to map between texture charts.")
                points = np.column_stack((x, y))
                t = np.clip(((points - start) @ direction) / length_squared, 0.0, 1.0)
                closest = start + t[:, None] * direction
                distance = np.linalg.norm(points - closest, axis=1)
                normal = np.array([-other_direction[1], other_direction[0]]) / other_length
                if np.dot(other_third - other_start, normal) < 0:
                    normal = -normal
                mapped = (other_start + t[:, None] * other_direction
                          + distance[:, None] * (other_length / np.sqrt(length_squared)) * normal)
                mapped_x = np.clip(np.rint(mapped[:, 0]).astype(int), 0, width - 1)
                mapped_y = np.clip(np.rint(mapped[:, 1]).astype(int), 0, height - 1)
                result[y, x] = pixels[nearest_color[0, mapped_y, mapped_x], nearest_color[1, mapped_y, mapped_x]]

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
    parser.add_argument("seams", type=Path, help="Matching seam CSV")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    flood_fill_texture_map(args.texture, args.mesh, args.seams, args.output)


if __name__ == "__main__":
    main()
