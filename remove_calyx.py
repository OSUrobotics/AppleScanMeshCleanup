"""Split a textured OBJ at its saved calyx ring and grid-patch the remaining apple."""

import argparse
import json
import math
import shlex
from pathlib import Path

import numpy as np
from meshlib import mrmeshnumpy as mn
from meshlib import mrmeshpy as mm
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve

import numpy as np
from seam_topology import SeamInfo, SeamTopology
from obj_geometry import get_vertex_star, create_adjacency_structure, bfs_graph, read_obj, write_obj
from filename_fixing import write_material_file


def _remove_and_renumber(vertex_loc: np.ndarray, 
                         vertex_uv: np.ndarray, 
                         vertex_normals: np.ndarray, 
                         vertex_uv_ids: np.ndarray,
                         face_triangles: np.ndarray, 
                         face_uv_triangles: np.ndarray,
                         face_norm_triangles: np.ndarray,
                         topology: SeamTopology,
                         kill_faces: list[int]
                         ):
    """ Remove the faces (and the adjacent vertices) from the mesh and renumber the remaining vertices and faces."""
    # Get the adjacent vertices for the faces to be killed
    kill_vs = set()
    for f_idx in kill_faces:
        kill_vs.update(face_triangles[f_idx])

    # Remove the calyx faces from the face triangles
    map_vs = {}  # Keep track of new vertex ids for renumbering
    map_uvs_vs = {}  # Keep track of new vertex ids for renumbering
    n_vertices = vertex_loc.shape[0]
    count = 0
    count_vt = 0
    vts_to_delete = set()
    new_vertex_loc = []
    new_vertex_norm = []
    new_vertex_uv = []
    for indx in range(0, n_vertices):
        if indx in kill_vs:
            # Delete this vertex and its texture coordinates
            for uv_idx in vertex_uv_ids[indx]:
                vts_to_delete.add(uv_idx)
                map_uvs_vs[uv_idx] = -1
            map_vs[indx] = -1
        else:
            map_vs[indx] = count
            new_vertex_loc.append(vertex_loc[indx, :])
            new_vertex_norm.append(vertex_normals[indx, :])
            count += 1
            for uv_idx in vertex_uv_ids[indx]:
                map_uvs_vs[uv_idx] = count_vt
                new_vertex_uv.append(vertex_uv[uv_idx, :])
                count_vt += 1

    topology.renumber_seams(map_vs)
    map_fs = {} 
    count_f = 0  # Keep track of new face ids for renumbering
    new_face_triangles = []
    new_face_uv_triangles = []
    new_face_normals = []
    for f_idx in range(0, face_triangles.shape[0]):
        if f_idx in kill_faces:
            map_fs[f_idx] = -1
        else:
            map_fs[f_idx] = count_f
            new_face_triangles.append([map_vs[vid] for vid in face_triangles[f_idx]])
            new_face_uv_triangles.append([map_uvs_vs[uv] for uv in face_uv_triangles[f_idx]])
            new_face_normals.append(face_norm_triangles[f_idx])
            count_f += 1

    topology.renumber_faces(map_fs)
    return new_vertex_loc, new_vertex_uv, new_vertex_norm, new_face_triangles, new_face_uv_triangles, new_face_normals


def _cutout_calyx_and_save(input_path: Path, output_path: Path):
    """Find the component inside the ring without relying on seam vertex order."""
    input_path, output_path = Path(input_path), Path(output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input mesh not found: {input_path}")
    if input_path.suffix.lower() != ".obj" or output_path.suffix.lower() != ".obj":
        raise ValueError("Input and output meshes must be OBJ files.")
    seam_path = input_path.with_name(
        input_path.stem + "_seam_topology.json"
    )
    calyx_sub_path = output_path.parent / "Calyx"
    calyx_sub_path.mkdir(exist_ok=True)
    calyx_path = calyx_sub_path / (output_path.stem + "_calyx.obj")
    output_seams = calyx_sub_path / (output_path.stem + "_seam_topology.json")
    calyx_texture = calyx_sub_path / (output_path.stem + ".jpg")

    # This input mesh and topology should have the stem ring and the calyx ring
    # Do this so it doesn't duplicate vertices
    vertex_loc, vertex_uv, vert_uv_ids, vertex_normals, face_triangles, face_uv_triangles, face_norm_triangles, mtllibs = read_obj(input_path)
    with seam_path.open(encoding="utf-8") as stream:
        topology = SeamTopology.from_dict(json.load(stream))

    if "calyx" not in topology.seams or "stem" not in topology.seams:
        raise ValueError("Seam topology must contain a calyx ring and a stem ring.")

    mesh = mn.meshFromFacesVerts(
        face_triangles.astype(np.int32), vertex_loc.astype(np.float32), duplicateNonManifoldVertices=False
    )

    bottom_vid = topology.bottom_vertex
    calyx_seam = topology.seams["calyx"]
    edge_adj = create_adjacency_structure(mesh, set())
    adj_vs = bfs_graph(edge_adj, bottom_vid, set(calyx_seam))
    # Get all the faces adjacent to adj_vs
    adj_faces = []
    for f_idx, tri in enumerate(face_triangles):
        if any(v in adj_vs for v in tri):
            adj_faces.append(f_idx)

    non_calyx_faces = [f_idx for f_idx in range(face_triangles.shape[0]) if f_idx not in adj_faces] 
    new_vertex_loc, new_vertex_uv, new_vertex_norm, new_face_triangles, new_face_uv_triangles, new_face_normals = _remove_and_renumber(vertex_loc=vertex_loc, 
                                                                                                                                       vertex_uv=vertex_uv, 
                                                                                                                                       vert_uv_ids=vert_uv_ids, 
                                                                                                                                       vertex_normals=vertex_normals, 
                                                                                                                                       face_triangles=face_triangles,
                                                                                                                                       face_uv_triangles=face_uv_triangles,
                                                                                                                                       face_normals=face_norm_triangles,
                                                                                                                                       topology=topology,
                                                                                                                                       remove_fs=adj_faces)

    write_obj(output_path, new_vertex_loc, new_vertex_uv, new_vertex_norm, new_face_triangles, new_face_uv_triangles, new_face_normals, mtllibs)    

    lower = new_vertex_uv.min(axis=0)
    upper = new_vertex_uv.max(axis=0)
    extent = float(np.max(upper - lower))
    if extent <= 0:
        raise ValueError("Calyx UVs have zero extent.")
    # Eight output pixels of padding, with a uniform scale to avoid distortion.
    side = extent * 256 / 240
    origin = (lower + upper - side) / 2
    cropped_uvs = (new_vertex_uv - origin) / side
    with Image.open(calyx_texture) as image:
        width, height = image.size
        transform = (side * width / 256, 0, origin[0] * width,
                     0, side * height / 256, (1 - origin[1] - side) * height)
        mode = "RGBA" if "A" in image.getbands() or "transparency" in image.info else "RGB"
        cropped = image.convert(mode).transform((256, 256), Image.Transform.AFFINE, transform,
                                                resample=Image.Resampling.BICUBIC)
    cropped.save(calyx_texture)


def _remove_calyx_and_save(input_path, seam_path, calyx_texture, output_path):
    # This input mesh and topology should have the stem ring and the calyx ring
    # Do this so it doesn't duplicate vertices
    vertex_loc, vertex_uv, vert_uv_ids, vertex_normals, face_triangles, face_uv_triangles, mtllibs = read_obj(input_path)
    seam_file = seam_path / (input_path.stem + "_seam_topology.json")
    with seam_file.open(encoding="utf-8") as stream:
        topology = SeamTopology.from_dict(json.load(stream))

    vertex_loc, vertex_uv, vert_uv_ids, vertex_normals, face_triangles, face_uv_triangles, face_norm_triangles, mtllibs = read_obj(input_path)
    seam_file = seam_path / (input_path.stem + "_seam_topology.json")
    with seam_file.open(encoding="utf-8") as stream:
        topology = SeamTopology.from_dict(json.load(stream))

    if "calyx" not in topology.seams or "stem" not in topology.seams:
        raise ValueError("Seam topology must contain a calyx ring and a stem ring.")

    mesh = mn.meshFromFacesVerts(
        face_triangles.astype(np.int32), vertex_loc.astype(np.float32), duplicateNonManifoldVertices=False
    )

    bottom_vid = topology.bottom_vertex
    calyx_seam = topology.seams["calyx"]
    edge_adj = create_adjacency_structure(mesh, set())
    adj_vs = bfs_graph(edge_adj, bottom_vid, set(calyx_seam))
    # Get all the faces adjacent to adj_vs
    adj_faces = []
    for f_idx, tri in enumerate(face_triangles):
        if any(v in adj_vs for v in tri):
            adj_faces.append(f_idx)

    new_vertex_loc, new_vertex_uv, new_vertex_norm, new_face_triangles, new_face_uv_triangles, new_face_normals = _remove_and_renumber(vertex_loc=vertex_loc, 
                                                                                                                                       vertex_uv=vertex_uv, 
                                                                                                                                       vert_uv_ids=vert_uv_ids, 
                                                                                                                                       vertex_normals=vertex_normals, 
                                                                                                                                       face_triangles=face_triangles,
                                                                                                                                       face_uv_triangles=face_uv_triangles,
                                                                                                                                       face_normals=face_norm_triangles,
                                                                                                                                       topology=topology,
                                                                                                                                       remove_fs=adj_faces)

    

    patch_vertices, patch_uvs, patch_faces, center_vid = _grid_patch(vertices=np.array(new_vertex_loc),
                                                                     vertex_uv=np.array(new_vertex_uv),
                                                                     face_triangles=np.array(new_face_triangles),
                                                                     face_uv_triangles=np.array(new_face_uv_triangles),
                                                                     ring=topology.seams["calyx"],
                                                                     )

    for indx, v in enumerate(patch_vertices):
        new_vertex_loc.append(v)
        new_vertex_uv.append(patch_uvs[indx])

    for indx, faces in enumerate(patch_faces):
        new_face_triangles.append([count + vid for vid in faces])
    topology.bottom_vertex = count + center_vid
    count += len(patch_vertices)

    # Write out the new mesh.


def _cross(a, b, c):
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _delaunay_patch(points, triangles, sign, tolerance, scale):
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


def _unfold_patch_uvs(points, uvs, triangles, geometry_sign, tolerance):
    """Flip interior edges where doing so reduces UV inversions without folding the geometry."""
    triangles = triangles.copy()
    uv_sign = np.sign(sum(_cross(*uvs[face]) for face in triangles))
    if not uv_sign:
        raise ValueError("The calyx ring has zero UV area.")
    uv_tolerance = 1e-14
    while True:
        areas = np.array([uv_sign * _cross(*uvs[face]) for face in triangles])
        bad = np.flatnonzero(areas <= uv_tolerance)
        if not len(bad):
            return triangles
        edge_faces = {}
        for index, face in enumerate(triangles):
            for a, b in zip(face, np.roll(face, -1)):
                edge_faces.setdefault(tuple(sorted((int(a), int(b)))), []).append(index)
        best = None
        for index in bad:
            face = triangles[index]
            for i in range(3):
                a, b, c = map(int, (face[i], face[(i + 1) % 3], face[(i + 2) % 3]))
                neighbors = edge_faces[(min(a, b), max(a, b))]
                if len(neighbors) != 2:
                    continue
                other = neighbors[0] if neighbors[1] == index else neighbors[1]
                d = next(int(v) for v in triangles[other] if v != a and v != b)
                replacement = np.array([[c, a, d], [c, d, b]])
                if any(geometry_sign * _cross(*points[tri]) <= tolerance for tri in replacement):
                    continue
                new_areas = np.array([uv_sign * _cross(*uvs[tri]) for tri in replacement])
                old_areas = areas[[index, other]]
                old_score = (int(np.sum(old_areas <= uv_tolerance)),
                             float(np.maximum(uv_tolerance - old_areas, 0).sum()))
                new_score = (int(np.sum(new_areas <= uv_tolerance)),
                             float(np.maximum(uv_tolerance - new_areas, 0).sum()))
                improvement = (old_score[0] - new_score[0], old_score[1] - new_score[1])
                if improvement > (0, 1e-18) and (best is None or improvement > best[0]):
                    best = (improvement, index, other, replacement)
        if best is None:
            raise ValueError("The calyx center and ring UVs produce a degenerate or inverted grid patch.")
        _, index, other, replacement = best
        triangles[index], triangles[other] = replacement


def _grid_patch(vertices, faces, uv_faces, uvs, ring, inside, center_vertex, center):
    """Triangulate the ring in XZ, inserting a Cartesian grid and the center."""
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
    points = list(boundary)
    remaining = list(range(len(ring)))
    triangles = []
    while len(remaining) > 3:
        for i, current in enumerate(remaining):
            a, b, c = remaining[i - 1], current, remaining[(i + 1) % len(remaining)]
            if sign * _cross(points[a], points[b], points[c]) <= tolerance:
                continue
            if any(
                min(sign * _cross(points[a], points[b], points[p]),
                    sign * _cross(points[b], points[c], points[p]),
                    sign * _cross(points[c], points[a], points[p])) >= -tolerance
                for p in remaining if p not in (a, b, c)
            ):
                continue
            triangles.append((a, b, c))
            remaining.pop(i)
            break
        else:
            raise ValueError("Calyx ring is self-intersecting or cannot be triangulated in XZ.")
    triangles.append(tuple(remaining))

    center = np.asarray(center, dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("Calyx ring must have a finite 3D center.")
    grid_size = max(3, len(ring) // 20)
    spacing = 2 * float(np.max(np.abs(boundary - center[[0, 2]]))) / grid_size
    if spacing <= scale * 1e-8:
        raise ValueError("Calyx ring is too small for a grid.")
    def insert(point, required=False):
        matches = []
        for index, (a, b, c) in enumerate(triangles):
            crosses = np.array([sign * _cross(points[a], points[b], point),
                                sign * _cross(points[b], points[c], point),
                                sign * _cross(points[c], points[a], point)])
            if np.min(crosses) >= -tolerance:
                matches.append((index, (a, b, c), crosses))
        if not matches or any(
            np.linalg.norm(point - points[v]) <= scale * 1e-9
            for _, triangle, _ in matches for v in triangle
        ):
            if required:
                raise ValueError("Calyx center must be strictly inside the projected ring.")
            return
        new_index = len(points)
        replacement = []
        for _, triangle, crosses in matches:
            for edge in range(3):
                if crosses[edge] > tolerance:
                    replacement.append((triangle[edge], triangle[(edge + 1) % 3], new_index))
        if not replacement or (required and len(matches) == 1 and np.min(matches[0][2]) <= tolerance):
            raise ValueError("Calyx center lies on the boundary of the projected ring.")
        for index, _, _ in reversed(matches):
            triangles.pop(index)
        points.append(point)
        triangles.extend(replacement)
        return new_index

    center_v_id = insert(center[[0, 2]], required=True)
    edge_vectors = np.roll(boundary, -1, axis=0) - boundary
    edge_lengths = np.sum(edge_vectors * edge_vectors, axis=1)
    if np.any(edge_lengths <= tolerance):
        raise ValueError("Calyx ring has coincident projected vertices.")
    grid_extent = math.ceil(grid_size / 2)
    for x in range(-grid_extent, grid_extent + 1):
        for z in range(-grid_extent, grid_extent + 1):
            if x or z:
                point = center[[0, 2]] + spacing * np.array([x, z])
                t = np.clip(np.sum((point - boundary) * edge_vectors, axis=1) / edge_lengths, 0, 1)
                distance = np.min(np.linalg.norm(point - boundary - t[:, None] * edge_vectors, axis=1))
                if distance > 0.2 * spacing:
                    insert(point)
    points = np.asarray(points)
    triangles = _delaunay_patch(points, triangles, sign, tolerance, scale)

    def unique_uv(vertex):
        values = uvs[uv_faces[inside][faces[inside] == vertex]]
        if not len(values) or not np.allclose(values, values[0], rtol=0, atol=2e-6):
            raise ValueError(f"Calyx vertex {vertex} must have one UV coordinate on the calyx.")
        return values[0]

    ring_uvs = np.array([unique_uv(v) for v in ring])
    center_uv = unique_uv(center_vertex)
    # Harmonic interpolation preserves the boundary UVs and the original center UV.
    fixed = np.vstack((np.column_stack((vertices[ring, 1], ring_uvs)),
                       [float(np.mean(vertices[ring, 1])), *center_uv]))
    adjacency = [set() for _ in points]
    for a, b, c in triangles:
        for v, w in ((a, b), (b, c), (c, a)):
            adjacency[v].add(w)
            adjacency[w].add(v)
    n_fixed = len(fixed)
    rows, cols, data = [], [], []
    rhs = np.zeros((len(points) - n_fixed, 3))
    for vertex in range(n_fixed, len(points)):
        row = vertex - n_fixed
        rows.append(row)
        cols.append(row)
        data.append(len(adjacency[vertex]))
        for neighbor in adjacency[vertex]:
            if neighbor < n_fixed:
                rhs[row] += fixed[neighbor]
            else:
                rows.append(row)
                cols.append(neighbor - n_fixed)
                data.append(-1)
    if len(rhs):
        values = np.vstack((fixed, spsolve(coo_matrix((data, (rows, cols)),
                                                     shape=(len(rhs), len(rhs))).tocsr(), rhs)))
    else:
        values = fixed
    patch_vertices = np.column_stack((points[:, 0], values[:, 0], points[:, 1]))
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
    patch_uvs = values[:, 1:]
    triangles = _unfold_patch_uvs(points, patch_uvs, triangles, sign, tolerance)
    return patch_vertices, patch_uvs, triangles, center_v_id


def remove_calyx(input_path: Path, output_path: Path, seam_path: Path | None = None) -> Path:
    """Write the patched OBJ and a separate textured ``<output>_calyx.obj``."""

    input_path, output_path = Path(input_path), Path(output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input mesh not found: {input_path}")
    if input_path.suffix.lower() != ".obj" or output_path.suffix.lower() != ".obj":
        raise ValueError("Input and output meshes must be OBJ files.")
    seam_path = Path(seam_path) if seam_path is not None else input_path.with_name(
        input_path.stem + "_seam_topology.json"
    )

    _cutout_calyx_and_save(input_path, output_path)
    _remove_calyx_and_save(input_path, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, default=None, help="Directory the paths are relative to")
    parser.add_argument("--input-path", type=Path, required=True, help="Apple scan mesh")
    parser.add_argument("--output-path", type=Path, required=True, help="Patched mesh to write")
    parser
    args = parser.parse_args()
    input_path, output_path = args.input_path, args.output_path
    if args.root_dir is not None:
        input_path, output_path = args.root_dir / input_path, args.root_dir / output_path

    print(f"Wrote {remove_calyx(input_path, output_path)}")


if __name__ == "__main__":
    main()
