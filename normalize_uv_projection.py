"""Normalize an OBJ's three UV charts and warp its texture to match.

The top and bottom boundaries become circles of radius 0.1 centered at
(0.9, 0.1) and (0.1, 0.9). The body boundary becomes the diamond with
corners (0.5, 0), (1, 0.5), (0.5, 1), and (0, 0.5). Seam topology JSON
contains six ordered boundary paths, the calyx ring, and the top and bottom
center positions. The calyx ring maps to radius 0.05 on the bottom chart.
Each circular chart's first seam vertex lies on the positive U axis.
The warped texture is padded by four pixels outside the rasterized charts.
The output OBJ has a companion <output stem>_seam_topology.json with normalized UVs.
"""

import argparse
import json
import math
import warnings
from pathlib import Path

from meshlib import mrmeshnumpy as mn
from meshlib import mrmeshpy as mm

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

from filename_fixing import write_material_file
from seam_topology import SeamInfo, SeamTopology
from obj_geometry import read_obj, read_texture, read_mesh_no_textures, write_obj, create_adjacency_structure, bfs_graph, compute_edge_locs_line, compute_edge_locs_circle
from uv_texture import write_texture_uvs, _CAP_RADIUS, _CAP_CENTERS, _UV_PADDING
from projection_routines import place_in_circle, place_in_square, label_vs_and_fs

from scipy.sparse import csc_array
from scipy.sparse.linalg import spsolve


_SEAM_TOLERANCE = 2e-6
_CALYX_UV_RADIUS = 0.05
_TEXTURE_PADDING_PIXELS = 4
_DIAMOND_CORNERS = np.array(
    [[0.5, 0.0], [1.0, 0.5], [0.5, 1.0], [0.0, 0.5]]
)


def _parameterize(
    original_uvs: np.ndarray,
    faces: np.ndarray,
    seams: list[np.ndarray],
    center_uv_indices: list[np.ndarray],
    calyx_uv_candidates: list[np.ndarray],
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
            if chart_index == 1:
                ring = []
                for candidates in calyx_uv_candidates:
                    chart_matches = candidates[labels[candidates] == chart_label]
                    if len(chart_matches) != 1:
                        raise ValueError("Each calyx vertex must have exactly one UV on the bottom chart.")
                    ring.append(int(chart_matches[0]))
                if len(set(ring)) != len(ring) or set(ring) & (boundary_set | {int(matches[0])}):
                    raise ValueError("The calyx ring must have distinct interior UV vertices away from the center.")
                ring_edges = edges[np.isin(edges[:, 0], ring) & np.isin(edges[:, 1], ring)]
                ring_order = _ordered_boundary(ring_edges)
                if set(ring_order) != set(ring):
                    raise ValueError("Calyx vertices must form one connected ring on the bottom chart.")
                ring_source = perturbed_uvs[ring_order]
                ring_area = np.sum(
                    ring_source[:, 0] * np.roll(ring_source[:, 1], -1)
                    - ring_source[:, 1] * np.roll(ring_source[:, 0], -1)
                )
                lengths = np.linalg.norm(np.roll(ring_source, -1, axis=0) - ring_source, axis=1)
                if abs(ring_area) < 1e-14 or np.any(lengths <= 1e-14):
                    raise ValueError("The calyx ring must enclose a nonzero UV area with nonzero edges.")
                center_source = perturbed_uvs[int(matches[0])]
                ring_direction = ring_source[0] - center_source
                anchor_direction = perturbed_uvs[int(seam_indices[1][0])] - center_source
                if np.linalg.norm(ring_direction) <= 1e-14 or np.linalg.norm(anchor_direction) <= 1e-14:
                    raise ValueError("The calyx ring and bottom boundary must be away from the bottom center.")
                offset = (math.atan2(ring_direction[1], ring_direction[0])
                          - math.atan2(anchor_direction[1], anchor_direction[0]))
                angles = offset + (1 if ring_area > 0 else -1) * (
                    np.r_[0.0, np.cumsum(lengths[:-1])] / lengths.sum() * (2 * math.pi)
                )
                ring_targets = center + _CALYX_UV_RADIUS * np.column_stack((np.cos(angles), np.sin(angles)))
                fixed_indices = np.r_[fixed_indices, ring_order]
                fixed_targets = np.vstack((fixed_targets, ring_targets))
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
        source = np.asarray(input_image.convert("RGB"))

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


def place_uvs_in_new_locs(mesh: mm.Mesh, 
                          seam_topology: SeamTopology, 
                          vertex_uv: np.ndarray, 
                          vert_uv_ids: np.ndarray,
                          face_vert_ids: np.ndarray,
                          face_uv_ids: np.ndarray,
                          labels: dict):
    """ This takes the top and bottom regions and places them with a circle boundary, with each vertex
     at the center of its neighbors and (optionally), the center vertex at the origin
      The middle bit is mapped to a diamond-shaped region. """
    
    # Define the new seam by connecting the top and bottom seams at the mid point
    top_seam_vs = seam_topology.seams["top"].vertex_ids
    bottom_seam_vs = seam_topology.seams["bottom"].vertex_ids
    side_seam_vs = seam_topology.seams["side"].vertex_ids
    # All of the seam vertices in a set
    all_seam_vs = set(top_seam_vs) | set(bottom_seam_vs) | set(side_seam_vs)

    # Label vertices and faces into calyx, top, and middle regions based on the original UVs and faces
    # Note that vertices will show up in multiple regions if they are shared across faces
    bottom_vs, bottom_fs, top_vs, top_fs, middle_vs, middle_fs, edge_adj = labels

    # Find the bottom vertex that is closest to the bottom point seam
    bottom_pt = seam_topology.bottom_point
    # Find the bottom vertex that is closest to the bottom point
    verts = mn.getNumpyVerts(mesh)
    bottom_dists = np.linalg.norm(verts - bottom_pt, axis=1)
    bottom_vid = np.argmin(bottom_dists)
    if bottom_vid not in bottom_vs:
        raise ValueError("Closest bottom vertex is not in the bottom vertex set.")

    new_top_locs, map_top, _ = place_in_circle(mesh=mesh, edge_adjacency=edge_adj, boundary_vs=top_seam_vs, inside_vs=top_vs, center=seam_topology.top_vertex, radius=_CAP_RADIUS - 0.01)
    new_bottom_locs, map_bottom, _ = place_in_circle(mesh=mesh, edge_adjacency=edge_adj, boundary_vs=bottom_seam_vs, inside_vs=bottom_vs, center=bottom_vid, radius=_CAP_RADIUS - 0.01)
    new_bottom_locs[:, :] += _CAP_CENTERS[1]
    new_top_locs[:, :] += _CAP_CENTERS[0]

    diamond_corners = np.array(
        [[0.5, _UV_PADDING], [1.0 - _UV_PADDING, 0.5], [0.5, 1.0 - _UV_PADDING], [_UV_PADDING, 0.5]],
        dtype=np.float64,
            )

    new_diamond_locs, map_diamond, _, vs_left, vs_right = place_in_square(mesh=mesh, 
                                                                           edge_adjacency=edge_adj, 
                                                                           face_vertex_indices=face_vert_ids,
                                                                           seam_topology=seam_topology, 
                                                                           inside_vs=middle_vs,
                                                                           corners=diamond_corners)   

    # Now put the new uv locations back into new_vertex_uv
    new_vertex_uv = np.copy(vertex_uv)

    # Which uv vertices belong to the
    seams = [[], [], []]    
    for v in seam_topology.seams["top"].vertex_ids:
        seams[0].append(vert_uv_ids[v])
    for v in seam_topology.seams["bottom"].vertex_ids:
        seams[1].append(vert_uv_ids[v])
    for v in seam_topology.seams["side"].vertex_ids:
        seams[2].append(vert_uv_ids[v])
    seam_vt_ids = []
    for seam in seams:
        seam_side_a = [s[0] for s in seam[1:-1]]
        seam_side_b = [s[1] for s in seam[1:-1]]
        seam_vt_ids.append(seam_side_a)
        seam_vt_ids.append(seam_side_b)
    for indx in range(len(seam_vt_ids)):
        for indx2 in range(indx+1, len(seam_vt_ids)):
            if set(seam_vt_ids[indx]).intersection(set(seam_vt_ids[indx2])):
                raise ValueError(f"Overlap found between seam segments {indx} and {indx2}.")

    v_set = set()
    # For the top and bottom, jsut find the corresponding vertex texture id and put it there
    for fs, map_v, locs in zip((top_fs, bottom_fs), (map_top, map_bottom), (new_top_locs, new_bottom_locs)):
        for face in fs:
            # Get the face vertex vt ids
            for vt_id, vid in zip(face_uv_ids[face], face_vert_ids[face]):
                new_vertex_uv[vt_id] = locs[map_v[vid]]  
                v_set.add(vt_id)
    
    # The middle is a bit trickier because the seam vertices and corners are split across multiple faces
    left_faces = seam_topology.face_lists["left_side_seam"]
    right_faces = seam_topology.face_lists["right_side_seam"]
    for face in middle_fs:
        face_uv_id = face_uv_ids[face]
        face_id = face_vert_ids[face]
        for vid, vt_id in zip(face_id, face_uv_id):
            map_v = map_diamond.get(vid, None)
            if map_v is None:
                raise ValueError(f"Vertex {vid} not found in map_diamond.")
            if len(map_v) == 1:
                new_vertex_uv[vt_id] = new_diamond_locs[map_v[0]]
                v_set.add(vt_id)
            elif len(map_v) == 2:
                # It's either in the left or the right side
                print(f"loc 1 for vertex {vid} is {new_diamond_locs[map_v[0]]}")
                print(f"loc 2 for vertex {vid} is {new_diamond_locs[map_v[1]]}")
                if face in left_faces:
                    new_vertex_uv[vt_id] = new_diamond_locs[map_v[0]]
                    v_set.add(vt_id)
                    continue
                if face in right_faces:
                    new_vertex_uv[vt_id] = new_diamond_locs[map_v[1]]
                    v_set.add(vt_id)
                    continue
                raise ValueError(f"Vertex {vid} with multiple mappings is not on either side of the seam.")
            else:
                raise ValueError(f"Vertex {vid} has unexpected number of mappings in map_diamond: {len(map_v)}")    

    if len(v_set) != len(new_vertex_uv):
        print ("Warning: Some UVs were not assigned new locations.")
        # raise ValueError("Some UVs were not assigned new locations.")
    return new_vertex_uv, seams


def normalize_uv_projection(input_path: Path, output_path: Path, seam_path: Path) -> Path:
    """Normalize one textured OBJ using matching seam topology in seam_path."""
    input_path, seam_dir, output_path = Path(input_path), Path(seam_path), Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("The output OBJ must differ from the input OBJ.")
    if output_path.suffix.lower() != ".obj":
        raise ValueError("The normalized mesh output must be an OBJ file.")
    seam_file = seam_path / (input_path.stem + "_seam_topology.json")
    with seam_file.open(encoding="utf-8") as stream:
        topology = SeamTopology.from_dict(json.load(stream))
    if "calyx" not in topology.seams:
        raise ValueError(f"Seam topology {seam_path} must contain a calyx ring.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_texture = output_path.with_suffix(".jpg")
    output_material = output_path.with_suffix(".mtl")
    output_seams = output_path.with_name(output_path.stem + "_seam_topology.json")  

    mesh = read_mesh_no_textures(input_path)

    # Read in the obj geometry separated into vertex locations, UVs, normals, and faces
    vertex_loc, vertex_uv, vert_uv_ids, vertex_normals, face_triangles, face_uv_triangles, mtllibs = read_obj(input_path)
    material_path = input_path.with_suffix(".mtl")
    texture_path = read_texture(material_path)

    # Label vertices and faces into calyx, top, and middle regions based on the original UVs and faces
    # Note that vertices will show up in multiple regions if they are shared across faces
    labels = label_vs_and_fs(mesh=mesh, 
                             seam_topology=topology)

    # The new locations for the uv coordinates, with boundaries mapped to the two circles and the diamonds
    new_uv_locs, seams = place_uvs_in_new_locs(mesh=mesh,
                                                seam_topology=topology,
                                                vertex_uv=vertex_uv,
                                                vert_uv_ids=vert_uv_ids,
                                                face_vert_ids=face_triangles,
                                                face_uv_ids=face_uv_triangles,
                                                labels=labels)

    write_texture_uvs(seam_vs=seams, uvs=new_uv_locs, texture_path=output_texture)
    # Now copy the texture over by warping the faces to the new UV locations
    # For each of the original faces in each of the regions, get the new UV coordinates from the mapped locations

    # Do the actual texture copy
    _warp_texture(texture_path=texture_path, 
                  output_texture=output_texture, 
                  original_uvs=vertex_uv, 
                  normalized_uvs=new_uv_locs, 
                  faces=face_uv_triangles)

    # This is just a straight copy - nothing should have changed
    with output_seams.open("w", encoding="utf-8") as stream:
        json.dump(topology.to_dict(), stream)

    write_material_file(output_material, output_texture.name)
    write_obj(output_path=output_path, 
              vertex_locs=vertex_loc, 
              uv_verts=new_uv_locs, 
              vertex_normals=vertex_normals, 
              triangles=face_triangles, 
              uv_triangles=face_uv_triangles, 
              texture_name=output_texture.name, 
              material_name=output_material.name)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize an OBJ's UV charts and texture.")
    parser.add_argument("input", nargs="?", default="/Users/cindygrimm/PycharmProjects/data/ScannedModels/ProsserEnvyApples/BlenderRebaked/Envy_002_FullRes.obj", type=Path, help="Rebaked textured OBJ")
    parser.add_argument("output", nargs="?", default="/Users/cindygrimm/PycharmProjects/data/ScannedModels/ProsserEnvyApples/NormalizedTexture/Envy_002_FullRes_seam.obj", type=Path, help="Output OBJ path")
    parser.add_argument("seams", nargs="?", default="/Users/cindygrimm/PycharmProjects/data/ScannedModels/ProsserEnvyApples/Parameterized", type=Path, help="Directory containing seam topology JSON files")
    args = parser.parse_args()
    normalize_uv_projection(args.input, args.output, args.seams)


if __name__ == "__main__":
    main()