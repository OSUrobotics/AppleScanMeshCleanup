
"""
Create a cylindrical texture map for the apple
The apple meshes are spherical, more or less, with the stem pointing up when they come out of Revo Scan
The basic idea is to cut a circle out of the top and a circle out of the bottom, then unroll the middle
into a single texture. This creates three distinct texture map regions.

Approximate algorithm:
  1) Define an axis through the middle of the apple
  2) Define a circle that is 10% of the way down from the top and 1/3 of the maximum diameter
  3) Pick 8 vertices that are evenly separated by angle on the mesh that are close to that circle
  4) Join them together by shortest path
  5).   That defines the top seam
  6) Do the same thing for the bottom, with a slighly smaller circle to enclose the calyx
  7) Now join the circles along one edge
  8) Use conformal mapping to define the uv texture map coordinates
  """

import heapq
from collections import deque
from pathlib import Path
import struct
import zlib

import os
import csv
import numpy as np
import pymeshlab
import trimesh


def _write_blank_texture(texture_path: Path):
    """Write a solid-white 1024x1024 RGB PNG without external image dependencies."""
    def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        chunk = chunk_type + data
        return struct.pack(">I", len(data)) + chunk + struct.pack(">I", zlib.crc32(chunk))

    image_data = b"\x00" + b"\xff" * (1024 * 3)
    compressed_data = zlib.compress(image_data * 1024)
    png_data = (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", struct.pack(">IIBBBBB", 1024, 1024, 8, 2, 0, 0, 0))
        + png_chunk(b"IDAT", compressed_data)
        + png_chunk(b"IEND", b"")
    )
    texture_path.write_bytes(png_data)


def _write_material_file(material_path: Path, texture_filename: str):
    material_path.write_text(
        "# Blender 5.0.1 MTL File: 'process_apples.blend'\n"
        "# www.blender.org\n"
        "\n"
        "newmtl material_0\n"
        "Ns 0.000000\n"
        "Ka 1.000000 1.000000 1.000000\n"
        "Ks 1.000000 1.000000 1.000000\n"
        "Ke 0.000000 0.000000 0.000000\n"
        "Ni 1.500000\n"
        "d 1.000000\n"
        "illum 2\n"
        f"map_Kd {texture_filename}\n",
        encoding="utf-8",
    )


def _export_obj_with_uv(
    mesh: trimesh.Trimesh,
    vmapping: np.ndarray,
    indices: np.ndarray,
    uvs: np.ndarray,
    output_path: Path,
):
    if vmapping.ndim != 1 or uvs.ndim != 2 or uvs.shape[1] != 2:
        raise ValueError("Expected one-dimensional vertex mapping and an (N, 2) UV array.")
    if len(vmapping) != len(uvs):
        raise ValueError(
            f"Parameterizer returned {len(vmapping)} vertex mappings but {len(uvs)} UV coordinates."
        )
    if vmapping.size and (vmapping.min() < 0 or vmapping.max() >= len(mesh.vertices)):
        raise ValueError("Vertex mapping contains indices outside the source mesh.")
    output_vertex_ids = vmapping
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError("OBJ UV export expects triangular faces.")
    if indices.size and (indices.min() < 0 or indices.max() >= len(output_vertex_ids)):
        raise ValueError("Face indices are outside the output vertex/UV arrays.")

    material_path = output_path.with_suffix(".mtl")
    texture_path = output_path.with_suffix(".png")
    _write_blank_texture(texture_path)
    _write_material_file(material_path, texture_path.name)

    with output_path.open("w", encoding="utf-8", newline="\n") as obj_file:
        obj_file.write(f"mtllib {material_path.name}\n")
        for x, y, z in mesh.vertices:
            obj_file.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
        for u, v in uvs:
            obj_file.write(f"vt {u:.9g} {v:.9g}\n")
        obj_file.write("usemtl material_0\n")
        for face in indices:
            corners = " ".join(
                f"{int(output_vertex_ids[index]) + 1}/{int(index) + 1}" for index in face
            )
            obj_file.write(f"f {corners}\n")


def _load_mesh(input_path: Path):
    # 1. Load your cleaned, manifold apple mesh
    mesh = trimesh.load(str(input_path))
    print(f"Loaded mesh with {len(mesh.vertices)} vertices.")
    return mesh


def _create_adjacency_structure(mesh: trimesh.Trimesh):
    """Create the edge adjacency structure."""
    adjacency = [[] for _ in mesh.vertices]

    for (u, v), length in zip(mesh.edges_unique, mesh.edges_unique_length):
        u, v = int(u), int(v)
        weight = float(length)

        adjacency[u].append((v, weight))
        adjacency[v].append((u, weight))
    return adjacency


def _geodesic(adjacency: list[tuple[int, float]], v1: int, v2: int, no_cross_vs: list[int]):
    """Find the shortest path between two vertices along mesh edges."""

    if v1 == v2:
        return [v1], []

    distances = {v1: 0.0}
    previous = {}
    queue = [(0.0, v1)]
    while queue:
        distance_so_far, vertex = heapq.heappop(queue)
        if vertex == v2:
            break
        if vertex in no_cross_vs:
            continue
        if distance_so_far > distances[vertex]:
            continue
        for neighbor, edge_length in adjacency[vertex]:
            candidate_distance = distance_so_far + edge_length
            if candidate_distance < distances.get(neighbor, float("inf")):
                distances[neighbor] = candidate_distance
                previous[neighbor] = vertex
                heapq.heappush(queue, (candidate_distance, neighbor))

    if v2 not in previous:
        raise ValueError(f"No connected mesh-edge path exists between vertices {v1} and {v2}.")

    vertex_path = [v2]
    while vertex_path[-1] != v1:
        vertex_path.append(previous[vertex_path[-1]])
    vertex_path.reverse()
    seam_edges = list(zip(vertex_path[:-1], vertex_path[1:]))
    print(f"Found mesh-edge path of length {distances[v2]:.2f} using {len(seam_edges)} edges.")
    return vertex_path, seam_edges


def _define_seam(mesh: trimesh.Trimesh, 
                 adjacency: list[tuple[int, float]], 
                 no_cross_vs: list[int],
                 perc_top_bottom: float, 
                 perc_width: float, 
                 pt_center: np.array):
    """ Define a ring of connected edges on the top (or bottom) of the mesh
    Also return the first v point; we'll use that to do the side seam"""

    # Checks
    vertices = mesh.vertices
    if len(vertices) == 0:
        raise ValueError("Cannot define a seam on a mesh with no vertices.")
    if not 0.0 <= perc_top_bottom <= 1.0:
        raise ValueError("perc_top_bottom must be between 0 and 1.")
    if not 0.0 < perc_width <= 1.0:
        raise ValueError("perc_width must be greater than 0 and at most 1.")

    # Y is the vertical axis; the ring lies in the XZ plane.
    horizontal_extents = np.ptp(vertices[:, [0, 2]], axis=0)
    diameter = float(np.max(horizontal_extents))
    y_min = float(np.min(vertices[:, 1]))
    height = float(np.max(vertices[:, 1]) - y_min)
    target_y = 0.2 * pt_center[1] + 0.8 * (y_min + perc_top_bottom * height)
    radius = diameter * perc_width / 2.0

    # Place eight evenly spaced targets around the ring and choose the
    # closest unused mesh vertex to each target.
    angles = np.linspace(0.0, 2.0 * np.pi, num=16, endpoint=False)
    key_points = np.column_stack((
        pt_center[0] + radius * np.cos(angles),
        np.full(angles.shape, target_y),
        pt_center[2] + radius * np.sin(angles),
    ))
    vs = []
    for point in key_points:
        distances = np.linalg.norm(vertices - point, axis=1)
        for vertex_id in np.argsort(distances):
            vertex_id = int(vertex_id)
            if vertex_id not in vs:
                vs.append(vertex_id)
                break
        else:
            raise ValueError("Could not find eight distinct vertices for the seam.")

    # Add one more to make circle
    vs.append(vs[0])

    # Connect the points along the circle with geodesics
    seam_edges = []
    seam_vertices = []
    for v1, v2 in zip(vs[0:-1], vs[1:]):
        connect_vs, connect_edges = _geodesic(adjacency=adjacency, v1=v1, v2=v2, no_cross_vs=no_cross_vs)
        # Always one more vertex then edge, drop the last one
        seam_vertices.extend(connect_vs[0:-1])
        seam_edges.extend(connect_edges)
    # Duplicate the last vertex and edge
    seam_vertices.append(vs[0])
    return seam_vertices, seam_edges


def _side_seam(adjacency: list[tuple[int, float]], v1: int, v2: int, no_cross_vs: list[int]):
    """ The top and bottom seam are given as edges in the mesh. Connect them"""
    # Find a list of edges that connects the top seam to the bottom one along the x = 0 plane

    side_verts, side_seam = _geodesic(adjacency=adjacency, v1=v1, v2=v2, no_cross_vs=no_cross_vs)
    return side_verts, side_seam


def _label_seam_faces(mesh: trimesh.Trimesh,
                      side_seam_vs: set,
                      faces_component,
                      face_neighbors,
                      ):
        # Now get the ones in the middle mesh
    faces_left_seam = set()
    faces_right_seam = set()
    faces_vert_seam = set()
    for face_id, face in enumerate(mesh.faces):
        if not face_id in faces_component:
            continue

        b_found_v = False
        b_found_e = False
        for indx, vid in enumerate(face):
            # Get a list of all of the faces that are adjacent to one of the side seam vertices
            # They are either a left face, a right face, or just touching
            if vid in side_seam_vs:
                b_found_v = True
                indx_in_side_seam = side_seam_vs[vid]
                v_prev_indx = face[(indx+2)%3]
                if v_prev_indx in side_seam_vs:
                    indx_in_side_seam_prev = side_seam_vs[v_prev_indx]
                    if indx_in_side_seam == indx_in_side_seam_prev + 1:
                        faces_left_seam.add(face_id)
                        b_found_e = True
                    elif indx_in_side_seam == indx_in_side_seam_prev - 1:
                        faces_right_seam.add(face_id)
                        b_found_e = True
        if b_found_v and not b_found_e:
            faces_vert_seam.add(face_id)

    neigh_side = {}
    for face in faces_left_seam:
        neigh_side[face] = "Left"
    for face in faces_right_seam:
        neigh_side[face] = "Right"
    faces_vert_seam.difference_update(faces_left_seam | faces_right_seam)
    search_queue = deque(sorted(neigh_side))
    done = False
    while search_queue and not done:
        face_id = search_queue.popleft()
        for neighbor in face_neighbors[face_id]:
            if neighbor not in neigh_side:
                neigh_side[neighbor] = neigh_side[face_id]
                search_queue.append(neighbor)
        done = True
        for face_id in faces_vert_seam:
            if face_id not in neigh_side:
                done = False
                break

    for face_id in faces_vert_seam:
        side = neigh_side.get(face_id)
        if side == "Left":
            faces_left_seam.add(face_id)
        elif side == "Right":
            faces_right_seam.add(face_id)

    return faces_left_seam, faces_right_seam


def _parameterize_cut_mesh(
    mesh: trimesh.Trimesh,
    seam_edges: list[tuple[int, int]],
    side_seam_vertices: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    """Partition the mesh into three components using the edge seams provide.
       Essentially breadth first search on connected faces, skipping faces connected over seam edges
       Once the component face lists are created, need to flatten them out.
        Then adjust the uv coordinates so that they fit in the texture map.
         The large center unwrap gets flattened into a rectangle that stretches diagonally
          The top and bottom caps get placed in the upper left and lower right corners """

    if len(side_seam_vertices) < 2:
        raise ValueError("Expected at least two vertices along the side seam.")
    if len(set(side_seam_vertices)) != len(side_seam_vertices):
        raise ValueError("Side-seam vertices must not contain duplicates.")
    if any(vertex_id < 0 or vertex_id >= len(mesh.vertices) for vertex_id in side_seam_vertices):
        raise ValueError("A side-seam vertex index is outside the mesh.")
    endpoint_ids = {side_seam_vertices[0], side_seam_vertices[-1]}
    side_seam_vs = {}
    for indx, vid in enumerate(side_seam_vertices):
        side_seam_vs[vid] = indx

    seam_edge_ids = set()
    for edge in seam_edges:
        if len(edge) != 2:
            raise ValueError(f"Expected a seam edge with two vertex indices, got {edge!r}.")
        u, v = map(int, edge)
        if u == v or min(u, v) < 0 or max(u, v) >= len(mesh.vertices):
            raise ValueError(f"Invalid seam edge: {edge!r}.")
        seam_edge_ids.add(tuple(sorted((u, v))))

    # Loop through all the faces and assign each one to an edge.
    #.  Edges are defined by a tuple sorted by vertex index id
    edge_faces = {}
    for face_id, face in enumerate(mesh.faces):
        for u, v in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge_faces.setdefault(tuple(sorted((int(u), int(v)))), []).append(face_id)

    # Create a face neighbor structure, skipping any edges that are in the seams
    face_neighbors = [set() for _ in mesh.faces]
    boundary_face_ids = set()
    for edge, incident_faces in edge_faces.items():
        if edge in seam_edge_ids:
            boundary_face_ids.update(incident_faces)
            continue
        for face_id in incident_faces:
            face_neighbors[face_id].update(other for other in incident_faces if other != face_id)
    missing_seam_edges = seam_edge_ids - edge_faces.keys()
    if missing_seam_edges:
        raise ValueError(f"Seam edges are not present in the mesh: {sorted(missing_seam_edges)}.")

    # Do the breadth first search
    components = []
    unseen = set(range(len(mesh.faces)))
    while unseen:
        start = unseen.pop()
        component = {start}
        stack = [start]
        while stack:
            face_id = stack.pop()
            for neighbor in face_neighbors[face_id] & unseen:
                unseen.remove(neighbor)
                component.add(neighbor)
                stack.append(neighbor)
        components.append(np.asarray(sorted(component), dtype=np.int64))

    if not components:
        raise ValueError("Cannot parameterize a mesh with no faces.")
        
    chart_data = []
    for component_faces in components:
        component_vertex_ids = set(map(int, mesh.faces[component_faces].ravel()))
        is_middle_component = endpoint_ids.issubset(component_vertex_ids)
        source_faces = mesh.faces[component_faces]
        source_vertices = np.unique(source_faces)
        local_index = {int(vertex_id): index for index, vertex_id in enumerate(source_vertices)}
        local_corners = None
        if is_middle_component:
            faces_left_seam, faces_right_seam = _label_seam_faces(mesh=mesh, side_seam_vs=side_seam_vs, faces_component=component_faces, face_neighbors=face_neighbors)
            duplicated_local_indices = {}
            for vertex_id in side_seam_vertices:
                if vertex_id not in local_index:
                    raise ValueError(
                        f"Side-seam vertex {vertex_id} is missing from the middle component."
                    )
                duplicated_local_indices[vertex_id] = len(source_vertices) + len(duplicated_local_indices)
            source_vertices = np.concatenate(
                (source_vertices, np.asarray(side_seam_vertices, dtype=np.int64))
            )            

            local_faces = np.asarray(
                [
                    [
                        duplicated_local_indices.get(int(vertex_id), local_index[int(vertex_id)])
                        if int(face_id) in faces_right_seam
                        else local_index[int(vertex_id)]
                        for vertex_id in face
                    ]
                    for face_id, face in zip(component_faces, source_faces)
                ],
                dtype=np.int32,
            )

            seam_local_indices = {
                local_index[vertex_id] for vertex_id in side_seam_vertices
            } | set(duplicated_local_indices.values())
            referenced_local_indices = set(map(int, local_faces.ravel()))
            unreferenced_seam_indices = sorted(
                seam_local_indices - referenced_local_indices
            )
            if unreferenced_seam_indices:
                unreferenced_vertices = [
                    (local_id, int(source_vertices[local_id]))
                    for local_id in unreferenced_seam_indices
                ]
                raise ValueError(
                    "Unreferenced side-seam vertices in middle chart "
                    f"(local index, source vertex): {unreferenced_vertices}"
                )

            local_corners = [
                local_index[side_seam_vertices[0]],
                local_index[side_seam_vertices[-1]],
                duplicated_local_indices[side_seam_vertices[-1]],
                duplicated_local_indices[side_seam_vertices[0]],
            ]
        else:
            local_faces = np.asarray(
                [[local_index[int(vertex_id)] for vertex_id in face] for face in source_faces],
                dtype=np.int32,
            )

        local_vertices = np.asarray(mesh.vertices[source_vertices], dtype=np.float64)
        face_areas = trimesh.triangles.area(local_vertices[local_faces])
        zero_area_rows = np.flatnonzero(face_areas == 0.0)
        if zero_area_rows.size:
            source_face_ids = component_faces[zero_area_rows].tolist()
            raise ValueError(
                "Zero-area faces in chart: "
                f"local rows {zero_area_rows.tolist()}, "
                f"source mesh faces {source_face_ids}."
            )
        component_area = float(np.sum(face_areas))
        print(f"Component with {len(component_faces)} faces has area {component_area:.6g}.")

        if len(local_faces) == 1:
            p0, p1, p2 = local_vertices[local_faces[0]]
            edge = p1 - p0
            edge_length = float(np.linalg.norm(edge))
            if edge_length == 0:
                raise ValueError("Cannot parameterize a degenerate triangle.")
            axis = edge / edge_length
            third = p2 - p0
            x = float(np.dot(third, axis))
            y = float(np.linalg.norm(third - x * axis))
            if y == 0:
                raise ValueError("Cannot parameterize a degenerate triangle.")
            local_uvs = np.zeros((len(source_vertices), 2), dtype=np.float64)
            local_uvs[local_faces[0, 1]] = (edge_length, 0.0)
            local_uvs[local_faces[0, 2]] = (x, y)
        else:
            mesh_set = pymeshlab.MeshSet()
            
            mesh_set.add_mesh(
                pymeshlab.Mesh(vertex_matrix=local_vertices, face_matrix=local_faces)
            )
            mesh_set.compute_texcoord_parametrization_least_squares_conformal_maps()
            local_uvs = np.asarray(mesh_set.current_mesh().vertex_tex_coord_matrix(), dtype=np.float64)
            if local_uvs.shape != (len(source_vertices), 2):
                raise ValueError(
                    "PyMeshLab returned unexpected UV dimensions "
                    f"{local_uvs.shape} for {len(source_vertices)} vertices."
                )

        chart_data.append(
            (component_area, local_corners, source_vertices, local_faces, local_uvs)
        )

    cutout_charts = [chart for chart in chart_data if chart[1] is not None]
    if not cutout_charts:
        raise ValueError("Both side-seam endpoints must belong to one middle component.")
    if len(cutout_charts) > 1:
        raise ValueError("The side-seam endpoints ambiguously identify multiple components.")

    if len(chart_data) != 3:
        raise ValueError(
            f"Expected the cutout and two cap components, found {len(chart_data)} components."
        )
    chart_data.sort(
        key=lambda chart: (chart[1] is not None, chart[0]),
        reverse=True,
    )

    output_mapping = []
    output_faces = []
    output_uvs = []
    vertex_offset = 0
    # Putting the pixels at the corners slightly inside
    pad = 3.0 / 1024.0
    chart_id = 1
    for (_, local_corners, source_vertices, local_faces, local_uvs) in chart_data:
        if local_corners is not None:
            corner_uvs = local_uvs[local_corners]
            corner_center = corner_uvs.mean(axis=0)
            corner_order = sorted(
                range(4),
                key=lambda index: np.arctan2(
                    corner_uvs[index, 1] - corner_center[1],
                    corner_uvs[index, 0] - corner_center[0],
                ),
            )
            ordered_corners = corner_uvs[corner_order]
            source_extent = ordered_corners.max(axis=0) - ordered_corners.min(axis=0)
            if np.any(source_extent <= 0):
                raise ValueError("Cutout corners do not span a two-dimensional region.")

            diamond_corners = np.array(
                [[0.5, pad], [1.0 - pad, 0.5], [0.5, 1.0 - pad], [pad, 0.5]],
                dtype=np.float64,
            )
            homography_rows = []
            homography_targets = []
            for (x, y), (u, v) in zip(ordered_corners, diamond_corners):
                homography_rows.extend((
                    [x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y],
                    [0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y],
                ))
                homography_targets.extend((u, v))
            homography = np.append(
                np.linalg.solve(
                    np.asarray(homography_rows, dtype=np.float64),
                    np.asarray(homography_targets, dtype=np.float64),
                ),
                1.0,
            ).reshape(3, 3)
            homogeneous_uvs = np.column_stack((local_uvs, np.ones(len(local_uvs))))
            transformed_uvs = homogeneous_uvs @ homography.T
            denominators = transformed_uvs[:, 2]
            if np.any(np.isclose(denominators, 0.0)):
                raise ValueError("Cutout homography maps UVs to infinity.")
            packed_uvs = transformed_uvs[:, :2] / denominators[:, np.newaxis]
            centered_uvs = packed_uvs - 0.5
            max_extent = float(np.max(np.abs(centered_uvs)))
            if max_extent > 0.5 - pad:
                packed_uvs = 0.5 + centered_uvs * ((0.5 - pad) / max_extent)
            print(f"Largest cutout corner UVs: {packed_uvs[local_corners].tolist()}")
        else:
            uv_min = local_uvs.min(axis=0)
            uv_size = local_uvs.max(axis=0) - uv_min
            if np.any(uv_size <= 0):
                raise ValueError(f"Chart {chart_id} has zero UV extent.")
            available_size = 0.2 - 2 * pad
            scale = min(available_size / uv_size[0], available_size / uv_size[1])
            normalized_uvs = (local_uvs - uv_min) * scale
            if chart_id == 1:
                packed_uvs = normalized_uvs + [pad, 0.8 - pad]
                chart_id += 1
            else:
                packed_uvs = normalized_uvs + [0.8 - pad, pad]

        if np.any(packed_uvs < -1e-10) or np.any(packed_uvs > 1.0 + 1e-10):
            raise ValueError(f"Chart {chart_id} produced UV coordinates outside [0, 1].")
        packed_uvs = np.clip(packed_uvs, 0.0, 1.0)

        output_mapping.extend(source_vertices.tolist())
        output_uvs.extend(packed_uvs.tolist())
        output_faces.extend((local_faces + vertex_offset).tolist())
        vertex_offset += len(source_vertices)

    return (
        np.asarray(output_mapping, dtype=np.int64),
        np.asarray(output_faces, dtype=np.int64),
        np.asarray(output_uvs, dtype=np.float64),
    )


def _find_centers(input_path: Path, vertices: np.ndarray):
    input_path_endpts = input_path.with_name(input_path.name.replace("NoStemHiRes", ""))
    input_path_endpts = input_path_endpts.with_suffix(".txt")

    vertices = np.asarray(vertices, dtype=np.float64)

    # Y is the vertical axis; the ring lies in the XZ plane.
    horizontal_extents = np.ptp(vertices[:, [0, 2]], axis=0)
    diameter = float(np.max(horizontal_extents))
    y_min = float(np.min(vertices[:, 1]))
    height = float(np.max(vertices[:, 1]) - y_min)
    radius = 0.5 * diameter

    # Center of the mesh - should be close to 0,0,0
    pt_center = np.median(vertices, axis=0)
    verts_center = vertices[np.logical_and(np.abs(vertices[:, 0] - pt_center[0]) < 0.5 * radius,
                                        np.abs(vertices[:, 2] - pt_center[2]) < 0.5 * radius)]

    verts_top = verts_center[verts_center[:, 1] > y_min + height * 0.8]
    pt_center_top = verts_top[np.argmin(verts_top[:, 1])]

    verts_bot = verts_center[verts_center[:, 1] < y_min + height * 0.2]
    pt_center_bot = verts_bot[np.argmax(verts_bot[:, 1])]

    print(f"center top and bottom{pt_center_top}, {pt_center_bot}")

    end_pts = []
    if os.path.exists(input_path_endpts):
        with open(str(input_path_endpts), mode="r", encoding="utf-8") as file:
            end_pts = list(csv.reader(file))

        print(f"End points from text file {end_pts}")
        return [
            np.asarray([float(value) for value in point], dtype=np.float64)
            for point in end_pts
        ]
    return [pt_center_top, pt_center_bot]


def _write_seam_mesh(mesh: trimesh.Trimesh, vs: list[int], output_path: Path):
    seam_mesh = mesh.copy()
    vertex_colors = np.full((len(seam_mesh.vertices), 4), 255, dtype=np.uint8)
    seam_vertex_ids = np.asarray(vs, dtype=np.int64)
    if seam_vertex_ids.size and (seam_vertex_ids.min() < 0 or seam_vertex_ids.max() >= len(seam_mesh.vertices)):
        raise ValueError("Seam vertex indices are outside the mesh vertex array.")
    vertex_colors[seam_vertex_ids] = [0, 0, 0, 255]
    seam_mesh.visual.vertex_colors = vertex_colors

    seam_output_path = output_path.with_name(f"{output_path.stem}_seam.ply")
    seam_mesh.export(str(seam_output_path), file_type="ply")


def uv_texture(input_path: Path, output_path: Path):
    """Slice open the top and the bottom and unwrap the middle"""
    mesh = _load_mesh(input_path)

    centers = _find_centers(input_path=input_path, vertices=mesh.vertices)

    adjacency = _create_adjacency_structure(mesh=mesh)
    no_cross_vs = set()
    top_seam_vs, top_seam_es = _define_seam(mesh=mesh, adjacency=adjacency, no_cross_vs=no_cross_vs,
                                            perc_top_bottom=0.95, perc_width=0.5, pt_center=centers[0])
    _write_seam_mesh(mesh=mesh, vs=top_seam_vs, output_path=output_path)    
    for v in top_seam_vs:
        no_cross_vs.add(v)
    bottom_seam_vs, bottom_seam_es = _define_seam(mesh=mesh, adjacency=adjacency, no_cross_vs=no_cross_vs, 
                                                  perc_top_bottom=0.05, perc_width=0.6, pt_center=centers[1])
    _write_seam_mesh(mesh=mesh, vs=top_seam_vs + bottom_seam_vs, output_path=output_path)    
    for v in bottom_seam_vs:
        no_cross_vs.add(v)

    for indx in range(0, min(len(top_seam_vs), len(bottom_seam_vs))):
        no_cross_vs.remove(top_seam_vs[indx])
        no_cross_vs.remove(bottom_seam_vs[indx])
        try:
            side_seam_vs, side_seam_es = _side_seam(adjacency=adjacency, v1=top_seam_vs[indx], v2=bottom_seam_vs[indx], no_cross_vs=no_cross_vs)
            break
        except:
            no_cross_vs.add(top_seam_vs[indx])
            no_cross_vs.add(bottom_seam_vs[indx])

    seam_vs = top_seam_vs + bottom_seam_vs + side_seam_vs
    _write_seam_mesh(mesh=mesh, vs=seam_vs, output_path=output_path)    

    seam_edges = top_seam_es + bottom_seam_es + side_seam_es
    print("Parameterizing the mesh with the requested seam boundaries...")
    vmapping, indices, uvs = _parameterize_cut_mesh(mesh, seam_edges, side_seam_vs)

    # ==========================================
    # 5. EXPORT THE ASSET
    # ==========================================
    _export_obj_with_uv(mesh, vmapping, indices, uvs, output_path)
    print(f"✓ Asset exported with manual UV seams to: {output_path}")


def main():
    scan_dir = "/Users/cindygrimm/PycharmProjects/data/ScannedModels/"
    #parser = argparse.ArgumentParser(description="Generate a UV texture map for a mesh.")
    #parser.add_argument("input_path", type=Path, default=scan_dir + "/ProsserEnvyApples/BlenderTexturedFixed/Envy8NoStemHiRes.obj", help="Path to the input mesh.")
    #parser.add_argument("output_path", type=Path, default=scan_dir + "/ProsserEnvyApples/BlenderTexturedFixed/Envy8NoStemHiRes_uv.obj", help="Path for the unwrapped output mesh.")
    #args = parser.parse_args()
    #uv_texture(args.input_path, args.output_path)

    input_path = Path(scan_dir + "/ProsserEnvyApples/BlenderTexturedFixed/Envy8NoStemHiRes.obj")
    output_path = Path(scan_dir + "/ProsserEnvyApples/BlenderTexturedFixed/Envy8NoStemHiRes_uv.obj")
    uv_texture(input_path, output_path)


if __name__ == "__main__":
    main()