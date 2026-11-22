
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
import json
from collections import deque
from collections.abc import Sequence
from pathlib import Path
import struct
import zlib
import os
import csv
import numpy as np
import pymeshlab
import trimesh
from filename_fixing import _write_material_file, _split_filename_into_apple_type_and_number
from obj_geometry import read_obj_geometry
from seam_topology import SeamInfo, SeamTopology


def _write_blank_texture(n_vs: int, seam_vs: list[list[list[int]]], uvs: np.ndarray, texture_path: Path):
    """Write a white 1024x1024 RGB PNG with the top seam marked in green, the bottom in blue, and the side seams in black.
      The first set of vs in the top seam are the edge of the middle mesh
      The first set of vs in the bottom seam are the left edge of the middle mesh
      The two triplets are the duplicated corners of the middle mesh"""
    image_size = 1024

    image_data = np.full((image_size, image_size, 3), 255, dtype=np.uint8)
    # Color all the pixels that belong to vertices grey
    pixels = np.clip(uvs * image_size, 0, image_size - 1).astype(np.uint)
    image_data[image_size - 1 - pixels[:, 1], pixels[:, 0]] = (128, 128, 128)

    def color_pixels(uv_locs: np.ndarray, color: tuple[int, int, int]) -> None:
        pixels = np.clip(uv_locs * image_size, 0, image_size - 1).astype(np.uint)
        if pixels.ndim == 1:
            image_data[image_size - 1 - pixels[1], pixels[0]] = color
        else:
            image_data[image_size - 1 - pixels[:, 1], pixels[:, 0]] = color

    def color_seam(seam_vertices: list[list[int]], uvs: np.ndarray, color: tuple[int, int, int]) -> None:
        seam_vs_side1 = []
        seam_vs_side2 = []
        for vs in seam_vertices:
            seam_vs_side1.append(vs[0])
            seam_vs_side2.append(vs[1])
            
        uv_locs = uvs[seam_vs_side1]
        color_pixels(uv_locs, color)
            
        uv_locs = uvs[seam_vs_side2]
        color_pixels(uv_locs, (color[0] // 2, color[1] // 2, color[2] // 2))

    if len(seam_vs) != 3:
        raise ValueError("Expected top, bottom, and side seam vertex lists.")
    color_seam(seam_vs[0], uvs, (0, 255, 0))
    color_seam(seam_vs[1], uvs, (0, 0, 255))
    color_seam(seam_vs[2], uvs, (255, 0, 0))

    # Corners
    print(f"Corner vs: {seam_vs[2][0]}, {seam_vs[2][-1]}")
    """
    corner_v1 = seam_vs[2][0]
    corner_v2 = seam_vs[2][-1]
    for col, corners in zip([(255, 255, 0), (0, 255, 255)], [corner_v1, corner_v2]):
        for corner in corners:
            color_pixels(uvs[corner, :], col)
    """
    corner_v1 = seam_vs[2][0][1]
    corner_v2 = seam_vs[2][-1][1]
    color_pixels(uvs[corner_v1, :], (255, 255, 0))
    color_pixels(uvs[corner_v2, :], (0, 255, 255))

    scanlines = np.empty((image_size, image_size * 3 + 1), dtype=np.uint8)
    scanlines[:, 0] = 0
    scanlines[:, 1:] = image_data.reshape(image_size, image_size * 3)
    compressed_data = zlib.compress(scanlines.tobytes())

    def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        chunk = chunk_type + data
        return struct.pack(">I", len(data)) + chunk + struct.pack(">I", zlib.crc32(chunk))

    png_data = (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", struct.pack(">IIBBBBB", image_size, image_size, 8, 2, 0, 0, 0))
        + png_chunk(b"IDAT", compressed_data)
        + png_chunk(b"IEND", b"")
    )
    texture_path.write_bytes(png_data)


def _export_obj_with_uv(
    mesh: trimesh.Trimesh,
    seam_vs: list[list[list[int]]],
    vmapping: np.ndarray,
    face_indices: np.ndarray,
    uvs: np.ndarray,
    output_path: Path,
):
    """ Keeps the original vertices in mesh
        The texture map coordinates are in uvs. There are more texture map coords than vertices
        face_indices is a list of face indices
        vmapping is the mapping from the face vertex ids (which has the duplicate vertex ids) to the original vertices
        """
    if vmapping.ndim != 1 or uvs.ndim != 2 or uvs.shape[1] != 2:
        raise ValueError("Expected one-dimensional vertex mapping and an (N, 2) UV array.")
    if len(vmapping) != len(uvs):
        raise ValueError(
            f"Parameterizer returned {len(vmapping)} vertex mappings but {len(uvs)} UV coordinates."
        )
    if vmapping.size and (vmapping.min() < 0 or vmapping.max() >= len(mesh.vertices)):
        raise ValueError("Vertex mapping contains indices outside the source mesh.")

    if face_indices.ndim != 2 or face_indices.shape[1] != 3:
        raise ValueError("OBJ UV export expects triangular faces.")
    if face_indices.size and (face_indices.min() < 0 or face_indices.max() >= len(vmapping)):
        raise ValueError("Face indices are outside the output vertex/UV arrays.")

    material_path = output_path.with_suffix(".mtl")
    texture_path = output_path.with_suffix(".jpg")
    if seam_vs is not None:
        _write_blank_texture(n_vs=len(mesh.vertices), seam_vs=seam_vs, uvs=uvs, texture_path=texture_path)
    _write_material_file(material_path, texture_path.name)

    with output_path.open("w", encoding="utf-8", newline="\n") as obj_file:
        obj_file.write(f"mtllib {material_path.name}\n")
        for x, y, z in mesh.vertices:
            obj_file.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
        for u, v in uvs:
            obj_file.write(f"vt {u:.9g} {v:.9g}\n")
        obj_file.write("usemtl material_0\n")
        for face in face_indices:
            corners = " ".join(
                f"{int(vmapping[index]) + 1}/{int(index) + 1}" for index in face
            )
            obj_file.write(f"f {corners}\n")


def _load_mesh(input_path: Path):
    if input_path.suffix.lower() == ".obj":
        with input_path.open(encoding="utf-8") as source:
            vertices, faces = read_obj_geometry(source, input_path)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    else:
        mesh = trimesh.load(str(input_path), process=False)
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


def _orient_mesh(mesh: trimesh.Trimesh, top_vertex: np.ndarray, bottom_vertex: np.ndarray):
    """Orient the mesh by finding the transformation that puts the top vertex at 0,0,0 and orients the mesh 
       so that the bottom vertex lies along the negative y-axis. Then translate back up"""
    translation = -top_vertex
    mesh.vertices += translation
    new_top_vertex = top_vertex + translation
    new_bottom_vertex = bottom_vertex + translation

    axis_length = np.linalg.norm(new_bottom_vertex)
    if not np.isfinite(axis_length) or axis_length == 0:
        raise ValueError("Mesh endpoints must define a finite, non-zero axis.")
    direction = new_bottom_vertex / axis_length
    target = np.array([0, -1, 0])
    axis = np.cross(direction, target)
    if np.linalg.norm(axis) < 1e-8:
        if np.dot(direction, target) < 0:
            rotation_matrix = np.diag([1.0, -1.0, -1.0])
            mesh.vertices = mesh.vertices @ rotation_matrix.T
            new_bottom_vertex = new_bottom_vertex @ rotation_matrix.T
        return mesh, new_top_vertex, new_bottom_vertex
    axis /= np.linalg.norm(axis)
    angle = np.arccos(np.clip(np.dot(direction, target), -1.0, 1.0))
    rotation_matrix = trimesh.transformations.rotation_matrix(angle, axis)[:3, :3]
    mesh.vertices = mesh.vertices @ rotation_matrix.T

    new_bottom_vertex = new_bottom_vertex @ rotation_matrix.T

    return mesh, new_top_vertex, new_bottom_vertex

       
def _geodesic(adjacency: list[list[tuple[int, float]]], v1: int, v2: int, no_cross_vs: set[int]):
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
                 adjacency: list[list[tuple[int, float]]],
                 no_cross_vs: set[int],
                 b_is_top: bool,
                 vertex_heights: np.ndarray,
                 y_min: float,
                 y_max: float,
                 radius: float):
    """ Define a ring of connected edges on the top (or bottom) of the mesh
    Also return the first v point; we'll use that to do the side seam"""

    # Checks
    vertices = mesh.vertices
    if len(vertices) == 0:
        raise ValueError("Cannot define a seam on a mesh with no vertices.")

    # Y is the vertical axis; the ring lies in the XZ plane.
    height = y_max - y_min
    if not np.isfinite([y_min, y_max]).all() or height <= 0:
        raise ValueError("Seam topology must have finite, increasing Y bounds.")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Shoulder radius must be finite and positive.")
    if b_is_top:
        y_target = y_max
    else:
        y_target = y_min

    # Place eight evenly spaced targets around the ring and choose the
    # closest unused mesh vertex to each target.
    angles = np.linspace(0.0, 2.0 * np.pi, num=16, endpoint=False)
    key_points = np.column_stack((
        radius * np.cos(angles),
        np.full(angles.shape, y_target),
        radius * np.sin(angles),
    ))
    vs = []

    for point in key_points:
        # Look for the vertex that has a y in the valid y range and is closest
        # to the point's x and z values
        if b_is_top:
            valid_vs = vertex_heights > y_min + 0.7 * height
        else:
            valid_vs = vertex_heights < y_min + 0.3 * height
        distances_x = np.fabs(vertices[:, 0] - point[0])
        distances_z = np.fabs(vertices[:, 2] - point[2])
        distances = distances_x + distances_z
        for vertex_id in np.argsort(distances):
            vertex_id = int(vertex_id)
            if not valid_vs[vertex_id]:
                continue
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
    return seam_vertices, seam_edges, radius


_CAP_RADIUS = 0.1
_UV_PADDING = 3.0 / 1024.0
_CAP_CENTERS = np.array([
    [0.9 - _UV_PADDING, 0.1 + _UV_PADDING],
    [0.1 + _UV_PADDING, 0.9 - _UV_PADDING],
])


def _check_diamond(diamond_uvs: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Contract the body along the cap-to-cap diagonal when its triangles overlap a cap."""
    triangles = diamond_uvs[faces]
    overlaps = False
    for center in _CAP_CENTERS:
        relative = triangles - center
        following = np.roll(relative, -1, axis=1)
        cross = relative[:, :, 0] * following[:, :, 1] - relative[:, :, 1] * following[:, :, 0]
        inside = (np.all(cross >= 0, axis=1) | np.all(cross <= 0, axis=1)) & (
            np.abs(np.sum(cross, axis=1)) > 1e-15
        )
        edge = following - relative
        lengths_squared = np.sum(edge * edge, axis=2)
        fraction = np.zeros_like(lengths_squared)
        np.divide(-np.sum(relative * edge, axis=2), lengths_squared,
                  out=fraction, where=lengths_squared > 0)
        fraction = np.clip(fraction, 0, 1)
        nearest = relative + fraction[:, :, None] * edge
        distance_squared = np.sum(nearest * nearest, axis=2).min(axis=1)
        if np.any(inside | (distance_squared < _CAP_RADIUS ** 2)):
            overlaps = True
            break
    if not overlaps:
        return diamond_uvs

    diagonal = np.array([1.0, -1.0]) / np.sqrt(2.0)
    positions = diamond_uvs @ diagonal
    cap_positions = _CAP_CENTERS @ diagonal
    limit = min(cap_positions[0], -cap_positions[1]) - _CAP_RADIUS - _UV_PADDING
    if limit <= 0:
        raise ValueError("Cap charts leave no room for the middle chart.")
    extent = float(np.max(np.abs(positions)))
    scale = min(1.0, limit / extent)
    return diamond_uvs - (1.0 - scale) * positions[:, None] * diagonal


def _side_seam(adjacency: list[list[tuple[int, float]]], v1: int, v2: int, no_cross_vs: set[int]):
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


def _pack_cap_uvs(local_uvs: np.ndarray, center: np.ndarray, chart_id: int) -> np.ndarray:
    if not np.isfinite(local_uvs).all() or np.any(np.ptp(local_uvs, axis=0) <= 0):
        raise ValueError(f"Chart {chart_id} has invalid or zero UV extent.")
    offsets = local_uvs - np.median(local_uvs, axis=0)
    max_radius = float(np.max(np.linalg.norm(offsets, axis=1)))
    return center + offsets * (_CAP_RADIUS / max_radius)


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
        is_top_component = side_seam_vertices[0] in component_vertex_ids
        is_bot_component = side_seam_vertices[-1] in component_vertex_ids
        if not is_middle_component and not is_top_component and not is_bot_component:
            raise ValueError(f"End vertices not in component")
        
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
            (component_area, local_corners, source_vertices, local_faces, local_uvs, is_middle_component, is_top_component, is_bot_component)
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

    output_mapping = []
    output_faces = []
    output_uvs = []
    vertex_offset = 0
    # Putting the pixels at the corners slightly inside
    pad = _UV_PADDING
    chart_id = 1
    for (_, local_corners, source_vertices, local_faces, local_uvs, is_middle, is_top, is_bottom) in chart_data:
        if is_middle:
            if local_corners is None:
                raise ValueError(f"Middle chart did not have the corner vertices")

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
            packed_uvs = _check_diamond(packed_uvs, local_faces)
            print(f"Largest cutout corner UVs: {packed_uvs[local_corners].tolist()}")
        else:
            if is_bottom:
                packed_uvs = _pack_cap_uvs(local_uvs, _CAP_CENTERS[1], chart_id)
                chart_id += 1
            elif is_top:
                packed_uvs = _pack_cap_uvs(local_uvs, _CAP_CENTERS[0], chart_id)
            else:
                raise ValueError(f"Was not top, bottom, or middle")

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
    apple_name, apple_number = _split_filename_into_apple_type_and_number(input_path.name)
    for directory in ["BlenderStemRemoved", "BlenderNoStemBaked", "Voxelized"]:
        # Get the directory one up and over (root plus directory)
        seam_path = input_path.parent.parent / directory / f"{apple_name}_{apple_number}.txt"
        if os.path.exists(seam_path):
            input_path_endpts = seam_path
            break
        seam_path = input_path.parent.parent / directory / f"{apple_name}{apple_number}.txt"
        if os.path.exists(seam_path):
            input_path_endpts = seam_path
            break

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
        pts = []
        for point in end_pts:
            pt = []
            for xyz in point:
                pt_str = ""
                for c in xyz:
                    if c.isdigit() or c == "." or c == "-":
                        pt_str += c
                if pt_str != "":
                    pt.append(float(pt_str))
            pts.append(pt)
        if len(pts) < 2:
            print(f"WARNING: Only one point in end points file {input_path_endpts}")
            return [pt_center_top, pt_center_bot]
        return np.asarray(pts[0], dtype=np.float64), np.asarray(pts[1])
    print(f"WARNING: No center points file found at {input_path_endpts}, guessing good point using mesh geometry")
    return [pt_center_top, pt_center_bot]


def _write_seam_mesh(
    mesh: trimesh.Trimesh, vs: list[int], centers: Sequence[np.ndarray], output_path: Path
):
    seam_mesh = mesh.copy()
    vertex_colors = np.full((len(seam_mesh.vertices), 4), 255, dtype=np.uint8)
    seam_vertex_ids = np.asarray(vs, dtype=np.int64)
    if seam_vertex_ids.size and (seam_vertex_ids.min() < 0 or seam_vertex_ids.max() >= len(seam_mesh.vertices)):
        raise ValueError("Seam vertex indices are outside the mesh vertex array.")
    center_points = np.asarray(centers, dtype=np.float64)
    if center_points.shape != (2, 3) or not np.isfinite(center_points).all():
        raise ValueError("Expected finite top and bottom 3D center coordinates.")
    if not len(seam_mesh.vertices):
        raise ValueError("Cannot mark centers on a mesh without vertices.")
    center_vertex_ids = [
        int(np.argmin(np.linalg.norm(seam_mesh.vertices - center, axis=1)))
        for center in center_points
    ]
    center_region = set(center_vertex_ids)
    frontier = set(center_vertex_ids)
    neighbors = seam_mesh.vertex_neighbors
    for _ in range(6):
        frontier = {neighbor for vertex in frontier for neighbor in neighbors[vertex]} - center_region
        center_region.update(frontier)
    vertex_colors[seam_vertex_ids] = [0, 0, 0, 255]
    vertex_colors[list(center_region)] = [0, 255, 0, 255]
    seam_mesh.visual.vertex_colors = vertex_colors

    seam_output_path = output_path.with_name(f"{output_path.stem}_seam.ply")
    seam_mesh.export(str(seam_output_path), file_type="ply")


def _store_boundary_paths(seam_topology: SeamTopology, seams: list[list[int]],
                          seams_uv: list[list[list[int]]], uvs: np.ndarray) -> None:
    records = []
    for seam_index, uv_index, reverse in (
        (0, -1, False), (1, -1, False), (0, 0, False),
        (2, 0, False), (1, 0, True), (2, 1, True),
    ):
        ids = seams[seam_index]
        mapped = seams_uv[seam_index]
        if reverse:
            ids = list(reversed(ids))
            mapped = list(reversed(mapped))
        records.append((ids, [uvs[indices[uv_index]] for indices in mapped]))
    seam_topology.set_boundary_paths(records)
    
def _update_seam_topology(
    seam_topology: SeamTopology,
    mesh: trimesh.Trimesh,
    seam_vertices: list[list[int]],
    centers: list[np.ndarray],
    radii: tuple[float, float],
    map_vs_back: dict[int, list[int]],
    uvs: np.ndarray,
) -> None:
    def vertex_uvs(vertex_ids: list[int]) -> list:
        result = []
        for vertex_id in vertex_ids:
            if vertex_id not in map_vs_back:
                raise ValueError(f"Seam vertex {vertex_id} has no UV coordinates.")
            result.append(uvs[map_vs_back[vertex_id]].tolist())
        return result

    updated_uvs = {}
    for name in ("calyx", "stem"):
        if name not in seam_topology.seams:
            continue
        seam = seam_topology.seams[name]
        vertex_ids = seam.vertex_ids
        if any(vertex_id < 0 or vertex_id >= len(mesh.vertices) for vertex_id in vertex_ids):
            raise ValueError(f"{name} seam vertex indices are outside the UV mesh.")
        stored_locations = np.asarray(seam.vertex_locations, dtype=np.float64)
        if stored_locations.shape != (len(vertex_ids), 3) or not np.allclose(
            mesh.vertices[vertex_ids], stored_locations, rtol=1e-5, atol=1e-5
        ):
            raise ValueError(f"{name} seam vertices do not match the UV mesh.")
        updated_uvs[name] = vertex_uvs(vertex_ids)

    new_seams = {}
    for name, vertex_ids, center, radius in zip(
        ("top", "bottom", "side"),
        seam_vertices,
        (centers[0], centers[1], None),
        (radii[0], radii[1], None),
    ):
        new_seams[name] = SeamInfo(
            name=name,
            vertex_ids=[int(vertex_id) for vertex_id in vertex_ids],
            vertex_locations=mesh.vertices[vertex_ids].tolist(),
            uv_coords=vertex_uvs(vertex_ids),
            center=tuple(float(value) for value in center) if center is not None else None,
            radius=float(radius) if radius is not None else None,
        )

    for name, coords in updated_uvs.items():
        seam_topology.seams[name].uv_coords = coords
    for name, seam in new_seams.items():
        seam_topology.add_seam(name, seam)


def uv_texture(input_path: Path, output_path: Path, seam_topology: SeamTopology):
    """Slice open the top and the bottom and unwrap the middle"""
    top_radius = seam_topology.top_shoulder_radius
    bottom_radius = seam_topology.bottom_shoulder_radius
    if top_radius is None or bottom_radius is None:
        raise ValueError("Seam topology must contain top and bottom shoulder radii.")
    if not np.isfinite([top_radius, bottom_radius]).all() or min(top_radius, bottom_radius) <= 0:
        raise ValueError("Top and bottom shoulder radii must be finite and positive.")
    mesh = _load_mesh(input_path)

    centers = [
        np.array(seam_topology.top_point, dtype=np.float64),
        np.array(seam_topology.bottom_point, dtype=np.float64),
    ]
    mesh_oriented = mesh.copy()
    mesh_oriented, new_top_vertex, new_bottom_vertex = _orient_mesh(mesh_oriented, centers[0], centers[1])
    print(f"New top vertex {new_top_vertex}")
    print(f"New bottom vertex {new_bottom_vertex}")

    adjacency = _create_adjacency_structure(mesh=mesh_oriented)
    no_cross_vs = set()
    top_seam_vs, top_seam_es, top_radius = _define_seam(
        mesh=mesh_oriented, adjacency=adjacency, no_cross_vs=no_cross_vs, b_is_top=True,
        vertex_heights=mesh.vertices[:, 1], y_min=seam_topology.ymin, y_max=seam_topology.ymax,
        radius=top_radius,
    )
    _write_seam_mesh(mesh=mesh_oriented, vs=top_seam_vs, centers=[new_top_vertex, new_bottom_vertex], output_path=output_path)
    for v in top_seam_vs:
        no_cross_vs.add(v)
    bottom_seam_vs, bottom_seam_es, bottom_radius = _define_seam(
        mesh=mesh_oriented, adjacency=adjacency, no_cross_vs=no_cross_vs, b_is_top=False,
        vertex_heights=mesh.vertices[:, 1], y_min=seam_topology.ymin, y_max=seam_topology.ymax,
        radius=bottom_radius,
    )
    _write_seam_mesh(mesh=mesh_oriented, vs=top_seam_vs + bottom_seam_vs, centers=[new_top_vertex, new_bottom_vertex], output_path=output_path)
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

    seam_vs_flat = top_seam_vs + bottom_seam_vs + side_seam_vs
    _write_seam_mesh(mesh=mesh_oriented, vs=seam_vs_flat, centers=[new_top_vertex, new_bottom_vertex], output_path=output_path)

    seam_edges = top_seam_es + bottom_seam_es + side_seam_es
    print("Parameterizing the mesh with the requested seam boundaries...")
    vmapping, indices, uvs = _parameterize_cut_mesh(mesh, seam_edges, side_seam_vertices=side_seam_vs)

    # Export the mesh with the uv coordinates, create an image with the seams/vertices marked, and export the seams
    # This is the reverse of vmapping - it says which original vertices correspond to each mapped vertex.
    #.  Since the seam vertices were duplicated, there should be 2 for each original seam vertex, and 4 for the corners of the middle mesh
    map_vs_back = {}
    for indx in range(0, vmapping.shape[0]):
        v_map = vmapping[indx]
        if v_map not in map_vs_back:
            map_vs_back[v_map] = []
        map_vs_back[v_map].append(indx)
        
    # Figure out where each of the seam vertices went. 
    seam_vs_mapped = []
    seam_vs = [top_seam_vs[1:-1], bottom_seam_vs[1:-1], side_seam_vs]
    for seam in seam_vs:
        seam_mapped = [map_vs_back[v] for v in seam]
        seam_vs_mapped.append(seam_mapped)

    _update_seam_topology(
        seam_topology, mesh, [top_seam_vs[:-1], bottom_seam_vs[:-1], side_seam_vs],
        centers, (top_radius, bottom_radius), map_vs_back, uvs,
    )
    _store_boundary_paths(seam_topology, seam_vs, seam_vs_mapped, uvs)
    _export_obj_with_uv(mesh=mesh, seam_vs=seam_vs_mapped, vmapping=vmapping, face_indices=indices, uvs=uvs, output_path=output_path)    

    seam_path = output_path.with_name(output_path.stem + "_seam_topology.json")
    n_vertices = len(mesh.vertices)
    seam_topology.write_to_file(seam_path=seam_path, n_vertices=n_vertices)
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
    topology_path = input_path.with_name(input_path.stem.removesuffix("_voxel") + "_seam_topology.json")
    with topology_path.open(encoding="utf-8") as handle:
        seam_topology = SeamTopology.from_dict(json.load(handle))
    uv_texture(input_path, output_path, seam_topology)
    output_topology = output_path.with_name(output_path.stem + "_seam_topology.json")
    with output_topology.open("w", encoding="utf-8") as handle:
        json.dump(seam_topology.to_dict(), handle)


if __name__ == "__main__":
    main()