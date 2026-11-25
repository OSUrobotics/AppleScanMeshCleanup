"""Read OBJ geometry without splitting vertices at UV or normal seams."""

from collections.abc import Iterable
from pathlib import Path
import shlex

from meshlib import mrmeshpy as mm
from meshlib import mrmeshnumpy as mn

import numpy as np


def read_obj(input_path: Path):
    lines = input_path.read_text(encoding="utf-8").splitlines()
    # These are x,y,z or u,v coordinates
    vert_locs = []
    vert_norms = []
    vert_uvs = []
    # Faces are triplets of vertex indices (or uv indices)
    face_vids = []
    face_tex_ids = []
    # Mapping from vertex index to list of associated UV coordinates
    vert_uv_ids = {}  # List of UV coordinates corresponding to each vertex in the faces
    for i in range(len(vert_locs)):
        vert_uv_ids[i] = []
    mtllibs = []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v":
            if len(parts) < 4:
                raise ValueError(f"Invalid vertex coordinate in {input_path}: {line}")
            vert_locs.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif parts[0] == "vn":
            if len(parts) < 4:
                raise ValueError(f"Invalid normal coordinate in {input_path}: {line}")
            vert_norms.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif parts[0] == "vt":
            if len(parts) < 3:
                raise ValueError(f"Invalid texture coordinate in {input_path}: {line}")
            vert_uvs.append([float(parts[1]), float(parts[2])])
        elif parts[0] == "f":
            if len(parts) != 4:
                raise ValueError(f"Expected triangular OBJ faces in {input_path}.")
            face = []
            uv_face = []
            for corner in parts[1:]:
                # The format is either vid/vtexid or vid/vtexid/vnormid
                fields = corner.split("/")
                if len(fields) < 2 or not fields[1]:
                    raise ValueError(f"Face is missing texture coordinates: {line}")
                v_index = int(fields[0])
                uv_index = int(fields[1])
                if v_index < 1:
                    raise ValueError("OBJ vertex indices cannot be zero or negative.")
                if uv_index < 1:
                    raise ValueError("OBJ texture indices cannot be zero or negative.")
                face.append(v_index - 1)
                uv_face.append(uv_index - 1)
                if v_index -1 not in vert_uv_ids:
                    vert_uv_ids[v_index - 1] = [uv_index - 1]
                else:
                    vert_uv_ids[v_index - 1].append(uv_index - 1)
            face_vids.append(face)
            face_tex_ids.append(uv_face)
        elif parts[0] == "mtllib":
            mtllibs.extend(shlex.split(line)[1:])

    # Convert these to nump arrays of floats
    vertex_loc = np.asarray(vert_locs, dtype=np.float64)
    vertex_uv = np.asarray(vert_uvs, dtype=np.float64)
    vertex_normals = np.asarray(vert_norms, dtype=np.float64)

    # These are the face indices for vertices and UVs
    face_triangles = np.asarray(face_vids, dtype=np.int64)
    face_uv_triangles = np.asarray(face_tex_ids, dtype=np.int64)

    # These are checks that the vertex and face data are valid
    if vertex_loc.ndim != 2 or vertex_loc.shape[1] != 3 or not np.isfinite(vertex_loc).all():
        raise ValueError(f"Expected finite vertex coordinates in {input_path}.")
    if face_triangles.ndim != 2 or face_triangles.shape[1] != 3:
        raise ValueError(f"Expected triangular faces with vertex coordinates in {input_path}.")
    if face_triangles.min() < 0 or face_triangles.max() >= len(vert_locs):
        raise ValueError(f"Face vertex index is out of range in {input_path}.")
    if face_uv_triangles.min() < 0 or face_uv_triangles.max() >= len(vertex_uv):
        raise ValueError(f"Face UV index is out of range in {input_path}.")
    if len(mtllibs) != 1:
        raise ValueError(f"Expected exactly one OBJ material library in {input_path}.")

    for indx, uv_ids in vert_uv_ids.items():
        unique_ids = set(uv_ids)
        vert_uv_ids[indx] = list(unique_ids)
    return vertex_loc, vertex_uv, vert_uv_ids, vertex_normals, face_triangles, face_uv_triangles, mtllibs[0]


def read_mesh_no_textures(input_path: Path) -> mm.Mesh:
    vertex_loc, vertex_uv, vert_uv_ids, vertex_normals, face_triangles, face_uv_triangles, mtllibs = read_obj(input_path)

    mesh = mn.meshFromFacesVerts(
        face_triangles.astype(np.int32), vertex_loc.astype(np.float32), duplicateNonManifoldVertices=False
    )
    print("MeshLib valid vertex count:", mesh.topology.numValidVerts())
    return mesh


def read_texture(material_path: Path) -> Path:
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


def build_vmap(verts_orig: np.ndarray, verts_new: np.ndarray)->dict[int, int]:
    vmap = {}
    for i, v in enumerate(verts_orig):
        matches = np.where(np.all(verts_new == v, axis=1))[0]
        if len(matches) == 0:
            raise ValueError(f"Original vertex {i} not found in new vertices.")
        vmap[i] = matches[0]
    return vmap


def get_vertex_star(topo, v):
  if not topo.hasVert(v):
    return []

  # Get the first outgoing half-edge from vertex v
  start_edge = topo.edgeWithOrg(v)
  if not start_edge.valid():
    return []  # Isolated vertex with no connected faces

  neighbors = []
  curr_edge = start_edge

  # Circulate through all outgoing half-edges around v
  while True:
    dest_vert = topo.dest(curr_edge)
    neighbors.append(dest_vert.get())
    # In MeshLib, next() rotates to the next outgoing edge around the same origin vertex
    curr_edge = topo.next(curr_edge)
    if curr_edge == start_edge:
      break

  return neighbors


def create_adjacency_structure(mesh: mm.Mesh, no_cross_vs: set[int]) -> list[list[tuple[int, float]]]:
    """ Create an edge adjacency structure as a list of neighbors for each vertex. """
    adjacency = [[] for _ in range(mesh.topology.numValidVerts())]

    verts = mn.getNumpyVerts(mesh)

    for v in range(0, mesh.topology.numValidVerts()):
        v_neighs = get_vertex_star(mesh.topology, mm.VertId(v))
        for u in v_neighs:
            if u < v:
                continue
            if u in no_cross_vs or v in no_cross_vs:
                continue

            np.linalg.norm(verts[u] - verts[v])
            length = np.linalg.norm(verts[u] - verts[v])
            weight = float(length)

            adjacency[u].append((v, weight))
            adjacency[v].append((u, weight))
    return adjacency


def bfs_graph(adjacency: list[list[tuple[int, float]]], start: int, no_cross_vs: set[int]) -> list[int]:
    """ Perform BFS on the graph represented by the adjacency list starting from vertex 'start'. """
    visited = [False] * len(adjacency)
    queue = [start]
    visited[start] = True
    order = []

    while queue:
        v = queue.pop(0)
        order.append(v)
        for u, _ in adjacency[v]:
            if u in no_cross_vs:
                continue
            if not visited[u]:
                visited[u] = True
                queue.append(u)

    return order


def compute_edge_locs_line(verts: np.ndarray, v_ids: list[int], corner_start: np.ndarray, corner_end: np.ndarray):
    edge_lengths = []
    total_length = 0.0
    for i in range(len(v_ids) - 1):
        v0, v1 = verts[v_ids[i]], verts[v_ids[i + 1]]
        length = np.linalg.norm(v1 - v0)
        edge_lengths.append(length)
        total_length += length

    t = 0.0
    edge_locs = []
    for i in range(len(v_ids) - 1):
        loc = (1 - t) * corner_start + t * corner_end
        edge_locs.append(loc)
        t += edge_lengths[i] / total_length
    edge_locs.append(corner_end)
    return edge_locs


def compute_edge_locs_circle(verts: np.ndarray, v_ids: list[int], radius: float):
    edge_lengths = []
    total_length = 0.0
    for i in range(len(v_ids) - 1):
        v0, v1 = verts[v_ids[i]], verts[v_ids[i + 1]]
        length = np.linalg.norm(v1 - v0)
        edge_lengths.append(length)
        total_length += length

    if v_ids[0] != v_ids[-1]:
        edge_lengths.append(np.linalg.norm(verts[v_ids[-1]] - verts[v_ids[0]]))
        total_length += edge_lengths[-1]
        
    angle = 0.0
    edge_locs = []
    for i in range(len(v_ids) - 1):
        loc = np.array([radius * np.cos(angle), radius * np.sin(angle)])
        edge_locs.append(loc)
        angle += 2 * np.pi * edge_lengths[i] / total_length

    if v_ids[0] != v_ids[-1]:
        loc = np.array([radius * np.cos(angle), radius * np.sin(angle)])
        edge_locs.append(loc)
   
    return edge_locs


def write_obj(output_path: Path, 
              vertex_locs : np.ndarray,
              uv_verts : np.ndarray, 
              vertex_normals : np.ndarray, 
              triangles : np.ndarray, 
              uv_triangles : np.ndarray, 
              texture_name: str,
              material_name: str):

    b_has_normals = vertex_normals is not None and len(vertex_normals) > 0
    with output_path.open("w", encoding="utf-8", newline="\n") as obj_file:
        obj_file.write(f"mtllib {material_name}\n")
        obj_file.write(f"o {texture_name}\n")
        for v in vertex_locs:
            obj_file.write(f"v {v[0]:.9g} {v[1]:.9g} {v[2]:.9g}\n")
        for uv in uv_verts:
            obj_file.write(f"vt {uv[0]:.9g} {uv[1]:.9g}\n")
        for n in vertex_normals:
            obj_file.write(f"vn {n[0]:.9g} {n[1]:.9g} {n[2]:.9g}\n")

        obj_file.write(f"usemtl {material_name}\n")
        for tri, uv_tri in zip(triangles, uv_triangles):
            obj_file.write("f")
            for vi, uvi in zip(tri, uv_tri):
                if b_has_normals:
                    obj_file.write(f" {vi+1}/{uvi+1}/{vi+1}")
                else:
                    obj_file.write(f" {vi+1}/{uvi+1}")
            obj_file.write("\n")