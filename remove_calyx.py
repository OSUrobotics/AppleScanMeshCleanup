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
from uv_texture import _CAP_CENTERS, _CAP_RADIUS


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
    kill_face_set = set(kill_faces)
    # Keep shared boundary vertices that are still referenced by retained faces.
    used_vs = {
        int(vertex)
        for f_idx, face in enumerate(face_triangles)
        if f_idx not in kill_face_set
        for vertex in face
    }

    map_vs = {}  # Keep track of new vertex ids for renumbering
    map_uvs_vs = {}  # Keep track of new vertex ids for renumbering
    n_vertices = vertex_loc.shape[0]
    count = 0
    count_vt = 0
    new_vertex_loc = []
    new_vertex_uv = []
    for indx in range(0, n_vertices):
        if indx in used_vs:
            map_vs[indx] = count
            new_vertex_loc.append(vertex_loc[indx, :])
            count += 1
            for uv_idx in vertex_uv_ids.get(indx, []):
                map_uvs_vs[uv_idx] = count_vt
                new_vertex_uv.append(vertex_uv[uv_idx, :])
                count_vt += 1
        else:
            map_vs[indx] = -1

    topology.renumber_seams(map_vs)
    map_fs = {} 
    count_f = 0  # Keep track of new face ids for renumbering
    new_face_triangles = []
    new_face_uv_triangles = []
    new_face_normals = []
    for f_idx in range(0, face_triangles.shape[0]):
        if f_idx in kill_face_set:
            map_fs[f_idx] = -1
        else:
            map_fs[f_idx] = count_f
            new_face_triangles.append([map_vs[vid] for vid in face_triangles[f_idx]])
            new_face_uv_triangles.append([map_uvs_vs[uv] for uv in face_uv_triangles[f_idx]])
            new_face_normals.append(face_norm_triangles[f_idx])
            count_f += 1

    topology.renumber_faces(map_fs)
    return (new_vertex_loc, new_vertex_uv, list(vertex_normals), new_face_triangles,
            new_face_uv_triangles, new_face_normals)


def _calyx_faces(vertices, faces, seam, bottom_point):
    """Return the calyx ring and faces connected to the bottom without crossing it."""
    ring = np.asarray(seam.vertex_ids, dtype=int)
    if len(ring) < 3 or len(np.unique(ring)) != len(ring):
        raise ValueError("The calyx seam must contain at least three unique vertices.")
    if ring.min() < 0 or ring.max() >= len(vertices):
        raise ValueError("The calyx seam contains an out-of-range vertex.")
    if len(seam.vertex_locations) != len(ring) or not np.allclose(
        vertices[ring], seam.vertex_locations, rtol=0, atol=1e-4
    ):
        raise ValueError("Calyx seam vertex locations do not match the mesh.")
    center = np.asarray(seam.center, dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("The calyx seam must have a finite 3D center.")
    offsets = vertices[ring][:, [0, 2]] - center[[0, 2]]
    ring = ring[np.argsort(np.arctan2(offsets[:, 1], offsets[:, 0]), kind="stable")]

    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges = edges[~np.isin(edges, ring).any(axis=1)]
    edges = np.concatenate((edges, edges[:, ::-1]))
    adjacency = coo_matrix(
        (np.ones(len(edges), dtype=np.uint8), (edges[:, 0], edges[:, 1])),
        shape=(len(vertices), len(vertices)),
    ).tocsr()
    start = int(np.argmin(np.linalg.norm(vertices - np.asarray(bottom_point), axis=1)))
    if start in set(map(int, ring)):
        raise ValueError("The bottom point must be inside the calyx ring.")
    _, labels = connected_components(adjacency, directed=False)
    visited = labels == labels[start]
    inside = visited[faces].any(axis=1)
    return ring, inside, np.flatnonzero(visited)


def cutout_calyx_and_save(input_path: Path, output_path: Path):
    """Find the component inside the ring without relying on seam vertex order."""
    input_path, output_path = Path(input_path), Path(output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input mesh not found: {input_path}")
    if input_path.suffix.lower() != ".obj" or output_path.suffix.lower() != ".obj":
        raise ValueError("Input and output meshes must be OBJ files.")
    seam_file = input_path.parent / (input_path.stem + "_seam_topology.json")
    calyx_sub_path = output_path.parent / "Calyx"
    calyx_sub_path.mkdir(parents=True, exist_ok=True)
    calyx_path = calyx_sub_path / (output_path.stem + "_calyx.obj")

    vertex_loc, vertex_uv, vert_uv_ids, vertex_normals, face_triangles, face_uv_triangles, face_norm_triangles, mtllibs = read_obj(input_path)
    with seam_file.open(encoding="utf-8") as stream:
        topology = SeamTopology.from_dict(json.load(stream))

    if "calyx" not in topology.seams or "stem" not in topology.seams:
        raise ValueError("Seam topology must contain a calyx ring and a stem ring.")

    ring, inside, _ = _calyx_faces(
        vertex_loc, face_triangles, topology.seams["calyx"], topology.bottom_point
    )
    remove_non_calyx_faces = np.flatnonzero(~inside).tolist()
    new_vertex_loc, new_vertex_uv, new_vertex_norm, new_face_triangles, new_face_uv_triangles, new_face_normals = _remove_and_renumber(
        vertex_loc, vertex_uv, vertex_normals, vert_uv_ids, face_triangles,
        face_uv_triangles, face_norm_triangles, topology, remove_non_calyx_faces
    )

    input_texture = read_texture(input_path.parent / mtllibs)
    calyx_texture = calyx_sub_path / input_texture.name
    if input_texture.resolve() != calyx_texture.resolve():
        # Cut out the top quarter of the image and save it out as the calyx texture
        # Take the square at 2 * width * _CAP_RADIUS from the top left corner
        # and rescale it to to 256x256
        input_image = Image.open(input_texture)
        width, height = input_image.size
        ll = int(2 * width * _CAP_RADIUS)

        top_left_square = input_image.crop((0, 0, ll, ll))
        top_left_square = top_left_square.resize((512, 512))
        top_left_square.save(calyx_texture)

    center = np.asarray(topology.seams["calyx"].center, dtype=float)
    new_vertex_loc = np.asarray(new_vertex_loc, dtype=float)
    new_vertex_loc[:, [0, 2]] -= center[[0, 2]]
    calyx_material = calyx_path.with_suffix(".mtl")
    write_material_file(calyx_material, calyx_texture.name)

    scl = 0.5 / _CAP_RADIUS
    for uv in new_vertex_uv:
        uv[0] = (uv[0] - _CAP_CENTERS[1][0]) * scl + 0.5
        uv[1] = (uv[1] - _CAP_CENTERS[1][1]) * scl + 0.5

    write_obj(
        calyx_path, new_vertex_loc, new_vertex_uv, new_vertex_norm, new_face_triangles,
        new_face_uv_triangles, new_face_normals, calyx_path.stem, calyx_material.name
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, default="/Users/cindygrimm/PycharmProjects/data/ScannedModels/ProsserEnvyApples/", help="Directory the paths are relative to")
    parser.add_argument("input", nargs="?", default="PaddedTexture/Envy_002_FullRes.obj", type=Path, help="Rebaked textured OBJ")
    parser.add_argument("output", nargs="?", default="CalyxRemoved/Envy_002_FullRes.obj", type=Path, help="Output OBJ path")

    args = parser.parse_args()
    input_path = args.root_dir / args.input if args.root_dir is not None else args.input
    output_path = args.root_dir / args.output if args.root_dir is not None else args.output

    print(f"Wrote {cutout_calyx_and_save(input_path, output_path)}")


if __name__ == "__main__":
    main()
