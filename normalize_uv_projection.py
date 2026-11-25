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
from pathlib import Path

from meshlib import mrmeshnumpy as mn
from meshlib import mrmeshpy as mm

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from filename_fixing import write_material_file
from seam_topology import SeamTopology
from obj_geometry import read_obj, read_texture, read_mesh_no_textures, write_obj
from uv_texture import write_texture_uvs, _CAP_RADIUS, _CAP_CENTERS, _UV_PADDING
from projection_routines import place_in_circle, place_in_square, label_vs_and_fs


_TEXTURE_PADDING_PIXELS = 4


def _check_face_uv(vertex_uv: np.ndarray, 
                   face_vertex_ids: np.ndarray,
                   face_vertex_texture_ids: np.ndarray,
                   topology: SeamTopology,
                   ):
    """ Check the corners and make sure all the faces attached to the corners/seams are not stretched from
        one side to another."""
    top_corner_vid = topology.seams["top"].vertex_ids[0]
    bottom_corner_vid = topology.seams["bottom"].vertex_ids[0]
    side_seam = set(topology.seams["side"].vertex_ids)
    top_seam = set(topology.seams["top"].vertex_ids)
    bottom_seam = set(topology.seams["bottom"].vertex_ids)

    left_side_faces = set(topology.face_lists["left_side_seam"])
    right_side_faces = set(topology.face_lists["right_side_seam"])
    top_faces = set(topology.face_lists["top"])
    bottom_faces = set(topology.face_lists["bottom"])
    middle_faces = set(topology.face_lists["middle"])

    for face in middle_faces:
        count_top = 0
        count_bottom = 0
        count_side = 0
        b_has_top = False
        b_has_bottom = False
        for vid in face_vertex_ids[face]:
            if vid in top_seam:
                count_top += 1
            if vid in bottom_seam:
                count_bottom += 1
            if vid in side_seam:
                count_side += 1
            if vid == top_corner_vid:
                b_has_top = True
            if vid == bottom_corner_vid:
                b_has_bottom = True
        if count_top > 0 and count_bottom > 0:
            print(f"Face {face} is stretched across top and bottom seams")
        if count_top > 0 and count_side > 0:
            print(f"Face {face} is stretched across top and side seams")
        if count_bottom > 0 and count_side > 0:
            print(f"Face {face} is stretched across bottom and side seams")
        if b_has_top and b_has_bottom:
            print(f"Face {face} contains both top and bottom corners")

        uv_center = np.mean(vertex_uv[face_vertex_texture_ids[face]], axis=0)
        dist_center = np.linalg.norm(vertex_uv[face_vertex_texture_ids[face]] - uv_center, axis=1)
        if np.any(dist_center > 0.1):
            print(f"Face {face} is stretched in UV space")
            print(f" Count top {count_top} bottom {count_bottom} side {count_side}")
            print(f" Has top: {b_has_top} Has bottom: {b_has_bottom}")
            print(f" Is left {face in left_side_faces}")
            print(f" Is right {face in right_side_faces}")
            for vid in face_vertex_texture_ids[face]:
                print(f"  Vertex {vid} UV {vertex_uv[vid]}")


def _warp_texture_face(
    face: np.ndarray,
    target_pixels: np.ndarray,
    source_pixels: np.ndarray,
    source: np.ndarray,
    output: np.ndarray,
    covered: np.ndarray,
    bad_distance: float
) -> None:
    height, width = source.shape[:2]
    destination = target_pixels[face]
    triangle_middle = np.mean(destination, axis=0)
    avg_dist = np.mean(np.linalg.norm(destination - triangle_middle, axis=1))
    if avg_dist > bad_distance:
        print(f"Triangle stretched {avg_dist}, skipping")
        return False
    origin = source_pixels[face]
    a = destination[1] - destination[0]
    b = destination[2] - destination[0]
    determinant = a[0] * b[1] - a[1] * b[0]
    if abs(determinant) < 1e-12:
        print(f"Degenerate triangle with determinant {determinant}, skipping")
        return False
    minimum = np.maximum(np.floor(destination.min(axis=0)).astype(int), 0)
    maximum = np.minimum(np.ceil(destination.max(axis=0)).astype(int), [width - 1, height - 1])
    if np.any(minimum > maximum):
        print(f"Flipped {np.cross(destination[1] - destination[0], destination[2] - destination[0])}")
        return False
    x, y = np.meshgrid(
        np.arange(minimum[0], maximum[0] + 1),
        np.arange(minimum[1], maximum[1] + 1),
    )
    dx, dy = x - destination[0, 0], y - destination[0, 1]
    s = (dx * b[1] - dy * b[0]) / determinant
    t = (a[0] * dy - a[1] * dx) / determinant
    mask = (s >= -1e-8) & (t >= -1e-8) & (s + t <= 1 + 1e-8)
    if not np.any(mask):
        return True
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
    return True


def _warp_texture(
    texture_path: Path,
    output_texture: Path,
    original_uvs: np.ndarray,
    normalized_uvs: np.ndarray,
    faces: np.ndarray,
) -> None:
    with Image.open(texture_path) as input_image:
        source = np.asarray(input_image.convert("RGB"))

    f_failed = []
    height, width = source.shape[:2]
    output = np.zeros_like(source)
    covered = np.zeros((height, width), dtype=bool)
    scale = np.array([width - 1, height - 1])
    source_pixels = original_uvs * scale
    target_pixels = normalized_uvs * scale
    source_pixels[:, 1] = (height - 1) - source_pixels[:, 1]
    target_pixels[:, 1] = (height - 1) - target_pixels[:, 1]
    bad_dist = height * 0.2
    for fid, face in enumerate(faces):
        success = _warp_texture_face(face, target_pixels, source_pixels, source, output, covered, bad_dist)
        if not success:
            f_failed.append(fid)
    if not np.any(covered):
        raise ValueError("No texture pixels were covered by the normalized UV triangles.")
    distances, nearest = distance_transform_edt(~covered, return_indices=True)
    padding = (~covered) & (distances <= _TEXTURE_PADDING_PIXELS)
    output[padding] = output[nearest[0][padding], nearest[1][padding]]
    if f_failed:
        print(f"Failed to warp {len(f_failed)} faces: {f_failed}")
    Image.fromarray(output).save(output_texture)
    return f_failed


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
    bottom_vs, bottom_fs, top_vs, top_fs, middle_vs, middle_fs, bottom_vid, edge_adj = labels

    new_top_locs, map_top, _ = place_in_circle(mesh=mesh, edge_adjacency=edge_adj, boundary_vs=top_seam_vs, inside_vs=top_vs, center=seam_topology.top_vertex, radius=_CAP_RADIUS - 0.01)
    new_bottom_locs, map_bottom, _ = place_in_circle(mesh=mesh, edge_adjacency=edge_adj, boundary_vs=bottom_seam_vs, inside_vs=bottom_vs, center=bottom_vid, radius=_CAP_RADIUS - 0.01)
    new_bottom_locs[:, :] += _CAP_CENTERS[1]
    new_top_locs[:, :] += _CAP_CENTERS[0]

    diamond_corners = np.array(
        [[0.5, _UV_PADDING], [1.0 - _UV_PADDING, 0.5], [0.5, 1.0 - _UV_PADDING], [_UV_PADDING, 0.5]],
        dtype=np.float64,
            )

    new_diamond_locs, map_diamond, _, _, _ = place_in_square(mesh=mesh, 
                                                            edge_adjacency=edge_adj, 
                                                            face_vertex_indices=face_vert_ids,
                                                            seam_topology=seam_topology, 
                                                            inside_vs=middle_vs,
                                                            corners=diamond_corners)   

    # Now put the new uv locations back into new_vertex_uv
    new_vertex_uv = np.copy(vertex_uv)

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
    corner_top = seam_topology.seams["side"].vertex_ids[0]
    corner_bottom = seam_topology.seams["side"].vertex_ids[-1]
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
                #print(f"loc 1 for vertex {vid} is {new_diamond_locs[map_v[0]]}")
                #print(f"loc 2 for vertex {vid} is {new_diamond_locs[map_v[1]]}")
                if vid == corner_top:
                    if face in left_faces:
                        new_vertex_uv[vt_id] = new_diamond_locs[map_v[1]]
                        v_set.add(vt_id)
                    elif face in right_faces:
                        new_vertex_uv[vt_id] = new_diamond_locs[map_v[0]]
                        v_set.add(vt_id)
                    else:
                        raise ValueError(f"Corner vertex {vid} is not on either side of the seam.")
                elif face in left_faces:
                    new_vertex_uv[vt_id] = new_diamond_locs[map_v[0]]
                    v_set.add(vt_id)
                elif face in right_faces:
                    new_vertex_uv[vt_id] = new_diamond_locs[map_v[1]]
                    v_set.add(vt_id)
                else:
                    raise ValueError(f"Vertex {vid} with multiple mappings is not on either side of the seam.")
            else:
                raise ValueError(f"Vertex {vid} has unexpected number of mappings in map_diamond: {len(map_v)}")    

    if len(v_set) != len(new_vertex_uv):
        print ("Warning: Some UVs were not assigned new locations.")
        # raise ValueError("Some UVs were not assigned new locations.")
    return new_vertex_uv


def normalize_uv_projection(input_path: Path, output_path: Path, seam_path: Path) -> Path:
    """Normalize one textured OBJ using matching seam topology in seam_path."""
    input_path, seam_path, output_path = Path(input_path), Path(seam_path), Path(output_path)
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
    vertex_loc, vertex_uv, vert_uv_ids, vertex_normals, face_triangles, face_uv_triangles, face_norm_triangles, mtllibs = read_obj(input_path)
    material_path = input_path.with_suffix(".mtl")
    texture_path = read_texture(material_path)

    # Label vertices and faces into calyx, top, and middle regions based on the original UVs and faces
    # Note that vertices will show up in multiple regions if they are shared across faces
    labels = label_vs_and_fs(mesh=mesh, 
                             seam_topology=topology)

    # The new locations for the uv coordinates, with boundaries mapped to the two circles and the diamonds
    new_uv_locs = place_uvs_in_new_locs(mesh=mesh,
                                        seam_topology=topology,
                                        vertex_uv=vertex_uv,
                                        vert_uv_ids=vert_uv_ids,
                                        face_vert_ids=face_triangles,
                                        face_uv_ids=face_uv_triangles,
                                        labels=labels)

    _check_face_uv(vertex_uv=vertex_uv, 
                   face_vertex_ids=face_triangles, 
                   face_vertex_texture_ids=face_uv_triangles, 
                   topology=topology)
    _check_face_uv(vertex_uv=new_uv_locs, face_vertex_ids=face_triangles, face_vertex_texture_ids=face_uv_triangles, topology=topology)

    # Which uv vertices belong to the
    seams = [[], [], []]    
    for v in topology.seams["top"].vertex_ids:
        seams[0].append(vert_uv_ids[v])
    for v in topology.seams["bottom"].vertex_ids:
        seams[1].append(vert_uv_ids[v])
    for v in topology.seams["side"].vertex_ids:
        seams[2].append(vert_uv_ids[v])


    write_texture_uvs(seam_vs=seams, uvs=new_uv_locs, texture_path=output_texture)
    # Now copy the texture over by warping the faces to the new UV locations
    # For each of the original faces in each of the regions, get the new UV coordinates from the mapped locations

    # Do the actual texture copy
    f_failed = _warp_texture(texture_path=texture_path, 
                             output_texture=output_texture, 
                             original_uvs=vertex_uv, 
                             normalized_uvs=new_uv_locs, 
                             faces=face_uv_triangles)
    vt_count = [[] for _ in range(len(new_uv_locs))]
    for face in face_uv_triangles:
        for vt_id in face:
            vt_count[vt_id].append(face)
    if f_failed:
        for face in f_failed:
            if face in topology.face_lists["top"]:
                print(f"Face {face} is in top face list")
            if face in topology.face_lists["bottom"]:
                print(f"Face {face} is in bottom face list")
            if face in topology.face_lists["middle"]:
                print(f"Face {face} is in middle face list")
            if face in topology.face_lists["left_side_seam"]:
                print(f"Face {face} is in left side seam face list")
            if face in topology.face_lists["right_side_seam"]:
                print(f"Face {face} is in right side seam face list")
            for vid, v_t_id in zip(face_triangles[face], face_uv_triangles[face]):
                uv_loc = vertex_uv[v_t_id]
                print(f" Failed vertex {vid} with UV {v_t_id} {float(uv_loc[0]), float(uv_loc[1])}")
                if vid in topology.seams["top"].vertex_ids:
                    print(f" Vertex {vid} is in top seam")
                if vid in topology.seams["bottom"].vertex_ids:
                    print(f" Vertex {vid} is in bottom seam")
                if vid in topology.seams["side"].vertex_ids:
                    print(f" Vertex {vid} is in side seam")
        print(f"Failed to warp {len(f_failed)} faces: {f_failed}")

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
              face_norm_triangles=face_norm_triangles,
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