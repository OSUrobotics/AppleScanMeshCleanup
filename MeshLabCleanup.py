""" Clean up meshes
  Clean up 1 (process_mesh): run a MeshLab filter script on the input mesh and save the result.
     - this takes out non-manifold edges, disconnected components, fills holes, and does a downsample
     - this uses the pymeshlab librar
  Clean up 2 (voxelize): Use voxelization to remove small tunnels from the mesh. Also removes disconnected components and faces with zero area
     - After voxelization the mesh is parameterized for UV texture mapping
        - the middle of the mesh is unrolled into a cylindrical texture map placed diagonally
        - the top and bottom of the mesh are mapped to circular regions (top left and bottom right)
    - this operation uses meshlib (which is not smart enough to keep texture map coords, etc)
    - After the uv parameterization pymeshlab is used to write out the uv coordinates
  Clean up 3 (downsample to final): Uses pymesnlab and a meshlab script to reduce the mesh to the target counts. 
    - After downsample, the clicked end points are used to orient the mesh so that the stem is at 0, 0. This reorientation 
      is done with pymeshlab.

"""
import os
from pathlib import Path
import shutil
import tempfile
from typing import Callable

import numpy as np
import pymeshlab as pml
from meshlib import mrmeshpy as mm

from MeshLibCleanup import print_mesh_topology

from uv_texture import uv_texture, _find_centers

MeshProcessor = Callable[[Path, Path], Path]


def _write_material_file(material_path: Path, texture_filename: str):
    if "Envy" in os.path.basename(material_path):
        Ka = (0.15, 0.03, 0.03)
        Kd = (0.65, 0.12, 0.10)
        Ks = (0.18, 0.18, 0.18)
    elif "Gala" in os.path.basename(material_path):
        Ka = (0.18, 0.06, 0.03)
        Kd = (0.78, 0.28, 0.12)
        Ks = (0.15, 0.15, 0.15)
    elif "PinkLady" in os.path.basename(material_path):
        Ka = (0.19, 0.07, 0.07)
        Kd = (0.82, 0.30, 0.32)
        Ks = (0.15, 0.18, 0.15)
    elif "CosmicCrisp" in os.path.basename(material_path):
        Ka = (0.10, 0.02, 0.02)
        Kd = (0.42, 0.07, 0.09)
        Ks = (0.15, 0.15, 0.15)
    elif "HoneyCrisp" in os.path.basename(material_path):
        Ka = (0.17, 0.05, 0.03)
        Kd = (0.74, 0.22, 0.14)
        Ks = (0.15, 0.15, 0.15)
    else:
        Ka = (0.15, 0.03, 0.03)
        Kd = (0.65, 0.12, 0.10)
        Ks = (0.18, 0.18, 0.18)

    material_path.write_text(
        "# Blender 5.0.1 MTL File: 'process_apples.blend'\n"
        "# www.blender.org\n"
        "\n"
        "newmtl material_0\n"
        f"Ka {Ka[0]} {Ka[1]} {Ka[2]}\n"
        f"Kd {Kd[0]} {Kd[1]} {Kd[2]}\n"
        f"Ks {Ks[0]} {Ks[1]} {Ks[2]}\n"
        "Ns 30.000000\n"
        "Ni 1.4500000\n"
        "d 1.000000\n"
        "illum 2\n"
        f"map_Kd {texture_filename}\n",
        encoding="utf-8",
    )


def _rename_mesh(input_path: Path, output_path: Path, use_original_material_file: bool) -> None:
    """ Copy the obj file over and update material references if needed."""
    new_lines = []
    mtl_name = ""
    tex_name = ""
    with open(input_path, "r") as fIn:
        data = fIn.read()
        for s in data.splitlines():
            if ".mtl" in s:
                mtl_pieces = s.split(" ")
                mtl_name = mtl_pieces[-1]
                s = "mtllib " + output_path.with_suffix(".mtl").name
            elif "usemtl" in s and not use_original_material_file:
                s = "usemtl material_0"
            elif len(s) > 0 and s[0] == "o":
                mtl_pieces = s.split(" ")
                tex_name = mtl_pieces[-1]
                s = "o " + output_path.with_suffix("").name
            new_lines.append(s)

    with open(output_path, "w") as fOut:
        fOut.write("\n".join(new_lines))

    # Copy the texture file if it exists
    if tex_name == "":
        tex_name = input_path.name.rsplit(".", 1)[0]
    for ext in [".png", ".jpg", ".jpeg"]:
        if os.path.exists(input_path.with_name(tex_name + ext)):
            tex_name += ext
            break   
    tex_path = input_path.with_name(tex_name)
    if os.path.exists(tex_path):
        shutil.copy(tex_path, output_path.with_suffix(tex_path.suffix))

    if use_original_material_file and os.path.exists(input_path.basename + mtl_name):
        if os.path.exists(input_path.basename + mtl_name):
            new_lines = []
            with open(input_path.basename + mtl_name, "r") as fIn:
                data = fIn.read()
                for s in data.splitlines():
                    if "map_Kd" in s:
                        tex_pieces = s.split(" ")
                        s = f"map_Kd {tex_pieces[-1]}"
                    new_lines.append(s)

            with open(output_path.with_suffix(".mtl"), "w") as fOut:
                fOut.write("\n".join(new_lines))
    else:
        output_tex_name = output_path.name.replace(output_path.suffix, tex_path.suffix)
        _write_material_file(output_path.with_suffix(".mtl"), output_tex_name)

    print(f"Renaming mesh {input_path} to {output_path}, texture: {tex_name}, material: {mtl_name}")


def _process_mesh(input_path: Path, output_path: Path, mlx_script_path: Path) -> None:
    ms = pml.MeshSet()
    ms.load_new_mesh(str(input_path))

    print(f"Trying mesh {input_path}")
    ms.load_filter_script(str(mlx_script_path))
    ms.apply_filter_script()

    textures = ms.current_mesh().textures()
    if len(textures) > 1:
        raise ValueError(f"Expected at most one texture in {input_path}, found {len(textures)}.")

    with tempfile.TemporaryDirectory() as temporary:
        if textures:
            texture_name, texture = next(iter(textures.items()))
            renamed_texture = Path(temporary) / output_path.with_suffix(Path(texture_name).suffix).name
            texture.save(str(renamed_texture))
            ms.set_texture_per_mesh(textname=str(renamed_texture))

        ms.save_current_mesh(
            str(output_path),
            save_textures=True,
            save_vertex_color=True,
            save_vertex_normal=True,
            save_face_color=True,
            save_wedge_texcoord=True,
            save_wedge_normal=True,
        )

    print(f"Processed and saved: {output_path.name}")


def cleanup_mesh(
    input_path: Path,
    output_path: Path,
    mlx_script_path: Path,
) -> Path:
    print("Original Revo scan mesh topology")
    print_mesh_topology(input_path)

    _process_mesh(input_path, output_path, mlx_script_path)
    print("After MeshLab cleanup mesh topology")
    print_mesh_topology(output_path)
    return output_path


def _reorient_mesh(centers_path: Path, output_path: Path) -> Path:
    ms = pml.MeshSet()
    ms.load_new_mesh(str(output_path))
    mesh = ms.current_mesh()
    vertices = mesh.vertex_matrix()
    centers = _find_centers(input_path=centers_path, vertices=vertices)
    top, bottom = (np.asarray(point, dtype=np.float64) for point in centers)

    axis = top - bottom
    axis_length = np.linalg.norm(axis)
    if not np.all(np.isfinite(axis)) or axis_length == 0:
        raise ValueError("Mesh endpoints must define a finite, non-zero axis.")

    direction = axis / axis_length
    target = np.array([0.0, 1.0, 0.0])
    cross = np.cross(direction, target)
    cosine = float(np.dot(direction, target))
    cross_length_squared = float(np.dot(cross, cross))
    if cross_length_squared < 1e-15:
        rotation = np.eye(3) if cosine > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        skew = np.array(
            [
                [0.0, -cross[2], cross[1]],
                [cross[2], 0.0, -cross[0]],
                [-cross[1], cross[0], 0.0],
            ]
        )
        rotation = np.eye(3) + skew + (skew @ skew) / (1.0 + cosine)

    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = -(rotation @ top)
    ms.apply_filter("set_matrix", transformmatrix=transform, freeze=True)
    ms.save_current_mesh(
        str(output_path),
        save_textures=True,
        save_vertex_color=True,
        save_vertex_normal=True,
        save_face_color=True,
        save_wedge_texcoord=True,
        save_wedge_normal=True,
    )
    return output_path


def downsample_mesh(
    input_path: Path, output_path: Path, mlx_script_path: Path
) -> Path:
    print("Original textured mesh topology")
    print_mesh_topology(input_path)
    
    if "lowres" in str(mlx_script_path):
        output_path = output_path.with_name(output_path.name.replace("HiRes", "LowRes"))
    _process_mesh(input_path, output_path, mlx_script_path)
    path_reorient = input_path.parent.with_name(
        input_path.parent.name.replace("BlenderRebaked", "BlenderStemRemoved")
    ) / input_path.name
    _reorient_mesh(centers_path=path_reorient, output_path=output_path)
    print("Downsampled textured mesh topology")
    print_mesh_topology(output_path)
    return output_path


def _remove_unconnected(mesh: mm.Mesh) -> None:
    components = mm.getAllComponents(mm.MeshPart(mesh))
    if len(components) <= 1:
        return

    largest_component_index = max(
        range(len(components)), key=lambda index: components[index].count()
    )
    removed_face_count = 0
    for index, component in enumerate(components):
        if index == largest_component_index:
            continue
        removed_face_count += component.count()
        mesh.deleteFaces(component)

    print(
        f"Removed {len(components) - 1} disconnected components "
        f"({removed_face_count} faces)."
    )


def _remove_zero_area_face(mesh: mm.Mesh) -> None:
    zero_area_faces = mm.FaceBitSet()
    zero_area_faces.resize(mesh.topology.faceSize())
    for face_id in mesh.topology.getValidFaces():
        if mesh.area(face_id) == 0.0:
            zero_area_faces.set(face_id, True)

    removed_face_count = zero_area_faces.count()
    if removed_face_count == 0:
        return

    mesh.deleteFaces(zero_area_faces)
    existing_face_ids = {
        face_id.get() for face_id in mesh.topology.getValidFaces()
    }
    hole_edges = mesh.topology.findHoleRepresentiveEdges()
    if len(hole_edges) > 0:
        mm.fillHoles(mesh, hole_edges)

        patch_faces = mm.FaceBitSet()
        patch_faces.resize(mesh.topology.faceSize())
        for face_id in mesh.topology.getValidFaces():
            if face_id.get() not in existing_face_ids:
                patch_faces.set(face_id, True)

        if patch_faces.count() > 0:
            params = mm.FixMeshDegeneraciesParams()
            params.region = patch_faces
            mm.fixMeshDegeneracies(mesh, params)

    remaining_zero_area_faces = [
        face_id
        for face_id in mesh.topology.getValidFaces()
        if mesh.area(face_id) == 0.0
    ]
    if remaining_zero_area_faces:
        raise ValueError(
            "Zero-area faces remain after hole repair: "
            f"{[face_id.get() for face_id in remaining_zero_area_faces]}"
        )

    print(
        f"Removed {removed_face_count} zero-area faces and "
        f"patched {len(hole_edges)} resulting holes."
    )


def voxelize_mesh(input_path: Path, output_path: Path) -> Path:
    print("Original mesh topology")
    print_mesh_topology(input_path)

    # 1. Load the original mesh
    mesh = mm.loadMesh(str(input_path))

    # 2. Set up voxel-based repair parameters
    params = mm.GeneralOffsetParameters()

    # Define voxel size. Smaller = preserves more detail; Larger = closes larger tunnels.
    # The mesh is in millimeters. About 1-2 mm is enough to remove tunnels
    params.voxelSize = 0.5  # mm
    params.signDetectionMode = mm.SignDetectionMode.HoleWindingRule
    params.closeHolesInHoleWindingNumber = True

    # 3. Rebuild the mesh (this melts away tiny tunnels and bridges)
    fixed_mesh = mm.generalOffsetMesh(mm.MeshPart(mesh), 0.0, params)

    _remove_unconnected(fixed_mesh)
    _remove_zero_area_face(fixed_mesh)

    # 5. Save the cleaned manifold mesh
    mm.saveMesh(fixed_mesh, str(output_path))
    print("voxel fixed topology")
    print_mesh_topology(output_path)

    path_uv = output_path.with_name(input_path.name.replace("NoStemHiRes", ""))

    uv_texture(input_path=output_path, output_path=path_uv)


def _split_filename_into_apple_type_and_number(filename: str) -> tuple[str, int]:
    # Get the apple name part of the filename
    apple_name = ""
    for c in filename:
        if c.isdigit() or c == "_":
            break
        apple_name += c
    
    # Get the number out of the filename

    import re
    match = re.search(r"(\d+)", filename)
    if not match:
        raise ValueError(f"Filename does not match expected pattern: {filename}")
    number = int(match.group(1))
    return apple_name, number


def process_meshes(
    input_dir,
    output_dir,
    process_file: MeshProcessor,
    new_tag_output_filename=None,
    prompt_overwrite_older=False,
    check_existing_outputs=True,
    input_extensions=(".obj", ".ply", ".stl"),
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    print(f"Input files: {input_dir}")
    print(f"Output files: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    processed_files = []
    filenames = sorted(os.listdir(input_dir))

    # Did we do this one already? (the output time stamp is greater than the input)
    already_done = set()
    no_outputs = set()
    if check_existing_outputs:
        for filename in filenames:
            if not filename.lower().endswith(input_extensions):
                continue

            input_path = input_dir / filename
            output_path = output_dir / filename
            if new_tag_output_filename:
                # first strip out everything past the apple name and number by splitting file name at the end of the number
                apple_name, number = _split_filename_into_apple_type_and_number(filename)
                output_filename = f"{apple_name}_{number:03d}_{new_tag_output_filename}"

                output_path = output_path.with_name(output_filename + output_path.suffix)   
            if input_path.is_file() and output_path.is_file():
                if output_path.stat().st_mtime_ns > input_path.stat().st_mtime_ns:
                    already_done.add(output_path)
            elif output_path.is_file():
                no_outputs.add(output_path)

        if already_done:
            print("Output files that are already done:")
            for output_path in sorted(already_done):
                print(f"  {str(output_path.name)}")

    for filename in filenames:
        if not filename.lower().endswith(input_extensions):
            continue

        input_path = input_dir / filename
        output_path = output_dir / filename

        if new_tag_output_filename:
            # first strip out everything past the apple name and number by splitting file name at the end of the number
            apple_name, number = _split_filename_into_apple_type_and_number(filename)
            output_filename = f"{apple_name}_{number:03d}_{new_tag_output_filename}"

            output_path = output_path.with_name(output_filename + output_path.suffix)   

        if check_existing_outputs and output_path.exists():
            if output_path in already_done:
                if prompt_overwrite_older:
                    answer = input(f"Overwrite {output_path.name}? [y/n] ").strip().lower()
                    if answer not in ("y", "yes"):
                        continue
                else:
                    continue
        try:
            processed_path = process_file(input_path, output_path)
            processed_files.append(str(processed_path))
        except Exception as exc:
            print(f"Failed to process {output_path.name}: {exc}")

    return processed_files
