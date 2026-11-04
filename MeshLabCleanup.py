import os
from pathlib import Path
from typing import Callable

import pymeshlab as pml
from meshlib import mrmeshpy as mm

from MeshLibCleanup import (
    preserve_obj_material_assets,
    print_mesh_topology,
    remove_small_tunnels,
)


MeshProcessor = Callable[[Path, Path], Path]


def _process_mesh(input_path: Path, output_path: Path, mlx_script_path: Path) -> None:
    original_obj = (
        input_path.read_text(encoding="utf-8")
        if input_path.suffix.lower() == ".obj"
        else None
    )
    source_mesh = (
        mm.loadMesh(str(input_path))
        if original_obj is not None
        and any(line.lstrip().startswith("vt ") for line in original_obj.splitlines())
        else None
    )
    ms = pml.MeshSet()
    ms.load_new_mesh(str(input_path))

    print(f"Trying mesh {input_path}")
    ms.load_filter_script(str(mlx_script_path))
    ms.apply_filter_script()
    ms.save_current_mesh(str(output_path))
    if original_obj is not None:
        preserve_obj_material_assets(
            input_path,
            output_path,
            original_obj,
            source_mesh,
        )
    print(f"Processed and saved: {output_path.name}")


def cleanup_mesh(
    input_path: Path,
    output_path: Path,
    mlx_script_path: Path,
    post_tunnel_mlx_script_path: Path,
) -> Path:
    print("Original Revo scan mesh topology")
    print_mesh_topology(input_path)

    _process_mesh(input_path, output_path, mlx_script_path)
    print("After MeshLab cleanup mesh topology")
    print_mesh_topology(output_path)

    remove_small_tunnels(output_path, output_path)
    print("After removing small tunnels mesh topology")
    print_mesh_topology(output_path)

    _process_mesh(output_path, output_path, post_tunnel_mlx_script_path)
    print("After post-tunnel MeshLab processing mesh topology")
    print_mesh_topology(output_path)
    return output_path


def downsample_mesh(
    input_path: Path, output_path: Path, mlx_script_path: Path
) -> Path:
    print("Original textured mesh topology")
    print_mesh_topology(input_path)
    output_path = output_path.with_name(output_path.name.replace("HiRes", "LowRes"))
    _process_mesh(input_path, output_path, mlx_script_path)
    print("Downsampled textured mesh topology")
    print_mesh_topology(output_path)
    return output_path


def fix_tunnels_mesh(
    input_path: Path, output_path: Path, post_tunnel_mlx_script_path: Path
) -> Path:
    print("Original mesh topology")
    print_mesh_topology(input_path)
    remove_small_tunnels(input_path, output_path)
    _process_mesh(output_path, output_path, post_tunnel_mlx_script_path)
    print("Tunnels fixed topology")
    print_mesh_topology(output_path)
    print("\n")
    return output_path


def voxelize_mesh(
    input_path: Path, output_path: Path, post_tunnel_mlx_script_path: Path
) -> Path:
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

    # 5. Save the cleaned manifold mesh
    mm.saveMesh(fixed_mesh, str(output_path))
    print("voxel fixed topology")
    print_mesh_topology(output_path)

    _process_mesh(output_path, output_path, post_tunnel_mlx_script_path)
    print("voxel fixed topology")
    print_mesh_topology(output_path)
    print("\n")


def process_meshes(
    input_dir,
    output_dir,
    process_file: MeshProcessor,
    prompt_overwrite_older=False,
    check_existing_outputs=True,
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    print(f"Input files: {input_dir}")
    print(f"Output files: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    processed_files = []
    filenames = sorted(os.listdir(input_dir))

    older_outputs = set()
    if prompt_overwrite_older and check_existing_outputs:
        for filename in filenames:
            if not filename.lower().endswith((".obj", ".ply", ".stl")):
                continue

            input_path = input_dir / filename
            output_path = output_dir / filename
            if (
                input_path.is_file()
                and output_path.is_file()
                and output_path.stat().st_mtime_ns < input_path.stat().st_mtime_ns
            ):
                older_outputs.add(output_path)

        if older_outputs:
            print("Output files older than their matching input files:")
            for output_path in sorted(older_outputs):
                print(f"  {output_path.name}")

    for filename in filenames:
        if not filename.lower().endswith((".obj", ".ply", ".stl")):
            continue

        input_path = input_dir / filename
        output_path = output_dir / filename

        if check_existing_outputs and output_path.exists():
            if output_path not in older_outputs:
                continue
            answer = input(f"Overwrite {output_path.name}? [y/n] ").strip().lower()
            if answer not in ("y", "yes"):
                continue

        try:
            processed_path = process_file(input_path, output_path)
            processed_files.append(str(processed_path))
        except Exception as exc:
            print(f"Failed to process {output_path.name}: {exc}")

    return processed_files
