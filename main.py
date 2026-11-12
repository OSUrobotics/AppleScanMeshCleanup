import argparse
from functools import partial
from pathlib import Path

""" Going from oroginal Revo scans to consistantly parameterized, topologically spherical meshes with a cylindrical parameterization
Directories
   OriginalRevoScans : The original output meshes, labeled Apple#FullRes.obj
   MeshLabCleaned : Derived from the Revo scans by automatic filtering and downsampling
           - it is safe to overwrite these 
   BlenderStemRemoved : Manually cut out the stem 
   Voxelized : Generate a new, clean mesh using voxelization and build the cylindrical parameterization. Runs on BlenderStemRemoved
           - it is safe to overwrite these
   BlenderTextured : Manually Bake the texture map from the meshlabcleaned and clean up the stem texture
   NormalizedTexture : Normalize UV charts and warp the rebaked texture using seams from Voxelized
   Final : Generate Hi and Low res meshees from BlenderTextured.
   """
from MeshLabCleanup import (
    cleanup_mesh,
    downsample_mesh,
    process_meshes,
    voxelize_mesh,
    _rename_mesh
)

from uv_texture import uv_texture
from normalize_uv_projection import normalize_uv_projection


SCRIPT_DIR = Path(__file__).resolve().parent
MLX_SCRIPTS = {
    "cleanup": SCRIPT_DIR / "MeshLabScripts" / "MeshLabCleanup.mlx",
    "downsample_lowres": SCRIPT_DIR / "MeshLabScripts" / "downsample_lowres.mlx",
    "downsample_hires": SCRIPT_DIR / "MeshLabScripts" / "downsample_hires.mlx",
    "voxelize": SCRIPT_DIR / "MeshLabScripts" / "post_tunnel.mlx",
    "rename": SCRIPT_DIR / "MeshLabScripts" / "MeshLabCleanup.mlx",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clean up, downsample, voxelize, or normalize UV textures."
    )
    parser.add_argument(
        "--operation",
        required=True,
        choices=(*MLX_SCRIPTS, "normalize"),
        help="Operation to run; selects the input/output directories.",
    )
    parser.add_argument(
        "--root-dir",
        required=True,
        help="Top-level folder containing OriginalRevoScans, MeshLabCleaned, MeshLabDownsampled, and BlenderTextures.",
    )
    parser.add_argument(
        "--source-dir",
        required=False,
        help="One of the subdirectories: OriginalRevoScans, MeshLabCleaned, MeshLabDownsampled, and BlenderTextures.",
    )
    parser.add_argument(
        "--dest-dir",
        required=False,
        help="One of the subdirectories: MeshLabCleaned, MeshLabDownsampled, BlenderStemRemoved, Voxelized, BlenderRebaked, NormalizedTexture, and Final.",
    )
    parser.add_argument(
        "--dest-name",
        required=False,
        help="What to append after Apple# for the file name.",
    )
    parser.add_argument(
        "--use_smart_uv",
        required=False,
        action="store_true",
        help="Use Blender's smart uv projection to create uv texture coordinates",
    )
    parser.add_argument(
        "--use_original_material_file",
        required=False,
        action="store_true",
        help="Whether to use the original material file for the mesh.",
    )
    parser.add_argument(
        "--prompt-overwrite-older",
        required=False,
        action="store_true",
        help="List outputs older than their matching inputs and prompt before overwriting.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root_dir = Path(args.root_dir)

    check_existing = False
    if args.operation == "cleanup":
        # Takes the original scans from Revo and runs 
        #.  - the MeshLabCleanup.mlx to get rid of duplicate vertices, etc, then downsample
        # Starts with xFullResx and puts xFullResx into the cleaned directory
        input_dir = root_dir / "OriginalRevoScans"
        output_dir = root_dir / "MeshLabCleaned"
        process_file = partial(
            cleanup_mesh,
            mlx_script_path=MLX_SCRIPTS["cleanup"],
        )
        check_existing = True
    elif args.operation == "voxelize":
        # The meshes in MeshCleaned up are currently being hand cleaned with the stem removed and put in BlenderStemRemoved
        #.  - this re-meshes those hand cleaned up meshes and splits the texturemap
        input_dir = root_dir / "BlenderStemRemoved"
        output_dir = root_dir / "Voxelized"
        process_file = partial(
            voxelize_mesh,
        )
        check_existing = True
    elif args.operation == "downsample_lowres":
        #
        input_dir = root_dir / "BlenderRebaked"
        output_dir = root_dir / "Final"
        process_file = partial(
            downsample_mesh,
            mlx_script_path=MLX_SCRIPTS["downsample_lowres"],
        )
    elif args.operation == "downsample_hires":
        #
        input_dir = root_dir / "BlenderRebaked"
        output_dir = root_dir / "Final"
        process_file = partial(
            downsample_mesh,
            mlx_script_path=MLX_SCRIPTS["downsample_hires"],
        )
    elif args.operation == "normalize":
        input_dir = root_dir / "BlenderRebaked"
        seam_dir = root_dir / "Voxelized"
        output_dir = root_dir / "NormalizedTexture"

        def process_file(input_path: Path, output_path: Path) -> Path:
            seam_name = input_path.stem.replace("NoStemHiRes", "") + "_seam.csv"
            return normalize_uv_projection(input_path, seam_dir / seam_name, output_path)
    elif args.operation == "rename":
        input_dir = root_dir / args.source_dir
        output_dir = root_dir / args.dest_dir
        if not args.source_dir or not args.dest_dir:
            raise ValueError("Both --source-dir and --dest-dir must be specified for the rename operation.")
        process_file = partial(
            _rename_mesh,
            use_original_material_file=args.use_original_material_file,
        )
    else:
        raise ValueError(f"No valid command {args.operation}")

    processed_files = process_meshes(
        input_dir,
        output_dir,
        process_file,
        prompt_overwrite_older=args.prompt_overwrite_older,
        check_existing_outputs=check_existing,
        new_tag_output_filename=args.dest_name,
        input_extensions=(".obj",),
    )
    if args.operation == "normalize":
        expected_count = sum(
            path.is_file() and path.suffix.lower() == ".obj"
            for path in input_dir.iterdir()
        )
        if len(processed_files) != expected_count:
            raise RuntimeError(
                f"Normalized {len(processed_files)} of {expected_count} OBJ files; "
                "see the errors above for details."
            )


if __name__ == "__main__":
    main()
