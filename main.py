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
   Final : Generate Hi and Low res meshees from BlenderTextured.
   """
from MeshLabCleanup import (
    cleanup_mesh,
    downsample_mesh,
    process_meshes,
    voxelize_mesh
)

from uv_texture import uv_texture


SCRIPT_DIR = Path(__file__).resolve().parent
MLX_SCRIPTS = {
    "cleanup": SCRIPT_DIR / "MeshLabScripts" / "MeshLabCleanup.mlx",
    "downsample_lowres": SCRIPT_DIR / "MeshLabScripts" / "downsample_lowres.mlx",
    "downsample_hires": SCRIPT_DIR / "MeshLabScripts" / "downsample_hires.mlx",
    "voxelize": SCRIPT_DIR / "MeshLabScripts" / "post_tunnel.mlx",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clean up, downsample_lowres, downsample_hires, or voxelize."
    )
    parser.add_argument(
        "--operation",
        required=True,
        choices=MLX_SCRIPTS,
        help="Operation to run; selects the corresponding .mlx script and input/output directories.",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Top-level folder containing OriginalRevoScans, MeshLabCleaned, MeshLabDownsampled, and BlenderTextures.",
    )
    parser.add_argument(
        "--prompt-overwrite-older",
        action="store_true",
        help="List outputs older than their matching inputs and prompt before overwriting.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root_dir = Path(args.input_dir)

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
    elif args.operation == "downsample_lowres":
        #
        input_dir = root_dir / "BlenderTextured"
        output_dir = root_dir / "Final"
        process_file = partial(
            downsample_mesh,
            mlx_script_path=MLX_SCRIPTS["downsample_lowres"],
        )
    elif args.operation == "downsample_hiures":
        #
        input_dir = root_dir / "BlenderTextured"
        output_dir = root_dir / "Final"
        process_file = partial(
            downsample_mesh,
            mlx_script_path=MLX_SCRIPTS["downsample_hires"],
        )

    process_meshes(
        input_dir,
        output_dir,
        process_file,
        prompt_overwrite_older=args.prompt_overwrite_older,
        check_existing_outputs=check_existing,
    )


if __name__ == "__main__":
    main()
