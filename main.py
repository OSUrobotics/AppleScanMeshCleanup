import argparse
from functools import partial
from pathlib import Path

""" Going from oroginal Revo scans to consistantly parameterized, topologically spherical meshes with a cylindrical parameterization
Directories
   OriginalRevoScans : The original output meshes, labeled Apple#FullRes.obj
   MeshLabCleaned : Derived from the Revo scans by filteringg and downsampling
   BlenderTextured : Manually cut out the stem and regenerate/repair the top texture using smart UV mapping
   Remeshed : Generate a new, clean mesh using voxelization
   BlenderRemeshedTextured : Manually create the cylindrical texture mapping and bake the new texture in
   MeshLabDownsampled : Downsample the remeshed texture
   """
from MeshLabCleanup import (
    cleanup_mesh,
    downsample_mesh,
    fix_tunnels_mesh,
    process_meshes,
    voxelize_mesh
)


SCRIPT_DIR = Path(__file__).resolve().parent
MLX_SCRIPTS = {
    "cleanup": SCRIPT_DIR / "MeshLabScripts" / "MeshLabCleanup.mlx",
    "downsample": SCRIPT_DIR / "MeshLabScripts" / "downsample.mlx",
    "fixtunnels": SCRIPT_DIR / "MeshLabScripts" / "post_tunnel.mlx",
    "voxelize": SCRIPT_DIR / "MeshLabScripts" / "post_tunnel.mlx",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clean up, downsample, voxelize, or fix tunnels in meshes."
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
        #.  - remove tunnels
        #.  - the post_tunnel.mlx to clean up holes
        # Starts with xFullResx and puts xFullResx into the cleaned directory
        input_dir = root_dir / "OriginalRevoScans"
        output_dir = root_dir / "MeshLabCleaned"
        process_file = partial(
            cleanup_mesh,
            mlx_script_path=MLX_SCRIPTS["cleanup"],
            post_tunnel_mlx_script_path=MLX_SCRIPTS["fixtunnels"],
        )
    elif args.operation == "fixtunnels":
        # The meshes in MeshCleaned up are currently being hand cleaned up and put in Blender Textured
        #.  - this is meant to fix tunnels on those hand cleaned up meshes. It should not be required
        #.    after this round of clean up
        input_dir = root_dir / "BlenderTextured"
        output_dir = root_dir / "BlenderTexturedFixed"
        process_file = partial(
            fix_tunnels_mesh,
            post_tunnel_mlx_script_path=MLX_SCRIPTS["fixtunnels"],
        )
    elif args.operation == "voxelize":
        # The meshes in MeshCleaned up are currently being hand cleaned up and put in Blender Textured
        #.  - this is meant to re-mesh those hand cleaned up meshes. 
        input_dir = root_dir / "BlenderTextured"
        output_dir = root_dir / "BlenderTexturedFixed"
        process_file = partial(
            voxelize_mesh,
            post_tunnel_mlx_script_path=MLX_SCRIPTS["fixtunnels"],
        )
    elif args.operation == "downsample":
        #
        input_dir = root_dir / "BlenderTextured"
        output_dir = root_dir / "MeshLabDownsample"
        process_file = partial(
            downsample_mesh,
            mlx_script_path=MLX_SCRIPTS["downsample"],
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
