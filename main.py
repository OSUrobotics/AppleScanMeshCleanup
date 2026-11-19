import argparse
from functools import partial
import os
from pathlib import Path

""" Going from oroginal Revo scans to consistantly parameterized, topologically spherical meshes with a cylindrical parameterization
Directories
   OriginalRevoScans : The original output meshes, labeled Apple#FullRes.obj
   MeshLabCleaned : Derived from the Revo scans by automatic filtering and downsampling
           - it is safe to overwrite these 
   BlenderStemRemoved : Manually cut out the stem 
   Voxelized : Generate a new, clean mesh using voxelization and build the cylindrical parameterization. Runs on BlenderStemRemoved
           - it is safe to overwrite these
   Parameterized : Derived from Voxelized by adding the cylindrical parameterization.
           - it is safe to overwrite these
   BlenderTextured : Manually Bake the texture map from the meshlabcleaned and clean up the stem texture
   NormalizedTexture : Normalize UV charts and warp the rebaked texture using seams from Voxelized
   Final : Generate Hi and Low res meshees from BlenderTextured.
   """
from MeshLabCleanup import (
    cleanup_mesh,
    downsample_mesh,
    parameterize_mesh,
    process_meshes,
    voxelize_mesh,
    _rename_mesh
)
from bake_textures import (
    DEFAULT_CAGE_EXTRUSION,
    DEFAULT_MAX_RAY_DISTANCE,
    _projection_distance,
    bake_textures,
)
from align_meshes import align_meshes

from normalize_uv_projection import normalize_uv_projection


SCRIPT_DIR = Path(__file__).resolve().parent
MLX_SCRIPTS = {
    "cleanup": SCRIPT_DIR / "MeshLabScripts" / "MeshLabCleanup.mlx",
    "downsample_lowres": SCRIPT_DIR / "MeshLabScripts" / "downsample_lowres.mlx",
    "downsample_hires": SCRIPT_DIR / "MeshLabScripts" / "downsample_hires.mlx",
    "downsample_fullres": SCRIPT_DIR / "MeshLabScripts" / "downsample_fullres.mlx",
    "voxelize": SCRIPT_DIR / "MeshLabScripts" / "post_tunnel.mlx",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clean up, downsample, voxelize, parameterize, bake textures, normalize UV textures, or align with canonical."
    )
    parser.add_argument(
        "--root-dir",
        required=True,
        help="Path to the root directories containing the meshes to process.",
    )
    parser.add_argument(
        "--operation",
        required=True,
        choices=(*MLX_SCRIPTS, "rename", "parameterize", "normalize_uv_projection", "bake_textures", "align_meshes", "downsample"),
        help="Operation to run; selects the input/output directories.",
    )
    parser.add_argument(
        "--source-dir",
        required=False,
        help="One of the subdirectories: eg OriginalRevoScans, MeshLabCleaned, MeshLabDownsampled, and BlenderTextures.",
    )
    parser.add_argument(
        "--dest-dir",
        required=False,
        help="One of the subdirectories: egMeshLabCleaned, MeshLabDownsampled, BlenderStemRemoved, Voxelized, BlenderRebaked, NormalizedTexture, and Final.",
    )
    parser.add_argument(
        "--dest-name",
        required=False,
        help="What to append after Apple# for the file name.",
    )
    parser.add_argument(
        "--replace-existing",
        required=False,
        action="store_true",
        help="Overwrite existing files if they exist.",
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
    parser.add_argument(
        "--blender",
        required=False,
        default="/Applications/Blender.app/Contents/MacOS/Blender",
        help="Path to the Blender executable for bake_textures option.",
    )
    parser.add_argument(
        "--use_smart_uv",
        required=False,
        action="store_true",
        help="Use Blender's smart uv projection to override existing uv texture coordinates for bake_textures option.",
    )
    parser.add_argument(
        "--cage-extrusion", type=_projection_distance, default=DEFAULT_CAGE_EXTRUSION,
        help="Bake cage extrusion in scene units (default: 0.25).",
    )
    parser.add_argument(
        "--max-ray-distance", type=_projection_distance, default=DEFAULT_MAX_RAY_DISTANCE,
        help="Maximum bake ray distance in scene units (default: 0.35; 0 means unlimited).",
    )
    parser.add_argument(
        "--canonical", type=str, default="fullres",
        help="Canonical mesh to align to (default: fullres).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root_dir = Path(args.root_dir)

    replace_existing = args.replace_existing
    if args.operation == "cleanup":
        # Takes the original scans from Revo and runs 
        #.  - the MeshLabCleanup.mlx to get rid of duplicate vertices, etc, then downsample
        # Starts with xFullResx and puts xFullResx into the cleaned directory
        input_dir = root_dir / "OriginalRevoScans"
        output_dir = root_dir / "MeshLabCleaned"
        stem_dir = root_dir / "BlenderStemRemoved"
        if not os.path.exists(stem_dir):
            os.makedirs(stem_dir)
        
        process_file = partial(
            cleanup_mesh,
            mlx_script_path=MLX_SCRIPTS["cleanup"],
        )
    elif args.operation == "voxelize":
        # The meshes in MeshCleaned up are currently being hand cleaned with the stem removed and put in BlenderStemRemoved
        # They are then re-textured automatically and placed in BlenderNoStemRebaked
        #.  - this re-meshes those hand cleaned up meshes and splits the texturemap
        input_dir = root_dir / "BlenderNoStemRebaked"
        output_dir = root_dir / "Voxelized"
        process_file = partial(
            voxelize_mesh,
        )
    elif args.operation == "parameterize":
        # The meshes labeled _voxel are parameterized with cylindrical coordinates
        input_dir = root_dir / "Voxelized"
        output_dir = root_dir / "Parameterized"
        process_file = partial(
            parameterize_mesh,
        )
    elif args.operation == "downsample":
        # Do all three downsample operations: lowres, hires, and fullres
        input_dir = root_dir / "BlenderRebaked"
        output_dir = root_dir / "Final"
        process_file = partial(
            downsample_mesh,
            mlx_script_paths=[
                MLX_SCRIPTS["downsample_lowres"],
                MLX_SCRIPTS["downsample_hires"],
                MLX_SCRIPTS["downsample_fullres"],
            ],
        )
    elif args.operation == "downsample_lowres":
        #
        input_dir = root_dir / "BlenderRebaked"
        output_dir = root_dir / "Final"
        process_file = partial(
            downsample_mesh,
            mlx_script_paths=[MLX_SCRIPTS["downsample_lowres"]],
        )
    elif args.operation == "downsample_hires":
        #
        input_dir = root_dir / "BlenderRebaked"
        output_dir = root_dir / "Final"
        process_file = partial(
            downsample_mesh,
            mlx_script_paths=[MLX_SCRIPTS["downsample_hires"]],
        )
    elif args.operation == "downsample_fullres":
        #
        input_dir = root_dir / "BlenderRebaked"
        output_dir = root_dir / "Final"
        process_file = partial(
            downsample_mesh,
            mlx_script_paths=[MLX_SCRIPTS["downsample_fullres"]],
        )
    elif args.operation == "normalize":
        input_dir = root_dir / "BlenderRebaked"
        seam_dir = root_dir / "Parameterized"
        output_dir = root_dir / "NormalizedTexture"

        process_file = partial(
            normalize_uv_projection,
            seam_path=seam_dir
        )
    elif args.operation == "align_meshes":
        input_dir = root_dir / "NormalizedTexture"
        output_dir = root_dir / "AlignedMeshes"

        canonical_path = root_dir.parent / "Canonical" / f"apple_{args.canonical}.obj"
        process_file = partial(
            align_meshes,
            canonical_path=canonical_path
        )
    elif args.operation == "rename":
        input_dir = root_dir / args.source_dir
        output_dir = root_dir / args.dest_dir
        if not args.source_dir or not args.dest_dir:
            raise ValueError("Both --source-dir and --dest-dir must be specified for the rename operation.")
        process_file = partial(
            _rename_mesh,
            use_original_material_file=args.use_original_material_file,
        )
    elif args.operation == "bake_textures":
        if args.use_smart_uv:
            input_dir = root_dir / "BlenderStemRemoved"
            output_dir = root_dir / "BlenderNoStemRebaked"
        else:
            input_dir = root_dir / "Parameterized" 
            output_dir = root_dir / "BlenderRebaked"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        source_dir = root_dir / "BlenderRebaked_v2"
        process_file = partial(
            bake_textures,
            source_dir=source_dir,
            use_smart_uv=args.use_smart_uv,
            blender=args.blender,
            cage_extrusion=args.cage_extrusion,
            max_ray_distance=args.max_ray_distance,
        )
    elif args.operation == "normalize_uv_projection":
        input_dir = root_dir / "BlenderRebaked"
        seam_dir = root_dir / "Parameterized"
        output_dir = root_dir / "NormalizedTexture"

        process_file = partial(
            normalize_uv_projection,
            seam_path=seam_dir,
        )
    else:
        raise ValueError(f"No valid command {args.operation}")

    processed_files = process_meshes(
        input_dir,
        output_dir,
        process_file,
        prompt_overwrite_older=args.prompt_overwrite_older,
        replace_existing_outputs=replace_existing,
        new_tag_output_filename=args.dest_name,
        input_extensions=(".obj",),
    )


if __name__ == "__main__":
    main()
