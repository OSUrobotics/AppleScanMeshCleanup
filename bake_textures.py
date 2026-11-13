"""Bake a textured OBJ onto another OBJ using Blender

Use to repair the texture after removing the stem (use the smart_uv option to generate new UVs) 
Use to replace the texture after the voxelize operation (do NOT use the smart_uv option )
Can be run either in Blender or as a standalone Python script.

    python bake_textures.py target.obj source_dir output/result.obj --use_smart_uv

The output consists of result.obj, result.mtl, and result.png. By default the
target's existing UV coordinates are retained; --use_smart_uv replaces them.
The image dimensions are taken from the source texture(s). Blender must be on
PATH, or its executable can be supplied with --blender.
Use --cage-extrusion and --max-ray-distance to adjust bake projection distances
(defaults: 0.25 and 0.35, in Blender scene units).
"""

import argparse
import math
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


SCENE_PATH = Path(__file__).resolve().parent / "BlenderScripts" / "bake_textures_apples.blend"
DEFAULT_CAGE_EXTRUSION = 0.2
DEFAULT_MAX_RAY_DISTANCE = 0.45


def _projection_distance(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 3.4028234663852886e38:
        raise ValueError("Bake projection distances must be finite and between 0 and 3.4028234663852886e38.")
    return value


def _blender_executable(explicit):
    if explicit:
        executable = shutil.which(str(explicit))
    else:
        executable = shutil.which("blender")
        if not executable and sys.platform == "darwin":
            app_executable = Path("/Applications/Blender.app/Contents/MacOS/Blender")
            if app_executable.is_file():
                executable = str(app_executable)
    if not executable:
        raise FileNotFoundError("Blender executable not found; supply it with --blender.")
    return executable


def bake_textures(input_path, output_path, source_dir, use_smart_uv=False, blender=None,
                  cage_extrusion=DEFAULT_CAGE_EXTRUSION, max_ray_distance=DEFAULT_MAX_RAY_DISTANCE):
    """Bake source's texture onto target, writing a textured OBJ at destination.
        Input path and output path are both full paths
        Source path is the directory containing the mesh with the texture you want to bake onto input"""
    from filename_fixing import _split_filename_into_apple_type_and_number

    cage_extrusion = _projection_distance(cage_extrusion)
    max_ray_distance = _projection_distance(max_ray_distance)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input mesh not found: {input_path}")
    if input_path.suffix.lower() != ".obj":
        raise ValueError(f"Expected an OBJ mesh: {input_path}")
    if output_path.suffix.lower() != ".obj":
        raise ValueError("The destination must be an OBJ file path.")
    if not SCENE_PATH.is_file():
        raise FileNotFoundError(f"Blender settings file not found: {SCENE_PATH}")

    apple_name, apple_number = _split_filename_into_apple_type_and_number(input_path.name)
    # Look in the source directory for an obj file that starts with the apple name
    source_obj = None
    for path in source_dir.glob("*.obj"):
        apple_name_f, apple_number_f = _split_filename_into_apple_type_and_number(path.name)
        if apple_name_f == apple_name and apple_number_f == apple_number:
            source_obj = path
            break
    if source_obj is None:
        raise FileNotFoundError(f"No source OBJ found in {source_dir} starting with {apple_name}")

    command = [
        _blender_executable(blender), "--background", str(SCENE_PATH),
        "--python-exit-code", "1", "--python", str(Path(__file__).resolve()), "--",
        str(input_path), str(source_obj), str(output_path),
        "--cage-extrusion", str(cage_extrusion),
        "--max-ray-distance", str(max_ray_distance),
    ]
    if use_smart_uv:
        command.append("--use_smart_uv")
    print(*command, sep="\n")
    subprocess.run(command, check=True)
    return output_path


def _material_libraries(path):
    libraries = set()
    with path.open(encoding="utf-8") as mesh:
        for line in mesh:
            parts = shlex.split(line, comments=True)
            if parts and parts[0] == "mtllib":
                libraries.update((path.parent / name).resolve() for name in parts[1:])
    return libraries


def _bake_in_blender(target, source, destination, use_smart_uv,
                     cage_extrusion=DEFAULT_CAGE_EXTRUSION, max_ray_distance=DEFAULT_MAX_RAY_DISTANCE):
    import bpy

    cage_extrusion = _projection_distance(cage_extrusion)
    max_ray_distance = _projection_distance(max_ray_distance)
    scene = bpy.context.scene
    scene.render.bake.cage_extrusion = cage_extrusion
    scene.render.bake.max_ray_distance = max_ray_distance
    for obj in list(scene.objects):
        if obj.type == "MESH":
            bpy.data.objects.remove(obj, do_unlink=True)

    def import_mesh(path):
        before = set(bpy.data.objects)
        bpy.ops.wm.obj_import(filepath=str(path), use_split_objects=False, use_split_groups=False)
        objects = [obj for obj in bpy.data.objects if obj not in before and obj.type == "MESH"]
        if len(objects) != 1 or not objects[0].data.polygons:
            raise ValueError(f"Expected one nonempty mesh in {path}.")
        return objects[0]

    target_obj = import_mesh(target)
    source_obj = import_mesh(source)
    if not source_obj.data.uv_layers:
        raise ValueError(f"Source mesh has no UV coordinates: {source}")
    if not use_smart_uv and not target_obj.data.uv_layers:
        raise ValueError(f"Target mesh has no UV coordinates: {target}; use --use_smart_uv.")

    source_images = []
    protected = {target, source} | _material_libraries(target) | _material_libraries(source)
    for obj in (target_obj, source_obj):
        for material in obj.data.materials:
            if material and material.use_nodes:
                for node in material.node_tree.nodes:
                    if node.type == "TEX_IMAGE" and node.image:
                        image = node.image
                        image_path = Path(bpy.path.abspath(image.filepath)).resolve()
                        protected.add(image_path)
                        if obj == source_obj:
                            if not image_path.is_file():
                                raise FileNotFoundError(f"Source texture could not be loaded: {image_path}")
                            if not image.has_data:
                                image.reload()
                                if len(image.pixels) == 0:
                                    raise ValueError(f"Source texture could not be decoded: {image_path}")
                            source_images.append(image)
    if not source_images:
        raise ValueError(f"Source mesh has no image texture: {source}")

    outputs = {destination, destination.with_suffix(".mtl"), destination.with_suffix(".png")}
    if outputs & protected:
        raise ValueError("Output OBJ, MTL, and PNG must not overwrite an input mesh, material, or texture.")

    bpy.ops.object.select_all(action="DESELECT")
    target_obj.select_set(True)
    bpy.context.view_layer.objects.active = target_obj
    if use_smart_uv:
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.smart_project()
        bpy.ops.object.mode_set(mode="OBJECT")

    width = max(image.size[0] for image in source_images)
    height = max(image.size[1] for image in source_images)
    image = bpy.data.images.new("BakedTexture", width=width, height=height, alpha=True)
    material = bpy.data.materials.new(name="BakedMaterial")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    texture = nodes.new("ShaderNodeTexImage")
    texture.image = image
    nodes.active = texture
    target_obj.data.materials.clear()
    target_obj.data.materials.append(material)
    for polygon in target_obj.data.polygons:
        polygon.material_index = 0

    source_obj.select_set(True)
    bpy.context.view_layer.objects.active = target_obj
    bpy.ops.object.bake(type=scene.cycles.bake_type)
    material.node_tree.links.new(texture.outputs["Color"], nodes.get("Principled BSDF").inputs["Base Color"])

    destination.parent.mkdir(parents=True, exist_ok=True)
    image.filepath_raw = str(destination.with_suffix(".png"))
    image.file_format = "PNG"
    image.save()
    source_obj.select_set(False)
    bpy.ops.wm.obj_export(filepath=str(destination), export_selected_objects=True, export_materials=True)
    if not all(path.is_file() for path in outputs):
        raise RuntimeError(f"Blender did not produce all expected output files for {destination}.")


def _parse_args(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", nargs="?", default="/Users/cindygrimm/PycharmProjects/data/ScannedModels/ProsserEnvyApples/BlenderStemRemoved/Envy1NoStemHiRes.obj", type=Path, help="OBJ mesh to receive the baked texture.")
    parser.add_argument("source", nargs="?", default="/Users/cindygrimm/PycharmProjects/data/ScannedModels/ProsserEnvyApples/OriginalRevoScans", type=Path, help="Textured OBJ mesh to bake from.")
    parser.add_argument("destination", nargs="?", default="/Users/cindygrimm/PycharmProjects/data/ScannedModels/ProsserEnvyApples/BlenderNoStemRebaked/Envy_1_FullRes.obj", type=Path, help="Output OBJ file path.")
    parser.add_argument("--use_smart_uv", action="store_true", help="Replace the target UVs using Smart UV Project.")
    parser.add_argument("--cage-extrusion", type=_projection_distance, default=DEFAULT_CAGE_EXTRUSION,
                        help="Bake cage extrusion in scene units (default: 0.25).")
    parser.add_argument("--max-ray-distance", type=_projection_distance, default=DEFAULT_MAX_RAY_DISTANCE,
                        help="Maximum bake ray distance in scene units (default: 0.35; 0 means unlimited).")
    parser.add_argument("--blender", nargs="?", default="/Applications/Blender.app/Contents/MacOS/Blender", help="Path to the Blender executable.")
    return parser.parse_args(arguments)


if __name__ == "__main__":
    if "--" in sys.argv and "bpy" in sys.modules:
        args = _parse_args(sys.argv[sys.argv.index("--") + 1:])
        print(f" Baking texture for target: {args.target}")
        print(f" Source: {args.source}")
        print(f" Destination: {args.destination}")
        _bake_in_blender(args.target, args.source, args.destination, args.use_smart_uv,
                         args.cage_extrusion, args.max_ray_distance)
    else:
        args = _parse_args()
        bake_textures(input_path=args.target, output_path=args.destination, source_dir=args.source,
                      use_smart_uv=args.use_smart_uv, blender=args.blender,
                      cage_extrusion=args.cage_extrusion, max_ray_distance=args.max_ray_distance)
