import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from bake_textures import SCENE_PATH, _blender_executable


class BakeTargetImageTests(unittest.TestCase):
    def test_command_line_custom_projection_reaches_offset_source(self):
        try:
            blender = _blender_executable(None)
        except FileNotFoundError:
            raise unittest.SkipTest("Blender is not installed")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir = root / "sources"
            source_dir.mkdir()
            Image.new("RGB", (16, 16), (230, 30, 20)).save(source_dir / "source.png")
            (source_dir / "source.mtl").write_text(
                "newmtl source\nKd 1 1 1\nmap_Kd source.png\n", encoding="utf-8"
            )
            vertices = "v 0 0 {0}\nv 1 0 {0}\nv 1 1 {0}\nv 0 1 {0}\n"
            geometry = "vt 0.1 0.1\nvt 0.9 0.1\nvt 0.9 0.9\nvt 0.1 0.9\nf 1/1 2/2 3/3\nf 1/1 3/3 4/4\n"
            (source_dir / "Envy1Source.obj").write_text(
                "mtllib source.mtl\n" + vertices.format(0.3) + "usemtl source\n" + geometry,
                encoding="utf-8",
            )
            target = root / "Envy1Target.obj"
            target.write_text(vertices.format(0) + geometry, encoding="utf-8")
            output = root / "out" / "result.obj"
            subprocess.run(
                [sys.executable, str(Path(__file__).with_name("bake_textures.py")),
                 str(target), str(source_dir), str(output), "--blender", blender,
                 "--cage-extrusion", "0.4", "--max-ray-distance", "0.5"],
                check=True,
            )
            with Image.open(output.with_suffix(".png")) as image:
                red, green, blue = image.convert("RGB").getpixel((8, 8))
                self.assertGreater(red, 180)
                self.assertLess(green, 80)
                self.assertLess(blue, 80)

    def test_new_target_image_is_unlinked_during_bake_and_linked_afterward(self):
        try:
            blender = _blender_executable(None)
        except FileNotFoundError:
            raise unittest.SkipTest("Blender is not installed")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, color in (("source", (230, 30, 20)), ("target", (20, 30, 230))):
                Image.new("RGB", (16, 16), color).save(root / f"{name}.png")
                (root / f"{name}.mtl").write_text(
                    f"newmtl {name}\nKd 1 1 1\nmap_Kd {name}.png\n", encoding="utf-8"
                )
                z = 0.05 if name == "source" else 0
                (root / f"{name}.obj").write_text(
                    f"mtllib {name}.mtl\n"
                    f"v 0 0 {z}\nv 1 0 {z}\nv 1 1 {z}\nv 0 1 {z}\n"
                    "vt 0.1 0.1\nvt 0.9 0.1\nvt 0.9 0.9\nvt 0.1 0.9\n"
                    f"usemtl {name}\nf 1/1 2/2 3/3\nf 1/1 3/3 4/4\n",
                    encoding="utf-8",
                )

            code = """
import bpy
import linecache
import sys
from pathlib import Path
sys.path.insert(0, %r)
from bake_textures import _bake_in_blender
root = Path(%r)
states = []
def inspect_target(frame, event, arg):
    if (frame.f_code.co_name == '_bake_in_blender' and event == 'line' and
            linecache.getline(frame.f_code.co_filename, frame.f_lineno).strip().startswith('bpy.ops.object.bake(')):
        node = frame.f_locals['material'].node_tree.nodes.active
        states.append((node.type, node.image.name, node.image.source,
                       len(node.outputs['Color'].links), tuple(node.image.size),
                       bpy.context.scene.render.bake.cage_extrusion,
                       bpy.context.scene.render.bake.max_ray_distance))
    return inspect_target
sys.settrace(inspect_target)
_bake_in_blender(root / 'target.obj', root / 'source.obj', root / 'out' / 'result.obj', False)
_bake_in_blender(root / 'target.obj', root / 'source.obj', root / 'custom' / 'result.obj', False,
                 cage_extrusion=0.1, max_ray_distance=0.2)
sys.settrace(None)
assert len(states) == 2, states
for state, expected in zip(states, [(0.25, 0.35), (0.1, 0.2)]):
    assert state[0] == 'TEX_IMAGE' and state[1].startswith('BakedTexture'), state
    assert state[2:5] == ('GENERATED', 0, (16, 16)), state
    assert all(abs(actual - wanted) < 1e-6 for actual, wanted in zip(state[5:], expected)), state
target = bpy.context.view_layer.objects.active
nodes = target.active_material.node_tree.nodes
node = nodes.active
assert node.image.name.startswith('BakedTexture')
assert any(link.to_node == nodes.get('Principled BSDF') and
           link.to_socket.name == 'Base Color' for link in node.outputs['Color'].links)
""" % (str(Path(__file__).resolve().parent), str(root))
            subprocess.run(
                [blender, "--background", str(SCENE_PATH), "--python-exit-code", "1",
                 "--python-expr", code],
                check=True,
                cwd=Path(__file__).resolve().parent,
            )
            with Image.open(root / "out" / "result.png") as image:
                red, green, blue = image.convert("RGB").getpixel((8, 8))
                self.assertGreater(red, 180)
                self.assertLess(green, 80)
                self.assertLess(blue, 80)
            with Image.open(root / "target.png") as image:
                self.assertEqual(image.getpixel((8, 8)), (20, 30, 230))
            self.assertIn("map_Kd result.png", (root / "out" / "result.mtl").read_text())


if __name__ == "__main__":
    unittest.main()
