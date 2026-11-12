import subprocess
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from bake_textures import _blender_executable, bake_textures


class BakeTexturesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.blender = _blender_executable(None)
        except FileNotFoundError:
            raise unittest.SkipTest("Blender is not installed")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source.obj"
        self.target = self.root / "target.obj"
        self.output = self.root / "out" / "result.obj"
        Image.new("RGB", (16, 16), (230, 30, 20)).save(self.root / "source.png")
        (self.root / "source.mtl").write_text(
            "newmtl source\nKd 1 1 1\nmap_Kd source.png\n", encoding="utf-8"
        )
        self.source.write_text(self._mesh(0.05, True, True), encoding="utf-8")
        self.target.write_text(self._mesh(0, True, False), encoding="utf-8")

    @staticmethod
    def _mesh(z, uv, material):
        header = "mtllib source.mtl\n" if material else ""
        vertices = f"v 0 0 {z}\nv 1 0 {z}\nv 1 1 {z}\nv 0 1 {z}\n"
        coordinates = "vt 0.1 0.1\nvt 0.9 0.1\nvt 0.9 0.9\nvt 0.1 0.9\n" if uv else ""
        faces = "f 1/1 2/2 3/3\nf 1/1 3/3 4/4\n" if uv else "f 1 2 3\nf 1 3 4\n"
        return header + vertices + coordinates + ("usemtl source\n" if material else "") + faces

    def _assert_baked(self):
        self.assertTrue(self.output.is_file())
        self.assertTrue(self.output.with_suffix(".mtl").is_file())
        self.assertIn("map_Kd result.png", self.output.with_suffix(".mtl").read_text())
        with Image.open(self.output.with_suffix(".png")) as image:
            self.assertEqual(image.size, (16, 16))
            red, green, blue = image.convert("RGB").getpixel((8, 8))
            self.assertGreater(red, 180)
            self.assertLess(green, 80)
            self.assertLess(blue, 80)

    def test_bakes_using_existing_uvs(self):
        self.assertEqual(bake_textures(self.target, self.source, self.output, blender=self.blender), self.output.resolve())
        self._assert_baked()
        before = {line for line in self.target.read_text().splitlines() if line.startswith("vt ")}
        after = {line for line in self.output.read_text().splitlines() if line.startswith("vt ")}
        parse = lambda lines: {tuple(round(float(x), 5) for x in line.split()[1:3]) for line in lines}
        self.assertEqual(parse(before), parse(after))

    def test_smart_uv_creates_missing_uvs(self):
        self.target.write_text(self._mesh(0, False, False), encoding="utf-8")
        bake_textures(self.target, self.source, self.output, use_smart_uv=True, blender=self.blender)
        self._assert_baked()
        self.assertIn("\nvt ", self.output.read_text())

    def test_smart_uv_replaces_existing_uvs(self):
        bake_textures(self.target, self.source, self.output, use_smart_uv=True, blender=self.blender)
        self._assert_baked()
        before = {line for line in self.target.read_text().splitlines() if line.startswith("vt ")}
        after = {line for line in self.output.read_text().splitlines() if line.startswith("vt ")}
        parse = lambda lines: {tuple(round(float(x), 5) for x in line.split()[1:3]) for line in lines}
        self.assertNotEqual(parse(before), parse(after))

    def test_missing_uv_and_overwriting_input_fail(self):
        self.target.write_text(self._mesh(0, False, False), encoding="utf-8")
        with self.assertRaises(subprocess.CalledProcessError):
            bake_textures(self.target, self.source, self.output, blender=self.blender)
        self.assertFalse(self.output.exists())
        self.target.write_text(self._mesh(0, True, False), encoding="utf-8")
        with self.assertRaises(subprocess.CalledProcessError):
            bake_textures(self.target, self.source, self.source, blender=self.blender)
        self.assertIn("mtllib source.mtl", self.source.read_text())

    def test_missing_source_texture_fails(self):
        (self.root / "source.png").unlink()
        with self.assertRaises(subprocess.CalledProcessError):
            bake_textures(self.target, self.source, self.output, blender=self.blender)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
