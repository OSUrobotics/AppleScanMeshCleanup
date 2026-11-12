import tempfile
import unittest
from pathlib import Path

from PIL import Image

from MeshLabCleanup import _process_mesh


class MeshLabTextureNameTests(unittest.TestCase):
    def test_processed_mesh_uses_output_name_for_texture(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.obj"
            output = root / "result.obj"
            script = root / "empty.mlx"
            Image.new("RGB", (4, 4), (220, 30, 10)).save(root / "old_texture.png")
            (root / "source.mtl").write_text(
                "newmtl mat\nKd 1 1 1\nmap_Kd old_texture.png\n", encoding="utf-8"
            )
            source.write_text(
                "mtllib source.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\n"
                "vt 0 0\nvt 1 0\nvt 0 1\nusemtl mat\nf 1/1 2/2 3/3\n",
                encoding="utf-8",
            )
            script.write_text("<FilterScript></FilterScript>\n", encoding="utf-8")

            _process_mesh(source, output, script)

            self.assertTrue(output.is_file())
            self.assertIn("map_Kd result.png", (root / "result.obj.mtl").read_text())
            with Image.open(root / "result.png") as image:
                self.assertEqual(image.getpixel((0, 0))[:3], (220, 30, 10))
            self.assertTrue((root / "old_texture.png").is_file())

    def test_untextured_mesh_still_exports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.obj"
            output = root / "result.obj"
            script = root / "empty.mlx"
            source.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
            script.write_text("<FilterScript></FilterScript>\n", encoding="utf-8")

            _process_mesh(source, output, script)

            self.assertTrue(output.is_file())
            self.assertFalse((root / "result.png").exists())


if __name__ == "__main__":
    unittest.main()
