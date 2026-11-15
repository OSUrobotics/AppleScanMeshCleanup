import csv
import shutil
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from fix_texture_maps import flood_fill_texture_map
from MeshLabCleanup import downsample_mesh
import main


class FillTextureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.mesh = self.root / "Apple1.obj"
        self.seams = self.root / "Apple1_seam.csv"
        self.texture = self.root / "texture.png"
        self.output = self.root / "filled.png"

        top = [(0.1, 0.9), (0.3, 0.9), (0.2, 0.7)]
        bottom = [(0.7, 0.3), (0.9, 0.3), (0.8, 0.1)]
        body_top = [(0.2, 0.65), (0.4, 0.65), (0.6, 0.65), (0.8, 0.65)]
        body_bottom = [(0.2, 0.35), (0.4, 0.35), (0.6, 0.35), (0.8, 0.35)]
        points = top + bottom + body_top + body_bottom
        faces = [(0, 1, 2), (3, 4, 5)]
        for i in range(3):
            faces.extend([(6 + i, 7 + i, 10 + i), (7 + i, 11 + i, 10 + i)])
        with self.mesh.open("w", encoding="utf-8") as file:
            file.write("mtllib Apple1.mtl\n")
            for i, (u, v) in enumerate(points):
                file.write(f"v {u} {v} {0.1 * (i % 3)}\nvt {u} {v}\n")
            for face in faces:
                file.write("f " + " ".join(f"{i + 1}/{i + 1}" for i in face) + "\n")

        # The real exporter omits the first vertex from the two circular rings.
        groups = [
            ([1, 2], top[1:]),
            ([4, 5], bottom[1:]),
            ([1, 2], body_top[1:3]),
            ([0, 3], [body_top[3], body_bottom[3]]),
            ([4, 5], body_bottom[1:3]),
            ([3, 0], [body_bottom[0], body_top[0]]),
        ]
        with self.seams.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            for ids, uv in groups:
                writer.writerow(ids)
                writer.writerow([point[0] for point in uv])
                writer.writerow([point[1] for point in uv])

        image = Image.new("RGB", (101, 101), "white")
        draw = ImageDraw.Draw(image)
        xy = lambda p: (p[0] * 100, (1 - p[1]) * 100)
        draw.polygon([xy(p) for p in top], fill=(200, 10, 10))
        draw.polygon([xy(p) for p in bottom], fill=(10, 200, 10))
        draw.polygon([xy(p) for p in (body_top[0], body_top[-1], body_bottom[-1], body_bottom[0])],
                     fill=(10, 10, 200))
        image.save(self.texture)

    def test_fill_uses_opposite_seam_charts_and_preserves_existing_pixels(self):
        flood_fill_texture_map(self.texture, self.mesh, self.seams, self.output)
        with Image.open(self.texture) as original, Image.open(self.output) as filled:
            before, after = np.array(original), np.array(filled)
        nonwhite = np.any(before != 255, axis=2)
        np.testing.assert_array_equal(after[nonwhite], before[nonwhite])
        self.assertFalse(np.any(np.all(after == 255, axis=2)))
        np.testing.assert_array_equal(after[8, 20], [10, 10, 200])
        np.testing.assert_array_equal(after[33, 40], [200, 10, 10])
        np.testing.assert_array_equal(after[50, 81], [10, 10, 200])

    def test_transparency_and_in_place_output(self):
        with Image.open(self.texture) as image:
            rgba = np.array(image.convert("RGBA"))
        rgba[50, 50] = [3, 4, 5, 0]
        Image.fromarray(rgba).save(self.texture)
        flood_fill_texture_map(self.texture, self.mesh, self.seams, self.texture)
        with Image.open(self.texture) as result:
            self.assertEqual(result.getpixel((50, 50))[3], 255)
            self.assertEqual(result.getpixel((15, 20)), (200, 10, 10, 255))

    def test_rejects_mismatched_seam_and_uncolored_chart(self):
        self.seams.write_text(self.seams.read_text().replace("0.3", "0.31", 1))
        with self.assertRaisesRegex(ValueError, "does not match"):
            flood_fill_texture_map(self.texture, self.mesh, self.seams, self.output)
        self.assertFalse(self.output.exists())

    def test_rejects_chart_with_no_source_color(self):
        image = np.array(Image.open(self.texture))
        image[np.all(image == [200, 10, 10], axis=2)] = 255
        Image.fromarray(image).save(self.texture)
        with self.assertRaisesRegex(ValueError, "contains no non-white texture pixels"):
            flood_fill_texture_map(self.texture, self.mesh, self.seams, self.output)
        self.assertFalse(self.output.exists())

    def test_rejects_overwriting_mesh_or_seam(self):
        with self.assertRaisesRegex(ValueError, "must not overwrite"):
            flood_fill_texture_map(self.texture, self.mesh, self.seams, self.mesh)

    def test_command_line(self):
        subprocess.run(
            [sys.executable, str(Path(__file__).with_name("fix_texture_maps.py")),
             str(self.texture), str(self.mesh), str(self.seams), str(self.output)],
            check=True, capture_output=True, text=True,
        )
        self.assertTrue(self.output.is_file())

    def test_downsample_fills_each_generated_texture(self):
        source = self.root / "BlenderRebaked" / "Apple1.obj"
        destination = self.root / "Final" / "Apple1.obj"
        scripts = [Path("downsample_lowres.mlx"), Path("downsample_hires.mlx")]
        material = self.root / "Final" / "Apple1.mtl"
        with (patch("MeshLabCleanup._find_seam_file", return_value=self.seams),
              patch("MeshLabCleanup._process_mesh") as process,
              patch("MeshLabCleanup._reorient_mesh"),
              patch("MeshLabCleanup._read_obj", return_value=(None, None, None, material)),
              patch("MeshLabCleanup._read_texture", return_value=self.texture),
              patch("MeshLabCleanup.flood_fill_texture_map") as fill,
              patch("MeshLabCleanup.print_mesh_topology")):
            downsample_mesh(source, destination, scripts)
        self.assertEqual(process.call_count, 2)
        self.assertEqual(fill.call_count, 2)
        for call in fill.call_args_list:
            self.assertEqual(call.args, (self.texture, source, self.seams, self.texture))

    def test_downsample_respects_existing_outputs(self):
        source = self.root / "BlenderRebaked" / "Apple1.obj"
        destination = self.root / "Final" / "Apple1.obj"
        destination.parent.mkdir()
        destination.with_name("Apple_001_lowres.obj").write_text("existing")
        with (patch("MeshLabCleanup._find_seam_file", return_value=self.seams) as find_seam,
              patch("MeshLabCleanup._process_mesh") as process,
              patch("MeshLabCleanup.print_mesh_topology")):
            downsample_mesh(source, destination, [Path("downsample_lowres.mlx")], replace_existing=False)
        process.assert_not_called()
        find_seam.assert_not_called()

    def test_all_final_operations_use_filling_downsample(self):
        for operation, count in (("downsample", 3), ("downsample_lowres", 1),
                                 ("downsample_hires", 1), ("downsample_fullres", 1)):
            args = Namespace(root_dir=str(self.root), operation=operation, replace_existing=False,
                             prompt_overwrite_older=False, dest_name=None)
            with self.subTest(operation=operation), patch("main.parse_args", return_value=args), \
                    patch("main.process_meshes") as process:
                main.main()
            processor = process.call_args.args[2]
            self.assertIs(processor.func, downsample_mesh)
            self.assertEqual(len(processor.keywords["mlx_script_paths"]), count)
            self.assertFalse(processor.keywords["replace_existing"])

    def test_downsample_with_meshlab_fills_saved_texture(self):
        source_dir = self.root / "BlenderRebaked"
        seam_dir = self.root / "Voxelized"
        final_dir = self.root / "Final"
        for directory in (source_dir, seam_dir, final_dir):
            directory.mkdir()
        source = source_dir / "Apple1.obj"
        shutil.copy(self.mesh, source)
        shutil.copy(self.seams, seam_dir / "Apple1_seam.csv")
        shutil.copy(self.texture, source_dir / "texture.png")
        (source_dir / "Apple1.mtl").write_text(
            "newmtl apple\nKd 1 1 1\nmap_Kd texture.png\n", encoding="utf-8"
        )
        script = self.root / "downsample_lowres.mlx"
        shutil.copy(Path(__file__).with_name("MeshLabScripts") / script.name, script)
        with (patch("MeshLabCleanup._find_centers", return_value=([0, 1, 0], [0, 0, 0])),
              patch("MeshLabCleanup.print_mesh_topology")):
            result = downsample_mesh(source, final_dir / "Apple1.obj", [script])
        self.assertTrue(result.is_file())
        with Image.open(final_dir / "Apple_001_lowres.png") as filled:
            self.assertFalse(np.any(np.all(np.array(filled)[:, :, :3] == 255, axis=2)))
            self.assertEqual(filled.getpixel((20, 8))[:3], (10, 10, 200))
        with Image.open(source_dir / "texture.png") as original:
            self.assertEqual(original.getpixel((20, 8)), (255, 255, 255))


if __name__ == "__main__":
    unittest.main()
