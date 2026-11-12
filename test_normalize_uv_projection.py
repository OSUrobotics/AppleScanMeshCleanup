import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

import main
import normalize_uv_projection as normalization
from normalize_uv_projection import _read_obj, normalize_uv_projection


class NormalizeUVProjectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.input_dir = self.root / "BlenderRebaked"
        self.seam_dir = self.root / "Voxelized"
        self.input_dir.mkdir()
        self.seam_dir.mkdir()
        self.input_path = self.input_dir / "AppleNoStemHiRes.obj"
        self.seam_path = self.seam_dir / "Apple_seam.csv"
        self.output_path = self.root / "NormalizedTexture" / self.input_path.name

        top = [(0.99, 0.1), (0.9, 0.18), (0.81, 0.1), (0.9, 0.02)]
        bottom = [(0.19, 0.9), (0.1, 0.98), (0.01, 0.9), (0.1, 0.82)]
        body = [(0.49, 0.02), (0.98, 0.49), (0.51, 0.98), (0.02, 0.51)]
        charts = [top, bottom, body]
        centers = [(0.9, 0.1), (0.1, 0.9), (0.5, 0.5)]
        with self.input_path.open("w", encoding="utf-8") as obj:
            obj.write("mtllib AppleNoStemHiRes.mtl\n")
            for index in range(15):
                obj.write(f"v {index} 0 0\n")
            for chart, center in zip(charts, centers):
                for u, v in [*chart, center]:
                    obj.write(f"vt {u} {v}\n")
            obj.write("usemtl apple\n")
            for chart in range(3):
                first = 5 * chart + 1
                for index in range(4):
                    a, b, c = first + index, first + (index + 1) % 4, first + 4
                    obj.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")

        material = self.input_path.with_suffix(".mtl")
        material.write_text("newmtl apple\nKd 1 1 1\nmap_Kd source.png\n", encoding="utf-8")
        x, y = np.meshgrid(np.arange(128), np.arange(128))
        image = np.stack((x * 2, y * 2, np.full_like(x, 100)), axis=2).astype(np.uint8)
        Image.fromarray(image).save(self.input_dir / "source.png")

        with self.seam_path.open("w", encoding="utf-8", newline="") as seam_file:
            writer = csv.writer(seam_file)
            for ids, points in [
                (list(range(4)), top),
                (list(range(5, 9)), bottom),
                *[
                    ([10 + i, 10 + (i + 1) % 4], [body[i], body[(i + 1) % 4]])
                    for i in range(4)
                ],
            ]:
                writer.writerow(ids)
                writer.writerow([point[0] for point in points])
                writer.writerow([point[1] for point in points])

    def test_normalizes_boundaries_and_warps_texture(self):
        normalize_uv_projection(self.input_path, self.seam_path, self.output_path)
        _, uvs, faces, material = _read_obj(self.output_path)
        self.assertEqual(len(faces), 12)
        self.assertTrue(material.is_file())
        self.assertIn("map_Kd AppleNoStemHiRes.png", material.read_text())
        for indices, center in [(range(4), (0.9, 0.1)), (range(5, 9), (0.1, 0.9))]:
            radii = np.linalg.norm(uvs[list(indices)] - center, axis=1)
            np.testing.assert_allclose(radii, 0.1, atol=1e-8)
        for point in uvs[10:14]:
            self.assertAlmostEqual(abs(point[0] - 0.5) + abs(point[1] - 0.5), 0.5)
        with Image.open(self.output_path.with_suffix(".png")) as result:
            self.assertEqual(result.size, (128, 128))
            self.assertGreater(result.getpixel((64, 64))[3], 0)
            pixel = result.getpixel((64, 64))
            self.assertLess(abs(pixel[0] - 128), 5)
            self.assertLess(abs(pixel[1] - 126), 5)
            self.assertEqual(pixel[2], 100)

    def test_normalized_seam_csv_preserves_ids_and_uses_output_uvs(self):
        normalize_uv_projection(self.input_path, self.seam_path, self.output_path)
        output_seam = self.output_path.with_name(self.output_path.stem + "_seam.csv")
        input_records = normalization._read_seam_records(self.seam_path)
        output_records = normalization._read_seam_records(output_seam)
        _, before, _, _ = _read_obj(self.input_path)
        _, after, _, _ = _read_obj(self.output_path)
        indices = normalization._match_seam_uvs(before, [uv for _, uv in input_records])
        self.assertEqual(len(output_records), 6)
        for (old_ids, _), (new_ids, actual), uv_indices in zip(input_records, output_records, indices):
            self.assertEqual(new_ids, old_ids)
            np.testing.assert_allclose(actual, after[uv_indices], atol=1e-8)

    def test_refuses_to_overwrite_source_seam(self):
        self.seam_path = self.input_dir / "result_seam.csv"
        self.seam_path.write_text((self.seam_dir / "Apple_seam.csv").read_text())
        original = self.seam_path.read_bytes()
        with self.assertRaisesRegex(ValueError, "must not overwrite"):
            normalize_uv_projection(self.input_path, self.seam_path, self.input_dir / "result.obj")
        self.assertEqual(self.seam_path.read_bytes(), original)

    def test_rejects_mismatched_seam(self):
        with self.seam_path.open(encoding="utf-8", newline="") as seam_file:
            rows = list(csv.reader(seam_file))
        rows[1][0] = "0.75"
        with self.seam_path.open("w", encoding="utf-8", newline="") as seam_file:
            csv.writer(seam_file).writerows(rows)
        with self.assertRaisesRegex(ValueError, "does not match"):
            normalize_uv_projection(self.input_path, self.seam_path, self.output_path)
        self.assertFalse(self.output_path.exists())

    def test_ring_first_vertices_are_on_positive_u_axis_and_texture_follows(self):
        with self.seam_path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream))
        for start, offset in ((0, 1), (3, 2)):
            for row in range(start, start + 3):
                rows[row] = rows[row][offset:] + rows[row][:offset]
        with self.seam_path.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows(rows)

        normalize_uv_projection(self.input_path, self.seam_path, self.output_path)
        _, uvs, _, _ = _read_obj(self.output_path)
        np.testing.assert_allclose(uvs[1], (1.0, 0.1), atol=1e-8)
        np.testing.assert_allclose(uvs[7], (0.2, 0.9), atol=1e-8)
        with Image.open(self.output_path.with_suffix(".png")) as result:
            for destination, source in (
                ((0.95, 0.1), (0.9, 0.14)),
                ((0.15, 0.9), (0.055, 0.9)),
            ):
                x = round(destination[0] * 127)
                y = round((1 - destination[1]) * 127)
                pixel = np.asarray(result.getpixel((x, y)))
                expected = np.array([source[0] * 127 * 2, (1 - source[1]) * 127 * 2])
                np.testing.assert_allclose(pixel[:2], expected, atol=6)
                self.assertEqual(pixel[3], 255)

    def test_main_normalize_uses_voxelized_seams(self):
        with patch.object(sys, "argv", ["main.py", "--operation", "normalize", "--input-dir", str(self.root)]):
            main.main()
        self.assertTrue(self.output_path.is_file())
        self.assertTrue(self.output_path.with_suffix(".png").is_file())

    def test_main_reports_missing_seam(self):
        self.seam_path.unlink()
        with patch.object(sys, "argv", ["main.py", "--operation", "normalize", "--input-dir", str(self.root)]):
            with self.assertRaisesRegex(RuntimeError, "Normalized 0 of 1"):
                main.main()
        self.assertFalse(self.output_path.exists())

    def test_refuses_to_overwrite_source_texture(self):
        conflicting_output = self.input_dir / "source.obj"
        with self.assertRaisesRegex(ValueError, "must not overwrite"):
            normalize_uv_projection(self.input_path, self.seam_path, conflicting_output)
        self.assertFalse(conflicting_output.exists())

    def test_texture_padding_copies_nearest_color_without_changing_chart(self):
        source = self.root / "padding_source.png"
        output = self.root / "padding_output.png"
        image = np.zeros((21, 21, 4), dtype=np.uint8)
        image[:] = (40, 100, 200, 255)
        image[10, 10] = (9, 17, 25, 0)
        Image.fromarray(image).save(source)
        uvs = np.array([[0.5, 0.5], [0.6, 0.5], [0.5, 0.6]])
        normalization._warp_texture(source, output, uvs, uvs, np.array([[0, 1, 2]]))
        with Image.open(output) as padded_image:
            pixels = np.asarray(padded_image)
        np.testing.assert_array_equal(pixels[10, 10], (9, 17, 25, 0))
        np.testing.assert_array_equal(pixels[10, 6], pixels[10, 10])
        np.testing.assert_array_equal(pixels[10, 5], (0, 0, 0, 0))
        np.testing.assert_array_equal(pixels[10, 12], image[10, 12])
        np.testing.assert_array_equal(pixels[10, 16], pixels[10, 12])
        np.testing.assert_array_equal(pixels[10, 17], (0, 0, 0, 0))

    def test_standalone_cli_defaults_and_explicit_arguments(self):
        root = Path("/Users/cindygrimm/PycharmProjects/data/ScannedModels/ProsserEnvyApples")
        with patch.object(normalization, "normalize_uv_projection") as normalize:
            with patch.object(sys, "argv", ["normalize_uv_projection.py"]):
                normalization.main()
            normalize.assert_called_once_with(
                root / "BlenderRebaked/Envy1NoStemHiRes.obj",
                root / "Voxelized/Envy1_seam.csv",
                root / "NormalizedTexture/Envy1NoStemHiRes.obj",
            )
            normalize.reset_mock()
            with patch.object(
                sys, "argv", ["normalize_uv_projection.py", "input.obj", "seam.csv", "output.obj"]
            ):
                normalization.main()
            normalize.assert_called_once_with(
                Path("input.obj"), Path("seam.csv"), Path("output.obj")
            )


if __name__ == "__main__":
    unittest.main()
