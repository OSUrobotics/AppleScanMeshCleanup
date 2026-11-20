import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

import main
import normalize_uv_projection as normalization
from align_meshes import _seam_ids
from normalize_uv_projection import _read_obj, normalize_uv_projection
from seam_topology import SeamInfo, SeamTopology


class NormalizeUVProjectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.input_dir = self.root / "BlenderRebaked"
        self.seam_dir = self.root / "Parameterized"
        self.input_dir.mkdir()
        self.seam_dir.mkdir()
        self.input_path = self.input_dir / "Apple_001_NoStemHiRes.obj"
        self.seam_path = self.seam_dir / "Apple1_FullRes_seam_topology.json"
        self.output_path = self.root / "NormalizedTexture" / self.input_path.name

        top = [(0.99, 0.1), (0.9, 0.18), (0.81, 0.1), (0.9, 0.02)]
        bottom = [(0.19, 0.9), (0.1, 0.98), (0.01, 0.9), (0.1, 0.82)]
        body = [(0.49, 0.02), (0.98, 0.49), (0.51, 0.98), (0.02, 0.51)]
        calyx = [(0.14, 0.9), (0.1, 0.94), (0.06, 0.9), (0.1, 0.86)]
        charts = [top, bottom, body]
        centers = [(0.9, 0.1), (0.1, 0.9), (0.5, 0.5)]
        with self.input_path.open("w", encoding="utf-8") as obj:
            obj.write("mtllib Apple_001_NoStemHiRes.mtl\n")
            for index in range(19):
                obj.write(f"v {index} 0 0\n")
            for chart, center in zip(charts, centers):
                for u, v in [*chart, center]:
                    obj.write(f"vt {u} {v}\n")
            for u, v in calyx:
                obj.write(f"vt {u} {v}\n")
            obj.write("usemtl apple\n")
            for chart in range(3):
                if chart == 1:
                    continue
                first = 5 * chart + 1
                for index in range(4):
                    a, b, c = first + index, first + (index + 1) % 4, first + 4
                    obj.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")
            for index in range(4):
                a, b = 6 + index, 6 + (index + 1) % 4
                c, d = 16 + index, 16 + (index + 1) % 4
                for face in ((a, b, d), (a, d, c), (c, d, 10)):
                    obj.write("f " + " ".join(f"{vertex}/{vertex}" for vertex in face) + "\n")

        material = self.input_path.with_suffix(".mtl")
        material.write_text("newmtl apple\nKd 1 1 1\nmap_Kd source.png\n", encoding="utf-8")
        x, y = np.meshgrid(np.arange(128), np.arange(128))
        image = np.stack((x * 2, y * 2, np.full_like(x, 100)), axis=2).astype(np.uint8)
        Image.fromarray(image).save(self.input_dir / "source.png")

        topology = SeamTopology(0, 18, (4.1, 0, 0), (9.1, 0, 0))
        topology.set_boundary_paths([
            (list(range(4)), top),
            (list(range(5, 9)), bottom),
            *[([10 + i, 10 + (i + 1) % 4], [body[i], body[(i + 1) % 4]])
              for i in range(4)],
        ])
        topology.add_seam("calyx", SeamInfo(
            "calyx", [15, 16, 17, 18], [[index, 0, 0] for index in range(15, 19)],
            [[point] for point in calyx], (9.1, 0, 0), 1.0,
        ))
        self.seam_path.write_text(json.dumps(topology.to_dict()), encoding="utf-8")

    def _read_topology(self):
        return json.loads(self.seam_path.read_text(encoding="utf-8"))

    def _write_topology(self, data):
        self.seam_path.write_text(json.dumps(data), encoding="utf-8")

    def test_normalizes_boundaries_and_warps_texture(self):
        normalize_uv_projection(self.input_path, self.output_path, self.seam_dir)
        _, uvs, faces, material = _read_obj(self.output_path)
        self.assertEqual(len(faces), 20)
        self.assertTrue(material.is_file())
        self.assertIn("map_Kd Apple_001_NoStemHiRes.png", material.read_text())
        for indices, center in [(range(4), (0.9, 0.1)), (range(5, 9), (0.1, 0.9))]:
            radii = np.linalg.norm(uvs[list(indices)] - center, axis=1)
            np.testing.assert_allclose(radii, 0.1, atol=1e-8)
        np.testing.assert_allclose(uvs[[4, 9]], [(0.9, 0.1), (0.1, 0.9)], atol=1e-8)
        np.testing.assert_allclose(np.linalg.norm(uvs[15:19] - (0.1, 0.9), axis=1), 0.05, atol=1e-8)
        for point in uvs[10:14]:
            self.assertAlmostEqual(abs(point[0] - 0.5) + abs(point[1] - 0.5), 0.5)
        with Image.open(self.output_path.with_suffix(".png")) as result:
            self.assertEqual(result.size, (128, 128))
            self.assertGreater(result.getpixel((64, 64))[3], 0)
            pixel = result.getpixel((64, 64))
            self.assertLess(abs(pixel[0] - 128), 5)
            self.assertLess(abs(pixel[1] - 126), 5)
            self.assertEqual(pixel[2], 100)

    def test_selects_seam_by_apple_name_and_number(self):
        for name in ("Apple10_seam_topology.json", "Other1_seam_topology.json", "Apple_seam_topology.json"):
            (self.seam_dir / name).write_text("unrelated", encoding="utf-8")
        normalize_uv_projection(self.input_path, self.output_path, self.seam_dir)
        self.assertTrue(self.output_path.is_file())

    def test_normalization_does_not_accept_legacy_csv(self):
        self.seam_path.unlink()
        (self.seam_dir / "Apple1_seam.csv").touch()
        with self.assertRaisesRegex(FileNotFoundError, "No seam topology JSON found"):
            normalize_uv_projection(self.input_path, self.output_path, self.seam_dir)
        with self.assertRaisesRegex(ValueError, "Expected a seam topology JSON"):
            normalization._read_seam_data(self.seam_dir / "Apple1_seam.csv")

    def test_rejects_missing_or_ambiguous_seam_files(self):
        self.seam_path.unlink()
        (self.seam_dir / "Apple10_seam_topology.json").touch()
        with self.assertRaisesRegex(FileNotFoundError, "No seam topology JSON found"):
            normalize_uv_projection(self.input_path, self.output_path, self.seam_dir)
        self.seam_path.touch()
        (self.seam_dir / "Apple_001_seam_topology.json").touch()
        with self.assertRaisesRegex(ValueError, "Multiple seam topology JSON files match"):
            normalize_uv_projection(self.input_path, self.output_path, self.seam_dir)
        self.assertFalse(self.output_path.exists())

    def test_rejects_seam_file_instead_of_directory(self):
        with self.assertRaisesRegex(NotADirectoryError, "Seam directory"):
            normalize_uv_projection(self.input_path, self.output_path, self.seam_path)

    def test_normalizes_circular_charts_with_zero_length_boundary_edges(self):
        obj = self.input_path.read_text(encoding="utf-8")
        obj = obj.replace("vt 0.9 0.18\n", "vt 0.99 0.1\n")
        obj = obj.replace("vt 0.1 0.98\n", "vt 0.19 0.9\n")
        self.input_path.write_text(obj, encoding="utf-8")
        data = self._read_topology()
        for index in (0, 1):
            data["boundary_paths"][index]["uv_coords"][1] = data["boundary_paths"][index]["uv_coords"][0]
        self._write_topology(data)

        normalize_uv_projection(self.input_path, self.output_path, self.seam_dir)
        _, uvs, _, _ = _read_obj(self.output_path)
        for first, center in ((0, (0.9, 0.1)), (5, (0.1, 0.9))):
            np.testing.assert_allclose(
                np.linalg.norm(uvs[first:first + 4] - center, axis=1), 0.1, atol=1e-8
            )
            separation = np.linalg.norm(uvs[first] - uvs[first + 1])
            self.assertGreater(separation, 0)
            self.assertLess(separation, 1e-5)
        np.testing.assert_allclose(uvs[[4, 9]], [(0.9, 0.1), (0.1, 0.9)], atol=1e-8)

    def test_warns_and_perturbs_invalid_topology_without_mutating_source_uvs(self):
        _, uvs, faces, _ = _read_obj(self.input_path)
        original_uvs = uvs.copy()
        seams = normalization._read_seams(self.seam_path)
        centers = [np.array([4]), np.array([9])]
        calyx = [np.array([index]) for index in range(15, 19)]
        reference = normalization._parameterize(uvs, faces, seams, centers, calyx)
        for extra_faces in ([[0, 4, 2], [0, 4, 2]], [[4, 4, 4]]):
            with self.subTest(extra_faces=extra_faces):
                invalid_faces = np.vstack((faces, extra_faces))
                with self.assertWarnsRegex(RuntimeWarning, "does not repair the topology"):
                    result = normalization._parameterize(uvs, invalid_faces, seams, centers, calyx)
                self.assertTrue(np.isfinite(result).all())
                np.testing.assert_array_equal(uvs, original_uvs)
                np.testing.assert_allclose(result[[4, 9]], [(0.9, 0.1), (0.1, 0.9)])
                if len(extra_faces) == 2:
                    self.assertGreater(np.max(np.abs(result[:4] - reference[:4])), 0)
                    self.assertLess(np.max(np.abs(result[:4] - reference[:4])), 1e-6)

    def test_normalized_topology_preserves_ids_and_uses_output_uvs(self):
        normalize_uv_projection(self.input_path, self.output_path, self.seam_dir)
        output_seam = self.output_path.with_name(self.output_path.stem + "_seam_topology.json")
        input_records = normalization._read_seam_records(self.seam_path)
        output_records = normalization._read_seam_records(output_seam)
        _, before, _, _ = _read_obj(self.input_path)
        _, after, _, _ = _read_obj(self.output_path)
        indices = normalization._match_seam_uvs(before, [uv for _, uv in input_records])
        self.assertEqual(len(output_records), 6)
        for (old_ids, _), (new_ids, actual), uv_indices in zip(input_records, output_records, indices):
            self.assertEqual(new_ids, old_ids)
            np.testing.assert_allclose(actual, after[uv_indices], atol=1e-8)
        _, input_centers = normalization._read_seam_data(self.seam_path)
        _, output_centers = normalization._read_seam_data(output_seam)
        np.testing.assert_array_equal(output_centers, input_centers)
        self.assertEqual(_seam_ids(output_seam, 19), [ids for ids, _ in input_records])
        self.assertFalse(self.output_path.with_name(self.output_path.stem + "_seam.csv").exists())

    def test_normalizes_topology_json_and_updates_existing_seam_uvs(self):
        records, centers = normalization._read_seam_data(self.seam_path)
        topology = SeamTopology.from_dict(self._read_topology())
        topology.top_shoulder_radius = 2.0
        topology.bottom_shoulder_radius = 3.0
        topology.add_seam("stem", SeamInfo(
            "stem", [0], [[0, 0, 0]], [[records[0][1][0].tolist(), records[2][1][0].tolist()]], (0, 0, 0), 1.0,
        ))
        self._write_topology(topology.to_dict())
        normalize_uv_projection(self.input_path, self.output_path, self.seam_dir)
        output_path = self.output_path.with_name(self.output_path.stem + "_seam_topology.json")
        result = SeamTopology.from_dict(json.loads(output_path.read_text()))
        self.assertEqual((result.top_shoulder_radius, result.bottom_shoulder_radius), (2.0, 3.0))
        _, before, _, _ = _read_obj(self.input_path)
        _, after, _, _ = _read_obj(self.output_path)
        matches = normalization._match_seam_uvs(before, [np.asarray(topology.seams["stem"].uv_coords[0])])[0]
        np.testing.assert_allclose(result.seams["stem"].uv_coords[0], after[matches], atol=1e-8)
        np.testing.assert_array_equal(result.top_point, centers[0])
        for copies in result.seams["calyx"].uv_coords:
            self.assertAlmostEqual(np.linalg.norm(np.asarray(copies[0]) - (0.1, 0.9)), 0.05)

    def test_rejects_missing_or_invalid_calyx(self):
        original = self._read_topology()
        for change, error in (
            ("missing", "must contain a calyx ring"),
            ("duplicate", "distinct vertices"),
            ("unknown_uv", "do not match"),
            ("extra_uv", "do not match"),
            ("center", "distinct interior UV vertices"),
        ):
            with self.subTest(change=change):
                data = json.loads(json.dumps(original))
                calyx = data["seams"]["calyx"]
                if change == "missing":
                    del data["seams"]["calyx"]
                elif change == "duplicate":
                    calyx["vertex_ids"][0] = calyx["vertex_ids"][1]
                elif change == "unknown_uv":
                    calyx["uv_coords"][0] = [[0.4, 0.4]]
                elif change == "extra_uv":
                    calyx["uv_coords"][0].append([0.9, 0.1])
                else:
                    calyx["vertex_ids"][0] = 9
                    calyx["uv_coords"][0] = [[0.1, 0.9]]
                self._write_topology(data)
                with self.assertRaisesRegex(ValueError, error):
                    normalize_uv_projection(self.input_path, self.output_path, self.seam_dir)
                self.assertFalse(self.output_path.exists())
                self.assertFalse(self.output_path.with_suffix(".png").exists())

    def test_refuses_to_overwrite_source_topology(self):
        source = self.seam_path
        original = source.read_text(encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "must not overwrite"):
            normalize_uv_projection(self.input_path, self.seam_dir / "Apple1_FullRes.obj", self.seam_dir)
        self.assertEqual(source.read_text(encoding="utf-8"), original)

    def test_pins_nearest_3d_vertex_and_solves_remaining_interior(self):
        obj = self.input_path.read_text(encoding="utf-8")
        obj = obj.replace("v 18 0 0\n", "v 18 0 0\nv 19 0 0\n")
        obj = obj.replace("vt 0.9 0.18\n", "vt 0.97 0.18\n")
        obj = obj.replace("usemtl apple\n", "vt 0.945 0.13\nusemtl apple\n")
        obj = obj.replace(
            "f 1/1 2/2 5/5\n",
            "f 1/1 2/2 20/20\nf 2/2 5/5 20/20\nf 5/5 1/1 20/20\n",
        )
        self.input_path.write_text(obj, encoding="utf-8")
        data = self._read_topology()
        data["boundary_paths"][0]["uv_coords"][1][0] = 0.97
        data["top_point"] = [19.05, 0, 0]
        self._write_topology(data)

        normalize_uv_projection(self.input_path, self.output_path, self.seam_dir)
        _, uvs, _, _ = _read_obj(self.output_path)
        np.testing.assert_allclose(uvs[19], (0.9, 0.1), atol=1e-8)
        np.testing.assert_allclose(uvs[9], (0.1, 0.9), atol=1e-8)
        np.testing.assert_allclose(uvs[4], np.mean(uvs[[0, 1, 2, 3, 19]], axis=0), atol=1e-8)
        self.assertGreater(np.linalg.norm(uvs[4] - uvs[19]), 0.001)

    def test_center_vertex_mapping_uses_obj_face_indices(self):
        lines = [
            "v 0 0 0", "v 1 0 0", "v 2 0 0",
            "f -3/3 -2/1 -1/2",
        ]
        matches = normalization._center_uv_indices(
            lines, np.array([[2, 0, 1]]), np.array([[0.1, 0, 0], [1.9, 0, 0]]),
            self.input_path,
        )
        np.testing.assert_array_equal(matches[0], [2])
        np.testing.assert_array_equal(matches[1], [1])

    def test_rejects_missing_or_invalid_centers(self):
        original = self._read_topology()
        for top_point, error in (
            ([float("nan"), 0, 0], "finite 3D center"),
            ([1, 2], "finite 3D center"),
            ([0, 0, 0], "not on its boundary"),
            ([14, 0, 0], "exactly one UV coordinate"),
        ):
            with self.subTest(top_point=top_point):
                data = original.copy()
                data["top_point"] = top_point
                self._write_topology(data)
                with self.assertRaisesRegex(ValueError, error):
                    normalize_uv_projection(self.input_path, self.output_path, self.seam_dir)
                self.assertFalse(self.output_path.exists())

    def test_preserves_source_topology(self):
        original = self.seam_path.read_bytes()
        normalize_uv_projection(self.input_path, self.input_dir / "Apple_001.obj", self.seam_dir)
        self.assertEqual(self.seam_path.read_bytes(), original)
        self.assertTrue((self.input_dir / "Apple_001_seam_topology.json").is_file())

    def test_rejects_mismatched_seam(self):
        data = self._read_topology()
        data["boundary_paths"][0]["uv_coords"][0][0] = 0.75
        self._write_topology(data)
        with self.assertRaisesRegex(ValueError, "does not match"):
            normalize_uv_projection(self.input_path, self.output_path, self.seam_dir)
        self.assertFalse(self.output_path.exists())

    def test_ring_first_vertices_are_on_positive_u_axis_and_texture_follows(self):
        data = self._read_topology()
        for index, offset in ((0, 1), (1, 2)):
            for key in ("vertex_ids", "uv_coords"):
                row = data["boundary_paths"][index][key]
                data["boundary_paths"][index][key] = row[offset:] + row[:offset]
        self._write_topology(data)

        normalize_uv_projection(self.input_path, self.output_path, self.seam_dir)
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

    def test_main_normalize_uses_parameterized_seams(self):
        with patch.object(sys, "argv", ["main.py", "--operation", "normalize_uv_projection", "--root-dir", str(self.root)]):
            main.main()
        self.assertTrue(self.output_path.is_file())
        self.assertTrue(self.output_path.with_suffix(".png").is_file())

    def test_main_reports_missing_seam(self):
        self.seam_path.unlink()
        with patch.object(sys, "argv", ["main.py", "--operation", "normalize_uv_projection", "--root-dir", str(self.root)]):
            with patch("sys.stdout", new_callable=io.StringIO) as output:
                main.main()
        self.assertIn("No seam topology JSON found", output.getvalue())
        self.assertFalse(self.output_path.exists())

    def test_refuses_to_overwrite_source_texture(self):
        conflicting_output = self.input_dir / "source.obj"
        with self.assertRaisesRegex(ValueError, "must not overwrite"):
            normalize_uv_projection(self.input_path, conflicting_output, self.seam_dir)
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
                root / "BlenderRebaked/Envy_002_FullRes.obj",
                root / "NormalizedTexture/Envy_002_FullRes_seam.obj",
                root / "Parameterized",
            )
            normalize.reset_mock()
            with patch.object(
                sys, "argv", ["normalize_uv_projection.py", "input.obj", "output.obj", "seam_dir"]
            ):
                normalization.main()
            normalize.assert_called_once_with(
                Path("input.obj"), Path("output.obj"), Path("seam_dir")
            )


if __name__ == "__main__":
    unittest.main()
