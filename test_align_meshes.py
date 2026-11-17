import csv
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
from scipy.sparse import coo_matrix

import align_meshes as alignment
from make_apple_geometry import _create_apple


class AlignMeshesTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.canonical = self.root / "canonical.obj"
        _create_apple(12, self.canonical)
        self.seams = self.root / "canonical_seam.csv"
        lines, self.vertices, self.faces, uv, uv_faces, _ = alignment._mesh(self.canonical)
        vertex_uvs = alignment._vertex_uvs(self.vertices, self.faces, uv, uv_faces)
        self.poles = [
            min((np.linalg.norm(coord - center), vertex)
                for vertex, mappings in enumerate(vertex_uvs)
                for candidate_chart, coord in mappings if candidate_chart == chart)[1]
            for chart, center in ((0, np.array([0.9, 0.1])), (1, np.array([0.1, 0.9])))
        ]
        rotation = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
        self.scan_vertices = 1.4 * self.vertices @ rotation + [2, 3, 4]
        scan_dir = self.root / "NormalizedTexture"
        scan_dir.mkdir()
        voxel_dir = self.root / "Voxelized"
        voxel_dir.mkdir()
        self.scan = scan_dir / "scan.obj"
        shutil.copyfile(self.canonical.with_suffix(".mtl"), scan_dir / "canonical.mtl")
        shutil.copyfile(self.canonical.with_suffix(".png"), scan_dir / "canonical.png")
        vertex = 0
        scan_lines = []
        for line in lines:
            if line.startswith("v "):
                scan_lines.append("v " + " ".join(map(str, self.scan_vertices[vertex])))
                vertex += 1
            else:
                scan_lines.append(line)
        self.scan.write_text("\n".join(scan_lines) + "\n", encoding="utf-8")
        self.scan_seams = voxel_dir / "scan_seam.csv"
        self.scan_seams.write_bytes(self.seams.read_bytes())
        with self.scan_seams.open("a", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows(self.scan_vertices[self.poles])
        self.output = self.root / "aligned" / "result.obj"

    def test_similarity_alignment_and_correspondence(self):
        Image.new("RGB", (128, 128), (31, 127, 219)).save(self.scan.parent / "canonical.png")
        alignment.align_meshes(self.scan, self.output, self.canonical)
        _, fitted, faces, _, _, _ = alignment._mesh(self.output)
        displacement = np.linalg.norm(fitted - self.scan_vertices, axis=1)
        self.assertGreater(displacement.max(), 1e-4)
        self.assertLess(displacement.max(), 0.09)
        np.testing.assert_allclose(fitted[self.poles], self.scan_vertices[self.poles], atol=1e-7)
        self.assertTrue(self.output.with_suffix(".mtl").is_file())
        self.assertTrue(self.output.with_suffix(".png").is_file())
        self.assertIn("map_Kd result.png", self.output.with_suffix(".mtl").read_text())
        with Image.open(self.output.with_suffix(".png")) as texture:
            self.assertEqual(texture.size, (128, 128))
            pixels = np.asarray(texture.convert("RGBA"))
            self.assertGreater(np.count_nonzero(pixels[:, :, 3]), 100)
            np.testing.assert_array_equal(pixels[pixels[:, :, 3] > 0][0], [31, 127, 219, 255])
        with (self.output.parent / "result_correspondence.csv").open(newline="") as stream:
            rows = list(csv.reader(stream))
        self.assertEqual(len(rows), len(self.vertices) + 1)
        for vertex, row in enumerate(rows[1:]):
            self.assertEqual(int(row[0]), vertex)
            face = int(row[1])
            self.assertTrue(0 <= face < len(faces))
            weights = np.array(row[2:], dtype=float)
            self.assertAlmostEqual(float(weights.sum()), 1)
            self.assertLess(np.linalg.norm(weights @ fitted[faces[face]] - self.scan_vertices[vertex]), 0.09)

    def test_texture_resampling_uses_correspondences_and_chart_specific_seam_uvs(self):
        size = 256
        y, x = np.indices((size, size))
        pixels = np.stack((x, y, np.full_like(x, 40), np.full_like(x, 255)), axis=2).astype(np.uint8)
        source = self.root / "source.png"
        result = self.root / "resampled.png"
        Image.fromarray(pixels).save(source)
        faces = np.array([[0, 1, 2], [0, 3, 4]])
        canonical_uv = np.array([
            [.9, .1], [.98, .1], [.9, .18],
            [.2, .2], [.6, .2], [.2, .6],
        ])
        scan_uv = np.array([
            [.9, .1], [.98, .1], [.9, .18],
            [.25, .25], [.65, .25], [.25, .65],
        ])
        uv_faces = np.array([[0, 1, 2], [3, 4, 5]])
        correspondence = [
            (0, 0, 1., 0., 0.), (1, 0, 0., 1., 0.), (2, 0, 0., 0., 1.),
            (3, 1, 0., 1., 0.), (4, 1, 0., 0., 1.),
        ]
        transforms = [(np.array(center), np.eye(2)) for center in
                      ((.9, .1), (.1, .9), (.5, .5))]
        alignment._resample_texture(
            source, result, faces, scan_uv, uv_faces,
            faces, canonical_uv, uv_faces, correspondence, transforms,
        )
        with Image.open(result) as image:
            actual = np.asarray(image.convert("RGBA"))
        np.testing.assert_allclose(actual[178, 76], [89, 165, 40, 255], atol=1)
        np.testing.assert_allclose(actual[229, 232], [232, 229, 40, 255], atol=1)
        correspondence[2] = (2, 0, 0., 1., 0.)
        with self.assertWarnsRegex(UserWarning, "skipped 1 degenerate mapped triangles"):
            alignment._resample_texture(
                source, result, faces, scan_uv, uv_faces,
                faces, canonical_uv, uv_faces, correspondence, transforms,
            )

    def test_texture_resampling_uses_face_uv_side_of_duplicated_body_seam(self):
        source = self.root / "source.png"
        Image.new("RGB", (128, 128), (12, 45, 98)).save(source)
        template_faces = np.array([[0, 1, 2], [0, 1, 3]])
        template_uv = np.array([
            [.5, 0], [.45, .05], [.48, .06],
            [0, .5], [.05, .55], [.06, .48],
        ])
        template_uv_faces = np.array([[0, 1, 2], [3, 4, 5]])
        scan_faces = np.array([[0, 1, 2], [0, 1, 3]])
        scan_uv = template_uv.copy()
        scan_uv_faces = template_uv_faces.copy()
        correspondence = [
            (0, 0, 1., 0., 0.), (1, 0, 0., 1., 0.),
            (2, 1, 0., .1, .9), (3, 1, 0., 0., 1.),
        ]
        transforms = [(np.array(center), np.eye(2)) for center in
                      ((.9, .1), (.1, .9), (.5, .5))]
        with patch.object(alignment, "_warp_texture") as warp, self.assertWarnsRegex(
            UserWarning, "face-corner UV lookup"
        ):
            alignment._resample_texture(
                source, self.root / "result.png", scan_faces, scan_uv, scan_uv_faces,
                template_faces, template_uv, template_uv_faces, correspondence, transforms,
            )
        mapped = warp.call_args.args[3].reshape(-1, 3, 2)
        np.testing.assert_array_equal(warp.call_args.args[2].reshape(-1, 3, 2), scan_uv[scan_uv_faces])
        self.assertLess(np.max(np.linalg.norm(mapped - np.roll(mapped, 1, axis=1), axis=2)), 0.2)
        np.testing.assert_allclose(mapped[0, 0], template_uv[0])
        np.testing.assert_allclose(mapped[1, 0], template_uv[3])

    def test_seam_interpolation_preserves_order(self):
        source = np.array([[0, 0, 0], [0.25, 0, 0], [1, 0, 0]])
        target = np.array([[0, 0, 0], [0.5, 0, 0], [2, 0, 0]])
        ids, positions = alignment._resample_seam([0, 1, 2], [0, 1, 2], source, target, False)
        np.testing.assert_array_equal(ids, [0, 1, 2])
        np.testing.assert_allclose(positions[:, 0], [0, 0.5, 2])
        mapping = alignment._seam_correspondence([0, 1, 2], [0, 1, 2], source, target, False)
        self.assertEqual(mapping[1][:3], (1, 1, 2))
        self.assertEqual(mapping[1][3], 0)

    def test_alignment_does_not_require_calyx_peaks(self):
        seams = alignment._seam_ids(self.seams, len(self.vertices))
        bottom_height = float(np.mean(self.vertices[seams[1], 1]))
        vertices = self.vertices.copy()
        vertices[vertices[:, 1] < bottom_height, 1] = bottom_height
        lines = self.canonical.read_text(encoding="utf-8").splitlines()
        vertex = 0
        for index, line in enumerate(lines):
            if line.startswith("v "):
                lines[index] = "v " + " ".join(map(str, vertices[vertex]))
                vertex += 1
        self.canonical.write_text("\n".join(lines) + "\n", encoding="utf-8")

        with patch.object(alignment, "spsolve", wraps=alignment.spsolve) as solve:
            alignment.align_meshes(self.scan, self.output, self.canonical)
        fixed = set().union(*map(set, seams), set(self.poles))
        self.assertEqual(solve.call_args.args[0].shape, (len(vertices) - len(self.poles),) * 2)
        _, fitted, faces, _, _, _ = alignment._mesh(self.output)
        with (self.output.parent / "result_correspondence.csv").open(newline="") as stream:
            rows = list(csv.reader(stream))[1:]
        for vertex in set().union(*map(set, seams)) - set(self.poles):
            row = rows[vertex]
            weights = np.array(row[2:], dtype=float)
            self.assertLessEqual(np.count_nonzero(weights > 1e-10), 2)
            self.assertTrue(set(faces[int(row[1])][weights > 1e-10]) <= fixed)

    def test_refinement_alternates_averaging_and_projection_for_all_vertices(self):
        x = np.arange(12, dtype=float)
        fitted = np.column_stack((x, np.where(x % 2, 1., 0.), np.ones(len(x))))
        starts = np.arange(len(x) - 1)
        ends = starts + 1
        graph = coo_matrix(
            (np.ones(2 * len(starts)), (np.r_[starts, ends], np.r_[ends, starts])),
            shape=(len(x), len(x)),
        ).tocsr()
        scan = np.array([[-10., -10, 0], [30, -10, 0], [30, 10, 0], [-10, 10, 0]])
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        degree = np.asarray(graph.sum(axis=1)).ravel()
        smoothed = fitted.copy()
        for _ in range(6):
            smoothed = graph @ smoothed / degree[:, None]
            smoothed[:, 2] = 0
        locator = alignment._SurfaceLocator(scan, faces, np.zeros(len(faces), dtype=int))
        refined = alignment._refine_fit(fitted, graph, scan, faces, locator, np.zeros(len(x), dtype=int), set())
        np.testing.assert_allclose(refined, smoothed, atol=1e-12)
        self.assertTrue(np.all(np.linalg.norm(refined - fitted, axis=1) > 0))

    def test_refinement_projects_before_next_average(self):
        fitted = np.array([[-2., 0, 0], [1, 0, 2], [2, 0, 0]])
        graph = coo_matrix(
            (np.ones(6), ([0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1])), shape=(3, 3)
        ).tocsr()
        points = np.array([[-4., -4, 4], [-4, 4, 4], [0, -4, 0], [0, 4, 0],
                           [4, -4, 4], [4, 4, 4]])
        faces = np.array([[0, 2, 3], [0, 3, 1], [2, 4, 5], [2, 5, 3]])
        degree = np.asarray(graph.sum(axis=1)).ravel()
        locator = alignment._SurfaceLocator(points, faces, np.zeros(len(faces), dtype=int))
        smoothed = fitted.copy()
        for _ in range(6):
            smoothed = graph @ smoothed / degree[:, None]
            for vertex in range(len(smoothed)):
                face, weights = locator.locate(smoothed[vertex], 0, 0)
                smoothed[vertex] = weights @ points[faces[face]]
        late_projection = fitted.copy()
        for _ in range(6):
            late_projection = graph @ late_projection / degree[:, None]
        for vertex in range(len(late_projection)):
            face, weights = locator.locate(late_projection[vertex], 0, 0)
            late_projection[vertex] = weights @ points[faces[face]]
        refined = alignment._refine_fit(fitted, graph, points, faces, locator, np.zeros(len(fitted), dtype=int), set())
        np.testing.assert_allclose(refined, smoothed, atol=1e-12)
        self.assertGreater(np.linalg.norm(refined - late_projection), 1e-3)

    def test_midpoint_clips_faces_and_refinement_keeps_poles_fixed(self):
        points = np.array([[0., -2, 0], [1, 2, 0], [-1, 2, 0]])
        faces = np.array([[0, 1, 2]])
        axis = np.array([0., 1, 0])
        surface_points, surface_faces, locator = alignment._half_surface(points, faces, axis)
        for half in (0, 1):
            heights = surface_points[surface_faces[locator.indices[half]], 1]
            self.assertTrue(np.all(heights <= 1e-12) if half == 0 else np.all(heights >= -1e-12))
        rotated = points[:, [1, 0, 2]]
        rotated_points, rotated_faces, rotated_locator = alignment._half_surface(
            rotated, faces, np.array([1., 0, 0])
        )
        for half in (0, 1):
            heights = rotated_points[rotated_faces[rotated_locator.indices[half]], 0]
            self.assertTrue(np.all(heights <= 1e-12) if half == 0 else np.all(heights >= -1e-12))
        graph = coo_matrix(
            (np.ones(6), ([0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1])), shape=(3, 3)
        ).tocsr()
        fitted = np.array([[0., -2, 0], [0, -0.5, 0], [1, 2, 0]])
        refined = alignment._refine_fit(
            fitted, graph, surface_points, surface_faces, locator, np.array([0, 0, 1]), {0, 2}
        )
        np.testing.assert_array_equal(refined[[0, 2]], fitted[[0, 2]])
        self.assertLessEqual(refined[1, 1], 1e-12)
        self.assertGreater(refined[1, 1], fitted[1, 1])

    def test_irregular_scan_seams_are_reprojected_in_output(self):
        seams = alignment._oriented_seams(
            alignment._seam_ids(self.scan_seams, len(self.vertices)), self.scan_vertices, self.faces
        )
        junctions = set(seams[3]) & (set(seams[0]) | set(seams[1]))
        for index in (0, 1, 3):
            for i, vertex in enumerate(seams[index]):
                if vertex not in junctions:
                    self.scan_vertices[vertex] += [0.008 * (-1) ** i, 0.01 * np.sin(i), 0]
        lines = self.scan.read_text(encoding="utf-8").splitlines()
        vertex = 0
        for i, line in enumerate(lines):
            if line.startswith("v "):
                lines[i] = "v " + " ".join(map(str, self.scan_vertices[vertex]))
                vertex += 1
        self.scan.write_text("\n".join(lines) + "\n", encoding="utf-8")
        alignment.align_meshes(self.scan, self.output, self.canonical)
        _, fitted, _, _, _, _ = alignment._mesh(self.output)
        canonical_seams = alignment._oriented_seams(
            alignment._seam_ids(self.seams, len(self.vertices)), self.vertices, self.faces
        )
        expected = {}
        for index in range(6):
            ids, positions = alignment._resample_seam(
                canonical_seams[index], seams[index], self.vertices, self.scan_vertices,
                index in (0, 1, 2, 4),
            )
            for vertex, position in zip(ids, positions):
                expected.setdefault(int(vertex), position)
        locator = alignment._SurfaceLocator(
            self.scan_vertices, self.faces, np.zeros(len(self.faces), dtype=int)
        )
        self.assertGreater(max(np.linalg.norm(fitted[v] - p) for v, p in expected.items()), 1e-5)
        for vertex in expected:
            face, weights = locator.locate(fitted[vertex], 0, 0)
            np.testing.assert_allclose(
                fitted[vertex], weights @ self.scan_vertices[self.faces[face]], atol=1e-7
            )

    def test_reversed_seam_rows_match_original_orientation(self):
        alignment.align_meshes(self.scan, self.output, self.canonical)
        _, expected, _, _, _, _ = alignment._mesh(self.output)
        correspondence = (self.output.parent / "result_correspondence.csv").read_text()
        with self.scan_seams.open(newline="") as stream:
            rows = list(csv.reader(stream))
        for index in (0, 3, 6, 9, 12, 15):
            rows[index].reverse()
        with self.scan_seams.open("w", newline="") as stream:
            csv.writer(stream).writerows(rows)
        alignment.align_meshes(self.scan, self.output, self.canonical)
        _, fitted, _, _, _, _ = alignment._mesh(self.output)
        np.testing.assert_allclose(fitted, expected, atol=1e-7)
        self.assertEqual((self.output.parent / "result_correspondence.csv").read_text(), correspondence)

    def test_alignment_with_reoriented_uv_charts(self):
        alignment.align_meshes(self.scan, self.output, self.canonical)
        _, expected, _, _, _, _ = alignment._mesh(self.output)
        lines, _, _, uv, uv_faces, _ = alignment._mesh(self.scan)
        transforms = [
            (np.array([0.9, 0.1]), np.array([[0, -1], [1, 0]])),
            (np.array([0.1, 0.9]), np.array([[1, 0], [0, -1]])),
            (np.array([0.5, 0.5]), np.array([[0, -1], [-1, 0]])),
        ]
        transformed = uv.copy()
        for face in uv_faces:
            chart = alignment._chart(uv[face].mean(axis=0))
            center, matrix = transforms[chart]
            transformed[face] = (uv[face] - center) @ matrix.T + center
        index = 0
        for row, line in enumerate(lines):
            if line.startswith("vt "):
                lines[row] = "vt " + " ".join(map(str, transformed[index]))
                index += 1
        self.scan.write_text("\n".join(lines) + "\n", encoding="utf-8")
        alignment.align_meshes(self.scan, self.output, self.canonical)
        _, fitted, _, _, _, _ = alignment._mesh(self.output)
        np.testing.assert_allclose(fitted, expected, atol=1e-7)

    def test_rejects_inconsistent_repeated_seams(self):
        seams = alignment._seam_ids(self.seams, len(self.vertices))
        seams[2] = seams[2][:-2]
        with self.assertRaisesRegex(ValueError, "repeated seam rows"):
            alignment._oriented_seams(seams, self.vertices, self.faces)

    def test_reads_legacy_ten_row_canonical_seams(self):
        with self.seams.open(newline="") as stream:
            rows = list(csv.reader(stream))
        old_path = self.root / "legacy_seams.csv"
        with old_path.open("w", newline="") as stream:
            csv.writer(stream).writerows(rows[:6] + [rows[index] for index in (6, 9, 12, 15)])
        self.assertEqual(
            alignment._seam_ids(old_path, len(self.vertices)),
            alignment._seam_ids(self.seams, len(self.vertices)),
        )

    def test_independent_ring_reversals_and_changed_start(self):
        seams = alignment._seam_ids(self.seams, len(self.vertices))
        expected = alignment._oriented_seams(seams, self.vertices, self.faces)
        changed = [list(row) for row in seams]
        for index in (0, 1):
            ring = list(dict.fromkeys(changed[index]))
            ring = list(reversed(ring))
            changed[index] = ring[2:] + ring[:2]
        changed[3].reverse()
        actual = alignment._oriented_seams(changed, self.vertices, self.faces)
        self.assertEqual(actual, expected)

    def test_rejects_output_overwriting_source_texture(self):
        output = self.root / "canonical.obj"
        original = self.canonical.read_bytes()
        with self.assertRaisesRegex(ValueError, "must not overwrite"):
            alignment.align_meshes(self.scan, output, self.canonical)
        self.assertEqual(self.canonical.read_bytes(), original)

    def test_cli_arguments_are_optional(self):
        with patch.object(alignment, "align_meshes") as run:
            with patch.object(sys, "argv", ["align_meshes.py"]):
                alignment.main()
            self.assertEqual(run.call_count, 1)
            self.assertIn("NormalizedTexture", str(run.call_args.args[0]))

    def test_cli_passes_three_paths_in_api_order(self):
        argv = ["align_meshes.py", "scan.obj", "canonical.obj", "output.obj"]
        with patch.object(alignment, "align_meshes") as run, patch.object(sys, "argv", argv):
            alignment.main()
        run.assert_called_once_with(Path("scan.obj"), Path("output.obj"), Path("canonical.obj"))

    def test_missing_or_invalid_scan_endpoints_fail(self):
        with self.scan_seams.open(newline="") as stream:
            rows = list(csv.reader(stream))
        invalid_cases = [
            ([], "Expected top and bottom XYZ"),
            ([[1, 2], [3, 4, 5]], "finite 3D center"),
            ([["nan", 0, 0], [1, 2, 3]], "finite 3D center"),
            ([[1, 2, 3], [1, 2, 3]], "endpoints must differ"),
        ]
        for endpoint_rows, message in invalid_cases:
            with self.subTest(endpoint_rows=endpoint_rows):
                with self.scan_seams.open("w", newline="") as stream:
                    csv.writer(stream).writerows(rows[:18] + endpoint_rows)
                with self.assertRaisesRegex(ValueError, message):
                    alignment.align_meshes(self.scan, self.output, self.canonical)
                self.assertFalse(self.output.exists())

    def test_rejects_unnormalized_cap_seam(self):
        seams = alignment._oriented_seams(
            alignment._seam_ids(self.scan_seams, len(self.vertices)), self.scan_vertices, self.faces
        )
        _, _, _, uv, uv_faces, _ = alignment._mesh(self.scan)
        mappings = alignment._vertex_uvs(self.scan_vertices, self.faces, uv, uv_faces)
        vertex = seams[0][1]
        mappings[vertex] = [
            (chart, coord + np.array([0.01, 0])) if chart == 0 else (chart, coord)
            for chart, coord in mappings[vertex]
        ]
        with self.assertRaisesRegex(ValueError, "Run UV normalization"):
            alignment._validate_normalized_seams(seams, mappings, self.scan)

    def test_rejects_missing_cap_seam_vertex(self):
        seams = alignment._oriented_seams(
            alignment._seam_ids(self.scan_seams, len(self.vertices)), self.scan_vertices, self.faces
        )
        _, _, _, uv, uv_faces, _ = alignment._mesh(self.scan)
        mappings = alignment._vertex_uvs(self.scan_vertices, self.faces, uv, uv_faces)
        vertex = seams[0][1]
        mappings[vertex] = [(chart, coord) for chart, coord in mappings[vertex] if chart != 0]
        with self.assertRaisesRegex(ValueError, f"vertex {vertex}.*chart 0"):
            alignment._validate_normalized_seams(seams, mappings, self.scan)

    def test_batch_alignment_uses_normalized_mesh_and_default_canonical(self):
        import main

        argv = ["main.py", "--root-dir", str(self.root / "apples"), "--operation", "align_meshes"]
        with patch.object(sys, "argv", argv), patch.object(main, "process_meshes") as process:
            main.main()
        args = process.call_args.args
        self.assertEqual(args[0], self.root / "apples" / "NormalizedTexture")
        self.assertEqual(args[1], self.root / "apples" / "AlignedMeshes")
        self.assertEqual(args[2].keywords["canonical_path"], self.root / "Canonical" / "apple_fullres.obj")


if __name__ == "__main__":
    unittest.main()
