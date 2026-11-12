import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

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
        self.scan = self.root / "scan.obj"
        vertex = 0
        scan_lines = []
        for line in lines:
            if line.startswith("v "):
                scan_lines.append("v " + " ".join(map(str, self.scan_vertices[vertex])))
                vertex += 1
            else:
                scan_lines.append(line)
        self.scan.write_text("\n".join(scan_lines) + "\n", encoding="utf-8")
        self.endpoints = self.root / "endpoints.txt"
        np.savetxt(self.endpoints, self.scan_vertices[self.poles], delimiter=",")
        self.output = self.root / "aligned" / "result.obj"

    def test_similarity_alignment_and_correspondence(self):
        alignment.align_meshes(
            self.scan, self.canonical, self.seams, self.seams, self.endpoints, self.output
        )
        _, fitted, faces, _, _, _ = alignment._mesh(self.output)
        np.testing.assert_allclose(fitted, self.scan_vertices, atol=1e-7)
        self.assertTrue(self.output.with_suffix(".mtl").is_file())
        self.assertTrue(self.output.with_suffix(".png").is_file())
        with (self.output.parent / "result_correspondence.csv").open(newline="") as stream:
            rows = list(csv.reader(stream))
        self.assertEqual(len(rows), len(self.vertices) + 1)
        for vertex, row in enumerate(rows[1:]):
            self.assertEqual(int(row[0]), vertex)
            face = int(row[1])
            self.assertTrue(0 <= face < len(faces))
            weights = np.array(row[2:], dtype=float)
            self.assertAlmostEqual(float(weights.sum()), 1)
            np.testing.assert_allclose(weights @ fitted[faces[face]], self.scan_vertices[vertex], atol=1e-7)

    def test_seam_interpolation_preserves_order(self):
        source = np.array([[0, 0, 0], [0.25, 0, 0], [1, 0, 0]])
        target = np.array([[0, 0, 0], [0.5, 0, 0], [2, 0, 0]])
        ids, positions = alignment._resample_seam([0, 1, 2], [0, 1, 2], source, target, False)
        np.testing.assert_array_equal(ids, [0, 1, 2])
        np.testing.assert_allclose(positions[:, 0], [0, 0.5, 2])
        mapping = alignment._seam_correspondence([0, 1, 2], [0, 1, 2], source, target, False)
        self.assertEqual(mapping[1][:3], (1, 1, 2))
        self.assertEqual(mapping[1][3], 0)

    def test_calyx_peaks_follow_distinct_scan_protrusions(self):
        peaks = np.flatnonzero(np.isclose(self.vertices[:, 1], self.vertices[:, 1].min()))
        self.assertEqual(len(peaks), 5)
        self.scan_vertices[peaks, 1] -= 0.02
        lines = self.scan.read_text(encoding="utf-8").splitlines()
        vertex = 0
        for index, line in enumerate(lines):
            if line.startswith("v "):
                lines[index] = "v " + " ".join(map(str, self.scan_vertices[vertex]))
                vertex += 1
        self.scan.write_text("\n".join(lines) + "\n", encoding="utf-8")
        alignment.align_meshes(
            self.scan, self.canonical, self.seams, self.seams, self.endpoints, self.output
        )
        _, fitted, _, _, _, _ = alignment._mesh(self.output)
        np.testing.assert_allclose(fitted[peaks], self.scan_vertices[peaks], atol=1e-7)
        np.testing.assert_allclose(fitted[self.poles], self.scan_vertices[self.poles], atol=1e-7)

    def test_reversed_seam_rows_match_original_orientation(self):
        with self.seams.open(newline="") as stream:
            rows = list(csv.reader(stream))
        for index in (0, 3, 6, 9, 12, 15):
            rows[index].reverse()
        reversed_seams = self.root / "reversed_seams.csv"
        with reversed_seams.open("w", newline="") as stream:
            csv.writer(stream).writerows(rows)
        alignment.align_meshes(
            self.scan, self.canonical, reversed_seams, self.seams, self.endpoints, self.output
        )
        _, fitted, faces, _, _, _ = alignment._mesh(self.output)
        seam_ids = alignment._seam_ids(self.seams, len(self.vertices))
        for ids in seam_ids:
            np.testing.assert_allclose(fitted[ids], self.scan_vertices[ids], atol=1e-7)
        with (self.output.parent / "result_correspondence.csv").open(newline="") as stream:
            rows = list(csv.reader(stream))[1:]
        for vertex in set().union(*map(set, seam_ids)):
            row = rows[vertex]
            reconstructed = np.array(row[2:], dtype=float) @ fitted[faces[int(row[1])]]
            np.testing.assert_allclose(reconstructed, self.scan_vertices[vertex], atol=1e-7)

    def test_alignment_with_reoriented_uv_charts(self):
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
        alignment.align_meshes(
            self.scan, self.canonical, self.seams, self.seams, self.endpoints, self.output
        )
        _, fitted, _, _, _, _ = alignment._mesh(self.output)
        np.testing.assert_allclose(fitted, self.scan_vertices, atol=1e-7)

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
            alignment.align_meshes(
                self.scan, self.canonical, self.seams, self.seams, self.endpoints, output
            )
        self.assertEqual(self.canonical.read_bytes(), original)

    def test_cli_arguments_are_optional(self):
        with patch.object(alignment, "align_meshes") as run:
            with patch.object(sys, "argv", ["align_meshes.py"]):
                alignment.main()
            self.assertEqual(run.call_count, 1)
            self.assertIn("NormalizedTexture", str(run.call_args.args[0]))


if __name__ == "__main__":
    unittest.main()
