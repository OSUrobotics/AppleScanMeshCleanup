import csv
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

import make_apple_geometry as geometry
from normalize_uv_projection import _read_obj


class AppleGeometryUVTests(unittest.TestCase):
    def test_top_faces_are_near_equilateral_and_form_a_closed_mesh(self):
        for samples in (3, 12, 32, 64):
            with self.subTest(samples=samples):
                assembly, top_ring, bottom_ring, _ = geometry._create_apple_body(samples)
                first, last = geometry._create_apple_top(assembly, top_ring, samples)
                vertices = np.asarray(assembly.points)
                faces = np.asarray(assembly.triangles[first:last])
                triangles = vertices[faces]
                lengths = np.linalg.norm(triangles - np.roll(triangles, 1, axis=1), axis=2)
                ratios = lengths.max(axis=1) / lengths.min(axis=1)
                self.assertGreaterEqual(np.mean(ratios < 1.8), 0.9)
                self.assertLess(float(ratios.max()), 2.0)
                self.assertEqual(np.count_nonzero(np.all(vertices == [0, 0.5, 0], axis=1)), 1)
                geometry._create_apple_bottom(assembly, bottom_ring)
                mesh = geometry._to_mesh(assembly)
                self.assertEqual(len(mesh.topology.findHoleRepresentiveEdges()), 0)

    def test_body_latitudes_span_minus_70_to_plus_70(self):
        for samples in (3, 12, 32, 64):
            with self.subTest(samples=samples):
                assembly, top_ring, bottom_ring, side_seam = geometry._create_apple_body(samples)
                latitudes = [
                    math.degrees(math.asin(2 * assembly.points[vertex][1]))
                    for vertex in side_seam
                ]
                self.assertAlmostEqual(latitudes[0], -70)
                self.assertAlmostEqual(latitudes[-1], 70)
                self.assertAlmostEqual(latitudes[1], -70 + 70 / samples)
                np.testing.assert_allclose(
                    latitudes[2:], np.linspace(-70 + 140 / samples, 70, samples), atol=1e-12
                )
                self.assertEqual(len(top_ring), geometry._longitude_count(70, 140 / samples))
                self.assertEqual(len(bottom_ring), geometry._longitude_count(-70, 140 / samples))

    def test_seam_csv_contains_body_uvs_for_each_face_corner(self):
        assembly, top_ring, bottom_ring, side_seam = geometry._create_apple_body(32)
        top_faces = geometry._create_apple_top(assembly, top_ring, 32)
        bottom_faces = geometry._create_apple_bottom(assembly, bottom_ring)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "apple.obj"
            geometry._write_obj_with_uv(
                assembly, top_ring, bottom_ring, side_seam, top_faces, bottom_faces, path
            )
            _, uvs, uv_faces, _ = _read_obj(path)
            with path.with_name("apple_seam.csv").open(newline="") as stream:
                rows = list(csv.reader(stream))

        self.assertEqual(len(rows), 18)
        body_uvs = {}
        for face, face_uv in zip(assembly.triangles[:top_faces[0]], uv_faces[:top_faces[0]]):
            for vertex, uv_index in zip(face, face_uv):
                body_uvs.setdefault(vertex, []).append(uvs[uv_index])
        corners = [(0.5, 0), (1, 0.5), (0.5, 1), (0, 0.5), (0.5, 0)]
        for side in range(4):
            ids = [int(value) for value in rows[6 + side * 3]]
            u = np.array(rows[7 + side * 3], dtype=float)
            v = np.array(rows[8 + side * 3], dtype=float)
            self.assertEqual(len(ids), len(u))
            self.assertEqual(len(ids), len(v))
            coords = np.column_stack((u, v))
            np.testing.assert_allclose(coords[0], corners[side], atol=1e-8)
            np.testing.assert_allclose(coords[-1], corners[side + 1], atol=1e-8)
            for vertex, coord in zip(ids, coords):
                self.assertTrue(
                    any(np.allclose(coord, uv, atol=1e-8) for uv in body_uvs[vertex]),
                    f"Vertex {vertex}, side {side}, UV {coord} missing from OBJ",
                )

    def test_exported_body_top_seam_spans_requested_diamond_edge(self):
        assembly, top_ring, bottom_ring, side_seam = geometry._create_apple_body(32)
        top_faces = geometry._create_apple_top(assembly, top_ring, 32)
        bottom_faces = geometry._create_apple_bottom(assembly, bottom_ring)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "apple.obj"
            geometry._write_obj_with_uv(
                assembly, top_ring, bottom_ring, side_seam, top_faces, bottom_faces, path
            )
            _, uvs, uv_faces, _ = _read_obj(path)

        top_vertices = set(top_ring)
        seam_uvs = np.array([
            uvs[uv_index]
            for face, face_uv in zip(
                assembly.triangles[:top_faces[0]], uv_faces[:top_faces[0]]
            )
            for vertex, uv_index in zip(face, face_uv)
            if vertex in top_vertices
        ])
        self.assertGreater(len(seam_uvs), 0)
        np.testing.assert_allclose(seam_uvs[:, 0] - seam_uvs[:, 1], 0.5, atol=1e-8)
        self.assertTrue(np.any(np.all(np.isclose(seam_uvs, [0.5, 0.0]), axis=1)))
        self.assertTrue(np.any(np.all(np.isclose(seam_uvs, [1.0, 0.5]), axis=1)))


if __name__ == "__main__":
    unittest.main()
