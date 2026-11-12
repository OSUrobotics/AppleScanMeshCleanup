import csv
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

import align_meshes
import make_apple_geometry as geometry
import normalize_uv_projection as normalization
from normalize_uv_projection import _read_obj


_ENVY_NORMALIZED = Path(
    "/Users/cindygrimm/PycharmProjects/data/ScannedModels/"
    "ProsserEnvyApples/NormalizedTexture"
)


class AppleGeometryUVTests(unittest.TestCase):
    def test_calyx_is_refined_twice_without_splitting_seam(self):
        baseline_faces = {3: 225, 12: 303, 32: 537, 64: 901}
        for samples, original_faces in baseline_faces.items():
            with self.subTest(samples=samples):
                assembly, top, bottom, _ = geometry._create_apple_body(samples)
                geometry._create_apple_top(assembly, top, samples)
                body_faces = list(assembly.triangles)
                first, last = geometry._create_apple_bottom(assembly, bottom)
                self.assertEqual(assembly.triangles[:first], body_faces)
                self.assertEqual(last - first, 16 * original_faces - 5 * len(bottom))
                edges = {}
                for face in assembly.triangles:
                    for i in range(3):
                        edge = tuple(sorted((face[i], face[(i + 1) % 3])))
                        edges[edge] = edges.get(edge, 0) + 1
                self.assertTrue(all(count == 2 for count in edges.values()))
                for i, vertex in enumerate(bottom):
                    edge = tuple(sorted((vertex, bottom[(i + 1) % len(bottom)])))
                    self.assertEqual(edges[edge], 2)
                vertices = np.asarray(assembly.points)
                bottom_y = vertices[bottom[0], 1]
                peaks = vertices[np.isclose(vertices[:, 1], bottom_y - 0.04, atol=1e-10)]
                self.assertEqual(len(peaks), 5)

    @unittest.skipUnless(_ENVY_NORMALIZED.is_dir(), "Envy1 sample data is not available")
    def test_envy_and_canonical_seam_uv_placement_and_winding(self):
        source = _ENVY_NORMALIZED.parent / "BlenderRebaked" / "Envy1NoStemHiRes.obj"
        source_seams = _ENVY_NORMALIZED / "Envy1_seam.csv"
        normalized = _ENVY_NORMALIZED / "Envy1NoStemHiRes.obj"
        _, original_uvs, original_faces, _ = _read_obj(source)
        _, normalized_uvs, normalized_faces, _ = _read_obj(normalized)
        np.testing.assert_array_equal(original_faces, normalized_faces)
        records = normalization._read_seam_records(source_seams)
        matches = normalization._match_seam_uvs(original_uvs, [uv for _, uv in records])
        assembly, top, bottom, side = geometry._create_apple_body(32)
        with tempfile.TemporaryDirectory() as directory:
            canonical_path = Path(directory) / "canonical_seam.csv"
            scan_path = Path(directory) / "normalized_seam.csv"
            geometry._write_seam_csv(
                canonical_path, assembly, top, bottom, side,
                math.hypot(assembly.points[top[0]][0], assembly.points[top[0]][2]),
                math.hypot(assembly.points[bottom[0]][0], assembly.points[bottom[0]][2]),
            )
            normalization._write_seam_csv(scan_path, records, matches, normalized_uvs)
            canonical = normalization._read_seams(canonical_path)
            scan = normalization._read_seams(scan_path)

        for index, center in enumerate(((0.9, 0.1), (0.1, 0.9))):
            with self.subTest(seam=("top", "bottom")[index]):
                for coords in (canonical[index], scan[index]):
                    np.testing.assert_allclose(coords[0], np.array(center) + [0.1, 0], atol=1e-6)
                    np.testing.assert_allclose(np.linalg.norm(coords - center, axis=1), 0.1, atol=1e-6)
                def winding(coords):
                    radial = coords - center
                    return np.sum(radial[:, 0] * np.roll(radial[:, 1], -1)
                                  - radial[:, 1] * np.roll(radial[:, 0], -1))
                self.assertGreater(winding(canonical[index]) * winding(scan[index]), 0,
                                   "Circular seam UV directions differ")

        for index in range(2, 6):
            with self.subTest(body_side=index - 1):
                expected, actual = canonical[index], scan[index]
                start, end = expected[0], expected[-1]
                direction = end - start
                fraction = (actual - start) @ direction / (direction @ direction)
                projection = start + fraction[:, None] * direction
                self.assertLessEqual(
                    float(np.max(np.linalg.norm(actual - projection, axis=1))), 1e-6,
                    "Body seam is on a different diamond side",
                )
                self.assertGreaterEqual(fraction.min(), -1e-6)
                self.assertLessEqual(fraction.max(), 1 + 1e-6)
                self.assertLess(fraction[0], 0.02)
                self.assertGreater(fraction[-1], 0.98)
                self.assertTrue(np.all(np.diff(fraction) >= -1e-6), "Body seam direction differs")

    @unittest.skipUnless(_ENVY_NORMALIZED.is_dir(), "Envy1 sample data is not available")
    def test_envy_top_and_bottom_seams_have_same_3d_winding(self):
        mesh_path = _ENVY_NORMALIZED / "Envy1NoStemHiRes.obj"
        seam_path = _ENVY_NORMALIZED / "Envy1_seam.csv"
        _, vertices, faces, _, _, _ = align_meshes._mesh(mesh_path)
        top, bottom, *_ = align_meshes._seam_ids(seam_path, len(vertices))
        edge_array = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
        edge_array.sort(axis=1)
        edges = {tuple(edge) for edge in edge_array}

        centers = []
        areas = []
        for name, ids in (("top", top), ("bottom", bottom)):
            if ids[0] == ids[-1]:
                ids = ids[:-1]
            self.assertGreaterEqual(len(ids), 3, f"{name} seam needs at least three vertices")
            self.assertEqual(len(ids), len(set(ids)), f"{name} seam repeats a vertex")
            for first, second in zip(ids, ids[1:]):
                self.assertIn(tuple(sorted((first, second))), edges, f"{name} seam is not an edge path")
            points = vertices[ids]
            center = points.mean(axis=0)
            centered = points - center
            centers.append(center)
            areas.append(np.cross(centered, np.roll(centered, -1, axis=0)).sum(axis=0))

        axis = centers[1] - centers[0]
        self.assertGreater(np.linalg.norm(axis), 0, "Seam centers must be distinct")
        axis /= np.linalg.norm(axis)
        winding = [float(area @ axis) for area in areas]
        for name, area, signed in zip(("top", "bottom"), areas, winding):
            self.assertGreater(
                abs(signed), 1e-8 * np.linalg.norm(area),
                f"{name} seam has ambiguous winding about the apple axis",
            )
        self.assertGreater(
            winding[0] * winding[1], 0,
            f"Top and bottom seam winding differs: {winding}",
        )
        assembly, top_ring, bottom_ring, side_seam = geometry._create_apple_body(32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "canonical_seam.csv"
            geometry._write_seam_csv(
                path, assembly, top_ring, bottom_ring, side_seam,
                math.hypot(*[assembly.points[top_ring[0]][i] for i in (0, 2)]),
                math.hypot(*[assembly.points[bottom_ring[0]][i] for i in (0, 2)]),
            )
            canonical_seams = align_meshes._seam_ids(path, len(assembly.points))
        canonical_vertices = np.asarray(assembly.points)
        canonical_axis = (
            canonical_vertices[bottom_ring].mean(axis=0)
            - canonical_vertices[top_ring].mean(axis=0)
        )
        canonical_axis /= np.linalg.norm(canonical_axis)
        for name, ids, scan_winding in zip(("top", "bottom"), canonical_seams[:2], winding):
            points = canonical_vertices[ids[:-1]]
            centered = points - points.mean(axis=0)
            canonical_winding = float(
                np.cross(centered, np.roll(centered, -1, axis=0)).sum(axis=0) @ canonical_axis
            )
            self.assertGreater(
                canonical_winding * scan_winding, 0,
                f"Canonical {name} seam winding differs from Envy1",
            )

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
        chart_uvs = [{}, {}]
        for chart, face_range in enumerate((top_faces, bottom_faces)):
            for face, face_uv in zip(
                assembly.triangles[slice(*face_range)], uv_faces[slice(*face_range)]
            ):
                for vertex, uv_index in zip(face, face_uv):
                    chart_uvs[chart].setdefault(vertex, []).append(uvs[uv_index])
        top_ids = [int(value) for value in rows[0]]
        bottom_ids = [int(value) for value in rows[3]]
        self.assertEqual(top_ids, [top_ring[0], *reversed(top_ring[1:]), top_ring[0]])
        self.assertEqual(bottom_ids, [bottom_ring[0], *reversed(bottom_ring[1:]), bottom_ring[0]])
        self.assertEqual(top_ids, [int(value) for value in rows[6]])
        self.assertEqual(bottom_ids, list(reversed([int(value) for value in rows[12]])))
        for start, center in ((0, (0.9, 0.1)), (3, (0.1, 0.9))):
            ids = [int(value) for value in rows[start]]
            coordinates = np.column_stack((np.array(rows[start + 1], dtype=float),
                                           np.array(rows[start + 2], dtype=float)))
            for vertex, actual in zip(ids, coordinates):
                radius = math.hypot(*[assembly.points[vertex][i] for i in (0, 2)])
                expected = (
                    geometry._circle_uv(assembly.points[vertex], center, radius)
                    if start == 0 else geometry._bottom_uv(assembly.points[vertex], radius)
                )
                np.testing.assert_allclose(
                    actual, expected, atol=1e-8
                )
                self.assertTrue(any(np.allclose(actual, uv, atol=1e-8)
                                    for uv in chart_uvs[start // 3][vertex]))
        body_uvs = {}
        for face, face_uv in zip(assembly.triangles[:top_faces[0]], uv_faces[:top_faces[0]]):
            for vertex, uv_index in zip(face, face_uv):
                body_uvs.setdefault(vertex, []).append(uvs[uv_index])
        corners = [(1, 0.5), (0.5, 1), (0, 0.5), (0.5, 0), (1, 0.5)]
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
        np.testing.assert_allclose(seam_uvs.sum(axis=1), 1.5, atol=1e-8)
        self.assertTrue(np.any(np.all(np.isclose(seam_uvs, [1.0, 0.5]), axis=1)))
        self.assertTrue(np.any(np.all(np.isclose(seam_uvs, [0.5, 1.0]), axis=1)))


if __name__ == "__main__":
    unittest.main()
