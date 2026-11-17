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
    def test_texture_marks_south_pole_with_cyan_dot(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            obj_path = Path(directory) / "apple.obj"
            geometry._create_apple(16, obj_path)
            image = Image.open(obj_path.with_suffix(".png")).convert("RGB")
            vertices = np.array([
                [float(value) for value in line.split()[1:4]]
                for line in obj_path.read_text(encoding="utf-8").splitlines()
                if line.startswith("v ")
            ])
        lowest = vertices[:, 1].min()
        peaks = np.flatnonzero(vertices[:, 1] <= lowest + 1e-6)
        self.assertEqual(len(peaks), 1)
        bottom_radius = 0.5 * math.cos(math.radians(70))
        u, v = geometry._bottom_uv(tuple(vertices[peaks[0]]), bottom_radius)
        self.assertEqual(
            image.getpixel((int(u * image.width), int((1 - v) * image.height))),
            (0, 255, 255),
        )

    def test_texture_marks_non_seam_vertices_with_grey_dots(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            obj_path = Path(directory) / "apple.obj"
            geometry._create_apple(16, obj_path)
            image = Image.open(obj_path.with_suffix(".png")).convert("RGB")

            assembly, top_ring, bottom_ring, side_seam = geometry._create_apple_body(16)
            top_faces = geometry._create_apple_top(assembly, top_ring, 16)
            bottom_faces = geometry._create_apple_bottom(assembly, bottom_ring)
            seam_vertices = set(top_ring) | set(bottom_ring) | set(side_seam)
            texture_coordinates = []
            candidate_uv = None
            for line in obj_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if parts[0] == "vt":
                    texture_coordinates.append((float(parts[1]), float(parts[2])))
                elif parts[0] == "f":
                    for corner in parts[1:]:
                        vertex_index, texture_index = (
                            int(index) - 1 for index in corner.split("/")
                        )
                        face_index = texture_index // 3
                        if (
                            vertex_index not in seam_vertices
                            and (face_index < top_faces[0] or face_index >= top_faces[1])
                            and (face_index < bottom_faces[0] or face_index >= bottom_faces[1])
                        ):
                            candidate_uv = texture_coordinates[texture_index]
                            break
                    if candidate_uv is not None:
                        break

            self.assertIsNotNone(candidate_uv)
            u, v = candidate_uv
            self.assertEqual(
                image.getpixel((int(u * image.width), int((1 - v) * image.height))),
                (128, 128, 128),
            )

    def test_texture_circles_have_grey_center_and_positive_x_line(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "texture.png"
            geometry._write_texture_map(path)
            image = Image.open(path).convert("RGB")
        size = image.width

        def pixel(u, v):
            return image.getpixel((int(u * size), int((1 - v) * size)))

        for (cu, cv), base in (((0.9, 0.1), (0, 255, 0)), ((0.1, 0.9), (0, 0, 128))):
            self.assertEqual(pixel(cu, cv), (128, 128, 128))
            self.assertEqual(pixel(cu + 0.05, cv), (0, 0, 0))
            self.assertEqual(pixel(cu - 0.05, cv), base)
            self.assertEqual(pixel(cu, cv + 0.05), base)
        self.assertEqual(pixel(0.5, 0.5), (255, 0, 0))
        self.assertEqual(pixel(0.75, 0.25), (0, 0, 0))
        self.assertEqual(pixel(0.25, 0.75), (0, 0, 0))
        self.assertEqual(pixel(0.5, 0.25), (255, 0, 0))
        self.assertEqual(pixel(0.75, 0.75), (144, 238, 144))
        self.assertEqual(pixel(0.25, 0.25), (173, 216, 230))

    def test_bottom_grid_spacing_and_seam_csv(self):
        for samples in (3, 12, 32, 64):
            with self.subTest(samples=samples):
                assembly, top, bottom, side_seam = geometry._create_apple_body(samples)
                geometry._create_apple_top(assembly, top, samples)
                body_faces = list(assembly.triangles)
                first_new_point = len(assembly.points)
                first, last = geometry._create_apple_bottom(assembly, bottom)
                self.assertEqual(assembly.triangles[:first], body_faces)
                self.assertGreater(last, first)
                boundary_lengths = np.linalg.norm(
                    np.asarray(assembly.points)[bottom]
                    - np.roll(np.asarray(assembly.points)[bottom], -1, axis=0),
                    axis=1,
                )
                bottom_latitude = math.radians(-70.0)
                pole_distance = math.dist(
                    (0.5 * math.cos(bottom_latitude), 0.5 * math.sin(bottom_latitude)),
                    (0.0, -0.5),
                )
                target_edge_length = min(
                    float(boundary_lengths.mean()) / 2,
                    2 * pole_distance / 3,
                )
                ring_latitude = bottom_latitude - 2 * math.asin(target_edge_length)
                ring_count = max(
                    3,
                    round(
                        len(bottom)
                        * math.cos(ring_latitude)
                        / math.cos(bottom_latitude)
                    ),
                )
                transition_ring = np.asarray(
                    assembly.points[first_new_point:first_new_point + ring_count]
                )
                self.assertTrue(np.allclose(
                    transition_ring[:, 1],
                    0.5 * math.sin(ring_latitude),
                ))
                meridian_edge_length = math.dist(
                    (
                        0.5 * math.cos(bottom_latitude),
                        0.5 * math.sin(bottom_latitude),
                    ),
                    (0.5 * math.cos(ring_latitude), 0.5 * math.sin(ring_latitude)),
                )
                self.assertAlmostEqual(
                    meridian_edge_length,
                    target_edge_length,
                    places=12,
                )
                edges = {}
                for face in assembly.triangles:
                    for i in range(3):
                        edge = tuple(sorted((face[i], face[(i + 1) % 3])))
                        edges[edge] = edges.get(edge, 0) + 1
                self.assertTrue(all(count == 2 for count in edges.values()))
                for i, vertex in enumerate(bottom):
                    edge = tuple(sorted((vertex, bottom[(i + 1) % len(bottom)])))
                    self.assertEqual(edges[edge], 2)
                self.assertEqual(assembly.points[-1], (0.0, -0.5, 0.0))
                vertices = np.asarray(assembly.points)
                cap_faces = np.asarray(assembly.triangles[first:last])
                cap_edges = np.unique(
                    np.sort(
                        np.concatenate((
                            cap_faces[:, [0, 1]],
                            cap_faces[:, [1, 2]],
                            cap_faces[:, [2, 0]],
                        )),
                        axis=1,
                    ),
                    axis=0,
                )
                cap_lengths = np.linalg.norm(
                    vertices[cap_edges[:, 0]] - vertices[cap_edges[:, 1]],
                    axis=1,
                )
                expected_length = target_edge_length
                self.assertLess(
                    abs(float(cap_lengths.mean()) - expected_length) / expected_length,
                    0.3,
                )
                grid_vertices = np.asarray(
                    assembly.points[first_new_point + ring_count:-1]
                )
                gap = np.linalg.norm(
                    transition_ring[:, None, :] - grid_vertices[None, :, :],
                    axis=2,
                ).min()
                self.assertGreater(gap, 0.4 * expected_length)
                self.assertLess(gap, 0.8 * expected_length)
                with tempfile.TemporaryDirectory() as directory:
                    csv_path = Path(directory) / "seam.csv"
                    geometry._write_seam_csv(
                        csv_path,
                        assembly,
                        top,
                        bottom,
                        side_seam,
                        math.hypot(*[assembly.points[top[0]][i] for i in (0, 2)]),
                        math.hypot(*[assembly.points[bottom[0]][i] for i in (0, 2)]),
                    )
                    with csv_path.open(newline="") as stream:
                        rows = list(csv.reader(stream))
                seam_ids = {
                    int(vertex)
                    for row_index in (0, 3, 6, 9, 12, 15)
                    for vertex in rows[row_index]
                }
                self.assertTrue(all(vertex < first_new_point for vertex in seam_ids))

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
                self.assertEqual(len(top_ring), samples)
                self.assertEqual(len(bottom_ring), samples)
                self.assertTrue(np.allclose(np.diff(latitudes), np.diff(latitudes)[0]))

                vertices = np.asarray(assembly.points)
                faces = np.asarray(assembly.triangles)
                triangles = vertices[faces]
                lengths = np.linalg.norm(
                    triangles - np.roll(triangles, 1, axis=1),
                    axis=2,
                )
                ratios = lengths.max(axis=1) / lengths.min(axis=1)
                self.assertLess(float(np.quantile(ratios, 0.95)), 1.8)

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
