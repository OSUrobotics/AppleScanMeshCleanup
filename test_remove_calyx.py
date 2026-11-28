import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from obj_geometry import read_obj
from remove_calyx import _calyx_faces, _grid_patch, remove_calyx
from seam_topology import SeamInfo, SeamTopology


def _sphere(nlat=10, nlon=24, radius=10.0):
    vertices = [[0, radius, 0]]
    for i in range(1, nlat):
        theta = np.pi * i / nlat
        for j in range(nlon):
            phi = 2 * np.pi * j / nlon
            vertices.append([
                radius * np.sin(theta) * np.cos(phi),
                radius * np.cos(theta),
                radius * np.sin(theta) * np.sin(phi),
            ])
    vertices.append([0, -radius, 0])
    last = len(vertices) - 1
    faces = [[0, 1 + j, 1 + (j + 1) % nlon] for j in range(nlon)]
    for i in range(nlat - 2):
        for j in range(nlon):
            a, b = 1 + i * nlon + j, 1 + i * nlon + (j + 1) % nlon
            faces.extend([[a, a + nlon, b], [b, a + nlon, b + nlon]])
    base = 1 + (nlat - 2) * nlon
    faces.extend([[base + j, last, base + (j + 1) % nlon] for j in range(nlon)])
    return np.asarray(vertices), np.asarray(faces, dtype=int)


def _write_sample_apple(directory):
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "apple.obj"
    vertices, faces = _sphere()
    uvs = np.column_stack((0.5 + vertices[:, 0] / 20, 0.5 + vertices[:, 2] / 20))
    normals = vertices / np.linalg.norm(vertices, axis=1)[:, None]
    with source.open("w", encoding="utf-8") as obj:
        obj.write("mtllib apple.mtl\no apple\n")
        for vertex in vertices:
            obj.write(f"v {vertex[0]:.12g} {vertex[1]:.12g} {vertex[2]:.12g}\n")
        for uv in uvs:
            obj.write(f"vt {uv[0]:.12g} {uv[1]:.12g}\n")
        for normal in normals:
            obj.write(f"vn {normal[0]:.12g} {normal[1]:.12g} {normal[2]:.12g}\n")
        obj.write("usemtl material_0\n")
        for face in faces:
            obj.write("f " + " ".join(f"{v + 1}/{v + 1}/{v + 1}" for v in face) + "\n")

    (directory / "apple.mtl").write_text(
        "newmtl material_0\nmap_Kd apple.png\n", encoding="utf-8"
    )
    y, x = np.indices((1024, 1024))
    pixels = np.stack((x // 4, y // 4, np.full_like(x, 100)), axis=2).astype(np.uint8)
    Image.fromarray(pixels).save(directory / "apple.png")

    nlon = 24
    ring_start = 1 + (10 - 2) * nlon
    ring = list(range(ring_start, ring_start + nlon))
    top_ring = list(range(1, nlon + 1))
    bottom = tuple(vertices[-1])
    top = tuple(vertices[0])
    calyx_radius = float(np.max(np.linalg.norm(vertices[ring][:, [0, 2]], axis=1)))
    topology = SeamTopology(
        -10.0,
        10.0,
        top,
        bottom,
        {
            "calyx": SeamInfo("calyx", ring, vertices[ring].tolist(), bottom, calyx_radius),
            "stem": SeamInfo(
                "stem", top_ring, vertices[top_ring].tolist(), top, calyx_radius
            ),
        },
        top_vertex=0,
        bottom_vertex=len(vertices) - 1,
    )
    source.with_name("apple_seam_topology.json").write_text(
        json.dumps(topology.to_dict()), encoding="utf-8"
    )
    return source, vertices, faces, uvs, topology


def _read_mesh(path):
    vertices, _, _, _, faces, uv_faces, _, _ = read_obj(path)
    return vertices, faces, uv_faces


def _point_in_polygon(point, polygon):
    x, y = point
    inside = False
    for index in range(len(polygon)):
        x1, y1 = polygon[index]
        x2, y2 = polygon[(index + 1) % len(polygon)]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


class RemoveCalyxTest(unittest.TestCase):
    def test_splits_textured_apple_and_fills_ring(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, source_vertices, source_faces, source_uvs, topology = _write_sample_apple(
                Path(tmp)
            )
            ring, inside, _ = _calyx_faces(
                source_vertices, source_faces, topology.seams["calyx"], topology.bottom_point
            )
            output = remove_calyx(source, Path(tmp) / "out.obj")

            body_vertices, body_faces, body_uv_faces = _read_mesh(output)
            calyx_path = Path(tmp) / "Calyx" / "out_calyx.obj"
            calyx_vertices, calyx_faces, calyx_uv_faces = _read_mesh(calyx_path)

            def face_coordinates(vertices, faces):
                return {
                    tuple(sorted(tuple(np.round(vertices[v], 6)) for v in face))
                    for face in faces
                }

            calyx_center_xz = np.asarray(topology.seams["calyx"].center)[[0, 2]]
            expected_calyx_vertices = source_vertices.copy()
            expected_calyx_vertices[:, [0, 2]] -= calyx_center_xz
            calyx_face_set = face_coordinates(calyx_vertices, calyx_faces)
            original_calyx_faces = face_coordinates(
                expected_calyx_vertices, source_faces[inside]
            )
            self.assertEqual(calyx_face_set, original_calyx_faces)
            self.assertFalse(original_calyx_faces & face_coordinates(body_vertices, body_faces))
            for point in source_vertices[ring]:
                self.assertTrue(np.any(np.all(np.isclose(body_vertices, point), axis=1)))
                translated = point.copy()
                translated[[0, 2]] -= calyx_center_xz
                self.assertTrue(np.any(np.all(np.isclose(calyx_vertices, translated), axis=1)))

            edges = Counter(
                tuple(sorted((int(face[i]), int(face[(i + 1) % 3]))))
                for face in body_faces
                for i in range(3)
            )
            self.assertEqual(set(edges.values()), {2})

            center = np.asarray(topology.seams["calyx"].center)
            center_y = float(np.mean(source_vertices[ring, 1])) + 2
            center_id = np.flatnonzero(
                np.all(np.isclose(body_vertices[:, [0, 2]], center[[0, 2]]), axis=1)
                & np.isclose(body_vertices[:, 1], center_y)
            )
            self.assertEqual(len(center_id), 1)

            _, body_uvs, _, _, body_faces, body_uv_faces, _, _ = read_obj(output)
            center_uv_ids = body_uv_faces[body_faces == center_id[0]]
            np.testing.assert_allclose(
                body_uvs[center_uv_ids],
                np.tile(source_uvs[-1], (len(center_uv_ids), 1)),
                atol=1e-8,
            )
            source_points = {tuple(np.round(point, 6)) for point in source_vertices}
            new_vertex_ids = [
                index for index, point in enumerate(body_vertices)
                if tuple(np.round(point, 6)) not in source_points
            ]
            self.assertGreater(len(new_vertex_ids), 1)
            new_uv_ids = np.unique(body_uv_faces[np.isin(body_faces, new_vertex_ids)])
            uv_ring = source_uvs[ring]
            for uv in body_uvs[new_uv_ids]:
                self.assertTrue(_point_in_polygon(uv, uv_ring))

            _, calyx_uvs, _, _, _, calyx_uv_faces, _, _ = read_obj(calyx_path)
            source_calyx_uvs = {
                tuple(np.round(source_uvs[vertex], 8))
                for vertex in np.unique(source_faces[inside])
            }
            output_calyx_uvs = {
                tuple(np.round(uv, 8)) for uv in calyx_uvs[np.unique(calyx_uv_faces)]
            }
            self.assertEqual(output_calyx_uvs, source_calyx_uvs)
            self.assertEqual(
                (Path(tmp) / "Calyx" / "apple.png").read_bytes(),
                source.with_name("apple.png").read_bytes(),
            )
            self.assertEqual(
                output.parent.joinpath("apple.png").read_bytes(),
                source.with_name("apple.png").read_bytes(),
            )

    def test_rejects_mismatched_calyx_seam(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, _, _, _, _ = _write_sample_apple(Path(tmp))
            seam_path = source.with_name("apple_seam_topology.json")
            topology = json.loads(seam_path.read_text(encoding="utf-8"))
            topology["seams"]["calyx"]["vertex_locations"][0][0] += 1
            seam_path.write_text(json.dumps(topology), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "locations do not match"):
                remove_calyx(source, Path(tmp) / "out.obj")
            self.assertFalse((Path(tmp) / "out.obj").exists())

    def test_grid_patch_handles_concave_ring(self):
        xz = np.array([[0, 0], [3, 0], [3, 1], [1, 1], [1, 3], [0, 3], [.5, .5]])
        vertices = np.column_stack((xz[:, 0], np.zeros(len(xz)), xz[:, 1]))
        faces = np.array([[i, (i + 1) % 6, 6] for i in range(6)])
        uvs = xz / 4
        points, patch_uvs, triangles, _ = _grid_patch(
            vertices, faces, faces, uvs, list(range(6)), np.arange(6), 6, vertices[6]
        )
        projected = points[:, [0, 2]][triangles]
        cross = ((projected[:, 1, 0] - projected[:, 0, 0]) * (projected[:, 2, 1] - projected[:, 0, 1])
                 - (projected[:, 1, 1] - projected[:, 0, 1]) * (projected[:, 2, 0] - projected[:, 0, 0]))
        self.assertTrue(np.all(cross > 0))
        self.assertAlmostEqual(float(np.sum(cross) / 2), 5.0)
        np.testing.assert_allclose(patch_uvs[6], uvs[6])

    def test_patch_uses_minimum_grid_and_parabolic_height(self):
        xz = np.array([[-4, -4], [4, -4], [4, 4], [-4, 4], [0, 0]], dtype=float)
        vertices = np.column_stack((xz[:, 0], np.zeros(len(xz)), xz[:, 1]))
        faces = np.array([[i, (i + 1) % 4, 4] for i in range(4)])
        uvs = (xz + 4) / 8
        points, patch_uvs, _, _ = _grid_patch(
            vertices, faces, faces, uvs, list(range(4)), np.arange(4), 4, vertices[4]
        )
        grid = {tuple(point) for point in points[4:, [0, 2]]}
        expected = {(x * 8 / 3, z * 8 / 3) for x in range(-1, 2) for z in range(-1, 2)}
        np.testing.assert_allclose(sorted(grid), sorted(expected), atol=1e-10)
        np.testing.assert_allclose(patch_uvs[4], uvs[4])
        np.testing.assert_allclose(points[:4], vertices[:4])
        self.assertAlmostEqual(points[4, 1], 2)
        edge_point = next(point for point in points[5:] if np.allclose(point[[0, 2]], [8 / 3, 0]))
        self.assertAlmostEqual(edge_point[1], 2 * (1 - (2 / 3) ** 2))
        corner_point = next(point for point in points[5:] if np.allclose(point[[0, 2]], [8 / 3, 8 / 3]))
        self.assertAlmostEqual(corner_point[1], edge_point[1])

    def test_patch_preserves_nonflat_ring_and_lifts_center(self):
        xz = np.array([[-4, -4], [4, -4], [4, 4], [-4, 4], [0, 0]], dtype=float)
        vertices = np.column_stack((xz[:, 0], [1, 2, 3, 4, -1], xz[:, 1]))
        faces = np.array([[i, (i + 1) % 4, 4] for i in range(4)])
        points, _, _, _ = _grid_patch(
            vertices, faces, faces, (xz + 4) / 8, list(range(4)), np.arange(4), 4, vertices[4]
        )
        np.testing.assert_allclose(points[:4], vertices[:4])
        self.assertAlmostEqual(points[4, 1], 4.5)

    def test_grid_size_scales_with_ring_vertex_count(self):
        for count, expected_size in ((16, 3), (80, 4), (160, 8)):
            with self.subTest(count=count):
                angles = np.arange(count) * 2 * np.pi / count
                boundary = np.column_stack((np.cos(angles), np.sin(angles)))
                xz = np.vstack((boundary, [0, 0]))
                vertices = np.column_stack((xz[:, 0], np.zeros(len(xz)), xz[:, 1]))
                faces = np.array([[i, (i + 1) % count, count] for i in range(count)])
                uvs = 0.5 + 0.4 * xz
                points, _, _, _ = _grid_patch(
                    vertices, faces, faces, uvs, list(range(count)), np.arange(count), count, vertices[count]
                )
                grid = points[count:, [0, 2]]
                spacing = 2 / expected_size
                np.testing.assert_allclose(grid / spacing, np.round(grid / spacing), atol=1e-10)
                self.assertEqual(np.max(np.abs(np.round(grid / spacing))),
                                 expected_size // 2 - (expected_size % 2 == 0))

    def test_patch_avoids_uv_inversions_with_offset_center(self):
        count = 16
        angles = np.arange(count) * 2 * np.pi / count
        boundary = np.column_stack((np.cos(angles), np.sin(angles)))
        xz = np.vstack((boundary, [0, 0]))
        vertices = np.column_stack((xz[:, 0], np.zeros(len(xz)), xz[:, 1]))
        faces = np.array([[i, (i + 1) % count, count] for i in range(count)])
        uvs = np.vstack((0.5 + 0.4 * boundary, [0.54, 0.46]))
        points, patch_uvs, triangles, _ = _grid_patch(
            vertices, faces, faces, uvs, list(range(count)), np.arange(count), count, vertices[count]
        )
        np.testing.assert_allclose(patch_uvs[:count], uvs[:count])
        np.testing.assert_allclose(patch_uvs[count], uvs[count])
        for coordinates in (points[:, [0, 2]], patch_uvs):
            a = coordinates[triangles[:, 1]] - coordinates[triangles[:, 0]]
            b = coordinates[triangles[:, 2]] - coordinates[triangles[:, 0]]
            self.assertTrue(np.all(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0] > 0))

    def test_missing_input(self):
        with self.assertRaisesRegex(FileNotFoundError, "Input mesh not found"):
            remove_calyx(Path("missing.obj"), Path("out.ply"))


if __name__ == "__main__":
    unittest.main()
