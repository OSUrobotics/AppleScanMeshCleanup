import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
from meshlib import mrmeshnumpy as mn
from meshlib import mrmeshpy as mm
from PIL import Image

from make_apple_geometry import _BOTTOM_UV_CENTER, _create_apple
from normalize_uv_projection import _read_obj, _vertex_geometry
from remove_calyx import (
    _calyx_faces, _grid_patch, _pack_with_seams, _patch_stem, patch_stem_and_ring_calyx, remove_calyx,
)
from seam_topology import SeamInfo, SeamTopology


def _sphere(nlat=40, nlon=80, radius=50.0):
    verts = [[0, radius, 0]]
    for i in range(1, nlat):
        theta = np.pi * i / nlat
        for j in range(nlon):
            phi = 2 * np.pi * j / nlon
            verts.append([radius * np.sin(theta) * np.cos(phi), radius * np.cos(theta),
                          radius * np.sin(theta) * np.sin(phi)])
    verts.append([0, -radius, 0])
    last = len(verts) - 1
    faces = [[0, 1 + j, 1 + (j + 1) % nlon] for j in range(nlon)]
    for i in range(nlat - 2):
        for j in range(nlon):
            a, b = 1 + i * nlon + j, 1 + i * nlon + (j + 1) % nlon
            faces += [[a, a + nlon, b], [b, a + nlon, b + nlon]]
    base = 1 + (nlat - 2) * nlon
    faces += [[base + j, last, base + (j + 1) % nlon] for j in range(nlon)]
    return np.array(verts, np.float32), np.array(faces, np.int32)


class RemoveCalyxTest(unittest.TestCase):
    def test_packing_rejects_removed_seam_vertex(self):
        mesh = mm.makeUVSphere(1.0, 8, 8)
        vertices = mm.VertBitSet()
        vertices.resize(mesh.topology.vertSize())
        vertices.set(mm.VertId(0), True)
        mesh.deleteFaces(mm.getIncidentFaces(mesh.topology, vertices))
        topology = SeamTopology(-1, 1, (0, 1, 0), (0, -1, 0), {})
        topology.add_seam("calyx", SeamInfo("calyx", [0], [], [], (0, -1, 0), 1.0))

        with self.assertRaisesRegex(ValueError, "Seam vertex 0 was removed"):
            _pack_with_seams(mesh, topology)

    def test_patch_records_rings_and_remaps_calyx_vertices(self):
        mesh = mm.makeUVSphere(10.0, 16, 16)
        topology = SeamTopology(-10, 10, (0, 10, 0), (0, -10, 0), {})
        original_calyx_ids = []

        def patch_stem(*args, **kwargs):
            original_calyx_ids.extend(topology.seams["calyx"].vertex_ids)
            return _patch_stem(*args, **kwargs)

        with (
            patch("remove_calyx._high_curvature_vertices", return_value=[]) as curvature,
            patch("remove_calyx._bottom_center", return_value=(4.0, 5.0, 3.0)) as bottom_center,
            patch("remove_calyx._top_center", return_value=(np.array([0.1, 10, 0]), 4.0, 3.0, 0, [])) as top_center,
            patch("remove_calyx._patch_stem", side_effect=patch_stem),
        ):
            patch_stem_and_ring_calyx(mesh, topology, Path("unused.ply"))

        np.testing.assert_array_equal(bottom_center.call_args.kwargs["center"], topology.bottom_point)
        np.testing.assert_array_equal(top_center.call_args.kwargs["center"], topology.top_point)
        self.assertEqual([call.kwargs["height"] for call in curvature.call_args_list], [3.0, 3.0])
        self.assertEqual(set(topology.seams), {"calyx", "stem"})
        self.assertNotEqual(original_calyx_ids, topology.seams["calyx"].vertex_ids)
        self.assertEqual(topology.seams["calyx"].center, (0, -10, 0))
        self.assertAlmostEqual(topology.seams["calyx"].radius, 4.2)
        np.testing.assert_allclose(topology.seams["stem"].center,
                                   np.mean(topology.seams["stem"].vertex_locations, axis=0))
        self.assertEqual(topology.seams["stem"].radius, 1.5)
        self.assertEqual(topology.top_shoulder_radius, 4.0)
        self.assertEqual(topology.bottom_shoulder_radius, 5.0)
        vertices = mn.getNumpyVerts(mesh)
        for seam in topology.seams.values():
            self.assertGreater(len(seam.vertex_ids), 2)
            self.assertEqual(seam.uv_coords, [])
            self.assertTrue(all(mesh.topology.hasVert(mm.VertId(v)) for v in seam.vertex_ids))
            np.testing.assert_allclose(vertices[seam.vertex_ids], seam.vertex_locations)
        restored = SeamTopology.from_dict(json.loads(json.dumps(topology.to_dict())))
        self.assertEqual(restored.top_shoulder_radius, 4.0)
        self.assertEqual(restored.bottom_shoulder_radius, 5.0)

    def test_splits_textured_apple_and_fills_ring(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "apple.obj"
            _create_apple(60, source)
            source_lines, source_uvs, source_uv_faces, _ = _read_obj(source)
            source_vertices, source_faces = _vertex_geometry(source_lines, source_uv_faces, source)
            topology = SeamTopology.from_dict(json.loads(source.with_name("apple_seam_topology.json").read_text()))
            ring, inside, _ = _calyx_faces(source_vertices, source_faces, topology.seams["calyx"], topology.bottom_point)
            # Give the texture a position-dependent color to verify the crop's UV mapping.
            y, x = np.indices((1024, 1024))
            pixels = np.stack((x // 4, y // 4, np.full_like(x, 100)), axis=2).astype(np.uint8)
            Image.fromarray(pixels).save(source.with_suffix(".jpg"), quality=100, subsampling=0)
            output = remove_calyx(source, Path(tmp) / "out.obj")

            def read(path):
                lines, uvs, uv_faces, _ = _read_obj(path)
                vertices, faces = _vertex_geometry(lines, uv_faces, path)
                return vertices, faces, uvs, uv_faces

            body_vertices, body_faces, body_uvs, body_uv_faces = read(output)
            calyx_vertices, calyx_faces, calyx_uvs, calyx_uv_faces = read(Path(tmp) / "out_calyx.obj")

            def face_coordinates(vertices, faces):
                return {tuple(sorted(tuple(vertices[v]) for v in face)) for face in faces}

            original_calyx_faces = face_coordinates(source_vertices, source_faces[inside])
            self.assertEqual(face_coordinates(calyx_vertices, calyx_faces), original_calyx_faces)
            self.assertFalse(original_calyx_faces & face_coordinates(body_vertices, body_faces))
            for point in source_vertices[ring]:
                self.assertTrue(np.any(np.all(np.isclose(body_vertices, point, atol=1e-10), axis=1)))
                self.assertTrue(np.any(np.all(np.isclose(calyx_vertices, point, atol=1e-10), axis=1)))
            edges = Counter(tuple(sorted((int(face[i]), int(face[(i + 1) % 3]))))
                            for face in body_faces for i in range(3))
            self.assertEqual(set(edges.values()), {2})

            center_xz = np.asarray(topology.seams["calyx"].center)[[0, 2]]
            center_id = np.flatnonzero(
                np.all(np.isclose(body_vertices[:, [0, 2]], center_xz, atol=1e-10), axis=1)
                & (body_vertices[:, 1] < 0)
            )
            self.assertEqual(len(center_id), 1)
            center_uvs = body_uvs[body_uv_faces[body_faces == center_id[0]]]
            np.testing.assert_allclose(center_uvs, np.tile(_BOTTOM_UV_CENTER, (len(center_uvs), 1)), atol=1e-8)
            new_points = {tuple(p) for p in body_vertices} - {tuple(p) for p in source_vertices}
            self.assertGreater(len(new_points), 1)
            with Image.open(Path(tmp) / "out_calyx.png") as cropped, Image.open(source.with_suffix(".jpg")) as original:
                self.assertEqual(cropped.size, (256, 256))
                used_uvs = calyx_uvs[np.unique(calyx_uv_faces)]
                self.assertGreaterEqual(used_uvs.min(), 8 / 256 - 1e-8)
                self.assertLessEqual(used_uvs.max(), 1 - 8 / 256 + 1e-8)
                for face, uv_face in zip(calyx_faces[:10], calyx_uv_faces[:10]):
                    point = tuple(calyx_vertices[face[0]])
                    source_vertex = np.flatnonzero(np.all(source_vertices == point, axis=1))[0]
                    source_uv = source_uvs[source_uv_faces[source_faces == source_vertex][0]]
                    target_uv = calyx_uvs[uv_face[0]]
                    def sample(image, uv):
                        return np.asarray(image.getpixel((min(image.width - 1, int(uv[0] * image.width)),
                                                          min(image.height - 1, int((1 - uv[1]) * image.height)))))[:3]
                    np.testing.assert_allclose(sample(cropped, target_uv), sample(original, source_uv), atol=3)
            self.assertEqual(output.with_suffix(".jpg").read_bytes(), source.with_suffix(".jpg").read_bytes())

    def test_rejects_mismatched_calyx_seam(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "apple.obj"
            _create_apple(15, source)
            seam_path = source.with_name("apple_seam_topology.json")
            topology = json.loads(seam_path.read_text())
            topology["seams"]["calyx"]["vertex_locations"][0][0] += 1
            seam_path.write_text(json.dumps(topology))
            with self.assertRaisesRegex(ValueError, "locations do not match"):
                remove_calyx(source, Path(tmp) / "out.obj")
            self.assertFalse((Path(tmp) / "out.obj").exists())

    def test_grid_patch_handles_concave_ring(self):
        xz = np.array([[0, 0], [3, 0], [3, 1], [1, 1], [1, 3], [0, 3], [.5, .5]])
        vertices = np.column_stack((xz[:, 0], np.zeros(len(xz)), xz[:, 1]))
        faces = np.array([[i, (i + 1) % 6, 6] for i in range(6)])
        uvs = xz / 4
        points, patch_uvs, triangles = _grid_patch(
            vertices, faces, faces, uvs, list(range(6)), np.arange(6), 6, vertices[6]
        )
        projected = points[:, [0, 2]][triangles]
        cross = ((projected[:, 1, 0] - projected[:, 0, 0]) * (projected[:, 2, 1] - projected[:, 0, 1])
                 - (projected[:, 1, 1] - projected[:, 0, 1]) * (projected[:, 2, 0] - projected[:, 0, 0]))
        self.assertTrue(np.all(cross > 0))
        self.assertAlmostEqual(float(np.sum(cross) / 2), 5.0)
        np.testing.assert_allclose(patch_uvs[6], uvs[6])

    def test_patch_uses_eight_by_eight_cells_with_center(self):
        xz = np.array([[-4, -4], [4, -4], [4, 4], [-4, 4], [0, 0]], dtype=float)
        vertices = np.column_stack((xz[:, 0], np.zeros(len(xz)), xz[:, 1]))
        faces = np.array([[i, (i + 1) % 4, 4] for i in range(4)])
        uvs = (xz + 4) / 8
        points, patch_uvs, _ = _grid_patch(
            vertices, faces, faces, uvs, list(range(4)), np.arange(4), 4, vertices[4]
        )
        grid = {tuple(point) for point in points[4:, [0, 2]]}
        self.assertEqual(grid, {(x, z) for x in range(-3, 4) for z in range(-3, 4)})
        np.testing.assert_allclose(patch_uvs[4], uvs[4])

    def test_patch_avoids_uv_inversions_with_offset_center(self):
        count = 16
        angles = np.arange(count) * 2 * np.pi / count
        boundary = np.column_stack((np.cos(angles), np.sin(angles)))
        xz = np.vstack((boundary, [0, 0]))
        vertices = np.column_stack((xz[:, 0], np.zeros(len(xz)), xz[:, 1]))
        faces = np.array([[i, (i + 1) % count, count] for i in range(count)])
        uvs = np.vstack((0.5 + 0.4 * boundary, [0.54, 0.46]))
        points, patch_uvs, triangles = _grid_patch(
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
