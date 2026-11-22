import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
import trimesh
import uv_texture as uv_module
from meshlib import mrmeshnumpy as mn
from meshlib import mrmeshpy as mm

from seam_topology import SeamInfo, SeamTopology
from uv_texture import _define_seam, _load_mesh, _orient_mesh, _store_boundary_paths, _update_seam_topology, _write_seam_mesh, uv_texture


class UVSeamExportTests(unittest.TestCase):
    def test_standalone_saves_updated_topology(self):
        with tempfile.TemporaryDirectory() as temporary:
            input_path = Path(temporary) / "apple.obj"
            output_path = Path(temporary) / "apple_uv.obj"
            topology = SeamTopology(-1, 1, (0, 1, 0), (0, -1, 0))
            input_path.with_name("apple_seam_topology.json").write_text(json.dumps(topology.to_dict()))

            def update(input_path, output_path, seam_topology):
                seam_topology.top_shoulder_radius = 2.0

            with (patch("uv_texture.Path", side_effect=[input_path, output_path]),
                  patch("uv_texture.uv_texture", side_effect=update)):
                uv_module.main()
            result = json.loads(output_path.with_name("apple_uv_seam_topology.json").read_text())
            self.assertEqual(result["top_shoulder_radius"], 2.0)
            self.assertFalse(output_path.with_name("apple_uv_seam.csv").exists())

    def test_texture_real_mesh_preserves_and_updates_ring_vertices(self):
        mesh = mm.makeUVSphere(50.0, 32, 32)
        vertices = mn.getNumpyVerts(mesh)
        topology = SeamTopology(
            -50, 50, (0, 50, 0), (0, -50, 0), {},
            top_shoulder_radius=12.0, bottom_shoulder_radius=20.0,
        )
        for name, vertex_id in (("stem", 0), ("calyx", len(vertices) - 1)):
            topology.add_seam(name, SeamInfo(
                name, [vertex_id], vertices[[vertex_id]].tolist(), [], (0, 0, 0), 1
            ))
        with tempfile.TemporaryDirectory() as temporary:
            input_path = Path(temporary) / "apple1.obj"
            output_path = Path(temporary) / "apple1_uv.obj"
            mm.saveMesh(mesh, str(input_path))
            uv_texture(input_path, output_path, topology)
            self.assertTrue(output_path.is_file())
        self.assertEqual(set(topology.seams), {"stem", "calyx", "top", "bottom", "side"})
        for seam in topology.seams.values():
            self.assertEqual(len(seam.uv_coords), len(seam.vertex_ids))
            self.assertTrue(all(copies for copies in seam.uv_coords))

    def test_rejects_ring_vertices_that_no_longer_match_the_mesh(self):
        mesh = trimesh.creation.box()
        topology = SeamTopology(-1, 1, (0, 1, 0), (0, -1, 0), {})
        topology.add_seam("stem", SeamInfo("stem", [0], [[100, 100, 100]], [], (0, 0, 0), 1))
        with self.assertRaisesRegex(ValueError, "stem seam vertices do not match"):
            _update_seam_topology(topology, mesh, [[], [], []], [np.zeros(3), np.zeros(3)], (1, 1), {}, np.empty((0, 2)))

    def test_texture_stores_all_uv_copies_for_each_seam_vertex(self):
        vertices = [[0, 2, 0], [0, -2, 0]]
        for y in (1, -1):
            vertices.extend([[1, y, 0], [0, y, 1], [-1, y, 0], [0, y, -1]])
        faces = []
        for i in range(4):
            a, b = 2 + i, 2 + (i + 1) % 4
            c, d = 6 + i, 6 + (i + 1) % 4
            faces.extend([[0, b, a], [1, c, d], [a, b, c], [b, d, c]])
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        topology = SeamTopology(
            -3, 4, (1, 2, 0), (0, -2, 0), {},
            top_shoulder_radius=2.5, bottom_shoulder_radius=3.5,
        )
        for name, ids in (("stem", [0, 2]), ("calyx", [1, 6])):
            topology.add_seam(name, SeamInfo(name, ids, mesh.vertices[ids].tolist(), [], (0, 0, 0), 1))
        top = [2, 3, 4, 5, 2]
        bottom = [6, 7, 8, 9, 6]
        top_edges = list(zip(top[:-1], top[1:]))
        bottom_edges = list(zip(bottom[:-1], bottom[1:]))

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "apple.obj"
            with (
                patch("uv_texture._load_mesh", return_value=mesh),
                patch("uv_texture._orient_mesh", wraps=_orient_mesh) as orient,
                patch("uv_texture._define_seam", side_effect=[
                    (top, top_edges, 2.5), (bottom, bottom_edges, 3.5)
                ]) as define,
                patch("uv_texture._side_seam", return_value=([2, 6], [(2, 6)])),
                patch("uv_texture._write_seam_mesh"),
            ):
                uv_texture(Path("input.obj"), output, topology)

            obj_lines = output.read_text(encoding="utf-8").splitlines()
            obj_uvs = [list(map(float, line.split()[1:])) for line in obj_lines if line.startswith("vt ")]
            by_vertex = {}
            for line in obj_lines:
                if line.startswith("f "):
                    for corner in line.split()[1:]:
                        vertex_id, uv_id = (int(value) - 1 for value in corner.split("/"))
                        by_vertex.setdefault(vertex_id, set()).add(uv_id)
            for seam in topology.seams.values():
                for vertex_id, location, copies in zip(seam.vertex_ids, seam.vertex_locations, seam.uv_coords):
                    np.testing.assert_allclose(location, mesh.vertices[vertex_id])
                    self.assertEqual(len(copies), len(by_vertex[vertex_id]))
                    for uv in copies:
                        self.assertTrue(any(np.allclose(uv, obj_uvs[index]) for index in by_vertex[vertex_id]))

        self.assertEqual(set(topology.seams), {"stem", "calyx", "top", "bottom", "side"})
        self.assertEqual(topology.seams["top"].vertex_ids, top[:-1])
        self.assertEqual(topology.seams["bottom"].vertex_ids, bottom[:-1])
        self.assertEqual(topology.seams["side"].vertex_ids, [2, 6])
        self.assertEqual(topology.seams["top"].radius, 2.5)
        self.assertEqual(topology.seams["bottom"].radius, 3.5)
        self.assertIsNone(topology.seams["side"].center)
        self.assertGreater(len(topology.seams["stem"].uv_coords[1]), 1)
        np.testing.assert_array_equal(orient.call_args.args[1], topology.top_point)
        np.testing.assert_array_equal(orient.call_args.args[2], topology.bottom_point)
        for call in define.call_args_list:
            self.assertEqual((call.kwargs["y_min"], call.kwargs["y_max"]), (-3, 4))
            np.testing.assert_array_equal(call.kwargs["vertex_heights"], mesh.vertices[:, 1])
        self.assertEqual([call.kwargs["radius"] for call in define.call_args_list], [2.5, 3.5])

    def test_define_seam_uses_original_vertex_heights_and_topology_bounds(self):
        angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        vertices = np.vstack([
            np.column_stack((r * np.cos(angles), np.full(16, y), r * np.sin(angles)))
            for y in (-100, 100) for r in (2, 5)
        ])
        mesh = trimesh.Trimesh(vertices=vertices, faces=[], process=False)
        heights = np.array([10] * 32 + [-10] * 32)
        with patch("uv_texture._geodesic", side_effect=lambda adjacency, v1, v2, no_cross_vs: ([v1, v2], [(v1, v2)])):
            top, _, top_radius = _define_seam(mesh, [], set(), True, heights, -10, 10, 5.0)
            bottom, _, bottom_radius = _define_seam(mesh, [], set(), False, heights, -10, 10, 2.0)
        self.assertEqual(set(top), set(range(16, 32)))
        self.assertEqual(set(bottom), set(range(32, 48)))
        self.assertEqual((top_radius, bottom_radius), (5.0, 2.0))

    def test_texture_requires_shoulder_radii(self):
        topology = SeamTopology(-1, 1, (0, 1, 0), (0, -1, 0), {})
        with self.assertRaisesRegex(ValueError, "top and bottom shoulder radii"):
            uv_texture(Path("input.obj"), Path("output.obj"), topology)

    def test_define_seam_rejects_invalid_shoulder_radius(self):
        mesh = trimesh.creation.box()
        for radius in (0, -1, float("nan")):
            with self.subTest(radius=radius), self.assertRaisesRegex(ValueError, "Shoulder radius"):
                _define_seam(mesh, [], set(), True, mesh.vertices[:, 1], -1, 1, radius)

    def test_load_mesh_preserves_vertex_ids_and_orientation_handles_aligned_axis(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mesh.obj"
            path.write_text(
                "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 0\nf 1 2 3\nf 4 3 2\n",
                encoding="utf-8",
            )
            mesh = _load_mesh(path)
        self.assertEqual(len(mesh.vertices), 4)
        oriented, top, bottom = _orient_mesh(mesh, np.array([0., 1., 0.]), np.array([0., -1., 0.]))
        np.testing.assert_array_equal(top, [0, 0, 0])
        np.testing.assert_array_equal(bottom, [0, -2, 0])
        self.assertIs(oriented, mesh)

    def test_load_obj_preserves_unused_vertices_and_negative_indices_with_normals(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mesh.obj"
            path.write_text(
                "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 9 9 9\n"
                "vn 0 0 1\nvn 0 0 -1\n"
                "f -4//1 -3//1 -2//1 # first face\n"
                "f 1//2 3//2 2//2\n",
                encoding="utf-8",
            )
            mesh = _load_mesh(path)
        np.testing.assert_array_equal(mesh.vertices, [[0, 0, 0], [1, 0, 0], [0, 1, 0], [9, 9, 9]])
        np.testing.assert_array_equal(mesh.faces, [[0, 1, 2], [0, 2, 1]])

    def test_textured_obj_round_trip_preserves_geometry_and_seam_indices(self):
        vertices = [[0, 2, 0], [0, -2, 0]]
        for y in (1, -1):
            vertices.extend([[1, y, 0], [0, y, 1], [-1, y, 0], [0, y, -1]])
        vertices.append([9, 9, 9])
        faces = []
        for i in range(4):
            a, b = 2 + i, 2 + (i + 1) % 4
            c, d = 6 + i, 6 + (i + 1) % 4
            faces.extend([[0, b, a], [1, c, d], [a, b, c], [b, d, c]])
        top, bottom = [2, 3, 4, 5, 2], [6, 7, 8, 9, 6]
        topology = SeamTopology(
            -2, 2, (0, 2, 0), (0, -2, 0), {},
            top_shoulder_radius=2.5, bottom_shoulder_radius=3.5,
        )
        for name, ids in (("stem", [0]), ("calyx", [1])):
            topology.add_seam(name, SeamInfo(
                name, ids, [vertices[index] for index in ids], [], (0, 0, 0), 1
            ))

        with tempfile.TemporaryDirectory() as temporary:
            input_path = Path(temporary) / "input.obj"
            output_path = Path(temporary) / "output.obj"
            lines = ["v " + " ".join(map(str, vertex)) for vertex in vertices]
            lines.extend(["vt 0 0", "vt 1 1"])
            lines.extend(
                "f " + " ".join(f"{vertex + 1}/{1 + (face_index % 2)}" for vertex in face)
                for face_index, face in enumerate(faces)
            )
            input_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with (
                patch("uv_texture._define_seam", side_effect=[
                    (top, list(zip(top[:-1], top[1:])), 2.5),
                    (bottom, list(zip(bottom[:-1], bottom[1:])), 3.5),
                ]),
                patch("uv_texture._side_seam", return_value=([2, 6], [(2, 6)])),
                patch("uv_texture._write_seam_mesh"),
                patch.object(topology, "check_valid"),
            ):
                uv_texture(input_path, output_path, topology)

            obj_lines = output_path.read_text(encoding="utf-8").splitlines()
            obj_vertices = [list(map(float, line.split()[1:])) for line in obj_lines if line.startswith("v ")]
            obj_uvs = [list(map(float, line.split()[1:])) for line in obj_lines if line.startswith("vt ")]
            obj_faces = []
            by_vertex = {}
            for line in obj_lines:
                if line.startswith("f "):
                    corners = [tuple(int(value) - 1 for value in corner.split("/")) for corner in line.split()[1:]]
                    obj_faces.append(tuple(vertex for vertex, _ in corners))
                    for vertex, uv in corners:
                        by_vertex.setdefault(vertex, set()).add(uv)
            np.testing.assert_array_equal(obj_vertices, vertices)
            self.assertEqual(Counter(obj_faces), Counter(map(tuple, faces)))
            saved = json.loads(output_path.with_name("output_seam_topology.json").read_text())
            for seam in saved["seams"].values():
                self.assertEqual(len(seam["vertex_ids"]), len(seam["vertex_locations"]))
                self.assertEqual(len(seam["vertex_ids"]), len(seam["uv_coords"]))
                for vertex_id, location, copies in zip(seam["vertex_ids"], seam["vertex_locations"], seam["uv_coords"]):
                    np.testing.assert_array_equal(location, vertices[vertex_id])
                    self.assertEqual(len(copies), len(by_vertex[vertex_id]))
                    for uv in copies:
                        self.assertTrue(any(np.allclose(uv, obj_uvs[index]) for index in by_vertex[vertex_id]))

    def test_seam_mesh_marks_two_rings_around_each_center_green(self):
        mesh = trimesh.Trimesh(
            vertices=[[x, y, 0] for x in range(11) for y in range(2)],
            faces=[
                face
                for x in range(10)
                for face in ([2*x, 2*x + 1, 2*x + 2], [2*x + 1, 2*x + 3, 2*x + 2])
            ],
            process=False,
        )
        original_vertices = mesh.vertices.copy()
        original_colors = mesh.visual.vertex_colors.copy()
        centers = [np.array([0.05, 0.02, 0]), np.array([9.95, 1.02, 0])]
        with tempfile.TemporaryDirectory() as temporary:
            output_path = Path(temporary) / "apple.obj"
            _write_seam_mesh(mesh, [0, 4, 10], centers, output_path)
            exported = trimesh.load(output_path.with_name("apple_seam.ply"), process=False)

        np.testing.assert_array_equal(exported.vertices, original_vertices)
        green = {0, 1, 2, 3, 4, 17, 18, 19, 20, 21}
        for index, color in enumerate(exported.visual.vertex_colors):
            expected = [0, 255, 0, 255] if index in green else (
                [0, 0, 0, 255] if index == 10 else [255, 255, 255, 255]
            )
            np.testing.assert_array_equal(color, expected)
        np.testing.assert_array_equal(mesh.visual.vertex_colors, original_colors)

    def test_seam_mesh_rejects_invalid_center_coordinates(self):
        mesh = trimesh.creation.box()
        with tempfile.TemporaryDirectory() as temporary:
            output_path = Path(temporary) / "apple.obj"
            with self.assertRaisesRegex(ValueError, "finite top and bottom"):
                _write_seam_mesh(mesh, [], [np.array([0, 0, np.nan]), np.zeros(3)], output_path)
            self.assertFalse(output_path.with_name("apple_seam.ply").exists())

    def test_stores_ordered_uv_boundary_paths(self):
        seams = [[0], [1], [2]]
        seams_uv = [[[0, 1]], [[2, 3]], [[4, 5]]]
        uvs = np.array([
            [0.1, 0.2], [0.3, 0.4], [0.5, 0.6],
            [0.7, 0.8], [0.9, 1.0], [0.2, 0.3],
        ])
        topology = SeamTopology(-1, 1, (1.25, 2.5, -3.0), (-4.0, 5.5, 6.0))
        _store_boundary_paths(topology, seams, seams_uv, uvs)
        self.assertEqual([path["vertex_ids"] for path in topology.boundary_paths], [[0], [1], [0], [2], [1], [2]])
        self.assertEqual([path["uv_coords"][0] for path in topology.boundary_paths],
                         [[0.3, 0.4], [0.7, 0.8], [0.1, 0.2], [0.9, 1.0], [0.5, 0.6], [0.2, 0.3]])


if __name__ == "__main__":
    unittest.main()
