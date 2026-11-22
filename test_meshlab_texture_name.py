import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
from meshlib import mrmeshnumpy as mn

from MeshLabCleanup import _process_mesh, _read_mesh_no_textures, _reorient_mesh, downsample_mesh
from seam_topology import SeamInfo, SeamTopology


class MeshLabTextureNameTests(unittest.TestCase):
    def test_read_mesh_ignores_texture_seams_and_triangulates_faces(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mesh.obj"
            path.write_text(
                "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nv 5 5 5\n"
                "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nvn 0 0 1\n"
                "f -5/1/1 -4/2/1 -3/3/1 -2/4/1 # quad\n",
                encoding="utf-8",
            )
            mesh = _read_mesh_no_textures(path)
        np.testing.assert_array_equal(mn.getNumpyVerts(mesh), [
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [5, 5, 5],
        ])
        np.testing.assert_array_equal(mn.getNumpyFaces(mesh.topology), [[0, 1, 2], [0, 2, 3]])

    def test_read_mesh_rejects_invalid_or_discarded_faces(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mesh.obj"
            for face in ("f 0 2 3", "f 1 2 4", "f -4 -2 -1", "f 1 2", "f 1 2 2"):
                with self.subTest(face=face):
                    path.write_text(f"v 0 0 0\nv 1 0 0\nv 0 1 0\n{face}\n", encoding="utf-8")
                    with self.assertRaises(ValueError):
                        _read_mesh_no_textures(path)

    def test_processed_mesh_uses_output_name_for_texture(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.obj"
            output = root / "result.obj"
            script = root / "empty.mlx"
            Image.new("RGB", (4, 4), (220, 30, 10)).save(root / "old_texture.png")
            (root / "source.mtl").write_text(
                "newmtl mat\nKd 1 1 1\nmap_Kd old_texture.png\n", encoding="utf-8"
            )
            source.write_text(
                "mtllib source.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\n"
                "vt 0 0\nvt 1 0\nvt 0 1\nusemtl mat\nf 1/1 2/2 3/3\n",
                encoding="utf-8",
            )
            script.write_text("<FilterScript></FilterScript>\n", encoding="utf-8")

            _process_mesh(source, output, script)

            self.assertTrue(output.is_file())
            self.assertIn("map_Kd result.png", (root / "result.obj.mtl").read_text())
            with Image.open(root / "result.png") as image:
                self.assertEqual(image.getpixel((0, 0))[:3], (220, 30, 10))
            self.assertTrue((root / "old_texture.png").is_file())

    def test_untextured_mesh_still_exports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.obj"
            output = root / "result.obj"
            script = root / "empty.mlx"
            source.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
            script.write_text("<FilterScript></FilterScript>\n", encoding="utf-8")

            _process_mesh(source, output, script)

            self.assertTrue(output.is_file())
            self.assertFalse((root / "result.png").exists())

    def test_processed_mesh_clips_wedge_uvs_preserving_seams(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.obj"
            output = root / "result.obj"
            script = root / "empty.mlx"
            Image.new("RGB", (4, 4), (220, 30, 10)).save(root / "texture.png")
            (root / "source.mtl").write_text("newmtl mat\nmap_Kd texture.png\n")
            source.write_text(
                "mtllib source.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\n"
                "vt -0.2 0.3\nvt 1.2 0.8\nvt 0.2 1.3\n"
                "vt 0.7 -0.4\nvt 0.5 0.6\nvt 1.5 1.1\n"
                "usemtl mat\nf 1/1 2/2 3/3\nf 2/4 4/5 3/6\n"
            )
            script.write_text("<FilterScript></FilterScript>\n")

            _process_mesh(source, output, script)

            lines = output.read_text().splitlines()
            uvs = [list(map(float, line.split()[1:3])) for line in lines if line.startswith("vt ")]
            corners = [
                uvs[int(corner.split("/")[1]) - 1]
                for line in lines if line.startswith("f ")
                for corner in line.split()[1:]
            ]
            np.testing.assert_allclose(corners, [
                [0, 0.3], [1, 0.8], [0.2, 1],
                [0.7, 0], [0.5, 0.6], [1, 1],
            ], atol=1e-6)

    def test_reorient_transforms_mesh_and_topology(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "result.obj"
            source_seam = root / "source_seam_topology.json"
            vertices = np.array([[2, 1, 0], [0, 1, 0], [1, 2, 1], [1, 0, -1]])
            output.write_text(
                "".join(f"v {x} {y} {z}\n" for x, y, z in vertices)
                + "f 1 2 3\nf 1 4 2\nf 1 3 4\nf 2 4 3\n",
                encoding="utf-8",
            )
            topology = SeamTopology(0, 2, (2, 1, 0), (0, 1, 0),
                                    top_shoulder_radius=2, bottom_shoulder_radius=3)
            topology.add_seam("stem", SeamInfo("stem", [0, 2], vertices[[0, 2]].tolist(),
                                               [[[0.1, 0.2]], [[0.3, 0.4]]], (1, 2, 1), 0.5))
            topology.add_seam("side", SeamInfo("side", [], [], [], None, None))
            topology.boundary_paths = [{"vertex_ids": [0], "uv_coords": [[0.1, 0.2]]}]
            original = json.dumps(topology.to_dict())
            source_seam.write_text(original, encoding="utf-8")

            _reorient_mesh(output, source_seam)

            result = SeamTopology.from_dict(json.loads(
                (root / "result_seam_topology.json").read_text(encoding="utf-8")
            ))
            actual_vertices = np.array([
                list(map(float, line.split()[1:4])) for line in output.read_text().splitlines()
                if line.startswith("v ")
            ])
            np.testing.assert_allclose(result.top_point, [0, 0, 0], atol=1e-6)
            np.testing.assert_allclose(result.bottom_point, [0, -2, 0], atol=1e-6)
            np.testing.assert_allclose(result.seams["stem"].vertex_locations,
                                       actual_vertices[[0, 2]], atol=1e-6)
            np.testing.assert_allclose(result.seams["stem"].center, actual_vertices[2], atol=1e-6)
            self.assertIsNone(result.seams["side"].center)
            self.assertEqual(result.seams["stem"].vertex_ids, [0, 2])
            self.assertEqual(result.seams["stem"].uv_coords, topology.seams["stem"].uv_coords)
            self.assertEqual(result.boundary_paths, topology.boundary_paths)
            self.assertEqual(result.seams["stem"].radius, 0.5)
            self.assertEqual(result.top_shoulder_radius, 2)
            self.assertAlmostEqual(result.ymin, actual_vertices[:, 1].min())
            self.assertAlmostEqual(result.ymax, actual_vertices[:, 1].max())
            self.assertEqual(source_seam.read_text(encoding="utf-8"), original)

    def test_reorient_rejects_coincident_endpoints_without_modifying_mesh(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "result.obj"
            source = root / "source_seam_topology.json"
            original = "v 0 1 0\nv 1 0 0\nv 0 -1 0\nf 1 2 3\n"
            output.write_text(original)
            source.write_text(json.dumps(SeamTopology(-1, 1, (0, 1, 0), (0, 1, 0)).to_dict()))
            with self.assertRaisesRegex(ValueError, "non-zero axis"):
                _reorient_mesh(output, source)
            self.assertEqual(output.read_text(), original)
            self.assertFalse((root / "result_seam_topology.json").exists())

    def test_topology_transform_rotates_then_translates(self):
        topology = SeamTopology(-1, 1, (1, 2, 3), (0, 0, 0))
        topology.add_seam("stem", SeamInfo("stem", [2], [[2, 1, 0]], [[[0.2, 0.3]]],
                                           (0, 1, 0), 2))
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        topology.transform([10, 20, 30], rotation)
        np.testing.assert_allclose(topology.top_point, [8, 21, 33])
        np.testing.assert_allclose(topology.bottom_point, [10, 20, 30])
        np.testing.assert_allclose(topology.seams["stem"].vertex_locations, [[9, 22, 30]])
        np.testing.assert_allclose(topology.seams["stem"].center, [9, 20, 30])
        with self.assertRaisesRegex(ValueError, "finite"):
            topology.transform([0, 0, 0], np.full((3, 3), np.nan))

    def test_downsample_writes_companion_reoriented_topology(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir, final_dir = root / "NormalizedTexture", root / "Final"
            source_dir.mkdir()
            final_dir.mkdir()
            source = source_dir / "Apple1.obj"
            source.write_text(
                "mtllib Apple1.mtl\nv 2 0 0\nv 0 0 0\nv 1 1 0\n"
                "vt 0 0\nvt 1 0\nvt 0 1\nusemtl apple\nf 1/1 2/2 3/3\n"
            )
            source.with_suffix(".mtl").write_text("newmtl apple\nKd 1 1 1\nmap_Kd texture.png\n")
            Image.new("RGB", (8, 8), (20, 40, 60)).save(source_dir / "texture.png")
            topology = SeamTopology(0, 1, (2, 0, 0), (0, 0, 0))
            source.with_name("Apple1_seam_topology.json").write_text(json.dumps(topology.to_dict()))
            script = root / "downsample_fullres.mlx"
            script.write_text("<FilterScript></FilterScript>\n")
            with (patch("MeshLabCleanup.print_mesh_topology"),
                  patch("MeshLabCleanup.flood_fill_texture_map",
                        side_effect=lambda texture, mesh, seam, padded: Image.open(texture).save(padded))):
                result = downsample_mesh(source, final_dir / "Apple1.obj", [script])
            output_seam = result.with_name(result.stem + "_seam_topology.json")
            transformed = SeamTopology.from_dict(json.loads(output_seam.read_text()))
            np.testing.assert_allclose(transformed.top_point, [0, 0, 0], atol=1e-6)
            np.testing.assert_allclose(transformed.bottom_point, [0, -2, 0], atol=1e-6)
            self.assertTrue(result.is_file())


if __name__ == "__main__":
    unittest.main()
