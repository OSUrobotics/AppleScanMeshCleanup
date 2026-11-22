import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from meshlib import mrmeshnumpy as mn
from PIL import Image

from align_meshes import _mesh
from make_apple_geometry import _create_apple
from MeshLabCleanup import reparameterize_and_remove_calyx
from reconstruct_uv_by_projection import reconstruct_uv_by_projection
from remove_calyx import remove_calyx
from seam_topology import SeamInfo, SeamTopology


class ReconstructUVTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source.obj"
        self.output = self.root / "output" / "result.obj"
        self.vertices = np.array([
            [0, 2, 0], [1, 1, 0], [0, 1, 1], [-1, 1, 0], [0, 1, -1],
            [1, -1, 0], [0, -1, 1], [-1, -1, 0], [0, -1, -1], [0, -2, 0],
        ], dtype=np.float32)
        faces, face_uvs = [], []
        top_uv = np.array([[0.8, 0.8], [0.9, 0.8], [0.8, 0.9], [0.7, 0.8], [0.8, 0.7]])
        bottom_uv = top_uv - [0.6, 0.6]
        for i in range(4):
            j = (i + 1) % 4
            faces.append([0, 1 + j, 1 + i])
            face_uvs.append([top_uv[0], top_uv[1 + j], top_uv[1 + i]])
            faces.append([9, 5 + i, 5 + j])
            face_uvs.append([bottom_uv[0], bottom_uv[1 + i], bottom_uv[1 + j]])
            u, next_u = 0.1 + 0.2 * i, 0.1 + 0.2 * (i + 1)
            faces.extend([[1 + i, 1 + j, 5 + i], [1 + j, 5 + j, 5 + i]])
            face_uvs.extend([[[u, 0.6], [next_u, 0.6], [u, 0.4]],
                             [[next_u, 0.6], [next_u, 0.4], [u, 0.4]]])
        self.faces = np.asarray(faces, dtype=np.int32)
        self.face_uvs = np.asarray(face_uvs)
        self.topology = SeamTopology(-2, 2, (0, 2, 0), (0, -2, 0))
        self.topology.add_seam("calyx", SeamInfo("calyx", [5, 6, 7, 8],
                                                 self.vertices[[5, 6, 7, 8]].tolist(), [],
                                                 (0, -2, 0), 1))
        Image.new("RGB", (4, 4), (30, 60, 90)).save(self.root / "texture.png")
        (self.root / "source.mtl").write_text("newmtl apple\nKd 1 1 1\nmap_Kd texture.png\n")
        with self.source.open("w") as stream:
            stream.write("mtllib source.mtl\n")
            for point in self.vertices:
                stream.write("v " + " ".join(map(str, point)) + "\n")
            for uv in self.face_uvs.reshape(-1, 2):
                stream.write(f"vt {uv[0]} {uv[1]}\n")
            stream.write("usemtl apple\n")
            for index, face in enumerate(self.faces):
                stream.write("f " + " ".join(f"{v + 1}/{3 * index + c + 1}"
                                              for c, v in enumerate(face)) + "\n")

    def test_reconstructs_uvs_seams_and_texture(self):
        mesh = mn.meshFromFacesVerts(self.faces, self.vertices)
        reconstruct_uv_by_projection(self.source, mesh, self.topology, self.output)
        _, vertices, faces, uvs, uv_faces, material = _mesh(self.output)
        np.testing.assert_allclose(vertices, self.vertices)
        np.testing.assert_allclose(uvs[uv_faces], self.face_uvs, atol=2e-6)
        self.assertEqual(len(faces), len(self.faces))
        shared = [int(uv) for face, face_uv in zip(faces, uv_faces)
                  for vertex, uv in zip(face, face_uv) if vertex == 2]
        self.assertEqual(len(set(shared)), 2)
        self.assertEqual(set(self.topology.seams), {"top", "bottom", "side", "calyx"})
        self.assertEqual(set(self.topology.seams["top"].vertex_ids), {1, 2, 3, 4})
        self.assertEqual(set(self.topology.seams["bottom"].vertex_ids), {5, 6, 7, 8})
        self.assertEqual(self.topology.seams["side"].vertex_ids, [1, 5])
        self.assertEqual(len(self.topology.boundary_paths), 6)
        self.assertTrue(all(copies for copies in self.topology.seams["calyx"].uv_coords))
        self.assertIn("map_Kd result.png", material.read_text())
        self.assertEqual((self.output.parent / "result.png").read_bytes(),
                         (self.root / "texture.png").read_bytes())

    def test_projects_new_vertex_onto_source_triangle(self):
        triangle = self.faces[2]
        new_vertex = self.vertices[triangle].mean(axis=0)
        vertices = np.vstack((self.vertices, new_vertex)).astype(np.float32)
        replacement = np.array([[triangle[0], triangle[1], 10],
                                [triangle[1], triangle[2], 10],
                                [triangle[2], triangle[0], 10]], dtype=np.int32)
        faces = np.vstack((self.faces[:2], replacement, self.faces[3:]))
        mesh = mn.meshFromFacesVerts(faces, vertices)
        reconstruct_uv_by_projection(self.source, mesh, self.topology, self.output)
        _, _, output_faces, uvs, uv_faces, _ = _mesh(self.output)
        projected = [uvs[uv] for face, uv_face in zip(output_faces, uv_faces)
                     for vertex, uv in zip(face, uv_face) if vertex == 10]
        self.assertEqual(len(projected), 3)
        np.testing.assert_allclose(projected, [self.face_uvs[2].mean(axis=0)] * 3, atol=2e-6)

    def test_rejects_source_overwrite(self):
        mesh = mn.meshFromFacesVerts(self.faces, self.vertices)
        with self.assertRaisesRegex(ValueError, "must not overwrite"):
            reconstruct_uv_by_projection(self.source, mesh, self.topology, self.source)

    def test_rejects_missing_uv_charts(self):
        lines = self.source.read_text().splitlines()
        self.source.write_text("\n".join("vt 0.5 0.5" if line.startswith("vt ") else line
                                         for line in lines) + "\n")
        mesh = mn.meshFromFacesVerts(self.faces, self.vertices)
        with self.assertRaisesRegex(ValueError, "three connected source UV charts"):
            reconstruct_uv_by_projection(self.source, mesh, self.topology, self.output)
        self.assertFalse(self.output.exists())

    def test_existing_apple_and_caller_preserve_welded_geometry(self):
        source = self.root / "apple.obj"
        _create_apple(15, source)
        _, original_vertices, original_faces, _, _, _ = _mesh(source)
        output = self.root / "final" / "apple.obj"
        with (patch("MeshLabCleanup.print_mesh_topology"),
              patch("MeshLabCleanup.patch_stem_and_ring_calyx",
                    side_effect=lambda mesh, **kwargs: mesh),
              patch("MeshLabCleanup.remove_calyx") as remove):
            self.assertEqual(reparameterize_and_remove_calyx(source, output), output)
        intermediate = output.parent / "FullResParameterize" / output.name
        _, vertices, faces, _, _, _ = _mesh(intermediate)
        self.assertEqual(len(vertices), len(original_vertices))
        self.assertEqual(len(faces), len(original_faces))
        topology = json.loads(intermediate.with_name("apple_seam_topology.json").read_text())
        self.assertEqual(set(topology["seams"]), {"top", "bottom", "side", "calyx"})
        self.assertEqual(len(topology["boundary_paths"]), 6)
        remove.assert_called_once_with(input_path=intermediate, output_path=output,
                                       seam_path=intermediate.parent)
        result = remove_calyx(intermediate, output, seam_path=intermediate.parent)
        self.assertTrue(result.is_file())
        self.assertTrue(output.with_name("apple_calyx.obj").is_file())

    def test_patched_apple_reconstructs_and_removes_calyx(self):
        source = self.root / "apple.obj"
        _create_apple(24, source)
        lines = source.read_text().splitlines()
        source.write_text("\n".join(
            "v " + " ".join(str(float(value) * 20) for value in line.split()[1:4])
            if line.startswith("v ") else line for line in lines
        ) + "\n")
        seam_path = source.with_name("apple_seam_topology.json")
        topology = json.loads(seam_path.read_text())
        topology["ymin"] *= 20
        topology["ymax"] *= 20
        for key in ("top_point", "bottom_point"):
            topology[key] = [value * 20 for value in topology[key]]
        topology["seams"] = {}
        topology["boundary_paths"] = []
        seam_path.write_text(json.dumps(topology))
        output = self.root / "final" / "apple.obj"
        with (patch("MeshLabCleanup.print_mesh_topology"),
              patch("remove_calyx._high_curvature_vertices", return_value=[]),
              patch("remove_calyx._bottom_center", return_value=(1.5, 2.0, 3.0)),
              patch("remove_calyx._top_center",
                    return_value=(np.array([0.1, 10, 0]), 4.0, 3.0, 0, [])),
              contextlib.redirect_stdout(io.StringIO())):
            result = reparameterize_and_remove_calyx(source, output)
        self.assertEqual(result, output)
        self.assertTrue(output.is_file())
        self.assertTrue(output.with_name("apple_calyx.obj").is_file())


if __name__ == "__main__":
    unittest.main()
