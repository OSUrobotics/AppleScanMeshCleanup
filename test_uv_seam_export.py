import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

from uv_texture import _write_seam_mesh, _write_seams_as_txt_file


class UVSeamExportTests(unittest.TestCase):
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

    def test_writes_centers_after_seam_rows(self):
        seams = [[0], [1], [2]]
        seams_uv = [[[0, 1]], [[2, 3]], [[4, 5]]]
        uvs = np.array([
            [0.1, 0.2], [0.3, 0.4], [0.5, 0.6],
            [0.7, 0.8], [0.9, 1.0], [0.2, 0.3],
        ])
        centers = [np.array([1.25, 2.5, -3.0]), np.array([-4.0, 5.5, 6.0])]

        with tempfile.TemporaryDirectory() as temporary:
            output_path = Path(temporary) / "apple.obj"
            _write_seams_as_txt_file(seams, seams_uv, uvs, centers, output_path)
            with (Path(temporary) / "apple_seam.csv").open(newline="") as file:
                rows = list(csv.reader(file, skipinitialspace=True))

        self.assertEqual(len(rows), 20)
        self.assertEqual(rows[0], ["0", ""])
        self.assertEqual(rows[1], ["0.3", ""])
        self.assertEqual(rows[2], ["0.4", ""])
        self.assertEqual(rows[3:18], [
            ["1", ""], ["0.7", ""], ["0.8", ""],
            ["0", ""], ["0.1", ""], ["0.2", ""],
            ["2", ""], ["0.9", ""], ["1.0", ""],
            ["1", ""], ["0.5", ""], ["0.6", ""],
            ["2", ""], ["0.2", ""], ["0.3", ""],
        ])
        self.assertEqual([float(value) for value in rows[-2]], centers[0].tolist())
        self.assertEqual([float(value) for value in rows[-1]], centers[1].tolist())


if __name__ == "__main__":
    unittest.main()
