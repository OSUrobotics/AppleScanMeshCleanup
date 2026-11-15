import unittest
from unittest.mock import patch

import numpy as np
import trimesh

import uv_texture
from uv_texture import _CAP_CENTERS, _CAP_RADIUS, _UV_PADDING, _check_diamond


def signed_area(points):
    a, b = points[1] - points[0], points[2] - points[0]
    return a[0] * b[1] - a[1] * b[0]


class UVDiamondWarpTests(unittest.TestCase):
    def test_no_overlap_leaves_coordinates_unchanged(self):
        uvs = np.array([[0.5, 0.1], [0.7, 0.5], [0.5, 0.9], [0.3, 0.5]])
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        np.testing.assert_array_equal(_check_diamond(uvs, faces), uvs)

    def test_overlap_inside_triangle_with_no_vertex_in_circle(self):
        for center in _CAP_CENTERS:
            with self.subTest(center=center):
                uvs = center + np.array([[-0.2, -0.15], [0.2, -0.15], [0, 0.2]])
                faces = np.array([[0, 1, 2]])
                original = uvs.copy()
                self.assertTrue(np.all(np.linalg.norm(uvs - center, axis=1) > _CAP_RADIUS))
                result = _check_diamond(uvs, faces)
                np.testing.assert_array_equal(uvs, original)
                diagonal = np.array([1., -1.]) / np.sqrt(2)
                limit = abs(_CAP_CENTERS[0] @ diagonal) - _CAP_RADIUS - _UV_PADDING
                self.assertLessEqual(np.max(np.abs(result @ diagonal)), limit + 1e-12)
                np.testing.assert_allclose(result.sum(axis=1), uvs.sum(axis=1))
                self.assertGreater(signed_area(result) * signed_area(uvs), 0)

    def test_simultaneous_overlap_and_triangle_edges(self):
        uvs = np.array([[0.88, 0.13], [0.95, 0.11], [0.5, 0.53],
                        [0.12, 0.87], [0.05, 0.89]])
        faces = np.array([[0, 1, 2], [3, 4, 2], [0, 2, 3]])
        result = _check_diamond(uvs, faces)
        diagonal = np.array([1., -1.]) / np.sqrt(2)
        limit = abs(_CAP_CENTERS[0] @ diagonal) - _CAP_RADIUS - _UV_PADDING
        self.assertLessEqual(np.max(np.abs(result @ diagonal)), limit + 1e-12)
        for face in faces:
            self.assertGreater(signed_area(result[face]) * signed_area(uvs[face]), 0)
        self.assertLess(abs(result[2, 0] - result[2, 1]), abs(uvs[2, 0] - uvs[2, 1]))

    def test_edge_intersection_without_vertex_or_center_in_triangle(self):
        center = _CAP_CENTERS[0]
        uvs = center + np.array([[-0.2, 0.05], [0.2, 0.05], [0, 0.25]])
        self.assertTrue(np.all(np.linalg.norm(uvs - center, axis=1) > _CAP_RADIUS))
        result = _check_diamond(uvs, np.array([[0, 1, 2]]))
        diagonal = np.array([1., -1.]) / np.sqrt(2)
        limit = abs(_CAP_CENTERS[0] @ diagonal) - _CAP_RADIUS - _UV_PADDING
        self.assertLessEqual(np.max(result @ diagonal), limit + 1e-12)
        self.assertGreater(signed_area(result) * signed_area(uvs), 0)

    def test_no_warp_for_nonintersecting_triangle_in_cap_halfplane(self):
        uvs = np.array([[0.90, 0.50], [0.95, 0.50], [0.95, 0.55]])
        np.testing.assert_array_equal(_check_diamond(uvs, np.array([[0, 1, 2]])), uvs)

    def test_parameterization_applies_diamond_check(self):
        vertices = [[0, 2, 0], [0, -2, 0]]
        for y in (1, -1):
            vertices.extend([[1, y, 0], [0, y, 1], [-1, y, 0], [0, y, -1]])
        faces = []
        for i in range(4):
            a, b = 2 + i, 2 + (i + 1) % 4
            c, d = 6 + i, 6 + (i + 1) % 4
            faces.extend([[0, b, a], [1, c, d], [a, b, c], [b, d, c]])
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        seam_edges = [(2 + i, 2 + (i + 1) % 4) for i in range(4)]
        seam_edges += [(6 + i, 6 + (i + 1) % 4) for i in range(4)]
        seam_edges.append((2, 6))
        with patch("uv_texture._check_diamond", wraps=_check_diamond) as check:
            mapping, output_faces, uvs = uv_texture._parameterize_cut_mesh(mesh, seam_edges, [2, 6])
        check.assert_called_once()
        self.assertEqual(len(mapping), len(uvs))
        self.assertEqual(len(output_faces), len(faces))
        self.assertTrue(np.all((uvs >= 0) & (uvs <= 1)))


if __name__ == "__main__":
    unittest.main()
