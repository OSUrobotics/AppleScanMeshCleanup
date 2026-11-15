import unittest

import numpy as np

from uv_texture import _pack_cap_uvs


class UVCapPackingTests(unittest.TestCase):
    def test_scales_about_median_to_maximum_radius(self):
        uvs = np.array([[0, 0], [1, 0], [2, 1], [3, 2], [12, 4]], dtype=float)
        original = uvs.copy()
        for center in (np.array([0.9 - 3 / 1024, 0.1 + 3 / 1024]),
                       np.array([0.1 + 3 / 1024, 0.9 - 3 / 1024])):
            with self.subTest(center=center):
                result = _pack_cap_uvs(uvs, center, 1)
                np.testing.assert_allclose(np.median(result, axis=0), center)
                self.assertAlmostEqual(np.linalg.norm(result - center, axis=1).max(), 0.1)
                offsets = uvs - np.median(uvs, axis=0)
                scale = 0.1 / np.linalg.norm(offsets, axis=1).max()
                np.testing.assert_allclose(result, center + offsets * scale)
                self.assertTrue(np.all((result >= 3 / 1024 - 1e-12)
                                       & (result <= 1 - 3 / 1024 + 1e-12)))
        np.testing.assert_array_equal(uvs, original)

    def test_rejects_degenerate_and_nonfinite_charts(self):
        for uvs in (np.array([[1., 1.], [1., 2.]]),
                    np.array([[0., 0.], [1., np.nan]])):
            with self.subTest(uvs=uvs), self.assertRaisesRegex(ValueError, "UV extent"):
                _pack_cap_uvs(uvs, np.array([0.5, 0.5]), 1)


if __name__ == "__main__":
    unittest.main()
