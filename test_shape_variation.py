import tempfile
import unittest
from pathlib import Path

import numpy as np

import shape_variation


class ShapeVariationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.mesh_dir = self.root / "meshes"
        self.mesh_dir.mkdir()
        self.mesh_text = (
            "# simple triangle\n"
            "o sample\n"
            "v 0 0 0\n"
            "v 1 0 0\n"
            "v 0 1 0\n"
            "vt 0 0\n"
            "vt 1 0\n"
            "vt 0 1\n"
            "f 1/1 2/2 3/3\n"
        )
        for name, shift in (("a.obj", 0.0), ("b.obj", 0.2), ("c.obj", 0.4)):
            text = self.mesh_text.replace("v 1 0 0", f"v 1 {shift} 0")
            (self.mesh_dir / name).write_text(text, encoding="utf-8")
        self.model_path = self.root / "model.npz"

    def test_computes_and_saves_reconstructible_pca(self):
        pca = shape_variation.compute_shape_variation(self.mesh_dir, self.model_path)

        self.assertEqual(pca.components.shape, (1, 9))
        samples = np.stack(
            [shape_variation._read_obj(path)[1].reshape(-1) for path in sorted(self.mesh_dir.glob("*.obj"))]
        )
        np.testing.assert_allclose(
            pca.mean + pca.weights @ pca.components,
            samples,
            atol=1e-12,
        )
        with np.load(self.model_path, allow_pickle=False) as archive:
            np.testing.assert_allclose(archive["weights"], pca.weights)
            self.assertEqual(archive["template_name"].item(), "a.obj")

    def test_generates_repeatable_obj_variations_from_saved_model(self):
        shape_variation.compute_shape_variation(self.mesh_dir, self.model_path)
        first = shape_variation.generate_random_variations(
            self.model_path, 3, self.root / "first", seed=42
        )
        second = shape_variation.generate_random_variations(
            self.model_path, 3, self.root / "second", seed=42
        )

        self.assertEqual([path.name for path in first], [f"variation_{i:04d}.obj" for i in range(1, 4)])
        self.assertEqual([path.read_text() for path in first], [path.read_text() for path in second])
        self.assertIn("f 1/1 2/2 3/3", first[0].read_text())
        self.assertFalse(
            np.array_equal(
                shape_variation._read_obj(first[0])[1],
                shape_variation._read_obj(first[1])[1],
            )
        )

    def test_rejects_meshes_with_different_topology(self):
        (self.mesh_dir / "c.obj").write_text(
            self.mesh_text.replace("f 1/1 2/2 3/3", "f 1/1 3/3 2/2"),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "same vertex count and face topology"):
            shape_variation.compute_shape_variation(self.mesh_dir, self.model_path)

    def test_accepts_negative_obj_face_indices(self):
        mesh = self.root / "negative.obj"
        mesh.write_text(
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\n",
            encoding="utf-8",
        )
        _, _, faces = shape_variation._read_obj(mesh)
        self.assertEqual(faces, ((0, 1, 2),))

    def test_rejects_invalid_variation_count(self):
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            shape_variation.generate_random_variations(
                shape_variation.ShapePCA(
                    mean=np.zeros(0),
                    components=np.zeros((0, 0)),
                    explained_variance=np.zeros(0),
                    weights=np.zeros((0, 0)),
                    mesh_names=(),
                    template_lines=(),
                    template_name="",
                ),
                -1,
                self.root / "output",
            )

    def test_existing_outputs_require_explicit_replacement(self):
        shape_variation.compute_shape_variation(self.mesh_dir, self.model_path)
        with self.assertRaises(FileExistsError):
            shape_variation.compute_shape_variation(self.mesh_dir, self.model_path)

        output_dir = self.root / "variations"
        shape_variation.generate_random_variations(self.model_path, 1, output_dir, seed=7)
        with self.assertRaises(FileExistsError):
            shape_variation.generate_random_variations(self.model_path, 1, output_dir, seed=7)
        shape_variation.generate_random_variations(
            self.model_path, 1, output_dir, seed=7, replace_existing=True
        )


if __name__ == "__main__":
    unittest.main()
