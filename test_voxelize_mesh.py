import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from meshlib import mrmeshnumpy as mn
from meshlib import mrmeshpy as mm

from MeshLabCleanup import voxelize_mesh
from seam_topology import SeamInfo, SeamTopology


class VoxelizeMeshTests(unittest.TestCase):
    def test_initializes_seam_topology_from_voxelized_mesh_and_centers(self):
        mesh = mm.makeUVSphere(10.0, 16, 16)
        vertices = mn.getNumpyVerts(mesh).copy()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "input" / "apple1.obj"
            centers_dir = root / "BlenderStemRemoved"
            centers_dir.mkdir()
            (centers_dir / "apple_1.txt").write_text(
                "1,9,2\n3,-8,4\n", encoding="utf-8"
            )
            output_path = root / "output" / "apple1.obj"
            output_path.parent.mkdir()
            def record_uvs(*, input_path, output_path, seam_topology):
                seam_topology.add_seam("top", SeamInfo("top", [0], [[0, 0, 0]], [[[0.1, 0.2]]], None, None))

            with (
                patch("MeshLabCleanup.print_mesh_topology"),
                patch("MeshLabCleanup.mm.loadMesh", return_value=mesh),
                patch("MeshLabCleanup.mm.generalOffsetMesh", return_value=mesh),
                patch("MeshLabCleanup._remove_zero_area_face"),
                patch("MeshLabCleanup._remove_unconnected"),
                patch("MeshLabCleanup.mm.relaxKeepVolume"),
                patch("MeshLabCleanup.patch_stem_and_ring_calyx") as patch_rings,
                patch("MeshLabCleanup.mm.saveMesh"),
                patch("MeshLabCleanup.uv_texture", side_effect=record_uvs) as texture,
                patch("MeshLabCleanup.SeamTopology", wraps=SeamTopology) as topology_class,
            ):
                voxelize_mesh(input_path, output_path)
            saved = json.loads(output_path.with_name("apple1_seam_topology.json").read_text())
            self.assertEqual(saved["seams"]["top"]["uv_coords"], [[[0.1, 0.2]]])

        topology_class.assert_called_once()
        self.assertEqual(topology_class.call_args.kwargs["ymin"], float(vertices[:, 1].min()))
        self.assertEqual(topology_class.call_args.kwargs["ymax"], float(vertices[:, 1].max()))
        self.assertEqual(topology_class.call_args.kwargs["top_point"], (1.0, 9.0, 2.0))
        self.assertEqual(topology_class.call_args.kwargs["bottom_point"], (3.0, -8.0, 4.0))
        self.assertIs(patch_rings.call_args.kwargs["mesh"], mesh)
        self.assertIsInstance(patch_rings.call_args.kwargs["seam_topology"], SeamTopology)
        self.assertEqual(patch_rings.call_args.kwargs["seam_topology"].top_point, (1.0, 9.0, 2.0))
        self.assertIs(texture.call_args.kwargs["seam_topology"], patch_rings.call_args.kwargs["seam_topology"])


if __name__ == "__main__":
    unittest.main()
