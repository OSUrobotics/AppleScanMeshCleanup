import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from meshlib import mrmeshnumpy as mn
from meshlib import mrmeshpy as mm

from remove_calyx import _pack_with_seams, _patch_stem, patch_stem_and_ring_calyx, remove_calyx
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
            patch_stem_and_ring_calyx(mesh, topology)

        np.testing.assert_array_equal(bottom_center.call_args.kwargs["center"], topology.bottom_point)
        np.testing.assert_array_equal(top_center.call_args.kwargs["center"], topology.top_point)
        self.assertEqual([call.kwargs["height"] for call in curvature.call_args_list], [3.0, 3.0])
        self.assertEqual(set(topology.seams), {"calyx", "stem"})
        self.assertNotEqual(original_calyx_ids, topology.seams["calyx"].vertex_ids)
        self.assertEqual(topology.seams["calyx"].center, (0, -10, 0))
        self.assertAlmostEqual(topology.seams["calyx"].radius, 4.2)
        self.assertEqual(topology.seams["stem"].center, (0.1, 10, 0))
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

    def test_patches_sphere_bottom(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "sphere.obj"
            mm.saveMesh(mn.meshFromFacesVerts(*reversed(_sphere())), str(source))
            output = remove_calyx(source, Path(tmp) / "out.ply")

            patched = mm.loadMesh(str(output))
            calyx = mm.loadMesh(str(Path(tmp) / "out_calyx.ply"))
            self.assertEqual(patched.topology.findNumHoles(), 0)
            self.assertGreater(calyx.topology.numValidFaces(), 0)
            with open(Path(tmp) / "out_calyx_seam.csv") as handle:
                self.assertGreaterEqual(len(list(csv.reader(handle))) - 1, 8)

    def test_missing_input(self):
        with self.assertRaisesRegex(FileNotFoundError, "Input mesh not found"):
            remove_calyx(Path("missing.obj"), Path("out.ply"))


if __name__ == "__main__":
    unittest.main()
