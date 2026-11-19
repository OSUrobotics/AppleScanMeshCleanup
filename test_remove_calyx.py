import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from meshlib import mrmeshnumpy as mn
from meshlib import mrmeshpy as mm

from remove_calyx import remove_calyx


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
