import csv
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

import fix_texture_maps as fixing
from seam_topology import SeamInfo, SeamTopology

_legacy_available = (
    importlib.util.find_spec("legacy_seams") is not None
    and all(hasattr(fixing, name) for name in (
        "_copy_projected_face", "_fill_seam_pixels", "_project_patch", "_seam_edges", "_seam_patches"
    ))
)
if _legacy_available:
    from fix_texture_maps import (
        _copy_projected_face, _fill_seam_pixels, _project_patch, _seam_edges, _seam_patches,
        flood_fill_texture_map,
    )
    from MeshLabCleanup import downsample_mesh
    import main
    from normalize_uv_projection import _read_obj
    from legacy_seams import _read_legacy_seam_data


@unittest.skipUnless(_legacy_available, "Legacy seam-filling API and CSV reader are not present.")
class FillTextureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.mesh = self.root / "Apple1.obj"
        self.seams = self.root / "Apple1_seam.csv"
        self.texture = self.root / "texture.png"
        self.output = self.root / "filled.png"

        top = [(0.1, 0.9), (0.3, 0.9), (0.2, 0.7)]
        bottom = [(0.7, 0.3), (0.9, 0.3), (0.8, 0.1)]
        body_top = [(0.2, 0.65), (0.4, 0.65), (0.6, 0.65), (0.8, 0.65)]
        body_bottom = [(0.2, 0.35), (0.4, 0.35), (0.6, 0.35), (0.8, 0.35)]
        points = top + bottom + body_top + body_bottom + [(0.2, 0.83), (0.8, 0.23)]
        vertices = [(0, 1, 0), (1, 1, 0), (0.5, 1, 1),
                    (0, -1, 0), (1, -1, 0), (0.5, -1, 1),
                    (0.5, 1.2, 0.33), (0.5, -1.2, 0.33)]
        vertex_ids = [0, 1, 2, 3, 4, 5, 0, 1, 2, 0, 3, 4, 5, 3, 6, 7]
        faces = [(i, (i + 1) % 3, 14) for i in range(3)]
        faces += [(3 + i, 3 + (i + 1) % 3, 15) for i in range(3)]
        for i in range(3):
            faces.extend([(6 + i, 7 + i, 10 + i), (7 + i, 11 + i, 10 + i)])
        with self.mesh.open("w", encoding="utf-8") as file:
            file.write("mtllib Apple1.mtl\n")
            for x, y, z in vertices:
                file.write(f"v {x} {y} {z}\n")
            for u, v in points:
                file.write(f"vt {u} {v}\n")
            for face in faces:
                file.write("f " + " ".join(f"{vertex_ids[i] + 1}/{i + 1}" for i in face) + "\n")

        # The real exporter omits the first vertex from the two circular rings.
        groups = [
            ([1, 2], top[1:]),
            ([4, 5], bottom[1:]),
            ([1, 2], body_top[1:3]),
            ([0, 3], [body_top[3], body_bottom[3]]),
            ([4, 5], body_bottom[1:3]),
            ([3, 0], [body_bottom[0], body_top[0]]),
        ]
        with self.seams.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            for ids, uv in groups:
                writer.writerow(ids)
                writer.writerow([point[0] for point in uv])
                writer.writerow([point[1] for point in uv])

        image = Image.new("RGB", (101, 101), "white")
        draw = ImageDraw.Draw(image)
        xy = lambda p: (p[0] * 100, (1 - p[1]) * 100)
        draw.polygon([xy(p) for p in top], fill=(200, 10, 10))
        draw.polygon([xy(p) for p in bottom], fill=(10, 200, 10))
        draw.polygon([xy(p) for p in (body_top[0], body_top[-1], body_bottom[-1], body_bottom[0])],
                     fill=(10, 10, 200))
        image.save(self.texture)

    def test_fill_uses_opposite_seam_charts_and_preserves_existing_pixels(self):
        flood_fill_texture_map(self.texture, self.mesh, self.seams, self.output)
        with Image.open(self.texture) as original, Image.open(self.output) as filled:
            before, after = np.array(original), np.array(filled)
        nonwhite = np.any(before != 255, axis=2)
        np.testing.assert_array_equal(after[nonwhite], before[nonwhite])
        np.testing.assert_array_equal(after[11, 30], [255, 255, 255])
        np.testing.assert_array_equal(after[11, 10], [255, 255, 255])
        np.testing.assert_array_equal(after[19, 72], [200, 10, 10])
        np.testing.assert_array_equal(after[66, 30], [10, 200, 10])
        np.testing.assert_array_equal(after[0, 0], [255, 255, 255])

    def test_fill_accepts_topology_json_with_legacy_equivalent_paths(self):
        topology = SeamTopology(-1, 1, (0, 1, 0), (0, -1, 0))
        topology.set_boundary_paths(_read_legacy_seam_data(self.seams)[0])
        topology_path = self.root / "Apple1_seam_topology.json"
        topology_path.write_text(json.dumps(topology.to_dict()), encoding="utf-8")
        legacy_output = self.root / "legacy.png"
        flood_fill_texture_map(self.texture, self.mesh, self.seams, legacy_output)
        flood_fill_texture_map(self.texture, self.mesh, topology_path, self.output)
        with Image.open(legacy_output) as legacy, Image.open(self.output) as result:
            np.testing.assert_array_equal(np.asarray(result), np.asarray(legacy))

    def test_transparency_and_in_place_output(self):
        with Image.open(self.texture) as image:
            rgba = np.array(image.convert("RGBA"))
        rgba[10, 20] = [3, 4, 5, 0]
        rgba[50, 50] = [3, 4, 5, 0]
        Image.fromarray(rgba).save(self.texture)
        flood_fill_texture_map(self.texture, self.mesh, self.seams, self.texture)
        with Image.open(self.texture) as result:
            self.assertEqual(result.getpixel((20, 10)), (200, 10, 10, 255))
            self.assertEqual(result.getpixel((50, 50)), (3, 4, 5, 0))
            self.assertEqual(result.getpixel((15, 20)), (200, 10, 10, 255))

    def test_rejects_mismatched_seam_and_uncolored_chart(self):
        self.seams.write_text(self.seams.read_text().replace("0.3", "0.31", 1))
        with self.assertRaisesRegex(ValueError, "does not match"):
            flood_fill_texture_map(self.texture, self.mesh, self.seams, self.output)
        self.assertFalse(self.output.exists())

    def test_rejects_seam_uvs_with_different_mesh_vertices(self):
        contents = self.mesh.read_text()
        self.mesh.write_text(contents.replace("f 1/7 ", "f 2/7 "))
        with self.assertRaisesRegex(ValueError, "same mesh vertices"):
            flood_fill_texture_map(self.texture, self.mesh, self.seams, self.output)
        self.assertFalse(self.output.exists())

    def test_does_not_invent_colors_for_white_source_chart(self):
        with Image.open(self.texture) as source:
            image = np.array(source)
        image[np.all(image == [200, 10, 10], axis=2)] = 255
        Image.fromarray(image).save(self.texture)
        flood_fill_texture_map(self.texture, self.mesh, self.seams, self.output)
        with Image.open(self.output) as output:
            self.assertFalse(np.any(np.all(np.array(output) == [200, 10, 10], axis=2)))

    def test_grows_exactly_five_vertex_rings(self):
        faces = []
        for i in range(8):
            a, b = 2 * i, 2 * i + 1
            faces.extend([(a, b, a + 2), (b, a + 3, a + 2)])
        faces = np.array(faces)
        seam = [((100, 101, 102, 1), (0, 1, 2, 0))]
        [(face_ids, pins, _)] = list(_seam_patches(faces, seam))
        self.assertEqual(pins, {0: 100, 1: 101})
        self.assertEqual(face_ids, list(range(10)))

    def test_excludes_corner_faces_from_all_seam_patches(self):
        _, uvs, faces, _ = _read_obj(self.mesh)
        _, edges, excluded = _seam_edges(uvs, faces, _read_legacy_seam_data(self.seams)[0])
        self.assertEqual(excluded, {(6, 7, 10), (9, 12, 13)})
        patches = list(_seam_patches(faces, edges, excluded))
        self.assertTrue(patches)
        self.assertTrue(any(8 in face_ids for face_ids, _, _ in patches))
        self.assertTrue(any(0 in face_ids for face_ids, _, _ in patches))
        for face_ids, pins, seam in patches:
            self.assertTrue(all(tuple(sorted(faces[face_id])) not in excluded for face_id in face_ids))
            source = seam[1]
            self.assertTrue(any(set(source[:2]).issubset(faces[face_id]) for face_id in face_ids))
            self.assertTrue(set(pins).issubset(set(faces[face_ids].ravel())))

    def test_corner_face_does_not_seed_or_bridge_neighborhoods(self):
        faces = np.array([[0, 1, 2], [1, 2, 3], [3, 4, 5],
                          [100, 101, 102]])
        seam = [((100, 101, 102, 1), (0, 1, 2, 0))]
        self.assertEqual(list(_seam_patches(faces, seam, {(0, 1, 2)})), [])
        patches = list(_seam_patches(faces, seam, {(1, 2, 3)}))
        self.assertEqual(len(patches), 1)
        self.assertEqual(patches[0][0], [0])

    def test_fill_seam_pixels_averages_valid_neighbors_even_inside_chart(self):
        image = np.full((5, 6, 4), 255, dtype=np.uint8)
        image[1, 2] = [20, 40, 60, 255]
        image[3, 2] = [60, 80, 100, 255]
        image[2, 1] = [2, 3, 4, 0]
        image[2, 4] = [100, 20, 30, 255]
        coordinates = np.array([[2, 2], [4, 2]])
        edges = [((0, 1, 0, 0), (0, 1, 0, 0))]
        self.assertEqual(_fill_seam_pixels(image, coordinates, edges), 0)
        np.testing.assert_array_equal(image[2, 2], [40, 60, 80, 255])
        np.testing.assert_array_equal(image[2, 3], [60, 47, 63, 255])
        np.testing.assert_array_equal(image[2, 4], [100, 20, 30, 255])
        np.testing.assert_array_equal(image[2, 1], [2, 3, 4, 0])

    def test_seam_missing_without_valid_neighbors_remains_and_is_reported(self):
        image = np.full((3, 5, 3), 255, dtype=np.uint8)
        coordinates = np.array([[0, 0], [4, 0]])
        edges = [((0, 1, 0, 0), (0, 1, 0, 0))]
        self.assertEqual(_fill_seam_pixels(image, coordinates, edges), 5)
        self.assertTrue(np.all(image == 255))

    def test_fill_seam_propagates_through_consecutive_missing_pixels(self):
        image = np.full((3, 7, 3), 255, dtype=np.uint8)
        image[1, 0] = [20, 40, 60]
        coordinates = np.array([[0, 1], [6, 1]])
        edges = [((0, 1, 0, 0), (0, 1, 0, 0))]
        self.assertEqual(_fill_seam_pixels(image, coordinates, edges), 0)
        np.testing.assert_array_equal(image[1], np.tile([20, 40, 60], (7, 1)))
        np.testing.assert_array_equal(image[0], np.full((7, 3), 255))

    def test_long_projected_triangles_are_not_rasterized(self):
        source_face = np.array([[0., 0.], [0.1, 0.], [0., 0.1]])
        long_face = np.array([[0., 0.], [0.251, 0.], [0., 0.1]])
        boundary_face = np.array([[0., 0.], [0.25, 0.], [0.125, 0.125]])
        self.assertGreater(np.linalg.norm(long_face[1] - long_face[0]), 0.25)
        self.assertEqual(np.linalg.norm(boundary_face[1] - boundary_face[0]), 0.25)
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        faces = np.array([[0, 1, 2]])
        edge = ((0, 1, 2, 0), (0, 1, 2, 0))
        projected = {i: point for i, point in enumerate(long_face)}
        with (patch("fix_texture_maps._read_obj", return_value=([], source_face, faces, self.root / "mesh.mtl")),
              patch("fix_texture_maps._mesh_geometry", return_value=(vertices, faces)),
              patch("fix_texture_maps._read_legacy_seam_data", return_value=([], None)),
              patch("fix_texture_maps._seam_edges", return_value=(None, [edge], set())),
              patch("fix_texture_maps._seam_patches", return_value=[([0], {0: 0, 1: 1}, edge)]),
              patch("fix_texture_maps._project_patch", return_value=(projected, 1)),
              patch("fix_texture_maps._copy_projected_face") as copy,
              patch("fix_texture_maps._fill_seam_pixels", return_value=0),
              self.assertWarnsRegex(RuntimeWarning, "UV edges longer than 0.25")):
            flood_fill_texture_map(self.texture, self.mesh, self.seams, self.output)
        copy.assert_not_called()
        projected = {i: point for i, point in enumerate(boundary_face)}
        with (patch("fix_texture_maps._read_obj", return_value=([], source_face, faces, self.root / "mesh.mtl")),
              patch("fix_texture_maps._mesh_geometry", return_value=(vertices, faces)),
              patch("fix_texture_maps._read_legacy_seam_data", return_value=([], None)),
              patch("fix_texture_maps._seam_edges", return_value=(None, [edge], set())),
              patch("fix_texture_maps._seam_patches", return_value=[([0], {0: 0, 1: 1}, edge)]),
              patch("fix_texture_maps._project_patch", return_value=(projected, 1)),
              patch("fix_texture_maps._copy_projected_face") as copy,
              patch("fix_texture_maps._fill_seam_pixels", return_value=0)):
            flood_fill_texture_map(self.texture, self.mesh, self.seams, self.output)
        copy.assert_called_once()

    def test_pins_seam_and_reprojects_geometry_with_texture_detail(self):
        vertices = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0.5, -1, 0]])
        vertex_faces = np.array([[0, 1, 3], [0, 1, 2]])
        faces = np.array([[0, 1, 2], [3, 4, 5]])
        uvs = np.array([[0.2, 0.2], [0.4, 0.2], [0.3, 0.1],
                        [0.6, 0.5], [0.8, 0.5], [0.6, 0.7]])
        pins = {3: 0, 4: 1}
        seam = ((0, 1, 2, 0), (3, 4, 5, 1))
        projected, orientation = _project_patch(
            vertices, vertex_faces, faces, uvs, [1], pins, seam,
            {0: {0}, 1: {1}, 2: {3}, 3: {0}, 4: {1}, 5: {2}},
        )
        self.assertEqual(orientation, 1)
        np.testing.assert_allclose(projected[3], uvs[0])
        np.testing.assert_allclose(projected[4], uvs[1])
        np.testing.assert_allclose(projected[5], [0.2, 0.4], atol=1e-8)
        source = np.full((101, 101, 3), 255, dtype=np.uint8)
        y, x = np.indices((101, 101))
        source[..., 0] = x
        source[..., 1] = y
        source[..., 2] = 50
        result = np.full_like(source, 255)
        occupied = np.zeros((101, 101), dtype=bool)
        source_missing = np.ones((101, 101), dtype=bool)
        source_missing[30:51, 60:81] = False
        source_missing[65:81, 20:41] = True
        _copy_projected_face(result, source, source_missing, occupied,
                             np.array([projected[i] for i in (3, 4, 5)]), uvs[faces[1]].copy())
        np.testing.assert_array_equal(result[70, 25], source[40, 65])
        np.testing.assert_array_equal(result[0, 0], [255, 255, 255])

    def test_rejects_overwriting_mesh_or_seam(self):
        with self.assertRaisesRegex(ValueError, "must not overwrite"):
            flood_fill_texture_map(self.texture, self.mesh, self.seams, self.mesh)

    def test_command_line(self):
        subprocess.run(
            [sys.executable, str(Path(__file__).with_name("fix_texture_maps.py")),
             str(self.texture), str(self.mesh), str(self.seams), str(self.output)],
            check=True, capture_output=True, text=True,
        )
        self.assertTrue(self.output.is_file())

    def test_downsample_pads_once_and_copies_to_each_texture(self):
        source = self.root / "BlenderRebaked" / "Apple1.obj"
        destination = self.root / "Final" / "Apple1.obj"
        scripts = [Path("downsample_lowres.mlx"), Path("downsample_hires.mlx"),
                   Path("downsample_fullres.mlx")]
        source.parent.mkdir()
        destination.parent.mkdir()
        padded = source.with_name("Apple1_padded.png")
        source_material = source.with_suffix(".mtl")
        output_materials = [destination.with_name(f"Apple_001_{tag}.mtl")
                            for tag in ("lowres", "hires", "fullres")]
        output_textures = [material.with_suffix(".png") for material in output_materials]
        def read_obj(path):
            material = source_material if path == source else path.with_suffix(".mtl")
            return None, None, None, material
        def read_texture(material):
            return self.texture if material == source_material else material.with_suffix(".png")
        def fill_texture(texture, mesh, seam, output):
            Image.new("RGB", (16, 16), (12, 34, 56)).save(output)
        with (patch("MeshLabCleanup._find_seam_file", return_value=self.seams),
              patch("MeshLabCleanup._process_mesh") as process,
              patch("MeshLabCleanup._reorient_mesh") as reorient,
              patch("MeshLabCleanup._read_obj", side_effect=read_obj),
              patch("MeshLabCleanup._read_texture", side_effect=read_texture),
              patch("MeshLabCleanup.flood_fill_texture_map", side_effect=fill_texture) as fill,
              patch("MeshLabCleanup.print_mesh_topology")):
            downsample_mesh(source, destination, scripts)
        self.assertEqual(process.call_count, 3)
        self.assertEqual(
            [call.kwargs["seam_path"] for call in reorient.call_args_list],
            [self.seams] * 3,
        )
        fill.assert_called_once_with(self.texture, source, self.seams, padded)
        with Image.open(output_textures[0]) as lowres:
            self.assertEqual(lowres.size, (4, 4))
        for texture in output_textures[1:]:
            self.assertEqual(texture.read_bytes(), padded.read_bytes())

    def test_downsample_uses_companion_normalized_seam(self):
        source = self.root / "NormalizedTexture" / "Apple1.obj"
        destination = self.root / "Final" / "Apple1.obj"
        source.parent.mkdir()
        destination.parent.mkdir()
        seam = source.with_name("Apple1_seam_topology.json")
        seam.touch()
        source.with_name("Apple1_seam.csv").touch()
        source_material = source.with_suffix(".mtl")
        output_material = destination.with_name("Apple_001_lowres.mtl")
        with (patch("MeshLabCleanup._find_seam_file") as find_seam,
              patch("MeshLabCleanup._process_mesh"),
              patch("MeshLabCleanup._reorient_mesh"),
              patch("MeshLabCleanup._read_obj", side_effect=[
                  (None, None, None, source_material), (None, None, None, output_material)]),
              patch("MeshLabCleanup._read_texture", side_effect=[self.texture, output_material.with_suffix(".png")]),
              patch("MeshLabCleanup.flood_fill_texture_map",
                    side_effect=lambda texture, mesh, seam, output: Image.new("RGB", (16, 16)).save(output)) as fill,
              patch("MeshLabCleanup.shutil.copyfile"),
              patch("MeshLabCleanup.print_mesh_topology")):
            downsample_mesh(source, destination, [Path("downsample_lowres.mlx")])
        fill.assert_called_once_with(self.texture, source, seam, source.with_name("Apple1_padded.png"))
        find_seam.assert_not_called()

    def test_downsample_prefers_parameterized_and_falls_back_to_legacy(self):
        source = self.root / "BlenderRebaked" / "Apple1.obj"
        parameterized = self.root / "Parameterized"
        voxelized = self.root / "Voxelized"
        parameterized.mkdir()
        voxelized.mkdir()
        topology = parameterized / "Apple1_seam_topology.json"
        topology.touch()
        (voxelized / "Apple1_seam_topology.json").touch()
        legacy = voxelized / "Apple1_seam.csv"
        legacy.touch()
        with (patch("MeshLabCleanup._read_obj", return_value=(None, None, None, source.with_suffix(".mtl"))),
              patch("MeshLabCleanup._read_texture", return_value=self.texture),
              patch("MeshLabCleanup.flood_fill_texture_map", side_effect=ValueError("stop")) as fill,
              patch("MeshLabCleanup.print_mesh_topology")):
            for expected in (topology, legacy):
                with self.assertRaisesRegex(ValueError, "stop"):
                    downsample_mesh(source, self.root / "Final" / "Apple1.obj", [])
                self.assertEqual(fill.call_args.args[2], expected)
                topology.unlink(missing_ok=True)

    def test_downsample_stops_if_padding_fails(self):
        source = self.root / "BlenderRebaked" / "Apple1.obj"
        material = source.with_suffix(".mtl")
        with (patch("MeshLabCleanup._find_seam_file", return_value=self.seams),
              patch("MeshLabCleanup._read_obj", return_value=(None, None, None, material)),
              patch("MeshLabCleanup._read_texture", return_value=self.texture),
              patch("MeshLabCleanup.flood_fill_texture_map", side_effect=ValueError("invalid seams")),
              patch("MeshLabCleanup._process_mesh") as process,
              patch("MeshLabCleanup.print_mesh_topology")):
            with self.assertRaisesRegex(ValueError, "invalid seams"):
                downsample_mesh(source, self.root / "Final" / "Apple1.obj",
                                [Path("downsample_lowres.mlx")])
        process.assert_not_called()

    def test_downsample_respects_existing_outputs(self):
        source = self.root / "BlenderRebaked" / "Apple1.obj"
        destination = self.root / "Final" / "Apple1.obj"
        destination.parent.mkdir()
        destination.with_name("Apple_001_lowres.obj").write_text("existing")
        with (patch("MeshLabCleanup._find_seam_file", return_value=self.seams) as find_seam,
              patch("MeshLabCleanup._process_mesh") as process,
              patch("MeshLabCleanup.print_mesh_topology")):
            downsample_mesh(source, destination, [Path("downsample_lowres.mlx")], replace_existing=False)
        process.assert_not_called()
        find_seam.assert_not_called()

    def test_all_final_operations_use_filling_downsample(self):
        for operation, count in (("downsample", 3), ("downsample_lowres", 1),
                                 ("downsample_hires", 1), ("downsample_fullres", 1)):
            args = Namespace(root_dir=str(self.root), operation=operation, replace_existing=False,
                             prompt_overwrite_older=False, dest_name=None)
            with self.subTest(operation=operation), patch("main.parse_args", return_value=args), \
                    patch("main.process_meshes") as process:
                main.main()
            processor = process.call_args.args[2]
            self.assertIs(processor.func, downsample_mesh)
            self.assertEqual(len(processor.keywords["mlx_script_paths"]), count)
            self.assertFalse(processor.keywords["replace_existing"])

    def test_downsample_with_meshlab_fills_saved_texture(self):
        source_dir = self.root / "BlenderRebaked"
        seam_dir = self.root / "Voxelized"
        final_dir = self.root / "Final"
        for directory in (source_dir, seam_dir, final_dir):
            directory.mkdir()
        source = source_dir / "Apple1.obj"
        shutil.copy(self.mesh, source)
        shutil.copy(self.seams, seam_dir / "Apple1_seam.csv")
        shutil.copy(self.texture, source_dir / "texture.png")
        (source_dir / "Apple1.mtl").write_text(
            "newmtl apple\nKd 1 1 1\nmap_Kd texture.png\n", encoding="utf-8"
        )
        scripts = [Path(__file__).with_name("MeshLabScripts") / f"downsample_{tag}.mlx"
                   for tag in ("lowres", "hires", "fullres")]
        with (patch("MeshLabCleanup._find_centers", return_value=([0, 1, 0], [0, 0, 0])),
              patch("MeshLabCleanup.print_mesh_topology")):
            result = downsample_mesh(source, final_dir / "Apple1.obj", scripts)
        self.assertTrue(result.is_file())
        padded = source_dir / "Apple1_padded.png"
        with Image.open(padded) as filled:
            self.assertEqual(filled.getpixel((72, 19))[:3], (200, 10, 10))
            self.assertEqual(filled.getpixel((0, 0))[:3], (255, 255, 255))
        for tag in ("lowres", "hires", "fullres"):
            self.assertEqual((final_dir / f"Apple_001_{tag}.png").read_bytes(), padded.read_bytes())
        with Image.open(source_dir / "texture.png") as original:
            self.assertEqual(original.getpixel((20, 8)), (255, 255, 255))


class TextureExpansionTests(unittest.TestCase):
    def setUp(self):
        count = 12
        self.seam = list(range(count))
        angles = np.arange(count) * (2 * np.pi / count)
        self.vertices = np.array([
            [radius * np.cos(angle), radius * np.sin(angle), 0]
            for radius in (1, 2, 3, 4)
            for angle in angles
        ])
        self.faces = np.array([
            triangle
            for ring in range(3)
            for i in range(count)
            for triangle in (
                [ring * count + i, ring * count + (i + 1) % count, (ring + 1) * count + i],
                [ring * count + (i + 1) % count, (ring + 1) * count + (i + 1) % count,
                 (ring + 1) * count + i],
            )
        ])
        self.vertex_faces, self.adjacency = fixing._mesh_adjacency(self.vertices, self.faces)

    def test_ring_boundary_and_circle_projection(self):
        patch, outer, cut = fixing._get_outer_boundary(
            self.faces, self.vertex_faces, self.seam, 2,
            allowed_faces=set(range(len(self.faces))), vertices=self.vertices, adjacency=self.adjacency,
        )
        self.assertEqual(len(patch), 48)
        self.assertEqual(set(outer), set(range(24, 36)))
        self.assertEqual(cut[0], self.seam[0])
        self.assertEqual(cut[-1], outer[0])
        center = np.array([0.3, 0.4])
        seam_uvs = np.asarray(fixing.compute_edge_locs_circle(self.vertices, self.seam, 0.09)) + center
        face_ids, projected = fixing._circle_patch(
            self.vertices, self.faces, patch, self.seam, outer, seam_uvs, center,
        )
        positions = {}
        for face, coords in zip(face_ids, projected):
            for vertex, coord in zip(self.faces[face], coords):
                positions[int(vertex)] = coord
        np.testing.assert_allclose([positions[v] for v in self.seam], seam_uvs)
        np.testing.assert_allclose(
            [np.linalg.norm(positions[v] - center) for v in outer],
            fixing._CAP_RADIUS * 1.2,
        )

    def test_line_projection_splits_and_offsets_the_closed_ring(self):
        patch, outer, cut = fixing._get_outer_boundary(
            self.faces, self.vertex_faces, self.seam, 2,
            allowed_faces=set(range(len(self.faces))), vertices=self.vertices, adjacency=self.adjacency,
        )
        start, end = np.array([0., 0.]), np.array([1., 0.])
        ids, projected = fixing._line_patch(
            self.vertices, self.faces, patch, self.seam, outer, cut, start, end, np.array([0., -0.2]),
        )
        seam_anchor = [coord for face, coords in zip(ids, projected)
                       for vertex, coord in zip(self.faces[face], coords) if vertex == self.seam[0]]
        outer_anchor = [coord for face, coords in zip(ids, projected)
                        for vertex, coord in zip(self.faces[face], coords) if vertex == outer[0]]
        self.assertTrue(any(np.allclose(point, start) for point in seam_anchor))
        self.assertTrue(any(np.allclose(point, end) for point in seam_anchor))
        self.assertTrue(any(np.allclose(point, [0., -0.2]) for point in outer_anchor))
        self.assertTrue(any(np.allclose(point, [1., -0.2]) for point in outer_anchor))
        expected = fixing.compute_edge_locs_line(self.vertices, self.seam, start, end)
        for vertex in self.seam[1:]:
            positions = [coord for face, coords in zip(ids, projected)
                         for v, coord in zip(self.faces[face], coords) if v == vertex]
            np.testing.assert_allclose(positions, np.tile(expected[vertex], (len(positions), 1)))

    def test_outer_boundary_follows_seam_orientation_on_skewed_mesh(self):
        vertices = self.vertices.copy()
        vertices[self.seam[1]] = vertices[35] + [0.01, 0, 0]
        vertex_faces, adjacency = fixing._mesh_adjacency(vertices, self.faces)
        _, outer, _ = fixing._get_outer_boundary(
            self.faces, vertex_faces, self.seam, 2,
            allowed_faces=set(range(len(self.faces))), vertices=vertices, adjacency=adjacency,
        )
        self.assertEqual((outer[1] - outer[0]) % len(self.seam), 1)

    def test_ring_expansion_fills_a_hole_touching_outer_boundary_at_one_vertex(self):
        face_index = 2 * len(self.seam) + 4
        a, b, c = self.faces[face_index]
        vertices = np.vstack((self.vertices, self.vertices[[a, b, c]].mean(axis=0)))
        center = len(vertices) - 1
        faces = np.vstack((self.faces[:face_index], [[a, b, center], [b, c, center], [c, a, center]],
                           self.faces[face_index + 1:]))
        vertex_faces, _ = fixing._mesh_adjacency(vertices, faces)
        patch = set(range(2 * len(self.seam) * 2 + 2)) - {face_index + 1, face_index + 2}
        repaired = fixing._fill_patch_holes(faces, vertex_faces, patch, set(range(len(faces))))
        self.assertTrue({face_index + 1, face_index + 2}.issubset(repaired))
        self.assertFalse(any(face >= 2 * len(self.seam) * 2 + 2 for face in repaired))
        boundary = fixing._boundary_edges(faces, repaired)
        degrees = {}
        for edge in boundary:
            for vertex in edge:
                degrees[vertex] = degrees.get(vertex, 0) + 1
        self.assertTrue(all(degree == 2 for degree in degrees.values()))

    def test_side_patch_projects_inside_pixels_outside_opposite_side(self):
        face_triangles = np.array([[0, 1, 4]])
        face_uvs = np.array([[0, 1, 2]])
        uvs = np.array([[0.25, 0.25], [0.25, 0.75], [0.35, 0.5]])
        vertex_faces = [[0], [0], [], [], [0]]
        ids, projected = fixing._side_patch(
            face_triangles, face_uvs, uvs, vertex_faces, {0}, {0}, [0, 1, 2, 3],
            (np.array([0.25, 0.25]), np.array([0.25, 0.75])),
            (np.array([0.75, 0.25]), np.array([0.75, 0.75])),
        )
        self.assertEqual(ids, [0])
        np.testing.assert_allclose(projected[0, 2], [0.85, 0.5])

    def test_side_patch_does_not_cross_excluded_corner_faces(self):
        faces = np.array([[0, 1, 2], [1, 2, 3], [3, 4, 5]])
        uvs = np.array([[0.25, 0.25], [0.3, 0.25], [0.3, 0.5],
                        [0.35, 0.5], [0.35, 0.75], [0.4, 0.75]])
        vertex_faces, _ = fixing._mesh_adjacency(np.column_stack((uvs, np.zeros(6))), faces)
        ids, _ = fixing._side_patch(
            faces, faces, uvs, vertex_faces, {0}, {0, 2}, [0],
            (np.array([0.25, 0.25]), np.array([0.25, 0.75])),
            (np.array([0.75, 0.25]), np.array([0.75, 0.75])),
        )
        self.assertEqual(ids, [0])

    def test_copy_preserves_occupied_pixels_and_fills_only_projected_region(self):
        source = np.full((7, 7, 3), [10, 20, 30], dtype=np.uint8)
        output = np.zeros_like(source)
        output[2, 2] = [77, 88, 99]
        occupied = np.zeros((7, 7), dtype=bool)
        occupied[2, 2] = True
        source_pixels = np.array([[1., 1.], [5., 1.], [1., 5.]])
        projected = np.array([[[1 / 6, 5 / 6], [5 / 6, 5 / 6], [1 / 6, 1 / 6]]])
        scratch = np.zeros_like(source)
        scratch_covered = np.zeros((7, 7), dtype=bool)
        fixing._copy_projected_faces(source, output, occupied, source_pixels, np.array([[0, 1, 2]]),
                                     [0], projected, scratch, scratch_covered, set())
        np.testing.assert_array_equal(output[2, 2], [77, 88, 99])
        np.testing.assert_array_equal(output[1, 1], [10, 20, 30])
        np.testing.assert_array_equal(output[6, 6], [0, 0, 0])
        self.assertFalse(scratch_covered.any())

    def test_corner_face_exclusion_includes_shared_and_distinct_seam_vertices(self):
        faces = np.array([
            [0, 5, 6], [1, 4, 7], [2, 8, 9], [3, 4, 10],
            [1, 5, 6], [4, 5, 6], [1, 3, 7], [5, 6, 7],
        ])
        vertex_faces, _ = fixing._mesh_adjacency(np.zeros((11, 3)), faces)
        seam_ids = {"top": [0, 1], "bottom": [2, 3], "side": [0, 2, 4]}
        self.assertEqual(fixing._seam_corner_faces(vertex_faces, seam_ids), {0, 1, 2, 3})

    def test_copy_skips_excluded_faces(self):
        source = np.full((9, 9, 3), [10, 20, 30], dtype=np.uint8)
        output = np.zeros_like(source)
        occupied = np.zeros((9, 9), dtype=bool)
        pixels = np.array([[0., 0.], [8., 0.], [0., 8.]])
        targets = np.array([
            [[0, 1], [0.5, 1], [0, 0.5]],
            [[0.5, 0.5], [1, 0.5], [1, 0]],
        ])
        fixing._copy_projected_faces(
            source, output, occupied, pixels, np.array([[0, 1, 2], [0, 1, 2]]),
            [0, 1], targets, np.zeros_like(source), np.zeros((9, 9), dtype=bool), {0},
        )
        np.testing.assert_array_equal(output[1, 1], [0, 0, 0])
        np.testing.assert_array_equal(output[6, 7], [10, 20, 30])

    def test_top_patch_skips_long_edges_even_when_they_follow_the_seam(self):
        points = np.array([
            [[0, 0], [0.01, 0], [0, 0]],
            [[0, 0], [0.010001, 0], [0, 0]],
            [[0.5, 0], [1, 0.5], [0.75, 0.25]],
            [[0, 0], [0.02, 0], [0, 0]],
        ])
        already_excluded = {3}
        with self.assertWarnsRegex(RuntimeWarning, r"2 .*top.*0\.01"):
            result = fixing._exclude_long_top_seam_faces([0, 1, 2, 3], points, already_excluded)
        self.assertEqual(result, {1, 2, 3})
        self.assertEqual(already_excluded, {3})

    def test_refuses_to_replace_input_mesh(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mesh.obj"
            with self.assertRaisesRegex(ValueError, "must not overwrite"):
                fixing.flood_fill_texture_map(path.with_suffix(".png"), path, path.with_suffix(".json"), path)

    def test_expands_complete_mesh_and_preserves_original_chart_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh = root / "input.obj"
            vertices, uvs, faces, face_uvs = [], [], [], []
            regions = {name: set() for name in (
                "top", "bottom", "middle", "left_side_seam", "right_side_seam"
            )}
            count, levels, cap_rings = 16, 12, 8
            angles = np.arange(count) * 2 * np.pi / count
            body = []
            for level in range(levels + 1):
                ring = []
                for angle in angles:
                    ring.append(len(vertices))
                    vertices.append([np.cos(angle), 1 - 2 * level / levels, np.sin(angle)])
                body.append(ring)

            corners = np.array([[0.5, fixing._UV_PADDING], [1 - fixing._UV_PADDING, 0.5],
                                [0.5, 1 - fixing._UV_PADDING], [fixing._UV_PADDING, 0.5]])
            body_uvs = []
            for level in range(levels + 1):
                t = level / levels
                ring = []
                for i in range(count + 1):
                    s = i / count
                    top = (1 - s) * corners[0] + s * corners[1]
                    bottom = (1 - s) * corners[3] + s * corners[2]
                    ring.append(len(uvs))
                    uvs.append((1 - t) * top + t * bottom)
                body_uvs.append(ring)

            def add_face(region, vertex_ids, uv_ids):
                face_id = len(faces)
                regions[region].add(face_id)
                faces.append(vertex_ids)
                face_uvs.append(uv_ids)
                return face_id

            for level in range(levels):
                for i in range(count):
                    j = (i + 1) % count
                    a, b = body[level][i], body[level][j]
                    c, d = body[level + 1][i], body[level + 1][j]
                    ua, ub = body_uvs[level][i:i + 2]
                    uc, ud = body_uvs[level + 1][i:i + 2]
                    first = add_face("middle", [a, b, c], [ua, ub, uc])
                    second = add_face("middle", [b, d, c], [ub, ud, uc])
                    if i == 0:
                        regions["left_side_seam"].add(first)
                    if i == count - 1:
                        regions["right_side_seam"].add(second)

            centers = []
            for cap_index, name in enumerate(("top", "bottom")):
                rings = [body[0 if name == "top" else -1]]
                uv_rings = []
                center_uv = fixing._CAP_CENTERS[cap_index]
                height = 1 if name == "top" else -1
                for ring_index in range(cap_rings):
                    radius = 1 - ring_index / cap_rings
                    if ring_index:
                        ring = []
                        for angle in angles:
                            ring.append(len(vertices))
                            vertices.append([radius * np.cos(angle), height, radius * np.sin(angle)])
                        rings.append(ring)
                    uv_ring = []
                    for angle in angles:
                        uv_ring.append(len(uvs))
                        uvs.append(center_uv + 0.09 * radius * np.array([np.cos(angle), np.sin(angle)]))
                    uv_rings.append(uv_ring)
                centers.append(len(vertices))
                vertices.append([0, height, 0])
                center_uv_index = len(uvs)
                uvs.append(center_uv)
                for ring_index in range(cap_rings - 1):
                    for i in range(count):
                        j = (i + 1) % count
                        a, b = rings[ring_index][i], rings[ring_index][j]
                        c, d = rings[ring_index + 1][i], rings[ring_index + 1][j]
                        ua, ub = uv_rings[ring_index][i], uv_rings[ring_index][j]
                        uc, ud = uv_rings[ring_index + 1][i], uv_rings[ring_index + 1][j]
                        add_face(name, [a, b, c], [ua, ub, uc])
                        add_face(name, [b, d, c], [ub, ud, uc])
                for i in range(count):
                    j = (i + 1) % count
                    add_face(name, [rings[-1][i], rings[-1][j], centers[-1]],
                             [uv_rings[-1][i], uv_rings[-1][j], center_uv_index])

            topology = SeamTopology(-1, 1, (0, 1, 0), (0, -1, 0),
                                    top_vertex=centers[0], bottom_vertex=centers[1],
                                    face_lists=regions)
            for name, ids in (("top", body[0]), ("bottom", body[-1]),
                              ("side", [ring[0] for ring in body])):
                topology.add_seam(name, SeamInfo(name, ids, [vertices[v] for v in ids], None, None))
            mesh.with_name("input_seam_topology.json").write_text(json.dumps(topology.to_dict()))
            mesh.with_suffix(".mtl").write_text("newmtl material_0\nmap_Kd input.png\n")
            with mesh.open("w") as file:
                file.write("mtllib input.mtl\n")
                for vertex in vertices:
                    file.write("v " + " ".join(map(str, vertex)) + "\n")
                for uv in uvs:
                    file.write("vt " + " ".join(map(str, uv)) + "\n")
                file.write("vn 0 1 0\n")
                for face, face_uv in zip(faces, face_uvs):
                    file.write("f " + " ".join(f"{v + 1}/{uv + 1}/1" for v, uv in zip(face, face_uv)) + "\n")
            image = Image.new("RGB", (256, 256), (0, 0, 0))
            occupied = Image.new("1", image.size)
            bottom_mask = Image.new("1", image.size)
            draw, mask_draw = ImageDraw.Draw(image), ImageDraw.Draw(occupied)
            bottom_draw = ImageDraw.Draw(bottom_mask)
            uv_array = np.asarray(uvs)
            coords = uv_array * 255
            coords[:, 1] = 255 - coords[:, 1]
            for region, color in (("middle", (20, 40, 200)), ("top", (200, 30, 30)),
                                  ("bottom", (30, 200, 30))):
                for face in regions[region]:
                    polygon = [tuple(coords[uv]) for uv in face_uvs[face]]
                    draw.polygon(polygon, fill=color)
                    mask_draw.polygon(polygon, fill=1)
                    if region == "bottom":
                        bottom_draw.polygon(polygon, fill=1)
            image_pixels = np.array(image)
            lower_half = np.indices(image_pixels.shape[:2])[0] > (1 - fixing._CAP_CENTERS[1, 1]) * 255
            image_pixels[np.asarray(bottom_mask, dtype=bool) & lower_half] = [200, 150, 30]
            image = Image.fromarray(image_pixels)
            image.save(root / "input.png")
            output = root / "out" / "result.obj"
            with (self.assertWarnsRegex(RuntimeWarning, r"projected top-seam faces.*0\.01"),
                  patch("fix_texture_maps._copy_projected_faces", wraps=fixing._copy_projected_faces) as copy,
                  patch("fix_texture_maps._side_patch", wraps=fixing._side_patch) as side):
                fixing.expand_texture_map(mesh, output)
            vertex_faces, _ = fixing._mesh_adjacency(np.asarray(vertices), np.asarray(faces))
            seam_ids = {name: topology.seams[name].vertex_ids for name in ("top", "bottom", "side")}
            excluded = fixing._seam_corner_faces(vertex_faces, seam_ids)
            self.assertTrue(excluded)
            self.assertEqual(copy.call_count, 6)
            for index, call in enumerate(copy.call_args_list):
                expected_excluded = excluded.copy()
                if index == 1:
                    ids, projected = call.args[5:7]
                    lengths = np.linalg.norm(projected - np.roll(projected, -1, axis=1), axis=2)
                    expected_excluded.update(
                        face for face, face_lengths in zip(ids, lengths) if np.any(face_lengths > 0.01)
                    )
                self.assertEqual(call.args[-1], expected_excluded)
            self.assertEqual(side.call_count, 2)
            for call in side.call_args_list:
                self.assertFalse(call.args[4] & excluded)
                self.assertFalse(call.args[5] & excluded)
            with Image.open(output.with_suffix(".png")) as result:
                actual = np.asarray(result)
            original = np.asarray(image)
            mask = np.asarray(occupied, dtype=bool)
            np.testing.assert_array_equal(actual[mask], original[mask])
            self.assertGreater(np.count_nonzero(np.any(actual[~mask] != original[~mask], axis=1)), 100)
            for center, direction in zip(fixing._CAP_CENTERS, (1, -1)):
                pixel = np.rint((center + [0, direction * 0.105]) * 255).astype(int)
                np.testing.assert_array_equal(actual[255 - pixel[1], pixel[0]], [20, 40, 200])
            for start, end, outward, color in (
                (corners[0], corners[1], [1, -1], [0, 0, 0]),
                (corners[3], corners[2], [-1, 1], [30, 200, 30]),
            ):
                point = 0.7 * start + 0.3 * end + np.asarray(outward) * (0.05 / np.sqrt(2))
                pixel = np.rint(point * 255).astype(int)
                np.testing.assert_array_equal(actual[255 - pixel[1], pixel[0]], color)
            for fraction, color in ((0.25, [30, 200, 30]), (0.75, [200, 150, 30])):
                point = ((1 - fraction) * corners[3] + fraction * corners[2]
                         + np.array([-1, 1]) * (0.03 / np.sqrt(2)))
                pixel = np.rint(point * 255).astype(int)
                np.testing.assert_array_equal(actual[255 - pixel[1], pixel[0]], color)
            self.assertEqual(mesh.with_name("input_seam_topology.json").read_bytes(),
                             output.with_name("result_seam_topology.json").read_bytes())
            self.assertIn("map_Kd result.png", output.with_suffix(".mtl").read_text())


if __name__ == "__main__":
    unittest.main()

if __name__ == "__main__":
    unittest.main()
