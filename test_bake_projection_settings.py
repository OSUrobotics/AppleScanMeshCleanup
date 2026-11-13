import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import bake_textures as baking
import main


class BakeProjectionSettingsTests(unittest.TestCase):
    def test_api_forwards_default_and_custom_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "Envy1Target.obj"
            target.touch()
            source_dir = root / "sources"
            source_dir.mkdir()
            (source_dir / "Envy1Source.obj").touch()
            output = root / "result.obj"
            with patch.object(baking, "_blender_executable", return_value="blender"), \
                    patch.object(baking.subprocess, "run") as run:
                baking.bake_textures(target, output, source_dir)
                command = run.call_args.args[0]
                self.assertEqual(command[command.index("--cage-extrusion") + 1], "0.25")
                self.assertEqual(command[command.index("--max-ray-distance") + 1], "0.35")
                baking.bake_textures(target, output, source_dir, cage_extrusion=0.4, max_ray_distance=0.6)
                command = run.call_args.args[0]
                self.assertEqual(command[command.index("--cage-extrusion") + 1], "0.4")
                self.assertEqual(command[command.index("--max-ray-distance") + 1], "0.6")

    def test_cli_defaults_overrides_and_invalid_values(self):
        args = baking._parse_args(["target.obj", "source", "out.obj"])
        self.assertEqual((args.cage_extrusion, args.max_ray_distance), (0.25, 0.35))
        args = baking._parse_args([
            "target.obj", "source", "out.obj", "--cage-extrusion", "0.4",
            "--max-ray-distance", "0",
        ])
        self.assertEqual((args.cage_extrusion, args.max_ray_distance), (0.4, 0.0))
        for value in ("-1", "nan", "inf", "1e100"):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    baking._parse_args(["--cage-extrusion", value])
                with self.assertRaises(ValueError):
                    baking._projection_distance(value)

    def test_main_forwards_values_to_bake(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = ["main.py", "--root-dir", temporary, "--operation", "bake_textures",
                    "--cage-extrusion", "0.4", "--max-ray-distance", "0.6"]
            with patch("sys.argv", args), patch.object(main, "process_meshes", return_value=[]) as process:
                main.main()
            bake = process.call_args.args[2]
            self.assertEqual(bake.keywords["cage_extrusion"], 0.4)
            self.assertEqual(bake.keywords["max_ray_distance"], 0.6)

    def test_api_rejects_invalid_projection_values(self):
        for value in (-1, float("nan"), float("inf"), 1e100):
            with self.subTest(value=value), self.assertRaises(ValueError):
                baking.bake_textures(Path("target.obj"), Path("output.obj"), Path("."),
                                     cage_extrusion=value)


if __name__ == "__main__":
    unittest.main()
