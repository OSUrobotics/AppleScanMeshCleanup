"""Compute PCA shape variation from OBJ meshes with matching vertex topology.

Run ``python shape_variation.py analyze MESH_DIRECTORY shape_pca.npz`` to save
the mean shape, principal components, eigenvalues, and per-mesh weights.
Run ``python shape_variation.py generate shape_pca.npz COUNT OUTPUT_DIRECTORY``
to create random OBJ variations.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ShapePCA:
    """PCA data and the OBJ template needed to write generated meshes."""

    mean: np.ndarray
    components: np.ndarray
    explained_variance: np.ndarray
    weights: np.ndarray
    mesh_names: tuple[str, ...]
    template_lines: tuple[str, ...]
    template_name: str


def _read_obj(path: Path) -> tuple[list[str], np.ndarray, tuple[tuple[int, ...], ...]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    vertices = []
    faces = []

    for line_number, line in enumerate(lines, start=1):
        fields = line.partition("#")[0].split()
        if not fields:
            continue
        if fields[0] == "v":
            if len(fields) < 4:
                raise ValueError(f"{path}:{line_number}: expected three vertex coordinates.")
            try:
                vertex = [float(value) for value in fields[1:4]]
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: invalid vertex coordinates.") from error
            if not np.isfinite(vertex).all():
                raise ValueError(f"{path}:{line_number}: vertex coordinates must be finite.")
            vertices.append(vertex)
        elif fields[0] == "f":
            if len(fields) < 4:
                raise ValueError(f"{path}:{line_number}: a face must have at least three vertices.")
            face = []
            for corner in fields[1:]:
                try:
                    index = int(corner.split("/", 1)[0])
                except ValueError as error:
                    raise ValueError(f"{path}:{line_number}: invalid face vertex index.") from error
                if index == 0:
                    raise ValueError(f"{path}:{line_number}: OBJ vertex indices cannot be zero.")
                resolved = index - 1 if index > 0 else len(vertices) + index
                if resolved < 0 or resolved >= len(vertices):
                    raise ValueError(f"{path}:{line_number}: face vertex index is out of range.")
                face.append(resolved)
            faces.append(tuple(face))

    if not vertices:
        raise ValueError(f"{path} contains no vertices.")
    if not faces:
        raise ValueError(f"{path} contains no faces.")
    return lines, np.asarray(vertices, dtype=np.float64), tuple(faces)


def _mesh_paths(mesh_paths: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(mesh_paths, (str, Path)):
        source = Path(mesh_paths)
        if source.is_dir():
            paths = sorted(
                (path for path in source.iterdir() if path.is_file() and path.suffix.lower() == ".obj"),
                key=lambda path: path.name.casefold(),
            )
        else:
            paths = [source]
    else:
        paths = [Path(path) for path in mesh_paths]
    if len(paths) < 2:
        raise ValueError("At least two OBJ meshes are required to compute shape variation.")
    return paths


def _save_shape_pca(pca: ShapePCA, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as stream:
        np.savez_compressed(
            stream,
            mean=pca.mean,
            components=pca.components,
            explained_variance=pca.explained_variance,
            weights=pca.weights,
            mesh_names=np.asarray(pca.mesh_names, dtype=np.str_),
            template_lines=np.asarray(pca.template_lines, dtype=np.str_),
            template_name=np.asarray(pca.template_name, dtype=np.str_),
        )


def compute_shape_variation(
    mesh_paths: str | Path | Iterable[str | Path],
    output_path: str | Path,
) -> ShapePCA:
    """Fit PCA to OBJ vertex positions and save the model and sample weights.

    ``mesh_paths`` may be a directory of OBJ files, one path per mesh, or an
    iterable of paths. All meshes must have identical face indices and vertex
    ordering.
    """
    output_path = Path(output_path)
    paths = _mesh_paths(mesh_paths)
    template_lines, template_vertices, template_faces = _read_obj(paths[0])
    all_vertices = [template_vertices]

    for path in paths[1:]:
        _, vertices, faces = _read_obj(path)
        if len(vertices) != len(template_vertices) or faces != template_faces:
            raise ValueError(
                f"Mesh {path} does not have the same vertex count and face topology as {paths[0]}."
            )
        all_vertices.append(vertices)

    samples = np.stack(all_vertices).reshape(len(paths), -1)
    mean = samples.mean(axis=0)
    centered = samples - mean
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    eigenvalues = singular_values**2 / (len(paths) - 1)

    if len(eigenvalues) and eigenvalues[0] > 0:
        tolerance = np.finfo(np.float64).eps * max(centered.shape) * eigenvalues[0]
        keep = eigenvalues > tolerance
        eigenvalues = eigenvalues[keep]
        components = right_vectors[keep]
    else:
        eigenvalues = np.empty(0, dtype=np.float64)
        components = np.empty((0, samples.shape[1]), dtype=np.float64)

    pca = ShapePCA(
        mean=mean,
        components=components,
        explained_variance=eigenvalues,
        weights=centered @ components.T,
        mesh_names=tuple(path.name for path in paths),
        template_lines=tuple(template_lines),
        template_name=paths[0].name,
    )
    _save_shape_pca(pca, output_path)
    return pca


def _load_shape_pca(model: ShapePCA | str | Path) -> ShapePCA:
    if isinstance(model, ShapePCA):
        pca = model
    else:
        with np.load(Path(model), allow_pickle=False) as archive:
            pca = ShapePCA(
                mean=np.array(archive["mean"], dtype=np.float64),
                components=np.array(archive["components"], dtype=np.float64),
                explained_variance=np.array(archive["explained_variance"], dtype=np.float64),
                weights=np.array(archive["weights"], dtype=np.float64),
                mesh_names=tuple(str(name) for name in archive["mesh_names"]),
                template_lines=tuple(str(line) for line in archive["template_lines"]),
                template_name=str(archive["template_name"].item()),
            )

    dimension = pca.mean.size
    if pca.mean.ndim != 1 or dimension == 0 or dimension % 3:
        raise ValueError("The PCA mean must contain a non-empty sequence of XYZ vertex positions.")
    if pca.components.ndim != 2 or pca.components.shape[1] != dimension:
        raise ValueError("PCA components do not match the mean shape.")
    if pca.explained_variance.shape != (len(pca.components),):
        raise ValueError("PCA explained variance does not match the component count.")
    if pca.weights.ndim != 2 or pca.weights.shape[1] != len(pca.components):
        raise ValueError("PCA sample weights do not match the component count.")
    if not all(
        np.isfinite(array).all()
        for array in (pca.mean, pca.components, pca.explained_variance, pca.weights)
    ):
        raise ValueError("PCA data must contain only finite values.")
    if np.any(pca.explained_variance < 0):
        raise ValueError("PCA explained variance cannot be negative.")
    if not pca.template_lines or not pca.template_name:
        raise ValueError("The PCA model is missing its OBJ template.")
    vertex_count = sum(
        1 for line in pca.template_lines if line.partition("#")[0].split()[:1] == ["v"]
    )
    if vertex_count * 3 != dimension:
        raise ValueError("The PCA mean does not match the template OBJ vertex count.")
    return pca


def _write_obj(template_lines: tuple[str, ...], vertices: np.ndarray, output_path: Path) -> None:
    vertex_index = 0
    output_lines = []
    vertex_line = re.compile(r"^(\s*v\s+)(\S+)(\s+)(\S+)(\s+)(\S+)(.*)$")
    for line in template_lines:
        if line.partition("#")[0].split()[:1] != ["v"]:
            output_lines.append(line)
            continue
        match = vertex_line.match(line)
        if match is None:
            raise ValueError(f"Cannot preserve vertex line in OBJ template: {line!r}")
        x, y, z = (format(float(value), ".17g") for value in vertices[vertex_index])
        output_lines.append(
            f"{match.group(1)}{x}{match.group(3)}{y}{match.group(5)}{z}{match.group(7)}"
        )
        vertex_index += 1

    output_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")


def generate_random_variations(
    pca_components: ShapePCA | str | Path,
    num_variations: int,
    output_dir: str | Path,
    seed: int | None = None,
) -> list[Path]:
    """Write random OBJ meshes sampled from a saved shape PCA model.

    Component weights are sampled independently from normal distributions
    scaled by each component's standard deviation.
    """
    if isinstance(num_variations, bool) or not isinstance(num_variations, int) or num_variations < 0:
        raise ValueError("The number of variations must be a non-negative integer.")
    pca = _load_shape_pca(pca_components)
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ValueError("The random seed must be an integer.")

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    width = max(4, len(str(num_variations)))
    output_paths = [
        directory / f"variation_{index:0{width}d}.obj"
        for index in range(1, num_variations + 1)
    ]

    rng = np.random.default_rng(seed)
    random_weights = rng.normal(size=(num_variations, len(pca.components)))
    random_weights *= np.sqrt(pca.explained_variance)
    generated = pca.mean + random_weights @ pca.components
    for output_path, flat_vertices in zip(output_paths, generated):
        _write_obj(
            pca.template_lines,
            flat_vertices.reshape(-1, 3),
            output_path,
        )
    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    root_dir = "/Users/cindygrimm/PycharmProjects/data/ScannedModels/ProsserEnvyApples"
    analyze = commands.add_parser("analyze", help="Compute and save PCA shape parameters.")
    analyze.add_argument("--mesh-directory", nargs="?", default="AlignedMeshes", type=str)
    analyze.add_argument("--root-dir", nargs="?", default=root_dir, type=str)
    analyze.add_argument("--output_name", nargs="?", default="model.npz", type=str)

    generate = commands.add_parser("generate", help="Generate random OBJ variations.")
    generate.add_argument("--root-dir", nargs="?", default=root_dir, type=str)
    generate.add_argument("--pca-file", nargs="?", default="AlignedMeshes/model.npz", type=Path)
    generate.add_argument("--count", nargs="?", default=1, type=int)
    generate.add_argument("--output-dir", nargs="?", default="Variations", type=Path)
    generate.add_argument("--seed", type=int, help="Seed for repeatable random variations.")

    args = parser.parse_args()
    if args.command == "analyze":
        mesh_path = Path(args.root_dir) / args.mesh_directory
        output_dir = mesh_path / args.output_name
        pca = compute_shape_variation(
            mesh_path, output_dir
        )
        print(
            f"Saved PCA for {len(pca.mesh_names)} meshes with "
            f"{len(pca.components)} components to {output_dir}."
        )
    else:
        pca_file = Path(args.root_dir) / args.pca_file
        paths = generate_random_variations(
            pca_file,
            args.count,
            Path(args.root_dir) / args.output_dir,
            seed=args.seed
        )
        print(f"Wrote {len(paths)} shape variations to {Path(args.root_dir) / args.output_dir}.")


if __name__ == "__main__":
    main()
