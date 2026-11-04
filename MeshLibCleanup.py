import shlex
import shutil
from pathlib import Path

from meshlib import mrmeshpy as mm


_TEXTURE_MAP_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def _copy_asset(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return destination


def _copy_material_file(source: Path, destination: Path) -> None:
    _copy_asset(source, destination)
    for line in source.read_text(encoding="utf-8").splitlines():
        try:
            parts = shlex.split(line, comments=True)
        except ValueError:
            continue
        if not parts or not (
            parts[0].lower().startswith("map_")
            or parts[0].lower() in {"bump", "disp", "decal", "norm", "refl"}
        ):
            continue
        texture_reference = Path(parts[-1])
        if texture_reference.suffix.lower() not in _TEXTURE_MAP_EXTENSIONS:
            continue
        texture_source = (
            texture_reference
            if texture_reference.is_absolute()
            else source.parent / texture_reference
        )
        texture_destination = (
            texture_reference
            if texture_reference.is_absolute()
            else destination.parent / texture_reference
        )
        _copy_asset(texture_source, texture_destination)


def _copy_obj_material_assets(
    input_path: Path,
    output_path: Path,
    original_obj: str,
    source_mesh: mm.Mesh | None = None,
) -> None:
    material_references = []
    for line in original_obj.splitlines():
        try:
            parts = shlex.split(line, comments=True)
        except ValueError:
            continue
        if parts and parts[0].lower() == "mtllib":
            material_references.extend(parts[1:])

    if not material_references:
        fallback_material = input_path.with_suffix(".mtl")
        if fallback_material.is_file():
            material_references.append(fallback_material.name)
    output_references = []
    for reference in material_references:
        material_path = Path(reference)
        if not material_path.is_absolute():
            material_path = input_path.parent / material_path
        if not material_path.is_file():
            raise FileNotFoundError(f"Material file not found: {material_path}")

        material_reference = Path(reference)
        if material_reference.is_absolute():
            material_reference = Path(material_path.name)
        material_destination = output_path.parent / material_reference
        _copy_material_file(material_path, material_destination)
        output_reference = material_reference.as_posix()
        output_references.append(output_reference)

    if output_references:
        output_lines = [
            line
            for line in output_path.read_text(encoding="utf-8").splitlines()
            if not line.lstrip().lower().startswith("mtllib ")
        ]
        output_lines[0:0] = [
            f"mtllib {shlex.quote(reference)}" for reference in output_references
        ]
        output_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")

    if source_mesh is not None:
        _restore_obj_texture_coordinates(
            original_obj, output_path, source_mesh
        )


def preserve_obj_material_assets(
    input_path: str | Path,
    output_path: str | Path,
    original_obj: str,
    source_mesh: mm.Mesh | None = None,
) -> None:
    _copy_obj_material_assets(
        Path(input_path), Path(output_path), original_obj, source_mesh
    )


def _resolve_obj_index(index: str, values: list) -> int:
    value = int(index)
    return value - 1 if value > 0 else len(values) + value


def _position_key(position: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(round(value, 6) for value in position)


def _parse_obj_texture_faces(original_obj: str) -> list[dict]:
    vertices = []
    texture_coordinates = []
    texture_faces = []
    material = None

    for line in original_obj.splitlines():
        parts = line.split("#", 1)[0].split()
        if not parts:
            continue
        if parts[0] == "v" and len(parts) >= 4:
            vertices.append(tuple(map(float, parts[1:4])))
        elif parts[0] == "vt" and len(parts) >= 3:
            texture_coordinates.append(tuple(map(float, parts[1:3])))
        elif parts[0] == "usemtl":
            material = " ".join(parts[1:]) or None
        elif parts[0] == "f" and len(parts) >= 4:
            corners = []
            for corner in parts[1:]:
                indices = corner.split("/")
                vertex_index = _resolve_obj_index(indices[0], vertices)
                texture_index = (
                    _resolve_obj_index(indices[1], texture_coordinates)
                    if len(indices) > 1 and indices[1]
                    else None
                )
                corners.append(
                    (
                        vertices[vertex_index],
                        texture_coordinates[texture_index]
                        if texture_index is not None
                        else None,
                    )
                )

            for index in range(1, len(corners) - 1):
                triangle = (corners[0], corners[index], corners[index + 1])
                texture_faces.append(
                    {
                        "corners": triangle,
                        "material": material,
                    }
                )

    return texture_faces


def _restore_obj_texture_coordinates(
    original_obj: str, output_path: Path, source_mesh: mm.Mesh
) -> None:
    source_faces_by_geometry: dict[tuple, list[dict]] = {}
    for source_face in _parse_obj_texture_faces(original_obj):
        geometry_key = tuple(
            sorted(_position_key(corner[0]) for corner in source_face["corners"])
        )
        source_faces_by_geometry.setdefault(geometry_key, []).append(source_face)

    topology = source_mesh.topology
    source_data_by_face = {}
    valid_faces = topology.getValidFaces()
    for face_index in range(topology.faceSize()):
        face_id = mm.FaceId(face_index)
        if not valid_faces.test(face_id):
            continue

        vertex_ids = topology.getTriVerts(face_id)
        positions = [
            tuple(source_mesh.points[vertex_id][axis] for axis in range(3))
            for vertex_id in vertex_ids
        ]
        geometry_key = tuple(sorted(_position_key(position) for position in positions))
        matching_faces = source_faces_by_geometry.get(geometry_key)
        if not matching_faces:
            continue

        source_face = matching_faces[0]
        source_faces_by_geometry[geometry_key] = matching_faces[1:]
        uv_by_vertex = {}
        for vertex_id, position in zip(vertex_ids, positions):
            position_key = _position_key(position)
            for corner_position, texture_coordinate in source_face["corners"]:
                if (
                    _position_key(corner_position) == position_key
                    and texture_coordinate is not None
                ):
                    uv_by_vertex[vertex_id.get()] = texture_coordinate
                    break

        if len(uv_by_vertex) == 3:
            source_data_by_face[face_id.get()] = {
                "uv_by_vertex": uv_by_vertex,
                "material": source_face["material"],
            }

    if not source_data_by_face:
        return

    texture_source_region = mm.FaceBitSet(topology.faceSize())
    for face_index in source_data_by_face:
        texture_source_region.set(mm.FaceId(face_index), True)

    output_lines = output_path.read_text(encoding="utf-8").splitlines()
    output_vertices = []
    non_face_lines = []
    face_lines = []
    for line in output_lines:
        parts = line.split()
        if parts and parts[0] == "v" and len(parts) >= 4:
            output_vertices.append(tuple(map(float, parts[1:4])))
            non_face_lines.append(line)
        elif parts and parts[0] == "f":
            face_lines.append(line)
        elif parts and parts[0] in ("vt", "usemtl"):
            continue
        else:
            non_face_lines.append(line)

    texture_lines = []
    rewritten_faces = []
    next_texture_index = 1
    for line in face_lines:
        parts = line.split()
        corners = []
        output_vertex_ids = []
        for corner in parts[1:]:
            indices = corner.split("/")
            vertex_index = _resolve_obj_index(indices[0], output_vertices)
            output_vertex_ids.append(vertex_index)
            corners.append((indices, output_vertices[vertex_index]))

        centroid = tuple(
            sum(position[axis] for _, position in corners) / len(corners)
            for axis in range(3)
        )
        projection = source_mesh.findClosestPoint(
            mm.Vector3f(*centroid), region=texture_source_region
        )
        source_face_id = projection.proj.face.get()
        source_data = source_data_by_face.get(source_face_id)
        if source_data is None:
            rewritten_faces.append(line)
            continue

        texture_indices = []
        for _, position in corners:
            projected_corner = source_mesh.findClosestPoint(
                mm.Vector3f(*position), region=texture_source_region
            )
            corner_source_data = source_data_by_face[
                projected_corner.proj.face.get()
            ]
            weighted_vertices = projected_corner.mtp.getWeightedVerts(topology)
            u = 0.0
            v = 0.0
            for weighted_vertex in weighted_vertices:
                uv = corner_source_data["uv_by_vertex"].get(
                    weighted_vertex.v.get()
                )
                if uv is None:
                    break
                u += uv[0] * weighted_vertex.weight
                v += uv[1] * weighted_vertex.weight
            else:
                texture_lines.append(f"vt {u} {v}")
                texture_indices.append(next_texture_index)
                next_texture_index += 1

        if len(texture_indices) != len(corners):
            rewritten_faces.append(line)
            continue

        material = source_data["material"]
        rewritten_faces.append(
            f"usemtl {material}" if material else ""
        )
        rewritten_faces.append(
            "f "
            + " ".join(
                f"{indices[0]}/{texture_index}"
                + (f"/{indices[2]}" if len(indices) > 2 and indices[2] else "")
                for (indices, _), texture_index in zip(corners, texture_indices)
            )
        )

    if not texture_lines:
        return

    first_face_index = next(
        (index for index, line in enumerate(non_face_lines) if line.lstrip().startswith("f ")),
        len(non_face_lines),
    )
    non_face_lines[first_face_index:first_face_index] = texture_lines
    output_path.write_text(
        "\n".join(non_face_lines + [line for line in rewritten_faces if line]) + "\n",
        encoding="utf-8",
    )


def print_mesh_topology(mesh_path: str | Path) -> int:
    """Print vertex, face, edge counts and the Euler characteristic."""
    mesh = mm.loadMesh(str(mesh_path))
    topology = mesh.topology
    vertices = topology.numValidVerts()
    faces = topology.numValidFaces()
    edges = topology.computeNotLoneUndirectedEdges()
    euler_characteristic = vertices + faces - edges

    non_manifold_edges = len(mm.findMultipleEdges(topology))
    mesh_copy = mm.Mesh(mesh)
    duplications = mm.std_vector_MeshBuilder_VertDuplication()
    mm.duplicateMultiHoleVertices(mesh_copy, 1, duplications)
    non_manifold_vertices = len({dup.srcVert.get() for dup in duplications})

    print(
        f"Topology for {Path(mesh_path).name}: "
        f"vertices={vertices}, faces={faces}, edges={edges}, \n"
        f"Euler characteristic (V + F - E)={euler_characteristic}\n"
        f"non-manifold edges={non_manifold_edges}, "
        f"non-manifold vertices={non_manifold_vertices}, "
    )
    return euler_characteristic


def remove_small_tunnels(
    input_path: str | Path,
    output_path: str | Path,
    max_tunnel_length: float = 1e-2,
) -> Path:
    """Remove tunnels up to the given length, measured in the mesh's model units."""
    if max_tunnel_length <= 0:
        raise ValueError("max_tunnel_length must be greater than zero")

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    original_obj = (
        input_path.read_text(encoding="utf-8")
        if input_path.suffix.lower() == ".obj"
        else None
    )
    mesh = mm.loadMesh(str(input_path))
    source_mesh = (
        mm.Mesh(mesh)
        if original_obj is not None
        and any(line.lstrip().startswith("vt ") for line in original_obj.splitlines())
        else None
    )
    settings = mm.DetectTunnelSettings()
    settings.maxTunnelLength = max_tunnel_length
    tunnel_faces = mm.detectTunnelFaces(mesh, settings)

    if tunnel_faces.count():
        print(f"Found {tunnel_faces.count()} holes")
        mm.expand(mesh.topology, tunnel_faces)
        mesh.deleteFaces(tunnel_faces)

        for edge in mesh.topology.findHoleRepresentiveEdges():
            left_face = mesh.topology.left(edge)
            right_face = mesh.topology.right(edge)
            if (
                left_face.valid() and tunnel_faces.test(left_face)
            ) or (
                right_face.valid() and tunnel_faces.test(right_face)
            ):
                mm.fillHole(mesh, edge)

    mm.saveMesh(mesh, str(output_path))
    if original_obj is not None:
        _copy_obj_material_assets(
            input_path, output_path, original_obj, source_mesh
        )
    return output_path
