"""
Creates the canonical apple mesh, the corresponding texture map with the tops and bottoms cut off
 - the body texture map is mapped to a dianmond diagonally in the image
 - the bottom texture map is in the upper left, the top in the bottom right. Both are circular

The mesh is a spherical lat-long parameterized mesh with longitude density adjusted to keep the
triangles roughly equilateral. The Y axis runs from the bottom (-Y) to the top (+Y).
 - the top mesh is created by placing a grid of vertices along the top hemisphere, with one vertex at the pole.
 - the bottom mesh has five downward-pointing cones distributed around the calyx and joined to the body ring.


"""

import argparse
import csv
import math
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
import zlib
from meshlib import mrmeshpy as mm
from MeshLabCleanup import _write_material_file


_TEXTURE_SIZE = 1024
_TOP_UV_CENTER = (0.9, 0.1)
_BOTTOM_UV_CENTER = (0.1, 0.9)
_UV_RADIUS = 0.1


@dataclass
class _MeshAssembly:
    points: list[tuple[float, float, float]] = field(default_factory=list)
    triangles: list[tuple[int, int, int]] = field(default_factory=list)


def _longitude_count(latitude: float, latitude_step: float) -> int:
    latitude_step_radians = math.radians(latitude_step)
    circumference = 2.0 * math.pi * math.cos(math.radians(latitude))
    return max(5, math.ceil(circumference / latitude_step_radians))


def _add_ring(ms: _MeshAssembly, latitude: float, n_samples: int) -> list[int]:
    latitude_radians = math.radians(latitude)
    radius = 0.5 * math.cos(latitude_radians)
    y = 0.5 * math.sin(latitude_radians)
    first = len(ms.points)
    for i in range(n_samples):
        longitude = 2.0 * math.pi * i / n_samples
        ms.points.append((radius * math.cos(longitude), y, -radius * math.sin(longitude)))
    return list(range(first, first + n_samples))


def _connect_rings(
    ms: _MeshAssembly,
    lower_ring: list[int],
    upper_ring: list[int],
) -> None:
    lower_count = len(lower_ring)
    upper_count = len(upper_ring)
    lower_index = 0
    upper_index = 0

    while lower_index < lower_count or upper_index < upper_count:
        lower_boundary = (lower_index + 1) * upper_count
        upper_boundary = (upper_index + 1) * lower_count
        lower_vertex = lower_ring[lower_index % lower_count]
        upper_vertex = upper_ring[upper_index % upper_count]

        if lower_boundary == upper_boundary:
            next_lower = lower_ring[(lower_index + 1) % lower_count]
            next_upper = upper_ring[(upper_index + 1) % upper_count]
            ms.triangles.append((lower_vertex, next_lower, next_upper))
            ms.triangles.append((lower_vertex, next_upper, upper_vertex))
            lower_index += 1
            upper_index += 1
        elif lower_boundary < upper_boundary:
            next_lower = lower_ring[(lower_index + 1) % lower_count]
            ms.triangles.append((lower_vertex, next_lower, upper_vertex))
            lower_index += 1
        else:
            next_upper = upper_ring[(upper_index + 1) % upper_count]
            ms.triangles.append((lower_vertex, next_upper, upper_vertex))
            upper_index += 1


def _create_apple_body(
    n_samples: int,
) -> tuple[_MeshAssembly, list[int], list[int], list[int]]:
    """Create the body of the apple as a lat-long parameterized mesh."""
    if n_samples < 3:
        raise ValueError("n_samples must be at least 3.")

    ms = _MeshAssembly()
    latitudes = [-70.0 + 140.0 * i / n_samples for i in range(n_samples + 1)]
    latitude_step = 140.0 / n_samples
    adjacent_ring_latitude = (latitudes[0] + latitudes[1]) / 2.0
    latitudes.insert(1, adjacent_ring_latitude)
    rings = [
        _add_ring(
            ms,
            latitude,
            _longitude_count(
                latitude,
                latitude_step / 2.0 if latitude == adjacent_ring_latitude else latitude_step,
            ),
        )
        for latitude in latitudes
    ]
    for lower_ring, upper_ring in zip(rings, rings[1:]):
        _connect_rings(ms, lower_ring, upper_ring)
    return ms, rings[-1], rings[0], [ring[0] for ring in rings]


def _create_apple_top(
    ms: _MeshAssembly,
    top_ring: list[int],
    n_samples: int,
) -> tuple[int, int]:
    """Subdivide the spherical cap with rings matching the body sampling density."""
    first_face = len(ms.triangles)
    x, y, z = ms.points[top_ring[0]]
    start_latitude = math.degrees(math.atan2(y, math.hypot(x, z)))
    body_step = 140.0 / n_samples
    intervals = max(1, math.ceil((90.0 - start_latitude) / body_step))
    cap_step = (90.0 - start_latitude) / intervals
    cap_ring = top_ring
    for index in range(1, intervals):
        latitude = start_latitude + index * cap_step
        next_ring = _add_ring(ms, latitude, _longitude_count(latitude, cap_step))
        _connect_rings(ms, cap_ring, next_ring)
        cap_ring = next_ring

    pole = len(ms.points)
    ms.points.append((0.0, 0.5, 0.0))
    for i, vertex in enumerate(cap_ring):
        ms.triangles.append((vertex, cap_ring[(i + 1) % len(cap_ring)], pole))
    return first_face, len(ms.triangles)


def _create_apple_bottom(
    ms: _MeshAssembly,
    bottom_ring: list[int],
) -> tuple[int, int]:
    """Create five cone shapes distributed around the bottom pole for the calyx."""
    first_face = len(ms.triangles)
    lobe_count = 5
    subdivision_levels = 4
    bottom_y = ms.points[bottom_ring[0]][1]
    bottom_radius = math.hypot(ms.points[bottom_ring[0]][0], ms.points[bottom_ring[0]][2])
    lobe_radius = bottom_radius * 0.45
    lobe_vertices = []
    lobe_edges: set[tuple[int, int]] = set()
    edge_subdivisions: dict[tuple[int, int, int], int] = {}

    def edge_vertex(apex: int, endpoint: int, level: int) -> int:
        if level == subdivision_levels:
            return endpoint
        start, end = sorted((apex, endpoint))
        canonical_level = level if apex == start else subdivision_levels - level
        edge = (start, end, canonical_level)
        vertex = edge_subdivisions.get(edge)
        if vertex is None:
            fraction = canonical_level / subdivision_levels
            point = ms.points[start]
            other = ms.points[end]
            position = tuple(
                (1.0 - fraction) * a + fraction * b
                for a, b in zip(point, other)
            )
            if (start, end) in lobe_edges:
                position = (
                    position[0],
                    position[1] + 0.03 * 4.0 * fraction * (1.0 - fraction),
                    position[2],
                )
            vertex = len(ms.points)
            ms.points.append(position)
            edge_subdivisions[edge] = vertex
        return vertex

    def subdivide_cone_face(
        apex: int,
        first: int,
        second: int,
        subdivide_base: bool = False,
    ) -> None:
        rows = [[apex]]
        final_level = subdivision_levels if subdivide_base else subdivision_levels - 1
        for level in range(1, final_level + 1):
            row = []
            for position in range(level + 1):
                if position == 0:
                    row.append(edge_vertex(apex, first, level))
                elif position == level:
                    row.append(edge_vertex(apex, second, level))
                elif level == subdivision_levels:
                    row.append(edge_vertex(first, second, position))
                else:
                    fraction = level / subdivision_levels
                    side = position / level
                    apex_point = ms.points[apex]
                    first_point = ms.points[first]
                    second_point = ms.points[second]
                    vertex = len(ms.points)
                    ms.points.append(tuple(
                        (1.0 - fraction) * apex_coordinate
                        + fraction * (
                            (1.0 - side) * first_coordinate
                            + side * second_coordinate
                        )
                        for apex_coordinate, first_coordinate, second_coordinate
                        in zip(apex_point, first_point, second_point)
                    ))
                    row.append(vertex)
            rows.append(row)

        for level in range(final_level):
            previous_row = rows[level]
            row = rows[level + 1]
            for position in range(level + 1):
                ms.triangles.append((
                    previous_row[position],
                    row[position],
                    row[position + 1],
                ))
                if position < level:
                    ms.triangles.append((
                        previous_row[position],
                        row[position + 1],
                        previous_row[position + 1],
                    ))

        if not subdivide_base:
            previous_row = rows[-1]
            for position in range(subdivision_levels - 1):
                ms.triangles.append((
                    previous_row[position],
                    first,
                    previous_row[position + 1],
                ))
            ms.triangles.append((previous_row[-1], first, second))

    for i in range(lobe_count):
        angle = 2.0 * math.pi * (i + 0.5) / lobe_count
        lobe_vertices.append(len(ms.points))
        ms.points.append((
            lobe_radius * math.cos(angle),
            bottom_y - 0.04,
            -lobe_radius * math.sin(angle),
        ))

    lobe_edges = {
        tuple(sorted((lobe_vertices[i], lobe_vertices[(i + 1) % lobe_count])))
        for i in range(lobe_count)
    }

    for i, lobe in enumerate(lobe_vertices):
        start = round(i * len(bottom_ring) / lobe_count)
        end = round((i + 1) * len(bottom_ring) / lobe_count)
        for edge in range(start, end):
            outer = bottom_ring[edge % len(bottom_ring)]
            next_outer = bottom_ring[(edge + 1) % len(bottom_ring)]
            subdivide_cone_face(lobe, next_outer, outer)

        boundary_vertex = bottom_ring[end % len(bottom_ring)]
        next_lobe = lobe_vertices[(i + 1) % lobe_count]
        subdivide_cone_face(boundary_vertex, lobe, next_lobe, subdivide_base=True)

    pole = len(ms.points)
    ms.points.append((0.0, bottom_y + 0.02, 0.0))
    for i, lobe in enumerate(lobe_vertices):
        next_lobe = lobe_vertices[(i + 1) % lobe_count]
        subdivide_cone_face(pole, next_lobe, lobe, subdivide_base=True)

    boundary_edges = {
        tuple(sorted((vertex, bottom_ring[(i + 1) % len(bottom_ring)])))
        for i, vertex in enumerate(bottom_ring)
    }
    for _ in range(2):
        midpoints: dict[tuple[int, int], int] = {}
        refined = []

        def midpoint(a: int, b: int) -> int:
            edge = tuple(sorted((a, b)))
            if edge not in midpoints:
                midpoints[edge] = len(ms.points)
                ms.points.append(tuple((x + y) / 2 for x, y in zip(ms.points[a], ms.points[b])))
            return midpoints[edge]

        for a, b, c in ms.triangles[first_face:]:
            edges = [(a, b), (b, c), (c, a)]
            fixed = [i for i, edge in enumerate(edges) if tuple(sorted(edge)) in boundary_edges]
            if fixed:
                if len(fixed) != 1:
                    raise ValueError("A calyx triangle has multiple boundary edges.")
                vertices = (a, b, c)
                i = fixed[0]
                a, b, c = vertices[i], vertices[(i + 1) % 3], vertices[(i + 2) % 3]
                bc, ca = midpoint(b, c), midpoint(c, a)
                refined.extend(((a, b, bc), (a, bc, ca), (ca, bc, c)))
            else:
                ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
                refined.extend(((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)))
        ms.triangles[first_face:] = refined
    return first_face, len(ms.triangles)


def _circle_uv(
    point: tuple[float, float, float],
    center: tuple[float, float],
    ring_radius: float,
) -> tuple[float, float]:
    scale = _UV_RADIUS / ring_radius
    return center[0] + point[0] * scale, center[1] - point[2] * scale


def _middle_uv(
    point: tuple[float, float, float],
    longitude: float,
    bottom_y: float,
    top_y: float,
) -> tuple[float, float]:
    height = (top_y - point[1]) / (top_y - bottom_y)
    return 0.5 + 0.5 * (longitude - height), 1.0 - 0.5 * (longitude + height)


def _bottom_uv(
    point: tuple[float, float, float], ring_radius: float,
) -> tuple[float, float]:
    u, v = _circle_uv(point, _BOTTOM_UV_CENTER, ring_radius)
    return u, 2 * _BOTTOM_UV_CENTER[1] - v


def _longitude(point: tuple[float, float, float]) -> float:
    return (math.atan2(-point[2], point[0]) % (2 * math.pi)) / (2 * math.pi)


def _write_texture_map(texture_path: Path) -> None:
    image = bytearray()
    for row in range(_TEXTURE_SIZE):
        v = 1.0 - (row + 0.5) / _TEXTURE_SIZE
        image.append(0)
        for column in range(_TEXTURE_SIZE):
            u = (column + 0.5) / _TEXTURE_SIZE
            color = (255, 255, 255)
            if abs(u - 0.5) + abs(v - 0.5) <= 0.5:
                color = (255, 0, 0)
            if (u - _TOP_UV_CENTER[0]) ** 2 + (v - _TOP_UV_CENTER[1]) ** 2 <= _UV_RADIUS**2:
                color = (0, 255, 0)
            if (u - _BOTTOM_UV_CENTER[0]) ** 2 + (v - _BOTTOM_UV_CENTER[1]) ** 2 <= _UV_RADIUS**2:
                color = (0, 0, 255)
            image.extend(color)

    def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        chunk = chunk_type + data
        return (
            struct.pack(">I", len(data))
            + chunk
            + struct.pack(">I", zlib.crc32(chunk))
        )

    texture_path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(
            b"IHDR",
            struct.pack(">IIBBBBB", _TEXTURE_SIZE, _TEXTURE_SIZE, 8, 2, 0, 0, 0),
        )
        + png_chunk(b"IDAT", zlib.compress(image))
        + png_chunk(b"IEND", b"")
    )


def _write_seam_csv(
    csv_path: Path,
    ms: _MeshAssembly,
    top_ring: list[int],
    bottom_ring: list[int],
    side_seam: list[int],
    top_radius: float,
    bottom_radius: float,
) -> None:
    """Write vertex, U and V rows for both rings and the four body-chart sides."""
    def write_rows(writer, vertices, uvs):
        writer.writerow(vertices)
        writer.writerow([uv[0] for uv in uvs])
        writer.writerow([uv[1] for uv in uvs])

    def write_ring_rows(
        writer,
        ring: list[int],
        center: tuple[float, float],
        radius: float,
    ) -> None:
        closed_ring = [ring[0], *reversed(ring[1:]), ring[0]]
        uvs = [
            _bottom_uv(ms.points[vertex], radius)
            if center == _BOTTOM_UV_CENTER else _circle_uv(ms.points[vertex], center, radius)
            for vertex in closed_ring
        ]
        write_rows(writer, closed_ring, uvs)

    top_side = [top_ring[0], *reversed(top_ring[1:]), top_ring[0]]
    right_side = list(reversed(side_seam))
    bottom_side = [*bottom_ring, bottom_ring[0]]
    left_side = side_seam
    bottom_y = ms.points[bottom_ring[0]][1]
    top_y = ms.points[top_ring[0]][1]

    side_longitudes = [
        [1.0, *[_longitude(ms.points[vertex]) for vertex in reversed(top_ring[1:])], 0.0],
        [0.0] * len(right_side),
        [*[_longitude(ms.points[vertex]) for vertex in bottom_ring], 1.0],
        [1.0] * len(left_side),
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as seam_file:
        writer = csv.writer(seam_file)
        write_ring_rows(writer, top_ring, _TOP_UV_CENTER, top_radius)
        write_ring_rows(writer, bottom_ring, _BOTTOM_UV_CENTER, bottom_radius)
        for side, longitudes in zip(
            (top_side, right_side, bottom_side, left_side), side_longitudes
        ):
            uvs = [
                _middle_uv(ms.points[vertex], angle, bottom_y, top_y)
                for vertex, angle in zip(side, longitudes)
            ]
            write_rows(writer, side, uvs)


def _write_obj_with_uv(
    ms: _MeshAssembly,
    top_ring: list[int],
    bottom_ring: list[int],
    side_seam: list[int],
    top_faces: tuple[int, int],
    bottom_faces: tuple[int, int],
    output_path: Path,
) -> None:
    top_radius = math.hypot(ms.points[top_ring[0]][0], ms.points[top_ring[0]][2])
    bottom_radius = math.hypot(
        ms.points[bottom_ring[0]][0],
        ms.points[bottom_ring[0]][2],
    )
    bottom_y = ms.points[bottom_ring[0]][1]
    top_y = ms.points[top_ring[0]][1]
    material_path = output_path.with_suffix(".mtl")
    texture_path = output_path.with_suffix(".png")
    seam_path = output_path.with_name(f"{output_path.stem}_seam.csv")
    _write_texture_map(texture_path)
    _write_material_file(material_path, texture_path.name)
    _write_seam_csv(
        seam_path,
        ms,
        top_ring,
        bottom_ring,
        side_seam,
        top_radius,
        bottom_radius,
    )

    with output_path.open("w", encoding="utf-8", newline="\n") as obj_file:
        obj_file.write(f"mtllib {material_path.name}\n")
        for x, y, z in ms.points:
            obj_file.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
        obj_file.write("usemtl apple\n")
        for face_index, face in enumerate(ms.triangles):
            is_top = top_faces[0] <= face_index < top_faces[1]
            is_bottom = bottom_faces[0] <= face_index < bottom_faces[1]
            longitudes = [_longitude(ms.points[vertex]) for vertex in face]
            if not is_top and not is_bottom and max(longitudes) - min(longitudes) > 0.5:
                longitudes = [
                    longitude + 1.0 if longitude < 0.5 else longitude
                    for longitude in longitudes
                ]

            face_uvs = []
            for corner, vertex in enumerate(face):
                point = ms.points[vertex]
                if is_top:
                    uv = _circle_uv(point, _TOP_UV_CENTER, top_radius)
                elif is_bottom:
                    uv = _bottom_uv(point, bottom_radius)
                else:
                    uv = _middle_uv(point, longitudes[corner], bottom_y, top_y)
                face_uvs.append(uv)

            vt_indices = [3 * face_index + corner + 1 for corner in range(3)]
            for u, v in face_uvs:
                obj_file.write(f"vt {u:.9g} {v:.9g}\n")
            corners = " ".join(
                f"{vertex + 1}/{vt_index}"
                for vertex, vt_index in zip(face, vt_indices)
            )
            obj_file.write(f"f {corners}\n")


def _to_mesh(ms: _MeshAssembly) -> mm.Mesh:
    vertices = mm.std_vector_Vector3_float(
        [mm.Vector3f(*point) for point in ms.points]
    )
    triangles = mm.std_vector_std_array_Id_VertTag_3ul(
        [
            mm.ThreeVertIds([mm.VertId(vertex) for vertex in triangle])
            for triangle in ms.triangles
        ]
    )
    return mm.Mesh.fromTriangles(mm.VertCoords(vertices), mm.Triangulation(triangles))


def _create_apple(n_samples: int, output_path: Path) -> mm.Mesh:
    ms, top_ring, bottom_ring, side_seam = _create_apple_body(n_samples)
    top_faces = _create_apple_top(ms, top_ring, n_samples)
    bottom_faces = _create_apple_bottom(ms, bottom_ring)
    mesh = _to_mesh(ms)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_obj_with_uv(
        ms,
        top_ring,
        bottom_ring,
        side_seam,
        top_faces,
        bottom_faces,
        output_path,
    )
    return mesh


def _main(argv=None):
    parser = argparse.ArgumentParser(description='Generate a procedural apple mesh with a calyx.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--variety', default='random', help='random or a named cultivar')
    parser.add_argument('--n_samples', default=64, type=int)
    parser.add_argument('--blemishes', type=float, default=0.20)
    parser.add_argument('--output-file', default='/Users/cindygrimm/PycharmProjects/data/ScannedModels/apple.obj', help='Path to the output OBJ mesh')
    args = parser.parse_args(argv)

    _create_apple(n_samples=args.n_samples, output_path=Path(args.output_file))


if __name__ == '__main__':
    # Blender's arguments precede --; ignore them when running from the Text Editor.
    _main(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
