"""
Creates the canonical apple mesh, the corresponding texture map with the tops and bottoms cut off
 - the body texture map is mapped to a dianmond diagonally in the image
 - the bottom texture map is in the upper left, the top in the bottom right. Both are circular

The mesh is a spherical lat-long parameterized mesh with longitude density adjusted to keep the
triangles roughly equilateral. The Y axis runs from the bottom (-Y) to the top (+Y).
The top cap uses latitude rings; the bottom has latitude rings down to a
calyx boundary at -85 degrees and a Cartesian grid inside that boundary.


"""

import argparse
import math
import struct
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
import zlib
from meshlib import mrmeshpy as mm
from filename_fixing import _write_material_file
from seam_topology import SeamInfo, SeamTopology
import numpy as np
from scipy.spatial import Delaunay
from MeshLabCleanup import check_topology
from remove_calyx import _pack_with_seams


_TEXTURE_SIZE = 1024
_TOP_UV_CENTER = (0.9, 0.1)
_BOTTOM_UV_CENTER = (0.1, 0.9)
_UV_RADIUS = 0.1
_UV_CALYX_RADIUS = 0.05
_CENTER_DOT_RADIUS = 0.012
_AXIS_LINE_HALF_WIDTH = 0.004
_SEAM_LINE_HALF_WIDTH = 0.004
_PEAK_DOT_RADIUS = 0.0015
_VERTEX_DOT_RADIUS = 0.003


@dataclass
class _MeshAssembly:
    points: list[tuple[float, float, float]] = field(default_factory=list)
    triangles: list[tuple[int, int, int]] = field(default_factory=list)
    calyx_ring: list[int] = field(default_factory=list)


def _longitude_count(latitude: float, latitude_step: float) -> int:
    latitude_step_radians = math.radians(latitude_step)
    circumference = 2.0 * math.pi * math.cos(math.radians(latitude))
    return max(5, math.ceil(circumference / latitude_step_radians))


def _body_longitude_count(latitude: float, latitude_step: float) -> int:
    circumference = 2.0 * math.pi * math.cos(math.radians(latitude))
    return max(3, round(circumference / math.radians(latitude_step)))


def _body_latitude_step(n_samples: int) -> float:
    target_step = 360.0 * math.cos(math.radians(70.0)) / n_samples
    return 140.0 / math.ceil(140.0 / target_step)


def _add_ring(ms: _MeshAssembly, latitude: float, n_samples: int) -> list[int]:
    latitude_radians = math.radians(latitude)
    radius = 0.5 * math.cos(latitude_radians)
    y = 0.5 * math.sin(latitude_radians)
    first = len(ms.points)
    for i in range(n_samples):
        longitude = 2.0 * math.pi * i / n_samples
        ms.points.append((radius * math.cos(longitude), y, radius * math.sin(longitude)))
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
            ms.triangles.append((lower_vertex, next_upper, next_lower))
            ms.triangles.append((lower_vertex, upper_vertex, next_upper))
            lower_index += 1
            upper_index += 1
        elif lower_boundary < upper_boundary:
            next_lower = lower_ring[(lower_index + 1) % lower_count]
            ms.triangles.append((lower_vertex, upper_vertex, next_lower))
            lower_index += 1
        else:
            next_upper = upper_ring[(upper_index + 1) % upper_count]
            ms.triangles.append((lower_vertex, upper_vertex, next_upper))
            upper_index += 1


def _create_apple_body(
    n_samples: int,
) -> tuple[_MeshAssembly, list[int], list[int], list[int]]:
    """Create a lat-long body with n_samples vertices on each seam ring."""
    if n_samples < 3:
        raise ValueError("n_samples must be at least 3.")

    ms = _MeshAssembly()
    endpoint_latitude = 70.0
    latitude_step = _body_latitude_step(n_samples)
    latitude_intervals = round(140.0 / latitude_step)
    latitudes = [
        -endpoint_latitude + latitude_step * index
        for index in range(latitude_intervals + 1)
    ]
    rings = []
    for index, latitude in enumerate(latitudes):
        longitude_count = (
            n_samples
            if index in (0, latitude_intervals)
            else _body_longitude_count(latitude, latitude_step)
        )
        rings.append(_add_ring(ms, latitude, longitude_count))
    for lower_ring, upper_ring in zip(rings, rings[1:]):
        _connect_rings(ms, lower_ring, upper_ring)
    return ms, rings[-1], rings[0], [ring[0] for ring in rings]


def _create_apple_top(
    ms: _MeshAssembly,
    top_ring: list[int],
    n_samples: int,
) -> tuple[int, int, int]:
    """Subdivide the spherical cap with rings matching the body sampling density."""
    first_face = len(ms.triangles)
    x, y, z = ms.points[top_ring[0]]
    start_latitude = math.degrees(math.atan2(y, math.hypot(x, z)))
    body_step = _body_latitude_step(n_samples)
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
        ms.triangles.append((vertex, pole, cap_ring[(i + 1) % len(cap_ring)]))
    return first_face, len(ms.triangles), pole


def _create_apple_bottom(
    ms: _MeshAssembly,
    bottom_ring: list[int],
) -> tuple[int, int, int]:
    """Connect the body to the calyx ring, then fill it with a projected grid."""
    first_face = len(ms.triangles)
    boundary_edge_lengths = [
        math.dist(
            ms.points[vertex],
            ms.points[bottom_ring[(index + 1) % len(bottom_ring)]],
        )
        for index, vertex in enumerate(bottom_ring)
    ]
    target_edge_length = sum(boundary_edge_lengths) / len(boundary_edge_lengths)
    bottom_latitude = math.degrees(
        math.atan2(
            ms.points[bottom_ring[0]][1],
            math.hypot(ms.points[bottom_ring[0]][0], ms.points[bottom_ring[0]][2]),
        )
    )
    calyx_latitude = -85.0
    intervals = max(1, math.ceil(
        0.5 * math.radians(bottom_latitude - calyx_latitude) / target_edge_length
    ))
    previous_ring = bottom_ring
    for index in range(1, intervals + 1):
        latitude = bottom_latitude + (calyx_latitude - bottom_latitude) * index / intervals
        circumference = math.pi * math.cos(math.radians(latitude))
        ring_count = max(3, round(circumference / target_edge_length))
        ring = _add_ring(ms, latitude, ring_count)
        _connect_rings(ms, ring, previous_ring)
        previous_ring = ring
    ms.calyx_ring = previous_ring

    boundary = np.asarray(
        [(ms.points[vertex][0], ms.points[vertex][2]) for vertex in ms.calyx_ring],
        dtype=np.float64,
    )
    radius = max(np.linalg.norm(boundary, axis=1))
    target_edge_length = min(target_edge_length, radius)
    grid_radius = radius - target_edge_length / 2
    if grid_radius <= 0:
        raise ValueError("The bottom boundary is too small for the requested grid spacing.")
    n_grid_points = max(3, math.ceil(grid_radius / target_edge_length))
    grid_spacing = grid_radius / n_grid_points
    for _ in range(n_grid_points):
        grid_extent = math.ceil(grid_radius / grid_spacing)
        grid_axis = np.arange(-grid_extent, grid_extent + 1) * grid_spacing
        grid_points = []
        if target_edge_length < 0.7 * radius:
            for x in grid_axis:
                for z in grid_axis:
                    if x * x + z * z >= grid_radius * grid_radius:
                        continue
                    crosses = [
                        (boundary[(index + 1) % len(boundary), 0] - boundary[index, 0])
                        * (z - boundary[index, 1])
                        - (boundary[(index + 1) % len(boundary), 1] - boundary[index, 1])
                        * (x - boundary[index, 0])
                        for index in range(len(boundary))
                    ]
                    if all(value >= -1e-12 for value in crosses) or all(
                        value <= 1e-12 for value in crosses
                    ):
                        grid_points.append((x, z))
        grid_points.append((0.0, 0.0))
        unique_points = np.unique(np.vstack((boundary, grid_points)), axis=0)
        simplices = Delaunay(unique_points).simplices
        edge_indices = np.unique(
            np.sort(
                np.concatenate((
                    simplices[:, [0, 1]],
                    simplices[:, [1, 2]],
                    simplices[:, [2, 0]],
                )),
                axis=1,
            ),
            axis=0,
        )
        coordinates = unique_points
        heights = -np.sqrt(np.maximum(0.0, 0.25 - np.sum(coordinates**2, axis=1)))
        surface_points = np.column_stack((coordinates[:, 0], heights, coordinates[:, 1]))
        edge_lengths = np.linalg.norm(
            surface_points[edge_indices[:, 0]] - surface_points[edge_indices[:, 1]],
            axis=1,
        )
        measured_length = float(edge_lengths.mean())
        if (target_edge_length >= 0.7 * radius
                or abs(measured_length - target_edge_length) <= target_edge_length * 0.03):
            break
        grid_spacing = min(
            grid_spacing * target_edge_length / measured_length,
            grid_radius * 0.8,
        )

    boundary_indices = np.array(
        [int(np.flatnonzero(np.all(unique_points == point, axis=1))[0]) for point in boundary]
    )
    boundary_vertex_by_point = {
        int(point_index): int(vertex)
        for point_index, vertex in zip(boundary_indices, ms.calyx_ring)
    }
    point_indices = []
    for point_index, point in enumerate(unique_points):
        if point_index in boundary_vertex_by_point:
            point_indices.append(boundary_vertex_by_point[point_index])
            continue
        x, z = point
        if abs(x) <= 1e-12 and abs(z) <= 1e-12:
            point_indices.append(-1)
        else:
            y = -math.sqrt(max(0.0, 0.25 - x * x - z * z))
            point_indices.append(len(ms.points))
            ms.points.append((float(x), y, float(z)))
    south_pole = len(ms.points)
    ms.points.append((0.0, -0.5, 0.0))
    for point_index, vertex in enumerate(point_indices):
        if vertex == -1:
            point_indices[point_index] = south_pole

    for simplex in simplices:
        face = [point_indices[int(index)] for index in simplex]
        a, b, c = (np.asarray(ms.points[vertex]) for vertex in face)
        normal = np.cross(b - a, c - a)
        if float(normal @ (a + b + c)) < 0:
            face[1], face[2] = face[2], face[1]
        ms.triangles.append(tuple(face))

    return first_face, len(ms.triangles), south_pole


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
    point: tuple[float, float, float], ring_radius: float, calyx_radius: float,
) -> tuple[float, float]:
    """Map the calyx and surrounding annulus to their respective UV radii."""
    radius = math.hypot(point[0], point[2])
    if radius <= calyx_radius:
        scale = _UV_CALYX_RADIUS / calyx_radius
    else:
        uv_radius = _UV_CALYX_RADIUS + (radius - calyx_radius) * (
            _UV_RADIUS - _UV_CALYX_RADIUS
        ) / (ring_radius - calyx_radius)
        scale = uv_radius / radius
    return _BOTTOM_UV_CENTER[0] + point[0] * scale, _BOTTOM_UV_CENTER[1] + point[2] * scale


def _longitude(point: tuple[float, float, float]) -> float:
    return (math.atan2(-point[2], point[0]) % (2 * math.pi)) / (2 * math.pi)


def _write_texture_map(
    texture_path: Path,
    peak_uvs: tuple[tuple[float, float], ...] = (),
    vertex_uvs: tuple[tuple[float, float], ...] = (),
) -> None:
    vertex_dot_pixels = set()
    for vertex_u, vertex_v in vertex_uvs:
        column_min = max(
            0, math.ceil((vertex_u - _VERTEX_DOT_RADIUS) * _TEXTURE_SIZE - 0.5)
        )
        column_max = min(
            _TEXTURE_SIZE - 1,
            math.floor((vertex_u + _VERTEX_DOT_RADIUS) * _TEXTURE_SIZE - 0.5),
        )
        row_min = max(
            0,
            math.ceil((1 - vertex_v - _VERTEX_DOT_RADIUS) * _TEXTURE_SIZE - 0.5),
        )
        row_max = min(
            _TEXTURE_SIZE - 1,
            math.floor((1 - vertex_v + _VERTEX_DOT_RADIUS) * _TEXTURE_SIZE - 0.5),
        )
        for row in range(row_min, row_max + 1):
            pixel_v = 1.0 - (row + 0.5) / _TEXTURE_SIZE
            for column in range(column_min, column_max + 1):
                pixel_u = (column + 0.5) / _TEXTURE_SIZE
                if (
                    (pixel_u - vertex_u) ** 2 + (pixel_v - vertex_v) ** 2
                    <= _VERTEX_DOT_RADIUS**2
                ):
                    vertex_dot_pixels.add((row, column))

    image = bytearray()
    for row in range(_TEXTURE_SIZE):
        v = 1.0 - (row + 0.5) / _TEXTURE_SIZE
        image.append(0)
        for column in range(_TEXTURE_SIZE):
            u = (column + 0.5) / _TEXTURE_SIZE
            color = (255, 255, 255)
            if abs(u - 0.5) + abs(v - 0.5) <= 0.5:
                color = (255, 0, 0)
            # The side seam maps to the lower-right (u - v = 0.5) and upper-left (v - u = 0.5) diamond edges.
            if (
                abs(abs(u - v) - 0.5) <= _SEAM_LINE_HALF_WIDTH * math.sqrt(2)
                and abs(u - 0.5) + abs(v - 0.5) <= 0.5 + 2 * _SEAM_LINE_HALF_WIDTH
            ):
                color = (0, 0, 0)
            # The top ring seam is the upper-right edge (u + v = 1.5), the bottom ring seam the lower-left edge (u + v = 0.5).
            if abs(u - 0.5) + abs(v - 0.5) <= 0.5 + 2 * _SEAM_LINE_HALF_WIDTH:
                if abs(u + v - 1.5) <= _SEAM_LINE_HALF_WIDTH * math.sqrt(2):
                    color = (144, 238, 144)
                elif abs(u + v - 0.5) <= _SEAM_LINE_HALF_WIDTH * math.sqrt(2):
                    color = (173, 216, 230)
            if (u - _TOP_UV_CENTER[0]) ** 2 + (v - _TOP_UV_CENTER[1]) ** 2 <= _UV_RADIUS**2:
                color = (0, 255, 0)
            if (u - _BOTTOM_UV_CENTER[0]) ** 2 + (v - _BOTTOM_UV_CENTER[1]) ** 2 <= _UV_RADIUS**2:
                color = (100, 50, 228)
                if (u - _BOTTOM_UV_CENTER[0]) ** 2 + (v - _BOTTOM_UV_CENTER[1]) ** 2 <= _UV_CALYX_RADIUS**2:
                    color = (50, 100, 228)
            for center in (_TOP_UV_CENTER, _BOTTOM_UV_CENTER):
                du, dv = u - center[0], v - center[1]
                if du * du + dv * dv <= _CENTER_DOT_RADIUS**2:
                    color = (128, 128, 128)
                elif 0.0 < du <= _UV_RADIUS and abs(dv) <= _AXIS_LINE_HALF_WIDTH:
                    color = (0, 0, 0)
            if (row, column) in vertex_dot_pixels:
                color = (128, 128, 128)
            for peak_u, peak_v in peak_uvs:
                if (u - peak_u) ** 2 + (v - peak_v) ** 2 <= _PEAK_DOT_RADIUS**2:
                    color = (0, 255, 255)
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


def _seam_boundary_records(
    ms: _MeshAssembly,
    top_ring: list[int],
    bottom_ring: list[int],
    side_seam: list[int],
    top_radius: float,
    bottom_radius: float,
) -> list[tuple[list[int], list[tuple[float, float]]]]:
    def ring_record(
        ring: list[int],
        center: tuple[float, float],
        radius: float,
    ):
        closed_ring = [ring[0], *reversed(ring[1:]), ring[0]]
        uvs = [_circle_uv(ms.points[vertex], center, radius) for vertex in closed_ring]
        if center == _BOTTOM_UV_CENTER:
            uvs = [(u, 2 * center[1] - v) for u, v in uvs]
        return closed_ring, uvs

    top_side = [top_ring[0], *reversed(top_ring[1:]), top_ring[0]]
    right_side = side_seam
    bottom_side = [*bottom_ring, bottom_ring[0]]
    left_side = list(reversed(side_seam))
    bottom_y = ms.points[bottom_ring[0]][1]
    top_y = ms.points[top_ring[0]][1]

    side_longitudes = [
        [1.0, *[_longitude(ms.points[vertex]) for vertex in reversed(top_ring[1:])], 0.0],
        [0.0] * len(right_side),
        [*[_longitude(ms.points[vertex]) for vertex in bottom_ring], 1.0],
        [1.0] * len(left_side),
    ]
    records = [
        ring_record(top_ring, _TOP_UV_CENTER, top_radius),
        ring_record(bottom_ring, _BOTTOM_UV_CENTER, bottom_radius),
    ]
    for side, longitudes in zip((top_side, right_side, bottom_side, left_side), side_longitudes):
        uvs = [
            _middle_uv(ms.points[vertex], angle, bottom_y, top_y)
            for vertex, angle in zip(side, longitudes)
        ]
        records.append((side, uvs))
    return records


def _create_topology(
    ms: _MeshAssembly,
    top_ring: list[int],
    bottom_ring: list[int],
    side_seam: list[int],
    top_faces: tuple[int, int, int],
    bottom_faces: tuple[int, int, int],
    output_path: Path,
) -> None:
    top_radius = math.hypot(ms.points[top_ring[0]][0], ms.points[top_ring[0]][2])
    bottom_radius = math.hypot(
        ms.points[bottom_ring[0]][0],
        ms.points[bottom_ring[0]][2],
    )
    if not ms.calyx_ring:
        raise ValueError("The bottom cap must have a calyx ring.")
    calyx_point = ms.points[ms.calyx_ring[0]]
    calyx_radius = math.hypot(calyx_point[0], calyx_point[2])
    # Add the calyx ring to the seam topology records
    if not 0 < calyx_radius < bottom_radius or not 0 < _UV_CALYX_RADIUS < _UV_RADIUS:
        raise ValueError("The calyx radii must be positive and smaller than their bottom boundary radii.")
    seam_path = output_path.with_name(f"{output_path.stem}_seam_topology.json")
    records = _seam_boundary_records(ms, top_ring, bottom_ring, side_seam, top_radius, bottom_radius)
    top_point = ms.points[top_faces[2]]
    bottom_point = ms.points[bottom_faces[2]]
    topology = SeamTopology(
        ymin=bottom_point[1], ymax=top_point[1], top_point=top_point, bottom_point=bottom_point,
        top_shoulder_radius=top_radius, bottom_shoulder_radius=bottom_radius,
    )
    topology.set_boundary_paths(records)
    uv_copies = {}
    for ids, uvs in records:
        for vertex, uv in zip(ids, uvs):
            copies = uv_copies.setdefault(vertex, [])
            if uv not in copies:
                copies.append(uv)
    for name, ids, center, radius in (
        ("top", top_ring, top_point, top_radius),
        ("bottom", bottom_ring, bottom_point, bottom_radius),
        ("side", side_seam, None, None),
    ):
        topology.add_seam(name, SeamInfo(
            name, ids, [ms.points[vertex] for vertex in ids],
            [uv_copies[vertex] for vertex in ids], center, radius,
        ))
    topology.add_seam("calyx", SeamInfo(
        "calyx", ms.calyx_ring, [ms.points[vertex] for vertex in ms.calyx_ring],
        [[_bottom_uv(ms.points[vertex], bottom_radius, calyx_radius)] for vertex in ms.calyx_ring],
        bottom_point, calyx_radius,
    ))
    topology.top_vertex = top_faces[2]
    topology.bottom_vertex = bottom_faces[2]
    topology.write_to_file(seam_path, len(ms.points))


def _write_obj_with_uv(
    ms: _MeshAssembly,
    top_ring: list[int],
    bottom_ring: list[int],
    side_seam: list[int],
    top_faces: tuple[int, int, int],
    bottom_faces: tuple[int, int, int],
    output_path: Path,
) -> None:
    _check_mesh_topology(ms)
    top_radius = math.hypot(ms.points[top_ring[0]][0], ms.points[top_ring[0]][2])
    bottom_radius = math.hypot(
        ms.points[bottom_ring[0]][0],
        ms.points[bottom_ring[0]][2],
    )
    if not ms.calyx_ring:
        raise ValueError("The bottom cap must have a calyx ring.")
    calyx_point = ms.points[ms.calyx_ring[0]]
    calyx_radius = math.hypot(calyx_point[0], calyx_point[2])
    # Add the calyx ring to the seam topology records
    if not 0 < calyx_radius < bottom_radius or not 0 < _UV_CALYX_RADIUS < _UV_RADIUS:
        raise ValueError("The calyx radii must be positive and smaller than their bottom boundary radii.")
    bottom_y = ms.points[bottom_ring[0]][1]
    top_y = ms.points[top_ring[0]][1]
    material_path = output_path.with_suffix(".mtl")
    texture_path = output_path.with_suffix(".jpg")
    lowest_y = min(point[1] for point in ms.points)
    peak_uvs = tuple(
        _bottom_uv(point, bottom_radius, calyx_radius)
        for point in ms.points
        if point[1] <= lowest_y + 1e-6
    )
    seam_vertices = set(top_ring) | set(bottom_ring) | set(side_seam)
    vertex_uvs: dict[int, tuple[float, float]] = {}
    _write_material_file(material_path, texture_path.name)
    uv_indices: dict[tuple[int, str, str], int] = {}
    with output_path.open("w", encoding="utf-8", newline="\n") as obj_file:
        obj_file.write(f"mtllib {material_path.name}\n")
        for x, y, z in ms.points:
            obj_file.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
        obj_file.write("usemtl material_0\n")
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
                    uv = _bottom_uv(point, bottom_radius, calyx_radius)
                else:
                    uv = _middle_uv(point, longitudes[corner], bottom_y, top_y)
                face_uvs.append(uv)
                if vertex not in seam_vertices:
                    vertex_uvs.setdefault(vertex, uv)

            vt_indices = []
            for vertex, (u, v) in zip(face, face_uvs):
                key = (vertex, f"{u:.9g}", f"{v:.9g}")
                if key not in uv_indices:
                    uv_indices[key] = len(uv_indices) + 1
                    obj_file.write(f"vt {key[1]} {key[2]}\n")
                vt_indices.append(uv_indices[key])
            corners = " ".join(
                f"{vertex + 1}/{vt_index}"
                for vertex, vt_index in zip(face, vt_indices)
            )
            obj_file.write(f"f {corners}\n")
    _write_texture_map(texture_path, peak_uvs, tuple(vertex_uvs.values()))


def _compact_vertices(
    ms: _MeshAssembly,
    top_ring: list[int],
    bottom_ring: list[int],
    side_seam: list[int],
    top_faces: tuple[int, int, int],
    bottom_faces: tuple[int, int, int],
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    used = {vertex for face in ms.triangles for vertex in face}
    if any(vertex < 0 or vertex >= len(ms.points) for vertex in used):
        raise ValueError("A face references an invalid vertex.")
    vertex_map = {old: new for new, old in enumerate(sorted(used))}
    for vertices in (top_ring, bottom_ring, side_seam, ms.calyx_ring):
        if any(vertex not in vertex_map for vertex in vertices):
            raise ValueError("A seam references an unused or invalid vertex.")
    if top_faces[2] not in vertex_map or bottom_faces[2] not in vertex_map:
        raise ValueError("A pole references an unused or invalid vertex.")
    ms.points = [ms.points[vertex] for vertex in sorted(used)]
    ms.triangles = [tuple(vertex_map[vertex] for vertex in face) for face in ms.triangles]
    for vertices in (top_ring, bottom_ring, side_seam, ms.calyx_ring):
        vertices[:] = [vertex_map[vertex] for vertex in vertices]
    return (
        (top_faces[0], top_faces[1], vertex_map[top_faces[2]]),
        (bottom_faces[0], bottom_faces[1], vertex_map[bottom_faces[2]]),
    )


def _check_mesh_topology(ms: _MeshAssembly) -> None:
    used = set()
    edges = Counter()
    directed_edges = Counter()
    links: dict[int, dict[int, set[int]]] = {}
    for face in ms.triangles:
        if len(face) != 3 or len(set(face)) != 3:
            raise ValueError("Mesh faces must have three distinct vertices.")
        if any(vertex < 0 or vertex >= len(ms.points) for vertex in face):
            raise ValueError("A face references an invalid vertex.")
        used.update(face)
        edges.update(tuple(sorted((face[i], face[(i + 1) % 3]))) for i in range(3))
        directed_edges.update((face[i], face[(i + 1) % 3]) for i in range(3))
        for i, vertex in enumerate(face):
            first, second = face[(i + 1) % 3], face[(i + 2) % 3]
            link = links.setdefault(vertex, {})
            link.setdefault(first, set()).add(second)
            link.setdefault(second, set()).add(first)
    if len(used) != len(ms.points):
        raise ValueError("Mesh contains unused vertices.")
    if not edges or any(count != 2 for count in edges.values()):
        raise ValueError("Every mesh edge must have exactly two incident faces.")
    if any(directed_edges[(a, b)] != 1 or directed_edges[(b, a)] != 1 for a, b in edges):
        raise ValueError("Adjacent mesh faces must have consistent winding.")
    for link in links.values():
        visited = set()
        pending = [next(iter(link))]
        while pending:
            vertex = pending.pop()
            if vertex not in visited:
                visited.add(vertex)
                pending.extend(link[vertex] - visited)
        if len(visited) != len(link):
            raise ValueError("Mesh contains a non-manifold vertex.")
    if len(ms.points) - len(edges) + len(ms.triangles) != 2:
        raise ValueError("Mesh must have Euler characteristic 2.")
    points = np.asarray(ms.points, dtype=np.float64)
    triangles = points[np.asarray(ms.triangles)]
    signed_volume = np.einsum(
        "ij,ij->", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])
    )
    if signed_volume <= 0:
        raise ValueError("Mesh faces must be oriented outward.")


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
    side_seam.reverse()

    top_faces = _create_apple_top(ms, top_ring, n_samples)
    bottom_faces = _create_apple_bottom(ms, bottom_ring)
    top_faces, bottom_faces = _compact_vertices(
        ms, top_ring, bottom_ring, side_seam, top_faces, bottom_faces
    )
    _check_mesh_topology(ms)
    mesh = _to_mesh(ms)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_obj_with_uv(
        ms,
        top_ring,
        bottom_ring,
        side_seam,
        top_faces,
        bottom_faces,
        output_path=output_path,
    )
    _create_topology(ms, top_ring, bottom_ring, side_seam, top_faces, bottom_faces, output_path)
    return mesh


def _main(argv=None):
    parser = argparse.ArgumentParser(description='Generate a procedural apple mesh with a calyx.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--variety', default='random', help='random or a named cultivar')
    parser.add_argument('--n_samples', default=64, type=int)
    parser.add_argument('--blemishes', type=float, default=0.20)
    parser.add_argument('--output-file', default='/Users/cindygrimm/PycharmProjects/data/ScannedModels/Canonical/apple.obj', help='Path to the output OBJ mesh')
    args = parser.parse_args(argv)

    output_path = Path(args.output_file)
    for n_samples, tag in ((15, "lowres"), (33, "hires"), (60, "fullres")):
        output_path_tagged = output_path.with_name(output_path.stem + f"_{tag}.obj")
        mesh = _create_apple(n_samples=n_samples, output_path=output_path_tagged)
        check_topology(output_path_tagged, output_path_tagged, mesh=mesh)


if __name__ == '__main__':
    # Blender's arguments precede --; ignore them when running from the Text Editor.
    _main(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
