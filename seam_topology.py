"""
Store information about the following:
Top and bottom vertex locations
Top and bottom closest vertex ids (id for stem and calyx patching)"""

import math

class SeamInfo:
    """ Stores the vertex ids, their 3D locations, and their uv coordinates for one seam. Also includes the seam name (one of top, bottom, side, calyx, stem)
    Keeps the center used to create the seam (for the top, bottom, calyx, and stem) as well as the radius of the seam ring.
    uv_coords contains one list of UV pairs per vertex, including all chart copies.
    The side seam has no ring center or radius, so both are None.
    Can calculate the average of the seam vertex positions and the span of the height to determine the vertical extent of the seam.
    Supports reading and writing to a .json file.
    Supports renumbering the vertices if the vertex indices change in the mesh.
    """
    def __init__(self, name: str, vertex_ids: list, vertex_locations: list, uv_coords: list,
                 center: tuple | None, radius: float | None):
        self.name = name
        self.vertex_ids = vertex_ids
        self.vertex_locations = vertex_locations
        self.uv_coords = uv_coords
        self.center = center
        self.radius = radius

    def average_position(self) -> tuple:
        if not self.vertex_locations:
            return (0.0, 0.0, 0.0)
        x = sum(v[0] for v in self.vertex_locations) / len(self.vertex_locations)
        y = sum(v[1] for v in self.vertex_locations) / len(self.vertex_locations)
        z = sum(v[2] for v in self.vertex_locations) / len(self.vertex_locations)
        return (x, y, z)

    def vertical_extent(self) -> float:
        if not self.vertex_locations:
            return 0.0
        ys = [v[1] for v in self.vertex_locations]
        return max(ys) - min(ys)

    def renumber_vertices(self, vertex_map: dict) -> None:
        """Renumber the vertex ids based on a provided mapping from old to new ids."""
        self.vertex_ids = [vertex_map.get(vid, vid) for vid in self.vertex_ids]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "vertex_ids": self.vertex_ids,
            "vertex_locations": self.vertex_locations,
            "uv_coords": self.uv_coords,
            "center": self.center,
            "radius": self.radius,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SeamInfo":
        return cls(
            name=data["name"],
            vertex_ids=data["vertex_ids"],
            vertex_locations=data["vertex_locations"],
            uv_coords=data["uv_coords"],
            center=data["center"],
            radius=data["radius"],
        )


class SeamTopology:
    """ Stores the ymin/max, the top and bottom points initially selected by the user, and the seam information for the seams of the mesh.  
    Supports reading and writing to a .json file.
    Supports adding a seam with a name and SeamInfo object.
    """
    def __init__(self, ymin: float, ymax: float, top_point: tuple, bottom_point: tuple, seams: dict | None = None,
                 top_shoulder_radius: float | None = None, bottom_shoulder_radius: float | None = None,
                 boundary_paths: list[dict] | None = None):
        self.ymin = ymin
        self.ymax = ymax
        self.top_point = top_point
        self.bottom_point = bottom_point
        self.seams = {} if seams is None else seams
        self.top_shoulder_radius = top_shoulder_radius
        self.bottom_shoulder_radius = bottom_shoulder_radius
        self.boundary_paths = [] if boundary_paths is None else boundary_paths

    def to_dict(self) -> dict:
        return {
            "ymin": self.ymin,
            "ymax": self.ymax,
            "top_point": self.top_point,
            "bottom_point": self.bottom_point,
            "top_shoulder_radius": self.top_shoulder_radius,
            "bottom_shoulder_radius": self.bottom_shoulder_radius,
            "seams": {name: seam.to_dict() for name, seam in self.seams.items()},
            "boundary_paths": self.boundary_paths,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SeamTopology":
        seams = {name: SeamInfo.from_dict(seam) for name, seam in data["seams"].items()}
        return cls(
            ymin=data["ymin"],
            ymax=data["ymax"],
            top_point=data["top_point"],
            bottom_point=data["bottom_point"],
            seams=seams,
            top_shoulder_radius=data.get("top_shoulder_radius"),
            bottom_shoulder_radius=data.get("bottom_shoulder_radius"),
            boundary_paths=data.get("boundary_paths"),
        )

    def set_boundary_paths(self, records) -> None:
        """Store top and bottom cap paths followed by the four body-chart paths."""
        paths = []
        for vertex_ids, coordinates in records:
            ids = [int(vertex_id) for vertex_id in vertex_ids]
            uvs = [[float(value) for value in uv] for uv in coordinates]
            if not ids or len(ids) != len(uvs) or min(ids) < 0:
                raise ValueError("Seam boundary paths require matching, non-negative vertex IDs and UVs.")
            if any(len(uv) != 2 or not all(math.isfinite(value) for value in uv) for uv in uvs):
                raise ValueError("Seam boundary paths require finite 2D UV coordinates.")
            paths.append({"vertex_ids": ids, "uv_coords": uvs})
        if len(paths) != 6:
            raise ValueError("Expected six ordered seam boundary paths.")
        self.boundary_paths = paths

    def add_seam(self, name: str, seam_info: SeamInfo) -> None:
        """Add a seam with the given name and SeamInfo object."""
        self.seams[name] = seam_info

    def renumber_seams(self, vertex_map: dict) -> None:
        """Renumber the vertex ids for all seams based on a provided mapping from old to new ids."""
        for seam in self.seams.values():
            seam.renumber_vertices(vertex_map)
        for path in self.boundary_paths:
            path["vertex_ids"] = [vertex_map.get(vid, vid) for vid in path["vertex_ids"]]