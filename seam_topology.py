"""
Store information about the following:
Top and bottom vertex locations
Top and bottom closest vertex ids (id for stem and calyx patching)"""

import json
import math

from pathlib import Path
import numpy as np


class SeamInfo:
    """ Stores the vertex ids, their 3D locations, and their uv coordinates for one seam. Also includes the seam name (one of top, bottom, side, calyx, stem)
    Keeps the center used to create the seam (for the top, bottom, calyx, and stem) as well as the radius of the seam ring.
    uv_coords contains one list of UV pairs per vertex, including all chart copies.
    The side seam has no ring center or radius, so both are None.
    Can calculate the average of the seam vertex positions and the span of the height to determine the vertical extent of the seam.
    Supports reading and writing to a .json file.
    Supports renumbering the vertices if the vertex indices change in the mesh.
    """
    def __init__(self, name: str, vertex_ids: list, vertex_locations: list,
                 center: tuple | None, radius: float | None):
        self.name = name
        self.vertex_ids = vertex_ids
        self.vertex_locations = vertex_locations
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
        #self.vertex_ids = [vertex_map.get(vid, vid) for vid in self.vertex_ids]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "vertex_ids": self.vertex_ids,
            "vertex_locations": self.vertex_locations,
            "center": self.center,
            "radius": self.radius,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SeamInfo":
        return cls(
            name=data["name"],
            vertex_ids=data["vertex_ids"],
            vertex_locations=data["vertex_locations"],
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
                 boundary_paths: list[dict] | None = None, top_vertex: int = -1, bottom_vertex: int = -1,
                 face_lists: dict | None = None):
        self.ymin = ymin
        self.ymax = ymax
        self.top_point = top_point
        self.bottom_point = bottom_point
        self.seams = {} if seams is None else seams
        self.top_shoulder_radius = top_shoulder_radius
        self.bottom_shoulder_radius = bottom_shoulder_radius
        self.boundary_paths = [] if boundary_paths is None else boundary_paths
        self.top_vertex = top_vertex
        self.bottom_vertex = bottom_vertex
        self.face_lists = {} if face_lists is None else face_lists

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
            "top_vertex": self.top_vertex,
            "bottom_vertex": self.bottom_vertex,
            "face_lists": {name: list(faces) for name, faces in self.face_lists.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SeamTopology":
        seams = {name: SeamInfo.from_dict(seam) for name, seam in data["seams"].items()}
        face_lists = {name: set(faces) for name, faces in data.get("face_lists", {}).items()} 
        return cls(
            ymin=data["ymin"],
            ymax=data["ymax"],
            top_point=data["top_point"],
            bottom_point=data["bottom_point"],
            seams=seams,
            top_shoulder_radius=data.get("top_shoulder_radius"),
            bottom_shoulder_radius=data.get("bottom_shoulder_radius"),
            boundary_paths=data.get("boundary_paths"),
            top_vertex=data.get("top_vertex", -1),
            bottom_vertex=data.get("bottom_vertex", -1),
            face_lists=face_lists,
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
        if self.top_vertex != -1:
            self.top_vertex = vertex_map.get(self.top_vertex, self.top_vertex)
        if self.bottom_vertex != -1:
            self.bottom_vertex = vertex_map.get(self.bottom_vertex, self.bottom_vertex)
        for name, faces in self.face_lists.items():
            self.face_lists[name] = {vertex_map.get(vid, vid) for vid in faces}

    def renumber_faces(self, face_map: dict) -> None:
        """Renumber the face ids for all face lists based on a provided mapping from old to new ids."""
        for name, faces in self.face_lists.items():
            self.face_lists[name] = {face_map.get(fid, fid) for fid in faces}

    def transform(self, translation, rotation) -> None:
        """Apply rotation then translation to 3D points; mesh Y bounds must be recomputed by the caller."""
        translation = np.asarray(translation, dtype=np.float64)
        rotation = np.asarray(rotation, dtype=np.float64)
        if (translation.shape != (3,) or rotation.shape != (3, 3)
                or not np.isfinite(translation).all() or not np.isfinite(rotation).all()):
            raise ValueError("Expected a finite 3D translation and 3x3 rotation.")

        def transform_point(point):
            return (rotation @ np.asarray(point, dtype=np.float64) + translation).tolist()

        self.ymax -= translation[1]
        self.ymin -= translation[1]
        self.top_point = transform_point(self.top_point)
        self.bottom_point = transform_point(self.bottom_point)
        for seam in self.seams.values():
            seam.vertex_locations = [transform_point(point) for point in seam.vertex_locations]
            if seam.center is not None:
                seam.center = transform_point(seam.center)

    def check_geometry(self) -> None:
        """ Check that the top, bottom, and side seams are oriented correctly. The side seam is along the angle
        0 direction (in the XY plane). that means the face normal for the faces adjacent to the side seam should
        point along the X axis."""
        if not "top" in self.seams:
            print("Not checking top seam because it is not present.")
            return
        
        # The face for the top seam is defined by top loc 1, top loc 0, side seam 1. Check the cross product
        # to ensure it points along the expected direction (e.g., along the Y axis for the top seam).
        walk_out_top = max(1, min(5, int(len(self.seams["top"].vertex_locations) / 4)))
        n_side = len(self.seams["side"].vertex_locations)
        walk_out_side = max(1, min(5, int(n_side / 4)))
        v0 = self.seams["top"].vertex_locations[walk_out_top]
        v1 = self.seams["top"].vertex_locations[0]
        s1 = self.seams["side"].vertex_locations[walk_out_side]
        face_normal_top = np.cross(np.asarray(v0) - np.asarray(v1), np.asarray(s1) - np.asarray(v1))
        if not face_normal_top[1] > 0.0:
            raise ValueError("Top seam face normal is not oriented clockwise along the Y axis.")

        # The face for the bottom seam is defined by bottom loc 1, bottom loc 0, side seam 0. Check the cross product
        # to ensure it points along the expected direction (e.g., along the -Y axis for the bottom seam).
        walk_out_bottom = max(1, min(5, int(len(self.seams["bottom"].vertex_locations) / 4)))
        v0 = self.seams["bottom"].vertex_locations[0]
        v1 = self.seams["bottom"].vertex_locations[walk_out_bottom]
        s0 = self.seams["side"].vertex_locations[n_side - 1 - walk_out_side]
        face_normal_bottom = np.cross(np.asarray(v0) - np.asarray(v1), np.asarray(s0) - np.asarray(v1))
        if not face_normal_bottom[1] < 0.0:
            raise ValueError("Bottom seam face normal is not oriented along the -Y axis.")
        
    def check_valid(self, vertex_count: int) -> None:
        """Check that all seams and boundary paths have valid vertex indices."""
        for seam in self.seams.values():
            if len(seam.vertex_ids) < 2 or min(seam.vertex_ids) < 0 or max(seam.vertex_ids) >= vertex_count:
                raise ValueError(f"Invalid seam vertex indices in input seam topology.")
        for path in self.boundary_paths:
            if len(path["vertex_ids"]) < 2 or min(path["vertex_ids"]) < 0 or max(path["vertex_ids"]) >= vertex_count:
                raise ValueError(f"Invalid boundary path vertex indices.")

        self.check_geometry()
        # For every seam except the side seam, ensure that the vertex indices are unique
        #. The side seam's start and end vertices should be in the top and bottom seam
        names = list(self.seams.keys())
        seams = list(self.seams.values())
        for indx in range(len(seams)):
            name1 = names[indx]
            seam1 = seams[indx]
            for indx2 in range(indx+1, len(seams)):
                name2 = names[indx2]
                seam2 = seams[indx2]
                duplicates = set(seam1.vertex_ids) & set(seam2.vertex_ids)
                if len(duplicates) > 1:
                    raise ValueError(f"Seam '{name1}' shares multiple vertex indices with seam '{name2}'.")
                elif len(duplicates) == 1:
                    duplicate_v = duplicates.pop()
                    if name1 == "top":
                        if seam2.vertex_ids[0] != duplicate_v or name2 != "side":
                            raise ValueError(f"Side seam start vertex must match the top seam vertex {duplicate_v}.")
                    elif name1 == "bottom":
                        if seam2.vertex_ids[-1] != duplicate_v or name2 != "side":
                            raise ValueError(f"Side seam end vertex must match the bottom seam vertex {duplicate_v}.")
                    else:
                        raise ValueError(f"Seam '{name1}' shares a vertex with seam '{name2}' that is not allowed.")
           
    def write_to_file(self, seam_path: Path, n_vertices: int) -> None:
        self.check_valid(n_vertices)
        with seam_path.open("w", encoding="utf-8") as seam_file:
            json.dump(self.to_dict(), seam_file)
