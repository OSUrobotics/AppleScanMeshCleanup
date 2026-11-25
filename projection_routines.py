"""Routines to project sections of the mesh
"""

import argparse
from pathlib import Path

from meshlib import mrmeshnumpy as mn
from meshlib import mrmeshpy as mm

import numpy as np
from scipy.sparse.linalg import spsolve

from seam_topology import SeamInfo, SeamTopology
from obj_geometry import create_adjacency_structure, bfs_graph, compute_edge_locs_line, compute_edge_locs_circle

from scipy.sparse import csc_array
from scipy.sparse.linalg import spsolve


def label_vs_and_fs(mesh: mm.Mesh, 
                    seam_topology: SeamTopology)->tuple[list[int], list[int]]:
    """ Label the uv coordinates and the faces with one of three labels: Bottom, Top, and middle """
    # Do a breadth_first search from the top and the bottom caps. The middle is what's left

    # The adjacency structure needs to be complete so that the place circle/diamond will work correctly
    edge_adj = create_adjacency_structure(mesh, no_cross_vs = set())

    # Don't cross these edges in the breadth first search
    no_cross_vs = set(seam_topology.seams["top"].vertex_ids) | set(seam_topology.seams["bottom"].vertex_ids) | set(seam_topology.seams["side"].vertex_ids)
    # Add the top seams vertices in - _place_circle will manually place those the boundary and skip them on the inside loop
    top_vs = set(bfs_graph(edge_adj, start = seam_topology.top_vertex, no_cross_vs = no_cross_vs))
    top_vs = top_vs | set(seam_topology.seams["top"].vertex_ids)

    # Start the bottom cap at the vertex closest to the user-marked bottom point
    if seam_topology.bottom_vertex != -1:
        bottom_vid = seam_topology.bottom_vertex
    else:
        verts = mn.getNumpyVerts(mesh)
        bottom_vid = np.argmin(verts[:, 1])
    bottom_vs = set(bfs_graph(edge_adj, start=bottom_vid, no_cross_vs=no_cross_vs))
    bottom_vs = bottom_vs | set(seam_topology.seams["bottom"].vertex_ids)

    # middle_vs are what's left
    middle_vs = set([indx for indx in range(mesh.topology.numValidVerts()) if indx not in top_vs and indx not in bottom_vs])
    # Add the top, bottom, and side seam vertices to the middle_vs set
    middle_vs = middle_vs | set(seam_topology.seams["side"].vertex_ids)
    middle_vs = middle_vs | set(seam_topology.seams["top"].vertex_ids)
    middle_vs = middle_vs | set(seam_topology.seams["bottom"].vertex_ids)
    check_overlap = (top_vs & bottom_vs) | (top_vs & middle_vs) | (bottom_vs & middle_vs)
    n_side_not_corner = len(seam_topology.seams["side"].vertex_ids) - 2
    if len(check_overlap) + n_side_not_corner != len(no_cross_vs):
        raise ValueError("Regions should only overlap at the seam vertices between top, bottom, and middle regions.")

    top_vs = top_vs | set(seam_topology.seams["top"].vertex_ids)
    bottom_vs = bottom_vs | set(seam_topology.seams["bottom"].vertex_ids)
    # Don't put the seam vs in to middle_vs 
    print(f"Number of vertices: {mesh.topology.numValidVerts()}")
    print(f"Number of seam vs {len(no_cross_vs)}")
    print(f"Number of top vertices: {len(top_vs)}, bottom vertices: {len(bottom_vs)}, middle vertices: {len(middle_vs)}, sum: {len(top_vs) + len(bottom_vs) + len(middle_vs)}")

    return bottom_vs, seam_topology.face_lists["bottom"], top_vs, seam_topology.face_lists["top"], middle_vs, seam_topology.face_lists["middle"], bottom_vid, edge_adj


def place_in_square(mesh: mm.Mesh, 
                     seam_topology: SeamTopology,
                     face_vertex_indices: np.ndarray,
                     edge_adjacency, 
                     inside_vs: list[int], 
                     corners: list[np.ndarray]
                     ):
    """ The corner vertex gets placed at the four  corners. The top and bottom seams go 
    along the upper right and lower left edges of the diamond. The side seam is duplicated and goes along the other two sides
    The vertex_uv locations in texture map space are used to distunguish which copy to use for each vertex star"""
    # The A matrix and b vector for the least squares problem will be set up here. 
    
    # Counts all the boundary vertices once
    top_seam_vs = set(seam_topology.seams["top"].vertex_ids)
    bottom_seam_vs = set(seam_topology.seams["bottom"].vertex_ids)
    side_seam_vs = set(seam_topology.seams["side"].vertex_ids)
    all_boundary = top_seam_vs | bottom_seam_vs | side_seam_vs
    all_inside = set(inside_vs) - all_boundary
    n_side_additional = len(seam_topology.seams["side"].vertex_ids)
    # Side seam is duplicated, corner vertex is placed in four places (one counted in all_boundary)
    n_constraints = len(all_boundary) + len(all_inside) + n_side_additional
    rows = []
    cols = []
    data = []
    #a_mat = np.zeros((n_constraints, n_constraints))
    b_vec = np.zeros((n_constraints, 2))

    map_vs = {}
    for vid in inside_vs:
        map_vs[vid] = []
    map_back_vs = {}


    # Make a flat list of all the boundary vs, duplicating the side vs and corner vs
    corner_top = (set(seam_topology.seams["side"].vertex_ids) & set(seam_topology.seams["top"].vertex_ids)).pop()
    corner_bottom = (set(seam_topology.seams["side"].vertex_ids) & set(seam_topology.seams["bottom"].vertex_ids)).pop()

    # Double check
    assert corner_top == seam_topology.seams["side"].vertex_ids[0] 
    assert corner_bottom == seam_topology.seams["side"].vertex_ids[-1] 
    assert corner_top == seam_topology.seams["top"].vertex_ids[0]
    assert corner_bottom == seam_topology.seams["bottom"].vertex_ids[0]

    # Top seam is going from corner 0 to corner 1
    #   So corner_top is mapped to corners 0 and 1, corner_bottom is mapped to corners 2 and 3
    count = 0
    for i, corner in enumerate([corner_top, corner_top, corner_bottom, corner_bottom]):
        map_vs[corner].append(count)
        map_back_vs[count] = corner
        rows.append(count)
        cols.append(count)
        data.append(1.0)
        #a_mat[count, count] = 1.0
        b_vec[count, :] = corners[i, :]
        count += 1

    # Now add in the seam boundaries, duplicating the side ones
    # Place the 0 corner at the sart of the top seam and the start of the side seam, 
    #    the 1 corner at the end of top seam and the start of the side seam
    #.   the 2 corner at end of the side seam and the end of the bottom seam
    #.   the 3 corner at the start of the bottom seam and the end of the side seam
    verts = mn.getNumpyVerts(mesh)
    top_vids = seam_topology.seams["top"].vertex_ids    
    top_vids_locs = [v for v in top_vids]
    top_vids_locs.append(top_vids[0])
    top_locs = compute_edge_locs_line(verts, top_vids_locs, corners[0], corners[1])
    side_1_vids = seam_topology.seams["side"].vertex_ids
    side_1_locs = compute_edge_locs_line(verts, side_1_vids, corners[1], corners[2])
    bottom_vids = list(reversed(seam_topology.seams["bottom"].vertex_ids))
    bottom_vids_locs = [v for v in bottom_vids]
    bottom_vids_locs.append(bottom_vids[0])
    bottom_locs = compute_edge_locs_line(verts, bottom_vids_locs, corners[2], corners[3])
    side_2_vids = list(reversed(seam_topology.seams["side"].vertex_ids))
    side_2_locs = compute_edge_locs_line(verts, side_2_vids, corners[3], corners[0])
    # Only pin the middle vertices on the seams - corners are already done. The top and bottom seams start with the corner vertex
    vids = [top_vids[1:], 
            side_1_vids[1:-1],
            bottom_vids[0:-1],
            side_2_vids[1:-1]]
    locs = [top_locs[1:-1], 
            side_1_locs[1:-1], 
            bottom_locs[1:-1], 
            side_2_locs[1:-1]]
    corner_count = 0
    for vids, locs in zip(vids, locs):
        print(f"Corner start {corners[corner_count]}, locs start {locs[0]}")
        print(f"Corner end {corners[(corner_count + 1) % 4]}, locs end {locs[-1]}")
        for indx, vid in enumerate(vids):
            map_vs[vid].append(count)
            map_back_vs[count] = vid
            rows.append(count)
            cols.append(count)
            data.append(1.0)
            #a_mat[count, count] = 1.0
            b_vec[count, :] = locs[indx]
            count += 1
        corner_count += 1
    n_edge_and_corner = count

    for v in inside_vs:
        if v in all_boundary:
            if v == corner_top or v == corner_bottom:
                if len(map_vs[v]) != 2:
                    raise ValueError(f"Corner vertex {v} does not have exactly 2 mapped vertices.")
            elif v in top_seam_vs or v in bottom_seam_vs:
                if len(map_vs[v]) != 1:
                    raise ValueError(f"Top or bottom seam vertex {v} does not have exactly 1 mapped vertex.")
            else:
                if not len(map_vs[v]) == 2:
                    raise ValueError(f"Side seam vertex {v} should have 2 mapped vertices.")
            continue
        elif not len(map_vs[v]) == 0:
            raise ValueError(f"Inside vertex {v} has a mapped vertex, but shouldn't {map_vs[v]}.")
        map_vs[v] = [count]
        map_back_vs[count] = v
        count += 1

    if count != n_constraints:
        raise ValueError(f"Count of mapped vertices {count} does not match number of constraints {n_constraints}.") 

    # Get all the vertices of the faces on the right and left side
    vs_on_left_side, vs_on_right_side = _expand_face_side_lists(edge_adjacency=edge_adjacency,
                                                                seam_topology=seam_topology, 
                                                                face_vert_ids=face_vertex_indices)

    for v, v_copies in map_vs.items():
        if len(v_copies) > 1:
            if v not in side_seam_vs:
                raise ValueError(f"Vertex {v} has multiple copies but is not on the side seam.")
            
    # Now place the inside the vertices as 1/n * their neighbors
    for i, v in enumerate(all_inside):
        row = map_vs[v][0]
        assert row >= n_edge_and_corner

        rows.append(row)
        cols.append(row)
        data.append(1.0)
        #a_mat[row, row] = 1.0
        neighbors = [n for n, _ in edge_adjacency[v]]
        n_neighbors = len(neighbors)
        perc = 1.0 / n_neighbors

        count_left = 0
        count_right = 0
        for neighbor in neighbors:
            if neighbor in vs_on_left_side:
                count_left += 1
            if neighbor in vs_on_right_side:
                count_right += 1

        for neighbor in neighbors:
            map_v = map_vs.get(neighbor, None)
            if map_v is None:
                raise ValueError(f"Neighbor vertex {neighbor} not found in map_vs.")
            if len(map_v) == 1:
                # Another inside vertex or top/bottom edge, just take its mapped index
                col = map_v[0]
                if neighbor in top_seam_vs or neighbor in bottom_seam_vs:
                    assert col < n_edge_and_corner
                else:
                    assert col >= n_edge_and_corner
            else:
                if neighbor == corner_top:
                    # This one is backwards
                    if count_left > 0 and count_right == 0:
                        col = map_v[1]
                    elif count_right > 0 and count_left == 0:
                        col = map_v[0]
                    else:
                        for nv, _ in edge_adjacency[neighbor]:
                            print(f"Neighbor vertex {nv} of vertex {v}")
                        raise ValueError(f"Neighbor vertex {neighbor} not found on either side of the seam.")
                else:
                    if count_left > 0 and count_right == 0:
                        col = map_v[0]
                    elif count_right > 0 and count_left == 0:
                        col = map_v[1]
                    else:
                        for nv, _ in edge_adjacency[neighbor]:
                            print(f"Neighbor vertex {nv} of vertex {v}")
                        raise ValueError(f"Neighbor vertex {neighbor} not found on either side of the seam.")
                assert col < n_edge_and_corner
            rows.append(row)
            cols.append(col)
            data.append(-perc)
            #a_mat[row, col] = -perc

    touch_col = np.zeros(n_constraints, dtype=np.int32)
    touch_row = np.zeros(n_constraints, dtype=np.int32)
    for row, col in zip(rows, cols):
        touch_col[col] += 1
        touch_row[row] += 1
    for indx in range(n_edge_and_corner):
        assert touch_row[indx] == 1
    for indx, nc in enumerate(touch_col):
        if nc == 1:
            print(f"Column with only one touch: {indx}, {nc}")

    a_mat = csc_array((data, (rows, cols)), shape=(n_constraints, n_constraints))
    x = spsolve(a_mat, b_vec)
    return x, map_vs, map_back_vs, vs_on_left_side, vs_on_right_side


def _expand_face_side_lists(edge_adjacency, 
                            seam_topology: SeamTopology, 
                            face_vert_ids: np.ndarray): 
    """ The faces originally labeled left and right are those only actually sharing an edge of the seam.
    This expands the face list (and vertex list) to include all of the faces/vertices adjacent"""
    top_seam_vs = set(seam_topology.seams["top"].vertex_ids)
    bottom_seam_vs = set(seam_topology.seams["bottom"].vertex_ids)
    side_seam_vs = set(seam_topology.seams["side"].vertex_ids)
    all_boundary = top_seam_vs | bottom_seam_vs | side_seam_vs

    vs_on_right_side = set()
    vs_on_left_side = set()
    right_side_faces = set(seam_topology.face_lists["right_side_seam"])
    left_side_faces = set(seam_topology.face_lists["left_side_seam"])
    top_faces = set(seam_topology.face_lists["top"])
    bottom_faces = set(seam_topology.face_lists["bottom"])
    if left_side_faces.intersection(right_side_faces):
        raise ValueError(f"Left and right side faces should not overlap {left_side_faces.intersection(right_side_faces)}.")

    faces_to_label = set()
    for indx, face in enumerate(face_vert_ids): 
        if indx in left_side_faces or indx in right_side_faces:
            continue       
        if indx in top_faces or indx in bottom_faces:
            continue
        count_seam = np.count_nonzero(set(face) & side_seam_vs)
        count_top = np.count_nonzero(set(face) & top_seam_vs)
        count_bottom = np.count_nonzero(set(face) & bottom_seam_vs)
        if count_seam > 0:
            # Face has two vertices along the top or bottom seam, and only one of the face (this is a corner face)
            if count_top == 2 or count_bottom == 2:
                continue
            if count_top > 0 or count_bottom > 0:
                print(f"Corner face detected: {indx} {count_seam}, {count_top}, {count_bottom}")
            faces_to_label.add(indx)

    print(f"Unassigned faces {len(faces_to_label)}")
    for vs, sl in zip((vs_on_left_side, vs_on_right_side), (left_side_faces, right_side_faces)):
        for face in sl:
            for vid in face_vert_ids[face]:
                if vid not in all_boundary:
                    vs.add(int(vid))

        b_expanded = True
        while b_expanded:
            b_expanded = False
            vs_to_add = set()
            for v in vs:
                # Look at the neighbors of the vertices that are in the current set of left/right seam vs
                for neighbor, _ in edge_adjacency[v]:
                    # Skip neighbors on the side seam so we don't cross it
                    if neighbor in side_seam_vs:
                        continue

                    faces_labeled = set()
                    # Look for faces that are in the unlabeled set that have neighbor in them
                    for indx, unlabeled_face in enumerate(faces_to_label):
                        if neighbor in face_vert_ids[unlabeled_face]:
                            sl.add(unlabeled_face)
                            faces_labeled.add(unlabeled_face)
                            if neighbor not in top_seam_vs and neighbor not in bottom_seam_vs:
                                vs_to_add.add(neighbor)
                            b_expanded = True
                    faces_to_label = faces_to_label - faces_labeled

            vs.update(vs_to_add)

        for face in sl:
            for vid in face_vert_ids[face]:
                if vid not in all_boundary:
                    vs.add(int(vid))
        print(f"Unassigned faces {len(faces_to_label)}")

    if left_side_faces.intersection(right_side_faces):
        raise ValueError(f"Left and right side faces should not overlap {vs_on_left_side.intersection(right_side_faces)}.")
    if vs_on_left_side.intersection(vs_on_right_side):
        raise ValueError(f"Left and right side vertices should not overlap {vs_on_left_side.intersection(vs_on_right_side)}.")
    seam_topology.face_lists["left_side_seam"] = left_side_faces
    seam_topology.face_lists["right_side_seam"] = right_side_faces
    return vs_on_left_side, vs_on_right_side


def place_in_circle(mesh: mm.Mesh, edge_adjacency, boundary_vs: list[int], inside_vs: list[int], center: int, radius = 1.0):
    """ Set up a least squares problem to place the inside vertices in a circle defined by the boundary vertices and the center. 
    Use the edge lengths as weights in the least squares problem to maintain the relative distances between vertices. 
    Place the boundary vertices on the circle based on their original edge lengths.
    Place the center vertex at the origin."""

    count = 0
    map_vs = {center: count}
    map_back_vs = {count: center}

    count += 1
    for i, v in enumerate(boundary_vs):
        map_vs[v] = count
        map_back_vs[count] = v
        count += 1

    # Include the center vertex because it is pinned,
    boundary_set = set(boundary_vs) | set([center])
    for i, v in enumerate(inside_vs):
        if v in boundary_set:
            continue
        map_vs[v] = count
        map_back_vs[count] = v
        count += 1

    # The A matrix and b vector for the least squares problem will be set up here. 
    n_constraints = len(inside_vs)

    if count != n_constraints:
        raise ValueError(f"Expected {n_constraints} constraints, but got {count}.")

    rows = []
    cols = []
    data = []
    # a_mat = np.zeros((n_constraints, n_constraints))
    b_vec = np.zeros((n_constraints, 2))

    # For now, we just place the boundary vertices on the circle and the center at the origin.
    # Compute the total perimeter length of the boundary
    verts = mn.getNumpyVerts(mesh)

    # Place the center vertex at the origin
    count_constraints = 0
    rows.append(count_constraints)
    cols.append(count_constraints)
    data.append(1.0)
    b_vec[count_constraints] = np.array([0.0, 0.0])
    count_constraints += 1

    # Place the boundary vertices on the circle based on their original edge lengths
    edge_locs = compute_edge_locs_circle(verts, boundary_vs, radius)
    for i in range(len(boundary_vs)):
        rows.append(count_constraints)
        cols.append(count_constraints)
        data.append(1.0)
        loc = edge_locs[i]
        b_vec[count_constraints] = loc
        count_constraints += 1

    # Now place the inside the vertices as 1/n * their neighbors
    for i, v in enumerate(inside_vs):
        if v in boundary_set or v == center:
            continue
        rows.append(count_constraints)
        cols.append(count_constraints)
        data.append(1.0)
        neighbors = [n for n, _ in edge_adjacency[v]]
        n_neighbors = len(neighbors)
        perc = 1.0 / n_neighbors
        for neighbor in neighbors:
            map_v = map_vs.get(neighbor, None)
            if map_v is None:
                print(f"vertex {v}: Neighbor vertex {neighbor} not found in map_vs.")
                raise ValueError(f"Neighbor vertex {neighbor} not found in map_vs.")
            rows.append(count_constraints)
            cols.append(map_v)
            data.append(-perc)
        count_constraints += 1

    a_mat = csc_array((data, (rows, cols)), shape=(n_constraints, n_constraints))
    x = spsolve(a_mat, b_vec)
    return x, map_vs, map_back_vs


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize an OBJ's UV charts and texture.")
    parser.add_argument("input", nargs="?", default="/Users/cindygrimm/PycharmProjects/data/ScannedModels/ProsserEnvyApples/BlenderRebaked/Envy_002_FullRes.obj", type=Path, help="Rebaked textured OBJ")
    parser.add_argument("output", nargs="?", default="/Users/cindygrimm/PycharmProjects/data/ScannedModels/ProsserEnvyApples/NormalizedTexture/Envy_002_FullRes_seam.obj", type=Path, help="Output OBJ path")
    parser.add_argument("seams", nargs="?", default="/Users/cindygrimm/PycharmProjects/data/ScannedModels/ProsserEnvyApples/Parameterized", type=Path, help="Directory containing seam topology JSON files")
    args = parser.parse_args()


if __name__ == "__main__":
    main()