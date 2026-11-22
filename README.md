# Usage and flow

This is primarily support code for cleaning up Revo-scanned meshes of apples and secondarily code for aligning all of those meshes in order to create a "more of the same" apple generator script that tries to match the specific variety's variation. 

Caveat: Most of this code is LLM generated (with a lot of guidance) and uses PyMeshLib, MeshLib, and Blender functionality. 

## Cleanup process:
This is probably useful, regardless of if you want to use the meshes to make more of the same. RevoScan produces high density, really ugly meshes, with a lot of small tunnels and non-manifold edges and semi-janky texture map (lots of small patches). If you try to downsample this to a more reasonable size (like 5,000 faces) it produces gaps in the texture maps and dense clusters of vertices around bad areas, with big polygons elsewhere. The process here cleans up the mesh and produces a texture map in three, well defined pieces. This makes fixing the texture map a lot easier and supports extensive downsampling.

A note on scanning: You need to leave the stem on the apple. Otherwise, RevoScan can't align the meshes (top and side scan - see instructions below). But, because you have a stem, you have to take it out again... that's why there's one manual step to remove the stem.

- Scan the apples with a table-top scanner using RevoScan (or equiv)
- Use **cleanup** to downsample and clean up mesh artifacts
- (Manual) remove the stem, break any tunnels in the calyx, and mark the top (stem junction) and bottom (center of calyx)
- Use Blender's baking function with smart uv mapping to fix the texture map (**bake_textures** with use_smart_uv option)
- Use **voxelization** to create a clean mesh with spherical topology.
- Use **parameterize** to add the stem/calyx rings and three texture seams.
-- The top and bottom charts are centered using the median of its UV vertices and uniformly scaled to a maximum radius of 0.1 (diameter 0.2), with padding from the texture edges.
-- The middle of the apple is mapped to a rectangle placed diagonally in the texture map
-- If the middle chart overlaps either the top or bottom chart, it is contracted along the cap-to-cap diagonal until it clears both the top and bottom charts, preserving triangle orientation.
-- Voxelization writes `_seam_topology.json` with the Y bounds and selected center points. Parameterization adds the top and bottom shoulder radii, calyx, stem, top, bottom, and side seam data, and six ordered UV boundary paths. The shoulder radii determine the top and bottom UV seam targets. Each seam vertex has a list of its UV coordinates, including copies on different charts; the side seam has no ring center or radius.
-- Parameterization preserves OBJ vertex order and face vertex indices, even when the input has existing UV or normal seams. The output's seam vertex IDs refer to its original OBJ vertex indices.
-- The standalone **uv_texture** entry point requires a matching `_seam_topology.json` containing both shoulder radii.
-- The mesh `_seam.ply` in the Parameterization folder marks seam vertices black and the vertices nearest the top and bottom centers, plus coloring the seam vertices unique colors.
- Use **bake_textures** again (Blender's baking function w/o use-smart-uv) to re-create the texture map with the new mesh.
- (Manual) Use Blender's texture paint (or any painting program) to clean up any holes in the texture and erase the stem 
- Use **uv_normalization** to warp the texture maps so that the top and bottom are circles and the diagonal has straight lines. In the bottom chart the 0.05 radius is where the calyx boundary is placed. This also updates a seam JSON file.
- Use **downsample_fullres** to create the cleaned up (but much smaller) mesh that should be visually identical to the original scan. This also orients the mesh along the y axis, and place the stem at the origin
-- Some padding is also applied to the texture map so if the mesh smoothing causes seam uv coordinates to go over the boundary a bit it's ok
-- Each output gets a companion seam file with transformed 3D points. Its seam vertex IDs, positions, and UVs describe the source mesh, not the decimated mesh; they are not remapped after downsampling. Legacy CSV seams must include top and bottom 3D points.
- Use **downsample_hires** to produce a smaller mesh.
- Use **remove_calyx** to cut out the calyx and replace it with a grid of faces. Also saves out the calyx with it's own texture map
- UV reconstruction projects from the source OBJ onto the patched mesh, preserving the texture and rebuilding top, bottom, and side seams along target edges. It requires three connected source UV charts; seams are approximate where the meshes differ, and ambiguous seam paths are errors.
- Use **align_meshes** to align the remove calyx mesh with both the hi and low res canonical meshes
- Use **downsample_lowres** to produce a lowres mesh that has both a downsampled texture map and a flat calyx.

Most of the operations can be run stand-alone using the main function in the corresponding file. Your mileage may vary.

There's more information on how to clean up the scans and the Blender processing here: [Instructions](https://docs.google.com/document/d/1VNYZVNQCl2NPY8_LQbEhR_85cBMGDxbPyeexvK_tSUA/edit?tab=t.0)

The launch.json file has examples of calling each of these (the --operation name is given in parenthesis above). The directories are largely made for you and you just supply the root directory. Keep all the apples of one type in one directory. Name the scans some version os AppleType#Whatever.obj when you create them and put them in a folder OriginalRevoScans. 

The code will not over-write an existing file unless replace_existing is set. 

## Normalization process

Alignment reads each input mesh's topology JSON from the supplied seam
directory; its calyx vertex IDs and endpoints must match the input mesh.
The canonical topology JSON is in the same
folder as the canonical OBJ; legacy seam CSV files are also accepted, but
cannot supply calyx seams and produce a warning.
The canonical body ends at -70° latitude; its bottom cap has rings down to a
calyx boundary at -85°, with spacing based on the body boundary edges. In the
bottom UV chart, the calyx ring maps to `_UV_CALYX_RADIUS` and the outer boundary
remains at `_UV_RADIUS`.
Run normalization before alignment; rebaked UV charts are not necessarily
normalized. The batch command uses `--operation align_meshes` and selects
`Canonical/apple_fullres.obj` by default (or `--canonical hires`, etc.).
The standalone command accepts `input.obj canonical.obj output.obj`.
The aligned OBJ uses the canonical UVs and material, with a JPEG texture
resampled from the input scan texture using the exported vertex-to-face
correspondences. The JPEG has the input texture's dimensions; triangle
rasterization uses bilinear sampling and adds four pixels of chart padding.
Source coordinates come from each face corner, including duplicated seam UVs.
When a correspondence creates a disproportionately stretched triangle, the
resampler warns and uses the transformed face-corner UV to locate the affected
corner on the canonical chart; it warns if a stretched triangle remains.
Degenerate mapped triangles are skipped with a warning, so distorted or
incomplete correspondences can leave gaps or overlaps in the resampled texture.
The geometry fit uses 3D coordinates only. It initially aligns and scales the
pole axes, assuming the meshes already share their rotation around that axis.
It matches the calyx rings by their 3D orientation and relative distance along
each ring, pins canonical calyx vertices to the scan ring, and maps scan calyx
vertices to canonical ring edges. Ring sampling may differ between meshes.
The top, bottom, and side seams are not matched. It divides
each mesh halfway between its extrema along its top-to-bottom pole axis, and
projects vertices onto the corresponding scan half. Scan triangles crossing
the midpoint are clipped for projection; the canonical mesh remains connected
across the boundary. The pole vertices remain pinned to the scan vertices
nearest the endpoints through four neighbor-averaging and reprojection
iterations. Other vertices use 3D surface projection for correspondence.
UV coordinates are used only afterward for texture transfer; the charts should
have the same orientation. The fit does not guarantee an intersection-free
mesh or exact vertex-to-vertex matches.

## Mesh sizes
- The original meshes (from RevoScan) are typically on the order of a million faces.
- The voxelized scans are typically on the order of 200,000 to 400,000 faces
- The FullRes scans (made by using meshlab to downsample the rebaked voxelized ones) are on the order of 40,000 faces
- The FullRes canonical mesh has 20,000 faces (no messy calyx geometry)
- The HiRes canonical mesh has 6,000 faces (no messy calyx geometry)
- The LowRes canonical mesh has 1,300 faces

