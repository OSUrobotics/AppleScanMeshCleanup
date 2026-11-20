# Usage and flow

This is primarily support code for cleaning up Revo-scanned meshes of apples and secondarily code for aligning all of those meshes in order to create a "more of the same" apple generator script that tries to match the specific variety's variation. 

Caveat: Most of this code is LLM generated (with a lot of guidance) and uses PyMeshLib, MeshLib, and Blender functionality. 

## Cleanup process:
This is probably useful, regardless of if you want to use the meshes to make more of the same. RevoScan produces high density, really ugly meshes, with a lot of small tunnels and non-manifold edges and semi-janky texture map (lots of small patches). If you try to downsample this to a more reasonable size (like 5,000 faces) it produces gaps in the texture maps and dense clusters of vertices around bad areas, with big polygons elsewhere. The process here cleans up the mesh and produces a texture map in three, well defined pieces. This makes fixing the texture map a lot easier and supports extensive downsampling.

A note on scanning: You need to leave the stem on the apple. Otherwise, RevoScan can't align the meshes (top and side scan - see instructions below). But, because you have a stem, you have to take it out again... that's why there's one manual step to remove the stem.

- Scan the apples with a table-top scanner using RevoScan (or equiv)
- Downsample and clean up mesh artifacts (cleanup)
- (Manual) remove the stem, break any tunnels in the calyx, and mark the top (stem junction) and bottom (center of calyx)
- Use Blender's baking function with smart uv mapping to fix the texture map (bake_textures with use_smart_uv option)
- Use voxelization to create a clean mesh with spherical topology (voxelize), then add the stem/calyx rings and three texture seams (parameterize).
- Each cap chart is centered using the median of its UV vertices and uniformly scaled to a maximum radius of 0.1 (diameter 0.2), with padding from the texture edges.
- If the middle chart overlaps either cap, it is contracted along the cap-to-cap diagonal until it clears both caps, preserving triangle orientation.
- Voxelization writes `_seam_topology.json` with the Y bounds and selected center points. Parameterization adds the top and bottom shoulder radii, calyx, stem, top, bottom, and side seam data, and six ordered UV boundary paths. The shoulder radii determine the top and bottom UV seam targets. Each seam vertex has a list of its UV coordinates, including copies on different charts; the side seam has no ring center or radius.
- The standalone UV entry point requires a matching `_seam_topology.json` containing both shoulder radii; for a `_voxel` mesh it looks for the JSON without the `_voxel` suffix.
- The companion `_seam.ply` marks seam vertices black and the vertices nearest the top and bottom centers, plus their first six rings of mesh neighbors, green; green takes precedence where they overlap.
- UV normalization maps the mesh vertices nearest those 3D points to the centers of the top and bottom circular charts; each must have an interior UV on its respective chart. It writes a companion topology JSON with normalized seam UVs.
- Normalization looks in `Parameterized` for a matching `_seam_topology.json`, or a legacy `_seam.csv` if no JSON exists, using the apple name and number and ignoring zero-padding and suffixes after the number. Missing or multiple matches are errors.
- Degenerate or non-manifold UV edges produce a warning and small, deterministic UV perturbations during normalization. This does not repair invalid topology, so parameterization can still fail.
- Use Blender's baking function to re-create the texture map (bake_textures, no use_smart_uv option)
- (Manual) Use Blender's texture paint (or any painting program) to clean up any holes in the texture and erase the stem 
- Downsample to a high, medium, and low res version, orient along the y axis, and place the stem at the origin (downsample_fullres, downsample_hires, downsample_lowres)
- Downsampling pads the original texture once per mesh, saving `<mesh stem>_padded` with the original image extension in the input directory, then copies it to each Final texture. It uses the input mesh's companion topology JSON or legacy CSV when present, otherwise matching seams from `Parameterized` or legacy `Voxelized` data. Padding extends colors across seams using five rings of neighboring faces, with seam vertices pinned to the opposite chart. Faces with edges on both the side seam and a top or bottom seam are excluded. Flipped or degenerate reprojected faces and those with a UV edge longer than 0.25 are skipped with a warning. Only white RGB or fully transparent pixels outside the original UV charts are filled by reprojection; pixels beyond the projected faces remain unchanged. Missing pixels along seam edges are filled using the mean of valid adjacent pixels, including inside the chart. The mesh UVs and seam data must correspond; missing or ambiguous seams are errors.

To fill an image independently, run `python fix_texture_maps.py TEXTURE MESH.obj SEAMS.json OUTPUT.png` (legacy seam CSV input is also supported). The output can be the input image; the original mesh and seam file are never overwritten.

There's more information on how to clean up the scans and the Blender processing here: [Instructions](https://docs.google.com/document/d/1VNYZVNQCl2NPY8_LQbEhR_85cBMGDxbPyeexvK_tSUA/edit?tab=t.0)

The launch.json file has examples of calling each of these (the --operation name is given in parenthesis above). The directories are largely made for you and you just supply the root directory. Keep all the apples of one type in one directory. Name the scans some version os AppleType#Whatever.obj when you create them and put them in a folder OriginalRevoScans. 

The code will not over-write an existing file unless replace_existing is set. 

## Normalization process

Alignment reads meshes from `NormalizedTexture`, using their matching
the companion topology JSON, or matching `Parameterized` seam data (with legacy
`Voxelized` CSV fallback), for seam vertices and the scan's top and bottom XYZ
endpoints. The canonical topology JSON is in the same folder as the canonical OBJ;
legacy canonical seam CSV files are also accepted.
The canonical body ends at -70° latitude; its bottom cap has rings down to a
calyx boundary at -85°, with spacing based on the body boundary edges.
Run normalization before alignment; rebaked UV charts are not necessarily
normalized. The batch command uses `--operation align_meshes` and selects
`Canonical/apple_fullres.obj` by default (or `--canonical hires`, etc.).
The standalone command accepts `input.obj canonical.obj output.obj`.
The aligned OBJ uses the canonical UVs and material, with a PNG texture
resampled from the input scan texture using the exported vertex-to-face
correspondences. The PNG has the input texture's dimensions; triangle
rasterization uses bilinear sampling and adds four pixels of chart padding.
Source coordinates come from each face corner, including duplicated seam UVs.
When a correspondence creates a disproportionately stretched triangle, the
resampler warns and uses the transformed face-corner UV to locate the affected
corner on the canonical chart; it warns if a stretched triangle remains.
Degenerate mapped triangles are skipped with a warning, so distorted or
incomplete correspondences can leave gaps or overlaps in the resampled texture.
The fit does not impose special calyx peak or seam constraints. It divides
each mesh halfway between its extrema along its top-to-bottom pole axis, and
projects vertices onto the corresponding scan half. Scan triangles crossing
the midpoint are clipped for projection; the canonical mesh remains connected
across the boundary. The pole vertices remain pinned to the scan vertices
nearest the endpoints through six neighbor-averaging and reprojection
iterations. UV seam and pole
correspondences are still used for texture transfer. This does not guarantee
an intersection-free mesh or exact vertex-to-vertex matches.
