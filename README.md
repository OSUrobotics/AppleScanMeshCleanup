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
- Use voxelization to create a clean mesh with spherical topology and three texture seams: a circle on the top, one on the bottom, and a side seam (voxelize)
- Use Blender's baking function to re-create the texture map (bake_textures, no use_smart_uv option)
- (Manual) Use Blender's texture paint (or any painting program) to clean up any holes in the texture and erase the stem 
- Downsample to a high, medium, and low res version, orient along the y axis, and place the stem at the origin (downsample_fullres, downsample_hires, downsample_lowres)

There's more information on how to clean up the scans and the Blender processing here: [Instructions](https://docs.google.com/document/d/1VNYZVNQCl2NPY8_LQbEhR_85cBMGDxbPyeexvK_tSUA/edit?tab=t.0)

The launch.json file has examples of calling each of these (the --operation name is given in parenthesis above). The directories are largely made for you and you just supply the root directory. Keep all the apples of one type in one directory. Name the scans some version os AppleType#Whatever.obj when you create them and put them in a folder OriginalRevoScans. 

The code will not over-write an existing file unless --overwrite-- is specified. 

## Normalization process
