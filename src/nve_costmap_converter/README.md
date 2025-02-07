# Costmap Converter for 3D to 2.5 

This package takes a 3D voxel cloud with voxel wise traversability estimates and generates a 2.5D costmap out of it. From the incoming point cloud it generates a ground surface in the form of a 2.5D map, where each "cell" consists of x,y,z coordinates and represents. The traversability cost are the average mean traversability estimate for each cell, from the ground cells up to "num_vert_voxels".

Given the frequent occlusions for natural environments, the resulting (patchy) costmap can be completed using either or a combination of patch and patch flood fill. Patch fill generates patches around existing measurements and considers the local neighbourhood. Flood fill does the same thing, but adds the virtual surfaces to the continuously. This makes the method directionally dependent and hence its recommended to use the patch fill first to alleviate this. This surfaces are marked as 'is_real=0' and 'is_real=0'.

The incoming point cloud is expected to have the fields \<x,y,z,prob\>. 

Published topics are either costmap as pcl2 or dense_costmap as DensePCL2. They will only publish if latched. 

## Installation and dependencies
- layered_costmap
- ohm
Note: Ohm is only used to generate the hash-keys. Can easily be replaced with openVDB, voxblox or any other 3D hashing library.

## Parameters

| Parameter | Description | Default value | Value ranges | 
| -------- | ------- |  -------- | ------- |
|  rate | Rate of the main loop $[Hz]$ | 5  | 1 - 10 |
| map_res | Voxel size of the incoming voxel cloud | 0.1 | 0.001-1.0 |
| world_frame | Frame of the static world | odom | - |
| robot_frame | Frame of the robot | ground_link | base_link, ground_link |
| local_map_bounds | Bounds of the local costmap origin at robot pose [m]| [-1.0, -1.0, -3.0, 1.0, 1.0, 3 ] | - |
| num_vertical_voxels | Number of vertical voxels to consider with and above the ground cell (includes the ground voxel). Robot height and voxel size determine this. | 10 | 8 -15 |
| use_virtual | Flat to enable virtual surfaces (patch and/or flood fill) | True | {False, True} |
| use_patch_fill | Flat to enable patch fill | True | {False, True} |
| use_patch_flood_fill | Flat to enable patch **flood** fill. This is done after patch_fill, if enabled, to ensure more local consistent surfaces | True | {False, True} |
| patch_leaf_size | Determines the number of cells considered in the adjacency calculations n+1 | 2 | 1-3 |
| min_adj_voxel | The minimum number of adjacent cells to generate a valid estimate for a cell | 3 | 3-8 | 
