# Costmap Converter for 3D to 2.5 

This package takes a 3D voxel cloud with voxel wise traversability estimates and generates a 2.5D costamp out of it. From the incoming point cloud it generates a ground surface in the form of a 2.5D map, where each "cell" consists of x,y,z coordinates and represents. The traversablitliy cost are the average mean traverability estimate for each cell, from the ground cells up to "num_vert_voxels".

Given the frequent occlusions for natural environments, the resulitng (patchy) costmap can be completed using either or a combination of patch and patch flood fill. Patch fill genrates patches around exising meassurments and consideres the local neighboorhood. Flodd fill does the same thing, but adds the virtual surfaces to the contiously. This makes the method directionally dependent and hence its recomended to use the patch fill first to aliviate this. This surfaces are marked as 'is_real=0' and 'is_real=0'.

The incoming point cloud is expected to have the fields <x,y,z,prob>. 

Published topics are either costmap as pcl2 or dense_costmap as DensePCL2. They will only publish if latched. 

## Installation and depenencies
- layered_costmap
- ohm
Note: Ohm is only used to generate the hash-keys. Can easly be replaced with openVDB, voxeblox or any other 3D hashing library.

## Parameters

| Parameter | Description | Default value | Value ranges | 
| -------- | ------- |  -------- | ------- |
|  rate | Rate of the main loop $[Hz]$ | 5  | 1 - 10 |
| map_res | Voxel size of the incoming voxel cloud | 0.1 | 0.001-1.0 |
| world_frame | Frame of the static world | odom | - |
| robot_frame | Frame of the robot | groun_link | base_link, ground_link |
| local_map_bounds | Bounds of the local costmap origin at robot pose [m]| [-1.0, -1.0, -3.0, 1.0, 1.0, 3 ] | - |
| num_vertical_voxels | Number of vertical voxels to consider with and above the ground cell (includes the ground voxel). Robot height and voxel size determin this. | 10 | 8 -15 |
| use_virtual | Flat to enable virtual surfaces (pathc and/or flood fill) | True | {Fasle, True} |
| use_patch_fill | Flat to enable patch fill | True | {Fasle, True} |
| use_patch_flood_fill | Flat to enable patch **flood** fill. This is done after patch_fill, if enabled, to ensure more local consistente surfaces | True | {Fasle, True} |
| patch_leaf_size | Determins the number of cells considered in the adjacency calculations n+1 | 2 | 1-3 |
| min_adj_voxel | The minimum number of adjacent cells to generate a valid estimate for a cell | 3 | 3-8 | 
