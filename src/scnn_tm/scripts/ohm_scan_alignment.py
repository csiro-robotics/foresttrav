import numpy as np
import pandas as pd
import open3d as o3d
import copy


csv_file = "/data/debug/ohm_scan_alignment/1635393844.041707_pose_cloud.csv"
ohm_global_ply_path = "/data/debug/ohm_scan_alignment/global_2021_10_28_04_03_11Z_tm_map_v0.1_cloud.ply"


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.0, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    b =  source_temp.transform(transformation)
    o3d.visualization.draw_geometries([b, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])


if __name__== '__main__':
  
  # Read the ohm_scans
  df = pd.read_csv(csv_file)
  ohm_scan_cloud = o3d.geometry.PointCloud()
  ohm_scan_cloud.points =  o3d.utility.Vector3dVector( df[["x","y","z"]].to_numpy())
  
  # Read the ply files
  ohm_global_cloud = o3d.io.read_point_cloud(ohm_global_ply_path)
  
  # Crop the cloud to fit around the scan 
  axis_aligned_bb = o3d.geometry.AxisAlignedBoundingBox()
  axis_aligned_bb.min_bound = np.array([-7, -7, -3])
  axis_aligned_bb.max_bound = np.array([7, 7, 3])
  ohm_global_cloud_cropped = ohm_global_cloud.crop(axis_aligned_bb)
  ohm_scan_cloud_cropped = ohm_scan_cloud.crop(axis_aligned_bb)
  
  # Ge initial guess. Note the cloud should be in the "map" frame and the transform should be uniform
  T_global_scan = np.eye((4))
  # T_global_scan[0,3] = 0.05 
  # T_global_scan[1,3] = -0.05
  # T_global_scan[2,3] = 0.5
  
  # print("Apply point-to-point ICP")
  threshold = 0.25
  reg_p2p = o3d.pipelines.registration.registration_icp(
    ohm_scan_cloud_cropped,ohm_global_cloud_cropped , threshold, T_global_scan,
    o3d.pipelines.registration.TransformationEstimationPointToPoint())
  print(reg_p2p)
  print("Transformation is:")
  print(reg_p2p.transformation)

  draw_registration_result( ohm_global_cloud_cropped,ohm_scan_cloud_cropped, reg_p2p.transformation)
  