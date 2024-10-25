// Copyright (c) 2021
// Commonwealth Scientific and Industrial Research Organisation (CSIRO)
// ABN 41 687 119 230
//
// Author: Fabio Ruetz

#include <gtest/gtest.h>

#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "nve_tools/TrajectoryLoader.h"

/// ToDo (rue011): Could copy the files with CMake into the installation directory... See ohm
std::string traj_test_path = "/workspaces/nve_docker_ws/src/nve_tools/test/data/TestTraj.txt";
std::string semantic_test_traj_file = "/workspaces/nve_docker_ws/src/nve_tools/test/data/TestTrajLabled.txt";

/// Note: quaterions are stores in glm in [x, y, z, w] fashion!
/// So use accessor q.x == q[0],...., q.w== q[3]

TEST(TrajectoryLoader, Load)
{ nve_tools::TrajectoryIO traj_loader;
  EXPECT_TRUE(traj_loader.load(traj_test_path));
  EXPECT_FALSE(traj_loader.load(traj_test_path + "sasd"));
}

TEST(TrajectoryLoader, Initial_Entry)
{
  nve_tools::TrajectoryIO traj_loader;
  traj_loader.load(traj_test_path);
  EXPECT_EQ(traj_loader.points().size(), traj_loader.quats().size());

  // Check first entry
  double time = 0.0;
  double t0 = traj_loader.times().front();
  EXPECT_DOUBLE_EQ(t0, time);

  glm::dvec3 pos(0, 0, 0);
  glm::dvec3 &p0 = traj_loader.points().front();
  EXPECT_EQ(p0[0], pos[0]);
  EXPECT_EQ(p0[1], pos[1]);
  EXPECT_EQ(p0[2], pos[2]);

  glm::dvec4 quat(0, 0, 0, 1);
  glm::dvec4 quat0 = traj_loader.quats().front();
  EXPECT_EQ(quat0.x, quat.z);
  EXPECT_EQ(quat0.y, quat.y);
  EXPECT_EQ(quat0.z, quat.z);
  EXPECT_EQ(quat0.w, quat.w);
  EXPECT_EQ(quat0.w, 1);

}


TEST(TrajectoryLoader, Last_Entry)
{
  nve_tools::TrajectoryIO traj_loader;
  traj_loader.load(traj_test_path);

  EXPECT_EQ(traj_loader.points().size(), traj_loader.quats().size());

  // Check last entry
  double time = 1.2;
  double tn = traj_loader.times().back();
  EXPECT_DOUBLE_EQ(tn, time);

  glm::dvec3 pos(1.0, 1.1, 1.2);
  glm::dvec3 &pn = traj_loader.points().back();
  EXPECT_EQ(pn[0], pos[0]);
  EXPECT_EQ(pn[1], pos[1]);
  EXPECT_EQ(pn[2], pos[2]);

  glm::dvec4 quat(0.1, 0.1, 0.1, 0.7);
  glm::dvec4 quatn = traj_loader.quats().back();
  EXPECT_EQ(quatn.x, quat.z);
  EXPECT_EQ(quatn.y, quat.y);
  EXPECT_EQ(quatn.z, quat.z);
  EXPECT_EQ(quatn.w, quat.w);
}

TEST(TrajectoryLoader, Pose_interpolation)
{
  nve_tools::TrajectoryIO traj_loader;
  traj_loader.load(traj_test_path);

  double tk = 0.3;
  glm::dvec4 quat_45( 0.0, 0.0, 0.423, 0.906);

  glm::dvec3 pos_interp;
  glm::dvec4 quat_interpol;

  traj_loader.nearestPose(tk, pos_interp, quat_interpol);
  EXPECT_NEAR(quat_45[0], quat_interpol[0], 0.001);
  EXPECT_NEAR(quat_45[1], quat_interpol[1], 0.001);
  EXPECT_NEAR(quat_45[2], quat_interpol[2], 0.001);
  EXPECT_NEAR(quat_45[3], quat_interpol[3], 0.001);
}

TEST(TrajectoryLoader, ClampPoseInitial)
{
  nve_tools::TrajectoryIO traj_loader;
  traj_loader.load(traj_test_path);

  /// Should clamp to the intial pose
  double tk = -0.3;
  glm::dvec3 pose_tk;
  glm::dvec4 quat_tk;

  auto index = traj_loader.nearestPose(tk, pose_tk, quat_tk);


  glm::dvec3 pos_eval(0.0, 0.0, 0.0);
  EXPECT_EQ(index, size_t(0));
  EXPECT_DOUBLE_EQ(pose_tk[0], pos_eval[0]);
  EXPECT_DOUBLE_EQ(pose_tk[1], pos_eval[1]);
  EXPECT_DOUBLE_EQ(pose_tk[2], pos_eval[2]);

  glm::dvec4 quat_eval( 0.0, 0.0, 0.0, 1.0);
  EXPECT_NEAR(quat_eval[0], quat_tk[0], 0.001);
  EXPECT_NEAR(quat_eval[1], quat_tk[1], 0.001);
  EXPECT_NEAR(quat_eval[2], quat_tk[2], 0.001);
  EXPECT_NEAR(quat_eval[3], quat_tk[3], 0.001);
  EXPECT_NEAR(quat_tk.w, 1.0, 0.001);
}

TEST(TrajectoryLoader, ClampPoseFinal)
{
  nve_tools::TrajectoryIO traj_loader;
  traj_loader.load(traj_test_path);

  /// Should clamp to the final pose
  double tk = 110.3;
  glm::dvec3 pose_tk;
  glm::dvec4 quat_tk;

  auto index = traj_loader.nearestPose(tk, pose_tk, quat_tk);


  glm::dvec3 pos_eval(1.0, 1.1, 1.2);
  EXPECT_EQ(index, traj_loader.points().size() - 1);
  EXPECT_DOUBLE_EQ(pose_tk[0], pos_eval[0]);
  EXPECT_DOUBLE_EQ(pose_tk[1], pos_eval[1]);
  EXPECT_DOUBLE_EQ(pose_tk[2], pos_eval[2]);

  glm::dvec4 quat_eval( 0.1, 0.1, 0.1, 0.7);
  EXPECT_NEAR(quat_eval[0], quat_tk[0], 0.001);
  EXPECT_NEAR(quat_eval[1], quat_tk[1], 0.001);
  EXPECT_NEAR(quat_eval[2], quat_tk[2], 0.001);
  EXPECT_NEAR(quat_eval[3], quat_tk[3], 0.001);
  EXPECT_NEAR(0.7, quat_tk.w, 0.001);
}

TEST(SemanticTrajectory, NoSemantics)
{
  nve_tools::TrajectoryIO traj_loader;
  traj_loader.load(traj_test_path);
  glm::dvec3 pose_tk;
  glm::dvec4 quat_tk;
  int label_k;

  auto index_0 = traj_loader.nearestPose(0, pose_tk, quat_tk, label_k);
  EXPECT_EQ(index_0, size_t(0));
  EXPECT_EQ(label_k, -1);  // Cannot work, as size_t is strictly positive ...
}




TEST(SemanticTrajectory, SemanticLabels)
{
  nve_tools::TrajectoryIO traj_loader;
  traj_loader.load(semantic_test_traj_file);
  glm::dvec3 pose_tk;
  glm::dvec4 quat_tk;
  int label_k;

  EXPECT_TRUE(traj_loader.has_labels());
  
  // Clamp begin
  auto index_0 = traj_loader.nearestPose(0.0001, pose_tk, quat_tk, label_k);
  EXPECT_EQ(label_k, -1); /// Default behaviour to prevent times outside of class to be read
  EXPECT_EQ(index_0, 0 );

  // Middle
  auto index_middle = traj_loader.nearestPose(0.2, pose_tk, quat_tk, label_k);
  EXPECT_EQ(label_k, 2.0);
  EXPECT_EQ(index_middle, 1);

  // Clamp end()
  auto index_5 = traj_loader.nearestPose(22312, pose_tk, quat_tk, label_k);
  EXPECT_EQ(label_k, traj_loader.labels().back());
  EXPECT_EQ(index_5, traj_loader.points().size() - 1);
}