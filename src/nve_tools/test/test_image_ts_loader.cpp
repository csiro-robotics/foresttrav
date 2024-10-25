// Copyright (c) 2021
// Commonwealth Scientific and Industrial Research Organisation (CSIRO)
// ABN 41 687 119 230
//
// Author: Fabio Ruetz

#include <gtest/gtest.h>

#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "nve_tools/ImageTSLoader.h"

/// ToDo (rue011): Could copy the files with CMake into the installation directory... See ohm
std::string file_test_path = "/workspaces/nve_docker_ws/src/nve_tools/test/data/camera0_test.timestamps";

TEST(ImageLoader, Load)
{
  nve_tools::ImageTSLoader img_ts_loader;
  EXPECT_TRUE(img_ts_loader.load(file_test_path));
  EXPECT_FALSE(img_ts_loader.load(file_test_path + "sasd"));
}
