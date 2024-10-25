// Copyright (c) 2022
// Commonwealth Scientific and Industrial Research Organisation (CSIRO)
//
// Author: Fabio Ruetz
// Adapted from Tomas Low
#include "nve_tools/TrajectoryLoader.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>

namespace nve_tools
{

    /**Loads the trajectory into the supplied vector and returns if successful*/
    bool TrajectoryIO::load(const std::string &file_name)
    {
      std::cout << "[TrajectoryLoader] Loading trajectory " << file_name << std::endl;
      std::string line;
      int size = 0;
      bool has_labels = false;

      // Pre-counting the size of the file
      {
        std::ifstream ifs(file_name.c_str(), std::ios::in);
        if (!ifs)
        {
          std::cerr << "1) Failed to open trajectory file: " << file_name << std::endl;
          return false;
        }
        assert(ifs.is_open());
        getline(ifs, line);

        if (line.find("label") != std::string::npos)
        {
          has_labels = true;
        }

        while (!ifs.eof())
        {
          getline(ifs, line);
          if(line.empty())
          {
            break;
          }
          size++;
        }
      }

      std::ifstream ifs(file_name.c_str(), std::ios::in);
      if (!ifs)
      {
        std::cerr << "2) Failed to open trajectory file: " << file_name << std::endl;
        return false;
      }
      getline(ifs, line); // We expect header to be removed here
      points_.resize(size);
      quats_.resize(size);
      times_.resize(size);
      if (has_labels)
        labels_.resize(size);
      bool ordered = true;

      for (int i = 0; i < size; i++)
      {
        if (!ifs)
        {
          std::cerr << "Invalid stream when loading trajectory file1: " << file_name << std::endl;
          return false;
        }

        getline(ifs, line);
        std::istringstream iss(line);
        
        if(line.empty())
        {
          break;
        }
        
        if (has_labels)
        {
          iss >> times_[i] >> points_[i][0] >> points_[i][1] >> points_[i][2] >> quats_[i].w >> quats_[i].x >>
              quats_[i].y >> quats_[i].z >> labels_[i];
        }
        else
        {
          iss >> times_[i] >> points_[i][0] >> points_[i][1] >> points_[i][2] >> quats_[i].w >> quats_[i].x >>
              quats_[i].y >> quats_[i].z;
        }

        // Force a normalisation
        quats_[i] = glm::normalize( quats_[i]);
        
        if (i > 0 && times_[i] < times_[i - 1])
          ordered = false;
      }
      // if (!ifs) /// Not sure what this does
      // {
      //   std::cerr << "Invalid stream when loading trajectory file: " << file_name << std::endl;
      //   return false;
      // }

      if (!ordered)
      {
        std::cout << "Warning: trajectory times not ordered. Ordering them now." << std::endl;

        struct Temp // ToDo(rue011): This is ugly, move to top of file
        {
          double time;
          size_t index;
        };

        std::vector<Temp> time_list(times_.size());
        for (size_t i = 0; i < time_list.size(); i++)
        {
          time_list[i].time = times_[i];
          time_list[i].index = i;
        }
        std::sort(time_list.begin(), time_list.end(), [](const Temp &a, const Temp &b)
                  { return a.time < b.time; });

        std::vector<glm::dvec3> new_points(points_.size());
        std::vector<glm::dvec4> new_quats(points_.size());
        std::vector<double> new_times(times_.size());
        std::vector<int> new_labels(points_.size());

        for (size_t i = 0; i < points_.size(); i++)
        {
          new_points[i] = points_[time_list[i].index];
          new_quats[i] = new_quats[time_list[i].index];
          new_times[i] = times_[time_list[i].index];

          if (has_labels)
          {
            new_labels[i] = labels_[time_list[i].index];
          }

          if (!(i % 100))
          {
            std::cout << "time: " << new_times[i] - new_times[0] << std::endl;
          }
        }
        points_ = std::move(new_points);
        quats_ = std::move(new_quats);
        times_ = std::move(new_times);

        if (has_labels)
        {
          labels_ = std::move(new_labels);
        }
        std::cout << "Finished sorting" << std::endl;
      }

      return true;
    }

    size_t TrajectoryIO::nearestPose(double time, glm::dvec3 &output_pos, glm::dvec4 &output_quat)
    {
      if (points_.empty())
      {
        return -1;
      }
      if (points_.size() == 1)
      {
        output_pos = points_[0];
        output_quat = quats_[0];
        return 0;
      }

      double ratio = time;
      size_t index = getIndexAndNormaliseTime(ratio); // This function takes a time and modifies it to a ratio

      // Clamp to initial and last transform
      if (index == 0 || index >= (points_.size() - 1))
      {
        output_pos = points_[index];
        output_quat = quats_[index];
        return index;
      }

      // Interpolate translation
      output_pos = points_[index] * (1 - ratio) + points_[index + 1] * ratio;

      // Interpolate rotation
      glm::dquat quat1;
      quat1.w = quats_[index].w;
      quat1.x = quats_[index].x;
      quat1.y = quats_[index].y;
      quat1.z = quats_[index].z;

      glm::dquat quat2;
      quat2.w = quats_[index + 1].w;
      quat2.x = quats_[index + 1].x;
      quat2.y = quats_[index + 1].y;
      quat2.z = quats_[index + 1].z;

      auto q_temp = glm::slerp(quat1, quat2, ratio);
      output_quat.w = q_temp.w;
      output_quat.x = q_temp.x;
      output_quat.y = q_temp.y;
      output_quat.z= q_temp.z; // ToDo(rue011): Ugly and verbose conversion

      output_quat = glm::normalize(output_quat);

      return index;
    }

    size_t TrajectoryIO::nearestPose(double time, glm::dvec3 &output_pos, glm::dvec4 &output_quat, int &label)
    {
      size_t index = nearestPose(time, output_pos, output_quat);

      double ratio = time;
      index = getIndexAndNormaliseTime(ratio);

      index = ratio >= 0.5 ? index + 1 : index;

      if (0 >= index || labels_.size() != points_.size() || labels_.empty())
      {
        label = -1;
        return 0;
      }
      label = labels_[index];
      return index;
    }

    bool TrajectoryIO::write(std::string file_name, bool skip)
    {
      std::ofstream ofs(file_name);
      ofs.unsetf(std::ios::floatfield);
      ofs.precision(15);
      if (!ofs.is_open() || points_.empty())
      {
        return false;
      }

      // Header
      if (!labels_.empty())
      {
        ofs << "%time x y z qw qx qy qz label" << std::endl;
      }
      else
      {
        ofs << "%time x y z qw qx qy qz" << std::endl;
      }

      for (size_t i = 0; i < points_.size(); i++)
      {

        ofs << std::fixed << times_[i] << " " << points_[i].x << " " << points_[i].y << " " << points_[i].z << " "
            << quats_[i].w << " " << quats_[i].x << " " << quats_[i].y << " " << quats_[i].z;
        if (!labels_.empty())
        {
          ofs << " " << labels_[i];
        }
        ofs << std::endl;
      }

      ofs.close();
      return true;
    }

    void TrajectoryIO::assignLabelToPose(double time, int label)
    {
      if (labels_.empty() || times_.size() != labels_.size())
      {
        // std::cerr << "Could not assign label. Check if labels are initializer or time and labels size mach" << std::endl;
        return void();
      }
      double ratio = time;
      auto index = getIndexAndNormaliseTime(ratio);
      index = ratio > 0.5 ? index + 1 : index;
      labels_[index] = label;
    }

} // namespace nve_tools
