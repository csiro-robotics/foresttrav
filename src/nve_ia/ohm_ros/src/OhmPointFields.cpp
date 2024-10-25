// Copyright (c) 2022
// Commonwealth Scientific and Industrial Research Organisation (CSIRO)
// 
//
// Original Author: Thomas Hines <thomas.hines@data61.csiro.au>
// Adapted: Fabio Ruetz  <fabio.ruetz@data61.csiro.au>
#include "OhmPointFields.h"

namespace nve_ros
{
PointCloudFields::PointCloudFields(const uint8_t coord_field_type, const uint8_t intensity_field_type)
{
  x_ = addField("x", coord_field_type, 1);
  y_ = addField("y", coord_field_type, 1);
  z_ = addField("z", coord_field_type, 1);
  intensity_ = addField("intensity", intensity_field_type, 1);
}

sensor_msgs::PointField *PointCloudFields::addField(const char *const name, const uint8_t type, const unsigned count)
{
  if (field_count_ == fields_.size())
  {
    throw std::length_error("PointCloudFields: Tried to add a field but internal fields array is already fully used.");
  }

  sensor_msgs::PointField *field = &fields_[field_count_++];
  field->name = name;
  field->offset = unsigned(point_step_);
  field->datatype = type;
  field->count = count;

  switch (type)
  {
  case sensor_msgs::PointField::INT8:
  case sensor_msgs::PointField::UINT8:
    point_step_ += 1;
    break;
  case sensor_msgs::PointField::INT16:
  case sensor_msgs::PointField::UINT16:
    point_step_ += 2;
    break;
  case sensor_msgs::PointField::INT32:
  case sensor_msgs::PointField::UINT32:
  case sensor_msgs::PointField::FLOAT32:
    point_step_ += 4;
    break;
  case sensor_msgs::PointField::FLOAT64:
    point_step_ += 8;
    break;
  default:
    throw std::runtime_error("PointCloudFields: Unknown PointField type");
  }

  return field;
}

size_t PointCloudFields::size() const
{
  return field_count_;
}

const sensor_msgs::PointField &PointCloudFields::operator[](const size_t pos) const
{
  return fields_[pos];
}

uint32_t PointCloudFields::pointStep() const
{
  return point_step_;
}

const sensor_msgs::PointField *PointCloudFields::x() const
{
  return x_;
}

const sensor_msgs::PointField *PointCloudFields::y() const
{
  return y_;
}

const sensor_msgs::PointField *PointCloudFields::z() const
{
  return z_;
}

const sensor_msgs::PointField *PointCloudFields::intensity() const
{
  return intensity_;
}

bool populateCloudMessage(sensor_msgs::PointCloud2 &map_cloud, const ohm::OccupancyMap &map,
                          const CloudPointFunc &voxel_func, const PointCloudFields &fields,
                          const CloudExtendedAttributesFunc &write_extended_fields)
{
  // Initialise cloud properties
  map_cloud.height = 1u;
  map_cloud.width = 0u;
  map_cloud.is_bigendian = false;
  map_cloud.point_step = fields.pointStep();
  map_cloud.row_step = 0u;
  map_cloud.is_dense = false;

  // Initialise cloud fields
  map_cloud.fields.reserve(fields.size());
  for (size_t index = 0; index < fields.size(); ++index)
  {
    map_cloud.fields.emplace_back(fields[index]);
  }

  // Check coordinate and intensity fields are supported types
  if ((fields.x() == nullptr) || (fields.y() == nullptr) || (fields.z() == nullptr) ||
      (fields.intensity() == nullptr) || (fields.x()->datatype != sensor_msgs::PointField::FLOAT32) ||
      (fields.y()->datatype != sensor_msgs::PointField::FLOAT32) ||
      (fields.z()->datatype != sensor_msgs::PointField::FLOAT32) ||
      (fields.intensity()->datatype != sensor_msgs::PointField::UINT16))
  {
    throw std::runtime_error("PointCloudFields: x, y, z and intensity fields are required. "
                             "x, y and z must be FLOAT32 and intensity must be UINT16.");
  }

  // Check map has occupancy
  ohm::Voxel<const float> voxel_occupancy(&map, map.layout().occupancyLayer());
  if (!voxel_occupancy.isLayerValid())
  {
    return false;
  }

  // Iterate over all chunks in the map
  std::vector<const ohm::MapChunk *> map_chunks;
  map.enumerateRegions(map_chunks);
  for (const ohm::MapChunk *const chunk : map_chunks)
  {
    // Check chunk has occupancy
    voxel_occupancy.setKey(ohm::Key(chunk->region.coord, 0, 0, 0), chunk);
    if (voxel_occupancy.isValid())
    {
      // Check every voxel in chunk
      do
      {
        glm::dvec3 voxel_pos{
          std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::quiet_NaN(),
        };
        uint16_t intensity_value = 0;
        if (voxel_func(voxel_occupancy, &voxel_pos, &intensity_value))
        {
          // Record the write offset.
          const size_t write_index = map_cloud.data.size();

          // Add data to the cloud array.
          for (uint32_t index = 0; index < fields.pointStep(); ++index)
          {
            map_cloud.data.push_back(0);
          }

          // Write to the target location.
          uint8_t *const point_mem = &map_cloud.data[write_index];
          *reinterpret_cast<float *>(point_mem + fields.x()->offset) = float(voxel_pos.x);
          *reinterpret_cast<float *>(point_mem + fields.y()->offset) = float(voxel_pos.y);
          *reinterpret_cast<float *>(point_mem + fields.z()->offset) = float(voxel_pos.z);
          *reinterpret_cast<uint16_t *>(point_mem + fields.intensity()->offset) = intensity_value;
          if (write_extended_fields)
          {
            write_extended_fields(fields, point_mem, voxel_occupancy, voxel_pos);
          }

          ++map_cloud.width;
        }
      } while (voxel_occupancy.nextInRegion());
    }
  }

  if (map_cloud.width <= 0)
  {
    return false;
  }

  map_cloud.row_step = map_cloud.point_step * map_cloud.width;

  return true;
}

}  // namespace ohmmapping