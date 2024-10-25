// Copyright (c) 2022
// Commonwealth Scientific and Industrial Research Organisation (CSIRO)
// ABN 41 687 119 230
//
// Original Author: Thomas Hines <thomas.hines@data61.csiro.au>
// Adapted: Fabio Ruetz  <fabio.ruetz@data61.csiro.au>
#ifndef OHM_POINTFIELDS_H
#define OHM_POINTFIELDS_H

#include <ohm/Voxel.h>

#include <sensor_msgs/PointCloud2.h>

namespace nve_ros
{
class PointCloudFields;

/// Function used to resolve a voxel position and intensity with
/// @c populateCloudMessage()
/// @param voxel_occupancy The voxel with its occupancy layer
/// @param[out] voxel_pos Set to the position values to use in the cloud
/// @param[out] intensity_value Set to the intensity value to use in the cloud
/// @return true iff the voxel should be written to the cloud
/// The function body should set
using CloudPointFunc = std::function<bool(const ohm::Voxel<const float> &, glm::dvec3 *, uint16_t *)>;

/// Function used to write optional fields in @c populateCloudMessage()
/// @param fields The fields being added to the cloud
/// @param[out] point_mem Write the point's data here
/// @param voxel_occupancy The voxel with its occupancy layer
/// @param voxel_pos The position values to use in the cloud
using CloudExtendedAttributesFunc =
  std::function<void(const PointCloudFields &, uint8_t *, const ohm::Voxel<const float> &, const glm::dvec3 &)>;

/// Defines the point cloud fields for @c populateCloudMessage(). Fields x, y, z
/// and intensity are added immediately and 4 additional fields may be added.
class PointCloudFields
{
public:
  /// The max number of fields that can be added (increase this as more fields
  /// are required)
  const static size_t kMaxSize{ 23 };

  /// Initialise the default fields x, y, z, intensity.
  /// @param coord_field_type The point field type for x, y, z. Default to @c FLOAT32 .
  /// @param intensity_field_type The point field type for intensity. Default to @c UINT16 .
  PointCloudFields(uint8_t coord_field_type = sensor_msgs::PointField::FLOAT32,
                   uint8_t intensity_field_type = sensor_msgs::PointField::UINT16);

  /// Add a field.
  /// @param name The field name.
  /// @param type The @c PointField type.
  /// @param count The number of elements in the field.
  /// @return A pointer to the field details or null on failure.
  sensor_msgs::PointField *addField(const char *name, uint8_t type, uint32_t count = 1);

  /// @return The number of fields that have been added
  size_t size() const;

  /// @param pos Get the field at this index
  /// @return The field at index @p pos
  const sensor_msgs::PointField &operator[](size_t pos) const;

  /// @return The number of bytes that will be written for each point
  uint32_t pointStep() const;

  /// @return A pointer to the x coordinate field or nullptr if there is not one
  const sensor_msgs::PointField *x() const;

  /// @return A pointer to the y coordinate field or nullptr if there is not one
  const sensor_msgs::PointField *y() const;

  /// @return A pointer to the z coordinate field or nullptr if there is not one
  const sensor_msgs::PointField *z() const;

  /// @return A pointer to the intensity field or nullptr if there is not one
  const sensor_msgs::PointField *intensity() const;

private:
  /// The fields
  std::array<sensor_msgs::PointField, kMaxSize> fields_;

  /// The number of fields that have been added
  unsigned field_count_{ 0 };

  /// The number of bytes that will be written for each point
  uint32_t point_step_{ 0 };

  /// A pointer to the x coordinate field or nullptr if there is not one
  sensor_msgs::PointField *x_{ nullptr };

  /// A pointer to the y coordinate field or nullptr if there is not one
  sensor_msgs::PointField *y_{ nullptr };

  /// A pointer to the z coordinate field or nullptr if there is not one
  sensor_msgs::PointField *z_{ nullptr };

  /// A pointer to the intensity field or nullptr if there is not one
  sensor_msgs::PointField *intensity_{ nullptr };
};

/// Populate a PointCloud2 message from an OccupancyMap.
///
/// The functor objects are used to help add additional fields and resolve voxel values.
///
/// @param init_fields Called to initialise the point cloud message fields. The default is to use
///     @c initialiseOhmMapCloudFields() and only needs to be replaced when adding additional fields afterwards.
///     Arguments: @p map_cloud
/// @param voxel_func Invoked for each candidate voxel from @p map. A true return value indicates a valid voxel
///     to write to the cloud.
///     Arguments: @c current_voxel, [out] voxel_pos, [out] voxel_intensity
/// @param write_extended_fields Called for each point to populate additional fields for the voxel point. Only needed
/// when
///     overriding @p init_fields .
///     Arguments: PointCloudFields, mem (pointer to the start of the point memory block), @c current_voxel,
///     @c voxel_pos
bool populateCloudMessage(sensor_msgs::PointCloud2 &map_cloud, const ohm::OccupancyMap &map,
                          const CloudPointFunc &voxel_func, const PointCloudFields &fields = PointCloudFields(),
                          const CloudExtendedAttributesFunc &write_extended_fields = CloudExtendedAttributesFunc());

}  // namespace ohmmapping

#endif  // OHM_MAPPING_POINT_CLOUD_2_WRITER_H