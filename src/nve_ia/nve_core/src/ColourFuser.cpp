#include "ColourFuser.h"

namespace nve_core
{

ColourFuserEP::ColourFuserEP(std::shared_ptr<ohm::OccupancyMap> map)
  : map_(map)
{
  colour_layer_ = ohm::Voxel<ohm::AppearanceVoxel>(map_.get(), map_->layout().appearanceLayer());
}

void ColourFuserEP::integrate(const std::vector<Eigen::Vector3d> &points, const std::vector<Eigen::Vector3d> &colours)
{
  for (auto i = 0; i < points.size(); i++)
  {
    integrate(points[i], colours[i]);
  }
}

void ColourFuserEP::integrate(const Eigen::Vector3d &point, const Eigen::Vector3d &colour)
{
  auto key = map_->voxelKey(glm::dvec3(point.x(), point.y(), point.z()));
  ohm::setVoxelKey(key, colour_layer_);

  if (colour_layer_.isValid())
  {
    colour_layer_.read(&colour_voxel_);
  }

  ++colour_voxel_.count;

  /// If the voxel is not initialize we set the voxel to the current colour meassurment with a high
  /// variance
  if (colour_voxel_.count == 1)
  {
    colour_voxel_.red[0] = static_cast<float>(colour[0]);  // Measurement is mean
    colour_voxel_.red[1] = 0.5;
    colour_voxel_.green[0] = colour[1];  // Measurement is mean
    colour_voxel_.green[1] = 0.5;
    colour_voxel_.blue[0] = colour[2];  // Measurement is mean
    colour_voxel_.blue[1] = 0.5;
  }

  ohm::updateAppearanceVoxel(colour_voxel_.red, colour[0], colour_voxel_.count);
  ohm::updateAppearanceVoxel(colour_voxel_.green, colour[1], colour_voxel_.count);
  ohm::updateAppearanceVoxel(colour_voxel_.blue, colour[2], colour_voxel_.count);

  colour_layer_.write(colour_voxel_);
}

bool ColourFuserEP::getColour(const Eigen::Vector3d &position, Eigen::Vector3d &colour)
{
  auto key = map_->voxelKey(glm::dvec3(position.x(), position.y(), position.z()));
  return getColour(key, colour);
}

bool ColourFuserEP::getColour(const ohm::Key &key, Eigen::Vector3d &colour)
{
  ohm::setVoxelKey(key, colour_layer_);
  if (not colour_layer_.isValid())
  {
    return false;
  }

  colour_layer_.read(&colour_voxel_);

  if (colour_voxel_.count < 1)  // We have not observed a colour
  {
    return false;
  }

  colour = Eigen::Vector3d(colour_voxel_.red[0], colour_voxel_.green[0], colour_voxel_.blue[0]);

  return true;
}

bool ColourFuserEP::getColourAndPosition(const ohm::Key &key, Eigen::Vector3d &pos, Eigen::Vector3d &colour)
{
  ohm::setVoxelKey(key, colour_layer_);
  if (not colour_layer_.isValid())
  {
    return false;
  }

  colour_layer_.read(&colour_voxel_);
  if (colour_voxel_.count < 1)  // We have not observed a colour
  {
    return false;
  }

  glm::dvec3 postion = map_->voxelCentreGlobal(key);

  colour = Eigen::Vector3d(colour_voxel_.red[0], colour_voxel_.green[0], colour_voxel_.blue[0]);
  pos = Eigen::Vector3d(postion[0], postion[1], postion[2]);

  return true;
}


bool ColourFuserEP::setColour(const ohm::Key &key, const Eigen::Vector3f &colour)
{
  ohm::setVoxelKey(key, colour_layer_);
  if (not colour_layer_.isValid())
  {
    return false;
  }
  colour_voxel_ = ohm::AppearanceVoxel();

  colour_voxel_.red[0] = colour[0];
  colour_voxel_.green[0] = colour[1];
  colour_voxel_.blue[0] = colour[2];

  colour_layer_.write(colour_voxel_);
}


OhmColourSetter::OhmColourSetter(std::shared_ptr<ohm::OccupancyMap> map)
  : map_(map)
{
  colour_layer_ = ohm::Voxel<ohm::VoxelRGBA>(map_.get(), map_->layout().rgbaLayer());
}


bool OhmColourSetter::setColour(const ohm::Key &key, const Eigen::Vector3d &colour)
{
  ohm::setVoxelKey(key, colour_layer_);
  if (not colour_layer_.isValid())
  {
    return false;
  }
  colour_voxel_ = ohm::VoxelRGBA();

  colour_voxel_.red = static_cast<u_int8_t>(colour[0] * 255.0);
  colour_voxel_.green = static_cast<u_int8_t>(colour[1] * 255.0);
  colour_voxel_.blue = static_cast<u_int8_t>(colour[2] * 255.0);
  colour_voxel_.alpha = 255;

  colour_layer_.write(colour_voxel_);
}


bool OhmColourSetter::getColour(const ohm::Key &key, Eigen::Vector3d &colour)
{
  ohm::setVoxelKey(key, colour_layer_);
  if (not colour_layer_.isValid())
  {
    return false;
  }

  colour_layer_.read(&colour_voxel_);

  colour = Eigen::Vector3d(double(colour_voxel_.red) / 255.0, double(colour_voxel_.green) / 255.0,
                           double(colour_voxel_.blue) / 255.0);

  return true;
}


}  // namespace nve_core
