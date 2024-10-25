#include "RayBuffer.h"

namespace nve_core {

RayBuffer::~RayBuffer()
{
  if (cloud_loader_.isOpen())
  {
    cloud_loader_.close();
  }
}

bool RayBuffer::get_points_around(const double timestamp, std::vector<glm::dvec3> &ray_endpoints)
{
  /// Error handling
  update_around_ts(timestamp);

  ray_endpoints.clear();
  ray_endpoints.reserve(inner_ts_ / outer_ts_ * buffered_points_.size() * 1.2);

  for (auto &sample : buffered_points_)
  {
    if (std::abs(sample.timestamp - timestamp) <= inner_ts_)
    {
      ray_endpoints.emplace_back(sample.position);
    }
    else if (sample.timestamp > timestamp + inner_ts_)
    {
      // Specific case where the sample points
      break;
    }
  }

  /// Go trough the points and get the fucking examples
  return true;
}

bool RayBuffer::need_to_update(const double time_stamp)
{
  // We have no points, so lets update
  if (buffered_points_.empty())
  {
    return true;
  }
  // Innder stride is to small, we need more points
  if ((time_stamp + inner_ts_ > target_timestamp_ + outer_ts_) or
      (time_stamp - inner_ts_ < target_timestamp_ - outer_ts_))
  {
    return true;
  }

  /// TODO: Handle this condition
  if (time_stamp < target_timestamp_)
  {
    return true;
  }

  return false;
}

void RayBuffer::update_around_ts(const double time_stamp)
{
  if (need_to_update(time_stamp))
  {
    // Only place we update the target time
    target_timestamp_ = time_stamp;

    remove_non_valid_samples();

    update_buffers();
  }
}

void RayBuffer::remove_non_valid_samples()
{
  if (buffered_points_.size() < 1)
  {
    return void();
  }

  std::vector<slamio::CloudPoint> temp_samples;
  temp_samples.reserve(buffered_points_.size());

  for (const auto &sample : buffered_points_)
  {
    if (std::abs(sample.timestamp - target_timestamp_) < outer_ts_)
    {
      temp_samples.emplace_back(sample);
    }
  }

  buffered_points_.swap(temp_samples);
}

/// TODO: We are not considering the single point which is just outside the update
void RayBuffer::update_buffers()
{
  slamio::CloudPoint sample;

  while (cloud_loader_.readNext(sample))
  {
    if (std::abs(sample.timestamp - target_timestamp_) <= outer_ts_)
    {
      buffered_points_.emplace_back(sample);
      buffered_timestamps_.emplace_back(sample.timestamp);
    }
    else if (sample.timestamp > target_timestamp_ + outer_ts_)
    {
      // Specific case where the sample points
      // Need to add last point to not loose data
      buffered_points_.emplace_back(sample);
      return void();
    }
    else
    {
      continue;
    }
  }
  /// We reached end of file, so no time can be found
  end_of_file_ = true;
}

bool RayBuffer::open_file(std::string file, const double inner_ts, const double outer_ts)
{
  inner_ts_ = inner_ts;
  outer_ts_ = outer_ts;

  cloud_loader_.setDesiredChannels(slamio::DataChannel::Position| slamio::DataChannel::Time );
  return cloud_loader_.open(file.c_str());
}

}  // namespace nve_ia
