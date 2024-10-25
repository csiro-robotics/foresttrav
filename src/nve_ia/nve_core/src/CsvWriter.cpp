#include "CsvWriter.h"

#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>


namespace nve_core
{
namespace io
{
CsvWriter::CsvWriter(std::string file_name)
  : output_file_(file_name)
{
  ofs_ = std::make_shared<std::ofstream>(output_file_);
  if (!(*ofs_).is_open())
  {
    std::cerr << "Could not open file to write to " << output_file_ << std::endl;
  }
  (*ofs_) << std::setprecision(8);
}

int CsvWriter::writeFeaturesToCSV(std::string output, Features &features, std::vector<std::string> &feature_header)
{
  if (!(*ofs_).is_open() || features.empty())
  {
    std::cerr << "Could open file to write to " << output_file_ << std::endl;
    return -2;
  }

  if (features[0].size() == feature_header.size() && !feature_header.empty())
  {
    if (!writeHeader(feature_header))
      return -2;
  }

  (*ofs_) << std::fixed << std::setprecision(8);
  for (auto feature : features)
  {
    streamFeature(feature);
  }

  return 0;
}

bool CsvWriter::writeHeader(const std::vector<std::string> &feature_header)
{
  if (!(*ofs_).is_open() || feature_header.empty())
  {
    return false;
  }

  *ofs_ << feature_header[0];
  for (int j = 1; j < feature_header.size(); j++)
  {
    *ofs_ << "," << feature_header[j];
  }
  *ofs_ << std::endl;

  return true;
}

void CsvWriter::streamFeature(const FeatureVector &feature_vector)
{
  if (!(*ofs_).is_open())
  {
    return void();
  }

  *ofs_ << feature_vector[0];  // Class label
  for (int j = 1; j < feature_vector.size(); j++)
  {
    if (std::isfinite(feature_vector[j]))
    {
      *ofs_ << "," << feature_vector[j];
    }
    else
    {
      *ofs_ << "," << 0.0;
    }
  }
  *ofs_ << std::endl;
}

void CsvWriter::close()
{
  (*ofs_).close();
}

}  // namespace utils
}  // namespace nve_ia