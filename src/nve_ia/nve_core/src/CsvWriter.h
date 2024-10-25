#ifndef CONVERSION_CSV_H
#define CONVERSION_CSV_H

#include <memory>
#include <string>
#include <vector>

namespace nve_core
{
namespace io
{
using FeatureVector = std::vector<double>;
using Features = std::vector<FeatureVector>;

class CsvWriter
{
public:
  CsvWriter(std::string file_name);

  ///
  void setOutputFile(std::string file_name) { output_file_ = file_name; };

  bool writeHeader(const std::vector<std::string> &feature_header);

  void streamFeature(const FeatureVector &feature_vectors);

  /// Write vector with feature vectors
  int writeFeaturesToCSV(std::string output, Features &features, std::vector<std::string> &feature_header);

  void close();

private:
  std::string output_file_;
  std::shared_ptr<std::ofstream> ofs_;
};


}  // namespace utils
}  // namespace nve_ia

#endif