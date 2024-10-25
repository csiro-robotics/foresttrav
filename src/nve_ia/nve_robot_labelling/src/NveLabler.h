#ifndef NVE_LABELER_H
#define NVE_LABELER_H

#include "nve_core/Types.h"


#include <ohm/ohm/KeyHash.h>
#include <ohm/ohm/OccupancyMap.h>
#include <ohm/ohm/VoxelSemanticLabel.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <set>
#include <string>

namespace nve_lfe
{

enum TeLabel
{
  Uninitialized = 0,
  Collision = 1,  // This can refer to non-traversbale or a collision event
  Free = 2, // This can refer to traversable or free space depending on the labeller
  Unknown = 3,
};

/// @brief Virtual strategy base class
class Labler
{
public:
  virtual ~Labler() = default;

  virtual void processEvent(const Eigen::Affine3d &T_mr, const TeLabel label, const double *prob,
                            const bool *override) = 0;

  virtual auto getSemanticData(const ohm::Key &key, ohm::SemanticLabel &semantic_label) -> bool = 0;

  /// @brief  Sets the prior to the map for the labler, i.e. from hand labels
  virtual auto addMapPrior(const ohm::Key key, const TeLabel label) ->void = 0;
};




}  // namespace nve_lfe

#endif  // Header Guard