#pragma once

#include "Geometry/Lpoint.hpp"
#include "Geometry/Lpoint64.hpp"
#include "Geometry/point.hpp"
#include "octree.hpp"
#include "octree_linear.hpp"
#include "PointEncoding/common.hpp"
#include "PointEncoding/morton_encoder.hpp"
#include "PointEncoding/hilbert_encoder.hpp"

template <PointType T>
std::string getPointName();

template <template <typename, typename> class Octree_t>
std::string getOctreeName();

namespace PointEncoding {
    template <typename Encoder_t>
    std::string getEncoderName();
}