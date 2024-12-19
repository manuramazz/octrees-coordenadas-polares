#include "type_names.hpp"

// Specializations for getPointName
template <>
std::string getPointName<Lpoint64>() {
    return "Lpoint64";
}

template <>
std::string getPointName<Lpoint>() {
    return "Lpoint";
}

template <>
std::string getPointName<Point>() {
    return "Point";
}

// Specializations for getOctreeName
template <>
std::string getOctreeName<Octree>() {
    return "Pointer";
}

template <>
std::string getOctreeName<LinearOctree>() {
    return "Linear";
}

// Specializations for getEncoderName
namespace PointEncoding {
    template <>
    std::string getEncoderName<PointEncoding::NoEncoder>() {
        return "Unencoded";
    }

    template <>
    std::string getEncoderName<PointEncoding::MortonEncoder32>() {
        return "MortonEncoder32";
    }

    template <>
    std::string getEncoderName<PointEncoding::MortonEncoder64>() {
        return "MortonEncoder64";
    }

    template <>
    std::string getEncoderName<PointEncoding::HilbertEncoder64>() {
        return "HilbertEncoder64";
    }
}
