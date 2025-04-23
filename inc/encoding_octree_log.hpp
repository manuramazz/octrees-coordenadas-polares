#pragma once

#include <string>
#include <iostream>
#include <iomanip>
#include "PointEncoding/point_encoder.hpp"
#include "type_names.hpp"

/// @brief Object with logs of the encoding and build time for both octrees and some basic info
struct EncodingOctreeLog {
    // Data cloud size
    size_t cloudSize = 0;
    std::string pointType = "";

    // The encoder being used
    std::string encoderType;

    // Encoding and sorting times
    double encodingTime = 0.0;
    double sortingTime = 0.0;

    // Octree general parameters
    size_t MAX_POINTS = 0;
    double MIN_OCTANT_RADIUS = 0.0;
    std::string octreeType = "";

    // Build step times (vary between LinearOctree and Octree)
    double boundingBoxTime = 0.0;
    double encodingTime2 = 0.0;
    double leafConstructionTime = 0.0;
    double internalMemAllocTime = 0.0;
    double internalConstructionTime = 0.0;
    double geometryTime = 0.0;
    double totalTime = 0.0;

    // Mem used in bytes
    size_t memoryUsed = 0;

    // Amount of nodes
    size_t totalNodes = 0;
    size_t leafNodes = 0;
    size_t internalNodes = 0;

    // Max depth and min radius at max depth
    size_t maxDepthSeen = 0;
    double minRadiusAtMaxDepth = 0.0;

    friend std::ostream& operator<<(std::ostream& os, const EncodingOctreeLog& log) {
        std::string memoryUsedStr = std::to_string(log.memoryUsed / (1024.0 * 1024)) + " MB";
        os << "Encoding and octree construction log:\n";
        os << std::left << std::setw(LOG_FIELD_WIDTH) << "Cloud size:" << log.cloudSize << "\n";
        os << std::left << std::setw(LOG_FIELD_WIDTH) << "Point type:" << log.pointType << "\n";
        os << std::left << std::setw(LOG_FIELD_WIDTH) << "Encoder type:" << log.encoderType << "\n";
        os << std::left << std::setw(LOG_FIELD_WIDTH) << "Encoding time:" << log.encodingTime << " sec\n";
        os << std::left << std::setw(LOG_FIELD_WIDTH) << "Sorting time:" << log.sortingTime << " sec\n";
        os << std::left << std::setw(LOG_FIELD_WIDTH) << "Octree type:" << log.octreeType << "\n";
        os << std::left << std::setw(LOG_FIELD_WIDTH) << "Finding bounding box time:" << log.boundingBoxTime << " sec\n";
        os << std::left << std::setw(LOG_FIELD_WIDTH) << "Leaf construction time:" << log.leafConstructionTime << " sec\n";
        os << std::left << std::setw(LOG_FIELD_WIDTH) << "Internal mem. alloc time:" << log.internalMemAllocTime << " sec\n";
        os << std::left << std::setw(LOG_FIELD_WIDTH) << "Internal part construction time:" << log.internalConstructionTime << " sec\n";
        os << std::left << std::setw(LOG_FIELD_WIDTH) << "Compute extra geometry time:" << log.geometryTime << " sec\n";
        os << std::left << std::setw(LOG_FIELD_WIDTH) << "Total time to build octree:" << log.totalTime << " sec\n\n";
        os << std::left << std::setw(LOG_FIELD_WIDTH) << "Octree memory:" << memoryUsedStr << "\n";
        os << std::left << std::setw(LOG_FIELD_WIDTH) << "Number of nodes:" << log.totalNodes << "\n";
        os << std::left << std::setw(LOG_FIELD_WIDTH) << "  Leafs:" << log.leafNodes << "\n";
        os << std::left << std::setw(LOG_FIELD_WIDTH) << "  Internal nodes:" << log.internalNodes << "\n";
        os << std::left << std::setw(LOG_FIELD_WIDTH) << "Max. depth seen:" << log.maxDepthSeen << "\n";
        os << std::left << std::setw(LOG_FIELD_WIDTH) << "Min. radii seen:" << log.minRadiusAtMaxDepth << "\n\n";
        return os;
    }

    void toCSV(std::ostream& out) const {
        out  << pointType << ","
             << octreeType << ","
             << encoderType << ","
             << cloudSize << ","
             << encodingTime << ","
             << sortingTime << ","
             << MAX_POINTS << ","
             << MIN_OCTANT_RADIUS << ","
             << boundingBoxTime << ","
             << leafConstructionTime << ","
             << internalMemAllocTime << ","
             << internalConstructionTime << ","
             << geometryTime << ","
             << totalTime << ","
             << memoryUsed << ","
             << totalNodes << ","
             << leafNodes << ","
             << internalNodes << ","
             << maxDepthSeen << ","
             << minRadiusAtMaxDepth
             << "\n";
    }
    
    static void writeCSVHeader(std::ostream& out) {
        out  << "point_type,"
             << "oct_type,"
             << "enc_type,"
             << "cloud_size,"
             << "enc_time,"
             << "sort_time,"
             << "max_leaf_points,"
             << "min_oct_radius,"
             << "bbox_time,"
             << "leaf_constr_time,"
             << "internal_alloc_time,"
             << "internal_constr_time,"
             << "extra_geo_time,"
             << "octree_build_time,"
             << "octree_memory,"
             << "number_of_nodes,"
             << "leaf_nodes,"
             << "internal_nodes,"
             << "max_depth_seen,"
             << "min_radii_seen\n";
    }
    
};
