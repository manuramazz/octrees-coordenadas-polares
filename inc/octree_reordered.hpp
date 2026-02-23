#pragma once


/**
* @class LinearOctree
* 
* @brief
* 
* @authors Manuel Ramallo Blanco 
* 
* @date 18/02/2026
* 
*/

enum class ReorderMode {
    None,
    Cylindrical,
    Spherical
};

struct CylindricalCoord {
    double theta;
    double r;
    double z;
};

struct SphericalCoord {
    double alpha;
    double beta;
    double r;
};
/*
idea: cuando quiera almacenar las coords hago un struct extendedPoint con Point y sphe/cilCoord 
*/

template<typename Octree_t, typename Point_t>
class OctreeReordered
{
public:
    static void reorderLeaves(Octree_t& octree, std::vector<Point_t>& points,ReorderMode mode){
        /// TODO: paralelización con OpenMP, revisar que leafRange contiene los índices de los puntos dentro de la hoja del vector points
        if (mode == ReorderMode::None)
            return;
        size_t numLeaves = octree.getLeaves();
        for (size_t i=0; i < numLeaves; i++){
            std::pair<size_t, size_t> leafRange = octree.getLeafRange(i);
            Point leafCenter = octree.getLeafCenter(i);

            if (mode == ReorderMode::Cylindrical) {
                reorderPointsCylindrical(points, leafRange, leafCenter);
            } else if (mode == ReorderMode::Spherical) {
                reorderPointsSpherical(points, leafRange, leafCenter);
            }
        }
    }
private:
    static void reorderPointsCylindrical(std::vector<Point_t>& points, std::pair<size_t, size_t> leafRange, Point leafCenter) {
        std::sort(points.begin() + leafRange.first, points.begin() + leafRange.second, 
            [&leafCenter](const Point_t& a, const Point_t& b) {
                double thetaA = atan2(a.y - leafCenter.y, a.x - leafCenter.x);
                double thetaB = atan2(b.y - leafCenter.y, b.x - leafCenter.x);
                return thetaA < thetaB;
            }
        );
    }

    static void reorderPointsSpherical(std::vector<Point_t>& points, std::pair<size_t, size_t> leafRange, Point leafCenter) {
        // Implement spherical reordering logic here
    }
}