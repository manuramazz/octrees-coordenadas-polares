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
    static void reorderLeaves(Octree_t& octree,
                              std::vector<Point_t>& points,
                              std::vector<typename PointEncoding::key_t>* codes = nullptr,
                              std::optional<std::vector<PointMetadata>>* meta_opt = nullptr,
                              ReorderMode mode = ReorderMode::None){
        /// TODO: paralelización con OpenMP, revisar que leafRange contiene los índices de los puntos dentro de la hoja del vector points
        if (mode == ReorderMode::None)
            return;
        size_t numLeaves = octree.getLeaves();
        #pragma omp parallel for schedule(dynamic)
        for (size_t i=0; i < numLeaves; i++){
            std::pair<size_t, size_t> leafRange = octree.getLeafRange(i);
            if (leafRange.first >= leafRange.second) continue;
            Point leafCenter = octree.getLeafCenter(i);
            //Reordenación consistente con codes y metadata (si existen)
            if (codes && *codes) {
                // Construir vector de índices local
                std::vector<size_t> idxs(leafRange.second - leafRange.first);
                for (size_t i = 0; i < idxs.size(); ++i) idxs[i] = leafRange.first + i;

                auto comparator = [&](size_t ia, size_t ib) {
                    const auto &a = points[ia];
                    const auto &b = points[ib];
                    if (mode == ReorderMode::Cylindrical) {
                        double thetaA = atan2(a.getY() - leafCenter.getY(), a.getX() - leafCenter.getX());
                        double thetaB = atan2(b.getY() - leafCenter.getY(), b.getX() - leafCenter.getX());
                        return thetaA < thetaB;
                    } else { // spherical
                        double alphaA = atan2(a.getY() - leafCenter.getY(), a.getX() - leafCenter.getX());
                        double betaA = atan2(a.getZ() - leafCenter.getZ(), sqrt(pow(a.getX() - leafCenter.getX(), 2) + pow(a.getY() - leafCenter.getY(), 2)));
                        double alphaB = atan2(b.getY() - leafCenter.getY(), b.getX() - leafCenter.getX());
                        double betaB = atan2(b.getZ() - leafCenter.getZ(), sqrt(pow(b.getX() - leafCenter.getX(), 2) + pow(b.getY() - leafCenter.getY(), 2)));
                        return (alphaA == alphaB) ? (betaA < betaB) : (alphaA < alphaB);
                    }
                };

                std::stable_sort(idxs.begin(), idxs.end(), comparator);

                // A partir de los indices ordenados, construir vectores temporales de puntos, codes y meta
                // para luego copiar de vuelta al vector original en la posición correcta
                std::vector<Point_t> points_buf;
                std::vector<typename PointEncoding::key_t> codes_buf;
                std::optional<std::vector<PointMetadata>> meta_buf;
                points_buf.reserve(idxs.size());
                if (codes) codes_buf.reserve(idxs.size());
                if (meta_opt && *meta_opt) meta_buf.emplace().reserve(idxs.size());

                for (auto id : idxs) {
                    points_buf.push_back(points[id]);
                    if (codes) codes_buf.push_back((*codes)[id]);
                    if (meta_opt && *meta_opt) meta_buf->push_back((*meta_opt)->at(id));
                }

                // Copy back
                for (size_t i = 0; i < idxs.size(); ++i) {
                    points[leafRange.first + i] = std::move(points_buf[i]);
                    if (codes) (*codes)[leafRange.first + i] = std::move(codes_buf[i]);
                    if (meta_opt && *meta_opt) (*meta_opt)->at(leafRange.first + i) = std::move(meta_buf->at(i));
                }
            } else {
                if (mode == ReorderMode::Cylindrical) {
                    reorderPointsCylindrical(points, leafRange, leafCenter);
                } else if (mode == ReorderMode::Spherical) {
                    reorderPointsSpherical(points, leafRange, leafCenter);
                }

            }
        }
        
    }

private:
    static void reorderPointsCylindrical(std::vector<Point_t>& points, std::pair<size_t, size_t> leafRange, Point leafCenter) {
        std::stable_sort(points.begin() + leafRange.first, points.begin() + leafRange.second, 
            [&leafCenter](const Point_t& a, const Point_t& b) {
                double thetaA = atan2(a.y - leafCenter.getY(), a.x - leafCenter.getX());
                double thetaB = atan2(b.y - leafCenter.getY(), b.x - leafCenter.getX());
                return thetaA < thetaB;
            }
        );
    }

    static void reorderPointsSpherical(std::vector<Point_t>& points, std::pair<size_t, size_t> leafRange, Point leafCenter) {
         std::stable_sort(points.begin() + leafRange.first, points.begin() + leafRange.second, 
            [&leafCenter](const Point_t& a, const Point_t& b) {
                double alphaA = atan2(a.y - leafCenter.getY(), a.x - leafCenter.getX());
                double betaA = atan2(a.z - leafCenter.getZ(), sqrt(pow(a.x - leafCenter.getX(), 2) + pow(a.y - leafCenter.getY(), 2)));
                double alphaB = atan2(b.y - leafCenter.getY(), b.x - leafCenter.getX());
                double betaB = atan2(b.z - leafCenter.getZ(), sqrt(pow(b.x - leafCenter.getX(), 2) + pow(b.y - leafCenter.getY(), 2)));
                return (alphaA == alphaB) ? (betaA < betaB) : (alphaA < alphaB);
            }
        );
    }
}