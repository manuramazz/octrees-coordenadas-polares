#pragma once

#include <vector>
#include <random>
#include "Lpoint.hpp"

struct SearchSet {
    const size_t numSearches;
    std::vector<Point> searchPoints;
    std::vector<uint32_t> searchKNNLimits;
    constexpr static uint32_t MIN_KNN = 5;
    constexpr static uint32_t MAX_KNN = 100;
    std::mt19937 rng;

    template <PointType Point_t>
    SearchSet(size_t numSearches, const std::vector<Point_t>& points)
        : numSearches(numSearches) {

        rng.seed(42);
        searchPoints.resize(numSearches);
        searchKNNLimits.resize(numSearches);

        std::uniform_int_distribution<size_t> indexDist(0, points.size() - 1);
        std::uniform_int_distribution<size_t> knnDist(MIN_KNN, MAX_KNN);

        for (size_t i = 0; i < numSearches; ++i) {
            searchPoints[i] = points[indexDist(rng)];
            searchKNNLimits[i] = knnDist(rng);
        }
    }
};
