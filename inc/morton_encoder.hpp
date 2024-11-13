#include "libmorton/morton.h"
#include "point.hpp"
#include "Box.hpp"

using morton_t = uint_fast64_t;
using coords_t = uint_fast32_t;

namespace MortonEncoder {
    // 21 octal 7s (i.e. all bits set to 1 except MSB)
    constexpr unsigned MAX_DEPTH = 21;
    constexpr double EPS = 1.0f / (1 << MAX_DEPTH);
    constexpr morton_t LAST_CODE = 0x8000000000000000;
    constexpr uint32_t UNUSED_BITS = 1;

    inline void getAnchorCoords(const Point &center, const Vector &radii, const Point& p, coords_t &x, coords_t &y, coords_t &z) {
        float x_transf = ((p.getX() - center.getX())  + radii.getX()) / (2 * radii.getX());
        float y_transf = ((p.getY() - center.getY())  + radii.getY()) / (2 * radii.getY());
        float z_transf = ((p.getZ() - center.getZ())  + radii.getZ()) / (2 * radii.getZ());
        if(x_transf + EPS >= 1.0f) x_transf = 1.0f - EPS;
        if(y_transf + EPS >= 1.0f) y_transf = 1.0f - EPS;
        if(z_transf + EPS >= 1.0f) z_transf = 1.0f - EPS;


        x = (coords_t) (x_transf * (1 << (MAX_DEPTH)));
        y = (coords_t) (y_transf * (1 << (MAX_DEPTH)));
        z = (coords_t) (z_transf * (1 << (MAX_DEPTH)));
    }

    inline morton_t encodeMortonPoint(const Point &center, const Vector &radii, const Point& p) {
        // Utility method combining the two above
        coords_t x, y, z;
        getAnchorCoords(center, radii, p, x, y, z);
        return libmorton::morton3D_64_encode(x, y, z);
    }

    std::vector<morton_t> sortPoints(std::vector<Lpoint> &points) {
        // We use a vector of pairs as an intermediate step, but this could also be done directly
        // TODO: optimize this heavily
        Vector radii;
        Point center = mbb(points, radii);
        std::vector<std::pair<morton_t, Lpoint>> encoded_points;
        encoded_points.reserve(points.size());
        for(size_t i = 0; i < points.size(); i++) {
            encoded_points.emplace_back(encodeMortonPoint(center, radii, points[i]), points[i]);
        }

        std::sort(encoded_points.begin(), encoded_points.end(),
            [](const auto& a, const auto& b) {
                return a.first < b.first;  // Compare only the morton codes
        });
        
        // Copy back sorted codes and points
        std::vector<morton_t> codes(points.size());
        for(size_t i = 0; i < points.size(); i++) {
            codes[i] = encoded_points[i].first;
            points[i] = encoded_points[i].second;
        }
        return codes;
    }

    inline void decodeMorton(morton_t code, coords_t &x, coords_t &y, coords_t &z) {
        libmorton::morton3D_64_decode(code, x, y, z);
    }

    constexpr uint32_t countLeadingZeros(uint64_t x)
    {
        #if defined(__GNUC__) || defined(__clang__)
            if (x == 0) return 8 * sizeof(uint64_t);
            return __builtin_clzll(x);
        #else
            uint32_t depth = 0;
            for (; x != 1; x >>= 3, depth++);
            return depth;
        #endif
    }

    constexpr bool isPowerOf8(morton_t n) {
        morton_t lz = countLeadingZeros(n - 1) - UNUSED_BITS;
        return lz % 3 == 0 && !(n & (n - 1));
    }

    // Get the level in the octree of the given morton code
    inline uint32_t getLevel(morton_t range) {
        assert(isPowerOf8(range));
        if(range == LAST_CODE)
            return 0U;
        return (countLeadingZeros(range - 1) - UNUSED_BITS) / 3;
    }

    // Get the sibling ID of the code at a given level
    constexpr uint32_t getSiblingId(morton_t code, uint32_t level) {
        // Shift 3*(21-level) to get the 3 bits corresponding to the level
        return (code >> (3u * (MAX_DEPTH - level))) & 7u;
    }   

    // Get the maximum range allowed in a level of the tree, equal to 1 << 3*(maxdepth-treeLevel)
    // e.g. at level 0 the range is the entire 63 bit span, at level 10 the range is 11*3 bit span
    // at level 20 (last to minimum), the range will just be 8 between each node, i.e. the 8 siblings that
    // can be on max level 21 between two nodes at level 20
    constexpr morton_t nodeRange(uint32_t treeLevel)
    {
        assert(treeLevel < MAX_DEPTH);
        uint32_t shifts = MAX_DEPTH - treeLevel;

        return 1ul << (3u * shifts);
    }


};