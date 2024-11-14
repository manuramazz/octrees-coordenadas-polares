#include "libmorton/morton.h"
#include "point.hpp"
#include "Box.hpp"

using morton_t = uint_fast64_t;
using coords_t = uint_fast32_t;

class MortonEncoder {
    public:
        static constexpr unsigned MAX_DEPTH = 21;
        static constexpr double EPS = 1.0f / (1 << MAX_DEPTH); // Unit of length of the encoded coordinates
        static constexpr morton_t LAST_CODE = 0x8000000000000000;
        static constexpr uint32_t UNUSED_BITS = 1;

        MortonEncoder(std::vector<Lpoint> &points): points(points) {
            Vector radii;
            Point center = mbb(points, radii);
            bbox = Box(center, radii);

            // Compute the physical half lengths for multiplying with the encoded coordinates
            halfLengths[0] = 0.5f * EPS * (bbox.maxX() - bbox.minX());
            halfLengths[1] = 0.5f * EPS * (bbox.maxY() - bbox.minY());
            halfLengths[2] = 0.5f * EPS * (bbox.maxZ() - bbox.minZ());

            for(int i = 0; i<=MAX_DEPTH; i++) {
                coords_t sideLength = (1u << (MAX_DEPTH - i));
                precomputedRadii[i] = Vector(
                    sideLength * halfLengths[0],
                    sideLength * halfLengths[1],
                    sideLength * halfLengths[2]
                );
            }
        }

        inline std::pair<Point, Vector> getCenterAndRadii(morton_t code, uint32_t level) {
            // Decode the points
            coords_t min_x, min_y, min_z;
            decodeMorton(code, min_x, min_y, min_z);

            // Find the physical center by multiplying the encoding with the halflength
            // to get to the low corner of the cell, and then adding the radii of the cell
            Point center = Point(
                bbox.minX() + min_x * halfLengths[0], 
                bbox.minY() + min_y * halfLengths[1], 
                bbox.minZ() + min_z * halfLengths[2]
            ) + precomputedRadii[level];
            
            return {center, precomputedRadii[level]};
        }

        std::vector<morton_t> sortPoints() {
            // We use a vector of pairs as an intermediate step, but this could also be done directly
            // TODO: optimize this heavily
            std::vector<std::pair<morton_t, Lpoint>> encoded_points;
            encoded_points.reserve(points.size());
            for(size_t i = 0; i < points.size(); i++) {
                encoded_points.emplace_back(encodeMortonPoint(points[i]), points[i]);
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



        static constexpr uint32_t countLeadingZeros(uint64_t x)
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

        static constexpr bool isPowerOf8(morton_t n) {
            morton_t lz = countLeadingZeros(n - 1) - UNUSED_BITS;
            return lz % 3 == 0 && !(n & (n - 1));
        }

        // Get the level in the octree of the given morton code
        static inline uint32_t getLevel(morton_t range) {
            assert(isPowerOf8(range));
            if(range == LAST_CODE)
                return 0U;
            return (countLeadingZeros(range - 1) - UNUSED_BITS) / 3;
        }

        // Get the sibling ID of the code at a given level
        static constexpr uint32_t getSiblingId(morton_t code, uint32_t level) {
            // Shift 3*(21-level) to get the 3 bits corresponding to the level
            return (code >> (3u * (MAX_DEPTH - level))) & 7u;
        }   

        // Get the maximum range allowed in a level of the tree, equal to 1 << 3*(maxdepth-treeLevel)
        // e.g. at level 0 the range is the entire 63 bit span, at level 10 the range is 11*3 bit span
        // at level 20 (last to minimum), the range will just be 8 between each node, i.e. the 8 siblings that
        // can be on max level 21 between two nodes at level 20
        static constexpr morton_t nodeRange(uint32_t treeLevel)
        {
            assert(treeLevel < MAX_DEPTH);
            uint32_t shifts = MAX_DEPTH - treeLevel;

            return 1ul << (3u * shifts);
        }

        
        static constexpr morton_t encodePlaceholderBit(morton_t code, int prefixLength) {
            int nShifts = 3 * MAX_DEPTH - prefixLength;
            morton_t ret = code >> nShifts;
            morton_t placeHolderMask = 1UL << prefixLength;

            return placeHolderMask | ret;
        }

        static constexpr uint32_t decodePrefixLength(morton_t code) {
            return 8 * sizeof(morton_t) - 1 - countLeadingZeros(code);
        }

        static constexpr morton_t decodePlaceholderBit(morton_t code) {
            int prefixLength        = decodePrefixLength(code);
            morton_t placeHolderMask = 1UL << prefixLength;
            morton_t ret             = code ^ placeHolderMask;

            return ret << (3 * MAX_DEPTH - prefixLength);
        }


        static constexpr int32_t commonPrefix(morton_t key1, morton_t key2) {
            return int32_t(countLeadingZeros(key1 ^ key2)) - UNUSED_BITS;
        }

        static constexpr unsigned octalDigit(morton_t code, uint32_t position) {
            return (code >> (3u * (MAX_DEPTH - position))) & 7u;
        }

        // Getters
        const Box& getPointsBbox() const { return bbox; }
        const Vector getPointsRadii() const { return bbox.radii(); }
        const Point getPointsCenter() const { return bbox.center(); }

        private:
            Box bbox = Box(Point(), Vector());
            std::vector<Lpoint> &points;
            Vector precomputedRadii[MAX_DEPTH + 1];
            float halfLengths[3];

            // This methods should not be called from the outside, as they use geometric information computed here
            // (radii and center) of the point cloud
            inline morton_t encodeMortonPoint(const Point& p) {
                // Utility method combining the two above
                coords_t x, y, z;
                getAnchorCoords(p, x, y, z);
                return libmorton::morton3D_64_encode(x, y, z);
            }
            inline void getAnchorCoords(const Point& p, coords_t &x, coords_t &y, coords_t &z) {
                // Put physical coords into the unit cube
                float x_transf = ((p.getX() - bbox.center().getX())  + bbox.radii().getX()) / (2 * bbox.radii().getX());
                float y_transf = ((p.getY() - bbox.center().getY())  + bbox.radii().getY()) / (2 * bbox.radii().getY());
                float z_transf = ((p.getZ() - bbox.center().getZ())  + bbox.radii().getZ()) / (2 * bbox.radii().getZ());

                // Edge case of coordinates falling exactly into 1, which would be problematic
                if(x_transf + EPS >= 1.0f) x_transf = 1.0f - EPS;
                if(y_transf + EPS >= 1.0f) y_transf = 1.0f - EPS;
                if(z_transf + EPS >= 1.0f) z_transf = 1.0f - EPS;
                
                // Scale to [0,2^L]^3 for morton encoding
                x = (coords_t) (x_transf * (1 << (MAX_DEPTH)));
                y = (coords_t) (y_transf * (1 << (MAX_DEPTH)));
                z = (coords_t) (z_transf * (1 << (MAX_DEPTH)));
            }
            
            static inline void decodeMorton(morton_t code, coords_t &x, coords_t &y, coords_t &z) {
                libmorton::morton3D_64_decode(code, x, y, z);
            }
};