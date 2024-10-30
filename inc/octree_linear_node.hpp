/**
 * A class representing a single node in a Linear Octree, with its points and associated Morton code
 */
#pragma once
#include "Lpoint.hpp"
using morton_t = uint_fast64_t;
using coords_t = uint_fast32_t;

class LinearOctreeNode {
    private:
        std::vector<Lpoint*> points;
        morton_t code;
        uint8_t depth;
    public:

        LinearOctreeNode(std::vector<Lpoint*> p, morton_t c, uint8_t d): points(p), code(c), depth(d) {};


        // Comparators based on Morton code
        bool operator==(const LinearOctreeNode& other) const {
            return this->code == other.code;
        }
        bool operator!=(const LinearOctreeNode& other) const {
            return !(*this == other);
        }
        bool operator<(const LinearOctreeNode& other) const {
            return this->code < other.code;
        }
        
        friend class LinearOctree;
};

