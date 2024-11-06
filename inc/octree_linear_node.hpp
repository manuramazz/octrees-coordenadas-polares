/**
 * A simple container representing a single node in a Linear Octree, with its points and associated Morton code
 * 
 */
#pragma once
#include "Lpoint.hpp"

// Here we also define a couple shorthands for the integers used in morton encoding
using morton_t = uint_fast64_t;
using coords_t = uint_fast32_t;

class LinearOctreeNode {
    private:
        std::vector<Lpoint*> points;
        morton_t code;

    public:
        LinearOctreeNode(std::vector<Lpoint*> p, morton_t c, uint8_t d): points(p), code(c) {};

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
        
        inline bool isLeaf() {
            return !points.empty();
        }

        // Since this is just a container, we can declare LinearOctree as a friend class
        friend class LinearOctree;
};

