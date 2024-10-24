/**
 * A linear (map-based) implementation of the Octree using Morton codes for quick access with good spacial locality
 * 
 * Pablo Díaz Viñambres 22/10/24
 * 
 * Some implementation details from https://geidav.wordpress.com/2014/08/18/advanced-octrees-2-node-representations/
 */


#pragma once

#include "octree_v2.hpp"
#include "Lpoint.hpp"

class LinearOctreeMap;

class LinearOctree : public OctreeV2 {
private:
    LinearOctreeMap* map;
    uint64_t code;
    uint32_t childMask;

    int getNextChildIndex() {
        int index;
        #if defined(__GNUC__)
        index = childMask ? 32 - __builtin_clz(childMask) : 0;
        #else
        if (childMask == 0) {
            index = 0;
        } else {
            index = 0;
            uint32_t mask = childMask;
            while (mask >>= 1) {
                ++index;
            }
        }
        #endif
        return index;
    };

public:
    LinearOctree();
    explicit LinearOctree(std::vector<Lpoint>& points);
    explicit LinearOctree(std::vector<Lpoint*>& points);
    LinearOctree(const Point& center, float radius);
    LinearOctree(Point center, float radius, std::vector<Lpoint*>& points);
    LinearOctree(Point center, float radius, std::vector<Lpoint>& points);

    [[nodiscard]] const OctreeV2* getOctant(int index) const override;
    [[nodiscard]] OctreeV2* getOctant(int index) override;
    
    void createOctants() override;
    void clearOctants() override;

    bool maxDepthReached() const override;

    // Friend declaration to allow LinearOctreeMap access
    friend class LinearOctreeMap;
};