/**
 * A pointer based octree implementation
 * 
 * Pablo Díaz Viñambres 22/10/24
 */

#pragma once

#include "octree_v2.hpp"
#include "Lpoint.hpp"
#include <vector>

class PointerOctree : public OctreeV2
{
private:
    std::vector<OctreeV2> octants_;

public:
    // Constructors
    PointerOctree();
    explicit PointerOctree(std::vector<Lpoint>& points);
    explicit PointerOctree(std::vector<Lpoint*>& points);
    PointerOctree(const Point& center, float radius);
    PointerOctree(Point center, float radius, std::vector<Lpoint*>& points);
    PointerOctree(Point center, float radius, std::vector<Lpoint>& points);

    [[nodiscard]] const OctreeV2* getOctant(int index) const override;
    void setOctant(int index, const OctreeV2& octant) override;
    void setOctants(const std::vector<OctreeV2>& octants) override;
    
    void addOctant(const OctreeV2& octant) override;

    void createOctants() override;
    void clearOctants() override;

};