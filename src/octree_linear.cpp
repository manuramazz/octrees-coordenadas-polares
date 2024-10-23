#include "point.hpp"
#include "octree_linear.hpp"
#include "octree_linear_map.hpp"

LinearOctree::LinearOctree() : OctreeV2() {};
LinearOctree::LinearOctree(const Point& center, float radius) : OctreeV2(center, radius) {};


LinearOctree::LinearOctree(std::vector<Lpoint>& points) {

};

LinearOctree::LinearOctree(std::vector<Lpoint*>& points) {

};


LinearOctree::LinearOctree(Point center, float radius, std::vector<Lpoint*>& points): OctreeV2(center, radius) {
    
};

LinearOctree::LinearOctree(Point center, float radius, std::vector<Lpoint>& points): OctreeV2(center, radius) {
    
};

// Creates the new octants for the current node
void LinearOctree::createOctants()
{
	for (size_t i = 0; i < OCTANTS_PER_NODE; i++)
	{
		auto newCenter = center_;
		newCenter.setX(newCenter.getX() + radius_ * ((i & 4U) != 0U ? 0.5F : -0.5F));
		newCenter.setY(newCenter.getY() + radius_ * ((i & 2U) != 0U ? 0.5F : -0.5F));
		newCenter.setZ(newCenter.getZ() + radius_ * ((i & 1U) != 0U ? 0.5F : -0.5F));
        auto newOctant = LinearOctree(newCenter, 0.5F * radius_);
		addOctant(newOctant);
	}
}

const OctreeV2* LinearOctree::getOctant(int index) const {
    return map->getChild(code, index);
}

void LinearOctree::setOctant(int index, const OctreeV2& octant) {
    map->replaceChild(code, index, static_cast<const LinearOctree&>(octant));
}

void LinearOctree::setOctants(const std::vector<OctreeV2>& octants) {
    for (int i = 0; i < 8; i++) {
        if (i < octants.size()) {
            setOctant(i, octants[i]);
        }
    }
}

void LinearOctree::addOctant(const OctreeV2& octant) {
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
    setOctant(index, octant);
}

void LinearOctree::clearOctants() {
    for (int i = 0; i < 8; i++) {
        map->clearChild(code, i);
    }
}

const LinearOctree* create(const Point& center, float radius) {
	return new LinearOctree(center, radius);
}