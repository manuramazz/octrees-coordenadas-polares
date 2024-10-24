#include "point.hpp"
#include "octree_linear.hpp"
#include "octree_linear_map.hpp"
#include <Box.hpp>

LinearOctree::LinearOctree() : OctreeV2() {};
LinearOctree::LinearOctree(const Point& center, float radius) : OctreeV2(center, radius) {};


LinearOctree::LinearOctree(std::vector<Lpoint>& points) {
	center_ = mbb(points, radius_);
	map = new LinearOctreeMap();
	buildOctree(points);
};

LinearOctree::LinearOctree(std::vector<Lpoint*>& points) {
	center_ = mbb(points, radius_);
	map = new LinearOctreeMap();
	buildOctree(points);
};

LinearOctree::LinearOctree(Point center, float radius, std::vector<Lpoint*>& points): OctreeV2(center, radius) {
	map = new LinearOctreeMap();
	buildOctree(points);
};

LinearOctree::LinearOctree(Point center, float radius, std::vector<Lpoint>& points): OctreeV2(center, radius) {
	map = new LinearOctreeMap();
	buildOctree(points);
};

// Creates the new octants for the current node
void LinearOctree::createOctants()
{
	for (size_t i = 0; i < OCTANTS_PER_NODE; i++)
	{
		auto newCenter = center_;
		std::cout << "creating octants" << std::endl;
		newCenter.setX(newCenter.getX() + radius_ * ((i & 4U) != 0U ? 0.5F : -0.5F));
		newCenter.setY(newCenter.getY() + radius_ * ((i & 2U) != 0U ? 0.5F : -0.5F));
		newCenter.setZ(newCenter.getZ() + radius_ * ((i & 1U) != 0U ? 0.5F : -0.5F));
        auto newOctant = LinearOctree(newCenter, 0.5F * radius_);
		newOctant.map = map;
		newOctant.code = LinearOctreeMap::getChildCode(code, i);
		auto index = getNextChildIndex();
		std::cout << "octants created" << std::endl;
        map->replaceChild(code, index, static_cast<const LinearOctree&>(newOctant));
		std::cout << "octants done" << std::endl;
	}
	octantsCreated = true;
}

const OctreeV2* LinearOctree::getOctant(int index) const {
    return map->getChild(code, index);
}

OctreeV2* LinearOctree::getOctant(int index) {
    return map->getChild(code, index);
}

void LinearOctree::clearOctants() {
    for (int i = 0; i < 8; i++) {
        map->clearChild(code, i);
    }
}

bool LinearOctree::maxDepthReached() const {
	return map->getDepth(code) > LinearOctreeMap::MAX_DEPTH;
}