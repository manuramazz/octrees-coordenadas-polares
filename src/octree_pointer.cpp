#include "octree_pointer.hpp"
#include "point.hpp"
#include "Box.hpp"

PointerOctree::PointerOctree() : OctreeV2() {};
PointerOctree::PointerOctree(const Point& center, float radius) : OctreeV2(center, radius) {};

PointerOctree::PointerOctree(std::vector<Lpoint>& points) {
	octants_.reserve(OCTANTS_PER_NODE);
	buildOctree(points);
};

PointerOctree::PointerOctree(std::vector<Lpoint*>& points) {
	center_ = mbb(points, radius_);
	octants_.reserve(OCTANTS_PER_NODE);
	buildOctree(points);
};

PointerOctree::PointerOctree(Point center, float radius, std::vector<Lpoint*>& points) : OctreeV2(center, radius) {
	octants_.reserve(OCTANTS_PER_NODE);
	buildOctree(points);
};

PointerOctree::PointerOctree(Point center, float radius, std::vector<Lpoint>& points): OctreeV2(center, radius) {
	octants_.reserve(OCTANTS_PER_NODE);
	buildOctree(points);
};

// Creates the new octants for the current node
void PointerOctree::createOctants()
{
	for (size_t i = 0; i < OCTANTS_PER_NODE; i++)
	{
		auto newCenter = center_;
		newCenter.setX(newCenter.getX() + radius_ * ((i & 4U) != 0U ? 0.5F : -0.5F));
		newCenter.setY(newCenter.getY() + radius_ * ((i & 2U) != 0U ? 0.5F : -0.5F));
		newCenter.setZ(newCenter.getZ() + radius_ * ((i & 1U) != 0U ? 0.5F : -0.5F));
        auto newOctant = PointerOctree(newCenter, 0.5F * radius_);
		addOctant(newOctant);
	}
}

const OctreeV2* PointerOctree::getOctant(int index) const {
    return &octants_[index];
}

void PointerOctree::setOctant(int index, const OctreeV2& octant) {
    octants_[index] = octant;
}

void PointerOctree::setOctants(const std::vector<OctreeV2>& octants) {
    octants_ = octants;
}

void PointerOctree::addOctant(const OctreeV2& octant) {
    octants_.emplace_back(octant);
}

void PointerOctree::clearOctants() {
    octants_.clear();
}

const PointerOctree* create(const Point& center, float radius) {
	return new PointerOctree(center, radius);
}