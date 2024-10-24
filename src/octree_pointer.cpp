#include "octree_pointer.hpp"
#include "point.hpp"
#include "Box.hpp"

PointerOctree::PointerOctree() : OctreeV2() {};
PointerOctree::PointerOctree(const Point& center, float radius) : OctreeV2(center, radius) {};

PointerOctree::PointerOctree(std::vector<Lpoint>& points) {
	center_ = mbb(points, radius_);
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
		octants_.emplace_back(newCenter, 0.5F * radius_);
	}
	octantsCreated = true;
}

const OctreeV2* PointerOctree::getOctant(int index) const {
	return &octants_[index];
}

OctreeV2* PointerOctree::getOctant(int index) {
	return &octants_[index];
}

void PointerOctree::clearOctants() {
    octants_.clear();
}