//
// Created by ruben.laso on 13/10/22.
//

#ifndef KERNEL2D_HPP
#define KERNEL2D_HPP

#include "KernelAbstract.hpp"
#include "PointEncoding/common.hpp"

class Kernel2D : public KernelAbstract
{
	public:
	Kernel2D(const Point& center, const double radius) : KernelAbstract(center, radius) {}
	Kernel2D(const Point& center, const Point& radii) : KernelAbstract(center, radii) {}

	[[nodiscard]] bool boxOverlap(const Point& center, const double radius) const override
	/**
 * @brief Checks if a given octant overlaps with the given kernel in 2 dimensions
 * @param octant
 * @return
 */
	{
		if (center.getX() + radius < boxMin().getX() || center.getY() + radius < boxMin().getY()) { return false; }

		if (center.getX() - radius > boxMax().getX() || center.getY() - radius > boxMax().getY()) { return false; }

		return true;
	}
	[[nodiscard]] bool boxOverlap(const Point& center, const Vector& radii) const override
	/**
 * @brief Checks if a given octant overlaps with the given kernel in 2 dimensions
 * @param octant
 * @return
 */
	{
		if (center.getX() + radii.getX() < boxMin().getX() || center.getY() + radii.getY() < boxMin().getY()) { return false; }

		if (center.getX() - radii.getX() > boxMax().getX() || center.getY() - radii.getY() > boxMax().getY()) { return false; }

		return true;
	}

	template <typename Encoder>
	[[nodiscard]] std::pair<typename Encoder::key_t, typename Encoder::key_t> encodeBounds(const Box& bbox) const {
		typename Encoder::coords_t x, y, z;
		typename Encoder::key_t encodedMin, encodedMax;
		Point realBoxMin = Point(boxMin().getX(), boxMin().getY(), bbox.minZ());
		Point realBoxMax = Point(boxMax().getX(), boxMax().getY(), bbox.maxZ());
		PointEncoding::getAnchorCoords<Encoder>(realBoxMin, bbox, x, y, z);
		encodedMin = Encoder::encode(x, y, z);
		PointEncoding::getAnchorCoords<Encoder>(realBoxMax, bbox, x, y, z);
		encodedMax = Encoder::encode(x, y, z);
		return std::make_pair(encodedMin, encodedMax);
	}
};

#endif /* end of include guard: KERNEL2D_HPP */
