/**
 * A linear (map-based) implementation of the Octree using Morton codes for quick access with good spacial locality
 * 
 * Pablo Díaz Viñambres 22/10/24
 * 
 */


#pragma once

#include "Lpoint.hpp"
#include "octree_linear_node.hpp"
#include "Box.hpp"
#include <stack>
#include <map>
#include <bitset>
#include "libmorton/morton.h"

class LinearOctree {
private:
    static constexpr unsigned int MAX_POINTS        = 100;
	static constexpr float        MIN_OCTANT_RADIUS = 0.1;
    static constexpr unsigned int MAX_DEPTH         = 19;
	static constexpr size_t       DEFAULT_KNN       = 100;

    /* 
     * To construct the Morton encodings for a point (p1,p2,p3) on a depth d, we first find its anchor (~ octant center) 
     * by multiplying each coordinate by 2^l, then we convert it to integer via floor.
     * 
     * Once we have the anchor (x,y,z), we Morton encode it by interleaving its most significant 19 bits on each coordinate
     * 
     * We have a total of 64 bits, 7 go for the depth and 57 for the interleaved anchor coordinates (19 each)
     * 
     * Since the max depth is 19, there are 2 depth bits unused, but allocating them to the other part serves no purpose as we
     * at least need 3 for another depth level.
     * 
     * Format:
     * x18y18z18x17y17z17 ... x1y1z1x0y0z0d6...d0
     */
    std::unordered_map<morton_t, LinearOctreeNode*> nodes;

    Point center; // Center of the point cloud at depth level 0
    Vector radii; // Vector of radii of the point cloud at depth level 0

    // Insert points into the octree by computing their bins, and adds nodes to keep processing to the queue
    void insertPoints(std::vector<Lpoint*>& points, uint8_t depth, std::stack<LinearOctreeNode*>& subdivision_stack) {
        std::unordered_map<morton_t, std::vector<Lpoint*>> bins;
        for(int i = 0; i<points.size(); i++) {
            auto point = points[i];
            // Shift and scale coordinates into [0, 1]^3
            // x' = ((x - c_x) + r_x) / (2*r_x)
            float x_transf = ((point->getX() - center.getX())  + radii.getX()) / (2 * radii.getX());
            float y_transf = ((point->getY() - center.getY())  + radii.getY()) / (2 * radii.getY());
            float z_transf = ((point->getZ() - center.getZ())  + radii.getZ()) / (2 * radii.getZ());

            // Get the integer coordinates by multiplying by 2^depth and then taking floor
            coords_t x = (coords_t) (x_transf * (1 << depth));
            coords_t y = (coords_t) (y_transf * (1 << depth));
            coords_t z = (coords_t) (z_transf * (1 << depth));

            // Compute the morton code and push point into corresponding bin
            morton_t code = libmorton::morton3D_64_encode(x, y, z);

            // Shift by 7 to pack bit depth into it
            code = code << 7;
            code = code | depth;

            bins[code].push_back(point);
        } 
        
        // Add good nodes to the octree, reject and put into subdivision stack the others
        for (auto& [code, binPoints] : bins) {
            auto node = new LinearOctreeNode(binPoints, code, depth);
            // Insert into octree
            // TODO: also add min size condition
            if(binPoints.size() <= MAX_POINTS || depth == MAX_DEPTH ) {
                assert(nodes.count(code) == 0);
                // std::cout << "Inserted node " << std::bitset<64>(code) << " with " << binPoints.size() << " points and depth = " << (code & 0x7f) << "\n"; 
                nodes[code] = node;
            } else {
                // Push into queue for future subdivision until we have small amount of points
                subdivision_stack.push(node);
            }
        }
    }

public:
    LinearOctree();

    explicit LinearOctree(std::vector<Lpoint>& points) {
        center = mbb(points, radii);
        std::vector<Lpoint*> points_p;
        points_p.reserve(points.size());
        for (auto& point : points) {
            points_p.push_back(&point);
        }
        buildOctree(points_p);

        // int total = 0;
        // for (auto& [code, node] : nodes) {
        //     total += node->points.size();
        // }
        // std::cout << "Total inserted " << total << " out of " << points.size() << " points\n";
    }

	explicit LinearOctree(std::vector<Lpoint*>& points);

	LinearOctree(const Point& center, float radius);
	LinearOctree(Point center, float radius, std::vector<Lpoint*>& points);
	LinearOctree(Point center, float radius, std::vector<Lpoint>& points);

    void buildOctree(std::vector<Lpoint*>& points) {
        // Compute morton code for each point at depth 0        
        std::stack<LinearOctreeNode*> subdivision_stack;
        insertPoints(points, 0, subdivision_stack);

        // Process points that we still need to subdivide
        while(!subdivision_stack.empty()) {
            auto node = subdivision_stack.top();
            subdivision_stack.pop();

            // Reprocess the points in the node
            insertPoints(node->points, node->depth+1, subdivision_stack);

            // Destroy the old node
            delete node;
        }
    }
};