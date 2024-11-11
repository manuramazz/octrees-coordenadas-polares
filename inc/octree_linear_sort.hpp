/**
 * A linear (map-based) implementation of the Octree using Morton codes for quick access with good spacial locality
 * 
 * Pablo Díaz Viñambres 22/10/24
 * 
 */


#pragma once

#include "Lpoint.hpp"
#include "Box.hpp"
#include <stack>
#include <bitset>
#include <unordered_map>
#include <unordered_set>
#include <filesystem>
#include "libmorton/morton.h"
#include "NeighborKernels/KernelFactory.hpp"
#include "TimeWatcher.hpp"
#include "morton_encoder.hpp"

using morton_t = uint_fast64_t;
using coords_t = uint_fast32_t;

class LinearOctreeSort {
private:
    static constexpr unsigned int MAX_POINTS        = 100;
	static constexpr float        MIN_OCTANT_RADIUS = 0.1;
	static constexpr size_t       DEFAULT_KNN       = 100;
	static constexpr short        OCTANTS_PER_NODE  = 8;

    /**
     * Another (more correct) linear octree implementation based on the cornerstone project
     * https://github.com/sekelle/cornerstone-octree/tree/master
     * 
     * This linear octree is built by storing offsets to the positions of an array of points sorted by their morton codes. 
     * For each leaf of the octree, there is an element in this array that points to the index of the first point in that leaf.
     * Since the points are sorted, the next element on the array - 1 contains the index of the last point in that leaf.
     * 
     * Refer to the cornerstone octree paper for documentation about the data structure 
     * https://arxiv.org/pdf/2307.06345
     */
    std::vector<morton_t> octree; 
    std::vector<size_t> counts;

    // The sorted vector of codes of the points
    std::vector<morton_t> &codes;

    uint32_t pointsSize;

    // Comput the array of rebalancing decisions (g1)
    bool rebalanceDecision(std::vector<uint32_t> &nodeOps) {
        bool converged = true;
        for(int i = 0; i<nodeOps.size(); i++) {
            uint32_t decision = calculateNodeOp(i);
            if(!decision != 1) converged = false;
        }
        return converged;
    } 

    // Calculate the operation to be done in this node
    uint32_t calculateNodeOp(uint32_t index) {
        auto [sibling, level] = siblingAndLevel(index);

        if(sibling > 0) {
            // We have 8 siblings next to each other, could merge this node if the count of all siblings is less MAX_COUNT
            uint32_t parentIndex = index - sibling;
            // Should not be bigger than 2^32
            size_t parentCount =    counts[parentIndex]   + counts[parentIndex+1]+ 
                                    counts[parentIndex+2] + counts[parentIndex+3]+ 
                                    counts[parentIndex+4] + counts[parentIndex+5]+
                                    counts[parentIndex+6] + counts[parentIndex+7];
            if(parentCount <= MAX_POINTS)
                return 0; // merge
        }
        
        uint32_t nodeCount = counts[index];
        // Decide if we split or not
        // TODO: higher order splits
        // if (nodeCount] > MAX_POINTS * 512 && level + 3 < MortonEncoder::MAX_DEPTH) { return 4096; } // split into 4 layers
        // if (nodeCount > MAX_POINTS * 64 && level + 2 < MortonEncoder::MAX_DEPTH) { return 512; }   // split into 3 layers
        // if (nodeCount > MAX_POINTS * 8 && level + 1 < MortonEncoder::MAX_DEPTH) { return 64; }     // split into 2 layers
        if (nodeCount > MAX_POINTS && level < MortonEncoder::MAX_DEPTH) { return 8; } // split into 1 layer
        
        return 1; // dont do nothing
    }

    // Get the sibling ID and level of the node in the octree
    inline std::pair<uint32_t, uint32_t> siblingAndLevel(uint32_t index) {
        morton_t node = octree[index];
        morton_t range = octree[index+1] - node;

        uint32_t level = MortonEncoder::getLevel(range);
        if(level == 0) {
            return {-1, level};
        }

        uint32_t siblingId = MortonEncoder::getSiblingId(node, level);
        // Checks if all siblings are on the tree, to do this, checks if the difference between the two parent nodes corresponding
        // to the code parent and the next parent is the range spanned by two consecutive codes at that level
        bool siblingsOnTree = (octree[index - siblingId + 8] - octree[index - siblingId]) == MortonEncoder::nodeRange(level - 1);
        if(!siblingsOnTree) siblingId = -1;

        return {siblingId, level};
    }

    // Build the new tree using the rebalance decision array
    void rebalanceTree(std::vector<morton_t> &newTree, std::vector<uint32_t> &nodeOps) {
        uint32_t n = octree.size() - 1;

        // g2, exclusive scan
        exclusiveScan(nodeOps.data(), n+1);

        newTree.resize(nodeOps[n] + 1);
        for (uint32_t i = 0; i < n; ++i) {
            processNode(i, nodeOps, newTree);
        }

        newTree.back() = octree.back();
    }

    // Construct new octree value for the given index
    void processNode(uint32_t index, std::vector<uint32_t> &nodeOps, std::vector<morton_t> &newTree) {
        morton_t node = octree[index];
        morton_t range = octree[index+1] - node;

        uint32_t level = MortonEncoder::getLevel(range);

        uint32_t opCode       = nodeOps[index + 1] - nodeOps[index]; // The original value of the opCode (before exclusive scan)
        uint32_t newNodeIndex = nodeOps[index]; // The new position to put the node into (nodeOps value after exclusive scan)

        if(opCode == 1) {
            // do nothing, just copy node into new position
            newTree[index] = node;
        } else if(opCode == 8) {
            // Split the node into 8
            for(int sibling = 0; sibling < OCTANTS_PER_NODE; sibling++) {
                newTree[newNodeIndex + sibling] = node + sibling * MortonEncoder::nodeRange(level + 1);
            }
        } else {
            // TODO: higher order splits
        }
    }

    void computeNodeCounts() {
        
    }

    template<class T>
    void exclusiveScan(T* out, size_t numElements) {
        exclusiveScanSerialInplace(out, numElements, T(0));
    }

    template<class T>
    T exclusiveScanSerialInplace(T* out, size_t num_elements, T init)
    {
        T a = init;
        T b = init;
        for (size_t i = 0; i < num_elements; ++i)
        {
            a += out[i];
            out[i] = b;
            b      = a;
        }
        return b;
    }

public:
    LinearOctreeSort() = default;
    
    explicit LinearOctreeSort(std::vector<morton_t> &codes): codes(codes) {
        buildOctree();
    }

    void buildOctree() {
        // Builds the octree sequentially using the cornerstone algorithm

        // We start with 0, 7777...777 (in octal)
        octree = {0, MortonEncoder::LAST_CODE};
        counts = {(uint32_t) codes.size()};
        
        while(!updateOctree())
            ;
    }

    bool updateOctree() {
        std::vector<uint32_t> nodeOps(codes.size());
        bool converged = rebalanceDecision(nodeOps);

        std::vector<morton_t> newTree;

        rebalanceTree(newTree, nodeOps);
        swap(octree, newTree);

        counts.resize(newTree.size()-1);
        computeNodeCounts();
    }
};