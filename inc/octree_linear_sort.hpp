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
     * Another (more correct) linear octree implementation based on the cornerstone octree project:
     * https://github.com/sekelle/cornerstone-octree/tree/master
     * 
     * This linear octree is built by storing offsets to the positions of an array of points sorted by their morton codes. 
     * For each leaf of the octree, there is an element in this array listthat points to the index of the first point in that leaf.
     * Since the points are sorted, the next element on the array - 1 contains the index of the last point in that leaf.
     * 
     * Refer to the cornerstone octree paper for documentation about the data structure 
     * https://arxiv.org/pdf/2307.06345
     */

    // Number of leaves and internal nodes in the octree
    uint32_t nLeaf, nInternal;

    // Total number of nodes in the octree
    uint32_t nTotal;

    // Leaves of the octree in cornerstone format and counts of particles in each leaf
    std::vector<morton_t> leaves; 
    std::vector<size_t> counts;

    // SFC key and level of each node (Waren-Salmon placeholder bit format)
    std::vector<morton_t> prefixes;

    // Index of the first child of each node (if 0 we have a leaf)
    std::vector<uint32_t> offsets;

    // Parent index of every group of 8 sibling nodes
    std::vector<uint32_t> parents;

    // First node index of every tree level (L+2 elements where L is MAX_DEPTH)
    std::vector<uint32_t> levelRange = std::vector<uint32_t>(MortonEncoder::MAX_DEPTH + 2);

    // Maps between the level-key sorted layout and the intermedaite binary layout
    std::vector<uint32_t> internalToLeaf;
    std::vector<uint32_t> leafToInternal;

    // The sorted vector of codes of the points
    std::vector<morton_t> &codes;

    // Comput the array of rebalancing decisions (g1)
    bool rebalanceDecision(std::vector<uint32_t> &nodeOps) {
        bool converged = true;
        for(int i = 0; i<leaves.size()-1; i++) {
            nodeOps[i] = calculateNodeOp(i);
            if(nodeOps[i] != 1) converged = false;
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
    inline std::pair<int32_t, uint32_t> siblingAndLevel(uint32_t index) {
        morton_t node = leaves[index];
        morton_t range = leaves[index+1] - node;
        uint32_t level = MortonEncoder::getLevel(range);
        if(level == 0) {
            return {-1, level};
        }

        uint32_t siblingId = MortonEncoder::getSiblingId(node, level);

        // Checks if all siblings are on the tree, to do this, checks if the difference between the two parent nodes corresponding
        // to the code parent and the next parent is the range spanned by two consecutive codes at that level
        bool siblingsOnTree = leaves[index - siblingId + 8] == (leaves[index - siblingId] + MortonEncoder::nodeRange(level - 1));
        if(!siblingsOnTree) siblingId = -1;

        return {siblingId, level};
    }

    
    static void printMortonCode(morton_t code) {
        // Print the bits in groups of 3 to represent each level
        std::cout << std::bitset<1>(code >> 63) << " ";
        for (int i = 62; i >= 0; i -= 3) {
            std::cout << std::bitset<3>((code >> (i - 2)) & 0b111) << " ";
        }
        std::cout << std::endl;
    }

    // Build the new tree using the rebalance decision array
    void rebalanceTree(std::vector<morton_t> &newTree, std::vector<uint32_t> &nodeOps) {
        uint32_t n = leaves.size() - 1;

        // g2, exclusive scan
        exclusiveScan(nodeOps.data(), n+1);

        newTree.resize(nodeOps[n] + 1);
        newTree.back() = leaves.back();
        for (uint32_t i = 0; i < n; ++i) {
            processNode(i, nodeOps, newTree);
        }
    }

    // Construct new octree value for the given index
    void processNode(uint32_t index, std::vector<uint32_t> &nodeOps, std::vector<morton_t> &newTree) {
        morton_t node = leaves[index];
        morton_t range = leaves[index+1] - node;

        uint32_t level = MortonEncoder::getLevel(range);

        uint32_t opCode       = nodeOps[index + 1] - nodeOps[index]; // The original value of the opCode (before exclusive scan)
        uint32_t newNodeIndex = nodeOps[index]; // The new position to put the node into (nodeOps value after exclusive scan)

        if(opCode == 1) {
            // do nothing, just copy node into new position
            newTree[newNodeIndex] = node;
            // assert(MortonEncoder::isPowerOf8(newTree[newNodeIndex + 1] - newTree[newNodeIndex]));
        } else if(opCode == 8) {
            // Split the node into 8
            for(int sibling = 0; sibling < OCTANTS_PER_NODE; sibling++) {
                newTree[newNodeIndex + sibling] = node + sibling * MortonEncoder::nodeRange(level + 1);
            }
            // assert(MortonEncoder::isPowerOf8(newTree[newNodeIndex + 8] - newTree[newNodeIndex + 7]));
        } else {
            // TODO: higher order splits
        }
    }

    // Count number of particles in each octree node
    void computeNodeCounts() {
        uint32_t n = leaves.size() - 1;
        uint32_t codes_size = codes.size();
        uint32_t firstNode = 0;
        uint32_t lastNode = n;

        if(codes.size() > 0) {
            firstNode = std::upper_bound(leaves.begin(), leaves.end(), codes[0]) - leaves.begin() - 1;
            lastNode = std::upper_bound(leaves.begin(), leaves.end(), codes[codes_size-1]) - leaves.begin();
            assert(firstNode <= lastNode);
        } else {
            firstNode = n, lastNode = n;
        }

        // Fill non-populated parts of the octree with zeros
        for(uint32_t i = 0; i<firstNode; i++)
            counts[i] = 0;
        for(uint32_t i = lastNode; i<lastNode; i++)
            counts[i] = 0;

        // TODO: count guessing, parallelizing
        size_t nNonZeroNodes = lastNode - firstNode;
        exclusiveScan(counts.data() + firstNode, nNonZeroNodes);

        for(uint32_t i = 0; i<nNonZeroNodes; i++) {
            counts[i + firstNode] = calculateNodeCount(leaves[i+firstNode], leaves[i+firstNode+1]);
        }
    }
    unsigned calculateNodeCount(morton_t keyStart, morton_t keyEnd) {
        auto rangeStart = std::lower_bound(codes.begin(), codes.end(), keyStart);
        auto rangeEnd   = std::lower_bound(codes.begin(), codes.end(), keyEnd);
        size_t count    = rangeEnd - rangeStart;
        // TODO: should we use maxCount??
        return count;
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

    constexpr uint32_t binaryKeyWeight(morton_t key, unsigned level)
    {
        uint32_t ret = 0;
        for (uint32_t l = 1; l <= level + 1; ++l)
        {
            uint32_t digit = MortonEncoder::octalDigit(key, l);
            ret += digitWeight(digit);
        }
        return ret;
    }

    constexpr int32_t digitWeight(uint32_t digit) {
        int32_t fourGeqMask = -int32_t(digit >= 4);
        return ((7 - digit) & fourGeqMask) - (digit & ~fourGeqMask);
    }

    void createUnsortedLayout() {
        for(int i = 0; i<nLeaf; i++) {
            morton_t key = leaves[i];
            uint32_t level = MortonEncoder::getLevel(leaves[i+1] - key);
            prefixes[i + nInternal] = MortonEncoder::encodePlaceholderBit(key, 3*level);
            internalToLeaf[i + nInternal] = i + nInternal;

            uint32_t prefixLength = MortonEncoder::commonPrefix(key, leaves[i+1]);
            if(prefixLength % 3 == 0 && i < nLeaf - 1) {
                uint32_t octIndex = (i + binaryKeyWeight(key, prefixLength / 3)) / 7;
                prefixes[octIndex] = MortonEncoder::encodePlaceholderBit(key, prefixLength);
                internalToLeaf[octIndex] = octIndex;
            }
        }
    }

    // Determine octree subdivision level boundaries
    void getLevelRange() {
        for(uint32_t level = 0; level <= MortonEncoder::MAX_DEPTH; level++) {
            auto it = std::lower_bound(prefixes.begin(), prefixes.end(), MortonEncoder::encodePlaceholderBit(0, 3 * level));
            levelRange[level] = std::distance(prefixes.begin(), it);
        }
        levelRange[MortonEncoder::MAX_DEPTH + 1] = nTotal;
    }

    // Extract parent/child relationships from binary tree and translate to sorted order
    void linkTree() {
        for(int i = 0; i<nInternal; i++) {
            uint32_t idxA = leafToInternal[i];
            morton_t prefix = prefixes[idxA];
            morton_t nodeKey = MortonEncoder::decodePlaceholderBit(prefix);
            unsigned prefixLength = MortonEncoder::decodePrefixLength(prefix);
            unsigned level = prefixLength / 3;
            assert(level < MortonEncoder::MAX_DEPTH);

            morton_t childPrefix = MortonEncoder::encodePlaceholderBit(nodeKey, prefixLength + 3);

            uint32_t leafSearchStart = levelRange[level + 1];
            uint32_t leafSearchEnd   = levelRange[level + 2];
            uint32_t childIdx = std::distance(prefixes.begin(), 
                std::lower_bound(prefixes.begin() + leafSearchStart, prefixes.begin() + leafSearchEnd, childPrefix));

            if (childIdx != leafSearchEnd && childPrefix == prefixes[childIdx]) {
                offsets[idxA] = childIdx;
                // We only store the parent once for every group of 8 siblings.
                // This works as long as each node always has 8 siblings.
                // Subtract one because the root has no siblings.
                parents[(childIdx - 1) / 8] = idxA;
            }
        } 
    }

public:
    LinearOctreeSort() = default;
    
    explicit LinearOctreeSort(std::vector<morton_t> &codes): codes(codes) {
        buildOctreeLeaves();

        resize();

        buildOctreeInternal();
    }

    void buildOctreeLeaves() {
        // Builds the octree sequentially using the cornerstone algorithm

        // We start with 0, 7777...777 (in octal)
        leaves = {0, MortonEncoder::LAST_CODE};
        counts = {(uint32_t) codes.size()};

        while(!updateOctreeLeaves())
            ;
        

        // Compute the final sizes of the octree
        nLeaf = leaves.size() - 1; // TODO: shouldnt this be -1?
        nInternal = (nLeaf - 1) / 7;
        nTotal = nLeaf + nInternal;
    }

    bool updateOctreeLeaves() {
        std::vector<uint32_t> nodeOps(leaves.size());
        bool converged = rebalanceDecision(nodeOps);

        std::vector<morton_t> newTree;
        rebalanceTree(newTree, nodeOps);
        counts.resize(newTree.size()-1);
        swap(leaves, newTree);

        computeNodeCounts();
        return converged;
    }

    void resize() {
        // Resize the other fields
        prefixes.resize(nTotal);
        offsets.resize(nTotal+1);
        parents.resize((nTotal-1) / 8);
        internalToLeaf.resize(nTotal);
        leafToInternal.resize(nTotal);
    }

    void buildOctreeInternal() {
        createUnsortedLayout();
        // Sort by key where the keys are the prefixes and the values to sort internalToLeaf
        std::vector<std::pair<morton_t, uint32_t>> prefixes_internalToLeaf(nTotal);
        for(int i = 0; i<nTotal; i++) {
            prefixes_internalToLeaf[i] = {prefixes[i], internalToLeaf[i]};
        }
        std::stable_sort(prefixes_internalToLeaf.begin(), prefixes_internalToLeaf.end(), [](const auto &t1, const auto &t2) {
            return t1.first < t2.first;
        });

        // Compute the reverse mapping leafToInternal
        for (uint32_t i = 0; i < nTotal; ++i) {
            leafToInternal[internalToLeaf[i]] = i;
        }

        // Offset by the number of internal nodes
        for (uint32_t i = 0; i < nTotal; ++i) {
            internalToLeaf[i] -= nInternal;
        }

        // Find the LO array
        getLevelRange();

        // Clear child offsets
        std::fill(offsets.begin(), offsets.end(), 0);

        // Compute the links
        linkTree();
    }

    void printArray(std::vector<uint32_t> &arr) {
        for(int i = 0; i<arr.size();i++) {
            std::cout << i << " -> " << arr[i] << std::endl;
        }
    }
    void printArrayMortonCodes(std::vector<morton_t> &arr){
        for(int i = 0; i<arr.size();i++) {
            std::cout << i << " -> "; printMortonCode(arr[i]);
        }
    }
};