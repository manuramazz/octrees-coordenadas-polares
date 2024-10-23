#pragma once
#include <unordered_map>
#include "octree_linear.hpp"

class LinearOctreeMap {
private:
    static constexpr int MAX_DEPTH = 64 / 3;
    std::unordered_map<uint64_t, LinearOctree> nodes;

    const LinearOctree* getNode(uint64_t locCode) const;
    void replaceNode(uint64_t locCode, const LinearOctree& node);
    void clearNode(uint64_t locCode);

public:
    LinearOctreeMap() = default;
    ~LinearOctreeMap() = default;

    const LinearOctree* getChild(uint64_t locCode, int index) const;
    const LinearOctree* getParent(uint64_t locCode) const;
    void replaceChild(uint64_t locCode, int index, const LinearOctree& node);
    void replaceParent(uint64_t locCode, const LinearOctree& node);
    void clearChild(uint64_t locCode, int index);

    static uint64_t getParentCode(int locCode);
    static uint64_t getChildCode(int locCode, int index);
    static size_t getDepth(uint64_t locCode);
};