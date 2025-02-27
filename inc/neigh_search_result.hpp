#pragma once
#include <vector>
#include <iterator>

template <typename Point_t>
class NeighSearchResult {
    public:
        std::vector<uint32_t> octantIndexes;
        std::vector<Point_t*> extraPoints;
        std::vector<Point_t> emptyPointCloud;
        std::vector<std::pair<size_t, size_t>> emptyInternalLayoutRanges;
        std::vector<Point_t>& pointCloud = emptyPointCloud;
        std::vector<std::pair<size_t, size_t>>& internalLayoutRanges = emptyInternalLayoutRanges;
        size_t numberOfPoints = 0;

        NeighSearchResult() = default;

        NeighSearchResult(std::vector<Point_t>& pointCloud, 
                          std::vector<std::pair<size_t, size_t>>& internalLayoutRanges)
            : pointCloud(pointCloud), internalLayoutRanges(internalLayoutRanges) {}

        // Copy constructor
        NeighSearchResult(const NeighSearchResult& other) = default;
    
        // Move constructor
        NeighSearchResult(NeighSearchResult&& other) noexcept = default;
            
        // Copy assignment operator
        NeighSearchResult& operator=(const NeighSearchResult& other) {
            if (this != &other) {
                octantIndexes = other.octantIndexes;
                extraPoints = other.extraPoints;
                pointCloud = other.pointCloud;
                internalLayoutRanges = other.internalLayoutRanges;
                numberOfPoints = other.numberOfPoints;
            }
            return *this;
        }

        // Move assignment operator
        NeighSearchResult& operator=(NeighSearchResult&& other) noexcept = default;

        class Iterator {
            public:
                using iterator_category = std::forward_iterator_tag;
                using value_type = Point_t;
                using difference_type = std::ptrdiff_t;
                using pointer = const Point_t*;
                using reference = const Point_t&;
        
                Iterator(const NeighSearchResult& result, size_t octant_idx, size_t point_idx)
                    : result(result), currentOctant(octant_idx), pointIndex(point_idx) {
                    updateCurrentPoint();
                }
        
                reference operator*() const { return *currentPoint; }
                pointer operator->() const { return currentPoint; }
        
                Iterator& operator++() {
                    ++pointIndex;
                    updateCurrentPoint();
                    return *this;
                }
        
                Iterator operator++(int) {
                    Iterator temp = *this;
                    ++(*this);
                    return temp;
                }
        
                bool operator==(const Iterator& other) const {
                    return currentOctant == other.currentOctant && pointIndex == other.pointIndex;
                }
        
                bool operator!=(const Iterator& other) const { return !(*this == other); }
        
            private:
                const NeighSearchResult& result;
                size_t currentOctant;
                size_t pointIndex;
                pointer currentPoint = nullptr;
        
                void updateCurrentPoint() {
                    while (currentOctant < result.octantIndexes.size()) {
                        size_t startIndex = result.internalLayoutRanges[result.octantIndexes[currentOctant]].first;
                        size_t endIndex = result.internalLayoutRanges[result.octantIndexes[currentOctant]].second;
                        if (startIndex + pointIndex < endIndex) {
                            currentPoint = &result.pointCloud[startIndex + pointIndex];
                            return;
                        }
                        ++currentOctant;
                        pointIndex = 0;
                    }
                    if (pointIndex < result.extraPoints.size()) {
                        currentPoint = result.extraPoints[pointIndex];
                        return;
                    }
                    currentPoint = nullptr;
                }
            };
        
        Iterator begin() const {
            return Iterator(*this, 0, 0);
        }

        Iterator end() const {
            return Iterator(*this, octantIndexes.size(), extraPoints.size());
        }

        size_t size() const {
            return numberOfPoints;
        }
};
