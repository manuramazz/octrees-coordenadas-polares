#pragma once

#include "benchmarking.hpp"
#include "octree.hpp"
#include "octree_linear.hpp"
#include "octree_linear_old.hpp"
#include "morton_encoder.hpp"
#include <random>
#include "point.hpp"
#include <omp.h>
#include "NeighborKernels/KernelFactory.hpp"


class OctreeBenchmark {
    private:
        const size_t search_size;
        constexpr static float min_radius = 0.01;
        constexpr static float max_radius = 100.0;
        constexpr static uint32_t min_knn = 5;
        constexpr static uint32_t max_knn = 30000;
        
        // Copy points for the linear octree because it reorders them and also to be more fair on comparisons
        // i.e. neither tree has points already in-cache when executing after the other
        std::vector<Lpoint> &points, &lOctPoints;

        Octree* pOct = nullptr;
        LinearOctreeOld* oldLOct = nullptr;
        LinearOctree *lOct = nullptr;
        std::mt19937 rng;
        
        std::vector<size_t> searchPointIndexes;
        std::vector<float> searchRadii;
        std::vector<Vector> innerRingRadii;
        std::vector<Vector> outerRingRadii;
        std::vector<uint32_t> searchKNNLimits;

        std::vector<std::vector<Lpoint*>> searchResultsPointer;
        std::vector<std::vector<Lpoint*>> searchResultsOldLinear;
        std::vector<std::vector<Lpoint*>> searchResultsLinear;

        std::vector<size_t> numNeighResultsPointer;
        std::vector<size_t> numNeighResultsOldLinear;
        std::vector<size_t> numNeighResultsLinear;

        void allocateSearchSetMemory() {
            searchResultsPointer.resize(search_size);
            searchResultsOldLinear.resize(search_size);
            searchResultsLinear.resize(search_size);
            numNeighResultsPointer.resize(search_size);
            numNeighResultsOldLinear.resize(search_size);
            numNeighResultsLinear.resize(search_size);
            searchPointIndexes.resize(search_size);
            searchRadii.resize(search_size);
            innerRingRadii.resize(search_size);
            outerRingRadii.resize(search_size);
            searchKNNLimits.resize(search_size);
        }

        void generateSearchSet() {
            rng.seed(42);
            std::uniform_int_distribution<size_t> indexDist(0, points.size()-1);
            std::uniform_real_distribution<float> radiusDist(min_radius, max_radius);
            std::uniform_int_distribution<size_t> knnDist(min_knn, max_knn);

            
            #pragma omp parallel for
                for(int i = 0; i<search_size; i++) {
                    searchPointIndexes[i] = indexDist(rng);
                    searchRadii[i] = radiusDist(rng);
                    innerRingRadii[i] = Vector(radiusDist(rng), radiusDist(rng), radiusDist(rng));
                    outerRingRadii[i] = Vector(innerRingRadii[i].getX() + radiusDist(rng), 
                                                innerRingRadii[i].getY() + radiusDist(rng), 
                                                innerRingRadii[i].getZ() + radiusDist(rng));
                    searchKNNLimits[i] = knnDist(rng);
                    searchResultsPointer[i].clear();
                    searchResultsOldLinear[i].clear();
                    searchResultsLinear[i].clear();
                    numNeighResultsPointer[i] = 0;
                    numNeighResultsOldLinear[i] = 0;
                    numNeighResultsLinear[i] = 0;
                }
        }

        bool checkNeighSearchResults() {
            bool correct = true;
            for(int i = 0; i<search_size; i++) {
                auto point = searchResultsPointer[i];
                auto lin = searchResultsLinear[i];
                if(point.size() != lin.size()) {
                    std::cout << "Wrong search result size in set " << i << 
                    "\n Pointer = " << point.size() << " Linear = " << lin.size() << std::endl << 
                    "Search center: " << points[searchPointIndexes[i]] << " with radii " << searchRadii[i] << std::endl;
                    correct = false;
                } else {
                    sort(point.begin(), point.end(), [&] (Lpoint *p, Lpoint* q) -> bool {
                        return p->id() < q->id();
                    });
                    sort(lin.begin(), lin.end(), [&] (Lpoint *p, Lpoint* q) -> bool {
                        return p->id() < q->id();
                    });
                    for(int j = 0; j<point.size(); j++){
                        if(point[j]->id() != lin[j]->id()) {
                            std::cout << "Wrong search result point in set " << i << " at index " << j <<
                            "\n Pointer = " << point[j]->id() << " Linear =  " << lin[j]->id() << std::endl;
                            correct = false;
                        }
                    }
                }
            }
            return correct;
        }

        bool checkNumNeighResults() {
            bool correct = true;
            for(int i = 0; i<search_size; i++) {
                size_t point = numNeighResultsPointer[i];
                size_t lin = numNeighResultsLinear[i];
                if(point != lin) {
                    std::cout << "Wrong search result size in set " << i << 
                    "\n Pointer = " << point << " Linear = " << lin << std::endl << 
                    "Search center: " << points[searchPointIndexes[i]] << " with radii " << searchRadii[i] << std::endl;
                    correct = false;
                }
            }
            return correct;
        }

        bool checkNeighKNNResults() {
            bool correct = true;
            for(int i = 0; i<search_size; i++) {
                auto point = searchResultsPointer[i];
                auto lin = searchResultsLinear[i];
                if(point.size() != lin.size()) {
                    std::cout << "Wrong search result size in set " << i << 
                    "\n Pointer = " << point.size() << " Linear = " << lin.size() << std::endl << 
                    "Search center: " << points[searchPointIndexes[i]] << " with max KNN " << searchKNNLimits[i] << std::endl;
                    correct = false;
                } else {
                    sort(point.begin(), point.end(), [&] (Lpoint *p, Lpoint* q) -> bool {
                        return p->id() < q->id();
                    });
                    sort(lin.begin(), lin.end(), [&] (Lpoint *p, Lpoint* q) -> bool {
                        return p->id() < q->id();
                    });
                    for(int j = 0; j<point.size(); j++){
                        if(point[j]->id() != lin[j]->id()) {
                            std::cout << "Wrong search result point in set " << i << " at index " << j <<
                            "\n Pointer = " << point[j]->id() << " Linear =  " << lin[j]->id() << std::endl;
                            correct = false;
                        }
                    }
                }
            }
            return correct;
        }

        bool checkRingNeighSearchResults() {
            bool correct = true;
            for(int i = 0; i<search_size; i++) {
                auto point = searchResultsPointer[i];
                auto lin = searchResultsLinear[i];
                if(point.size() != lin.size()) {
                    std::cout << "Wrong search result size in set " << i << 
                    "\n Pointer = " << point.size() << " Linear = " << lin.size() << std::endl << 
                    "Search center: " << points[searchPointIndexes[i]] << " with inner radii of the ring " << innerRingRadii[i] << std::endl <<
                    " and outer radii of the ring " << outerRingRadii[i] << std::endl;
                    correct = false;
                } else {
                    sort(point.begin(), point.end(), [&] (Lpoint *p, Lpoint* q) -> bool {
                        return p->id() < q->id();
                    });
                    sort(lin.begin(), lin.end(), [&] (Lpoint *p, Lpoint* q) -> bool {
                        return p->id() < q->id();
                    });
                    for(int j = 0; j<point.size(); j++){
                        if(point[j]->id() != lin[j]->id()) {
                            std::cout << "Wrong search result point in set " << i << " at index " << j <<
                            "\n Pointer = " << point[j]->id() << " Linear =  " << lin[j]->id() << std::endl;
                            correct = false;
                        }
                    }
                }
            }
            return correct;
        }


        void octreePointerBuild() {
            // Pointer based octree
            if(pOct != nullptr)
                delete pOct;
            pOct = new Octree(points);
        }
        void octreeOldLinearBuild() {
            // Old linear octree impl.
            if(oldLOct != nullptr)
                delete oldLOct;
            oldLOct = new LinearOctreeOld(points);
        }
        void octreeLinearBuild() {
            // Linear octree impl.
            if(lOct != nullptr)
                delete lOct;
            lOct = new LinearOctree(lOctPoints);
        }

        template<Kernel_t kernel>
        void pointerOctreeSearchNeigh() {
            #pragma omp parallel for schedule(static)
                for(int i = 0; i<search_size; i++) {
                    searchResultsPointer[i] = pOct->searchNeighbors<kernel>(points[searchPointIndexes[i]], searchRadii[i]);
                }
        }

        template<Kernel_t kernel>
        void oldLinearSearchNeigh() {
            #pragma omp parallel for schedule(static)
                for(int i = 0; i<search_size; i++) {
                    searchResultsPointer[i] = oldLOct->searchNeighbors<kernel>(points[searchPointIndexes[i]], searchRadii[i]);
                }
        }

        template<Kernel_t kernel>
        void linearOctreeNeighborSearch() {
            #pragma omp parallel for schedule(static)
                for(int i = 0; i<search_size; i++) {
                    searchResultsLinear[i] = lOct->searchNeighbors<kernel>(points[searchPointIndexes[i]], searchRadii[i]);
                }
        }

        template<Kernel_t kernel>
        void pointerOctreeNumNeigh() {
            #pragma omp parallel for schedule(static)
                for(int i = 0; i<search_size; i++) {
                    numNeighResultsPointer[i] = pOct->numNeighbors<kernel>(points[searchPointIndexes[i]], searchRadii[i]);
                }
        }

        template<Kernel_t kernel>
        void oldLinearOctreeNumNeigh() {
            #pragma omp parallel for schedule(static)
                for(int i = 0; i<search_size; i++) {
                    numNeighResultsOldLinear[i] = oldLOct->numNeighbors<kernel>(points[searchPointIndexes[i]], searchRadii[i]);
                }
        }

        template<Kernel_t kernel>
        void linearOctreeNumNeigh() {
            #pragma omp parallel for schedule(static)
                for(int i = 0; i<search_size; i++) {
                    numNeighResultsLinear[i] = lOct->numNeighbors<kernel>(points[searchPointIndexes[i]], searchRadii[i]);
                }
        }

        void pointerOctreeKNN() {
            #pragma omp parallel for schedule(static)
                for(int i = 0; i<search_size; i++) {
                    searchResultsPointer[i] = pOct->KNN(points[searchPointIndexes[i]], searchKNNLimits[i], searchKNNLimits[i]);
                }
        }

        void oldLinearOctreeKNN() {
            #pragma omp parallel for schedule(static)
                for(int i = 0; i<search_size; i++) {
                    searchResultsOldLinear[i] = oldLOct->KNN(points[searchPointIndexes[i]], searchKNNLimits[i], searchKNNLimits[i]);
                }
        }

        void linearOctreeKNN() {
            #pragma omp parallel for schedule(static)
                for(int i = 0; i<search_size; i++) {
                    searchResultsLinear[i] = lOct->KNN(points[searchPointIndexes[i]], searchKNNLimits[i], searchKNNLimits[i]);
                }
        }

        void pointerOctreeRingSearchNeigh() {
            #pragma omp parallel for schedule(static)
                for(int i = 0; i<search_size; i++) {
                    searchResultsPointer[i] = pOct->searchNeighborsRing(points[searchPointIndexes[i]], innerRingRadii[i], outerRingRadii[i]);
                }
        }

        void oldLinearOctreeRingSearchNeigh() {
            #pragma omp parallel for schedule(static)
                for(int i = 0; i<search_size; i++) {
                    searchResultsPointer[i] = oldLOct->searchNeighborsRing(points[searchPointIndexes[i]], innerRingRadii[i], outerRingRadii[i]);
                }
        }

        void linearOctreeRingSearchNeigh() {
            #pragma omp parallel for schedule(static)
                for(int i = 0; i<search_size; i++) {
                    searchResultsLinear[i] = lOct->searchNeighborsRing(points[searchPointIndexes[i]], innerRingRadii[i], outerRingRadii[i]);
                }
        }

    public:
        OctreeBenchmark(std::vector<Lpoint> &points, std::vector<Lpoint> &lsOctreePoints, size_t searchSetsSize = 100) : 
            points(points), lOctPoints(lsOctreePoints), search_size(searchSetsSize) {
            allocateSearchSetMemory();
            octreePointerBuild();
            // octreeOldLinearBuild();
            octreeLinearBuild();
        }

        void benchmarkbuild(size_t repeats) {
            benchmarking::benchmark("Pointer octree build", repeats, [this]() { octreePointerBuild(); });
            // benchmarking::benchmark("Old linear octree build", repeats, [this]() { octreeOldLinearBuild(); });
            benchmarking::benchmark("Linear octree build", repeats, [this]() { octreeLinearBuild(); });
        }  

        template<Kernel_t kernel>
        void benchmarkSearchNeigh(size_t repeats = 1, bool checkResults = false) {
            const auto kernelStr = kernelToString(kernel);
            generateSearchSet();
            std::cout << "Generated search set containing " << search_size << " operations" << std::endl;
            benchmarking::benchmark(std::string("Pointer octree neighbor search with kernel ") + kernelStr, repeats, [this]() { pointerOctreeSearchNeigh<kernel>(); });
            // benchmarking::benchmark(std::string("Old linear octree neighbor search with kernel ") + kernelStr, repeats, [this]() { oldLinearSearchNeigh<kernel>(); });
            benchmarking::benchmark(std::string("Linear octree neighbor search with kernel ") + kernelStr, repeats, [this]() { linearOctreeNeighborSearch<kernel>(); });
            if(checkResults && checkNeighSearchResults())
                std::cout << "All neighbors search methods yield the same results!" << std::endl;
        }

        template<Kernel_t kernel>
        void benchmarkNumNeigh(size_t repeats = 1, bool checkResults = false) {
            const auto kernelStr = kernelToString(kernel);
            generateSearchSet();
            std::cout << "Generated search set containing " << search_size << " operations" << std::endl;
            benchmarking::benchmark(std::string("Pointer octree neighbor count with kernel ") + kernelStr, repeats, [this]() { pointerOctreeNumNeigh<kernel>(); });
            // benchmarking::benchmark(std::string("Old linear octree neighbor count with kernel ") + kernelStr, repeats, [this]() { oldLinearOctreeNumNeigh<kernel>(); });
            benchmarking::benchmark(std::string("Linear octree neighbor count with kernel ") + kernelStr, repeats, [this]() { linearOctreeNumNeigh<kernel>(); });
            if(checkResults && checkNumNeighResults())
                std::cout << "All count neighbors methods yield the same results!" << std::endl;
        }

        void benchmarkKNN(size_t repeats = 1, bool checkResults = false) {
            generateSearchSet();
            benchmarking::benchmark("Pointer octree KNN search", repeats, [this]() { pointerOctreeKNN(); });
            // benchmarking::benchmark("Old linear octree KNN search", repeats, [this]() { oldLinearOctreeKNN(); });
            benchmarking::benchmark("Linear octree KNN search", repeats, [this]() { linearOctreeKNN(); });
            if(checkResults && checkNeighSearchResults())
                std::cout << "All KNN search methods yield the same results!" << std::endl;
        }

        void benchmarkRingSearchNeigh(size_t repeats = 1, bool checkResults = false) {
            generateSearchSet();
            benchmarking::benchmark("Pointer octree ring neighbor search", repeats, [this]() { pointerOctreeRingSearchNeigh(); });
            // benchmarking::benchmark("Old linear octree ring neighbor search", repeats, [this]() { oldLinearOctreeRingSearchNeigh(); });
            benchmarking::benchmark("Linear octree ring neighbor search", repeats, [this]() { linearOctreeRingSearchNeigh(); });
            if(checkResults && checkRingNeighSearchResults())
                std::cout << "All KNN search methods yield the same results!" << std::endl;
        }

};
