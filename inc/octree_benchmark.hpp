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
        const size_t searchSize;

        constexpr static bool CHECK_RESULTS = false;
        constexpr static float MIN_RADIUS = 0.01;
        constexpr static float MAX_RADIUS = 100.0;
        constexpr static uint32_t MIN_KNN = 5;
        constexpr static uint32_t MAX_KNN = 30000;

        const char* BENCHMARK_FILE_PATH = "benchmark_results.csv";

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
            searchPointIndexes.resize(searchSize);
            searchRadii.resize(searchSize);
            innerRingRadii.resize(searchSize);
            outerRingRadii.resize(searchSize);
            searchKNNLimits.resize(searchSize);
        }

        void allocateResultCheckingMemory() {
            searchResultsPointer.resize(searchSize);
            searchResultsOldLinear.resize(searchSize);
            searchResultsLinear.resize(searchSize);
            numNeighResultsPointer.resize(searchSize);
            numNeighResultsOldLinear.resize(searchSize);
            numNeighResultsLinear.resize(searchSize);
        }

        void generateSearchSet() {
            rng.seed(42);
            std::uniform_int_distribution<size_t> indexDist(0, points.size()-1);
            std::uniform_real_distribution<float> radiusDist(MIN_RADIUS, MAX_RADIUS);
            std::uniform_int_distribution<size_t> knnDist(MIN_KNN, MAX_KNN);

            
            #pragma omp parallel for
                for(int i = 0; i<searchSize; i++) {
                    searchPointIndexes[i] = indexDist(rng);
                    searchRadii[i] = radiusDist(rng);
                    innerRingRadii[i] = Vector(radiusDist(rng), radiusDist(rng), radiusDist(rng));
                    outerRingRadii[i] = Vector(innerRingRadii[i].getX() + radiusDist(rng), 
                                                innerRingRadii[i].getY() + radiusDist(rng), 
                                                innerRingRadii[i].getZ() + radiusDist(rng));
                    searchKNNLimits[i] = knnDist(rng);
                }
        }

        bool checkNeighSearchResults() {
            bool correct = true;
            for(int i = 0; i<searchSize; i++) {
                auto point = searchResultsPointer[i];
                auto oldLin = searchResultsLinear[i];
                if(point.size() != oldLin.size()) {
                            std::cout << "Wrong search result size in set " << i << 
                    "\n Pointer = " << point.size() << " Old linear = " << oldLin.size() << std::endl << 
                    "Search center: " << points[searchPointIndexes[i]] << " with radii " << searchRadii[i] << std::endl;
                    correct = false;
                } else {
                    sort(point.begin(), point.end(), [&] (Lpoint *p, Lpoint* q) -> bool {
                        return p->id() < q->id();
                    });
                    sort(oldLin.begin(), oldLin.end(), [&] (Lpoint *p, Lpoint* q) -> bool {
                        return p->id() < q->id();
                    });
                    for(int j = 0; j<point.size(); j++){
                        if(point[j]->id() != oldLin[j]->id()) {
                            std::cout << "Wrong search result point in set " << i << " at index " << j <<
                            "\n Pointer = " << point[j]->id() << " Old linear =  " << oldLin[j]->id() << std::endl;
                            correct = false;
                        }
                    }
                }
            }

            for(int i = 0; i<searchSize; i++) {
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
            for(int i = 0; i<searchSize; i++) {
                size_t point = numNeighResultsPointer[i];
                size_t oldLin = numNeighResultsLinear[i];
                if(point != oldLin) {
                    std::cout << "Wrong search result size in set " << i << 
                    "\n Pointer = " << point << " Old linear = " << oldLin << std::endl << 
                    "Search center: " << points[searchPointIndexes[i]] << " with radii " << searchRadii[i] << std::endl;
                    correct = false;
                }
            }
            for(int i = 0; i<searchSize; i++) {
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
            for(int i = 0; i<searchSize; i++) {
                auto point = searchResultsPointer[i];
                auto oldLin = searchResultsLinear[i];
                if(point.size() != oldLin.size()) {
                    std::cout << "Wrong search result size in set " << i << 
                    "\n Pointer = " << point.size() << " Old linear = " << oldLin.size() << std::endl << 
                    "Search center: " << points[searchPointIndexes[i]] << " with max KNN " << searchKNNLimits[i] << std::endl;
                    correct = false;
                } else {
                    sort(point.begin(), point.end(), [&] (Lpoint *p, Lpoint* q) -> bool {
                        return p->id() < q->id();
                    });
                    sort(oldLin.begin(), oldLin.end(), [&] (Lpoint *p, Lpoint* q) -> bool {
                        return p->id() < q->id();
                    });
                    for(int j = 0; j<point.size(); j++){
                        if(point[j]->id() != oldLin[j]->id()) {
                            std::cout << "Wrong search result point in set " << i << " at index " << j <<
                            "\n Pointer = " << point[j]->id() << " Old linear =  " << oldLin[j]->id() << std::endl;
                            correct = false;
                        }
                    }
                }
            }

            for(int i = 0; i<searchSize; i++) {
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
            for(int i = 0; i<searchSize; i++) {
                auto point = searchResultsPointer[i];
                auto oldLin = searchResultsLinear[i];
                if(point.size() != oldLin.size()) {
                    std::cout << "Wrong search result size in set " << i << 
                    "\n Pointer = " << point.size() << " Old linear = " << oldLin.size() << "\n" << 
                    "  Search center: " << points[searchPointIndexes[i]] << "\n" <<
                    "  Inner radii of the ring " << innerRingRadii[i] << "\n" <<
                    "   Outer radii of the ring " << outerRingRadii[i] << std::endl;
                    correct = false;
                } else {
                    sort(point.begin(), point.end(), [&] (Lpoint *p, Lpoint* q) -> bool {
                        return p->id() < q->id();
                    });
                    sort(oldLin.begin(), oldLin.end(), [&] (Lpoint *p, Lpoint* q) -> bool {
                        return p->id() < q->id();
                    });
                    for(int j = 0; j<point.size(); j++){
                        if(point[j]->id() != oldLin[j]->id()) {
                            std::cout << "Wrong search result point in set " << i << " at index " << j <<
                            "\n Pointer = " << point[j]->id() << " Old linear =  " << oldLin[j]->id() << std::endl;
                            correct = false;
                        }
                    }
                }
            }
            for(int i = 0; i<searchSize; i++) {
                auto point = searchResultsPointer[i];
                auto lin = searchResultsLinear[i];
                if(point.size() != lin.size()) {
                    std::cout << "Wrong search result size in set " << i << 
                    "\n Pointer = " << point.size() << " Linear = " << lin.size() << "\n" << 
                    "  Search center: " << points[searchPointIndexes[i]] << "\n" <<
                    "  Inner radii of the ring " << innerRingRadii[i] << "\n" <<
                    "   Outer radii of the ring " << outerRingRadii[i] << std::endl;
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
                for(int i = 0; i<searchSize; i++) {
                    if(CHECK_RESULTS){
                        searchResultsPointer[i] = pOct->searchNeighbors<kernel>(points[searchPointIndexes[i]], searchRadii[i]);
                    } else{
                        (void) pOct->searchNeighbors<kernel>(points[searchPointIndexes[i]], searchRadii[i]);
                        
                    }
                }
        }

        template<Kernel_t kernel>
        void oldLinearSearchNeigh() {
            #pragma omp parallel for schedule(static)
                for(int i = 0; i<searchSize; i++) {
                    if(CHECK_RESULTS){
                        searchResultsOldLinear[i] = oldLOct->searchNeighbors<kernel>(points[searchPointIndexes[i]], searchRadii[i]);
                    }else{
                        (void) oldLOct->searchNeighbors<kernel>(points[searchPointIndexes[i]], searchRadii[i]);
                        
                    }
                }
        }

        template<Kernel_t kernel>
        void linearOctreeNeighborSearch() {
            #pragma omp parallel for schedule(static)
                for(int i = 0; i<searchSize; i++) {
                    if(CHECK_RESULTS){
                        searchResultsLinear[i] = lOct->searchNeighbors<kernel>(points[searchPointIndexes[i]], searchRadii[i]);
                    } else {
                        (void) lOct->searchNeighbors<kernel>(points[searchPointIndexes[i]], searchRadii[i]);
                    }
                }
        }

        template<Kernel_t kernel>
        void pointerOctreeNumNeigh() {
            #pragma omp parallel for schedule(static)
                for(int i = 0; i<searchSize; i++) {
                    if(CHECK_RESULTS) {
                        numNeighResultsPointer[i] = pOct->numNeighbors<kernel>(points[searchPointIndexes[i]], searchRadii[i]);
                    } else {
                        (void) pOct->numNeighbors<kernel>(points[searchPointIndexes[i]], searchRadii[i]);
                    }
                }
        }

        template<Kernel_t kernel>
        void oldLinearOctreeNumNeigh() {
            #pragma omp parallel for schedule(static)
                for(int i = 0; i<searchSize; i++) {
                    if(CHECK_RESULTS) {
                        numNeighResultsOldLinear[i] = oldLOct->numNeighbors<kernel>(points[searchPointIndexes[i]], searchRadii[i]);
                    } else {
                        (void) oldLOct->numNeighbors<kernel>(points[searchPointIndexes[i]], searchRadii[i]);
                    }
                }
        }

        template<Kernel_t kernel>
        void linearOctreeNumNeigh() {
            #pragma omp parallel for schedule(static)
                for(int i = 0; i<searchSize; i++) {
                    if(CHECK_RESULTS) {
                        numNeighResultsLinear[i] = lOct->numNeighbors<kernel>(points[searchPointIndexes[i]], searchRadii[i]);
                    } else {
                        (void) lOct->numNeighbors<kernel>(points[searchPointIndexes[i]], searchRadii[i]);
                    }
                }
        }

        void pointerOctreeKNN() {
            #pragma omp parallel for schedule(static)
                for(int i = 0; i<searchSize; i++) {
                    if(CHECK_RESULTS) {
                        searchResultsPointer[i] = pOct->KNN(points[searchPointIndexes[i]], searchKNNLimits[i], searchKNNLimits[i]);
                    } else {
                        (void) pOct->KNN(points[searchPointIndexes[i]], searchKNNLimits[i], searchKNNLimits[i]);
                    }
                }
        }

        void oldLinearOctreeKNN() {
            #pragma omp parallel for schedule(static)
                for(int i = 0; i<searchSize; i++) {
                    if(CHECK_RESULTS) {
                        searchResultsOldLinear[i] = oldLOct->KNN(points[searchPointIndexes[i]], searchKNNLimits[i], searchKNNLimits[i]);
                    } else {
                        (void) oldLOct->KNN(points[searchPointIndexes[i]], searchKNNLimits[i], searchKNNLimits[i]);
                    }
                }
        }

        void linearOctreeKNN() {
            #pragma omp parallel for schedule(static)
                for(int i = 0; i<searchSize; i++) {
                    if(CHECK_RESULTS) {
                        searchResultsLinear[i] = lOct->KNN(points[searchPointIndexes[i]], searchKNNLimits[i], searchKNNLimits[i]);
                    } else {
                        (void) lOct->KNN(points[searchPointIndexes[i]], searchKNNLimits[i], searchKNNLimits[i]);
                    }
                }
        }

        void pointerOctreeRingSearchNeigh() {
            #pragma omp parallel for schedule(static)
                for(int i = 0; i<searchSize; i++) {
                    if(CHECK_RESULTS) {    
                        searchResultsPointer[i] = pOct->searchNeighborsRing(points[searchPointIndexes[i]], 
                            innerRingRadii[i], outerRingRadii[i]);
                    } else {
                        (void) pOct->searchNeighborsRing(points[searchPointIndexes[i]], 
                            innerRingRadii[i], outerRingRadii[i]);
                    }
                }
        }

        void oldLinearOctreeRingSearchNeigh() {
            #pragma omp parallel for schedule(static)
                for(int i = 0; i<searchSize; i++) {
                    if(CHECK_RESULTS) {
                        searchResultsOldLinear[i] = oldLOct->searchNeighborsRing(points[searchPointIndexes[i]], 
                            innerRingRadii[i], outerRingRadii[i]);
                    } else {
                        (void) oldLOct->searchNeighborsRing(points[searchPointIndexes[i]], 
                            innerRingRadii[i], outerRingRadii[i]);
                    }
                }
        }

        void linearOctreeRingSearchNeigh() {
            #pragma omp parallel for schedule(static)
                for(int i = 0; i<searchSize; i++) {
                    if(CHECK_RESULTS) {
                        searchResultsLinear[i] = lOct->searchNeighborsRing(points[searchPointIndexes[i]], 
                            innerRingRadii[i], outerRingRadii[i]);
                    } else {
                        (void) lOct->searchNeighborsRing(points[searchPointIndexes[i]], 
                            innerRingRadii[i], outerRingRadii[i]);
                    }
                }
        }

        inline std::string getCurrentDate() {
            auto t = std::time(nullptr);
            auto tm = *std::localtime(&t);
            std::ostringstream oss;
            oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
            return oss.str();
        }

        inline void appendToCsv(const std::string& octree, const std::string& operation, 
                            const std::string& kernel, const benchmarking::Stats<>& stats) {
            // Open the file in append mode
            std::ofstream file(BENCHMARK_FILE_PATH, std::ios::app);
            if (!file.is_open()) {
                throw std::ios_base::failure(std::string("Failed to open benchmark output file: ") + BENCHMARK_FILE_PATH);
            }

            // Check if the file is empty and append header if it is
            if (file.tellp() == 0) {
                file << "date,octree,operation,kernel,search_size,repeats,accumulated,mean,median,stdev,used_warmup\n";
            }

            // Append the benchmark data
            file << getCurrentDate() << ',' 
                << octree << ',' 
                << operation << ',' 
                << kernel << ',' 
                << searchSize << ',' 
                << stats.size() << ','
                << stats.accumulated() << ',' 
                << stats.mean() << ',' 
                << stats.median() << ',' 
                << stats.stdev() << ','
                << stats.usedWarmup() << '\n';
        }
    
    public:
        OctreeBenchmark(std::vector<Lpoint> &points, std::vector<Lpoint> &lsOctreePoints, size_t searchSetsSize = 100) : 
            points(points), lOctPoints(lsOctreePoints), searchSize(searchSetsSize) {
            allocateSearchSetMemory();
            if(CHECK_RESULTS)
                allocateResultCheckingMemory();
            octreePointerBuild();
            // octreeOldLinearBuild();
            octreeLinearBuild();
        }

        void benchmarkbuild(size_t repeats = 1) {
            auto statsPointer = benchmarking::benchmark(repeats, [this]() { octreePointerBuild(); });
            appendToCsv("pointer", "build", "NA", statsPointer);

            // auto statsOldLinear = benchmarking::benchmark(repeats, [this]() { octreeOldLinearBuild(); });
            // appendToCsv("oldLinear", "build", "N/A", statsOldLinear);

            auto statsLinear = benchmarking::benchmark(repeats, [this]() { octreeLinearBuild(); });
            appendToCsv("linear", "build", "NA", statsLinear);
        }

        template<Kernel_t kernel>
        void benchmarkSearchNeigh(size_t repeats = 1) {
            const auto kernelStr = kernelToString(kernel);

            generateSearchSet();

            auto statsPointer = benchmarking::benchmark(repeats, [this]() { pointerOctreeSearchNeigh<kernel>(); });
            appendToCsv("pointer", "neighSearch", kernelStr, statsPointer);

            // auto statsOldLinear = benchmarking::benchmark(repeats, [this]() { oldLinearSearchNeigh<kernel>(); });
            // appendToCsv("oldLinear", "neighSearch", kernelStr, statsOldLinear);

            auto statsLinear = benchmarking::benchmark(repeats, [this]() { linearOctreeNeighborSearch<kernel>(); });
            appendToCsv("linear", "neighSearch", kernelStr, statsLinear);
            
            if (CHECK_RESULTS && checkNeighSearchResults())
                std::cout << "All neighbors search methods yield the same results!" << std::endl;

            std::cout << "Benchmark search neighbors with kernel " << kernelStr << " completed" << std::endl;
        }

        template<Kernel_t kernel>
        void benchmarkNumNeigh(size_t repeats = 1) {
            const auto kernelStr = kernelToString(kernel);

            generateSearchSet();

            auto statsPointer = benchmarking::benchmark(repeats, [this]() { pointerOctreeNumNeigh<kernel>(); });
            appendToCsv("pointer", "numNeighSearch", kernelStr, statsPointer);

            // auto statsOldLinear = benchmarking::benchmark(repeats, [this]() { oldLinearOctreeNumNeigh<kernel>(); });
            // appendToCsv("oldLinear", "numNeighSearch", kernelStr, statsOldLinear);

            auto statsLinear = benchmarking::benchmark(repeats, [this]() { linearOctreeNumNeigh<kernel>(); });
            appendToCsv("linear", "numNeighSearch", kernelStr, statsLinear);
            
            if(CHECK_RESULTS && checkNumNeighResults())
                std::cout << "All count neighbors methods yield the same results!" << std::endl;
            
            std::cout << "Benchmark search num neighbors with kernel " << kernelStr << " completed" << std::endl;
        }

        void benchmarkKNN(size_t repeats = 1) {
            generateSearchSet();

            auto statsPointer = benchmarking::benchmark(repeats, [this]() { pointerOctreeKNN(); });
            appendToCsv("pointer", "KNN", "NA", statsPointer);

            // auto statsOldLinear = benchmarking::benchmark(repeats, [this]() { oldLinearOctreeKNN(); });
            // appendToCsv("oldLinear", "KNN", "NA", statsOldLinear);

            auto statsLinear = benchmarking::benchmark(repeats, [this]() { linearOctreeKNN(); });
            appendToCsv("linear", "KNN", "NA", statsLinear);

            if(CHECK_RESULTS && checkNeighSearchResults())
                std::cout << "All KNN search methods yield the same results!" << std::endl;
            
            std::cout << "Benchmark KNN completed" << std::endl;
        }

        void benchmarkRingSearchNeigh(size_t repeats = 1) {
            generateSearchSet();
            
            auto statsPointer = benchmarking::benchmark(repeats, [this]() { pointerOctreeRingSearchNeigh(); });
            appendToCsv("pointer", "ringNeighSearch", "NA", statsPointer);

            // auto statsOldLinear = benchmarking::benchmark(repeats, [this]() { oldLinearOctreeRingSearchNeigh(); });
            // appendToCsv("oldLinear", "ringNeighSearch", "NA", statsOldLinear);

            auto statsLinear = benchmarking::benchmark(repeats, [this]() { linearOctreeRingSearchNeigh(); });
            appendToCsv("linear", "ringNeighSearch", "NA", statsLinear);

            if(CHECK_RESULTS && checkRingNeighSearchResults())
                std::cout << "All KNN search methods yield the same results!" << std::endl;
            
            std::cout << "Benchmark search neighbors on ring completed" << std::endl;
        }
};
