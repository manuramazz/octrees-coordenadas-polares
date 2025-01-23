#include "util.hpp"
#include "type_names.hpp"
#include "TimeWatcher.hpp"
#include "handlers.hpp"
#include "main_options.hpp"
#include "octree.hpp"
#include "octree_linear.hpp"
#include <filesystem> // Only C++17 and beyond
#include <iomanip>
#include <iostream>
#include "benchmarking.hpp"
#include <random>
#include "NeighborKernels/KernelFactory.hpp"
#include "octree_benchmark.hpp"
#include "Geometry/Lpoint.hpp"
#include "Geometry/Lpoint64.hpp"
#include "PointEncoding/morton_encoder.hpp"
#include "PointEncoding/hilbert_encoder.hpp"
#include <new>

namespace fs = std::filesystem;

template <template <typename, typename> class Octree_t, PointType Point_t, typename Encoder_t>
std::shared_ptr<ResultSet<Point_t>> runSearchBenchmark(std::ofstream &outputFile, std::vector<Point_t>& points,
  std::shared_ptr<const SearchSet> searchSet, std::string comment = "") {
  OctreeBenchmark<Octree_t, Point_t, Encoder_t> ob(points, mainOptions.numSearches, searchSet, outputFile, mainOptions.checkResults, comment);
  ob.searchBench(mainOptions.benchmarkRadii, REPEATS, mainOptions.numSearches);
  return ob.getResultSet();
}

template <template <typename, typename> class Octree_t, PointType Point_t, typename Encoder_t>
std::shared_ptr<ResultSet<Point_t>> runSearchImplComparisonBenchmark(std::ofstream &outputFile, std::vector<Point_t>& points,
  std::shared_ptr<const SearchSet> searchSet, std::string comment = "") {
  OctreeBenchmark<Octree_t, Point_t, Encoder_t> ob(points, mainOptions.numSearches, searchSet, outputFile, mainOptions.checkResults, comment);
  ob.searchImplComparisonBench(mainOptions.benchmarkRadii, REPEATS, mainOptions.numSearches);
  return ob.getResultSet();
}

template <template <typename, typename> class Octree_t, PointType Point_t, typename Encoder_t, Kernel_t kernel>
std::shared_ptr<ResultSet<Point_t>> runSingleKernelRadiiBenchmark(std::ofstream &outputFile, std::vector<Point_t>& points,
  std::shared_ptr<const SearchSet> searchSet, const float radius, std::string comment = "") {
  OctreeBenchmark<Octree_t, Point_t, Encoder_t> ob(points, mainOptions.numSearches, searchSet, outputFile, mainOptions.checkResults, comment);
  // only 1 repeat and no warmup
  ob.template benchmarkSearchNeighSeq<kernel>(1, radius);
  return ob.getResultSet();
}

/**
 * Benchmark neighSearch and numNeighSearch for a given octree configuration (point type + encoder).
 * Compares LinearOctree and PointerOctree. If passed PointEncoding::NoEncoder, only PointerOctree is used.
 */
template <PointType Point_t, typename Encoder_t>
void searchBenchmark(std::ofstream &outputFile) {
  TimeWatcher tw;
  tw.start();
  auto points = readPointCloud<Point_t>(mainOptions.inputFile);
  tw.stop();

  std::cout << "Number of read points: " << points.size() << "\n";
  std::cout << "Time to read points: " << tw.getElapsedDecimalSeconds()
            << " seconds\n";
  checkVectorMemory(points);
  std::shared_ptr<const SearchSet> searchSet = std::make_shared<const SearchSet>(mainOptions.numSearches, points);

  if constexpr (std::is_same_v<Encoder_t, PointEncoding::NoEncoder>) {
    // Only do pointer octree, since we are not encoding the points
    runSearchBenchmark<Octree, Point_t, PointEncoding::NoEncoder>(outputFile, points, searchSet);
  } else {
    // Do both linear (which encodes and sorts the points) and pointer octree after it
    auto resultsLinear = runSearchBenchmark<LinearOctree, Point_t, Encoder_t>(outputFile, points, searchSet);
    auto resultsPointer = runSearchBenchmark<Octree, Point_t, Encoder_t>(outputFile, points, searchSet);
    if(mainOptions.checkResults) {
      resultsLinear->checkResults(resultsPointer);
    }
  }
}

/**
 * Benchmark oldNeighSearch vs neighSearch and oldNumNeighSearch vs numNeighSearch for a given octree configuration (point type + encoder),
 * in order to see the performance improvement of the new implementation.
 * 
 * Only uses LinearOctree, since that's where the implementation is being improved.
 */
template <PointType Point_t, typename Encoder_t>
void searchImplComparisonBenchmark(std::ofstream &outputFile) {
  TimeWatcher tw;
  tw.start();
  auto points = readPointCloud<Point_t>(mainOptions.inputFile);
  tw.stop();

  std::cout << "Number of read points: " << points.size() << "\n";
  std::cout << "Time to read points: " << tw.getElapsedDecimalSeconds()
            << " seconds\n";
  checkVectorMemory(points);

  // Generate a shared search set for each benchmark execution
  std::shared_ptr<const SearchSet> searchSet = std::make_shared<const SearchSet>(mainOptions.numSearches, points);

  auto results = runSearchImplComparisonBenchmark<LinearOctree, Point_t, Encoder_t>(outputFile, points, searchSet);

  // Check the results if needed
  if(mainOptions.checkResults) {
    results->checkResultsAlgo();
  }
}

/**
 * Runs neighSearch for every point in the point cloud, comparing the performance 
 * of sequential vs shuffled searchSets due to cache losses.
 * 
 * Only uses one kernel and one radii to not make the benchmark too long.
 * 
 * Uses LinearOctree
 */
template <PointType Point_t, typename Encoder_t, Kernel_t kernel>
void sequentialVsShuffleBenchmark(std::ofstream &outputFile, const float radius) {
  TimeWatcher tw;
  tw.start();
  auto points = readPointCloud<Point_t>(mainOptions.inputFile);
  tw.stop();

  std::cout << "Number of read points: " << points.size() << "\n";
  std::cout << "Time to read points: " << tw.getElapsedDecimalSeconds()
            << " seconds\n";
  checkVectorMemory(points);

  // Generate a shared search set for each benchmark execution

  // In this benchmark we only do one radii and one kernel, because otherwise it would be too much
  // and we are only interested in the difference between sequential and shuffled points
  std::shared_ptr<SearchSet> searchSetShuffle = std::make_shared<SearchSet>(points, true);
  std::cout << "Running search over shuffled search set of size " << searchSetShuffle->numSearches << std::endl;
  runSingleKernelRadiiBenchmark<LinearOctree, Point_t, Encoder_t, kernel>(outputFile, points, searchSetShuffle, radius, "Shuffled");
  searchSetShuffle->searchPoints.clear();
  searchSetShuffle->searchKNNLimits.clear();

  std::shared_ptr<SearchSet> searchSetSeq = std::make_shared<SearchSet>(points, false);
  std::cout << "Running search over sequential search set of size" << searchSetSeq->numSearches << std::endl;
  runSingleKernelRadiiBenchmark<LinearOctree, Point_t, Encoder_t, kernel>(outputFile, points, searchSetSeq, radius, "Sequential");
}

int main(int argc, char *argv[]) {
  setDefaults();
  processArgs(argc, argv);
  
  fs::path inputFile = mainOptions.inputFile;
  std::string fileName = inputFile.stem();

  if (!mainOptions.outputDirName.empty()) {
    mainOptions.outputDirName = mainOptions.outputDirName / fileName;
  }

  // Handling Directories
  createDirectory(mainOptions.outputDirName);

  // Print three decimals
  std::cout << std::fixed;
  std::cout << std::setprecision(3);

  // Open the benchmark output file
  std::string csvFilename = mainOptions.inputFileName + "-" + getCurrentDate() + ".csv";
  std::filesystem::path csvPath = mainOptions.outputDirName / csvFilename;
  std::ofstream outputFile(csvPath, std::ios::app);
  if (!outputFile.is_open()) {
      throw std::ios_base::failure(std::string("Failed to open benchmark output file: ") + csvPath.string());
  }

  switch(mainOptions.benchmarkMode) {
    case BenchmarkMode::SEARCH:
      searchBenchmark<Lpoint64, PointEncoding::HilbertEncoder3D>(outputFile);
      searchBenchmark<Lpoint64, PointEncoding::MortonEncoder3D>(outputFile);
      searchBenchmark<Lpoint64, PointEncoding::NoEncoder>(outputFile);
    break;
    case BenchmarkMode::COMPARE:
      searchImplComparisonBenchmark<Lpoint64, PointEncoding::HilbertEncoder3D>(outputFile);
      searchImplComparisonBenchmark<Lpoint64, PointEncoding::MortonEncoder3D>(outputFile);
    break;
    case BenchmarkMode::SEQUENTIAL:
      sequentialVsShuffleBenchmark<Lpoint64, PointEncoding::HilbertEncoder3D, Kernel_t::sphere>(outputFile, 5.0);
    break;
  }

  return EXIT_SUCCESS;
}
