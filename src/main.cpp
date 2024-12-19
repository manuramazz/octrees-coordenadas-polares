#include "util.hpp"
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

// Global benchmark parameters
const std::vector<float> BENCHMARK_RADII = {0.5, 1.0, 2.5, 5.0};
constexpr size_t REPEATS = 5;
constexpr size_t NUM_SEARCHES = 100;
constexpr bool CHECK_RESULTS = true;

template <typename T>
void checkVectorMemory(std::vector<T> vec) {
    std::cout << "Size in memory: " << (sizeof(std::vector<T>) + (sizeof(T) * vec.size())) / (1024.0 * 1024.0) << "MB" << std::endl;

    void* data = vec.data();
    // Check if the data is aligned to cache liens
    constexpr std::size_t CACHE_LINE_SIZE = std::hardware_destructive_interference_size;
    if (reinterpret_cast<std::uintptr_t>(data) % CACHE_LINE_SIZE == 0) {
        std::cout << "The vector's data is aligned to a cache line!" << std::endl;
    } else {
        std::cout << "The vector's data is NOT aligned to a cache line." << std::endl;
    }
}

template <template <typename, typename> class Octree_t, typename Point_t, typename Encoder_t>
std::shared_ptr<ResultSet<Point_t>> buildAndRunBenchmark(std::ofstream &outputFile, std::vector<Point_t>& points,
  std::shared_ptr<const SearchSet> searchSet, std::string comment = "") {
  OctreeBenchmark<Octree_t, Point_t, Encoder_t> ob(points, NUM_SEARCHES, searchSet, outputFile, CHECK_RESULTS, comment);
  OctreeBenchmark<Octree_t, Point_t, Encoder_t>::runFullBenchmark(ob, BENCHMARK_RADII, REPEATS, NUM_SEARCHES);
  return ob.getResultSet();
}

template <template <typename, typename> class Octree_t, typename Point_t, typename Encoder_t>
std::shared_ptr<ResultSet<Point_t>> buildAndRunAlgoComparisonBenchmark(std::ofstream &outputFile, std::vector<Point_t>& points,
  std::shared_ptr<const SearchSet> searchSet, std::string comment = "") {
  OctreeBenchmark<Octree_t, Point_t, Encoder_t> ob(points, NUM_SEARCHES, searchSet, outputFile, CHECK_RESULTS, comment);
  OctreeBenchmark<Octree_t, Point_t, Encoder_t>::runAlgoComparisonBenchmark(ob, BENCHMARK_RADII, REPEATS, NUM_SEARCHES);
  return ob.getResultSet();
}

template <PointType Point_t, typename Encoder_t>
void octreeComparisonBenchmark(std::ofstream &outputFile) {
  // TODO: maybe a better idea is to choose radii based on point cloud density
  TimeWatcher tw;
  tw.start();
  auto points = readPointCloud<Point_t>(mainOptions.inputFile);
  tw.stop();

  std::cout << "Number of read points: " << points.size() << "\n";
  std::cout << "Time to read points: " << tw.getElapsedDecimalSeconds()
            << " seconds\n";
  checkVectorMemory(points);

  // Generate a shared search set for each benchmark execution
  std::shared_ptr<const SearchSet> searchSet = std::make_shared<const SearchSet>(NUM_SEARCHES, points);

  auto resultsPointer = buildAndRunBenchmark<Octree, Point_t, PointEncoding::NoEncoder>(outputFile, points, searchSet);
  auto resultsLinear = buildAndRunBenchmark<LinearOctree, Point_t, Encoder_t>(outputFile, points, searchSet);
  auto resultsPointerSorted = buildAndRunBenchmark<Octree, Point_t, Encoder_t>(outputFile, points, searchSet);

  // Check the results if needed
  if(CHECK_RESULTS) {
    resultsLinear->checkResults(resultsPointerSorted);
  }
}


// To test different implementations of the same methods (i.e. numNeighbors vs numNeighborsOld)
template <PointType Point_t, typename Encoder_t = PointEncoding::MortonEncoder64>
void algorithmComparisonBenchmark(std::ofstream &outputFile) {
  // TODO: maybe a better idea is to choose radii based on point cloud density
  TimeWatcher tw;
  tw.start();
  auto points = readPointCloud<Point_t>(mainOptions.inputFile);
  tw.stop();

  std::cout << "Number of read points: " << points.size() << "\n";
  std::cout << "Time to read points: " << tw.getElapsedDecimalSeconds()
            << " seconds\n";
  checkVectorMemory(points);

  // Generate a shared search set for each benchmark execution
  std::shared_ptr<const SearchSet> searchSet = std::make_shared<const SearchSet>(NUM_SEARCHES, points);

  auto results = buildAndRunAlgoComparisonBenchmark<LinearOctree, Point_t, Encoder_t>(outputFile, points, searchSet);

  // Check the results if needed
  if(CHECK_RESULTS) {
    results->checkResultsAlgo();
  }
}

void printKey(uint64_t key) {
  for(int i=20; i>=0; i--) {
    std::cout << std::bitset<3>(key >> (3*i)) << " ";
  }
  std::cout << std::endl;
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

  octreeComparisonBenchmark<Lpoint64, PointEncoding::HilbertEncoder64>(outputFile);
  octreeComparisonBenchmark<Lpoint64, PointEncoding::MortonEncoder64>(outputFile);

  // Grid 16x16x16 with some modifications, save its encoding to file for visualization
  // constexpr int GRID_SIZE = 2;
  // std::vector<Point_t> points;
  // std::vector<std::pair<Encoder_t::key_t, Point_t>> codes;
  // for(int i = -GRID_SIZE; i<=GRID_SIZE; i++) {
  // for(int j = -GRID_SIZE; j<=GRID_SIZE; j++) {
  // for(int k = -GRID_SIZE; k<=GRID_SIZE; k++) {
  //   Point_t p = Point_t(i*GRID_SIZE*GRID_SIZE + j*GRID_SIZE + k, k, j, i);
  //   points.push_back(p);
  // }}}
  // Vector radii;
  // Point p = mbb(points, radii);
  // Box bbox = Box(p, radii);
  // for(int i = 0; i<points.size(); i++){
  //   Encoder_t::coords_t x, y, z;
  //   PointEncoding::getAnchorCoords<Encoder_t>(points[i], bbox, x, y, z);
  //   codes.push_back({Encoder_t::encode(x, y, z), points[i]});
  // }

  // std::sort(codes.begin(), codes.end(),
  //   [](const auto& a, const auto& b) {
  //       return a.first < b.first;  // Compare only the codes
  //   }
  // );

  // std::ofstream pointsFile("points.txt",  std::ios::out);
  // for(int i = 0; i<codes.size();i++)
  //   pointsFile << codes[i].second << std::endl;
  // pointsFile.close();

  // auto searchSet = std::make_shared<const SearchSet>(NUM_SEARCHES, points);

  // auto pointerOct = Octree<Point_t, PointEncoding::NoEncoder>(points);
  // auto resPointer = pointerOct.numNeighbors(kernelFactory<Kernel_t::sphere>(searchSet.get()->searchPoints[4], BENCHMARK_RADII[0]));

  // auto octHilbert = LinearOctree<Point_t, PointEncoding::HilbertEncoder64>(points);
  // auto resHilbert = octHilbert.numNeighbors(kernelFactory<Kernel_t::sphere>(searchSet.get()->searchPoints[4], BENCHMARK_RADII[0]));
  // std::ofstream hilbertFile("hilbert.txt", std::ios::out);
  // octHilbert.writeOctree(hilbertFile);
  // hilbertFile.close();


  
  // auto octMorton = LinearOctree<Point_t, PointEncoding::MortonEncoder64>(points);
  // auto resMorton = octMorton.numNeighbors(kernelFactory<Kernel_t::sphere>(searchSet.get()->searchPoints[4], BENCHMARK_RADII[0]));
  // std::ofstream mortonFile("morton.txt", std::ios::out);
  // octMorton.writeOctree(mortonFile);
  // mortonFile.close();
  // std::cout << "Searched for point: " << searchSet.get()->searchPoints[4] << " with radius " << BENCHMARK_RADII[0] << std::endl;
  // std::cout << "Hilbert: " << resHilbert << ", Morton: " << resMorton << ", Pointer: " << resPointer << std::endl;

  // uint64_t key = 05;
  // key <<= 20*3;
  // uint32_t x, y, z;
  // std::cout << "key ";
  // printKey(key);
  // PointEncoding::HilbertEncoder64::decode(key, x, y, z);
  // std::cout << "hilbert coords:\n" << std::bitset<32>(x) << "\n" << std::bitset<32>(y) << "\n" << std::bitset<32>(z) << std::endl;
  // std::cout << ((1 << 21) - x) << std::endl;
  // PointEncoding::MortonEncoder64::decode(key, x, y, z);
  // std::cout << "morton coords:\n" << std::bitset<32>(x) << "\n" << std::bitset<32>(y) << "\n" << std::bitset<32>(z) << std::endl;

  return EXIT_SUCCESS;
}
