#include <filesystem>
#include <iomanip>
#include <iostream>
#include <random>
#include <optional>
#include "util.hpp"
#include "type_names.hpp"
#include "TimeWatcher.hpp"
#include "handlers.hpp"
#include "main_options.hpp"
#include "benchmarking.hpp"
#include "octree_benchmark.hpp"
#include "octree.hpp"
#include "linear_octree.hpp"
#include "NeighborKernels/KernelFactory.hpp"
#include "Geometry/point.hpp"
#include "Geometry/Lpoint.hpp"
#include "Geometry/Lpoint64.hpp"
#include "Geometry/PointMetadata.hpp"
#include "PointEncoding/morton_encoder.hpp"
#include "PointEncoding/hilbert_encoder.hpp"
#include "result_checking.hpp"
#include "omp.h"

namespace fs = std::filesystem;

/**
 * @brief Benchmark neighSearch and numNeighSearch for a given octree configuration (point type + encoder).
 * Compares LinearOctree and PointerOctree. If passed PointEncoding::NoEncoder, only PointerOctree is used.
 */
template <PointType Point_t, typename Encoder_t>
void searchBenchmark(std::ofstream &outputFile) {
  auto pointMetaPair = readPointsWithMetadata<Point_t>(mainOptions.inputFile);
  std::vector<Point_t> points = std::move(pointMetaPair.first);
  std::optional<std::vector<PointMetadata>> metadata = std::move(pointMetaPair.second);
  if constexpr (std::is_same_v<Encoder_t, PointEncoding::NoEncoder>) {
    const SearchSet<Point_t> searchSet = SearchSet<Point_t>(mainOptions.numSearches, points);
    // Only do pointer octree, since we are not encoding the points
    OctreeBenchmark<Octree, Point_t, Encoder_t> obPointer(points, searchSet, outputFile, metadata, false);
    obPointer.searchBench();
  } else {
    // Sort the point cloud
    PointEncoding::sortPoints<Encoder_t, Point_t>(points, metadata);
    // Create the searchSet (WARMING: this should be done after sorting since it indexes points!)

    const SearchSet<Point_t> searchSet = SearchSet<Point_t>(mainOptions.numSearches, points);
    // Do both linear (which encodes and sorts the points) and pointer octree after it
    OctreeBenchmark<LinearOctree, Point_t, Encoder_t> obLinear(points, searchSet, outputFile, metadata, true);
    obLinear.searchBench();
    obLinear.deleteOctree();
    OctreeBenchmark<Octree, Point_t, Encoder_t> obPointer(points, searchSet, outputFile, metadata, true);
    obPointer.searchBench();
    obPointer.deleteOctree();
    if(mainOptions.checkResults) {
        ResultChecking::checkResultsLinearVsPointer(obLinear.getResultSet(), obPointer.getResultSet());
    }
  }
}

/**
 * @brief Benchmark oldNeighSearch vs neighSearch and oldNumNeighSearch vs numNeighSearch for a given octree configuration (point type + encoder),
 * in order to see the performance improvement of the new implementation.
 * 
 * Only uses LinearOctree, since that's where the implementation is being improved, so don't pass PointEncoding::NoEncoder!
 */
template <PointType Point_t, typename Encoder_t>
void algoCompBenchmark(std::ofstream &outputFile) {
  auto pointMetaPair = readPointsWithMetadata<Point_t>(mainOptions.inputFile);
  std::vector<Point_t> points = std::move(pointMetaPair.first);
  std::optional<std::vector<PointMetadata>> metadata = std::move(pointMetaPair.second);
  // Sort the point cloud
  PointEncoding::sortPoints<Encoder_t, Point_t>(points, metadata);
  
  // Create the searchSet (WARMING: this should be done after sorting since it indexes points!)
  const SearchSet<Point_t> searchSet = SearchSet<Point_t>(mainOptions.numSearches, points);

  OctreeBenchmark<LinearOctree, Point_t, Encoder_t> ob(points, searchSet, outputFile, metadata, true);
  ob.searchImplComparisonBench();

  if(mainOptions.checkResults) {
    ResultChecking::checkResultsAlgoComp(ob.getResultSet());
  }
}

/**
 * @brief Runs the approximate searches benchmark with the configuration given in mainOptions
 * 
 * Only uses LinearOctree, so don't pass PointEncoding::NoEncoder!
 */
template <PointType Point_t, typename Encoder_t>
void approxSearchBenchmark(std::ofstream &outputFile) {
  auto pointMetaPair = readPointsWithMetadata<Point_t>(mainOptions.inputFile);
  std::vector<Point_t> points = std::move(pointMetaPair.first);
  std::optional<std::vector<PointMetadata>> metadata = std::move(pointMetaPair.second);
  // Sort the point cloud
  PointEncoding::sortPoints<Encoder_t, Point_t>(points, metadata);
  
  // Create the searchSet (WARMING: this should be done after sorting since it indexes points!)
  const SearchSet<Point_t> searchSet = SearchSet<Point_t>(mainOptions.numSearches, points);

  OctreeBenchmark<LinearOctree, Point_t, Encoder_t> ob(points, searchSet, outputFile, metadata, true);
  ob.approxSearchBench();

  if(mainOptions.checkResults) {
    ResultChecking::checkResultsApproxSearches(ob.getResultSet());
  }
}

/**
 * @brief Runs the parallel execution benchmark.
 * 
 * Only uses LinearOctree, so don't pass PointEncoding::NoEncoder!
 */
template <template <typename, typename> class Octree_t, PointType Point_t, typename Encoder_t>
void parallelScalabilityBenchmark(std::ofstream &outputFile) {
  auto pointMetaPair = readPointsWithMetadata<Point_t>(mainOptions.inputFile);
  std::vector<Point_t> points = std::move(pointMetaPair.first);
  std::optional<std::vector<PointMetadata>> metadata = std::move(pointMetaPair.second);
  // Sort the point cloud
  if constexpr (! std::is_same_v<Encoder_t, PointEncoding::NoEncoder>) {
    PointEncoding::sortPoints<Encoder_t, Point_t>(points, metadata);
  }  
  // Create the searchSet (WARMING: this should be done after sorting since it indexes points!)
  const SearchSet<Point_t> searchSet = SearchSet<Point_t>(mainOptions.numSearches, points);

  OctreeBenchmark<Octree_t, Point_t, Encoder_t> ob(points, searchSet, outputFile, metadata, true);
  ob.parallelScalabilityBenchmark();
}

template <PointType Point_t>
std::vector<Point_t> generateGridCloud(size_t n) {
  std::vector<Point_t> points;
  points.reserve(n * n * n);
  for (size_t i = 0; i < n; i++)  
    for (size_t j = 0; j < n; j++)  
      for (size_t k = 0; k < n; k++)  
        points.push_back(Point_t(i * n * n + j * n + k, i, j, k));
  return points;
}

/**
 * @brief Constructs a linear octree and logs it to two separate files
 *  octree-structure.txt contains the structure of the octree
 *  encoded-points.csv contains the encodings of each point and its x,y,z coordinates
 * 
 * Keep in mind that those files (specially the second, since it will be on the same order of magnitude as the .las file) 
 * will be huge for big clouds, so only use this for small clouds (e.g. <5M points)
 */
template <PointType Point_t, typename Encoder_t>
void linearOctreeLog(std::ofstream &outputFile, bool useGridCloud = false, size_t gridCloudSize = 16) {
  TimeWatcher tw;
  tw.start();
  std::vector<Point_t> points;
  std::string cloudName;
  if(useGridCloud) {
    points = generateGridCloud<Point_t>(gridCloudSize);
    cloudName = "regular-grid-" + std::to_string(gridCloudSize);
  } else {
    points = readPointCloud<Point_t>(mainOptions.inputFile);
    cloudName = mainOptions.inputFileName;
  }
  tw.stop();
  pointCloudReadLog(points, tw, mainOptions.inputFile);
  std::string encoderTypename = PointEncoding::getShortEncoderName<Encoder_t>();
  
  // Open the files for outputting the octree structure and the encoded points
  // Dont put the timestamp on this file to not occupy a lot of memory if we forget to remove them after user
  std::string octreeOutputFilename = cloudName + "-" + encoderTypename + "-octree-structure.txt";
  std::string encodedPointsFilename = cloudName + "-" + encoderTypename + "-encoded-points.csv";
  std::filesystem::path octreeOutputPath = mainOptions.outputDirName / octreeOutputFilename;
  std::filesystem::path encodedPointsPath = mainOptions.outputDirName / encodedPointsFilename;
  std::ofstream octreeOutputFile(octreeOutputPath, std::ios::out);
  std::ofstream encodedPointsFile(encodedPointsPath, std::ios::out);
  if (!octreeOutputFile.is_open() || !encodedPointsFile.is_open()) {
    throw std::ios_base::failure(std::string("Failed to open octree log output files"));
  }
  std::optional<std::vector<PointMetadata>> metadata = std::nullopt;
  LinearOctree<Point_t, Encoder_t> linearOctree(points, metadata, true);
  std::cout << "OCTREE LOGGING MODE -- Outputting linear octree structure built from point cloud " << cloudName << " to " 
    << octreeOutputFilename << " and encoded points to " << encodedPointsFilename << std::endl;
  linearOctree.logOctree(octreeOutputFile, encodedPointsFile);
  octreeOutputFile.close();
  encodedPointsFile.close();
}

void approximateSearchLog(std::ofstream &outputFile) {
  auto pointMetaPair = readPointsWithMetadata<Lpoint64>(mainOptions.inputFile);
  std::vector<Lpoint64> points = std::move(pointMetaPair.first);
  std::optional<std::vector<PointMetadata>> metadata = std::move(pointMetaPair.second);

  auto lin_oct = LinearOctree<Lpoint64, PointEncoding::MortonEncoder3D>(points, metadata, true);
  std::array<float, 5> tolerances = {5.0, 10.0, 25.0, 50.0, 100.0};
  float radius = 3.0;
  outputFile << "tolerance,upper,x,y,z\n";
  auto points_exact = lin_oct.searchNeighborsStruct<Kernel_t::sphere>(points[1234], radius);
  for(const Point &p: points_exact) {
    outputFile << "0.0,exact," << p.getX() << "," << p.getY() << "," << p.getZ() << "\n";
  }
  for(float tol: tolerances) {
    auto points_upper = lin_oct.searchNeighborsApprox<Kernel_t::sphere>(points[1234], 3.0, tol, true);
    auto points_lower = lin_oct.searchNeighborsApprox<Kernel_t::sphere>(points[1234], 3.0, tol, false);
    for(const Point &p: points_upper) {
      outputFile << tol << ",upper," << p.getX() << "," << p.getY() << "," << p.getZ() << "\n";
    }
    for(const Point &p: points_lower) {
      outputFile << tol << ",lower," << p.getX() << "," << p.getY() << "," << p.getZ() << "\n";
    }
  }
}

template <typename Encoder_t>
void outputReorderings(std::ofstream &outputFilePoints, std::ofstream &outputFileOct) {
  // std::vector<Lpoint64> points = generateGridCloud<Lpoint64>(16);
  // std::optional<std::vector<PointMetadata>> metadata;

  auto pointMetaPair = readPointsWithMetadata<Lpoint64>(mainOptions.inputFile);
  std::vector<Lpoint64> points = std::move(pointMetaPair.first);
  std::optional<std::vector<PointMetadata>> metadata = std::move(pointMetaPair.second);

  if constexpr (! std::is_same_v<Encoder_t, PointEncoding::NoEncoder>) {
    PointEncoding::sortPoints<Encoder_t, Lpoint64>(points, metadata);
  }
  std::optional<std::vector<PointMetadata>> emptyMetadata;
  outputFilePoints << std::fixed << std::setprecision(3); 
  for(size_t i = 0; i<points.size(); i++) {
    outputFilePoints <<  points[i].getX() << "," << points[i].getY() << "," << points[i].getZ() << "\n";
  }
  
  // build octree and output it
  if constexpr (! std::is_same_v<Encoder_t, PointEncoding::NoEncoder>) {
    auto oct = LinearOctree<Lpoint64, Encoder_t>(points, emptyMetadata, true);
    oct.logOctreeBounds(outputFileOct, 6);
  }
}
int main(int argc, char *argv[]) {
  // Set default OpenMP schedule: dynamic and auto chunk size
  omp_set_schedule(omp_sched_dynamic, 0);
  processArgs(argc, argv);
  std::cout << std::fixed << std::setprecision(3); 
  fs::path inputFile = mainOptions.inputFile;
  std::string fileName = inputFile.stem();

  if (!mainOptions.outputDirName.empty()) {
    mainOptions.outputDirName = mainOptions.outputDirName / fileName;
  }

  // Handling Directories
  createDirectory(mainOptions.outputDirName);

  // Open the benchmark output file
  std::string csvFilename = mainOptions.inputFileName + "-" + getCurrentDate() + ".csv";
  std::filesystem::path csvPath = mainOptions.outputDirName / csvFilename;
  std::ofstream outputFile(csvPath, std::ios::app);
  if (!outputFile.is_open()) {
      throw std::ios_base::failure(std::string("Failed to open benchmark output file: ") + csvPath.string());
  }

  switch(mainOptions.benchmarkMode) {
    case BenchmarkMode::SEARCH:
      searchBenchmark<Point, PointEncoding::NoEncoder>(outputFile);
      searchBenchmark<Point, PointEncoding::MortonEncoder3D>(outputFile);
      searchBenchmark<Point, PointEncoding::HilbertEncoder3D>(outputFile);
    break;
    case BenchmarkMode::COMPARE:
      algoCompBenchmark<Point, PointEncoding::HilbertEncoder3D>(outputFile);
    break;
    case BenchmarkMode::POINT_TYPE:
      searchBenchmark<Point, PointEncoding::HilbertEncoder3D>(outputFile);
      searchBenchmark<Lpoint64, PointEncoding::HilbertEncoder3D>(outputFile);
      searchBenchmark<Lpoint, PointEncoding::HilbertEncoder3D>(outputFile);
    break;
    case BenchmarkMode::APPROX:
      approxSearchBenchmark<Point, PointEncoding::HilbertEncoder3D>(outputFile);
    break;
    case BenchmarkMode::PARALLEL:
      parallelScalabilityBenchmark<Octree, Point, PointEncoding::NoEncoder>(outputFile);
      //parallelScalabilityBenchmark<Octree, Point, PointEncoding::MortonEncoder3D>(outputFile);
      //parallelScalabilityBenchmark<LinearOctree, Point, PointEncoding::MortonEncoder3D>(outputFile);
      parallelScalabilityBenchmark<Octree, Point, PointEncoding::HilbertEncoder3D>(outputFile);
      parallelScalabilityBenchmark<LinearOctree, Point, PointEncoding::HilbertEncoder3D>(outputFile);
    break;
    case BenchmarkMode::LOG_OCTREE:
      std::filesystem::path unencodedPath = mainOptions.outputDirName / "output_unencoded.csv";
      std::filesystem::path mortonPath = mainOptions.outputDirName / "output_morton.csv";
      std::filesystem::path hilbertPath = mainOptions.outputDirName / "output_hilbert.csv";
      std::filesystem::path unencodedPathOct = mainOptions.outputDirName / "output_unencoded_oct.csv";
      std::filesystem::path mortonPathOct = mainOptions.outputDirName / "output_morton_oct.csv";
      std::filesystem::path hilbertPathOct = mainOptions.outputDirName / "output_hilbert_oct.csv";
      // Open files
      std::ofstream unencodedFile(unencodedPath, std::ios::app);
      std::ofstream mortonFile(mortonPath, std::ios::app);
      std::ofstream hilbertFile(hilbertPath, std::ios::app);
      std::ofstream unencodedFileOct(unencodedPathOct, std::ios::app);
      std::ofstream mortonFileOct(mortonPathOct, std::ios::app);
      std::ofstream hilbertFileOct(hilbertPathOct, std::ios::app);
      
      if (!unencodedFile.is_open() || !mortonFile.is_open() || !hilbertFile.is_open() || 
          !unencodedFileOct.is_open() || !mortonFileOct.is_open() || !hilbertFileOct.is_open()) {
          throw std::ios_base::failure("Failed to open output files");
      }
      
      std::cout << "Output files created successfully." << std::endl;
      outputReorderings<PointEncoding::NoEncoder>(unencodedFile, unencodedFileOct);  
      outputReorderings<PointEncoding::MortonEncoder3D>(mortonFile, mortonFileOct);  
      outputReorderings<PointEncoding::HilbertEncoder3D>(hilbertFile, hilbertFileOct);  
      // approximateSearchLog(outputFile);
      // linearOctreeLog<Lpoint64, PointEncoding::MortonEncoder3D>(outputFile);
      // linearOctreeLog<Lpoint64, PointEncoding::MortonEncoder3D>(outputFile, true, 16);
      // linearOctreeLog<Lpoint64, PointEncoding::HilbertEncoder3D>(outputFile);
      // linearOctreeLog<Lpoint64, PointEncoding::HilbertEncoder3D>(outputFile, true, 16);
  }
  return EXIT_SUCCESS;
}
