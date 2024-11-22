#include "TimeWatcher.hpp"
#include "handlers.hpp"
#include "main_options.hpp"
#include "octree.hpp"
#include "octree_linear.hpp"
#include "octree_linear_old.hpp"
#include <filesystem> // Only C++17 and beyond
#include <iomanip>
#include <iostream>
#include "benchmarking.hpp"
#include <octree_benchmark.hpp>
#include <random>
#include "NeighborKernels/KernelFactory.hpp"
namespace fs = std::filesystem;

int main(int argc, char *argv[]) {
  setDefaults();
  processArgs(argc, argv);
  std::cout << "Size of Point: " << sizeof(Point) << " bytes\n";
  std::cout << "Size of Lpoint: " << sizeof(Lpoint) << " bytes\n";

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

  TimeWatcher tw;

  tw.start();
  std::vector<Lpoint> points = readPointCloud(mainOptions.inputFile);
  tw.stop();

  std::cout << "Number of read points: " << points.size() << "\n";
  std::cout << "Size of the points array: " << points.size() * sizeof(Lpoint) / (1024.0 * 1024.0) << " MB \n";
  std::cout << "Time to read points: " << tw.getElapsedDecimalSeconds()
            << " seconds\n";
  
  // Benchmark paramters
  const size_t benchmarkSize = 50;
  // TODO: maybe a better idea is to choose radii based on point cloud density
  const std::vector<float> benchmarkRadii = {0.5, 1.0, 2.5, 3.5, 5.0};
  const size_t repeats = 5;
  // TODO: For now we copy the points for the linear octree, since they get sorted and so its better if they are separate
  // Should change this to first execute all benchmarks in pointer octree, then sort, then execute them in linear octree
  // This way we don't waste memory and can run bigger examples.
  std::vector<Lpoint> lOctreePoints(points);
  OctreeBenchmark ob(points, lOctreePoints, benchmarkSize);
  

  std::cout << "Running benchmarks with search radii {";
  for(int i = 0; i<benchmarkRadii.size(); i++) {
    std::cout << benchmarkRadii[i];
    if(i != benchmarkRadii.size()-1) {
      std::cout << ", ";
    }
  }
  std::cout << "} with " << repeats << " repeats each over a set of " << benchmarkSize << " random center points." << std::endl;

  for(int i = 0; i<benchmarkRadii.size(); i++) {
    ob.benchmarkSearchNeigh<Kernel_t::sphere>(repeats, benchmarkRadii[i]);
    ob.benchmarkSearchNeigh<Kernel_t::circle>(repeats, benchmarkRadii[i]);
    ob.benchmarkSearchNeigh<Kernel_t::cube>(repeats, benchmarkRadii[i]);
    ob.benchmarkSearchNeigh<Kernel_t::square>(repeats, benchmarkRadii[i]);
    std::cout << "Benchmark search neighbors with radii " << benchmarkRadii[i] << " completed" << std::endl;
  }

  for(int i = 0; i<benchmarkRadii.size(); i++) {
    ob.benchmarkNumNeigh<Kernel_t::sphere>(repeats, benchmarkRadii[i]);
    ob.benchmarkNumNeigh<Kernel_t::circle>(repeats, benchmarkRadii[i]);
    ob.benchmarkNumNeigh<Kernel_t::cube>(repeats, benchmarkRadii[i]);
    ob.benchmarkNumNeigh<Kernel_t::square>(repeats, benchmarkRadii[i]);
    std::cout << "Benchmark number of neighbors with radii " << benchmarkRadii[i] << " completed" << std::endl;
  }

  // TODO: fix the implementation of this other two benchmarks
  // ob.benchmarkKNN(5);
  // ob.benchmarkRingSearchNeigh(5);

  return EXIT_SUCCESS;
}
