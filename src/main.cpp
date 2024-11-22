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
  // {0, 1, ..., 15}^3 testing grid
  // std::vector<Lpoint> points;
  // points.reserve(16*16*16);
  // for(int i = 0; i<16;i++) {
  //     for(int j = 0; j<16;j++) {
  //       for(int k = 0; k<16; k++) {
  //           points.push_back(Lpoint(Point(i*1.0, j*1.0, k*1.0)));
  //       }
  //   }  
  // }
  tw.stop();

  std::cout << "Number of read points: " << points.size() << "\n";
  std::cout << "Time to read points: " << tw.getElapsedDecimalSeconds()
            << " seconds\n";

  // Global Octree Creation
  // std::cout << "Building global octree..." << std::endl;
  // tw.start();
  // Octree gOctree(points);
  // tw.stop();
  // std::cout << "Time to build global octree: " << tw.getElapsedDecimalSeconds()
  //           << " seconds\n";
  // std::ofstream gOctreeStream(mainOptions.outputDirName / "global_octree.txt");
  // gOctree.writeOctree(gOctreeStream, 0);

  // Copy of the points for the linear octree
  const size_t benchmarkSize = 10000;
  std::vector<Lpoint> lOctreePoints(points);
  OctreeBenchmark ob(points, lOctreePoints, benchmarkSize);
  
  const std::vector<float> benchmarkRadii = {0.5, 1.0, 2.5, 5.0, 10.0};
  const size_t REPEATS = 5;
  std::cout << "Running benchmarks with sarch radii {";
  for(int i = 0; i<benchmarkRadii.size(); i++) {
    std::cout << benchmarkRadii[i];
    if(i != benchmarkRadii.size()-1) {
      std::cout << ", ";
    }
  }
  std::cout << "} with " << REPEATS << " repeats each over a set of " << benchmarkSize << "random center points." << std::endl;
  for(int i = 0; i<benchmarkRadii.size(); i++) {
    ob.benchmarkSearchNeigh<Kernel_t::sphere>(5, benchmarkRadii[i]);
    ob.benchmarkSearchNeigh<Kernel_t::circle>(5, benchmarkRadii[i]);
    ob.benchmarkSearchNeigh<Kernel_t::cube>(5, benchmarkRadii[i]);
    ob.benchmarkSearchNeigh<Kernel_t::square>(5, benchmarkRadii[i]);
    std::cout << "Benchmark search neighbors with radii " << benchmarkRadii[i] << " completed" << std::endl;
  }

  for(int i = 0; i<benchmarkRadii.size(); i++) {
    ob.benchmarkNumNeigh<Kernel_t::sphere>(5, benchmarkRadii[i]);
    ob.benchmarkNumNeigh<Kernel_t::circle>(5, benchmarkRadii[i]);
    ob.benchmarkNumNeigh<Kernel_t::cube>(5, benchmarkRadii[i]);
    ob.benchmarkNumNeigh<Kernel_t::square>(5, benchmarkRadii[i]);
    std::cout << "Benchmark number of neighbors with radii " << benchmarkRadii[i] << " completed" << std::endl;
  }

  // ob.benchmarkKNN(5);
  // ob.benchmarkRingSearchNeigh(5);

  return EXIT_SUCCESS;
}
