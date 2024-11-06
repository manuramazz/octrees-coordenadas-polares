#include "TimeWatcher.hpp"
#include "benchmarking.hpp"
#include "handlers.hpp"
#include "main_options.hpp"
#include "octree.hpp"
#include "octree_v2.hpp"
#include "octree_linear.hpp"
#include "octree_pointer.hpp"
#include <filesystem> // Only C++17 and beyond
#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

namespace fs = std::filesystem;

void read_cache_size(const std::string& path, long& size) {
    std::ifstream file(path);
    if (file.is_open()) {
        std::string line;
        std::getline(file, line);
        size = std::stol(line);
    } else {
        size = 0;
    }
}

void get_cache_info(long& l1_size, long& l2_size, long& l3_size, long& l4_size, long& cache_line_size) {
    cache_line_size = sysconf(_SC_LEVEL1_DCACHE_LINESIZE); // usually 64 bytes is the line size

    read_cache_size("/sys/devices/system/cpu/cpu0/cache/index0/size", l1_size); // L1 cache size
    read_cache_size("/sys/devices/system/cpu/cpu0/cache/index1/size", l2_size); // L2 cache size
    read_cache_size("/sys/devices/system/cpu/cpu0/cache/index2/size", l3_size); // L3 cache size
    read_cache_size("/sys/devices/system/cpu/cpu0/cache/index3/size", l4_size); // L4 cache size
}

void get_struct_info() {
    std::cout << "Size of Lpoint: " << sizeof(Lpoint) << " bytes\n";
    std::cout << "Size of Octree: " << sizeof(Octree) << " bytes\n";
    std::cout << "Size of PointerOctree: " << sizeof(PointerOctree) << " bytes\n";
    std::cout << "Size of LinearOctree: " << sizeof(LinearOctree) << " bytes\n";
}

int main(int argc, char *argv[]) {
  long l1_size, l2_size, l3_size, l4_size, cache_line_size;
  get_cache_info(l1_size, l2_size, l3_size, l4_size, cache_line_size);
  std::cout << "Cache information: \n\t L1 size: " 
            << l1_size << " KB\n\t L2 size: " 
            << l2_size << " KB\n\t L3 size: "
            << l3_size << " KB\n\t L4 size: "
            << l4_size << " KB\n\t Cache line size: " 
            << cache_line_size << " bytes " 
            << std::endl;

  get_struct_info();

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

  TimeWatcher tw;

  // tw.start();
  std::vector<Lpoint> points = readPointCloud(mainOptions.inputFile);
  // tw.stop();
  // std::cout << "Number of read points: " << points.size() << "\n";
  // std::cout << "Time to read points: " << tw.getElapsedDecimalSeconds()
  //           << " seconds\n";

  // // Global Octree Creation
  // std::cout << "Building global octree..." << std::endl;
  // tw.start();
  // Octree gOctree(points);
  // tw.stop();
  // std::cout << "Time to build global octree: " << tw.getElapsedDecimalSeconds()
  //           << " seconds\n";
  // std::ofstream gOctreeStream(mainOptions.outputDirName / "global_octree.txt");
  // gOctree.writeOctree(gOctreeStream, 0);


  // Global Pointer Octree Creation
  std::cout << "Building global (pointer) octree..." << std::endl;
  tw.start();
  PointerOctree pOctree(points);
  tw.stop();
  std::cout << "Time to build global (pointer) octree: " << tw.getElapsedDecimalSeconds()
            << " seconds\n";
  std::ofstream pOctreeStream(mainOptions.outputDirName / "pointer_octree.txt");
  pOctree.writeOctree(pOctreeStream, 0);

  std::cout << "Building global (linear) octree..." << std::endl;

  tw.start();
  LinearOctree lOctree(points);
  tw.stop();
  std::cout << "Time to build global (linear) octree: " << tw.getElapsedDecimalSeconds()
            << " seconds\n";

  benchmarking::runBenchmark(points, gOctree, pOctree, lOctree);

  return EXIT_SUCCESS;
}
