#ifndef CPP_MAIN_OPTIONS_HPP
#define CPP_MAIN_OPTIONS_HPP

#include <filesystem>
#include <getopt.h>
#include <iostream>
#include <vector>
#include "omp.h"
#include <set>
#include "NeighborKernels/KernelFactory.hpp"

namespace fs = std::filesystem;

enum BenchmarkMode { SEARCH, COMPARE, POINT_TYPE, APPROX, PARALLEL, LOG_OCTREE };

class main_options
{
public:
	// Files & paths
	fs::path inputFile{};
	fs::path outputDirName{"out"};
	std::string inputFileName{};

	// Benchmark parameters
	std::vector<float> benchmarkRadii{2.5, 5.0, 7.5, 10.0};
	size_t repeats{2};
	size_t numSearches{10000};
	bool sequentialSearches{false};
	bool searchAll{false};
	bool checkResults{false};
	bool useWarmup{true};
	BenchmarkMode benchmarkMode{SEARCH};
	std::vector<double> approximateTolerances{50.0};
	std::vector<int> numThreads{omp_get_max_threads()};
	std::set<Kernel_t> kernels{Kernel_t::sphere, Kernel_t::circle, Kernel_t::cube, Kernel_t::square};
};

extern main_options mainOptions;

enum LongOptions : int
{
	HELP = 0,
	RADII,
	REPEATS,
	SEARCHES,
	CHECK,
	BENCHMARK,
	NO_WARMUP,
	APPROXIMATE_TOLERANCES,
	NUM_THREADS,
	SEQUENTIAL_SEARCH_SET,
	KERNELS
};

// Define short options
const char* const short_opts = "h:i:o:r:t:s:cb:";

// Define long options
const option long_opts[] = {
	{ "help", no_argument, nullptr, LongOptions::HELP },
	{ "radii", required_argument, nullptr, LongOptions::RADII },
	{ "repeats", required_argument, nullptr, LongOptions::REPEATS },
	{ "searches", required_argument, nullptr, LongOptions::SEARCHES },
	{ "check", no_argument, nullptr, LongOptions::CHECK },
	{ "benchmark", required_argument, nullptr, LongOptions::BENCHMARK },
	{ "no-warmup", no_argument, nullptr, LongOptions::NO_WARMUP },
	{ "approx-tol", required_argument, nullptr, LongOptions::APPROXIMATE_TOLERANCES },
	{ "num-threads", required_argument, nullptr, LongOptions::NUM_THREADS},
	{ "sequential", no_argument, nullptr, LongOptions::SEQUENTIAL_SEARCH_SET},
	{ "kernels", required_argument, nullptr, LongOptions::KERNELS},
	{ nullptr, 0, nullptr, 0 }
};

void printHelp();
void setDefaults();
std::set<Kernel_t> parseKernelOptions(const std::string& kernelStr);
std::string getKernelListString();
void processArgs(int argc, char** argv);

#endif // CPP_MAIN_OPTIONS_HPP
