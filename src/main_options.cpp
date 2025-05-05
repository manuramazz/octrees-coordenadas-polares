#include "main_options.hpp"
#include <sstream>
#include <cstdlib>
#include <set>
#include "NeighborKernels/KernelFactory.hpp"
#include <unordered_map>

main_options mainOptions{};

void printHelp()
{
	std::cout
		<< "-h, --help: Show this message\n"
		   "-i: Path to input file\n"
		   "-o: Path to output file\n"
		   "-r, --radii: Benchmark radii (comma-separated, e.g., '2.5,5.0,7.5')\n"
		   "-s, --searches: Number of searches\n"
		   "-t, --repeats: Number of repeats\n"
		   "-c, --check: Enable result checking\n"
		   "-b, --benchmark: Benchmark to run:\n\t" 
		   "'srch' for comparison between pointer and linear octree, and between point encodings (default),\n\t"
		   "'comp' for comparison of different linear octree search methods,\n\t"
		   "'pt' for point type comparison,\n\t" 
		   "'approx' for approximate searches comparison\n\t"
		   "'struct' for comparing performance on raw vector returned vs structure with octants and extra points\n\t"
		   "'parallel' for a parallelism scalability benchmark across a number of threads spawned (passed with --num-threads) and with multiple OpenMP schedules\n\t"
		   "'log' for logging the entire linear octree built, use for debugging\n"
		   "--no-warmup: Disable warmup phase\n"
		   "--approx-tol: For specifying tolerance percentage in approximate searches (e.g. 80.0 = 80% tolerance on kernel size), format is list of doubles in format e.g. '10.0,50.0,100.0'\n"
		   "--num-threads: List of number of threads to use in the parallelism scalability benchmark (e.g. 1,2,4,8,16,32)\n"
		   "--sequential: Make the search set sequential instead of random\n"
		   "--kernels: Specify which kernels to use (comma-separated, e.g., 'sphere,cube' or 'all')\n"
		   "--max-leaf: Maximum numbers of points in an octree leaf (default = 128)\n";
	exit(1);
}

template <typename T>
std::vector<T> readVectorArg(const std::string& vStr)
{
	std::vector<T> v;
	std::stringstream ss(vStr);
	std::string token;

	while (std::getline(ss, token, ',')) {
		v.push_back(std::stof(token));
	}

	return v;
}

std::set<Kernel_t> parseKernelOptions(const std::string& kernelStr) {
    static const std::unordered_map<std::string, Kernel_t> kernelMap = {
        {"sphere", Kernel_t::sphere},
        {"circle", Kernel_t::circle},
        {"cube", Kernel_t::cube},
        {"square", Kernel_t::square}
    };

    std::set<Kernel_t> selectedKernels;

    if (kernelStr == "all") {
        for (const auto& [key, value] : kernelMap) {
            selectedKernels.insert(value);
        }
    } else {
        std::stringstream ss(kernelStr);
        std::string token;
        while (std::getline(ss, token, ',')) {
            auto it = kernelMap.find(token);
            if (it != kernelMap.end()) {
                selectedKernels.insert(it->second);
            } else {
                std::cerr << "Warning: Unknown kernel '" << token << "' ignored.\n";
            }
        }
    }

    return selectedKernels;
}

std::string getKernelListString() {
    std::ostringstream oss;
    auto it = mainOptions.kernels.begin();
    for (; it != mainOptions.kernels.end(); ++it) {
        oss << kernelToString(*it);
        if (std::next(it) != mainOptions.kernels.end()) {
            oss << ", ";
        }
    }
    return oss.str();
}


void processArgs(int argc, char** argv)
{
	while (true)
	{
		const auto opt = getopt_long(argc, argv, short_opts, long_opts, nullptr);

		if (opt == -1) { break; } // No more options to process

		switch (opt)
		{
			case 'h':
			case LongOptions::HELP:
				printHelp();
				break;
			case 'i':
				mainOptions.inputFile = fs::path(std::string(optarg));
				mainOptions.inputFileName = mainOptions.inputFile.stem().string();
				break;
			case 'o':
				mainOptions.outputDirName = fs::path(std::string(optarg));
				break;
			case 'r':
			case LongOptions::RADII:
				mainOptions.benchmarkRadii = readVectorArg<float>(std::string(optarg));
				break;
			case 't':
			case LongOptions::REPEATS:
				mainOptions.repeats = std::stoul(std::string(optarg));
				break;
			case 's':
			case LongOptions::SEARCHES:
				if (std::string(optarg) == "all") {
					mainOptions.searchAll = true;
					mainOptions.numSearches = 0;
				} else {
					mainOptions.numSearches = std::stoul(std::string(optarg));
				}
				break;
			case 'c':
			case LongOptions::CHECK:
				mainOptions.checkResults = true;
				break;
			case 'b':
			case LongOptions::BENCHMARK:
				if (std::string(optarg) == "srch") {
					mainOptions.benchmarkMode = SEARCH;
				} else if (std::string(optarg) == "comp") {
					mainOptions.benchmarkMode = COMPARE;
				} else if(std::string(optarg) == "pt") {
					mainOptions.benchmarkMode = POINT_TYPE;
				} else if(std::string(optarg) == "log") {
					mainOptions.benchmarkMode = LOG_OCTREE;
				} else if(std::string(optarg) == "approx") {
					mainOptions.benchmarkMode = APPROX;
				} else if(std::string(optarg) == "parallel") {
					mainOptions.benchmarkMode = PARALLEL;
				} else {
					std::cerr << "Invalid benchmark mode: " << optarg << "\n";
					printHelp();
				}
				break;
			case LongOptions::NO_WARMUP:
				mainOptions.useWarmup = false;
				break;
			case LongOptions::APPROXIMATE_TOLERANCES:
				mainOptions.approximateTolerances = readVectorArg<double>(std::string(optarg));
				break;
			case LongOptions::NUM_THREADS:
				mainOptions.numThreads = readVectorArg<int>(std::string(optarg));
				break;
			case LongOptions::SEQUENTIAL_SEARCH_SET:
				mainOptions.sequentialSearches = true;
				break;
			case LongOptions::KERNELS:
				mainOptions.kernels = parseKernelOptions(std::string(optarg));
				break;
			case LongOptions::MAX_POINTS_LEAF:
				mainOptions.maxPointsLeaf = std::stoul(std::string(optarg));
				break;
			default:
				printHelp();
				break;
		}
	}
}
