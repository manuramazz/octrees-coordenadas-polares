//
// Created by miguelyermo on 11/3/20.
//

#include "main_options.hpp"

#include <cmath>
#include <filesystem>
#include <vector>

namespace fs = std::filesystem;

main_options mainOptions{};

void printHelp()
{
	std::cout
	    << "-h: Show this message\n"
	       "-i: Path to input file\n"
	       "-o: Path to output file\n"
	       "-r: Benchmark radii (comma-separated, e.g., '2.5,5.0,7.5')\n"
	       "-s: Number of searches\n"
	       "-c: Enable result checking\n"
		   "-b: Benchmark to run: -b srch for search (default), -b comp for impl. comparison, -b seq for sequential vs shuffled points\n";
	exit(1);
}


void setDefaults()
{
	if (mainOptions.outputDirName.empty()) { mainOptions.outputDirName = "out"; }
	mainOptions.benchmarkMode = BenchmarkMode::SEARCH;
}

std::vector<float> parseRadii(const std::string& radiiStr)
{
	std::vector<float> radii;
	std::stringstream ss(radiiStr);
	std::string token;

	while (std::getline(ss, token, ',')) {
		radii.push_back(std::stof(token));
	}

	return radii;
}

void processArgs(int argc, char** argv)
{
	while (true)
	{
		const auto opt = getopt_long(argc, argv, short_opts, long_opts, nullptr);

		if (-1 == opt) { break; }

		switch (opt)
		{
				// Short Options
			case 'h': {
				printHelp();
				break;
			}
			case 'i': {
				mainOptions.inputFile = fs::path(std::string(optarg));
				std::cout << "Read file set to: " << mainOptions.inputFile << "\n";
				mainOptions.inputFileName = mainOptions.inputFile.stem().string();
				break;
			}
			case 'o': {
				mainOptions.outputDirName = fs::path(std::string(optarg));
				std::cout << "Output path set to: " << mainOptions.outputDirName << "\n";
				break;
			}
			case 'r': {
				mainOptions.benchmarkRadii = parseRadii(std::string(optarg));
				std::cout << "Benchmark radii set to: ";
				for (const auto& r : mainOptions.benchmarkRadii) std::cout << r << " ";
				std::cout << "\n";
				break;
			}
			case 's': {
				mainOptions.numSearches = std::stoul(std::string(optarg));
				std::cout << "Number of searches set to: " << mainOptions.numSearches << "\n";
				break;
			}
			case 'c': {
				mainOptions.checkResults = true;
				std::cout << "Result checking enabled\n";
				break;
			}
			case 'b': {
				std::string mode = std::string(optarg);
				if (mode == "srch") {
					mainOptions.benchmarkMode = BenchmarkMode::SEARCH;
					std::cout << "Benchmark mode set to: Linear Octree vs Octree in neighSearch and numNeighSearch\n";
				} else if (mode == "comp") {
					mainOptions.benchmarkMode = BenchmarkMode::COMPARE;
					std::cout << "Benchmark mode set to: Comparisons of implementations of neighSearch and numNeighSearch inside LinearOctree\n";
				} else if (mode == "seq") {
					mainOptions.benchmarkMode = BenchmarkMode::SEQUENTIAL;
					std::cout << "Benchmark mode set to: Sequential vs Shuffled search sets performance comparison\n";
				} else {
					std::cerr << "Invalid benchmark mode: " << mode << "\n";
					printHelp();
				}
				break;
			}
			case '?': // Unrecognized option
			default:
				printHelp();
				break;
		}
	}
}