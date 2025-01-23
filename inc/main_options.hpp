#ifndef CPP_MAIN_OPTIONS_HPP
#define CPP_MAIN_OPTIONS_HPP

//
// Created by miguelyermo on 11/3/20.
//

/*
* FILENAME :  main_options.h  
* PROJECT  :  rule-based-classifier-cpp
*
* DESCRIPTION :
*
*
*
*
*
* AUTHOR :    Miguel Yermo        START DATE : 18:50 11/3/20
*
*/

#include <filesystem>
#include <getopt.h>
#include <iostream>
#include <vector>

namespace fs = std::filesystem;

enum BenchmarkMode {SEARCH, COMPARE, SEQUENTIAL};

class main_options
{
	public:
	// Files & paths
	fs::path      inputFile{};
	fs::path      outputDirName{};
	std::string   inputFileName{};

	// Benchmark parameters
	std::vector<float> benchmarkRadii{2.5, 5.0, 7.5, 10.0};
	size_t             repeats{5};
	size_t             numSearches{10000};
	bool               checkResults{false};
	BenchmarkMode      benchmarkMode{SEARCH};
};

extern main_options mainOptions;


enum LongOptions : int
{
	HELP = 0, // Help message
	RADII,
	REPEATS,
	SEARCHES,
	CHECK,
	BENCHMARK
};

// Define short options
const char* const short_opts = "h:i:o:r:s:cb:";

// Define long options
const option long_opts[] = {
	{ "help", no_argument, nullptr, LongOptions::HELP },
	{ "radii", required_argument, nullptr, LongOptions::RADII },
	{ "repeats", required_argument, nullptr, LongOptions::REPEATS },
	{ "searches", required_argument, nullptr, LongOptions::SEARCHES },
	{ "check", no_argument, nullptr, LongOptions::CHECK },
	{ "benchmark", required_argument, nullptr, LongOptions::BENCHMARK },
	{ nullptr, 0, nullptr, 0 }
};

void printHelp();
void processArgs(int argc, char** argv);
void setDefaults();

#endif //CPP_MAIN_OPTIONS_HPP
