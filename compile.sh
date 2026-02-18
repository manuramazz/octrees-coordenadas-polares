#!/bin/bash

rm -rf build

cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-I$(pwd)/lib/LAStools/LASzip/include/laszip" .

cmake --build build


