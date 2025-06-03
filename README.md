# rule-based-classifier: C++ version.

## Background

LiDAR (Light and Ranging Detection) technology has now become the quintessential technique for collecting geospatial 
data from the earth's surface. This code implements a method for automatic classification of objects with LiDAR data, 
with the aim of detecting ground, vegetation and buildings in the cloud of points.


Original project: https://gitlab.citius.usc.es/lidar/rule-based-classifier.
		
## Installation

### Dependencies
- Eigen and Armadillo
  - Ubuntu
      ```bash
      sudo apt install libeigen3-dev libarmadillo-dev
      ```
  - ArchLinux
      ```bash
      sudo pacman -S eigen
      git clone https://aur.archlinux.org/armadillo.git lib/armadillo
      (cd armadillo && makepkg -si --noconfirm)
      ```
 
- LASTools:
    First we need the dependencies, listed at https://github.com/LAStools/LAStools:
    ```bash
    sudo apt-get install libjpeg62 libpng-dev libtiff-dev libjpeg-dev libz-dev libproj-dev liblzma-dev libjbig-dev libzstd-dev libgeotiff-dev libwebp-dev liblzma-dev libsqlite3-dev
    ```

    Now we clone the repo and build:
    ```bash
    git clone --depth 1 https://github.com/LAStools/LAStools lib/LAStools
    (cd lib/LAStools && cmake . && make)
    ```

### Compilation

In the project directory, just execute
  ```bash
  cmake -B build -DCMAKE_BUILD_TYPE=Release .
  cmake --build build
  ```

This creates the executable build/octrees-benchmark.

### Execution
    ./build/octrees-benchmark [-i input_file] [-o output_dir]

## Authorship
Grupo de Arquitectura de Computadores (GAC)  
Centro Singular de Investigación en Tecnologías Inteligentes (CiTIUS)  
Universidad de Santiago de Compostela (USC)  

Maintainers: 
- Miguel Yermo García ([miguel.yermo@usc.es](mailto:miguel.yermo@usc.es))
- Silvia Rodríguez Alcaraz ([silvia.alcaraz@usc.es](mailto:silvia.alcaraz@usc.es))