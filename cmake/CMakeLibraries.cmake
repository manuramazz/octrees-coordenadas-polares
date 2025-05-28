# ---  L I B R A R I E S  --- #
# --------------------------- #

# Add module directory to the include path.
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${PROJECT_SOURCE_DIR}/cmake/modules")

# Add lib/ folder to the list of folder where CMake looks for packages
set(LIB_FOLDER "${CMAKE_SOURCE_DIR}/lib")
list(APPEND CMAKE_MODULE_PATH ${LIB_FOLDER})

# OpenMP
find_package(OpenMP REQUIRED)
if (OPENMP_CXX_FOUND)
    message(STATUS "OpenMP found and to be linked")
else ()
    message(SEND_ERROR "Could not find OpenMP")
endif ()

# Armadillo
find_package(Armadillo REQUIRED)
if (TARGET armadillo::armadillo)
    message(STATUS "Dependency armadillo::armadillo found")
elseif (${ARMADILLO_FOUND})
    include_directories(${ARMADILLO_INCLUDE_DIR})
    message(STATUS "Armadillo include: " ${ARMADILLO_INCLUDE_DIR})
    message(STATUS "Armadillo libraries: " ${ARMADILLO_LIBRARIES})
else ()
    message(SEND_ERROR "Could find armadillo::armadillo")
endif ()

# Eigen3
find_package(Eigen3 REQUIRED)
if (TARGET Eigen3::Eigen)
    message(STATUS "Dependency Eigen3::Eigen found")
elseif (${EIGEN3_FOUND})
    include_directories(${EIGEN3_INCLUDE_DIR})
    message(STATUS "Eigen include: ${EIGEN3_INCLUDE_DIR}")
else ()
    message(SEND_ERROR "Could find Eigen3")
endif ()

# LASlib
find_package(LASLIB REQUIRED)
if (${LASLIB_FOUND})
  include_directories(${LASLIB_INCLUDE_DIR} ${LASZIP_INCLUDE_DIR})
    message(STATUS "LASlib include: ${LASLIB_INCLUDE_DIR} ${LASZIP_INCLUDE_DIR}")
else ()
    message(SEND_ERROR "Could not find LASLIB")
endif ()

# PCL
set(PCL_DIR "$ENV{HOME}/local/pcl/share/pcl-1.15")
find_package(PCL 1.3 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})