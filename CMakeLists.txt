cmake_minimum_required(VERSION 3.16)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(OTTER)

# -----------------------------------------------------------------------------
# Find deal.II
# -----------------------------------------------------------------------------
find_package(deal.II 9.6 REQUIRED
  HINTS
    ${DEAL_II_DIR}
    $ENV{DEAL_II_DIR}
    ../
    ../../
)

message(STATUS "Found deal.II version: ${deal.II_VERSION}")
message(STATUS "deal.II_DIR: ${deal.II_DIR}")
message(STATUS "DEAL_II_DIR cache variable: ${DEAL_II_DIR}")
message(STATUS "DEAL_II_DIR environment variable: $ENV{DEAL_II_DIR}")

if (NOT ${deal.II_FOUND})
    message(
            FATAL_ERROR
            "\n"
            "*** Could not locate deal.II. ***\n\n"
            "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
            "or set an environment variable \"DEAL_II_DIR\" that contains this path.")
endif ()

# -----------------------------------------------------------------------------
# Initialize deal.II cache variables
# -----------------------------------------------------------------------------
DEAL_II_INITIALIZE_CACHED_VARIABLES()

# -----------------------------------------------------------------------------
# Find libtiff
# -----------------------------------------------------------------------------
find_package(TIFF REQUIRED)

# -----------------------------------------------------------------------------
# Project library (this is what tests expect: ${PROJECT_NAME}-lib)
# -----------------------------------------------------------------------------
add_library(${PROJECT_NAME}-lib dummy.cc)

# Use plain signature (no PRIVATE/PUBLIC), to be consistent with deal.II macros
target_link_libraries(${PROJECT_NAME}-lib TIFF::TIFF)

# Let deal.II add its include paths, flags, libs, etc.
DEAL_II_SETUP_TARGET(${PROJECT_NAME}-lib)

# -----------------------------------------------------------------------------
# Main executable
# -----------------------------------------------------------------------------
include_directories(include)

add_executable(otter screened_poisson_gmg.cc)

# Link against our project library and TIFF as well
target_link_libraries(otter
    ${PROJECT_NAME}-lib
    TIFF::TIFF)

DEAL_II_SETUP_TARGET(otter)

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
enable_testing()

set(TEST_TARGET "otter")
DEAL_II_PICKUP_TESTS("${TEST_TARGET}")

add_subdirectory(tests)

