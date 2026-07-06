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

if (NOT deal.II_FOUND)
  message(FATAL_ERROR
    "\n"
    "*** Could not locate deal.II. ***\n\n"
    "You may want to either pass -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set the environment variable DEAL_II_DIR to this path."
  )
endif()

# -----------------------------------------------------------------------------
# Initialize deal.II cache variables
# -----------------------------------------------------------------------------
DEAL_II_INITIALIZE_CACHED_VARIABLES()

# -----------------------------------------------------------------------------
# Find libtiff
# -----------------------------------------------------------------------------
find_package(TIFF REQUIRED)

# -----------------------------------------------------------------------------
# Project interface library
# -----------------------------------------------------------------------------
add_library(${PROJECT_NAME}-lib INTERFACE)

target_include_directories(${PROJECT_NAME}-lib
  INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(${PROJECT_NAME}-lib
  INTERFACE
    TIFF::TIFF
)
# -----------------------------------------------------------------------------
# Main executable
# -----------------------------------------------------------------------------
add_executable(otter otter_driver.cc)

target_link_libraries(otter
  ${PROJECT_NAME}-lib
)

DEAL_II_SETUP_TARGET(otter)

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
enable_testing()

set(TEST_TARGET "otter")
DEAL_II_PICKUP_TESTS("${TEST_TARGET}")

add_subdirectory(tests)
