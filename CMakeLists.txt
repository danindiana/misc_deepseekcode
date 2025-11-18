cmake_minimum_required(VERSION 3.20)
project(InterSystemCommunication VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")

# Find optional packages
find_package(Eigen3 3.4 QUIET NO_MODULE)
find_package(OpenMP)

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/src/cpp
)

if(Eigen3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIR})
    message(STATUS "Eigen3 found: ${EIGEN3_INCLUDE_DIR}")
    add_definitions(-DHAVE_EIGEN3)
else()
    message(WARNING "Eigen3 not found. Some features will be disabled.")
    message(STATUS "To install Eigen3: sudo apt-get install libeigen3-dev")
endif()

# Library sources
set(LIB_SOURCES
    src/cpp/language_model.cpp
)

# Create static library
add_library(language_model_lib STATIC ${LIB_SOURCES})
target_include_directories(language_model_lib PUBLIC
    ${PROJECT_SOURCE_DIR}/src/cpp
)

# Link libraries to lib
if(Eigen3_FOUND)
    target_link_libraries(language_model_lib PUBLIC Eigen3::Eigen)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(language_model_lib PUBLIC OpenMP::OpenMP_CXX)
endif()

# Example executable
add_executable(language_model examples/cpp/main.cpp)
target_link_libraries(language_model language_model_lib)

# Installation
install(TARGETS language_model DESTINATION bin)

# Testing
enable_testing()
add_subdirectory(tests)

# Documentation
option(BUILD_DOCS "Build documentation" ON)
if(BUILD_DOCS)
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
        set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
        set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
        configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
        add_custom_target(docs
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen"
            VERBATIM)
    endif()
endif()
