cmake_minimum_required(VERSION 3.20)
project(InterSystemCommunication VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")

# Find required packages
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(OpenMP)

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/src/cpp
    ${EIGEN3_INCLUDE_DIR}
)

# Source files
set(SOURCES
    examples/cpp/language_model.cpp
)

# Create executable
add_executable(language_model ${SOURCES})

# Link libraries
target_link_libraries(language_model
    Eigen3::Eigen
)

# If OpenMP is found, link it
if(OpenMP_CXX_FOUND)
    target_link_libraries(language_model OpenMP::OpenMP_CXX)
endif()

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
