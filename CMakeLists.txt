cmake_minimum_required(VERSION 3.13)

project(leca)

include(FetchContent)

set(CMAKE_VERBOSE_MAKEFILE OFF)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    message("CMAKE_BUILD_TYPE not set, setting it to Release")
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG origin/master 
)

FetchContent_GetProperties(pybind11)

if(NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif()

find_package(OpenMP REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

############################################################################
# Add leca

set(LECA_INCLUDE_DIR "source/leca;")

include_directories(${LECA_INCLUDE_DIR})

set(LECA_SRC
    "source/leca/Module.cpp"
    "source/leca/Leca.cpp"
)

find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)

pybind11_add_module(leca ${LECA_SRC})

target_link_libraries(leca PUBLIC ${OpenMP_CXX_FLAGS})
