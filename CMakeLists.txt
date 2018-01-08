cmake_minimum_required(VERSION 2.8.12)
project(OpenSimplex)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if (NOT MSVC)
    add_compile_options(-std=c++11 -W -Wall -Wextra -pedantic -Wno-unused-parameter)
else ()
    add_compile_options(/W4)
endif ()

file(GLOB HEADERS "include/OpenSimplex/*")
file(GLOB SOURCES "src/OpenSimplex/*")

add_library(OpenSimplex ${HEADERS} ${SOURCES})

target_include_directories(OpenSimplex PUBLIC "include")

option(OPENSIMPLEX_BUILD_EXAMPLES "Build the example program." TRUE)
if (OPENSIMPLEX_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif ()
