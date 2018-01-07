cmake_minimum_required(VERSION 2.8.12)
project(OpenSimplexTest)

if (NOT MSVC)
add_compile_options(-std=c++11 -W -Wall -Wextra -pedantic -Wno-unused-parameter)
else ()
add_compile_options(/W4)
endif ()

add_executable(OpenSimplexTest OpenSimplexTest.cpp)
