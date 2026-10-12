# CMakeLists.txt
cmake_minimum_required(VERSION 3.20)
project(geodsd LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

include(FetchContent)

# 1. Fetch libigl (this will also handle Eigen)
FetchContent_Declare(
    libigl
    GIT_REPOSITORY https://github.com/libigl/libigl.git
    GIT_TAG v2.5.0  # Use the latest stable version
)

# This step downloads the library and makes its targets available
FetchContent_MakeAvailable(libigl)
# find_package(Eigen3 REQUIRED)       # or FetchContent
find_package(OpenMP)

file(GLOB CORE_SRC cpp/src/*.cpp)
add_library(geodsd_core STATIC ${CORE_SRC})
target_include_directories(geodsd_core PUBLIC cpp/include)
# Link against libigl
target_link_libraries(geodsd_core PUBLIC igl::core)
target_link_libraries(geodsd_core PUBLIC Eigen3::Eigen)
if(OpenMP_CXX_FOUND)
  target_link_libraries(geodsd_core PUBLIC OpenMP::OpenMP_CXX)
endif()

find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)
nanobind_add_module(_core bindings/module.cpp)
target_link_libraries(_core PRIVATE geodsd_core)
install(TARGETS _core LIBRARY DESTINATION CovBIM)