cmake_minimum_required(VERSION 3.18)
project(build_optimizer_backend CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# --- pybind11 -----------------------------------------------------------
# Prefer the pybind11 provided by the active Pixi/Python environment;
# fall back to fetching it if not found (e.g. a bare `cmake` invocation
# outside of `pixi run`).
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.6
  )
  FetchContent_MakeAvailable(pybind11)
endif()

# --- build_optimizer_backend extension module ---------------------------
pybind11_add_module(build_optimizer_backend
  src/bindings.cpp
  src/knapsack.cpp
)

target_include_directories(build_optimizer_backend PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

install(TARGETS build_optimizer_backend DESTINATION .)
