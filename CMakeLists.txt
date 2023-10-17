cmake_minimum_required(VERSION 3.15...3.26)

project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX)

add_library(skretrievalBuildProperties INTERFACE)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

python_add_library(_core MODULE src/main.cpp WITH_SOABI)
target_link_libraries(skretrievalBuildProperties INTERFACE pybind11::headers)

find_package(Eigen3 CONFIG)

if(Eigen3_FOUND)
  target_link_libraries(skretrievalBuildProperties INTERFACE Eigen3::Eigen)
else()
  include(ExternalProject)

  set(EIGEN_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/eigen-install/")

  ExternalProject_Add(
    eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/eigen-src"
    BINARY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/eigen-build"
    INSTALL_DIR "${EIGEN_INSTALL_DIR}"
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
               -DCMAKE_BUILD_TYPE=Release)

  file(MAKE_DIRECTORY ${EIGEN_INSTALL_DIR}/include) # avoid race condition

  add_library(eigenlib INTERFACE IMPORTED GLOBAL)
  add_dependencies(eigenlib eigen)

  set_target_properties(eigenlib PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                            ${EIGEN_INSTALL_DIR}/include/eigen3)

  target_link_libraries(skretrievalBuildProperties INTERFACE eigenlib)
endif()

target_link_libraries(_core PUBLIC skretrievalBuildProperties)
target_compile_definitions(_core PRIVATE VERSION_INFO=${PROJECT_VERSION})

install(TARGETS _core DESTINATION skretrieval)
