cmake_minimum_required(VERSION 3.16)

project(visionai_hardware
        VERSION 1.0.0
        DESCRIPTION "Native Linux hardware monitoring layer for linux-ai-vision"
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(VISIONAI_BUILD_TESTS "Build the native unit tests" ON)
option(VISIONAI_BUILD_PYTHON "Build the pybind11 Python extension" ON)
option(VISIONAI_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

add_library(visionai_warnings INTERFACE)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(visionai_warnings INTERFACE
    -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion)
  if(VISIONAI_WARNINGS_AS_ERRORS)
    target_compile_options(visionai_warnings INTERFACE -Werror)
  endif()
endif()

set(VISIONAI_HARDWARE_SOURCES
  hardware/sysfs.cpp
  hardware/cpu_monitor.cpp
  hardware/memory_monitor.cpp
  hardware/temperature_monitor.cpp
  hardware/gpu_monitor.cpp
  hardware/network_monitor.cpp
  hardware/disk_monitor.cpp
  hardware/monitor.cpp)

add_library(visionai_hardware_objects OBJECT ${VISIONAI_HARDWARE_SOURCES})
target_include_directories(visionai_hardware_objects PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(visionai_hardware_objects PRIVATE visionai_warnings)

add_library(visionai_hw SHARED $<TARGET_OBJECTS:visionai_hardware_objects> bindings/c_api.cpp)
target_include_directories(visionai_hw PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(visionai_hw PROPERTIES
  OUTPUT_NAME visionai_hw
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR})
target_link_libraries(visionai_hw PRIVATE visionai_warnings)
find_package(Threads REQUIRED)
target_link_libraries(visionai_hw PUBLIC Threads::Threads)

if(VISIONAI_BUILD_PYTHON)
  find_package(Python3 COMPONENTS Interpreter Development.Module QUIET)
  if(Python3_FOUND)
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
      OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
      RESULT_VARIABLE PYBIND11_QUERY_RESULT)
    if(PYBIND11_QUERY_RESULT EQUAL 0)
      list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
    endif()
  endif()

  find_package(pybind11 CONFIG QUIET)
  if(pybind11_FOUND)
    pybind11_add_module(_visionai_native bindings/python_bindings.cpp)
    target_link_libraries(_visionai_native PRIVATE visionai_warnings)
    target_link_libraries(_visionai_native PRIVATE visionai_hardware_objects)
    message(STATUS "pybind11 extension: building _visionai_native")
  else()
    message(STATUS
      "pybind11 not found: building libvisionai_hw.so only. "
      "The Python side will fall back to the ctypes C ABI or pure Python.")
  endif()
endif()

if(VISIONAI_BUILD_TESTS)
  enable_testing()
  add_executable(visionai_native_tests tests/native_tests.cpp)
  target_link_libraries(visionai_native_tests PRIVATE visionai_hardware_objects)
  target_link_libraries(visionai_native_tests PRIVATE visionai_warnings)
  add_test(NAME native_unit_tests COMMAND visionai_native_tests)
endif()
