cmake_minimum_required(VERSION 3.20)
project(metal_treeshap CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ASAN+UBSAN build for the portable targets (validation runs pass clean under both):
#   cmake -B build -DMETAL_TREESHAP_SANITIZE=ON
option(METAL_TREESHAP_SANITIZE "Build portable targets with ASAN+UBSAN" OFF)
if(METAL_TREESHAP_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=address,undefined)
endif()

# ---- Portable targets (build & run on any platform) ----
add_executable(test_preprocess tests/test_preprocess.cpp)
add_executable(test_property tests/test_property_additivity.cpp)
add_executable(reference_cli src/main_reference.cpp)

enable_testing()
add_test(NAME preprocess COMMAND test_preprocess)
add_test(NAME property_additivity COMMAND test_property)
# Golden test needs python + xgboost (verified on 2.0.3 and 3.1.2):
#   python tests/test_vs_xgboost.py $<TARGET_FILE:reference_cli>
# Fixture test needs only python + numpy (no xgboost):
#   python tests/test_fixture.py $<TARGET_FILE:reference_cli>

# ---- Metal targets (macOS / Apple Silicon only; Phase 1-2) ----
# The offline Metal toolchain (xcrun metal/metallib) ships with full Xcode or the
# optional "Metal Toolchain" download — NOT with Command Line Tools alone, and on some
# systems `xcrun --find metal` SUCCEEDS with a stub that then fails to execute
# (validation_v3 finding). So detection COMPILES A TINY PROBE end-to-end (metal -> air,
# metallib -> lib) and only then enables the metallib target. Macs without the toolchain
# use runtime compilation instead: metal_cli auto-falls back to
# newLibraryWithSource(shaders/treeshap.metal) — the path validation_v3 actually ran.
if(APPLE)
  option(METAL_TREESHAP_BUILD_SHADERS "Compile shaders/ to a metallib (needs the offline Metal toolchain)" ON)
  set(METAL_TOOLCHAIN_OK FALSE)
  if(METAL_TREESHAP_BUILD_SHADERS)
    set(_probe_dir ${CMAKE_CURRENT_BINARY_DIR}/metal_probe)
    file(MAKE_DIRECTORY ${_probe_dir})
    file(WRITE ${_probe_dir}/probe.metal
         "#include <metal_stdlib>\nusing namespace metal;\n"
         "kernel void probe(device float* o [[buffer(0)]], uint t [[thread_position_in_grid]]) { o[t] = float(t); }\n")
    execute_process(COMMAND xcrun -sdk macosx metal -std=metal3.0
                            -c ${_probe_dir}/probe.metal -o ${_probe_dir}/probe.air
                    RESULT_VARIABLE _metal_rc OUTPUT_QUIET ERROR_QUIET)
    if(_metal_rc EQUAL 0)
      execute_process(COMMAND xcrun -sdk macosx metallib ${_probe_dir}/probe.air
                              -o ${_probe_dir}/probe.metallib
                      RESULT_VARIABLE _metallib_rc OUTPUT_QUIET ERROR_QUIET)
      if(_metallib_rc EQUAL 0)
        set(METAL_TOOLCHAIN_OK TRUE)
      endif()
    endif()
  endif()

  if(METAL_TOOLCHAIN_OK)
    set(METALLIB ${CMAKE_CURRENT_BINARY_DIR}/treeshap.metallib)
    add_custom_command(
      OUTPUT ${METALLIB}
      COMMAND xcrun -sdk macosx metal -std=metal3.0 -O2
              -c ${CMAKE_CURRENT_SOURCE_DIR}/shaders/treeshap.metal
              -o ${CMAKE_CURRENT_BINARY_DIR}/treeshap.air
      COMMAND xcrun -sdk macosx metallib ${CMAKE_CURRENT_BINARY_DIR}/treeshap.air -o ${METALLIB}
      DEPENDS shaders/treeshap.metal
      COMMENT "Compiling treeshap.metal -> treeshap.metallib")
    add_custom_target(metallib ALL DEPENDS ${METALLIB})
  else()
    message(STATUS "Offline Metal toolchain unusable or disabled (probe compile failed); "
                   "skipping metallib. metal_cli will runtime-compile "
                   "shaders/treeshap.metal instead.")
  endif()

  # Metal fixture runner (the repository-reproducible validation harness). Requires
  # vendored metal-cpp headers: download from https://developer.apple.com/metal/cpp/
  # and unpack to third_party/metal-cpp (so that third_party/metal-cpp/Metal/Metal.hpp
  # exists). Works with OR without the offline toolchain (runtime source fallback).
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/metal-cpp/Metal/Metal.hpp)
    add_executable(metal_cli src/main_metal.cpp)
    target_include_directories(metal_cli PRIVATE
                               ${CMAKE_CURRENT_SOURCE_DIR}/third_party/metal-cpp)
    target_link_libraries(metal_cli "-framework Metal" "-framework Foundation"
                          "-framework QuartzCore")
    if(METAL_TOOLCHAIN_OK)
      add_dependencies(metal_cli metallib)
    endif()
    add_test(NAME fixture_metal
             COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_fixture.py
                     $<TARGET_FILE:reference_cli> --metal-cli $<TARGET_FILE:metal_cli>)
    set_tests_properties(fixture_metal PROPERTIES
                         ENVIRONMENT "METAL_TREESHAP_KERNEL=${CMAKE_CURRENT_SOURCE_DIR}/shaders/treeshap.metal")
  else()
    message(STATUS "third_party/metal-cpp not found; metal_cli target skipped "
                   "(see comment above for vendoring instructions).")
  endif()
endif()

# Portable fixture regression (needs python3 + numpy; no xgboost).
add_test(NAME fixture_portable
         COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_fixture.py
                 $<TARGET_FILE:reference_cli>)
