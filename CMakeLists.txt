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
add_executable(test_csv_io tests/test_csv_io.cpp)
add_executable(test_deterministic_plan tests/test_deterministic_plan.cpp)
add_executable(reference_cli src/main_reference.cpp)

enable_testing()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_test(NAME preprocess COMMAND test_preprocess)
add_test(NAME property_additivity COMMAND test_property)
add_test(NAME csv_io COMMAND test_csv_io)
add_test(NAME deterministic_plan COMMAND test_deterministic_plan)
add_test(NAME reference_cli_validation
         COMMAND ${Python3_EXECUTABLE}
                 ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_reference_cli.py
                 $<TARGET_FILE:reference_cli>)
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

  # Metal is a mandatory part of an Apple build: a missing dependency must fail at
  # configure time, never produce a deceptively green portable-only `ctest -R fixture`.
  # The pinned metal-cpp headers are committed under third_party/metal-cpp.
  set(METAL_CPP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/metal-cpp)
  if(NOT EXISTS ${METAL_CPP_DIR}/Metal/Metal.hpp OR
     NOT EXISTS ${METAL_CPP_DIR}/Foundation/Foundation.hpp)
    message(FATAL_ERROR
            "Pinned third_party/metal-cpp headers are missing; restore the complete "
            "repository checkout before configuring an Apple build")
  endif()

  add_executable(metal_cli src/main_metal.cpp)
  add_executable(test_metal_host tests/test_metal_host.cpp)
  add_executable(phase2_benchmark src/main_benchmark.cpp)
  foreach(target metal_cli test_metal_host phase2_benchmark)
    target_include_directories(${target} PRIVATE ${METAL_CPP_DIR})
    target_link_libraries(${target} "-framework Metal" "-framework Foundation"
                          "-framework QuartzCore")
  endforeach()
  if(METAL_TOOLCHAIN_OK)
    add_dependencies(metal_cli metallib)
    add_dependencies(phase2_benchmark metallib)
  endif()

  add_test(NAME metal_host_edges
           COMMAND test_metal_host ${CMAKE_CURRENT_SOURCE_DIR}/shaders/treeshap.metal)
  add_test(NAME metal_cli_validation
           COMMAND ${Python3_EXECUTABLE}
                   ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_metal_cli.py
                   $<TARGET_FILE:metal_cli>)
  add_test(NAME phase2_benchmark_smoke
           COMMAND phase2_benchmark
                   ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/deep31/paths.csv
                   ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/deep31/X.csv 1
                   --intercepts 0.5
                   --kernel ${CMAKE_CURRENT_SOURCE_DIR}/shaders/treeshap.metal
                   --expected
                   ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/deep31/expected_contribs.csv
                   --max-abs-error 0.001
                   --warmup 1 --iterations 2 --row-limit 4
                   --rows-per-simdgroup 7 --threads-per-threadgroup 64
                   --accumulation simdgroup --model-storage private)
  add_test(NAME phase2_benchmark_deterministic_smoke
           COMMAND phase2_benchmark
                   ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/deep31/paths.csv
                   ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/deep31/X.csv 1
                   --intercepts 0.5
                   --kernel ${CMAKE_CURRENT_SOURCE_DIR}/shaders/treeshap.metal
                   --expected
                   ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/deep31/expected_contribs.csv
                   --max-abs-error 0.001
                   --warmup 1 --iterations 2 --row-limit 4
                   --rows-per-simdgroup 7 --threads-per-threadgroup 64
                   --accumulation deterministic --deterministic-scratch-mib 1
                   --model-storage private)

  foreach(rows_per_simdgroup 1 7 1024)
    add_test(NAME fixture_metal_rps${rows_per_simdgroup}
             COMMAND ${Python3_EXECUTABLE}
                     ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_fixture.py
                     $<TARGET_FILE:reference_cli>
                     --metal-cli $<TARGET_FILE:metal_cli>
                     --metal-rows-per-simdgroup ${rows_per_simdgroup})
    set_tests_properties(
      fixture_metal_rps${rows_per_simdgroup}
      PROPERTIES ENVIRONMENT
                 "METAL_TREESHAP_KERNEL=${CMAKE_CURRENT_SOURCE_DIR}/shaders/treeshap.metal")
  endforeach()

  # One repository-permanent all-fixture differential for each Phase-2 path. The three
  # atomic row-bank variants above retain the Phase-1 boundary coverage; these two tests
  # ensure SIMD aggregation and deterministic reduction cannot silently regress.
  foreach(accumulation simdgroup deterministic)
    add_test(NAME fixture_metal_${accumulation}_rps7
             COMMAND ${Python3_EXECUTABLE}
                     ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_fixture.py
                     $<TARGET_FILE:reference_cli>
                     --metal-cli $<TARGET_FILE:metal_cli>
                     --metal-rows-per-simdgroup 7
                     --metal-accumulation ${accumulation})
    set_tests_properties(
      fixture_metal_${accumulation}_rps7
      PROPERTIES ENVIRONMENT
                 "METAL_TREESHAP_KERNEL=${CMAKE_CURRENT_SOURCE_DIR}/shaders/treeshap.metal")
  endforeach()
endif()

# Portable fixture regression (needs python3 + numpy; no xgboost).
add_test(NAME fixture_portable
         COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_fixture.py
                 $<TARGET_FILE:reference_cli>)
add_test(NAME phase2_workload_generator
         COMMAND ${Python3_EXECUTABLE}
                 ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/phase2_workloads.py hot
                 ${CMAKE_CURRENT_BINARY_DIR}/phase2_hot_smoke
                 --trees 4 --rows 9 --seed 7 --force)
