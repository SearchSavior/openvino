# OpenVINO C++ extension library exposing the custom ops:
#   - GatedDeltaRule        (linear-attn recurrence)
#   - FusedCausalConv1d     (depthwise causal conv with state; kept for
#                            reference only -- not applied by default)
#   - QuantizedKVCache      (int8 KV state update + f32 dequant for SDPA)
#   - QuantizedInt8SDPA     (fused SDPA that consumes i8 K/V directly)
#
# Build:
#   cmake -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j
# Produces libqwen3_ov_ext.so consumable by:
#   ov.Core().add_extension("<path>/libqwen3_ov_ext.so")
#   ov_genai.VLMPipeline(model_path, "CPU", extensions=["<path>/libqwen3_ov_ext.so"])
cmake_minimum_required(VERSION 3.16)
project(qwen3_ov_ext LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Locate the OpenVINO installed in the active Python interpreter.
if(NOT CMAKE_CROSSCOMPILING)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "from openvino.utils import get_cmake_path; print(get_cmake_path(), end='')"
        OUTPUT_VARIABLE OpenVINO_DIR_PY)
endif()
find_package(OpenVINO REQUIRED PATHS "${OpenVINO_DIR_PY}")

# DO NOT link libgomp. Mixing libgomp (OpenMP) with OV's libtbb in the same
# process causes severe thread-scheduler conflicts -- baseline decode drops
# 5x just from loading this .so. Our kernels' #pragma omp parallel become
# no-ops without -fopenmp; we accept single-threaded kernels in cpp_ext.
# The Python/ctypes path keeps its own libqwen3_kernels.so with libgomp,
# which works because it isn't loaded into the OV plugin's process.

set(SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/../kernels/kernels.c
    gated_delta_rule.cpp
    gated_delta_rule_v2.cpp
    fused_causal_conv1d.cpp
    quantized_kv_cache.cpp
    quantized_kv_cache_update.cpp
    quantized_int8_sdpa.cpp
    ov_extension.cpp
)

set(TARGET_NAME qwen3_ov_ext)
add_library(${TARGET_NAME} MODULE ${SRC})
set_target_properties(${TARGET_NAME} PROPERTIES PREFIX "lib")

target_compile_options(${TARGET_NAME} PRIVATE
    -O3 -march=native -ffast-math -Wall -Wextra)
# kernels.c uses sched_setaffinity / CPU_SET; expose them via _GNU_SOURCE.
target_compile_definitions(${TARGET_NAME} PRIVATE _GNU_SOURCE)

# kernels.c / kernels.h are the canonical copies under ../kernels (shared with
# the Python ctypes path). cpp_ext compiles them directly into the extension.
target_include_directories(${TARGET_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../kernels)
target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime)
