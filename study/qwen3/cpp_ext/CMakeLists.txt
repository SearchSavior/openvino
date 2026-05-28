# OpenVINO C++ extension library exposing two fused custom ops:
#   - GatedDeltaRule        (linear-attn recurrence)
#   - FusedCausalConv1d     (depthwise causal conv with state)
#
# Build:
#   cmake -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j
# Produces libqwen3_ov_ext.so consumable by:
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

find_package(OpenMP)

set(SRC
    kernels.c
    gated_delta_rule.cpp
    fused_causal_conv1d.cpp
    ov_extension.cpp
)

set(TARGET_NAME qwen3_ov_ext)
add_library(${TARGET_NAME} MODULE ${SRC})
set_target_properties(${TARGET_NAME} PROPERTIES PREFIX "lib")

target_compile_options(${TARGET_NAME} PRIVATE
    -O3 -march=native -ffast-math -Wall -Wextra)

target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${TARGET_NAME} PRIVATE OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
endif()
