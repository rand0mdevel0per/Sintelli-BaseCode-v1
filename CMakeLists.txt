# =============================================================================
# Sintelli Neural Network Simulator - CMake Build Configuration
# 
# Project: CUDA Neural Network Simulator
# Version: 1.0
# Date: 2025-10-03
# Github Repo: https://github.com/rand0mdevel0per/Sintelli-BaseCode-v1
#
# This file configures the CMake build system for the Sintelli Neural Network 
# Simulator. It specifies dependencies, compiler options, and build targets.
# =============================================================================

cmake_minimum_required(VERSION 3.18)

if(MSVC)
    add_compile_options(/utf-8)
endif()

if (MSVC)
    add_compile_options(/Zm200)
endif()

if (MSVC)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=\"/Zm200 /bigobj /utf-8\" -v -allow-unsupported-compiler")
endif()

set(CMAKE_VERBOSE_MAKEFILE ON)
set_property(GLOBAL PROPERTY RULE_MESSAGES ON)

set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -O1 -Wall ")

project(src CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# =============================================================================
# Python Configuration
# =============================================================================

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

message(STATUS "🐍 Python: ${Python3_VERSION}")
message(STATUS "   Include: ${Python3_INCLUDE_DIRS}")
message(STATUS "   Library: ${Python3_LIBRARIES}")

# =============================================================================
# CUDA Configuration
# =============================================================================

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

if (NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89")
endif ()

if (CUDAToolkit_FOUND)
    message(STATUS "🎯 CUDA: ${CUDAToolkit_VERSION}")
    message(STATUS "   Include: ${CUDAToolkit_INCLUDE_DIRS}")
endif ()

# =============================================================================
# Third-party Library Configuration
# =============================================================================

find_package(Threads REQUIRED)
find_package(ZeroMQ CONFIG REQUIRED)
find_package(CURL CONFIG REQUIRED)

# LZMA Configuration
find_path(LZMA_INCLUDE_DIR lzma.h)
find_library(LZMA_LIBRARY NAMES lzma liblzma)

if (LZMA_INCLUDE_DIR AND LZMA_LIBRARY)
    message(STATUS "✅ liblzma: ${LZMA_LIBRARY}")
else ()
    message(STATUS "⚠️  liblzma not found, using fallback")
endif ()

# =============================================================================
# Source Files Configuration
# =============================================================================

add_library(src SHARED
        deviceQueue.cu
        matrixMultiplex.cu
        isw.cuh
        cern.cuh
        conv16_res_msg.cuh
        sct.cuh
        hasher.cuh
        queue.cu
        NeuronModel.cu
        Neuron.cu
        sim.cu
        structs.cuh
        converter.cuh
        converter.cu
        bpe_tokenizer.cu
        smry.cu
        stb_image_write.h
        openai_client.cpp
        openai_client.h
        logic_semantic_matcher.cu
        GPUMutex.cu
        rag_knowledge_loader.cuh
        memory_slot.cuh
        feature_extractor.cu
        python_binding.cu
        external_storage_api.cuh
        semantic_query_engine.cu
        semantic_query_engine.cuh
        training_interval.cu
        gpu_containers.cuh
        NeuronOptimize.cu
        hasher_cu.cuh
        hasher_impl.cu
        ffi_bridge.cu
        string_utils.h
        string_utils.cpp
        string_utils.h
        feature_extractor.h
        smry.h
        semantic_query_engine.cpp
        external_storage_api.h
        semantic_matcher.h
        rag_knowledge_loader.h
        bpe_tokenizer.cpp
        converter.cpp
)

# =============================================================================
# Compilation Options
# =============================================================================

target_compile_options(src
        PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -fPIC>
        $<$<COMPILE_LANGUAGE:CUDA>:-expt-relaxed-constexpr>
)

target_compile_definitions(src
        PRIVATE
        #CUTLASS_AVAILABLE
        CROW_STANDALONE_ASIO
        $<$<NOT:$<BOOL:${LZMA_LIBRARY}>>:NO_LZMA_SUPPORT>
)

target_compile_definitions(src
        PRIVATE
        # USE_OPTIMIZED_KERNELS
)

# =============================================================================
# Include Path Configuration
# =============================================================================

set(CUTLASS_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cutlass")

target_include_directories(src
        PRIVATE
        ${Python3_INCLUDE_DIRS}
        ${CUTLASS_PATH}/include
        ${CUTLASS_PATH}/tools/util/include
        ${CMAKE_CURRENT_SOURCE_DIR}/json
        ${CMAKE_CURRENT_SOURCE_DIR}/onnx/include
        ${CMAKE_CURRENT_SOURCE_DIR}/zmq/include
        ${CUDAToolkit_INCLUDE_DIRS}
        $<$<BOOL:${LZMA_INCLUDE_DIR}>:${LZMA_INCLUDE_DIR}>
)

target_compile_definitions(src
        PRIVATE
        #CUTLASS_AVAILABLE
        CROW_STANDALONE_ASIO
        $<$<NOT:$<BOOL:${LZMA_LIBRARY}>>:NO_LZMA_SUPPORT>
)

target_link_libraries(src
        PRIVATE
        # CUDA libraries
        CUDA::cudart
        CUDA::curand
        CUDA::cublas

        # Python libraries
        ${Python3_LIBRARIES}

        # Windows system libraries
        ws2_32
        kernel32
        user32

        # Third-party libraries
        libzmq
        CURL::libcurl
        Threads::Threads

        # LZMA (if found)
        $<$<BOOL:${LZMA_LIBRARY}>:${LZMA_LIBRARY}>
)

# =============================================================================
# Target Properties Configuration
# =============================================================================

set_target_properties(src PROPERTIES
        PREFIX ""
        OUTPUT_NAME "sintelli_base"
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD 20
        POSITION_INDEPENDENT_CODE ON
)

# Use .pyd for Windows, .so for Linux
if (WIN32)
    set_target_properties(src PROPERTIES SUFFIX ".pyd")
else ()
    set_target_properties(src PROPERTIES SUFFIX ".so")
endif ()

# =============================================================================
# Installation Configuration
# =============================================================================

install(TARGETS src
        LIBRARY DESTINATION ${Python3_SITELIB}
        RUNTIME DESTINATION ${Python3_SITELIB}
)


# =============================================================================

# Output Information

# =============================================================================


message(STATUS "")

message(STATUS "========================================")

message(STATUS "Project configuration completed")

message(STATUS "========================================")

message(STATUS "Output: sintelli_base${CMAKE_SHARED_LIBRARY_SUFFIX}")

message(STATUS "Python: ${Python3_VERSION}")

message(STATUS "CUDA: ${CUDAToolkit_VERSION}")

message(STATUS "========================================")

message(STATUS "")
