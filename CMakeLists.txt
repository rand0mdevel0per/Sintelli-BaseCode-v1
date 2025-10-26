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
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/utf-8")
endif()

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

if (CUDAToolkit_FOUND)
    message(STATUS "🎯 CUDA: ${CUDAToolkit_VERSION}")
    message(STATUS "   Include: ${CUDAToolkit_INCLUDE_DIRS}")
endif ()

# =============================================================================
# Third-party Library Configuration
# =============================================================================

find_package(Threads REQUIRED)
find_package(ZeroMQ CONFIG REQUIRED)
find_package(curl CONFIG REQUIRED)

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
        deviceQueue.cpp
        matrixMultiplex.cpp
        isw.hpp
        cern.cuh
        conv16_res_msg.cuh
        sct.hpp
        hasher.h
        wss.hpp
        queue.cpp
        NeuronModel.cu
        Neuron.cu
        sim.cu
        structs.h
        converter.h
        converter.cpp
        bpe_tokenizer.cpp
        smry.cpp
        model_process.cpp
        stb_image_write.h
        openai_client.cpp
        openai_client.h
        central_router.cpp
        logic_semantic_matcher.cpp
        GPUMutex.cu
        rag_knowledge_loader.cpp
        rag_knowledge_loader.h
        memory_slot.h
        feature_extractor.cpp
        python_binding.cu
        external_storage_api.h
        semantic_query_engine.cpp
        semantic_query_engine.h
        training_interval.cu
        gpu_containers.cuh
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
        CROW_STANDALONE_ASIO
        $<$<NOT:$<BOOL:${LZMA_LIBRARY}>>:NO_LZMA_SUPPORT>
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

# =============================================================================
# 🎯 Library Linking Configuration (Unified here with PRIVATE keyword)
# =============================================================================

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

message(STATUS "Test program: test_mtools")

message(STATUS "Python: ${Python3_VERSION}")

message(STATUS "CUDA: ${CUDAToolkit_VERSION}")

message(STATUS "========================================")

message(STATUS "")