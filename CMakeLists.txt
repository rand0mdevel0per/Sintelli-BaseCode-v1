# =============================================================================
# iFlow Neural Network Simulator - CMake构建配置
# 
# 项目: CUDA神经网络模拟器
# 版本: 1.0
# 日期: 2025-10-03
# =============================================================================

cmake_minimum_required(VERSION 3.18)

project(src CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# =============================================================================
# Python配置
# =============================================================================

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

message(STATUS "🐍 Python: ${Python3_VERSION}")
message(STATUS "   Include: ${Python3_INCLUDE_DIRS}")
message(STATUS "   Library: ${Python3_LIBRARIES}")

# =============================================================================
# CUDA配置
# =============================================================================

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CUDA_STANDARD 20)

if(CUDAToolkit_FOUND)
    message(STATUS "🎯 CUDA: ${CUDAToolkit_VERSION}")
    message(STATUS "   Include: ${CUDAToolkit_INCLUDE_DIRS}")
endif()

# =============================================================================
# 第三方库配置
# =============================================================================

find_package(Threads REQUIRED)
find_package(ZeroMQ CONFIG REQUIRED)
find_package(curl CONFIG REQUIRED)

# LZMA配置
find_path(LZMA_INCLUDE_DIR lzma.h)
find_library(LZMA_LIBRARY NAMES lzma liblzma)

if(LZMA_INCLUDE_DIR AND LZMA_LIBRARY)
    message(STATUS "✅ liblzma: ${LZMA_LIBRARY}")
else()
    message(STATUS "⚠️  liblzma not found, using fallback")
endif()

# =============================================================================
# 源文件配置
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
)

# =============================================================================
# 编译选项
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
# 包含路径配置
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
# 🎯 库链接配置（统一在这里，使用PRIVATE关键字）
# =============================================================================

target_link_libraries(src
        PRIVATE
        # CUDA库
        CUDA::cudart
        CUDA::curand
        CUDA::cublas

        # Python库
        ${Python3_LIBRARIES}

        # Windows系统库
        ws2_32
        kernel32
        user32

        # 第三方库
        libzmq
        CURL::libcurl
        Threads::Threads

        # LZMA（如果找到）
        $<$<BOOL:${LZMA_LIBRARY}>:${LZMA_LIBRARY}>
)

# =============================================================================
# 目标属性配置
# =============================================================================

set_target_properties(src PROPERTIES
        PREFIX ""
        OUTPUT_NAME "sintelli_base"
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD 20
        POSITION_INDEPENDENT_CODE ON
)

# Windows用.pyd，Linux用.so
if(WIN32)
    set_target_properties(src PROPERTIES SUFFIX ".pyd")
else()
    set_target_properties(src PROPERTIES SUFFIX ".so")
endif()

# =============================================================================
# 示例程序配置
# =============================================================================

# ExternalStorage API示例
add_executable(example_external_storage_api
    example_external_storage_api.cpp
)

target_include_directories(example_external_storage_api
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

target_link_libraries(example_external_storage_api
    PRIVATE
    src
)

# 增强版RAG演示程序
add_executable(demo_rag_enhanced
    demo_rag_enhanced.cpp
)

target_include_directories(demo_rag_enhanced
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

target_link_libraries(demo_rag_enhanced
    PRIVATE
    src
)

# Hugging Face RAG集成示例
add_executable(huggingface_rag_integration
    huggingface_rag_integration.cpp
)

target_include_directories(huggingface_rag_integration
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

target_link_libraries(huggingface_rag_integration
    PRIVATE
    src
)

# =============================================================================
# 安装配置
# =============================================================================

install(TARGETS src
        LIBRARY DESTINATION ${Python3_SITELIB}
        RUNTIME DESTINATION ${Python3_SITELIB}
)

# =============================================================================
# 输出信息
# =============================================================================

message(STATUS "")
message(STATUS "========================================")
message(STATUS "项目配置完成")
message(STATUS "========================================")
message(STATUS "输出: sintelli_base${CMAKE_SHARED_LIBRARY_SUFFIX}")
message(STATUS "Python: ${Python3_VERSION}")
message(STATUS "CUDA: ${CUDAToolkit_VERSION}")
message(STATUS "========================================")
message(STATUS "")