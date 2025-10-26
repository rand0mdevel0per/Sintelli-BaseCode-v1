# Sintelli-BaseCode-v1

## Overview

This is a CUDA-based neural network simulation project designed to build a large-scale neuronal network in three-dimensional space. The project leverages GPU parallel computing capabilities to simulate neuron behavior, connections, and information transmission.

## Key Features

- High-performance computing based on CUDA
- Three-dimensional spatial neuron network topology
- Support for adaptive message compression and routing
- Integrated KFE (Knowledge Feature Encoding) short-term memory system
- Support for convolution and GEMM inference operations
- C++/CUDA C mixed programming

## Technology Stack

- **Primary Language**: C++ (C++20 standard), CUDA C
- **Build System**: CMake (minimum version 3.18)
- **Dependency Management**: vcpkg
- **CUDA Architecture**: Supports modern GPU architectures such as sm_75, sm_80, sm_86
- **Third-party Libraries**: 
  - CUTLASS (for CUDA-optimized GEMM operations)
  - LibLZMA (data compression)
  - nlohmann/json (JSON processing)
  - Crow (C++ Web framework for potential web interfaces)

## Directory Structure

```
src/
├── Core source files (main.cu, Neuron.cu, NeuronModel.cu)
├── Device queue implementation (deviceQueue.cpp)
├── Matrix operations (matrixMultiplex.cpp)
├── Message encoding/decoding (converter.h, converter.cpp)
├── Data structure definitions (structs.h, conv16_res_msg.h)
├── Utility header files (hasher.h, isw.hpp, sct.hpp, wss.hpp)
├── Third-party libraries (cutlass/, json/)
├── Model files (models/)
├── Build directories (cmake-build-debug/, cmake-build-debug-visual-studio/)
└── Configuration files (CMakeLists.txt, vcpkg.json, .clangd)
```

## Core Components

### 1. Neuron Model (Neuron.cu)

Implementation of a complex neuron model that simulates biological neural computation with GPU acceleration.

#### Neuron Structure

Each neuron in the network is represented by the `Neuron` class, which contains:

- **3D Spatial Positioning**: Each neuron has coordinates (x, y, z) in a 3D grid, allowing for spatial organization and locality-based connections.
- **6-Directional Neighbor Connections**: Neurons can connect to neighbors in the ±X, ±Y, and ±Z directions, forming a 3D lattice structure.
- **4-Logical Port Input/Output System**: Each neuron has 4 logical ports for input and output, enabling multi-channel communication.
- **KFE Short-Term Memory System**: A knowledge feature encoding system for contextual memory and learning.
- **Convolution and GEMM Inference Capabilities**: Support for both convolutional and matrix multiplication-based computations.
- **Port Transformation Matrices**: Hebbian learning-based matrices for feature transformation between ports.

#### Core Computational Flow

The neuron's computation follows a detailed flow within the `step()` function:

1. **Message Processing**: Handle incoming messages from other neurons
   - If queue is not empty, process the message using `processMessage()`
   - Route or receive data messages, forward or reply to connection requests, process connection replies

2. **Input Processing**: Process data from input ports
   - For each port with input data, execute `processUpdate()`
   - Only proceed to output broadcast if there was input

3. **Output Broadcast**: Send results to connected neurons
   - For each output connection, create and route output messages
   - Update convolution kernels for all ports

4. **Maintenance**: Update internal state and perform housekeeping
   - Increment cycle counter
   - Perform KFE decay every 10 steps
   - Initiate neuron discovery every 100 steps (when activity > 0.3 and output connections < 1024)
   - Update multiplex matrices every 50 steps
   - Update neuron activity

#### Detailed Computational Algorithms

##### processUpdate Function Flow

The `processUpdate()` function is the core of neuron computation:

1. **Neighbor Input Aggregation**:
   - Reset `PS_aggregate` matrix to zero
   - For each of the 4 ports, if there's input:
     - Get front input from port queue
     - Transform input using `input_multiplex_array`
     - Extract convolution features using `extractConvFeatures()`
     - Calculate attention score
     - Aggregate features using `aggregateFeatures()`
     - Weight the input and accumulate in `PS_aggregate`

2. **Normalization**:
   - Normalize `PS_aggregate` by total weight if weight_sum > 1e-6

3. **Deviation Calculation**:
   - Compute prediction error: `Deviation[i][j] = PS_aggregate[i][j] - P_stable[i][j]`

4. **Selective SSM**:
   - Execute `selectiveSSM()` for state space modeling

5. **KFE Attention Computation**:
   - Compute STM aggregate utility using `computeKFEAttention()`

6. **Gating Decision**:
   - Determine whether to trigger GEMM:
     - Periodic heartbeat (every 16 steps)
     - High external demand (deviation_norm > 0.5)
     - Internal crisis (core_vulnerability > 0.7)
     - High internal attention (STM_aggregate_utility > 0.6)
   - Execute either `executeGEMMAndDRC()` or `executeMicroCorrection()`

7. **Output Broadcasting**:
   - Execute `broadcastOutput()`
   - Update convolution kernels for all ports using `updateConvKernels()`

##### GEMM Inference with DRC

The neuron's core computation is based on General Matrix Multiply (GEMM) operations with Dynamic Recalibration Correction (DRC):

1. **Positional Encoding**:
   - Add positional encoding to `P_Matrix` using `addPositionalEncoding()`

2. **GEMM Core Inference**:
   $P_{Next} = \text{GELU}(P_{Matrix} \times W_{predict} + M_{KFE})$

   Where:
   - $P_{Matrix}$ is the current state matrix (256×256)
   - $W_{predict}$ is the autoregressive weight matrix (256×256)
   - $M_{KFE}$ is the knowledge context matrix from KFE (256×256)
   - GELU is the Gaussian Error Linear Unit activation function

3. **Fixed Target Computation**:
   $T_{fixed} = \alpha \cdot PS_{aggregate} + (1-\alpha) \cdot P_{Next}$

4. **DRC Iterative Correction**:
   For 16 iterations:
   $P_{new} = P_{current} + V_{corr} + M_{attn} + V_{hist}$

   Where:
   - $V_{corr} = (T_{fixed} - P_{current}) \cdot \eta_{base}$ (basic correction term)
   - $M_{attn}$ is the attention-modulated correction
   - $V_{hist}$ is the historical momentum term

5. **Noise Prediction and Denoising**:
   - Predict noise using `predictNoise()`
   - Apply denoising process with cosine noise schedule

6. **State Synchronization**:
   - Copy `P_current` to `P_Matrix` and `P_stable`
   - Update core vulnerability using `updateCoreVulnerability()`

##### Selective State Space Model (SSM)

The neuron implements a selective SSM mechanism:

1. **Input Projection**:
   $B[i] = \text{GELU}(\frac{1}{256} \sum_{j=0}^{255} PS_{aggregate}[i][j])$ (Input gate)
   $C[i] = \text{GELU}(-\frac{1}{256} \sum_{j=0}^{255} PS_{aggregate}[i][j])$ (Output gate)

2. **State Update**:
   $\Delta[i] = B[i] \cdot PS_{aggregate}[i][0]$
   $h_{state}[i] = 0.9 \cdot h_{state}[i] + \Delta[i]$

3. **Output Projection**:
   $P_{Matrix}[i][j] += C[i] \cdot h_{state}[i]$

##### Convolution Operations

The neuron implements 8×8 convolution operations with stride=8 for feature extraction:

1. **Forward Convolution**:
   For input $I$ and kernel $K$:
   $O[i,j] = \text{ReLU}(\sum_{ki=0}^{7} \sum_{kj=0}^{7} I[i \cdot 8 + ki, j \cdot 8 + kj] \cdot K[ki,kj] + b)$

2. **Deconvolution**:
   For feature map $F$ and kernel $K$:
   $O[i \cdot 8 + ki, j \cdot 8 + kj] += F[i,j] \cdot K[ki,kj]$

3. **Feature Aggregation**:
   - Deconvolve 8 feature maps
   - Weighted fusion: $output[i][j] = \sum_{k=0}^{7} temp\_outputs[k][i][j] / 8.0$

4. **Kernel Update**:
   - Compute gradients using feature maps and deviation
   - Update weights: $input\_conv\_kernels[port][k].kernel[ki][kj] -= learning\_rate \cdot grad / (32.0 \cdot 32.0)$
   - Update bias: $input\_conv\_kernels[port][k].bias -= learning\_rate \cdot bias\_grad / (32.0 \cdot 32.0)$

##### Attention Mechanisms

###### KFE Attention

The Knowledge Feature Encoding (KFE) system uses attention mechanisms to focus on relevant knowledge fragments:

$\text{AttentionWeight} = \frac{1}{1 + e^{-\text{dot\_product}}}$

$\text{WeightedAttention} = \text{AttentionWeight} \cdot I_{core}$

$M_{KFE}[i,j] += \text{WeightedAttention} \cdot V_{mem}[i,j]$

###### Neighbor Aggregation Attention

The neuron aggregates inputs from neighbors using an attention mechanism:

$\text{score} = \frac{\sum_{i=0}^{255} \sum_{j=0}^{255} P_{Matrix}[i][j] \cdot \text{transformed\_input}[i][j]}{256}$

$\text{PS\_aggregate}[i][j] += \text{transformed\_input}[i][j] \cdot w \cdot \text{aggregated}[i][j] \cdot \text{score} + \frac{\text{wkv}}{\text{wkv} + \text{state}}$

###### Importance Computation

The neuron computes its importance for message routing:

$\text{importance} = 0.4 \cdot \text{core\_vulnerability} + 0.3 \cdot \text{activity} + 0.2 \cdot \min(\text{deviation\_norm}, 1.0) + 0.1 \cdot \text{conn\_ratio}$

#### Message Routing and Compression

Neurons communicate through adaptive message passing with three compression modes:

1. **MODE_FULL**: Full matrix transmission
2. **MODE_RESIDUAL**: Residual compression transmission
3. **MODE_CONV_ONLY**: Convolution feature transmission only

Messages are routed greedily in 3D space based on destination coordinates.

### 2. Device Queue (deviceQueue.cpp)

Implementation of a thread-safe CUDA device-side queue for message passing between neurons. The queue uses atomic operations to ensure thread safety:

```cpp
__device__ bool push(const T &item) {
    unsigned long long old_tail = atomicAdd(&tail, 1ULL);
    unsigned long long current_head = atomicAdd(&head, 0ULL);
    if (old_tail - current_head >= CAPACITY) {
        atomicAdd(&tail, -1LL);
        return false;
    }
    int pos = (int) (old_tail % CAPACITY);
    data[pos] = item;
    return true;
}
```

### 3. Neuron Model Management (NeuronModel.cu)

The `NeuronModel` class manages the 3D grid of neurons, handling:

- Neuron allocation and initialization
- Inter-neuron connectivity setup
- Integration with semantic matching systems
- Serialization and deserialization of model state

#### 3D Grid Initialization

Neurons are arranged in a 3D grid with size GRID_SIZE³. Each neuron is initialized with:

- Neighbor queue connections in 6 directions
- Random seed for stochastic operations
- KFE storage and query queues
- Semantic processing components

#### Parallel Processing Loop

The main processing loop executes neuron computations in parallel streams:

1. **Input Processing**: Process input blocks from various processors
2. **Neuron Computation**: Execute neuron computations in parallel streams
3. **Data Flow Management**: Manage matrix data flow between neurons
4. **Semantic Processing**: Handle semantic matching and logic injection
5. **Output Generation**: Generate output messages

### 4. Encoding/Decoding System (converter.h/cpp)

Implementation of matrix to UTF-8 string encoding functionality for data storage and transmission, using a custom encoding scheme that preserves numerical precision while reducing storage requirements.

## Build and Run

### Environment Requirements
- CUDA Toolkit (recommended 11.x or 12.x)
- Compiler supporting C++20 (Visual Studio 2022 recommended)
- CMake 3.18+
- vcpkg package manager

### Build Steps
```bash
# Using Visual Studio generator
cmake -B cmake-build-debug -S . -G "Visual Studio 17 2022"
cmake --build cmake-build-debug
```

### Run
```bash
# Run the generated executable
./cmake-build-debug/Debug/src.exe
```

## Development Conventions

1. **CUDA Programming**: Use modern CUDA programming practices, including unified memory, streams, and asynchronous operations
2. **Memory Management**: Manual memory management, avoiding STL containers like std::string that may have issues on the device side
3. **Error Handling**: Use CUDA error checking macros to ensure the correctness of GPU operations
4. **Code Style**: Follow C++ Core Guidelines, using clang-format to format code
5. **Naming Conventions**: 
   - Class names use PascalCase
   - Functions and variables use snake_case
   - Macro definitions use UPPER_SNAKE_CASE