# Sintelli-BaseCode-v1

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![C++20](https://img.shields.io/badge/C++-20-blue.svg)](https://en.cppreference.com/w/cpp/20)

**3D Neural Network Simulation powered by CUDA** 🧠⚡

## Overview

This is a CUDA-based neural network simulation project designed to build a large-scale neuronal network in three-dimensional space. The project leverages GPU parallel computing capabilities to simulate neuron behavior, connections, and information transmission.

## Key Features

- High-performance computing powered by CUDA
- Three-dimensional spatial neuron network topology
- Adaptive message compression and routing
- Integrated KFE (Knowledge Feature Encoding) short-term memory system
- Convolution and GEMM inference operations
- C++/CUDA mixed programming

## Technology Stack

- **Primary Languages**: C++20, CUDA C
- **Build System**: CMake (minimum version 3.18)
- **Dependency Management**: vcpkg
- **CUDA Architecture**: Supports modern GPU architectures (sm_75, sm_80, sm_86, sm_89)
- **Third-party Libraries**: 
  - CUTLASS (CUDA-optimized GEMM operations)
  - LibLZMA (data compression)
  - nlohmann/json (JSON processing)
  - Crow (C++ web framework)

## Directory Structure

```
src/
├── Core source files (main.cu, Neuron.cu, NeuronModel.cu)
├── Device queue implementation (deviceQueue.cpp)
├── Matrix operations (matrixMultiplex.cpp)
├── Message encoding/decoding (converter.h, converter.cpp)
├── Data structure definitions (structs.h, conv16_res_msg.h)
├── Utility headers (hasher.h, isw.hpp, sct.hpp, wss.hpp)
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

- **3D Spatial Positioning**: Each neuron has coordinates (x, y, z) in a 3D grid, enabling spatial organization and locality-based connections
- **6-Directional Neighbor Connections**: Neurons can connect to neighbors in ±X, ±Y, and ±Z directions, forming a 3D lattice structure
- **4-Logical Port I/O System**: Each neuron has 4 logical ports for input and output, enabling multi-channel communication
- **KFE Short-Term Memory System**: A knowledge feature encoding system for contextual memory and learning
- **Convolution and GEMM Inference**: Support for both convolutional and matrix multiplication-based computations
- **Port Transformation Matrices**: Hebbian learning-based matrices for feature transformation between ports

#### Core Computational Flow

The neuron's computation follows this flow in the `step()` function:

1. **Message Processing**: Handle incoming messages from other neurons
   - Process messages from queue using `processMessage()`
   - Route or receive data messages
   - Forward or reply to connection requests

2. **Input Processing**: Process data from input ports
   - For each port with input data, execute `processUpdate()`
   - Proceed to output broadcast only if there was input

3. **Output Broadcast**: Send results to connected neurons
   - Create and route output messages for each output connection
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
   - For each of the 4 ports with input:
     - Retrieve input from port queue
     - Transform input using `input_multiplex_array`
     - Extract convolution features via `extractConvFeatures()`
     - Calculate attention score
     - Aggregate features using `aggregateFeatures()`
     - Weight and accumulate in `PS_aggregate`

2. **Normalization**:
   - Normalize `PS_aggregate` by total weight if weight_sum > 1e-6

3. **Deviation Calculation**:
   - Compute prediction error: `Deviation[i][j] = PS_aggregate[i][j] - P_stable[i][j]`

4. **Selective SSM**:
   - Execute `selectiveSSM()` for state space modeling

5. **KFE Attention Computation**:
   - Compute STM aggregate utility using `computeKFEAttention()`

6. **Gating Decision**:
   - Determine whether to trigger GEMM based on:
     - Periodic heartbeat (every 16 steps)
     - High external demand (deviation_norm > 0.5)
     - Internal crisis (core_vulnerability > 0.7)
     - High internal attention (STM_aggregate_utility > 0.6)
   - Execute either `executeGEMMAndDRC()` or `executeMicroCorrection()`

7. **Output Broadcasting**:
   - Execute `broadcastOutput()`
   - Update convolution kernels for all ports via `updateConvKernels()`

##### GEMM Inference with DRC

The neuron's core computation uses General Matrix Multiply (GEMM) operations with Dynamic Recalibration Correction (DRC):

1. **Positional Encoding**:
   - Add positional encoding to `P_Matrix` using `addPositionalEncoding()`

2. **GEMM Core Inference**:
   ```
   P_Next = GELU(P_Matrix × W_predict + M_KFE)
   ```
   
   Where:
   - `P_Matrix`: Current state matrix (256×256)
   - `W_predict`: Autoregressive weight matrix (256×256)
   - `M_KFE`: Knowledge context matrix from KFE (256×256)
   - GELU: Gaussian Error Linear Unit activation function

3. **Fixed Target Computation**:
   ```
   T_fixed = α · PS_aggregate + (1-α) · P_Next
   ```

4. **DRC Iterative Correction** (16 iterations):
   ```
   P_new = P_current + V_corr + M_attn + V_hist
   ```
   
   Where:
   - `V_corr = (T_fixed - P_current) · η_base`: Basic correction term
   - `M_attn`: Attention-modulated correction
   - `V_hist`: Historical momentum term

5. **Noise Prediction and Denoising**:
   - Predict noise using `predictNoise()`
   - Apply denoising with cosine noise schedule

6. **State Synchronization**:
   - Copy `P_current` to `P_Matrix` and `P_stable`
   - Update core vulnerability via `updateCoreVulnerability()`

##### Selective State Space Model (SSM)

The neuron implements a selective SSM mechanism:

1. **Input Projection**:
   ```
   B[i] = GELU(mean(PS_aggregate[i][:])) 
   C[i] = GELU(-mean(PS_aggregate[i][:]))
   ```

2. **State Update**:
   ```
   Δ[i] = B[i] · PS_aggregate[i][0]
   h_state[i] = 0.9 · h_state[i] + Δ[i]
   ```

3. **Output Projection**:
   ```
   P_Matrix[i][j] += C[i] · h_state[i]
   ```

##### Convolution Operations

The neuron implements 8×8 convolution operations with stride=8 for feature extraction:

1. **Forward Convolution**:
   ```
   O[i,j] = ReLU(Σ(I[i·8+ki, j·8+kj] · K[ki,kj]) + b)
   ```

2. **Deconvolution**:
   ```
   O[i·8+ki, j·8+kj] += F[i,j] · K[ki,kj]
   ```

3. **Feature Aggregation**:
   - Deconvolve 8 feature maps
   - Weighted fusion: `output[i][j] = Σ(temp_outputs[k][i][j]) / 8.0`

4. **Kernel Update**:
   - Compute gradients using feature maps and deviation
   - Update weights: `kernel[ki][kj] -= learning_rate · grad / (32² )`
   - Update bias: `bias -= learning_rate · bias_grad / (32²)`

##### Attention Mechanisms

###### KFE Attention

The Knowledge Feature Encoding (KFE) system uses attention to focus on relevant knowledge fragments:

```
AttentionWeight = 1 / (1 + exp(-dot_product))
WeightedAttention = AttentionWeight · I_core
M_KFE[i,j] += WeightedAttention · V_mem[i,j]
```

###### Neighbor Aggregation Attention

```
score = Σ(P_Matrix[i][j] · transformed_input[i][j]) / 256
PS_aggregate[i][j] += transformed_input[i][j] · w · aggregated[i][j] · score + wkv/(wkv + state)
```

###### Importance Computation

```
importance = 0.4·core_vulnerability + 0.3·activity + 0.2·min(deviation_norm,1.0) + 0.1·conn_ratio
```

#### Message Routing and Compression

Neurons communicate through adaptive message passing with three compression modes:

1. **MODE_FULL**: Full matrix transmission
2. **MODE_RESIDUAL**: Residual compression transmission
3. **MODE_CONV_ONLY**: Convolution feature transmission only

Messages are routed greedily in 3D space based on destination coordinates.

### 2. Device Queue (deviceQueue.cpp)

Thread-safe CUDA device-side queue for message passing between neurons using atomic operations:

```cpp
__device__ bool push(const T &item) {
    unsigned long long old_tail = atomicAdd(&tail, 1ULL);
    unsigned long long current_head = atomicAdd(&head, 0ULL);
    if (old_tail - current_head >= CAPACITY) {
        atomicAdd(&tail, -1LL);
        return false;
    }
    int pos = (int)(old_tail % CAPACITY);
    data[pos] = item;
    return true;
}
```

### 3. Neuron Model Management (NeuronModel.cu)

The `NeuronModel` class manages the 3D grid of neurons:

- Neuron allocation and initialization
- Inter-neuron connectivity setup
- Integration with semantic matching systems
- Model state serialization/deserialization

#### 3D Grid Initialization

Neurons are arranged in a `GRID_SIZE³` 3D grid. Each neuron is initialized with:

- Neighbor queue connections in 6 directions
- Random seed for stochastic operations
- KFE storage and query queues
- Semantic processing components

#### Parallel Processing Loop

The main processing loop executes neuron computations in parallel:

1. **Input Processing**: Process input blocks from various processors
2. **Neuron Computation**: Execute neuron computations in parallel streams
3. **Data Flow Management**: Manage matrix data flow between neurons
4. **Semantic Processing**: Handle semantic matching and logic injection
5. **Output Generation**: Generate output messages

### 4. Encoding/Decoding System (converter.h/cpp)

Matrix to UTF-8 string encoding for data storage and transmission, using a custom encoding scheme that preserves numerical precision while reducing storage requirements.

## Build and Run

### Requirements
- CUDA Toolkit (11.x or 12.x recommended)
- C++20-capable compiler (Visual Studio 2022 recommended for Windows)
- CMake 3.18+
- vcpkg package manager

### Build Steps
```bash
# Configure with Visual Studio generator
cmake -B cmake-build-debug -S . -G "Visual Studio 17 2022"

# Build
cmake --build cmake-build-debug --config Release

# Or use Ninja for faster builds
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### Usage
Use the `sintelliv1` Python package to interact with the model:

```python
import sintelliv1.sintelli_base as sintelli

# Create and start model
sintelli.create_model(grid_size=10)
sintelli.start_model()

# Input text
sintelli.input(text="Hello, Sintelli!")

# Get output
output = sintelli.get_next_output(timeout=2.0)
print(output)

# Cleanup
sintelli.stop_model()
sintelli.destroy_model()
```

## Development Guidelines

1. **CUDA Programming**: Use modern CUDA practices including unified memory, streams, and asynchronous operations
2. **Memory Management**: Manual memory management; avoid STL containers like `std::string` that may have device-side issues
3. **Error Handling**: Use CUDA error checking macros to ensure correctness
4. **Code Style**: Follow C++ Core Guidelines; use clang-format for formatting
5. **Naming Conventions**: 
   - Classes: PascalCase (e.g., `NeuronModel`)
   - Functions/variables: snake_case (e.g., `process_update`)
   - Macros: UPPER_SNAKE_CASE (e.g., `CUDA_CHECK`)

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Citation

If you use this code in your research, please cite:

```bibtex
@software{sintelli2025,
  author = {rand0mdevel0per},
  title = {Sintelli: CUDA-based 3D Neural Network Simulation},
  year = {2025},
  url = {https://github.com/rand0mdevel0per/Sintelli-BaseCode-v1}
}
```

## Contact

- GitHub: [@rand0mdevel0per](https://github.com/rand0mdevel0per)
- Email: rand0mk4cas@gmail.com

---

**Built with 🐱 and CUDA**
