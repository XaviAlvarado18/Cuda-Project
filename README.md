# CUDA-Project

**Project #03 - Parallel and Distributed Computing Course**

This project focuses on implementing the Hough Transform using CUDA, leveraging different types of memory (Global, Constant, Shared, and Combined) to evaluate performance improvements in parallel computation.

## Table of Contents
- [Introduction](#introduction)
- [Implementation](#implementation)
- [Results](#results)
- [How to Run](#how-to-run)

---

## Introduction

The Hough Transform is a widely used technique for detecting shapes (like lines) in images. This project uses CUDA to accelerate the transform by distributing the computation across GPU threads. The goal is to compare the performance of different memory types in CUDA: Global, Constant, Shared, and a Combined approach.

---

## Implementation

### 1. Global Memory
Global memory is used to store the input image and accumulator. While it provides large storage, it has higher latencies and lower bandwidth compared to other memory types.

### 2. Constant Memory
Constant memory is leveraged for storing precomputed trigonometric values (`cos(θ)` and `sin(θ)`), which are read-only and shared across all threads. This reduces memory access overhead significantly for repeated operations.

### 3. Shared Memory
Shared memory is utilized within thread blocks to temporarily store accumulations, reducing access to slower global memory. Synchronization between threads ensures correctness, but it adds overhead.

### 4. Combined Memory
The Combined approach integrates the strengths of Global, Constant, and Shared memory:
- **Constant Memory** for precomputed values.
- **Shared Memory** for thread-local accumulations.
- **Global Memory** for final storage.  
This hybrid strategy aims to minimize latencies and maximize throughput.

---

## Results

### Average Execution Times
| Memory Type       | Average Time (ms) |
|-------------------|--------------------|
| **Global Memory** | 2.533             |
| **Constant Memory** | 2.333           |
| **Shared Memory** | 470.000           |
| **Combined Memory** | 1.834           |

The **Combined Memory** implementation demonstrated the best performance, highlighting the advantages of integrating memory strategies in CUDA programming.

### Observations
- Constant memory was efficient for read-only data shared across threads.
- Shared memory incurred significant overhead due to thread synchronization.
- The combined approach leveraged the strengths of each memory type for optimal results.

---

## How to Run

1. **Compile the CUDA program**:
   ```bash
   make
