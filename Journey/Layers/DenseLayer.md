# 🧱 DenseLayer.md

## 📝 Overview

The `DenseLayer` class implements a fully connected neural network layer, providing forward and backward propagation, gradient storage, and flexible weight and bias initialization. It supports various initialization methods and is designed to be a core building block in customizable neural networks.

---


## ✨ Key Features

- Fully connected layer with configurable input and output sizes.
- Supports multiple weight and bias initialization strategies.
- Caches inputs for backpropagation.
- Provides gradient accumulation and clearing.
- Includes utility functions for printing and summarizing parameters.

---


## 🏗️ Design Decisions

- Uses `std::vector>` for weight matrices and vectors for biases and gradients.
- Separates weight and bias initialization with flexible parameters including sparsity and random seed.
- Implements input caching to support gradient computation during backpropagation.
- Provides const-correct accessors and mutators with validation.

---


## 🛠️ Implementation Highlights

### Initialization
- Weight and bias initialization support various methods (Xavier, He, uniform, constant) with optional sparsity.
- Random seed parameter allows reproducible initialization.

### Forward Pass
- Computes output as `y = Wx + b`.
- Validates input size and parameter initialization.
- Caches input vector for backward pass.

### Backward Pass
- Computes gradients w.r.t input, weights, and biases.
- Accumulates gradients for batch updates.
- Validates gradient output size and input cache presence.

### Utilities
- `clearGradients()` resets accumulated gradients.
- `summary()` prints layer configuration and parameter count.
- `printWeights()` and `printBiases()` display parameters with formatting.

---


## 🚀 Usage Example

```cpp
DenseLayer layer(128, 64);
layer.initializeWeights(InitMethod::Xavier, 42);
layer.initializeBiases(InitMethod::Constant, 42, 0.1);

std::vector input(128, 1.0);
auto output = layer.forward(input);

std::vector grad_output(64, 0.5);
auto grad_input = layer.backward(grad_output);

layer.clearGradients();
layer.summary();
```

---


## ⚡ Performance and Limitations

- Uses nested vectors which may have performance overhead compared to contiguous memory.
- No batch processing support; designed for single input vectors.
- No GPU acceleration or parallelization.

---

## 🚧 Future Improvements

- Support batch inputs and outputs.
- Integrate with matrix libraries like Eigen for optimized computations.
- Add GPU acceleration support.
- Implement additional initialization methods.
- Provide serialization and deserialization of parameters.