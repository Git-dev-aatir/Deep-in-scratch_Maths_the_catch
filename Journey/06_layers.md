# 🧱 Layers Module Documentation

This document describes the design and functionality of the **layer classes** in the neural network library, including the abstract base class and concrete implementations of dense (fully connected) and activation layers.

---

## 🗂 Files Used

| File Path                             | Description                                        |
| ------------------------------------- | -------------------------------------------------- |
| `./include/Layers/BaseLayer.h`        | Abstract base class defining layer interface.      |
| `./include/Layers/DenseLayer.h`       | Dense (fully connected) layer declaration.         |
| `./src/Layers/DenseLayer.cpp`         | Dense layer method implementations.                |
| `./include/Layers/Activation_utils.h` | Activation function utilities and enums.           |
| `./include/Layers/ActivationLayer.h`  | Activation layer declaration.                      |
| `./src/Layers/Activation_utils.cpp`   | Activation functions implementation.               |
| `./src/Layers/ActivationLayer.cpp`    | Activation layer method implementations.           |
| `./include/Layers/Layers.h`           | Header aggregating DenseLayer and ActivationLayer. |

---

## 📄 BaseLayer.h

### 🧩 Abstract Base Class: `BaseLayer`

* **Purpose:** Defines the interface for all neural network layers to ensure a consistent API for:

  * Forward pass computation (`forward`).
  * Backward pass computation (`backward`).
  * Layer summary (`summary`).

* **Key Points:**

  * All methods are pure virtual, enforcing implementation in derived classes.
  * Enables polymorphism for heterogeneous neural network layers.
  * Virtual destructor for safe inheritance.

---

## 📄 DenseLayer.h / DenseLayer.cpp

### ⚙️ Class: `DenseLayer` (Fully Connected Layer)

* **Inherits:** `BaseLayer`

* **Data Members:**

  * `input_size`, `output_size`: Dimensions of input and output.
  * `weights` (2D vector): Weight matrix `[output_size x input_size]`.
  * `biases` (1D vector): Bias vector `[output_size]`.
  * `grad_weights`, `grad_biases`: Stored gradients.
  * `input_cache`: Cache of input for backpropagation.

* **Constructor:**

  * Initializes layer dimensions and optionally weights and biases.

* **Core Methods:**

  * `initializeWeights(...)` — Uses the **Initialization** utility for multiple weight init methods, including seed and sparsity control.
  * `initializeBiases(...)` — Bias initialization similar to weights.
  * `forward(const vector<double>& input)` — Computes output = weights \* input + biases, caches input.
  * `backward(const vector<double>& grad_output, double learning_rate)` — Computes gradients w.r.t inputs, weights, and biases; stores gradients.
  * `clearGradients()` — Resets gradients to zero.
  * `summary()` — Prints layer dimensions.

* **Getters/Setters:**

  * Accessors for weights, biases, and their gradients.

* **Design Change:**  
  **Parameter updates (e.g., `applyGradients`) are now handled by separate optimizer classes, not the layer itself.**  
  The layer only accumulates gradients; the optimizer is responsible for updating parameters using those gradients.

---

## 📄 Activation_utils.h / Activation_utils.cpp

### 🔢 Enum & Functions: `ActivationType` and Activation Utilities

* **Supported Activation Functions:**

  `RELU`, `LEAKY_RELU`, `SIGMOID`, `TANH`, `LINEAR`, `SOFTMAX`, `SELU`

* **Functions:**

  * `applyActivation(...)` — Element-wise application of the selected activation with numerically stable softmax.
  * `activationDerivative(...)` — Element-wise derivative for backpropagation. Throws error for softmax derivative (to be handled combined with cross-entropy).
  * `activationTypeToString(...)` — Converts enum to readable string.

---

## 📄 ActivationLayer.h / ActivationLayer.cpp

### ⚡ Class: `ActivationLayer`

* **Inherits:** `BaseLayer`

* **Data Members:**

  * `activation_type`: Enum specifying activation function.
  * `input_cache`: Stores input for derivative calculation.

* **Constructor:**

  * Takes `ActivationType` parameter.

* **Core Methods:**

  * `forward(...)` — Applies the activation function and caches input.
  * `backward(...)` — Multiplies incoming gradient by activation derivative.
  * `summary()` — Prints activation function name.

* **Notes:**

  * Supports parameters for LeakyReLU (`alpha`) and SELU (`alpha`, `lambda`).
  * Softmax derivative not computed here; handled externally with loss.

---

## 📄 Layers.h

* Convenience header that includes `DenseLayer.h` and `ActivationLayer.h`.

---

## 🛠 Problems Faced & Solutions

| Problem                                                                                                 | Solution                                                                                                                       |
| ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Gradient computation required caching inputs during forward pass to compute weight gradients correctly. | Cached inputs in `input_cache` during forward pass; used cached inputs in backward to compute gradients accurately.            |
| Derivative of Softmax is complex due to Jacobian; direct computation in activation caused errors.       | Threw an exception when derivative of Softmax requested; expect Softmax + CrossEntropy combined in loss for gradient handling. |
| Needed flexible initialization with various schemes (Xavier, He, Orthogonal) and sparsity control.      | Integrated `initializeParameters()` utility supporting multiple init methods, sparsity, and manual seed control.               |
| Orthogonal initialization requires square matrices but user may pass non-square dimensions.             | Added runtime check and error message to reject non-square matrices for orthogonal init.                                       |
| Lack of consistent interface caused maintenance and polymorphism issues.                                | Created abstract `BaseLayer` class with pure virtual methods to enforce consistent interface across all layers.                |
| Parameter update logic (`applyGradients`) was tightly coupled to layers.                                | **Moved all parameter updates to separate optimizer classes. Layers now only store parameters and gradients.**                 |

---

## 🎓 Key Learnings

* **Separation of Concerns:** Layers now only store and accumulate gradients; optimizers handle all parameter updates, making the codebase more modular and extensible.
* **Caching inputs during forward passes** is essential for correct backpropagation, especially in layers with parameters.
* **Handling Softmax derivatives separately** avoids errors and numerical instability in training.
* **Modular weight initialization utilities** with flexibility for various schemes and reproducibility greatly simplify model setup.
* **Clear error handling** (like for orthogonal init) prevents silent bugs and aids debugging.
* **Designing a clean, abstract base class interface** enforces consistent APIs and facilitates extensibility and polymorphism.
* **Batch and hardware support** are natural next steps for efficiency and scalability.

---

## 🚀 Future Improvements

- **Batch Processing:** Add batch versions of forward and backward methods for efficiency.
- **Advanced Layer Types:** Implement convolutional, recurrent, dropout, and normalization layers.
- **Regularization:** Add L1/L2 regularization hooks and support in backward pass.
- **Serialization:** Add save/load methods for weights and biases.
- **Hardware Acceleration:** Integrate SIMD, threading, or GPU support for matrix operations.
- **Layer Parameter Registration:** Expose generic parameter/gradient accessors for optimizers.
- **Unit Testing:** Expand tests for all edge cases, especially for batch and optimizer integration.
- **ONNX/Interoperability:** Support export to ONNX or other formats for deployment.
- **Visualization:** Add utilities for layer-wise output and gradient inspection.

---

# 🏁 Summary

This module provides the **foundation for building feed-forward neural networks**:

* **BaseLayer**: Abstract interface enforcing consistent API.
* **DenseLayer**: Fully connected layer with weight & bias management and gradient tracking.
* **ActivationLayer**: Supports common activation functions critical for non-linearity in neural nets.

**Parameter updates are now handled by dedicated optimizer classes, not by the layers themselves.**  
This separation of concerns enables more flexible, maintainable, and extensible neural network architectures, and aligns with best practices in modern deep learning frameworks.

---
