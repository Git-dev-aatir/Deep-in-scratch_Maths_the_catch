# 🧱 Layers Module Documentation

This document describes the design and functionality of the **layer classes** in the neural network library, including the abstract base class and concrete implementations of dense (fully connected) and activation layers.

---

## 🗂 Files Used

| File Path                             | Description                                        |
| ------------------------------------- | -------------------------------------------------- |
| `./include/Layers/BaseLayer.h`        | Abstract base class defining layer interface.      |
| `./include/Layers/DenseLayer.h`       | Dense (fully connected) layer declaration.         |
| `./src/Layers/denseLayer.cpp`         | Dense layer method implementations.                |
| `./include/Layers/Activation_utils.h` | Activation function utilities and enums.           |
| `./include/Layers/ActivationLayer.h`  | Activation layer declaration.                      |
| `./src/Layers/activation_utils.cpp`   | Activation functions implementation.               |
| `./src/Layers/activationLayer.cpp`    | Activation layer method implementations.           |
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

## 📄 DenseLayer.h / denseLayer.cpp

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
  * `backward(const vector<double>& grad_output, double learning_rate)` — Computes gradients w\.r.t inputs, weights, and biases; stores gradients.
  * `clearGradients()` — Resets gradients to zero.
  * `summary()` — Prints layer dimensions.

* **Getters/Setters:**

  * Accessors for weights, biases, and their gradients.

* **File References:**

  * Declaration: `./include/Layers/DenseLayer.h`
  * Implementation: `./src/Layers/denseLayer.cpp`

---

## 📄 Activation\_utils.h / activation\_utils.cpp

### 🔢 Enum & Functions: `ActivationType` and Activation Utilities

* **Supported Activation Functions:**

  `RELU`, `LEAKY_RELU`, `SIGMOID`, `TANH`, `LINEAR`, `SOFTMAX`, `SELU`

* **Functions:**

  * `applyActivation(...)` — Element-wise application of the selected activation with numerically stable softmax.
  * `activationDerivative(...)` — Element-wise derivative for backpropagation. Throws error for softmax derivative (to be handled combined with cross-entropy).
  * `activationTypeToString(...)` — Converts enum to readable string.

* **File References:**

  * Declaration: `./include/Layers/Activation_utils.h`
  * Implementation: `./src/Layers/activation_utils.cpp`

---

## 📄 ActivationLayer.h / activationLayer.cpp

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

* **File References:**

  * Declaration: `./include/Layers/ActivationLayer.h`
  * Implementation: `./src/Layers/activationLayer.cpp`

---

## 📄 Layers.h

* Convenience header that includes `DenseLayer.h` and `ActivationLayer.h`.

* **File Reference:** `./include/Layers/Layers.h`

---

## 🛠 Problems Faced & Solutions

| Problem                                                                                                 | Solution                                                                                                                       |
| ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Gradient computation required caching inputs during forward pass to compute weight gradients correctly. | Cached inputs in `input_cache` during forward pass; used cached inputs in backward to compute gradients accurately.            |
| Derivative of Softmax is complex due to Jacobian; direct computation in activation caused errors.       | Threw an exception when derivative of Softmax requested; expect Softmax + CrossEntropy combined in loss for gradient handling. |
| Needed flexible initialization with various schemes (Xavier, He, Orthogonal) and sparsity control.      | Integrated `initializeParameters()` utility supporting multiple init methods, sparsity, and manual seed control.               |
| Orthogonal initialization requires square matrices but user may pass non-square dimensions.             | Added runtime check and error message to reject non-square matrices for orthogonal init.                                       |
| Lack of consistent interface caused maintenance and polymorphism issues.                                | Created abstract `BaseLayer` class with pure virtual methods to enforce consistent interface across all layers.                |

---

## 🎓 Key Learnings

* Caching inputs during forward passes is essential for correct backpropagation, especially in layers with parameters.
* Handling Softmax derivatives separately avoids errors and numerical instability in training.
* Modular weight initialization utilities with flexibility for various schemes and reproducibility greatly simplify model setup.
* Clear error handling (like for orthogonal init) prevents silent bugs and aids debugging.
* Designing a clean, abstract base class interface enforces consistent APIs and facilitates extensibility and polymorphism.

---

# 🏁 Summary

This module provides the **foundation for building feed-forward neural networks**:

* **BaseLayer**: Abstract interface enforcing consistent API.
* **DenseLayer**: Fully connected layer with weight & bias management and gradient tracking.
* **ActivationLayer**: Supports common activation functions critical for non-linearity in neural nets.

Together, these classes enable creating trainable networks capable of learning via backpropagation and gradient descent.
