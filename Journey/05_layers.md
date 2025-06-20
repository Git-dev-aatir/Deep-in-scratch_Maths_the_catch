# 05\_layers.md

# 🧱 Layers Module Documentation

This document describes the design and functionality of the **layer classes** in the neural network library, including the abstract base class and concrete implementations of dense (fully connected) and activation layers.

---

## 📄 layers.h

### 🧩 Abstract Base Class: `Layer`

* **Purpose:**
  Defines the interface for all neural network layers to ensure a consistent API for **forward pass**, **backward pass**, and **summary**.

* **Key Methods:**

  * `forward(const vector<double>& input) -> vector<double>`
    ⚡ Pure virtual function. Computes the output of the layer given an input vector.
  * `backward(const vector<double>& grad_output, double learning_rate) -> vector<double>`
    🔄 Pure virtual function. Computes the gradient of the loss with respect to the input, using the gradient from the next layer (`grad_output`). Also allows for parameter updates.
  * `summary() const`
    📝 Pure virtual function. Prints a brief summary of the layer.
  * Virtual destructor 🗑️.

* **Notes:**

  * Enforces implementation of all methods in derived classes.
  * Supports polymorphism for heterogeneous layer stacks.

---

## 🏗 dense.h

### ⚙️ Class: `Dense` (Fully Connected Layer)

* **Inherits:** `Layer`

* **Data Members:**

  * `input_size`, `output_size` (size\_t): Dimensions of input and output.
  * `weights` (2D vector): Weight matrix `[output_size x input_size]`.
  * `biases` (1D vector): Bias vector `[output_size]`.
  * `grad_weights`, `grad_biases`: Stored gradients for use with optimizers.
  * `input_cache`: Cache of input from forward pass for backprop.

* **Constructor:**
  Initializes sizes and optionally weights and biases.

* **Methods:**

  * `initializeWeights(...)` 🎲
    Uses various initialization methods (Xavier, He, etc.) with support for sparsity and seed control.
  * `initializeBiases(...)` 🎲
    Similar initialization for biases.
  * `forward(const vector<double>& input)`
    Computes output = weights \* input + biases and caches input.
  * `backward(const vector<double>& grad_output, double learning_rate)`
    Calculates gradients w\.r.t. inputs, weights, biases, stores gradients, and performs parameter update via gradient descent.
  * `summary() const` 📝
    Prints layer info, e.g., input → output size.
  * Getter methods for weights, biases, and their gradients for optimizer use.

* **Notes:**

  * Parameter updates happen inside backward but can be decoupled for advanced optimizers later.
  * Enables full gradient tracking for training.

---

## ⚡ activation.h

### 🔢 Enum: `ActivationType`

Supported activation functions:
RELU, LEAKY\_RELU, SIGMOID, TANH, LINEAR, SOFTMAX, SELU

### Functions:

* `applyActivation(...)` 🎯
  Applies the chosen activation function element-wise on the input vector, with numerically stable softmax.

* `activationDerivative(...)` 🔍
  Calculates element-wise derivative for backpropagation. Throws error if derivative requested for softmax (handled with cross-entropy loss).

* `activationTypeToString(...)` 🆔
  Converts enum to human-readable string.

### ⚙️ Class: `ActivationLayer`

* **Inherits:** `Layer`

* **Data Members:**

  * `activation_type`: Chosen activation function.
  * `input_cache`: Stores input vector for computing derivative during backward.

* **Constructor:**
  Takes activation type.

* **Methods:**

  * `forward(const vector<double>& input)`
    Applies activation and caches input.
  * `backward(const vector<double>& grad_output, double learning_rate)`
    Multiplies upstream gradient by activation derivative.
  * `summary() const` 📝
    Prints the activation function name.

* **Notes:**

  * Supports LeakyReLU and SELU parameters (`alpha`, `lambda`) with defaults.
  * Softmax derivative is not computed here — it's combined with cross-entropy loss.

---

# 🏁 Summary

This module forms the **foundation** for building feed-forward neural networks:

* 🧩 Abstract base class `Layer` defines the required interface.
* ⚙️ `Dense` layer implements fully connected layers with weight & bias management plus gradient storage.
* ⚡ `ActivationLayer` supports common nonlinear activations crucial for learning complex functions.

Together, these components allow creating trainable networks capable of learning via gradient descent and backpropagation.
