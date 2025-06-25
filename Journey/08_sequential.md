Sure! Here’s your freshly generated **`Sequential.md`** file — complete with emoji headings and the "Future Improvements" you requested:

---

# 📦 Sequential Model Documentation

---

## 📝 Overview

The `Sequential` class is a container for stacking layers, inspired by frameworks like **PyTorch** and **Keras**.
It allows building models by simply chaining layers and handling their forward, backward passes and training loops internally.

**Key Features:**

* Flexible layer stacking using variadic templates
* Automatic weight initialization based on activation function types
* Integrated optimizer support
* Parameter initialization check to prevent accidental usage before setup

---

## 📂 File Structure

```
include/
└── Models/
    └── Sequential.h   # Header file for Sequential class
src/
└── Models/
    └── Sequential.cpp # Implementation file
```

---

## 🏛️ Classes

### 🔹 `Sequential`

#### Description:

Represents a sequential model container that owns and manages a list of layer pointers (`BaseLayer*`).

#### Important Notes:

* **Owns all layers:** The destructor deallocates all added layers.
* **Not copyable:** Copy constructor and copy assignment are deleted to prevent unsafe shallow copies.
* **Initialization required:** Throws an error if `forward()` or `train()` is called without prior parameter initialization.

---

## 🔧 Public Methods

| Return Type           | Method                                                                                                                                                 | Description                                                               |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------- |
| Constructor           | `Sequential(Layers... args)`                                                                                                                           | Variadic template to accept any number of layer pointers.                 |
| `void`                | `initializeParameters(unsigned int seed=MANUAL_SEED, double a=0, double b=1, double sparsity=0, double bias=0.1)`                                      | Initializes weights/biases of dense layers based on next activation type. |
| `std::vector<double>` | `forward(const std::vector<double>& input) const`                                                                                                      | Forward pass through all layers.                                          |
| `std::vector<double>` | `backward(const std::vector<double>& grad_output, double learning_rate=0.01)`                                                                          | Backward pass through all layers.                                         |
| `double`              | `train(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& Y_true, LossFunction, LossDerivative, Optimizer*)`           | Train using per-sample loss/gradient.                                     |
| `double`              | `train(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& Y_true, BatchLossFunction, BatchLossDerivative, Optimizer*)` | Train using batch-aware loss/gradient.                                    |
| `void`                | `summary() const`                                                                                                                                      | Prints model architecture summary.                                        |
| Destructor            | `~Sequential()`                                                                                                                                        | Deallocates all owned layers.                                             |

---

## ⚙️ Private Members

| Type                      | Name             | Description                                              |
| ------------------------- | ---------------- | -------------------------------------------------------- |
| `std::vector<BaseLayer*>` | `layers`         | Container holding pointers to all layers.                |
| `bool`                    | `is_initialized` | Tracks if parameters are initialized (default: `false`). |

---

## 🚨 Error Handling

* **Uninitialized Parameters Protection:**
  If `initializeParameters()` is not called before `forward()` or `train()`, the following runtime error is thrown:

  ```
  Error: Model parameters not initialized. Call initializeParameters() before training.
  ```

---

## 🛠️ Problems Faced & Solutions

| Problem                                                                   | Solution                                                                  |
| ------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| Users may forget to initialize parameters before `forward()` or `train()` | Introduced `is_initialized` flag and runtime checks                       |
| Potential unsafe copying of layers (dangling pointers)                    | Deleted copy constructor and assignment operator (`Rule of 5` compliance) |

---

## 🌱 Key Learnings

* Importance of enforcing **parameter initialization** before model usage.
* Preventing object slicing and dangling pointers via **explicit copy prevention**.
* Design choices in managing model ownership (layers are owned by the container).

---

## 🚧 Future Improvements

* **🚚 Move Semantics (C++11 and later):**
  To allow safe movement of `Sequential` objects:

  ```cpp
  Sequential(Sequential&& other) noexcept = default;
  Sequential& operator=(Sequential&& other) noexcept = default;
  ```

* **❌ Explicitly Delete Copy Operations:**
  To prevent accidental copying (since `BaseLayer*` is manually managed):

  ```cpp
  Sequential(const Sequential&) = delete;
  Sequential& operator=(const Sequential&) = delete;
  ```

* **🔄 `resetParameters()` Method (Optional):**
  Allow re-initialization of weights/biases without rebuilding the model:

  ```cpp
  void resetParameters();
  ```

* **🔧 `mutable` for `is_initialized` (Optional):**
  If in the future you need to modify `is_initialized` in `const` methods (like `forward()`), declare:

  ```cpp
  mutable bool is_initialized;
  ```

  *Not required currently but useful for lazy initialization patterns.*

* Export/Import functionality for **saving and loading models**.:

---

## 🎯 Summary

The `Sequential` class forms the **core model-building block** of this ML library, balancing flexibility, error safety, and simplicity — inspired by proven designs from PyTorch/Keras.
