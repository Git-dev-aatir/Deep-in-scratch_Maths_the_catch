# 📄 **Optimizers Module Documentation**

---

## 🧩 **1. Overview**

The **Optimizers** module implements various gradient-based optimization algorithms to update the parameters of neural network layers during training. These optimizers manage how weights and biases are adjusted to minimize the loss function via backpropagation.

---

## 📂 **2. Folder Structure**

| Folder                 | Description                            |
| ---------------------- | -------------------------------------- |
| `./include/Optimizers` | Header files for all optimizer classes |
| `./src/Optimizers`     | Implementation files for optimizers    |

---

## ⚙️ **3. Optimizer Base Class**

### Class: `Optimizer` (`BaseOptim.h`)

* **Purpose:** Abstract base class defining the interface and basic operations shared by all optimizers.

* **Key Members:**

  * `learning_rate` — Learning rate hyperparameter controlling step size.

* **Main Methods:**

  * `step(std::vector<DenseLayer*>& layers)` (pure virtual) — Updates parameters of all provided dense layers.
  * `before_epoch(...)`, `after_epoch(...)` — Hooks for optional actions around epochs.
  * `step_per_sample(...)`, `step_after_batch(...)` — Hooks for per-sample and per-batch updates.
  * `clear_gradients(...)` — Clears stored gradients on all layers.

---

## ⚡ **4. Optimizer Implementations**

### 4.1 Stochastic Gradient Descent (SGD) — `SGD.h` / `SGD.cpp`

* **Description:** Updates parameters **after every training sample** (stochastic).

* **Key Methods:**

  * `step_per_sample(layers)` — Updates weights and biases by subtracting `learning_rate * gradients`.
  * `step_after_batch(layers)` — No-op (empty implementation).
  * `step(layers)` — No-op (empty implementation).

* **Notes:**

  * Efficient for online learning.
  * Parameter updates are immediate per sample.

---

### 4.2 Batch Gradient Descent (BatchGD) — `BatchGD.h` / `BatchGD.cpp`

* **Description:** Updates parameters **once after processing the entire batch** of samples.

* **Key Methods:**

  * `step(layers)` — Applies parameter updates using accumulated gradients.
  * `step_per_sample(layers)` — No-op (empty implementation).
  * `step_after_batch(layers)` — Calls `step(layers)`.

* **Notes:**

  * Gradients are accumulated over the batch before update.
  * Can be computationally expensive for large batches.

---

### 4.3 Mini-Batch Gradient Descent (MiniBatchGD) — `MiniBatchGD.h` / `MiniBatchGD.cpp`

* **Description:** Updates parameters **after a fixed number of samples (mini-batch)** have been processed.

* **Data Members:**

  * `mini_batch_size` — Number of samples per mini-batch.
  * `sample_count` — Tracks number of samples processed in current mini-batch.

* **Key Methods:**

  * `step_per_sample(layers)` — Increments `sample_count`; calls `step()` and resets gradients when mini-batch size reached.
  * `step(layers)` — Performs parameter updates.
  * `step_after_batch(layers)` — Handles remaining samples if batch ends prematurely.

* **Notes:**

  * Balances between computational efficiency and noise reduction.
  * Commonly used in practical training.

---

## 🔧 **5. Implementation Details**

* All optimizers manipulate the weights and biases stored in `DenseLayer` instances by directly accessing their references using `const_cast` (to modify from `const` getters).

* The parameter updates apply the classic gradient descent formula:

  $$
  \theta \leftarrow \theta - \eta \cdot \nabla_\theta L
  $$

  where \$\eta\$ is the learning rate, and \$\nabla\_\theta L\$ is the gradient for weights or biases.

* Gradients are expected to be accumulated in `DenseLayer` during backpropagation before calling optimizer steps.

* `clear_gradients` is used after updates to reset gradient storage for the next iteration.

---

## ⚠️ **6. Problems Faced and Solutions**

| Problem                                                            | Solution                                                                                                          |
| ------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| Accessing non-const weights and biases from `const` getter methods | Used `const_cast` to obtain modifiable references for in-place updates in optimizer step functions.               |
| Ensuring correct gradient clearing after parameter updates         | Added `clear_gradients()` calls immediately after update steps in SGD and MiniBatchGD to prevent stale gradients. |
| Handling incomplete mini-batches at epoch end in MiniBatchGD       | Implemented `step_after_batch` to update parameters if samples remain unprocessed when epoch ends prematurely.    |
| Empty/no-op implementations in some overridden virtual methods     | Provided empty bodies to satisfy interface requirements without unintended side effects.                          |

---

## 📚 **7. Key Learnings**

* Design of an **abstract optimizer base class** with hooks facilitates easy extension and integration of new optimizers.
* Separation of **per-sample**, **per-batch**, and **epoch-level** update steps provides flexibility for different optimization strategies.
* **Direct manipulation of layer parameters** via references improves efficiency but requires careful const-correctness handling.
* **Mini-batch gradient descent** strikes a balance between noisy updates (SGD) and computational overhead (batch GD).
* Proper **gradient management** (accumulation and clearing) is critical for correct optimizer functionality.
* Clear modular design enables future addition of more advanced optimizers (Momentum, Adam, RMSProp, etc.).

---

## 🏷️ **Namespace**

All optimizer classes currently reside in the **global namespace** but can be encapsulated inside an `Optimizers` namespace in the future for better modularity.

---

## ⏳ **8. Future Improvements**

* Add more advanced optimizers like **Momentum**, **Adam**, **RMSProp**.
* Implement **learning rate scheduling** and **adaptive learning rates**.
* Integrate optimizer hooks for **regularization** (weight decay, dropout).
* Refactor to avoid `const_cast` by redesigning layer interfaces or exposing non-const parameter access.
* Support GPU acceleration and parallel batch updates.

---

# 🏁 **Summary**

The Optimizers module provides the fundamental building blocks for updating neural network parameters during training. It currently supports Stochastic, Batch, and Mini-Batch Gradient Descent with extensible design patterns to accommodate more sophisticated methods in future development.
