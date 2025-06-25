# 📐 **Losses**

---

## 📦 Loss Functions Module

This module provides **loss functions** and their **derivatives**, used for training and evaluating models like regressors, classifiers, and SVMs.

---

## 🔍 Overview

✔️ Supports **single sample** and **batch** versions
✔️ Handles **logits vs probabilities** (where applicable)
✔️ Performs **input validation** (size, emptiness, shape consistency)
✔️ Implements:

* Mean Squared Error (MSE)
* Mean Absolute Error (MAE)
* Binary Cross Entropy (BCE)
* Categorical Cross Entropy (CE)
* Hinge Loss (SVM)

---

## 📂 Files

| File                                   | Description                         |
| -------------------------------------- | ----------------------------------- |
| `include/Metrics/Losses.h`             | Function declarations (header file) |
| `src/Metrics/Losses/mse.cpp`           | MSE implementation                  |
| `src/Metrics/Losses/mae.cpp`           | MAE implementation                  |
| `src/Metrics/Losses/bce.cpp`           | BCE implementation                  |
| `src/Metrics/Losses/cross_entropy.cpp` | Cross Entropy implementation        |
| `src/Metrics/Losses/hinge.cpp`         | Hinge Loss implementation           |

---

## **General Notes**

* **Single Sample Input**: `std::vector<double>` (1D vector)

* **Batch Input**: `std::vector<std::vector<double>>` (2D matrix)

* **Error Checks**:

  * Shape matching (`y_true.size() == y_pred.size()`)
  * Non-empty vectors/batches
  * Consistent batch row sizes

* **Cross Entropy & BCE**:

  * Supports raw logits via `from_logits=true`
  * Automatically applies **sigmoid** (BCE) or **softmax** (Cross Entropy) when required.

---

## 📚 **Available Loss Functions**

---

## 🔹 1. **Mean Squared Error (MSE)**

| Function                 | Description                       |
| ------------------------ | --------------------------------- |
| `mse_loss()`             | Single sample MSE                 |
| `mse_loss_batch()`       | Batch MSE (averaged over samples) |
| `mse_derivative()`       | Gradient for single sample        |
| `mse_derivative_batch()` | Gradient for batch                |

#### **Loss Formula**:

$$
MSE = \frac{1}{C} \sum_{j=1}^{C} \left( y_{true}^{(j)} - y_{pred}^{(j)} \right)^2
$$

#### **Derivative (Per Element)**:

$$
\frac{\partial L}{\partial y_{pred}} = \frac{2}{C} \left( y_{pred} - y_{true} \right)
$$

---

## 🔹 2. **Mean Absolute Error (MAE)**

| Function                 | Description                   |
| ------------------------ | ----------------------------- |
| `mae_loss()`             | Single sample MAE             |
| `mae_loss_batch()`       | Batch MAE                     |
| `mae_derivative()`       | Subgradient for single sample |
| `mae_derivative_batch()` | Subgradient for batch         |

#### **Loss Formula**:

$$
MAE = \frac{1}{C} \sum_{j=1}^{C} \left| y_{true}^{(j)} - y_{pred}^{(j)} \right|
$$

#### **Derivative (Subgradient)**:

$$
\frac{\partial L}{\partial y_{pred}} = \frac{sign(y_{pred} - y_{true})}{C}
$$

---

## 🔹 3. **Binary Cross Entropy (BCE)**

| Function                 | Description                  |
| ------------------------ | ---------------------------- |
| `bce_loss()`             | Single sample BCE            |
| `bce_loss_batch()`       | Batch BCE                    |
| `bce_derivative()`       | Derivative for single sample |
| `bce_derivative_batch()` | Derivative for batch         |

#### **Loss Formula**:

$$
BCE = -\frac{1}{C} \sum_{j=1}^{C} \left[ y_{true}^{(j)} \log(y_{pred}^{(j)}) + (1 - y_{true}^{(j)}) \log(1 - y_{pred}^{(j)}) \right]
$$

* If **`from_logits=true`**, predictions pass through **sigmoid** internally.

#### **Derivative**:

$$
\frac{\partial L}{\partial y_{pred}} = \frac{(y_{pred} - y_{true})}{C \cdot y_{pred}(1 - y_{pred})}
$$

*(Logits handling includes scaling this via sigmoid derivative if `from_logits=true`.)*

---

## 🔹 4. **Categorical Cross Entropy (CE)**

| Function                           | Description                  |
| ---------------------------------- | ---------------------------- |
| `cross_entropy_loss()`             | Single sample CE             |
| `cross_entropy_loss_batch()`       | Batch CE                     |
| `cross_entropy_derivative()`       | Derivative for single sample |
| `cross_entropy_derivative_batch()` | Derivative for batch         |

#### **Loss Formula**:

$$
CE = -\sum_{j=1}^{C} y_{true}^{(j)} \log(y_{pred}^{(j)})
$$

* If **`from_logits=true`**, applies **softmax** internally to logits.

#### **Derivative**:

$$
\frac{\partial L}{\partial y_{pred}} = y_{pred} - y_{true}
$$

---

## 🔹 5. **Hinge Loss (SVM)**

| Function                   | Description                  |
| -------------------------- | ---------------------------- |
| `hinge_loss()`             | Single sample Hinge loss     |
| `hinge_loss_batch()`       | Batch Hinge loss             |
| `hinge_derivative()`       | Derivative for single sample |
| `hinge_derivative_batch()` | Derivative for batch         |

#### **Loss Formula**:

$$
Hinge = \frac{1}{C} \sum_{j=1}^{C} \max \left(0, 1 - y_{true}^{(j)} \cdot y_{pred}^{(j)} \right)
$$

#### **Derivative**:

$$
\frac{\partial L}{\partial y_{pred}} =
\begin{cases}
\frac{-y_{true}^{(j)}}{C} & \text{if } 1 - y_{true}^{(j)} \cdot y_{pred}^{(j)} > 0 \\
0 & \text{otherwise}
\end{cases}
$$

---

## ⚠️ Error Handling

✔️ Empty vector detection
✔️ Size mismatch (throws `std::invalid_argument`)
✔️ Batch shape consistency

---

## 🛠️ **Problems Faced & Solutions**

| Problem                                              | Solution                                                                                                                      |
| ---------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Handling derivative for softmax activation           | Decided to throw an explicit exception in derivative function; derivative handled with cross-entropy for numerical stability. |
| Inconsistent batch sizes causing crashes             | Added strict size and shape checks before computation; throws exceptions if mismatched.                                       |
| Correctly supporting logits vs probability inputs    | Introduced `from_logits` flag to toggle internal sigmoid/softmax application to raw model outputs.                            |
| Managing zero division in BCE derivative calculation | Added epsilon-clamping in denominator to avoid division by zero in edge cases.                                                |
| Ensuring correct sign in hinge loss derivative       | Carefully tested sign logic against known SVM hinge loss gradients.                                                           |

---

## 📚 **Key Learnings**

* Loss functions must validate inputs thoroughly to avoid runtime errors and inconsistent training results.
* Handling logits internally improves numerical stability and user convenience.
* Softmax derivative is complex and best combined with cross-entropy loss for efficiency and correctness.
* Clear exception handling aids debugging and guides users toward proper API usage.
* Batch versions improve performance but require careful shape management.

---

## 🏷️ Namespace

```
Losses
```

---

## ⏳ **Future Improvements (To-Do)**

* [ ] Add support for **weighted loss functions**.
* [ ] Implement **multi-label BCE**.
* [ ] Provide **loss reduction modes** (mean, sum, none).
* [ ] Support **masking** for padded sequences.
