# 03\_losses.md

## Loss Functions Module

This module provides a set of commonly used loss functions and their derivatives, designed to be used in neural network training and evaluation.

*NOTE* :
* All losses are made thinking of labels (y) as a vector of one-hot-vectors or 2-D matrices.
* The labels (y) can also be considered as a vector of single element sub-vectors (still 2-D matrices).

---

## **Available Loss Functions:**

### 1. **Mean Squared Error (MSE)**

* **Function**: `mse_loss(const vector<vector<double>> &y_true, const vector<vector<double>> &y_pred)`
* **Description**: Computes the Mean Squared Error between the predicted and true values.
* **Formula**:

  $$
  MSE = \frac{1}{N \times C} \sum_{i=1}^{N} \sum_{j=1}^{C} (y_{true}^{(i,j)} - y_{pred}^{(i,j)})^2
  $$
* **Returns**: `double` (MSE loss)
* **Throws**: `invalid_argument` if batch or class size mismatch occurs.

#### **Derivative:**

* **Function**: `mse_derivative(const vector<vector<double>> &y_true, const vector<vector<double>> &y_pred)`
* **Returns**:

  $$
  \frac{2(y_{pred} - y_{true})}{N \times C}
  $$
* **Output**: 2D vector matching the shape of input.

---

### 2. **Mean Absolute Error (MAE)**

* **Function**: `mae_loss(const vector<vector<double>> &y_true, const vector<vector<double>> &y_pred)`
* **Description**: Computes the Mean Absolute Error.
* **Formula**:

  $$
  MAE = \frac{1}{N \times C} \sum_{i=1}^{N} \sum_{j=1}^{C} |y_{true}^{(i,j)} - y_{pred}^{(i,j)}|
  $$
* **Returns**: `double` (MAE loss)
* **Throws**: `invalid_argument` if batch or class size mismatch occurs.

#### **Derivative (Subgradient):**

* **Function**: `mae_derivative(const vector<vector<double>> &y_true, const vector<vector<double>> &y_pred)`
* **Returns**:

  * `1 / (N * C)` if `y_pred > y_true`
  * `-1 / (N * C)` if `y_pred < y_true`
  * `0` otherwise.

---

### 3. **Binary Cross Entropy (BCE)**

* **Function**: `bce_loss(const vector<vector<double>> &y_true, const vector<vector<double>> &y_pred)`
* **Description**: Computes Binary Cross-Entropy loss for binary classification tasks.
* **Formula**:

  $$
  BCE = -\frac{1}{N \times C} \sum_{i=1}^{N} \sum_{j=1}^{C} \left[y_{true}^{(i,j)} \log(y_{pred}^{(i,j)}) + (1 - y_{true}^{(i,j)}) \log(1 - y_{pred}^{(i,j)})\right]
  $$
* **Returns**: `double` (BCE loss)
* **Note**: Clamps predictions to avoid log(0).

#### **Derivative:**

* **Function**: `bce_derivative(const vector<vector<double>> &y_true, const vector<vector<double>> &y_pred)`
* **Returns**:

  $$
  \frac{(y_{pred} - y_{true})}{y_{pred} (1 - y_{pred}) \times N \times C}
  $$
* **Clamped to avoid division by zero**.

---

### 4. **Categorical Cross-Entropy (Softmax Output)**

* **Function**: `cross_entropy_loss(const vector<vector<double>> &y_true, const vector<vector<double>> &y_pred)`
* **Description**: Computes Cross-Entropy loss for multi-class classification (softmax probabilities).
* **Formula**:

  $$
  CE = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{C} y_{true}^{(i,j)} \log(y_{pred}^{(i,j)})
  $$
* **Returns**: `double` (Cross-Entropy loss)
* **Note**: Only true class probabilities are considered (one-hot labels assumed).

#### **Derivative:**

* **Function**: `cross_entropy_derivative(const vector<vector<double>> &y_true, const vector<vector<double>> &y_pred)`
* **Returns**:

  $$
  \frac{y_{pred} - y_{true}}{N}
  $$

---

### 5. **Hinge Loss (SVM)**

* **Function**: `hinge_loss(const vector<vector<double>> &y_true, const vector<vector<double>> &y_pred)`
* **Description**: Computes the hinge loss for both binary and multi-class classification (SVM style).
* **Formula (Binary)**:

  $$
  Hinge = \frac{1}{N} \sum_{i=1}^{N} \max(0, 1 - y_{true}^{(i)} \cdot y_{pred}^{(i)})
  $$
* **Formula (Multi-class)**:

  $$
  Hinge = \frac{1}{N} \sum_{i=1}^{N} \sum_{j \ne y_{true}} \max(0, 1 - (y_{pred}^{(y_{true})} - y_{pred}^{(j)}))
  $$
* **Returns**: `double` (Hinge Loss)

#### **Derivative:**

* **Function**: `hinge_loss_derivative(const vector<vector<double>> &y_true, const vector<vector<double>> &y_pred)`
* **Returns**:

  * Binary case:

    $$
    \frac{-y_{true}}{N} \quad \text{if margin < 1}
    $$
  * Multi-class case: Gradient applied to correct and incorrect classes based on margin violations.

---

## **Error Handling:**

* All functions check for:

  * Matching shapes of `y_true` and `y_pred`
  * Non-zero batch size and class size
  * Consistent row sizes within `y_true` and `y_pred`
* `std::invalid_argument` exceptions are thrown on violations.

---

## **Namespace:**

```
Losses
```
