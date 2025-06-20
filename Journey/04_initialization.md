# 📄 **04\_initialization.md — Parameter Initialization Module Documentation**

---

## **1. Overview**

This module provides multiple **parameter initialization strategies** for neural networks and machine learning models ensuring proper **variance control and convergence**.

---

## **2. Initialization Methods Supported**

| Method         | Purpose                             | Equation / Range                             |
| -------------- | ----------------------------------- | -------------------------------------------- |
| Random Uniform | Random values in `[a, b]`           | `U(a, b)`                                    |
| Random Normal  | Gaussian with mean=`a`, std=`b`     | `N(a, b^2)`                                  |
| Xavier Uniform | Xavier Glorot for sigmoid/tanh      | `U(-sqrt(6 / (in+out)), sqrt(6 / (in+out)))` |
| Xavier Normal  | Xavier Gaussian for sigmoid/tanh    | `N(0, 2 / (in+out))`                         |
| He Uniform     | He Uniform for ReLU                 | `U(-sqrt(6 / in), sqrt(6 / in))`             |
| He Normal      | He Normal for ReLU                  | `N(0, 2 / in)`                               |
| LeCun Uniform  | LeCun Uniform for SELU              | `U(-sqrt(3 / in), sqrt(3 / in))`             |
| LeCun Normal   | LeCun Normal for SELU               | `N(0, 1 / in)`                               |
| Orthogonal     | Orthogonal matrix (for square only) | Gram-Schmidt                                 |
| Bias           | Constant bias value                 | `bias_value`                                 |

---

## **3. Function Summary**

### 🔹 `initializeParameters()`

**Arguments:**

| Parameter     | Description                                     |
| ------------- | ----------------------------------------------- |
| in\_features  | Number of input units                           |
| out\_features | Number of output units                          |
| method        | Initialization method (enum `InitMethod`)       |
| seed          | Manual random seed (default: 21)                |
| a, b          | Parameters for distribution (Uniform or Normal) |
| sparsity      | Fraction \[0,1] of weights forced to 0          |
| bias\_value   | Used when method is `BIAS`                      |

---

## **4. Special Features**

✔️ **Sparsity support** — Any method can set random elements to zero.
✔️ **Manual seed control** — Ensures reproducibility.
✔️ Orthogonalization uses **Gram-Schmidt**.
✔️ **Clamp applied for Normal Distribution** to avoid extreme values.

---

## **5. Important Notes**

* Orthogonal init requires **square matrices** (`in_features == out_features`).
* Bias initialized only via the `BIAS` method.
* Automatically handles **batch size 0/shape errors** via `assert`.

---

## **6. To-Do / Future Enhancements**

* [ ] Add **Kaiming initialization with fan\_in/fan\_out control**.
* [ ] Support for **mixed-type tensors**.
* [ ] **Layer-specific initialization wrappers**.

---

## **7. Limitations**

* No direct support for **per-layer customized scaling** yet.
* Orthogonal init error for non-square matrices is handled by error message.
