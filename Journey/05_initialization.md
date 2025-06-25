# 📄 **Parameter Initialization Module Documentation**

---

## 🧩 **1. Overview**

This module offers various **parameter initialization methods** for neural networks and machine learning models to ensure stable training and faster convergence.

---

## 📂 **2. Files Used**

| File                               | Description              |
| ---------------------------------- | ------------------------ |
| `./include/Utils/Initialization.h` | Header declarations      |
| `./src/Utils/initialization.cpp`   | Function implementations |

---

## ⚙️ **3. Initialization Methods Supported**

| Method         | Purpose                                        | Equation / Range                                                                                    |
| -------------- | ---------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| Random Uniform | Random values uniformly in `[a, b]`            | \$U(a, b)\$                                                                                         |
| Random Normal  | Gaussian with mean = `a`, std = `b`            | \$\mathcal{N}(a, b^2)\$, clamped to \[a, b]                                                         |
| Xavier Uniform | Glorot uniform for sigmoid/tanh activations    | \$U\left(-\sqrt{\frac{6}{\text{in} + \text{out}}}, \sqrt{\frac{6}{\text{in} + \text{out}}}\right)\$ |
| Xavier Normal  | Glorot normal for sigmoid/tanh activations     | \$\mathcal{N}\left(0, \frac{2}{\text{in} + \text{out}}\right)\$                                     |
| He Uniform     | He uniform for ReLU activations                | \$U\left(-\sqrt{\frac{6}{\text{in}}}, \sqrt{\frac{6}{\text{in}}}\right)\$                           |
| He Normal      | He normal for ReLU activations                 | \$\mathcal{N}\left(0, \frac{2}{\text{in}}\right)\$                                                  |
| LeCun Uniform  | LeCun uniform for SELU activations             | \$U\left(-\sqrt{\frac{3}{\text{in}}}, \sqrt{\frac{3}{\text{in}}}\right)\$                           |
| LeCun Normal   | LeCun normal for SELU activations              | \$\mathcal{N}\left(0, \frac{1}{\text{in}}\right)\$                                                  |
| Orthogonal     | Orthogonal matrix initialization (square only) | Gram-Schmidt process                                                                                |
| Bias           | Constant bias value                            | Constant value                                                                                      |

---

## 🔧 **4. Function Summary**

### 🔹 `initializeParameters()`

| Parameter      | Description                                     |
| -------------- | ----------------------------------------------- |
| `in_features`  | Number of input features (columns)              |
| `out_features` | Number of output features (rows)                |
| `method`       | Initialization method (`InitMethod` enum)       |
| `seed`         | Random seed (default: 21)                       |
| `a`, `b`       | Distribution parameters (lower/mean, upper/std) |
| `sparsity`     | Fraction $\[0, 1]\$ of weights to zero out      |
| `bias_value`   | Constant value for bias initialization          |

---

## 🌟 **5. Special Features**

✔️ Sparsity support: randomly zeroes weights per `sparsity` parameter
✔️ Manual seed control for reproducibility
✔️ Clamp applied for normal distribution samples to avoid outliers
✔️ Orthogonal initialization using Gram-Schmidt (only for square matrices)
✔️ Bias initialization sets all weights to `bias_value`

---

## ⚠️ **6. Important Notes**

* Orthogonal init requires square matrices (`in_features == out_features`).
* Bias init applies only when `method == InitMethod::BIAS`.
* Input validation via `assert` for dimensions and sparsity range.
* Orthogonal init prints error and returns zero matrix if dimensions mismatch.

---

## 🛠️ **7. Problems Faced & Solutions**

| Problem                                                       | Solution                                                                                                                |
| ------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Clamping normal distribution values to avoid extreme outliers | Implemented a `clamp` function restricting sampled values between `[a, b]` after sampling.                              |
| Orthogonal initialization limited to square matrices          | Added runtime check and error message; gracefully return zero matrix when invalid.                                      |
| Sparse initialization not native in std distributions         | Added a uniform random sparsity mask that zeroes weights probabilistically post initialization.                         |
| Bias vector initialization not matching weight init interface | Created special case to initialize bias as vector using the same initialization function with shape `[output_size, 1]`. |
| Ensuring reproducibility across runs with random seed         | Passed `seed` to random number generator (`std::mt19937`) used throughout initialization.                               |

---

## 📚 **8. Key Learnings**

* Proper initialization is critical to prevent vanishing/exploding gradients and improve training stability.
* Different activations require different initialization scaling (e.g., Xavier for sigmoid/tanh, He for ReLU).
* Sparse initialization can be useful for model compression or regularization but requires explicit masking logic.
* Orthogonal initialization requires careful matrix dimension handling and vector orthogonalization (Gram-Schmidt).
* Using explicit random seeds ensures deterministic behavior for debugging and experiments.

---

## 🗂️ **9. To-Do / Future Enhancements**

* [ ] Add Kaiming initialization with fan\_in/fan\_out controls.
* [ ] Support structured sparsity patterns.
* [ ] Extend support for convolutional kernels.
* [ ] Replace `assert` with exception handling.
* [ ] Add mixed-precision tensor initialization support.

---

## 🚧 **10. Limitations**

* No per-layer custom scaling or tuning yet.
* Orthogonal init restricted to square matrices; no fallback for non-square.

---

## 🏷️ **11. Namespace**

```
Utils
```
