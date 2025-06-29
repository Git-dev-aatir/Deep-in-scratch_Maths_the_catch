# ⚡ Activation Layer Documentation

## 📝 Overview
This document covers the **Activation Utilities** and **Activation Layer** components of the neural network library. These elements work together to provide non-linear transformations essential for neural network operations.

---

## 🔥 Activation Utilities

### 🧩 ActivationType Enum
| Enum Value      | Activation Function | Formula |
|-----------------|---------------------|---------|
| `RELU`         | Rectified Linear Unit | `f(x) = max(0, x)` |
| `LEAKY_RELU`   | Leaky ReLU          | `f(x) = x > 0 ? x : αx` |
| `SIGMOID`      | Sigmoid             | `f(x) = 1 / (1 + e^{-x})` |
| `TANH`         | Hyperbolic Tangent  | `f(x) = tanh(x)` |
| `LINEAR`       | Linear              | `f(x) = x` |
| `SOFTMAX`      | Softmax             | `f(x_i) = e^{x_i} / Σ e^{x_j}` |
| `SELU`         | Scaled ELU          | `λ * (x > 0 ? x : α(e^x - 1))` |

### ⚙️ Core Functions
```cpp
// Apply activation to input vector
std::vector<double> applyActivation(
const std::vector<double>& x,
ActivationType act_type,
double alpha = 0.01,
double lambda = 1.0507
);

// Compute activation derivatives
std::vector<double> activationDerivative(
const std::vector<double>& x,
ActivationType act_type,
double alpha = 0.01,
double lambda = 1.0507
);

// Convert enum to string
std::string activationTypeToString(ActivationType act_type);
```


### 🛠️ Implementation Highlights
1. **Numerical Stability**:
   - Softmax uses max subtraction: `exp(x_i - max(x))`
   - Handles near-zero sum cases
2. **Special Derivatives**:
   - Sigmoid: `f'(x) = f(x)(1 - f(x))`
   - Tanh: `f'(x) = 1 - f(x)^2`
   - SELU: `λ * (x > 0 ? 1 : αe^x)`
3. **Edge Cases**:
   - Returns uniform distribution when softmax sum ≈0
   - Throws errors for unsupported activation types

---

## ⚡ ActivationLayer Class

### 🏗️ Class Definition
```cpp
class ActivationLayer : public BaseLayer {
    ActivationType activation_type;
    std::vector<double> input_cache;
    double alpha; // For LeakyReLU/SELU
    double lambda; // For SELU

    public:
    ActivationLayer(ActivationType act_type,
    double alpha = 0.01,
    double lambda = 1.0507);
    std::vector<double> forward(const std::vector<double>& input) override;

    std::vector<double> backward(const std::vector<double>& grad_output) override;
    void summary() const override;

};
```


### 🔑 Key Features
- **Input Caching**: Stores raw inputs for efficient backpropagation
- **Parameter Handling**:
  - Auto-corrects SELU parameters to defaults
  - Configurable α and λ values
- **Special Case Handling**:
  - Softmax derivatives deferred to loss function
  - SELU uses paper defaults when unspecified

### 🚀 Forward Pass
```cpp
std::vector<double> ActivationLayer::forward(const std::vector<double>& input) {
    input_cache = input; // Cache inputs
    return applyActivation(input, activation_type, alpha, lambda);
}
```


### 🔙 Backward Pass
```cpp
std::vector<double> ActivationLayer::backward(
const std::vector<double>& grad_output)
{
    // Special handling for softmax
    if (activation_type == ActivationType::SOFTMAX) {
        return grad_output; // Defer to loss function
    }

    auto deriv = activationDerivative(input_cache, activation_type, alpha, lambda);
    std::vector<double> grad_input(grad_output.size());

    // Element-wise gradient multiplication
    for (size_t i = 0; i < grad_output.size(); ++i) {
        grad_input[i] = grad_output[i] * deriv[i];
    }
    return grad_input;
}
```


### 📊 Performance Characteristics
| Operation | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| Forward   | O(n)            | O(n)            |
| Backward  | O(n)            | O(1)            |

---

## 🚀 Usage Example
```cpp
// Create SELU activation layer
ActivationLayer selu_layer(ActivationType::SELU);

// Forward pass
std::vector<double> input = {1.0, -1.0, 0.5};
auto output = selu_layer.forward(input);

// Backward pass
std::vector<double> grad_output = {0.2, 0.2, 0.2};
auto grad_input = selu_layer.backward(grad_output);

// Inspect layer
selu_layer.summary();
// Output: Activation Layer: SELU (alpha=1.67326, lambda=1.0507)
```

---

## ⚠️ Limitations
1. **Batch Processing**:
   - Only supports single vectors, not batches
2. **Softmax Derivatives**:
   - Requires pairing with cross-entropy loss
3. **Parameter Learning**:
   - α and λ are fixed after construction
4. **Numerical Precision**:
   - Limited to double precision

## 🚧 Future Improvements
1. **Batch Processing Support**:
```cpp
std::vector<std::vector<double>> forwardBatch(
    const std::vector<std::vector<double>>& batch
);
```


2. **Learnable Parameters**:
```cpp
void setAlpha(double new_alpha);
void setLambda(double new_lambda);
```


3. **Advanced Activations**:
- Add GELU, Swish, Mish functions
- Support learnable PReLU

4. **GPU Acceleration**:
- Implement CUDA kernels for activations
- Add half-precision support

5. **Automatic Differentiation**:
```cpp
std::pair<std::vector<double>, std::vector<double>>
forwardWithDerivative(const std::vector<double>& input);
```

---


## 📊 Activation Function Recommendations
| Use Case               | Recommended Activation |
|------------------------|------------------------|
| Hidden Layers          | ReLU/Leaky ReLU       |
| Binary Classification  | Sigmoid               |
| Multi-class Output     | Softmax               |
| RNNs/LSTMs            | Tanh                  |
| Self-normalizing Nets  | SELU                  |
