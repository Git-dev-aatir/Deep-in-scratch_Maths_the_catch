# 🔥 Activation Functions Documentation

---

## 📝 Overview
The `Activations` namespace provides efficient implementations of common neural network activation functions and their derivatives. It supports:
- **Scalar operations**: Single-value computations
- **Vector operations**: Element-wise processing
- **Batch processing**: Efficient handling of multiple vectors

## 🧩 Core Functions

### 1. **Sigmoid**
```cpp
double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
```
**Formula**:  
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$  
**Derivative**:  
$$\sigma'(x) = \sigma(x) \cdot (1 - \sigma(x))$$

### 2. **ReLU**
```cpp
double relu(double x) { return (x > 0) ? x : 0; }
```
**Formula**:  
$$\text{ReLU}(x) = \max(0, x)$$  
**Derivative**:  
$$\text{ReLU}'(x) = \begin{cases} 1 & x > 0 \\ 0 & \text{otherwise} \end{cases}$$

### 3. **Tanh**
```cpp
double tanh(double x) { return std::tanh(x); }
```
**Formula**:  
$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$  
**Derivative**:  
$$\tanh'(x) = 1 - \tanh^2(x)$$

### 4. **Softmax**
```cpp
std::vector softmax(const std::vector& x) {
    double max_elem = *std::max_element(x.begin(), x.end());
    double sum = 0.0;
    std::vector exp_vals;
    for (double val : x) {
        double exp_val = std::exp(val - max_elem);
        exp_vals.push_back(exp_val);
        sum += exp_val;
    }
    // ... normalization
}
```
**Formula**:  
$$\text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}$$  
**Properties**:
- Outputs sum to 1
- Numerically stable (max subtraction)

---

## ⚙️ Implementation Highlights

### 1. **Numerical Stability**
- **Softmax**: Uses max subtraction to prevent overflow
  ```cpp
  double max_elem = *std::max_element(x.begin(), x.end());
  double exp_val = std::exp(val - max_elem);
  ```
- **Division Protection**: Handles near-zero sum cases
  ```cpp
  if (sum (x.size(), 1.0/x.size());
  ```

### 2. **Efficiency Optimizations**
- **Pre-allocation**: `reserve()` used before vector operations
- **Batch Processing**: Minimal overhead through vectorized calls
- **Reusable Logic**: Scalar functions drive vector/batch implementations

### 3. **Consistent Interface**
| Function Type | Sigmoid | ReLU | Tanh | Softmax |
|---------------|---------|------|------|---------|
| **Scalar**    | ✓       | ✓    | ✓    | ✗       |
| **Vector**    | ✓       | ✓    | ✓    | ✓       |
| **Batch**     | ✓       | ✓    | ✓    | ✓       |
| **Derivative**| ✓       | ✓    | ✓    | ✗       |

---

## 🚀 Usage Examples

### Single Value
```cpp
double x = 2.0;
double sig = Activations::sigmoid(x);
double rel = Activations::relu(x);
double d_sig = Activations::sigmoid_derivative(x);
```

### Vector Processing
```cpp
std::vector input = {1.0, -1.0, 0.5};
auto activated = Activations::tanh(input);
auto derivatives = Activations::relu_derivative(input);
```

### Batch Processing
```cpp
std::vector> batch = {
    {1.0, 2.0, 3.0},
    {-1.0, 0.0, 1.0}
};
auto softmax_results = Activations::softmax_batch(batch);
auto sig_derivs = Activations::sigmoid_derivative_batch(batch);
```

---

## ⚠️ Edge Cases & Validation

1. **Empty Input Handling**:
   ```cpp
   if (x.empty()) throw std::invalid_argument("softmax: Input vector cannot be empty");
   ```
2. **Zero Vectors**:
   - Softmax returns uniform distribution for near-zero sums
3. **Large Values**:
   - ReLU handles negative values correctly
   - Tanh stays within [-1, 1] range

---

## 📊 Performance Characteristics

| Operation | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| Scalar    | O(1)            | O(1)            |
| Vector    | O(n)            | O(n)            |
| Batch     | O(m×n)          | O(m×n)          |

---

## 🚧 Future Improvements

1. **GPU Acceleration**:
   ```cpp
   #ifdef USE_CUDA
   __device__ double cuda_sigmoid(double x);
   #endif
   ```

2. **Approximate Functions**:
   ```cpp
   double fast_sigmoid(double x) {
       return 0.5 * (x / (1 + abs(x))) + 0.5;
   }
   ```

3. **Sparse Activations**:
   ```cpp
   SparseVector relu_sparse(const SparseVector& x);
   ```

4. **Fused Operations**:
   ```cpp
   std::pair, std::vector> 
   sigmoid_with_derivative(const std::vector& x);
   ```

5. **Additional Activations**:
   ```cpp
   double swish(double x) { return x * sigmoid(x); }
   double gelu(double x) { /* Gaussian Error Linear Unit */ }
   ```
