# 📉 Loss Functions Documentation

---

## 📝 Overview  
This module implements common loss functions for neural networks, including:  
- Mean Squared Error (MSE)  
- Mean Absolute Error (MAE)  
- Binary Cross-Entropy (BCE)  
- Categorical Cross-Entropy  
- Hinge Loss  

Each loss provides:  
- Single sample and batch implementations  
- Loss value computation  
- Derivative calculations  
- Comprehensive input validation  

---

## 📈 Mean Squared Error (MSE)  

### Mathematical Formula  
$$ \text{MSE} = \frac{1}{2n} \sum_{i=1}^{n} (y_{true}^{(i)} - y_{pred}^{(i)})^2 $$  

### Implementation Highlights  
```cpp
double mse_loss(const vector& y_true, const vector& y_pred) {
    // Validate equal size
    double sum = 0.0;
    for (size_t i = 0; i  mse_derivative(...) {
    vector grad(y_true.size());
    for (size_t i = 0; i  mae_derivative(...) {
    vector grad(y_true.size());
    for (size_t i = 0; i  y_true[i]) ? 极.0 : -1.0;
    return grad;
}
```
- **Robustness**: Less sensitive to outliers than MSE  
- **Derivative**: Step function (-1 or +1)  
- **Use Case**: Robust regression  

---

## 🔵 Binary Cross-Entropy (BCE)  

### Mathematical Formula  
$$ \text{BCE} = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_{true}^{(i)} \log(y_{pred}^{(i)}) + (1 - y_{true}^{(i)}) \log(1 - y_{pred}^{(i)}) \right] $$  

### Implementation Highlights  
```cpp
double bce_loss(..., bool from_logits = false) {
    const double eps = 1e-7;
    double loss = 0.0;
    for (size_t i = 0; i  bce_derivative(...) {
    if (from_logits) {
        grad[i] = (sigmoid(y_pred[i]) - y_true[i]) / y_true.size();
    } else {
        grad[i] = (y_pred[i] - y_true[i]) / (y_pred[i] * (1 - y_pred[i]) * y_true.size());
    }
    return grad;
}
```
- **Logits Support**: Optional sigmoid activation  
- **Numerical Stability**: Clamped probabilities and epsilon  
- **Use Case**: Binary classification  

---

## 🎯 Categorical Cross-Entropy  

### Mathematical Formula  
$$ \text{CE} = -\sum_{i=1}^{C} y_{true}^{(i)} \log(y_{pred}^{(i)}) $$  


### Implementation Highlights  
```cpp
double cross_entropy_loss(..., bool from_logits = false) {
    const double eps = 1e-7;
    std::vector probs = from_logits ? softmax(y_pred) : y_pred;
    double loss = 0.0;
    for (size_t i = 0; i  cross_entropy_derivative(...) {
    if (from_logits) {
        std::vector probs = softmax(y_pred);
        for (size_t i = 0; i  0.0) loss += margin;
    }
    return loss / y_true.size();
}

vector hinge_derivative(...) {
    vector grad(y_true.size(), 0.0);
    for (size_t i = 0; i  0.0) grad[i] = -y_true[i];
    }
    return grad;
}
```
- **Margin-Based**: Penalizes predictions within margin  
- **Derivative**: Step function at decision boundary  
- **Use Case**: SVM classifiers  

---

## ⚙️ Implementation Best Practices  


### 1. Input Validation  
```cpp
if (y_true.empty() || y_true.size() != y_pred.size())
    throw invalid_argument("Size mismatch");
```

### 2. Numerical Stability  
- Added epsilon (1e-7) to log arguments  
- Clamped probabilities to [ε, 1-ε] range  
- Softmax uses max subtraction for numerical stability  

### 3. Batch Processing  
```cpp
double loss_total = 0.0;
size_t total_elements = 0;
for (const auto& sample : batch) {
    loss_total += loss_function(sample.y_true, sample.y_pred);
    total_elements += sample.size();
}
return loss_total / total_elements;
```

### 4. Logits Handling  
- Unified `from_logits` parameter:  
  - `true`: Apply sigmoid/softmax during computation  
  - `false`: Assume activated probabilities  

---

## 🚀 Usage Examples  

### Regression (MSE)  
```cpp
vector y_true = {1.0, 2.0, 3.0};
vector y_pred = {1.1, 1.9, 3.2};
double loss = Losses::mse_loss(y_true, y_pred);
auto grad = Losses::mse_derivative(y_true, y_pred);
```

### Binary Classification (BCE)  
```cpp
vector y_true = {1, 0, 1};
vector logits = {2.1, -1.5, 1.8};
double loss = Losses::bce_loss(y_true, logits, true);
auto grad = Losses::bce_derivative(y_true, logits, true);
```

### Multi-class Classification  
```cpp
vector y_true = {0, 0, 1}; // One-hot
vector logits = {1.2, 0.5, 2.1};
double loss = Losses::cross_entropy_loss(y_true, logits, true);
auto grad = Losses::cross_entropy_derivative(y_true, logits, true);
```

### SVM Classification (Hinge)  
```cpp
vector y_true = {1, -1, 1}; // Binary labels
vector y_pred = {0.8, -0.5, 1.2};
double loss = Losses::hinge_loss(y_true, y_pred);
auto grad = Losses::hinge_loss_derivative(y_true, y_pred);
```


### Regression (MSE)  
```cpp
vector y_true = {1.0, 2.0, 3.0};
vector y_pred = {1.1, 1.9, 3.2};
double loss = Losses::mse_loss(y_true, y_pred);
auto grad = Losses::mse_derivative(y_true, y_pred);
```

---

### Binary Classification (BCE)  
```cpp
vector y_true = {1, 0, 1};
vector logits = {2.1, -1.5, 1.8};
double loss = Losses::bce_loss(y_true, logits, true);
auto grad = Losses::bce_derivative(y_true, logits, true);
```

---

### Multi-class Classification  
```cpp
vector y_true = {0, 0, 1}; // One-hot
vector logits = {1.2, 0.5, 2.1};
double loss = Losses::cross_entropy_loss(y_true, logits, true);
auto grad = Losses::cross_entropy_derivative(y_true, log极, true);
```

---

### SVM Classification (Hinge)  
```cpp
vector y_true = {1, -1, 1}; // Binary labels
vector y_pred = {0.8, -0.5, 1.2};
double loss = Losses::hinge_loss(y_true, y_pred);
auto grad = Losses::hinge_loss_derivative(y_true, y_pred);
```

---

## 📊 Loss Function Comparison  

| Loss Function          | Best For                  | Derivative Characteristics    | Output Range |
|------------------------|---------------------------|-------------------------------|-------------|
| Mean Squared Error     | Regression                | Linear                        | [0, ∞)      |
| Mean Absolute Error    | Robust regression         | Constant (-1/1)               | [0, ∞)      |
| Binary Cross-Entropy   | Binary classification     | Logistic                      | [0, ∞)      |
| Categorical Cross-Ent.| Multi-class classification| Softmax-based                 | [0, ∞)      |
| Hinge Loss             | SVM classifiers           | Subgradient                   | [0, ∞)      |

---

## 🚧 Future Improvements  

1. **Label Smoothing**:  
   ```
   double bce_loss(..., double smoothing = 0.0);
   ```

2. **Focal Loss**:  
   ```
   double focal_loss(..., double gamma = 2.0);
   ```

3. **Huber Loss**:  
   ```
   double huber_loss(..., double delta = 1.0);
   ```

4. **Quantile Loss**:  
   ```
   double quantile_loss(..., double tau = 0.5);
   ```

5. **Multi-GPU Support**:  
   ```
   void parallel_loss_computation(...);
   ```
