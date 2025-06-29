# 🚀 Optimizers Documentation

---

## 📝 Overview  
The `BaseOptim` class defines the core interface for all optimization algorithms in the neural network library. It provides a standardized way to update model parameters during training.

---

## 🏗️ BaseOptim Class Design

```cpp
class BaseOptim {
public:
    virtual ~BaseOptim() = default;
    
    /**
     * @brief Update parameters for a set of layers
     * @param layers Vector of layer pointers to update
     * @param batch_size Batch size used in current step
     */
    virtual void step(std::vector layers, size_t batch_size) = 0;

    /**
     * @brief Post-step operations (e.g., state updates)
     */
    virtual void afterStep() = 0;
    
    /**
     * @brief Set the learning rate
     * @param l极 New learning rate value
     */
    virtual void setLearningRate(double lr) = 0;
    
    /**
     * @brief Decay the learning rate
     * @param decay_factor Multiplier for current learning rate
     */
    virtual void decayLearningRate(double decay_factor) = 0;
    
    /**
     * @brief Get current learning rate
     * @return Current learning rate value
     */
    virtual double getLearningRate() const = 0;
};
```

---

## 🔑 Key Methods

### 1. `step()`
- **Purpose**: Core parameter update logic
- **Parameters**:
  - `layers`: Pointers to layers with trainable parameters
  - `batch_size`: Current batch size (for gradient scaling)
- **Responsibilities**:
  - Apply optimization algorithm to update parameters
  - Use stored gradients from layers
  - Handle batch size scaling appropriately

---

### 2. `afterStep()`
- **Purpose**: Post-update operations
- **Typical Use Cases**:
  - Increment step counters
  - Update momentum buffers
  - Apply weight decay
  - Schedule learning rate changes

---

### 3. Learning Rate Management
| Method | Purpose | Parameters |
|--------|---------|------------|
| `setLearningRate()` | Set absolute learning rate | `lr`: New learning rate |
| `decayLearningRate()` | Multiply current LR | `decay_factor`: Multiplier (0.0-1.0) |
| `getLearningRate()` | Get current LR | Returns current value |

---

## 🏗️ Implementation Requirements

### Parameter Update Workflow
1. **Forward Pass**: Compute predictions
2. **Backward Pass**: Calculate gradients
3. **Optimization Step**:
   ```cpp
   optim.step(model.getLayers(), current_batch_size);
   optim.afterStep();
   ```
4. **Gradient Reset**: Clear gradients for next batch

### Layer Interface Expectations
Layers must implement:
```cpp
class BaseLayer {
public:
    // Gradient access
    virtual std::vector>& getGradWeights() = 0;
    virtual std::vector& getGradBiases() = 0;
    
    // Parameter access
    virtual std::vector>& getWeights() = 0;
    virtual std::vector& getBiases() = 0;
};
```

---

## 🚧 Future Optimizer Implementations

### 1. Stochastic Gradient Descent (SGD) 
>(already implented but can be improved)

```cpp
class SGD : public BaseOptim {
private:
    double lr;
    double momentum;
    // ... state buffers
public:
    void step(std::vector layers, size_t batch_size) override;
    // ... other overrides
};
```

### 2. Adam Optimizer
```cpp
class Adam : public BaseOptim {
private:
    double lr;
    double beta1, beta2;
    int step_count = 0;
    // ... moment buffers
public:
    void step(std::vector layers, size_t batch_size) override;
    void afterStep() override; // Update step count
    // ... other overrides
};
```

### 3. RMSprop
```cpp
class RMSprop : public BaseOptim {
private:
    double lr;
    double decay_rate;
    // ... squared gradient buffers
public:
    void step(std::vector layers, size极 batch_size) override;
    // ... other overrides
};
```

---

## ⚙️ Best Practices

---

### 1. Gradient Scaling
```cpp
// In step() implementation:
for (auto& grad : gradients) {
    grad /= batch_size;  // Average gradients over batch
}
```

### 2. State Initialization
```cpp
// In constructor:
Adam(double lr = 0.001, double beta1 = 0.9, double beta2 = 0.999) 
    : lr(lr), beta1(beta1), beta2(beta2) {}
```

### 3. Learning Rate Scheduling
```cpp
void afterStep() override {
    step_count++;
    if (step_count % decay_interval == 0) {
        lr *= decay_factor;
    }
}
```

### 4. Numerical Stability
```cpp
// Adam implementation example:
double m_hat = m / (1 - pow(beta1, step_count));
double v_hat = v / (1 - pow(beta2, step_count));
param -= lr * m_hat / (sqrt(v_hat) + epsilon);
```

---

## 🚀 Usage Example

```cpp
// Create model and optimizer
SequentialModel model;
model.addLayer(new DenseLayer(784, 128));
model.addLayer(new ActivationLayer(ActivationType::RELU));
model.addLayer(new DenseLayer(128, 10));

Adam optim(0.001);

// Training loop
for (auto& batch : dataloader) {
    auto output = model.forward(batch.input);
    auto loss = Losses::cross_entropy_loss(batch.target, output);
    auto grad = Losses::cross_entropy_derivative(batch.target, output);
    model.backward(grad);
    
    optim.step(model.getLayers(), batch.size());
    optim.afterStep();
    
    model.zeroGradients();
}
```

---

## 📊 Optimizer Comparison

| Optimizer | Key Features | Best For |
|-----------|-------------|----------|
| SGD       | Simple, no state | Basic models |
| SGD+Momentum | Velocity buffer | Faster convergence |
| Adam      | Adaptive LR, momentum | Most architectures |
| RMSprop   | Adaptive per-parameter LR | RNNs |
| Adagrad   | Accumulated squared gradients | Sparse data |

---

## 🚧 Future Improvements

1. **Weight Decay**:
   ```
   virtual void setWeightDecay(double decay) = 0;
   ```

2. **Gradient Clipping**:
   ```
   virtual void setGradientClip(double threshold) = 0;
   ```

3. **Per-Layer Settings**:
   ```
   void setLayerSettings(BaseLayer* layer, OptimSettings settings);
   ```

4. **Serialization**:
   ```
   virtual void saveState(const std::string& path) = 0;
   virtual void loadState(const std::string& path) = 0;
   ```
   