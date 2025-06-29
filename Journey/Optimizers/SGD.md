# 🚀 SGD Optimizer Documentation

---

## 📝 Overview
The `SGD` class implements the Stochastic Gradient Descent optimizer with momentum and learning rate scheduling. It extends the `BaseOptim` interface to provide:
- Basic SGD with or without momentum
- Per-layer velocity tracking
- Flexible learning rate scheduling
- Batch-size aware gradient 

---

## 🏗️ Class Definition
```cpp
class SGD : public BaseOptim {
private:
    double learning_rate;
    double initial_lr;
    double momentum;
    std::unordered_map>> velocity_weights;
    std::unordered_map> velocity_biases;
    std::function lr_scheduler;
    size_t step_count = 0;

public:
    SGD(double lr = 0.01, 
        double momentum = 0.0,
        std::function scheduler = nullptr);
    
    void step(std::vector layers, size_t batch_size) override;
    void afterStep() override;
    void setLearningRate(double lr) override;
    void decayLearningRate(double decay_factor) override;
    double getLearningRate() const override;
    
    void setMomentum(double m);
    void setLRScheduler(std::function scheduler);
    void resetStepCount();
};
```

---

## ⚙️ Key Features

### 1. **Momentum Implementation**
```cpp
// Weight update with momentum
velocity_weights[layer][i][j] = 
    momentum * velocity_weights[layer][i][j] + 
    lr * grad_weights[i][j];
new_weights[i][j] -= velocity_weights[layer][i][j];
```
- Maintains velocity buffers per parameter
- Momentum factor controls persistence of previous updates
- Default momentum = 0 (pure SGD)

### 2. **Learning Rate Scheduling**
```cpp
void afterStep() override {
    step_count++;
    if (lr_scheduler) {
        learning_rate = lr_scheduler(initial_lr, step_count);
    }
}
```
- Supports custom scheduler functions
- Scheduler signature: `double func(double initial_lr, size_t step)`
- Automatic step counting

### 3. **Batch-Size Aware Scaling**
```cpp
const double lr = this->learning_rate / batch_size;
```
- Normalizes learning rate by batch size
- Ensures consistent update magnitude across batch sizes

### 4. **Layer-Specific Buffers**
```cpp
if (velocity_weights.find(layer) == velocity_weights.end()) {
    // Initialize velocity buffers on first access
    velocity_weights[layer] = std::vector>(...);
    velocity_biases[layer] = std::vector(...);
}
```
- Lazy initialization of velocity buffers
- Automatic memory management
- Supports models with dynamic layers

---

## 🔧 Implementation Details

### Parameter Update Workflow
1. **Gradient Scaling**:
   ```cpp
   const double lr = learning_rate / batch_size;
   ```
2. **Velocity Update** (if momentum > 0):
   ```cpp
   velocity = momentum * previous_velocity + lr * gradient
   ```
3. **Parameter Update**:
   ```cpp
   parameter -= velocity  // With momentum
   parameter -= lr * gradient  // Without momentum
   ```
4. **Gradient Reset**:
   ```cpp
   dense_layer->clearGradients();
   ```

### Scheduler Integration
| Method | Description |
|--------|-------------|
| `setLRScheduler()` | Assigns custom scheduling function |
| `resetStepCount()` | Resets internal step counter to 0 |
| `afterStep()` | Automatically calls scheduler after each step |

### Dimension Handling
```cpp
const size_t output_size = weights.size();
const size_t input_size = weights.empty() ? 0 : weights[0].size();
```
- Robust dimension detection
- Handles empty layers gracefully
- Supports variable-sized layers

---

## 🚀 Usage Examples

### Basic SGD
```cpp
SGD optim(0.01);  // LR=0.01, no momentum
optim.step(model.getLayers(), batch_size);
```

### SGD with Momentum
```cpp
SGD optim(0.01, 0.9);  // Momentum=0.9
optim.step(model.getLayers(), batch_size);
optim.afterStep();  // For step counting
```

### Custom Learning Rate Schedule
```cpp
auto scheduler = [](double init_lr, size_t step) {
    return init_lr * exp(-0.1 * step);
};

SGD optim(0.1, 0.0, scheduler);  // Exponential decay

for (int epoch = 0; epoch (layer);
   if (!dense_layer) return;
   ```

2. **Gradient Scaling**:
   - Assumes gradients are accumulated (not averaged)
   - Divides learning rate by batch size

3. **Numerical Stability**:
   - No gradient clipping
   - No protection against large momentum values

---

## 🚧 Future Improvements

1. **Support for Other Layers**:
   ```cpp
   void updateConvLayer(ConvLayer* layer);
   void updateRNNLayer(RNNLayer* layer);
   ```

2. **Weight Decay**:
   ```cpp
   void setWeightDecay(double decay);
   ```

3. **Gradient Clipping**:
   ```cpp
   void setGradientClip(double threshold);
   ```

4. **Nesterov Momentum**:
   ```cpp
   void enableNesterov(bool enable);
   ```

5. **State Serialization**:
   ```cpp
   void saveState(const std::string& path);
   void loadState(const std::string& path);
   ```

---

## 📊 Performance Characteristics

| Operation | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| `step()` | O(P) where P=parameters | O(P) for velocity buffers |
| `afterStep()` | O(1) | O(1) |
| Layer update | O(output_size × input_size) | O(1) per layer |
