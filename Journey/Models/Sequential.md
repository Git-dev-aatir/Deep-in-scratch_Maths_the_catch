# 🧠 Sequential Model Documentation

---

## 📝 Overview
The `Sequential` class implements a neural network container that stacks layers in sequence. It provides:
- Layer management with ownership semantics
- Automatic parameter initialization
- Forward/backward propagation
- Training loop implementation
- Model inspection utilities

---

## 🏗️ Class Design

### Core Structure
```cpp
class Sequential {
private:
    std::vector> layers; // Owned layers
    bool is_initialized = false;

    // Variadic template helpers
    void addLayers() {}
    template
    void addLayers(First&& first, Rest&&... rest);
    
public:
    template
    Sequential(Layers&&... args);
    
    void initializeParameters(unsigned int seed = MANUAL_SEED, ...);
    std::vector forward(const std::vector& input) const;
    std::vector backward(const std::vector& grad_output);
    void summary() const;
    double train(...); // Two overloads
    void clearGradients();
    BaseLayer* operator[](size_t index);
    std::vector getLayers();
};
```

---

## 🔑 Key Features

### 1. **Layer Ownership**
- Uses `unique_ptr` for automatic memory management
- Prevents dangling pointers through exclusive ownership
- Transfers ownership in constructor:
  ```cpp
  Sequential model(
      new DenseLayer(784, 256),
      new ActivationLayer(ActivationType::RELU),
      new DenseLayer(256, 10)
  );
  ```

### 2. **Smart Parameter Initialization**
```cpp
void initializeParameters(...) {
    for (size_t i = 0; i (layers[i].get())) {
            InitMethod method = InitMethod::XAVIER_UNIFORM; // Default
            
            // Determine method based on next activation
            if (i + 1 (layers[i+1].get())) {
                    switch (act->getActivationType()) {
                        case ActivationType::RELU:
                        case ActivationType::LEAKY_RELU:
                            method = InitMethod::HE_UNIFORM; break;
                        // ... other cases
                    }
                }
            }
            dense->initializeWeights(method, ...);
            dense->initializeBiases(InitMethod::BIAS, ...);
        }
    }
}
```
- **Heuristic**: Chooses initialization based on next activation
  - He for ReLU/LeakyReLU
  - Xavier for Sigmoid/Tanh
  - LeCun for SELU

---

### 3. **Training Loop**
```cpp
double train(const Dataset& X_train, const Dataset& y_train,
             BaseOptim& optimizer, size_t batch_size,
             /* loss and grad functions */) 
{
    DataLoader loader(X_train, batch_size, true);
    double total_loss = 0.0;
    
    for (auto batch : loader) {
        // ... forward pass ...
        // ... loss calculation ...
        // ... backward pass ...
        
        optimizer.step(getLayers(), current_batch_size);
        optimizer.afterStep();
    }
    return total_loss;
}
```
- Supports two variants:
  1. Per-sample processing
  2. Batch-level operations
- Automatic batch management via `DataLoader`

---

### 4. **Propagation**
```cpp
// Forward pass
std::vector output = input;
for (auto& layer : layers) {
    output = layer->forward(output);
}

// Backward pass (reverse order)
std::vector grad = grad_output;
for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
    grad = (*it)->backward(grad);
}
```
- Maintains correct execution order
- Handles intermediate value passing

---

## 🚀 Usage Example

### Model Creation
```cpp
// Create network
Sequential model(
    new DenseLayer(784, 256),
    new ActivationLayer(ActivationType::RELU),
    new DenseLayer(256, 10),
    new ActivationLayer(ActivationType::SOFTMAX)
);

// Initialize parameters
model.initializeParameters(42); // Seed=42

// Train model
SGD optim(0.01);
model.train(X_train, y_train, optim, 64,
            Losses::cross_entropy_loss,
            Losses::cross_entropy_derivative);

// Inspect model
model.summary();
```

---

### Custom Training
```cpp
// Custom loss and gradient functions
auto my_loss = [](const vector& y_true, const vector& y_pred) {
    // ... custom implementation ...
};

model.train(X_train, y_train, optim, 128, my_loss, my_grad_fn);
```

---

## ⚙️ Implementation Details

### 1. **Gradient Management**
```cpp
void clearGradients() {
    for (auto& layer : getLayers()) {
        if (auto* dense = dynamic_cast(layer)) {
            dense->clearGradients();
        }
    }
}
```
- Clears accumulated gradients before each batch
- Only affects `DenseLayer` parameters

### 2. **Layer Access**
```cpp
BaseLayer* operator[](size_t index) {
    if (index >= layers.size()) throw out_of_range(...);
    return layers[index].get();
}

std::vector getLayers() {
    vector ptrs;
    for (auto& layer : layers) ptrs.push_back(layer.get());
    return ptrs;
}
```
- Safe index-based access
- Raw pointer access for optimizer integration

### 3. **Batch Processing**
```cpp
// Batch-level training variant
double train(..., 
    function>&, 
                   const vector>&)> batch_loss_fn,
    function>(...)> batch_grad_fn) 
{
    // ...
    vector> batch_preds;
    for (const auto& x : batch_data) {
        batch_preds.push_back(forward(x));
    }
    auto batch_grads = batch_grad_fn(batch_y, batch_preds);
    // ...
}
```
- More efficient than per-sample processing
- Enables vectorized operations

---

## ⚠️ Limitations

1. **Layer Type Restrictions**:
   - Only `DenseLayer` parameters are updated
   - Other layer types ignored during optimization

2. **Static Architecture**:
   - No support for branching architectures
   - No skip connections

3. **Input Dimension Handling**:
   - Requires manual input size specification
   - No automatic dimension inference

---

## 🚧 Future Improvements

1. **Dynamic Graph Support**:
   ```cpp
   void addLayer(BaseLayer* layer, size_t connect_to = -1);
   ```

2. **Automatic Shape Inference**:
   ```cpp
   void build(const vector& input_shape);
   ```

3. **Heterogeneous Layers**:
   ```cpp
   void addConvLayer(size_t filters, size_t kernel_size);
   ```

4. **Model Saving/Loading**:
   ```cpp
   void save(const string& path);
   void load(const string& path);
   ```

5. **Tensor Support**:
   ```cpp
   Tensor forward(const Tensor& input);
   ```

---

## 📊 Performance Characteristics

| Operation | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| Forward | O(L×P) | O(M) |
| Backward | O(L×P) | O(M) |
| Training Epoch | O(N×L×P/B) | O(B×M) |

Where:
- L: Number of layers
- P: Parameters per layer
- M: Maximum layer output size
- N: Training samples
- B: Batch size
