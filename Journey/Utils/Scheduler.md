# 🔄 Learning Rate Schedulers Documentation

---

## 📝 Overview
The `Schedulers` namespace provides factory functions for creating learning rate scheduler functions. These schedulers dynamically adjust the learning rate during training to improve convergence and model performance.

---

## 🧩 Scheduler Functions

### 1. **Cosine Annealing Scheduler**
```cpp
inline std::function cosine(double total_steps) {
    return [total_steps](double init, size_t step) {
        return init * 0.5 * (1 + cos(M_PI * step / total_steps));
    };
}
```

**Parameters**:
- `total_steps`: Total training steps for full annealing cycle

**Behavior**:
- Learning rate follows a cosine curve from `init` to near-zero
- Formula: $$ \text{lr} = \text{init} \times 0.5 \times \left(1 + \cos\left(\pi \times \frac{\text{step}}{\text{total\_steps}}\right)\right) $$

**Use Case**:
- Fine-tuning in later training stages
- Cyclical learning rate schedules

**Visualization**:
```
Learning Rate
  ^
  |     /\
  |    /  \
  |   /    \
  |  /      \
  |_/________\___> Steps
```

### 2. **Step Decay Scheduler**
```cpp
inline std::function step(size_t step_size, double gamma) {
    return [step_size, gamma](double init, size_t step) {
        return init * pow(gamma, floor(step/step_size));
    };
}
```

**Parameters**:
- `step_size`: Number of steps between decay events
- `gamma`: Multiplicative decay factor

**Behavior**:
- Learning rate decays by `gamma` every `step_size` steps
- Formula: $$ \text{lr} = \text{init} \times \gamma^{\lfloor \frac{\text{step}}{\text{step\_size}} \rfloor} $$

**Use Case**:
- Stable convergence in mid-training
- When validation loss plateaus

**Visualization**:
```
Learning Rate
  ^
  |___________
  |           \_________
  |                     \_____
  |____________________________> Steps
  ||
```

---

## ⚙️ Implementation Details

### 1. **Mathematical Foundation**
- **Cosine Annealing**: Based on half-period cosine function
- **Step Decay**: Exponential decay with floor function

### 2. **Numerical Stability**
- Protected against division by zero
- Uses standard math constants (M_PI defined if missing)

### 3. **Closure Design**
- Returns callable objects with captured configuration
- Stateless implementation
- Pure functional design

---

## 🚀 Usage Examples

### Integration with Optimizer
```cpp
SGD optim(0.1);  // Initial LR=0.1

// Create cosine scheduler for 1000 steps
auto cosine_scheduler = Schedulers::cosine(1000);
optim.setLRScheduler(cosine_scheduler);

// Create step decay scheduler (decay 0.8 every 100 steps)
auto step_scheduler = Schedulers::step(100, 0.8);
optim.setLRScheduler(step_scheduler);
```

### During Training Loop
```cpp
for (size_t step = 0; step  10) {
    optim.setLRScheduler(Schedulers::cosine(500));
}
```

---

## 🚧 Future Improvements

1. **Linear Warmup**:
   ```cpp
   inline std::function linear_warmup(size_t warmup_steps);
   ```

2. **Cyclical Cosine**:
   ```cpp
   inline std::function cosine_cyclic(size_t cycle_steps);
   ```

3. **Exponential Decay**:
   ```cpp
   inline std::function exponential(double gamma);
   ```

4. **Plateau Detection**:
   ```cpp
   inline std::function reduce_on_plateau(double factor, size_t patience);
   ```

5. **Custom Scheduler**:
   ```cpp
   template
   inline std::function custom(Func f);
   ```

---

## 📊 Scheduler Comparison

| Scheduler Type       | Parameters          | Best For                  | Behavior                  |
|----------------------|---------------------|---------------------------|---------------------------|
| Cosine Annealing     | `total_steps`       | Final convergence         | Smooth cyclic decay       |
| Step Decay           | `step_size, gamma`  | Stable training           | Sudden periodic drops    |
| Linear Warmup        | `warmup_steps`      | Early training            | Linear increase          |
| Exponential          | `gamma`             | Continuous decay          | Smooth exponential curve |
| Reduce on Plateau    | `factor, patience`  | Avoiding local minima     | Validation-based         |
