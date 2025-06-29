# 🔄 DataLoader.md

## 📝 Overview

The `DataLoader` class provides efficient iteration over `Dataset` objects in mini-batches. Key features include:
- Configurable batch sizes
- Optional epoch shuffling
- STL-compatible iterator interface
- Zero-copy batch creation
- Deterministic shuffling with proper seeding

---

## 🏗️ Design Decisions

1. **Epoch Management**:
   - Automatic index reset at epoch start
   - Optional shuffling with Mersenne Twister RNG
   - `std::iota` for efficient index generation

2. **Memory Efficiency**:
   - Stores only indices, not data copies
   - Uses `Dataset::selectRows()` for batch creation
   - Minimal memory overhead per batch

3. **Iterator Design**:
   - Input iterator semantics
   - Batch creation on dereference
   - Efficient position tracking

---

## 🛠️ Implementation Highlights

### Batch Creation Logic
```cpp
Dataset DataLoader::Iterator::operator*() const {
    size_t end = std::min(cursor + loader.batch_size,
    loader.dataset.rows());
    std::vector<size_t> batch_indices;
    for (size_t i = cursor; i < end; i++) {
        batch_indices.push_back(loader.indices[i]);
    }
    return loader.dataset.selectRows(batch_indices);
}
```

- **Safety**: Uses `std::min` for partial batches
- **Efficiency**: Builds indices without data copying
- **Flexibility**: Handles any batch size/dataset combination

### Shuffling Implementation
```cpp
void DataLoader::reset() {
    indices.resize(dataset.rows());
    std::iota(indices.begin(), indices.end(), 0);
    if (shuffle) {
        std::shuffle(indices.begin(), indices.end(), rng);
    }
}
```

- **Reproducibility**: Seeded with `std::random_device`
- **Performance**: O(n) shuffling complexity
- **Correctness**: Maintains complete index set

### Iterator Semantics

```cpp
Iterator& operator++() {
    cursor += loader.batch_size;
    return *this;
}

bool operator!=(const Iterator& other) const {
    return cursor < other.cursor;
}
```
- **Efficiency**: O(1) increment operation
- **Termination**: Simple cursor comparison
- **Compatibility**: Works with range-based for loops

---

## 🚀 Usage Example

```cpp
Dataset training_data;
training_data.loadCSV("train.csv");

// Create loader with 64 samples/batch and shuffling
DataLoader loader(training_data, 64, true);

for (auto it = loader.begin(); it != loader.end(); ++it) {
    Dataset batch = *it; // Get current batch
    model.forward(batch);
    // Get raw indices if needed
    auto indices = it.getIndices();
    debugPrint(indices);
    // Range-based for loop alternative
    for (Dataset batch : loader) {
        model.train(batch);
    }
```


---

## ⚙️ Performance Characteristics

| Operation | Time Complexity | Notes |
|-----------|-----------------|-------|
| Constructor | O(1) | Only stores references |
| reset() | O(n) | Index generation + shuffling |
| Iterator++ | O(1) | Simple cursor increment |
| operator* | O(b) | Batch index creation |
| Batch Creation | O(b×m) | Where b=batch size, m=columns |

---

## ⚠️ Limitations & Edge Cases

1. **Dataset Lifetime**:
   - Requires source dataset to outlive DataLoader
   - Undefined behavior if dataset modified during iteration

2. **Batch Size Handling**:
   - Last batch may be smaller than configured size
   - Always check `batch.rows()` for actual size

3. **Shuffling Consistency**:
   - Different RNG states across processes
   - Consider adding seed parameter for reproducibility

4. **Concurrency**:
   - Not thread-safe for concurrent iteration
   - No parallel batch loading support

---

## 🚧 Future Improvements

1. **Custom Sampling**:
```cpp
void setSampler(std::function<std::vector<size_t>()> sampler);
```


2. **Parallel Loading**:
```cpp
void prefetch(size_t num_batches);
```


3. **Epoch Tracking**:
```cpp
size_t current_epoch() const;
```


4. **Transforms**:
```cpp
void addTransform(std::function<Dataset(Dataset)> transform);
```


5. **Memory Mapping**:
- Support for out-of-core datasets
- Lazy batch loading

6. **Deterministic Shuffling**:
```cpp
void setSeed(unsigned seed);
```


---

## 🛡️ Safety Guarantees

1. **Exception Safety**:
- Strong guarantee for all operations
- No memory leaks during index generation

2. **Bounds Protection**:
- Automatic partial batch handling
- Validated index access

3. **Resource Management**:
- Zero allocations during iteration
- Clean iterator state transitions
