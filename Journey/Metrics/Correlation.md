# 📊 Correlation Metrics Documentation

---

## 📝 Overview

The correlation module provides statistical methods to analyze relationships between variables in datasets. It includes:
- Covariance matrix computation
- Pearson correlation matrix
- Attribute-target correlation analysis
- Visualization of highly correlated features

---

## 🧩 Core Functions

### 1. **`getShape()`**
```
template
std::tuple getShape(const std::vector>& dataset)
```
- **Purpose**: Validates dataset consistency and returns dimensions
- **Validation**: Checks all rows have same column count
- **Returns**: `(rows, columns)` tuple
- **Throws**: `invalid_argument` for inconsistent row sizes

---

### 2. **`computeCovarianceMatrix()`**
```
template
std::vector> computeCovarianceMatrix(...)
```
- **Implementation**:
  - Computes column means
  - Calculates centered values
  - Accumulates products of centered values
  - Normalizes by `n-1` (unbiased estimator)
- **Formula**:  
  $$\text{Cov}(X,Y) = \frac{1}{n-1} \sum (x_i - \bar{x})(y_i - \bar{y})$$

---

### 3. **`computeCorrelationMatrix()`**
```
template
std::vector> computeCorrelationMatrix(...)
```
- **Implementation**:
  - Uses covariance matrix
  - Computes standard deviations
  - Handles near-zero standard deviations
- **Formula**:  
  $$\rho_{X,Y} = \frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y}$$

---

### 4. **`computeCorrelationWithAttribute()`**
```
template
std::vector computeCorrelationWithAttribute(...)
```
- **Parameters**:
  - `target_col`: Index of target column (-1 for last column)
- **Implementation**:
  - Computes means for all columns
  - Accumulates covariances and variances
  - Handles division by zero

---

### 5. **`computeCorrelationWithTarget()`**
```
template
std::vector computeCorrelationWithTarget(...)
```
- **Validation**: Checks target vector length matches row count
- **Implementation**:
  - Computes dataset means and target mean
  - Accumulates covariances and variances
  - Uses same correlation formula as above

---

### 6. **Visualization Utilities**
```
void printSortedCorrelations(...)
void printHighlyCorrelatedFeatures(...)
```
- **Features**:
  - Sorts correlations by absolute value
  - Filters feature pairs by correlation threshold
  - Provides human-readable output

---

## ⚙️ Implementation Details

### Numerical Stability
- **Division Handling**: Checks for near-zero denominators
- **Two-Pass Algorithm**: Reduces floating-point error
- **Centered Values**: Computes before covariance

---

### Efficiency Optimizations
- **Precomputation**: Column means calculated once
- **Symmetric Optimization**: Only computes upper triangle
- **Memory Reuse**: Avoids unnecessary copies

---

### Validation
- **Empty Dataset Handling**: Returns zero matrix
- **Single Row Handling**: Returns zero covariance
- **Index Validation**: Checks target column bounds

---

## 🚀 Usage Example

```
vector> data = {
    {1.0, 2.0, 3.0},
    {1.5, 2.5, 3.5},
    {2.0, 3.0, 4.0}
};

// Compute full correlation matrix
auto corr_matrix = computeCorrelationMatrix(data);

// Find correlations with last column
auto correlations = computeCorrelationWithAttribute(data, -1);

// Print sorted correlations
printSortedCorrelations(correlations);

// Identify highly correlated features
printHighlyCorrelatedFeatures(corr_matrix, 0.95);
```

---

## ⚠️ Edge Case Handling

| Case                  | Handling                      |
|-----------------------|------------------------------|
| Empty dataset         | Returns (0,0) shape           |
| Single-row dataset    | Returns zero covariance       |
| Inconsistent rows     | Throws `invalid_argument`     |
| Invalid column index  | Throws `out_of_range`         |
| Near-zero variance    | Returns zero correlation      |
| Target size mismatch  | Throws `invalid_argument`     |

---

## 🚧 Future Improvements

1. **Support for Different Correlation Types**:
   ```
   enum class CorrelationType { PEARSON, SPEARMAN, KENDALL };
   ```

2. **Parallelization**:
   ```
   #pragma omp parallel for
   for (size_t i = 0; i > computeCorrelationPValues();
   ```

5. **Large Dataset Support**:
   ```
   class StreamingCorrelationCalculator {
       void addDataPoint(const vector& row);
       // ...
   };
   ```
