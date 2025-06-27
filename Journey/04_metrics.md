# 📊 **Metrics**

---

## 📦 Metrics Module Overview

The **Metrics** module provides functionality for **evaluating** and **analyzing** datasets and model outputs. It currently includes:

* **Loss functions** (detailed separately in [`Losses.md`](Losses.md))
* **Correlation analysis tools**

This document focuses on **Correlation utilities** and the Metrics folder organization.

---

## 📂 Folder Structure

| File                                   | Description                                                                 |
| -------------------------------------- | --------------------------------------------------------------------------- |
| `include/Metrics/Losses.h`             | Loss functions and derivatives (see [`Losses.md`](Losses.md))               |
| `include/Metrics/Correlations.h`       | Correlation analysis functions                                               |
| `src/Metrics/Losses/mse.cpp`           | Mean Squared Error loss implementation                                      |
| `src/Metrics/Losses/mae.cpp`           | Mean Absolute Error loss implementation                                     |
| `src/Metrics/Losses/bce.cpp`           | Binary Cross Entropy loss implementation                                    |
| `src/Metrics/Losses/cross_entropy.cpp` | Categorical Cross Entropy loss implementation                               |
| `src/Metrics/Losses/hinge.cpp`         | Hinge loss implementation                                                   |
| `src/Metrics/correlations.cpp`         | Correlation matrix and utilities implementation                             |

---

## 🔍 Correlation Utilities

These utilities compute **Pearson correlation coefficients** for statistical analysis:

### Core Features
* **Pearson Correlation Matrix**: Pairwise correlations between all dataset features
* **Target Correlation**: Feature correlations with specified target column/vector
* **Reporting Tools**:
  - Sorted correlations (by absolute value)
  - Highly correlated feature pairs (multicollinearity detection)

### Function Specifications

| Function                                                                             | Description                                                                                                        |
| ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| `computeCorrelationMatrix<T>(const vector<vector<T>>&)`                              | Returns 2D correlation matrix between all columns                                                                  |
| `computeCorrelationWithAttribute<T>(const vector<vector<T>>&, int)`                  | Correlations with target column (default: last column)                                                             |
| `computeCorrelationWithTarget<T>(const vector<vector<T>>&, const vector<T>&)`        | Correlations with external target vector                                                                           |
| `printSortedCorrelations(const vector<double>&, bool ascending=false)`               | Prints correlations sorted by absolute value (descending default)                                                  |
| `printHighlyCorrelatedFeatures(const vector<vector<double>>&, double threshold=0.8)` | Prints feature pairs with \|corr\| > threshold (sorted by index pairs)                                            |

---

## ⚠️ Usage Notes & Assumptions

1. **Data Requirements**:
   - Rectangular matrices (consistent row/column counts)
   - Numeric types (`int`, `double`, `float`)
   - No missing values (handling not implemented)

2. **Statistical Methods**:
   - Unbiased estimation (N-1 denominator)
   - Pearson correlation formula:
     $$
     r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}
     $$

3. **Error Handling**:
   - Throws `std::invalid_argument` for:
     - Empty datasets
     - Size mismatches
     - Invalid column indices
   - Returns 0 correlation for zero-variance features

---

## 🛠️ Development Journey

### Key Challenges & Solutions
| Challenge                                                      | Solution                                                                                              |
| -------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| **Division by zero** in standard deviation                     | Added variance threshold (1e-10) → set correlation=0                                                  |
| **Template type support** (int/double)                         | Used `static_cast<double>` for internal calculations                                                   |
| **Inconsistent row sizes**                                     | Implemented `getShape()` with rectangularity validation                                               |
| **Inefficient correlation matrix** computation (O(n³))         | Optimized to O(n²) with covariance reuse                                                              |
| **Duplicate feature pairs** in high-correlation reporting      | Used upper triangular iteration (j > i)                                                               |

### Performance Optimizations
1. **Covariance Matrix First**:
```
auto cov = computeCovarianceMatrix(data);
auto corr = convertToCorrelation(cov);
```
2. **Single-Pass Statistics**:
- Precompute means and variances together
3. **Symmetric Matrix Handling**:
- Compute only upper triangle
- Mirror results to lower triangle

### Key Insights
- **Validation First**: 100% input validation prevents runtime errors
- **Numerical Safety**: Zero-variance handling avoids NaNs
- **API Design**: Unified interface for different target types
- **Debugging**: Added detailed error messages with indexes
- **Testing**: Verified against NumPy's `np.corrcoef`

---

## 📚 Key Learnings

1. **Robustness > Performance**:
- Zero-variance checks are crucial
- Input validation prevents cryptic failures

2. **Template Challenges**:
- Explicit casting required for mixed-type safety
- Explicit instantiations needed for linking

3. **Optimization Path**:
```mermaid
graph LR;
    A[Naive O-n³] --> B[Covariance Reuse];
    B --> C[Symmetric Optimization];
    C --> D[Batch Processing];
```


4. **API Design Principles**:
- Consistent naming (`compute*` vs `print*`)
- Sensible defaults (last column as target)
- Const-correct parameters


---

## ⏳ Future Enhancements

### High Priority
| Feature                      | Description                                                                 |
| ---------------------------- | --------------------------------------------------------------------------- |
| **Missing Value Handling**   | NaN skipping/Imputation options                                             |
| **Alternative Correlations** | Spearman, Kendall Tau, and Point-Biserial                                  |
| **GPU Acceleration**         | CUDA implementation for large datasets                                      |
| **Streaming API**            | Online correlation computation for incremental data                          |

### Advanced Features
| Feature                      | Description                                                                 |
| ---------------------------- | --------------------------------------------------------------------------- |
| **Sparse Matrix Support**    | Compressed formats for high-dimensional data                                |
| **Statistical Significance** | p-value calculation for correlations                                        |
| **Confidence Intervals**     | 95% CI reporting for correlation values                                    |
| **Visualization Integration**| Plotting support (heatmaps, scatter matrices)                              |

### Performance & Usability
| Feature                      | Description                                                                 |
| ---------------------------- | --------------------------------------------------------------------------- |
| **Parallel Processing**      | OpenMP/Threading for multi-core systems                                    |
| **Memory-Mapped Files**      | Out-of-core computation for huge datasets                                  |
| **Configurable Output**      | CSV/JSON formatting options                                                |
| **Python Bindings**          | Pybind11 interface for Python integration                                  |

### Experimental
| Feature                      | Description                                                                 |
| ---------------------------- | --------------------------------------------------------------------------- |
| **Auto-Correlation**         | Time-series lag analysis                                                    |
| **Partial Correlation**      | Controlling for other variables                                            |
| **Categorical Correlation**  | Cramer's V, Theil's U for mixed data types                                 |


