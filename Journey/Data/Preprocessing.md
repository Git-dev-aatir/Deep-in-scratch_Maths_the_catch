# 🧹 Preprocessing.md

## 📝 Overview

The `Preprocessing` namespace provides essential data cleaning and transformation tools for preparing datasets before training neural networks. These functions operate directly on the `Dataset` class, enabling standardization, normalization, missing value handling, outlier removal, column manipulation, categorical encoding, and shuffling. Designed for practical machine learning workflows.

---

## ✨ Key Features

- **Scaling**: Standardize to zero mean/unit variance or normalize to [0,1] range
- **Missing Value Handling**: Detect, remove, or impute NaN values
- **Outlier Removal**: Filter rows using Z-Score or IQR methods
- **Column Operations**: Remove columns or expand categorical features to one-hot encoding
- **Data Shuffling**: Randomize row order for unbiased training

---

## 🏗️ Design Decisions

- **In-Place Operations**: All functions modify datasets directly for memory efficiency
- **Selective Processing**: Column parameters allow targeting specific features
- **NaN Safety**: Uses `quiet_NaN()` for missing values with consistent handling
- **Statistical Robustness**: Skips invalid operations (e.g., scaling constant columns)
- **Order Preservation**: Maintains row/column relationships where applicable

---

## 🛠️ Implementation Challenges & Solutions

### 1️⃣ Handling Problematic Columns
- **Challenge**: Scaling columns with zero variance causes division errors
- **Solution**: Skip processing for constant columns with warning

### 2️⃣ Accurate Outlier Detection
- **Challenge**: IQR method requires precise percentile calculation
- **Solution**: Use linear interpolation for Q1/Q3 with sorted data validation

### 3️⃣ Efficient One-Hot Encoding
- **Challenge**: Expanding columns without corrupting data order
- **Solution**: Pre-calculate category counts and build new matrix systematically

### 4️⃣ Safe Data Removal
- **Challenge**: Index shifting during row/column deletion
- **Solution**: Process columns in reverse order and use predicate-based row removal

---

## 🔍 Key Implementation Details

- **Standardization Formula**:  
  \( x' = \frac{x - \mu}{\sigma} \)
- **Normalization Formula**:  
  \( x' = \frac{x - \min}{\max - \min} \)
- **IQR Outlier Detection**:  
  Bounds: \([Q1 - 1.5 \times IQR, Q3 + 1.5 \times IQR]\)
- **One-Hot Encoding**: Converts integer categories to orthogonal binary vectors
- **Fisher-Yates Shuffle**: Efficient O(n) in-place randomization

---

## 🚀 Usage Example

```cpp
Dataset data;
data.loadCSV("data.csv");

// Handle missing values
Preprocessing::printMissingValues(data);
Preprocessing::imputeMissing(data, ImputeStrategy::Mean);

// Scale features
Preprocessing::standardize(data, {0, 2, 4}); // Specific columns
Preprocessing::minMaxNormalize(data); // All columns

// Process outliers
Preprocessing::dropOutliers(data, OutlierMethod::IQR, 1.5, {1, 3});

// Prepare categorical data
Preprocessing::oneHotEncode(data, {5}); // Column 5 = categories

// Final preparation
Preprocessing::shuffleRows(data);
data.saveCSV("cleaned_data.csv");
```


---

## ⚡ Performance & Limitations

| **Strength** | **Limitation** |
|--------------|----------------|
| Efficient in-place operations | No parallel processing |
| Handles large datasets | Only double precision support |
| Comprehensive NaN handling | No string category support |
| Preserves data relationships | Aggressive outlier removal |

---

## 🚧 Future Improvements

- [ ] **String Categorical Support**: Auto-convert text categories to integers
- [ ] **Advanced Imputation**: KNN-based missing value filling
- [ ] **Feature Engineering**: Polynomial feature creation
- [ ] **Binning**: Continuous value discretization
- [ ] **Memory Optimization**: Reference-based operations for large data
- [ ] **Progress Logging**: Verbose mode with progress indicators
