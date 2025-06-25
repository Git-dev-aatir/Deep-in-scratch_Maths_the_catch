# 🧹 **Data Preprocessing Module**

---

## **1. Overview**

This module provides essential **data preprocessing utilities** designed for Machine Learning and Data Analysis workflows in C++.
It offers functionality for:

1. **Handling missing values** via multiple strategies.
2. **Data normalization and standardization** to improve model performance and convergence.
3. **Outlier detection and removal** (Z-Score and IQR).
4. **Templated support** for numeric and string types.
5. **Column removal**, **missing value diagnostics**, and data cleanup utilities.

---

## **2. Files Involved**

```
/include/Preprocessing/Preprocessing.h        # Header with declarations
/src/Preprocessing/Preprocessing.cpp          # Implementation
/include/Preprocessing/Helper_functions.h     # Helper methods (trimming, parsing, missing check)
```

---

## **3. Features**

### 🔹 **Missing Value Handling**

✔️ Detect missing entries for:

* `float` / `double`: via `isnan()`
* `int`: via sentinel `std::numeric_limits<int>::min()`
* `string`: empty string (`""`)

✔️ Replace missing values using:

* **Mean**: averages available values
* **Median**: uses central tendency
* **Mode**: most frequent value
* **Custom value**: provided by the user

✔️ Remove rows with any missing values.

✔️ Locate and print missing value locations.

---

### 🔹 **Normalization & Standardization**

✔️ **Standardize**: Subtract mean, divide by standard deviation
✔️ **Normalize**: Scale to `[0, 1]` range

✔️ Operates only on **numeric types** (`int`, `double`)

✔️ Supports column selection: normalize specific columns, or all

✔️ Skips missing values in calculations

---

### 🔹 **Outlier Detection and Removal**

✔️ Two supported methods:

* **Z-Score**:

  * Compute mean & stddev, remove data beyond threshold
* **IQR (Interquartile Range)**:

  * Uses 1st and 3rd quartiles and removes values outside `[Q1 - k×IQR, Q3 + k×IQR]`

✔️ Fully supports column-specific outlier cleaning

✔️ Gracefully skips columns with insufficient data or zero stddev

---

### 🔹 **Column Removal**

✔️ Remove any set of columns using a list of indices

✔️ Retains relative column order

✔️ Safe against invalid inputs

✔️ Works for **any data type** (`int`, `double`, `string`)

---

### 🔹 **Helper Tools**

✔️ `get_median()` utility for numeric vectors

✔️ `isMissing()` overloaded for different datatypes using `enable_if`

✔️ `trim()` utilities to sanitize string tokens during parsing

✔️ `split()` handles both:

* Single-character delimiters
* Multi-space regex-based splitting

✔️ `parseToken<T>()` template + specializations for robust parsing with error fallback

---

## **4. Challenges & Solutions**

| Issue                                              | Solution                                                  |
| -------------------------------------------------- | --------------------------------------------------------- |
| Detecting missing values for various types         | Used SFINAE (`enable_if`) with special cases for `string` |
| Preventing flattening of columns (esp. labels)     | Manual reshaping retained dimensionality                  |
| Handling zero variance during standardization      | Skipped division to avoid numerical issues                |
| Avoiding accidental data loss on column operations | Defensive programming with sets and index tracking        |
| Consistent template usage without code bloat       | Explicit instantiation in `.cpp` for required types only  |

---

## **5. Learnings & Design Patterns**

✔️ Strong type constraints using `static_assert` or SFINAE lead to safer templates
✔️ Missing value detection is a core concern in C++ preprocessing
✔️ Explicit instantiations prevent template bloat in large systems
✔️ Preprocessing logic benefits from a separation between utility and core logic

---

## **6. To-Do / Future Roadmap**

* [ ] Add support for **mixed-type rows** in the same dataset
* [ ] Implement **label encoding** or **one-hot encoding** for categorical string data
* [ ] Add **random seed control** for reproducible preprocessing (e.g., for shuffling, imputation)
* [ ] Integrate **correlation analysis** (moved to future module for now)
* [ ] Add **transforms history tracking** (like scikit-learn's fit/transform)
* [ ] Add **outlier tagging** (instead of just removal)

---

## **7. Limitations**

* Z-Score and IQR methods are only for numeric datasets
* Missing values detected only if they match expected patterns (no token inference)
* No correlation matrix or feature selection (to be added separately)
* `unsqueeze()`-like shape transformations handled in Dataset Utility module, not here
* No automatic type inference — user must specify template usage explicitly

---

## **8. Version History**

| Version | Date       | Changes                                                               |
| ------- | ---------- | --------------------------------------------------------------------- |
| 1.2     | 2024–2025  | Initial support for missing value handling, normalize, standardize    |
| 1.3     | 23-06-2025 | Added outlier detection, column removal, better missing value support |
