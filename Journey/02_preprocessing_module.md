# 🧹 **Data Preprocessing Module — Documentation**

---

## **1. Overview**

This module provides essential **data preprocessing utilities** designed for Machine Learning and Data Analysis workflows in C++.
It offers functionality for:

1. **Handling missing values** via multiple strategies.
2. **Data normalization and scaling** to improve model convergence.
3. **Outlier detection and removal** (Z-Score and IQR supported).
4. **Correlation analysis** for feature selection.
5. Supporting **templated datatypes** (`int`, `double`, `std::string`) where applicable.
6. Utilities to **inspect data statistics** for informed preprocessing.
7. **Column removal** and **highly correlated feature identification**.

---

## **2. Files Involved**

```
/include/preprocessing.h
/src/preprocessing.cpp
```

---

## **3. Features**

### 🔹 **Missing Value Handling**

✔️ Detect missing entries (empty strings or sentinel values).

✔️ Replace missing values by:

* Mean (numeric only)
* Median (numeric only)
* Mode (categorical and numeric)
* Custom value (user-defined)

---

### 🔹 **Scaling & Normalization**

✔️ **Min-Max Scaling** — scales features to `[0, 1]`.

✔️ **Z-score Normalization** — standardizes to zero mean and unit variance.

✔️ Works with **int** and **double** datasets.

✔️ Automatically skips non-numeric columns during numeric transforms.

---

### 🔹 **Outlier Detection and Removal**

✔️ Supports two methods:

* **Z-Score**: Flags data points that are beyond the specified Z-score threshold.
* **IQR (Interquartile Range)**: Flags points outside the IQR \* threshold range.

✔️ Users can specify **threshold** and **columns** for targeted outlier removal.

✔️ Applicable only for **numeric datasets** (`int`, `double`).

---

### 🔹 **Correlation Analysis**

✔️ Computes **Pearson Correlation Matrix**.

✔️ Computes **correlation of all features with a target feature**.

✔️ Identifies and **prints highly correlated feature pairs** above a specified threshold.

✔️ Supports **sorting and printing correlations** based on absolute values.

---

### 🔹 **Column Removal**

✔️ Remove specified columns from the dataset based on **column indices**.

✔️ Safeguards against out-of-range errors.

✔️ Works on all **templated datatypes**.

---

### 🔹 **Data Inspection**

✔️ Compute basic statistics: mean, median, mode, variance, count missing.

✔️ Print per-column summary for quick checks.

---

## **4. Challenges & Solutions**

| Issue                                            | Solution                                                  |
| ------------------------------------------------ | --------------------------------------------------------- |
| Handling missing values differently by datatype. | Template specializations for numeric vs string types.     |
| Outlier detection requires statistical methods.  | Implemented Z-Score and IQR based outlier removal.        |
| Correlation requires numerical-only handling.    | Correlation functions restricted to numeric datatypes.    |
| Avoid data corruption during inplace transforms. | Designed functions to return new datasets (non-mutating). |

---

## **5. Learning Points**

✔️ Preprocessing improves downstream model performance.

✔️ Templates require thoughtful specialization for varied datatypes.

✔️ Separating numeric and categorical preprocessing clarifies logic.

✔️ Outlier removal and correlation analysis aid in **feature engineering**.

✔️ Data inspection and visualization of correlations help tailor preprocessing strategies effectively.

---

## **6. To-Do / Future Plans**

* Include string dataset support in various functions (for encoding).
* Support for mixed-type datasets (numerical + categorical together).
* Implement encoding options: One-hot, binary encoding.
* Handling missing data via predictive imputation (ML-based).
* Integration with **dataset utility module** for seamless workflow.
* Visual representation of correlations and outlier detections.

---

## **7. Known Limitations**

* Most advanced functions (outlier detection, correlation) currently support **numeric datatypes only**.
* No automatic detection of missing value tokens beyond empty strings.
* Does not yet handle mixed datatypes in a single dataset.
* No inplace preprocessing — functions return copies to avoid side effects.

---

## **8. Version**

* **v2.0** — Outlier detection, correlation analysis, column removal, and sorting of correlations added.
