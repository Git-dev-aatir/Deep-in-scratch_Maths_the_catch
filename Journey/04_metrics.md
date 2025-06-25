# 📊 **Metrics**

---

## 📦 Metrics Module Overview

The **Metrics** module provides functionality for **evaluating** and **analyzing** datasets and model outputs. It currently includes:

* **Loss functions** (detailed separately in `03_losses.md`)
* **Correlation analysis tools**

This document summarizes the **Correlation utilities** alongside a brief overview of the Metrics folder organization.

---

## 📂 Folder Structure

| File                                   | Description                                                         |
| -------------------------------------- | ------------------------------------------------------------------- |
| `include/Metrics/Losses.h`             | Declarations of loss functions and derivatives (see `03_losses.md`) |
| `include/Metrics/Correlations.h`       | Declarations of correlation analysis functions                      |
| `src/Metrics/Losses/mse.cpp`           | Mean Squared Error loss implementation                              |
| `src/Metrics/Losses/mae.cpp`           | Mean Absolute Error loss implementation                             |
| `src/Metrics/Losses/bce.cpp`           | Binary Cross Entropy loss implementation                            |
| `src/Metrics/Losses/cross_entropy.cpp` | Categorical Cross Entropy loss implementation                       |
| `src/Metrics/Losses/hinge.cpp`         | Hinge loss implementation                                           |
| `src/Metrics/correlations.cpp`         | Correlation matrix and correlation utilities implementation         |

---

## 🔍 Correlation Utilities

These utilities assist in statistical analysis by computing **Pearson correlation coefficients** among dataset features or between features and a target.

### Features

* **Pearson Correlation Matrix**
  Computes pairwise correlations between all columns (features) of a numeric dataset.

* **Correlation with Target Attribute**
  Computes correlation of each feature column with a specified target column.

* **Correlation with External Target Vector**
  Computes correlation of each feature column with an externally provided target vector.

* **Reporting Utilities**

  * Print correlations sorted by absolute values (ascending or descending).
  * Print pairs of features with high absolute correlation (above a threshold), useful for identifying multicollinearity.

---

## 📖 Function Summaries

| Function                                                                             | Description                                                                                                        |
| ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| `computeCorrelationMatrix<T>(const vector<vector<T>>&)`                              | Returns a 2D matrix of Pearson correlations between all columns.                                                   |
| `computeCorrelationWithAttribute<T>(const vector<vector<T>>&, int)`                  | Returns correlations between each column and the specified target column. Defaults to last column if not provided. |
| `computeCorrelationWithTarget<T>(const vector<vector<T>>&, const vector<T>&)`        | Returns correlations between each column and an external target vector.                                            |
| `printSortedCorrelations(const vector<double>&, bool ascending=false)`               | Prints sorted correlations by absolute value; descending by default.                                               |
| `printHighlyCorrelatedFeatures(const vector<vector<double>>&, double threshold=0.8)` | Prints all feature pairs with correlation magnitude above threshold, sorted by index pairs.                        |

---

## ⚠️ Notes and Usage

* Template functions support numeric types such as `int` and `double`.
* Assumes datasets are rectangular matrices (consistent row and column counts).
* Uses unbiased standard deviation and covariance calculations (dividing by N-1).
* Designed for use in exploratory data analysis, feature selection, and data quality assessment.
* Printing functions write to `std::cout` directly; redirect or adapt as needed.

---

## 🛠️ **Problems Faced & Solutions**

| Problem                                                           | Solution                                                                                             |
| ----------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Handling division by zero when computing standard deviation       | Added check to skip computation or set correlation to zero if variance is zero to avoid NaN results. |
| Ensuring template functions support both `int` and `double` types | Used static\_cast<double> to ensure internal computation in floating point for precision.            |
| Detecting inconsistent row sizes in datasets                      | Introduced strict assertions and runtime checks to verify that all rows have equal lengths.          |
| Printing cleanly sorted feature correlations                      | Used `std::pair` with index tracking and custom sorting to display features by absolute correlation. |
| Overlapping feature pairs reported in high correlation function   | Applied careful double-loop bounds and index ordering to avoid duplicate or redundant feature pairs. |

---

## 📚 **Key Learnings**

* **Robustness checks** (like variance zero-handling) are crucial in statistical computations to prevent runtime crashes or invalid values.
* **Templates require explicit casting** to maintain calculation precision across `int`, `float`, `double` without ambiguity.
* Proper **sorting and indexing of feature pairs** makes high-correlation reporting meaningful and user-friendly.
* Dataset **rectangularity (consistent row lengths)** must be validated before computation to ensure algorithm correctness.
* Diagnostic **print utilities** simplify exploratory data analysis and are useful beyond development/testing phases.

---

## 🏷️ Namespace

```
Metrics
```

---

## ⏳ **Future Enhancements**

* Support for other correlation measures (Spearman, Kendall Tau)
* Handling missing or NaN values gracefully
* Configurable output streams for print functions
* Parallelized or optimized computation for large datasets
* Integration with dataset loading utilities and feature metadata
