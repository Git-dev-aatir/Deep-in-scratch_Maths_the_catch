# 📦 **Dataset Utility Module**

---

## **1. Introduction**

This module provides a comprehensive suite of generic and templated **dataset loading, saving, and preprocessing utilities** for CSV, binary, and in-memory datasets. It serves as the foundational step for machine learning workflows in C++ by supporting:

* Flexible **data input/output** with support for numeric and string types.
* **Preprocessing utilities** like feature-label splitting, train-test splitting, and row selection.
* Basic **tensor-like operations** (reshape, squeeze, unsqueeze, transpose) paving the way toward tensor abstractions.
* Debug-friendly **printing and descriptive statistics** for quick dataset insights.

---

## **2. Files and Structure**

```
/include/Preprocessing/dataset_utils.h    // Template declarations, type aliases, and docstrings
/src/Preprocessing/dataset_utils.cpp      // Template implementations, explicit instantiations
/include/Preprocessing/Helper_functions.h // Inline helpers (trimming, splitting, median, isMissing)
/Datasets/                                // Dataset files (CSV, binary)
/Journey/01_dataset_utility.md            // Documentation (this file)
```

---

## **3. Module Features & Journey Notes**

### 🔹 **Dataset Loading (Text/Binary)**

* Load CSV and generic text files, supporting both single-character delimiters and multi-space delimiters using regex.
* Load binary datasets with specialization for `std::string` due to string length handling.
* Template specializations ensure type-safe parsing for:

  * `int`
  * `double` (with NaN-aware missing value checking)
  * `std::string`

#### Journey Note:

> Implementing `parseToken<T>()` highlighted the need for **type specialization** in C++ — especially for string parsing and robust error handling.

---

### 🔹 **Saving Datasets**

* Save datasets as CSV files for readability.
* Save/load binary files efficiently with explicit support for strings by saving lengths alongside character data.

#### Journey Note:

> Managing strings in binary format required explicit length encoding to ensure consistent reading/writing, unlike numeric types.

---

### 🔹 **Train/Test Splitting**

* User-configurable test fraction (default 20%).
* Optional shuffling with `std::shuffle` for randomized splits.
* Index generation factored out via `getIndices()` for easy reproducibility and unit testing.

#### Journey Note:

> Separating index generation improved modularity and will simplify adding seed control for reproducible experiments.

---

### 🔹 **Feature/Label Splitting**

* Splits dataset into feature matrix (`n x (m-1)`) and label matrix (`n x 1`), preserving label dimensionality as 2D for model compatibility.
* Avoids common dimension collapse bugs by wrapping labels in a `DataRow`.

#### Journey Note:

> Wrapping labels in 2D vectors mirrors PyTorch’s `.unsqueeze(1)` behavior, essential for later layer operations.

---

### 🔹 **Basic Tensor-Like Utilities**

* **`squeeze()`**: Flattens 2D datasets to 1D vectors.
* **`unsqueeze()`**: Adds a dimension to 1D vectors along axis 0 (row vector) or axis 1 (column vector).
* **`reshape()`**: Reshapes 2D vectors with dimension consistency checks.
* **`transpose()`**: Swaps rows and columns.

#### Journey Note:

> Inspired by PyTorch, these utilities represent the first steps toward full tensor abstractions in C++.

---

### 🔹 **Debugging Helpers**

* **`head()`**: Nicely formatted printout of first N rows, with column headers and spacing.
* **`printDimensions()`**: Prints `[rows x cols]` for quick dataset shape inspection.
* **`describeDataset()`**: Numeric summaries (mean, stddev, quartiles) with type checking to ensure numeric-only operation.

#### Journey Note:

> Achieving Pandas-like printing in C++ demanded careful formatting and stream manipulations.

---

## **4. Inline Helper Functions**

**Note:** Helper utilities such as string trimming (`ltrim()`, `rtrim()`, `trim()`), splitting (`split()`), missing value detection (`isMissing<T>()`), and statistical helpers (e.g., `get_median()`) are **defined separately** in **`Helper_functions.h`** and included in this module via `#include "Helper_functions.h"`. These functions support core dataset operations but are maintained independently for modularity and reuse.

---

## **5. Technical Challenges & Solutions**

| Challenge                                          | Solution                                                     |
| -------------------------------------------------- | ------------------------------------------------------------ |
| Handling CSV with **multiple spaces / delimiters** | Regex splitting or char delimiter based splitting            |
| Reliable **whitespace trimming**                   | Custom inline trim functions in Helper\_functions.h          |
| Binary I/O for **strings**                         | Length-prefix encoding and decoding                          |
| Preserving label dimensionality (avoid flattening) | Wrapping labels as 2D single-column DataRow                  |
| Reproducible **train/test splitting**              | Extracted index generation with optional shuffle             |
| Tensor-like reshaping utilities                    | Added `squeeze()`, `unsqueeze()`, `reshape()`, `transpose()` |

---

## **6. Key Functions — Summary**

| Function                   | Description                             | Template?          | Specialized for string? |
| -------------------------- | --------------------------------------- | ------------------ | ----------------------- |
| `loadDataset()`            | Load CSV/binary datasets                | Yes                | Yes                     |
| `saveDatasetToCSV()`       | Save dataset as CSV                     | Yes                | Yes                     |
| `saveDatasetToBinary()`    | Save dataset as binary                  | Yes                | Yes                     |
| `loadDatasetFromBinary()`  | Load dataset from binary                | Yes                | Yes                     |
| `head()`                   | Print first N rows formatted            | Yes                | Yes                     |
| `printDimensions()`        | Print dataset shape                     | Yes                | Yes                     |
| `describeDataset()`        | Numeric descriptive stats               | Yes (numeric-only) | No                      |
| `splitFeaturesAndLabels()` | Split features and labels               | Yes                | Yes                     |
| `getIndices()`             | Generate shuffled or sequential indices | No                 | N/A                     |
| `selectRowsByIndices()`    | Select rows by index                    | Yes                | Yes                     |
| `trainTestSplit()`         | Split dataset into train/test           | Yes                | Yes                     |
| `squeeze()`                | Flatten 2D to 1D vector                 | Yes                | Yes                     |
| `unsqueeze()`              | Add dimension to 1D vector              | Yes                | Yes                     |
| `reshape()`                | Reshape 2D vector                       | Yes                | Yes                     |
| `transpose()`              | Transpose 2D matrix                     | Yes                | Yes                     |

---

## **7. Learnings & Evolution**

* Template specialization is key for safe and efficient parsing of diverse data types.
* C++ requires explicit handling of binary data for non-POD types like strings.
* Inspired by PyTorch, tensor utilities like `squeeze` and `unsqueeze` were implemented early for smooth data transformations.
* Formatting output to mimic Python libraries took careful handling but greatly improves usability.
* Modularizing helper functions improved code clarity and reusability.

---

## **8. To-Do for Future Versions**

* [ ] Auto-detect and parse CSV headers.
* [ ] Support for rows with mixed data types (`int`, `double`, `string` in same row).
* [ ] Implement missing value handling for all types.
* [ ] Expand tensor utilities with batch dimensions, higher-dimensional tensors.
* [ ] More comprehensive error handling and input validation.

---

## **9. Limitations (as of v1.2)**

* No automatic header parsing in CSV files.
* Missing value handling limited to double NaN checks only.
* No support for heterogeneous data types in a single dataset row.
* `unsqueeze()` only supports axis 0 and 1.
* `describeDataset()` restricted to numeric types (due to static\_assert).
* Helper functions separated — require inclusion of `Helper_functions.h` for full functionality.

---

## **10. Version Log**

| Version | Date       | Changes                                                                                                           |
| ------- | ---------- | ----------------------------------------------------------------------------------------------------------------- |
| 1.0     | 12-05-2025 | Initial dataset loading/saving, basic parsing and printing utilities                                              |
| 1.1     | 05-06-2025 | Added binary I/O with string specialization, improved train/test split                                            |
| 1.2     | 19-06-2025 | Added tensor-like utilities (`squeeze()`, `unsqueeze()`, `reshape()`, `transpose()`), enhanced printing and stats |
