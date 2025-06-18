# 📦 **Dataset Utility Module — Documentation (Updated)**

---

## **1. Overview**

This module provides a **generic, reusable, and efficient utility** for handling datasets in C++ Machine Learning or Data Processing projects.
It offers functionality for:

1. **Loading** datasets from `.csv`, `.data`, and `.bin` formats.
2. **Saving** datasets as `.bin` to avoid repeated parsing.
3. **Splitting** datasets into training and test sets (with or without shuffling).
4. **Separating** features and labels.
5. **Viewing dataset heads** and dimensional info for debugging.
6. **Full template support**: `int`, `double`, `std::string`.

---

## **2. Files Involved**

```
/include/dataset_utils.h    // Header with template dataset functions
/src/main.cpp               // Example usage (if any)
/Datasets/                  // Data files: .csv, .data, .bin
```

---

## **3. Features**

### 🔹 **Loading**

✔️ From `.csv` / `.data` files with user-defined **delimiter** (`char`) and **multi-space support**.
✔️ From **binary (.bin)** files — faster loading for repeated experiments.

---

### 🔹 **Saving**

✔️ Save any dataset as a **binary file (.bin)**.
✔️ Special handling for `std::string` type.

---

### 🔹 **Splitting**

✔️ Split into **train-test sets** with user-controlled **test fraction** (default 0.2).
✔️ Supports **shuffling** with random indices.
✔️ Internally uses `getIndices()` to create random or ordered indices.

---

### 🔹 **Feature/Label Separation**

✔️ Splits dataset into **Features** (`n x m-1`) and **Labels** (`n x 1`).
✔️ Ensures **labels remain as 2D** (no dimension loss).

---

### 🔹 **Printing & Debugging**

✔️ `printDimensions()` — shows dataset shape `[rows x cols]`.
✔️ `head()` — prints the **first N rows** in a nicely formatted table view.
✔️ Auto-handles empty datasets gracefully.

---

### 🔹 **Template Support**

✔️ Generic functions work for these types:

```
int      double      std::string
```

✔️ Binary save/load **specialized for `std::string`** (with length encoding).

---

## **4. Technical Notes & Challenges**

| Challenge                                                   | Solution                                                   |
| ----------------------------------------------------------- | ---------------------------------------------------------- |
| CSV parsing with **multiple spaces or single delimiter**    | Regex used for space-splitting; delimiter-based otherwise. |
| String trimming for CSV lines needed                        | **Trimmed** before processing each line.                   |
| Binary save/load of `std::string` needs **length encoding** | Specialized template writes **length + content**.          |
| **Features/Labels shape issue** during splitting            | Explicit handling to wrap label in a **1-column DataRow**. |
| Needed flexible **train-test splitting** with shuffling     | Built **index generator function** (`getIndices`).         |

---

## **5. Important Functions**

| Function                   | Purpose                                   | Template?              |
| -------------------------- | ----------------------------------------- | ---------------------- |
| `loadDataset()`            | Load from CSV/data text files             | ✅ Yes                  |
| `saveDatasetToCSV()`       | Save dataset as CSV file                  | ✅ Yes                  |
| `saveDatasetToBinary()`    | Save as binary (specialized for string)   | ✅ Yes (Specialization) |
| `loadDatasetFromBinary()`  | Load from binary (specialized for string) | ✅ Yes (Specialization) |
| `head()`                   | Print first N rows with formatting        | ✅ Yes                  |
| `printDimensions()`        | Print shape of dataset                    | ✅ Yes                  |
| `splitFeaturesAndLabels()` | Separate Features & Labels                | ✅ Yes                  |
| `getIndices()`             | Get shuffled or ordered row indices       | ❌ No                   |
| `selectRowsByIndices()`    | Select rows based on index list           | ✅ Yes                  |
| `trainTestSplit()`         | Split dataset into training and test sets | ✅ Yes                  |

---

## **6. Learning Points**

✔️ **Templating allows single code base** for `int`, `double`, `string`.
✔️ Binary file I/O is **customized for strings** to avoid read errors.
✔️ `getIndices()` allows flexible shuffling logic separated from data logic.
✔️ Printing (`head()`) uses **formatted, column-aligned output**.

---

## **7. To-Do / Future Enhancements**

* [ ] CSV **header detection**.
* [ ] Allow **mixed-type columns** (e.g., string + float).
* [ ] Outlier handling (planned in Preprocessing Module).
* [ ] **Random seed control** for reproducibility in shuffling.

---

## **8. Limitations**

* Does not detect or skip **CSV headers**.
* Cannot handle **missing values (NaNs)** yet.
* **Mixed-type rows unsupported** in current version.

---

## **9. Version**

| Version | Date       | Changes                                                                            |
| ------- | ---------- | ---------------------------------------------------------------------------------- |
| 1.1     | 18-06-2025 | Updated for full binary support, feature/label split fix, new `head()` formatting. |
