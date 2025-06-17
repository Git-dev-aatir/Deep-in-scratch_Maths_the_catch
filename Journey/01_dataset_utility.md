Here’s a **unified, consistent, professional style** version of both your documentation files — blending strengths of both.

---

## **01\_dataset\_utility.md**

# 📦 **Dataset Utility Module — Documentation**

---

## **1. Overview**

This module provides a **flexible and reusable utility** for handling datasets in C++ Machine Learning projects.
It offers functionality for:

1. **Loading** datasets from `.csv`, `.data`, and `.bin` formats.
2. **Saving** datasets as `.bin` to avoid repeated parsing.
3. **Splitting** datasets into train/test sets with or without shuffling.
4. **Separating** features and labels.
5. Supporting **templated datatypes**: `int`, `double`, `std::string`.
6. **Printing** datasets and checking dimensions for debugging.

---

## **2. Files Involved**

```
/include/dataset_utils.h
/src/main.cpp
/Datasets/ (user-supplied)
```

---

## **3. Features**

### 🔹 **Loading**

✔️ From `.csv` / `.data` (custom delimiters supported).
✔️ From `.bin` (binary format) — fast reload.

---

### 🔹 **Saving**

✔️ Save any dataset as `.bin` for reuse.

---

### 🔹 **Splitting**

✔️ Random shuffling supported.
✔️ User-defined test fraction (e.g., 0.2 for 20% test set).
✔️ Returns separate train and test sets.

---

### 🔹 **Feature/Label Separation**

✔️ **Features**: all columns except last.
✔️ **Labels**: last column, shape `[n x 1]` (avoids shape bugs in future matrix operations).

---

### 🔹 **Printing & Debugging**

✔️ Print dataset shape (`[rows x columns]`).
✔️ Print head preview (first N rows).

---

### 🔹 **Template Support**

✔️ Generic for **int**, **double**, **std::string**.

---

## **4. Challenges & Solutions**

| Issue                                                         | Solution                                               |
| ------------------------------------------------------------- | ------------------------------------------------------ |
| Labels came as `[n x m]` instead of `[n x 1]`.                | Wrapped `row.back()` in a **new DataRow**.             |
| Auto-structured bindings (`auto [X, y] = ...`) caused errors. | Required **C++17** enabled in compiler.                |
| **Repeated reloading** from CSV/data slow during testing.     | Added **binary saving/loading** to speed up reload.    |
| Binary read/write must handle **different datatypes**.        | Wrote **template specializations**.                    |
| Needed flexible shuffling and splitting.                      | Returned **index vector separately** for easy slicing. |

---

## **5. Learning Points**

✔️ **Binary saving** greatly speeds up development cycles.
✔️ **Templates are powerful** but need proper specialization for strings.
✔️ Separating **indices from data** makes splits cleaner.
✔️ Keeping labels `[n x 1]` prevents downstream shape errors.

---

## **6. To-Do / Future Plans**

* Automatic header detection in CSVs.
* Allow setting random seed for reproducibility.
* Handle multiple datatypes in a single dataset (e.g., mix of numeric & string).
* Outlier handling (in Preprocessing module).

---

## **7. Known Limitations**

* No CSV header detection yet.
* Only single-type datasets per file (no mixed-type rows).
* Missing value handling not covered (in Preprocessing module).

---

## **8. Version**

* **v1.0** — Base dataset utility functions completed.


