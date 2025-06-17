# **01\_dataset\_utility**

📝 *Building a Flexible, Reusable Dataset Utility for C++ ML Projects*

---

## 📌 **Purpose**

Before diving into model-building, I wanted a **robust dataset handling utility** — a reusable tool for:

1. **Loading** datasets from `.data`, `.csv`, and `.bin` formats.
2. **Saving** datasets to `.bin` to avoid reloading for every run.
3. **Splitting** datasets into train/test sets with or without shuffling.
4. **Separating** features and labels cleanly.
5. Supporting **template data types**: `int`, `double`, `string` — not limited to any type.
6. Allowing **print and dimension checking** for easier debugging.

---

## 🔧 **Features Implemented**

* **Loading**:
  ✔️ From `.csv`, `.data` with custom delimiters (space, comma, etc.)
  ✔️ From `.bin` (binary format) — for faster reloads.

* **Saving**:
  ✔️ Dataset can be saved as `.bin`.

* **Splitting**:
  ✔️ Random shuffling option
  ✔️ User-defined test fraction (e.g., 0.2 for 20% test set)
  ✔️ Returns train/test datasets separately.

* **Feature/Label Split**:
  ✔️ Features = all columns except last
  ✔️ Labels = last column (kept as `[n x 1]` matrix)

* **Printing**:
  ✔️ Head preview (first `n` rows)
  ✔️ Print dataset dimensions: `[rows x columns]`

* **Template Support**:
  ✔️ Generic for **`int`**, **`double`**, **`string`** datasets.

---

## ⚠️ **Challenges Faced**

| Issue                                                                                   | Resolution                                                                          |
| --------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `splitFeaturesAndLabels` was producing `[n x 20]` instead of `[n x 1]` for labels.      | Fixed by wrapping `row.back()` inside a **new `DataRow`**, keeping shape `[n x 1]`. |
| Auto-structured bindings (`auto [X, y] = ...`) caused errors.                           | Required enabling **C++17** explicitly in compiler.                                 |
|Had to **load dataset again and again** every time when testing something which needed parsing the .csv or .data file everytime. | Saved the data loaded first time on a .bin file for faster reloads next time. (Can also used to save preprocessed training and test sets)
| Binary read/write had to be **templated properly** to handle `int`, `double`, `string`. | Wrote specialized template versions for each type.                                  |
| Needed **shuffled index list** separately from data split to make splitting flexible.   | Made `getIndices()` return shuffled or ordered indices to slice datasets properly.  |

---

## 🤔 **Learning Points**

1. **Dataset handling is crucial**: Repeated parsing/loading slows down training loops. Binary saving fixes this.
2. **C++ Templates are powerful** but require thoughtful specialization (e.g., `std::string` cannot be `read()`/`write()` like `double`).
3. **Shuffling and splitting by indices** is cleaner than directly splitting the data.
4. Keeping **labels as `[n x 1]` datasets** avoids shape bugs in future matrix operations.
5. Type printing in C++ (`typeid(var).name()`) gives **mangled names** — needs demangling for clean output.

---

## 💡 **Future Considerations**

* **Automatic header detection** in CSVs.
* Allow **random seed setting** for reproducible splits.
* Handle **data types** other than int, double and String.
* Allow storage of **multiple datatypes** in a single dataset. 

---

## ✅ **Checklist for Dataset Utility (Complete)**

* [x] Load from CSV/Data files
* [x] Save & load from Binary
* [x] Split into Train/Test
* [x] Split into Features/Labels
* [x] Print Dimensions
* [x] Template support (`int`, `double`, `string`)
* [x] Handle whitespace, trimming
* [x] Shuffle option
* [x] Reusable index generation

---

## 📚 **Related Files**

```
/include/dataset_utils.h
/src/main.cpp
/Datasets/ (user-supplied)
```
