# 📊 Dataset.md

## 📝 Overview

The `Dataset` class is a core component for data storage and manipulation in the neural network library. It provides robust support for loading, saving, inspecting, transforming, and splitting datasets, with both CSV and binary file format compatibility. The class is designed for flexibility and ease of use in typical machine learning workflows.

---

## ✨ Key Features

- **Flexible Data Loading**: Supports CSV (with customizable delimiters and header handling) and binary formats.
- **Data Saving**: Export datasets to CSV or binary formats for interoperability and efficiency.
- **Inspection Utilities**: Includes shape reporting, head display, and a detailed `describe()` method for column-wise statistics.
- **Data Manipulation**: Enables row selection, feature/label splitting, and train/test splitting (with optional stratification and shuffling).
- **Transformations**: Provides transpose, reshape, flatten, and in-place one-hot encoding for label data.
- **Robust Error Handling**: Throws exceptions for file errors, inconsistent dimensions, and invalid operations.

---

## 🏗️ Design Decisions

- **Row-major Storage**: Uses `std::vector<std::vector<double>>` for intuitive row-wise access and manipulation.
- **Dimension Validation**: Every load or modification validates row and column consistency to prevent subtle bugs.
- **In-place and Copy Operations**: Most data manipulations return new `Dataset` instances, while some (like `toOneHot()`) modify in place.
- **Statistical Reporting**: The `describe()` method computes count of nulls, unique values, mean, std, min, max, and percentiles for each column.
- **Stratified Splitting**: `trainTestSplit()` supports stratified sampling for imbalanced classification tasks.

---

## 🛠️ Problems Faced & Solutions

### 1. Inconsistent Row Dimensions in Input Files
- **Problem**: Real-world CSVs may have missing values or inconsistent columns.
- **Solution**: The implementation checks for row length consistency after loading and throws a descriptive exception if a mismatch is found.

### 2. Robust CSV Parsing
- **Problem**: CSV files may use spaces as delimiters or have multiple consecutive spaces.
- **Solution**: Added a `multiple_spaces` flag to handle such cases. When set and the delimiter is a space, the parser treats consecutive spaces as a single delimiter.

### 3. Stratified Splitting for Imbalanced Data
- **Problem**: Random splits can destroy class balance in small or imbalanced datasets.
- **Solution**: `trainTestSplit()` groups indices by class and splits within each group, guaranteeing class proportions are preserved in both train and test sets.

### 4. One-Hot Encoding Safety
- **Problem**: One-hot encoding is only valid for single-column integer label datasets.
- **Solution**: `toOneHot()` checks that the dataset has exactly one column and that all values are valid non-negative integers before encoding.

### 5. Binary File Compatibility
- **Problem**: Ensuring binary files are portable and robust to header changes.
- **Solution**: Binary files always begin with two `size_t` values (rows, cols) and then the data in row-major order. Skipping headers and partial writes are handled with care.

---

## 🔍 Notable Implementation Details

- **Percentile Calculation**: Uses linear interpolation between sorted values for accurate quantile estimation.
- **Describe Method**: Skips NaN values and reports null counts per column.
- **Row Selection**: `selectRows()` safely skips out-of-range indices.
- **Operator Overloading**: Provides both const and mutable row access via `operator[]`, with bounds checking.

---

## 🚀 Usage Example

```cpp
Dataset iris;
iris.loadCSV("iris.csv", ',', true);

// Show first 5 rows
iris.head();

// Print shape and statistics
iris.printShape();
iris.describe();

// Split features and labels (last column as label)
auto [X, y] = iris.splitFeaturesLabels(-1);

// Stratified train-test split (by label column)
auto [train, test] = iris.trainTestSplit(0.2, -1, true);

// One-hot encode labels
y.toOneHot();
```


---

## ⚡ Performance and Limitations

- **Performance**: For small-to-medium datasets, row-major storage is efficient and simple. For very large datasets, consider memory-mapped files or a flat buffer for improved cache locality.
- **Limitations**:
  - Only supports `double`-precision data.
  - No built-in support for missing value imputation or advanced preprocessing.
  - Column names are not stored or exported.
  - No parallel loading for very large files.

---

## 🚧 Future Improvements

- [ ] Support for column names and metadata.
- [ ] Templated data type support (float/int).
- [ ] Parallelized CSV and binary loading.
- [ ] Out-of-core (memory-mapped) dataset handling.
- [ ] Built-in normalization and missing value imputation.
- [ ] More flexible and robust error reporting.

