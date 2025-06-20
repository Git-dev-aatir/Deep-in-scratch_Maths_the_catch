# **ML & Deep Learning from Scratch in C++** 🧠💻

A handmade forge of machine learning and deep learning tools using pure number wizardry — to predict and understand the world from the ground up.

---

## 🧩 **Introduction**

This repository contains carefully crafted **Machine Learning and Deep Learning algorithms implemented entirely from scratch in C++** — no external ML libraries, no shortcuts. Just pure logic, mathematics, and a desire to demystify these black-box models.

The aim is educational: build everything yourself to fully understand data preprocessing, model training, inference, and evaluation.

---

## 📂 **Repository Structure**

```
/include
    
    /Preprocessing
        dataset_utils.h      # Templated Dataset loading, saving, splitting, printing
    
    /Losses
        loss functions (mse, mae, bce, cross_entropy, hinge_loss)

    /General
        general features (contains only initialization function for parameters)

    /Layers
        neural network layers (Dense, activation)
    

/src
    main.cpp             # Demo: loading data, splitting into train/test, feature/label

/Datasets
    ...                 # Your personal datasets (this folder is gitignored)

/Journey
    step1.md             # Detailed personal notes on dataset handling
    step2.md             # Notes on model design choices
    ...more journey logs

.gitignore
README.md
```

---

## 📦 **Features So Far**

* ✔️ **Custom Dataset Loader** (CSV / Data / Binary)
* ✔️ **Save/Load Datasets as .bin** — avoid repeated parsing
* ✔️ **Trim & Handle Flexible Delimiters** — automatic space handling
* ✔️ **Train/Test Split** with shuffling
* ✔️ **Feature/Label Split** — simple and safe
* ✔️ **Print Dataset Dimensions & Preview (head)**
* ✔️ **Documented Development Journey** — every major step is logged in `/Journey`

---

## 🔮 **Planned Algorithms**

> All from scratch — no `sklearn`, `TensorFlow`, or `Eigen`.

* [x] **Dataset Utilities**
* [ ] Linear Regression (in progress)
* [ ] Logistic Regression
* [ ] K-Nearest Neighbors (KNN)
* [ ] Decision Trees
* [ ] Naïve Bayes
* [ ] K-Means Clustering
* [ ] Principal Component Analysis (PCA)
* [ ] Perceptron
* [ ] Feedforward Neural Network
* [ ] Backpropagation
* [ ] Gradient Descent Variants (SGD, Momentum, Adam)

---

## ⚙️ **Planned Preprocessing Utilities**

* [ ] Feature Scaling (Normalization / Standardization)
* [ ] Missing Value Handling
* [ ] Categorical Encoding (One-Hot, Label Encoding)
* [ ] Feature Selection
* [ ] Dimensionality Reduction

---

## 🏗️ **Usage Example**

```cpp
#include "dataset_utils.h"

int main() {
    Dataset<double> housing = loadDatasetFromBinary<double>("Datasets/Boston_housing_dataset/data.bin");

    auto [train_set, test_set] = trainTestSplit(housing, 0.2, true);
    auto [X_train, y_train] = splitFeaturesAndLabels(train_set);
    auto [X_test, y_test] = splitFeaturesAndLabels(test_set);

    printDimensions(X_train);
    printDimensions(y_test);
}
```

---

## 📜 **Datasets**

The `/Datasets` folder is intentionally **excluded** from this repository (`.gitignore`).
Feel free to download and experiment with datasets of your choice (e.g., from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)).

---

## 📝 **Journey Logs**

Every step, experiment, design decision, or error made during this project is documented in the `/Journey` folder in Markdown format:

* [Step 1: Dataset Handling and Loading](./Journey/01_dataset_utility.md)
* [Step 2: Data Preprocessing Notes](./Journey/02_preprocessing_module.md)
* [Step 3: Losses](./Journey/03_losses.md)
* [Step 4: Initialization](./Journey/04_initialization.md)
* [Step 5: Neural Net Layers Implementation with Activation Functions](./Journey/05_layers.md)

✔️ Transparent and raw development logs
✔️ Useful for reflection and learning
✔️ You can follow how each part of this project came into being

---

## ⚙️ **Build & Run**

```bash
g++ -std=c++17 -o app src/main.cpp
./app
```

---

## 💡 **Vision**

✔️ Build **ML & DL models from first principles**
✔️ Demystify math-heavy concepts via real, working C++ code
✔️ Educate yourself and others — no hidden magic, no library dependency

---

## 🧪 **Future Roadmap**

* Unit tests (possibly with Catch2)
* Preprocessing utilities (scaling, missing values, encoding)
* Linear Algebra utilities
* Visualization support (optional)
* Experiments & benchmarks against standard datasets

---

## 🤝 **Contributing**

This is a personal educational project — feel free to suggest improvements, raise issues, or fork for your own ML adventures.

---

*No license has been assigned to this project yet. You are free to read, learn, and modify for personal use.*
