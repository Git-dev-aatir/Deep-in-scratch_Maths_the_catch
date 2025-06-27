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


Ah, I see what you're asking now! You're wondering how optimizers like **mini-batch** or **stochastic gradient descent (SGD)** control when to update the parameters—whether after processing each individual example, after a mini-batch (like 32 examples), or after the entire dataset (which would be **full-batch gradient descent**).

### **How the Optimizer Works with Mini-Batches**

The key thing to understand is that **the optimizer** doesn't directly control when to update the model parameters. It simply updates the parameters after the **gradient** is computed. The **training loop** (or `model.train()` function) controls the process of feeding data into the model, calculating gradients, and deciding when the optimizer should step (i.e., perform the parameter update).

### **Training with Mini-Batches:**

In **mini-batch gradient descent**, the key idea is to split your training dataset into smaller chunks (mini-batches) and then update the model's parameters after each mini-batch, rather than after each individual example or after the entire dataset.

#### Here's how it works step-by-step:

1. **Training Loop**: The loop controls how the data is divided into mini-batches.

   * For each epoch, the training set is divided into batches (e.g., 32 examples per batch).
   * You iterate over each batch, performing forward and backward passes.

2. **Gradient Calculation (Forward + Backward)**: After processing each mini-batch, the loss for that mini-batch is calculated, and gradients are computed for the model's parameters.

3. **Optimizer Step**: After the gradients are computed for each mini-batch, the optimizer uses those gradients to update the parameters.

#### How this is implemented:

* **Mini-batch size** is determined when you configure the **data loader** (or however you load your dataset).
* After each mini-batch, the optimizer performs an update.

### **Structure of the Training Loop with Mini-Batch Update**

Let’s walk through a conceptual example (similar to PyTorch-like pseudocode):

```cpp
void train(Model& model, DataLoader& train_data, Optimizer& optimizer, int batch_size, int epochs) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        model.train();  // Set model to training mode
        for (auto& batch : train_data.getMiniBatches(batch_size)) {  // Assuming getMiniBatches yields batches of size batch_size
            optimizer.zero_grad();  // Clear previous gradients

            auto inputs = batch.inputs;
            auto targets = batch.targets;

            // Forward pass
            auto predictions = model(inputs);

            // Compute loss
            auto loss = computeLoss(predictions, targets);

            // Backward pass (compute gradients)
            loss.backward();

            // After the gradients are computed for this mini-batch, optimizer updates the model parameters
            optimizer.step();
        }
    }
}
```

### **Details about the Optimizer:**

* **Optimizer** (like SGD, Adam) doesn't control when to update the parameters directly—it simply performs the update **after each call to `optimizer.step()`**.
* The optimizer **doesn't need to know** whether you're using mini-batch, stochastic, or full-batch gradient descent. This is controlled by the **training loop**, where you decide how much data to process before performing the update.
* After each mini-batch, the optimizer performs an update to the model parameters based on the gradients computed during the backward pass.

### **Mini-Batch Control:**

* **Mini-batch size** (e.g., 32) is controlled by the **data loader** or the way the data is partitioned into batches.
* The optimizer is agnostic to the batch size—it just performs an update after every gradient calculation (which comes after a batch is processed).

### **Full-Batch vs Mini-Batch vs Stochastic Gradient Descent:**

1. **Stochastic Gradient Descent (SGD)**:

   * Updates after each individual example.
   * Batch size = 1 (each update is based on a single example).
   * This is often slower but can help escape local minima due to its noisiness.

2. **Mini-Batch Gradient Descent**:

   * Updates after a fixed-sized batch (e.g., 32 or 64).
   * Often a good trade-off between the efficiency of full-batch gradient descent and the noisiness of SGD.

3. **Full-Batch Gradient Descent**:

   * Updates after processing the entire dataset (i.e., all training examples in one go).
   * Computationally expensive and might be slow for large datasets.

### **Key Takeaways:**

* **Mini-batch control** (how many examples to process before an update) is handled by the **training loop** (or `model.train()` function), specifically by the way you set up your data loading and batch processing.
* The **optimizer** just updates the model after every mini-batch (or full-batch or stochastic pass), based on the gradients calculated from that mini-batch.
* The optimizer does **not** decide when to update—it simply takes gradients from the training loop and applies the updates.
