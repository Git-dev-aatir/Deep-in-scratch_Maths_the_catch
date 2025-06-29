# Neural Network Library (C++)

## 📝 Overview  
This project implements a comprehensive neural network library in C++ with support for:
- Layer management (Dense, Activation)
- Optimization algorithms (SGD with momentum)
- Data handling (Dataset, DataLoader)
- Preprocessing utilities
- Loss functions and metrics
- Learning rate scheduling

## 🔑 Key Features  
- **Layers**: Dense, ReLU, Sigmoid, Tanh, Softmax  
- **Optimizers**: SGD with momentum and LR scheduling  
- **Data Handling**: Batch loading, shuffling, preprocessing  
- **Initialization**: Xavier, He, LeCun methods  
- **Losses**: MSE, MAE, Cross-Entropy, Hinge  
- **Utilities**: Activation functions, weight initialization  

## 📁 Folder Structure  
```
project-root/
├── include/               # Header files
│   ├── Data/              # Dataset, DataLoader, Preprocessing
│   ├── Layers/            # Layer implementations
│   ├── Metrics/           # Loss functions and metrics
│   ├── Models/            # Sequential model
│   ├── Optimizers/        # Optimization algorithms
│   └── Utils/             # Utility functions
├── src/                   # Implementation files
│   ├── Data/              # Dataset/DataLoader implementations
│   ├── Layers/            # Layer implementations
│   ├── Metrics/           # Loss function implementations
│   ├── Models/            # Sequential model implementation
│   ├── Optimizers/        # Optimizer implementations
│   └── Utils/             # Utility implementations
├── Journey/               # Documentation
│   ├── Data/              # Data-related docs
│   ├── Layers/            # Layer docs
│   ├── Metrics/           # Metrics docs
│   ├── Models/            # Model docs
│   ├── Optimizers/        # Optimizer docs
│   └── Utils/             # Utility docs
├── Examples/              # Usage examples
├── Makefile               # Build configuration
└── README.md              # This file
```

## 🛠️ Building  
1. **Prerequisites**: C++17 compiler (GCC/Clang)  
2. **Build**:  
```bash
make
```

## ▶️ Running Examples  
To compile and run an example file:  
```bash
make run FILE=Examples/your_example.cpp
```
Replace `your_example.cpp` with your example file path relative to project root.

## 🧹 Cleaning Build Artifacts  
```bash
make clean
```

## ⚠️ Note  
All file paths should be given relative to the project root directory.

## 🚀 Example Usage  
```cpp
#include "Models/Sequential.h"
#include "Layers/Layers.h"
#include "Optimizers/SGD.h"
#include "Metrics/Losses.h"

int main() {
    // Create network
    Sequential model(
        new DenseLayer(784, 256),
        new ActivationLayer(ActivationType::RELU),
        new DenseLayer(256, 10),
        new ActivationLayer(ActivationType::SOFTMAX)
    );

    // Initialize parameters
    model.initializeParameters(42);

    // Create optimizer
    SGD optim(0.01, 0.9); // LR=0.01, momentum=0.9

    // Training loop
    for (int epoch = 0; epoch < 10; epoch++) {
        double loss = model.train(X_train, y_train, optim, 64,
                                  Losses::cross_entropy_loss,
                                  Losses::cross_entropy_derivative);
        std::cout << "Epoch " << epoch << " loss: " << loss << "\n";
    }
    return 0;
}
```

## 📦 Dependencies  
- C++17 standard library  
- STL components only (no external dependencies)  

## 📚 Documentation  
Comprehensive documentation available in the `Journey/` directory:  
- Layer implementations  
- Optimization algorithms  
- Data handling  
- Utility functions  
- Model architecture  

## 📄 License  
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.