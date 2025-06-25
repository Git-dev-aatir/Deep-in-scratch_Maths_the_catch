#pragma once

#include <vector>
#include <iostream>
#include "../Layers/Layers.h"
#include "../Optimizers/Optim.h"

// Note : Sequential takes ownership of all layers within it 
//        and these layer pointers or layer shouldn't be used anywhere else
//        otherwise the destructor on destroying the layer may create a dangling pointer.

// Note : Current assumption is that only dense layers are trainable by optimizers.

/**
 * @brief Sequential container for stacking layers, similar to PyTorch.
 */
class Sequential {
private:
    /**
     * @brief Container to hold pointers to layers in the sequence.
     */
    std::vector<BaseLayer*> layers;

    /**
     * @brief Flag to track initialization of parameters.
     */
    bool is_initialized = false;

    /**
     * @brief Base case for recursive unpacking of variadic template arguments.
     * 
     * This function is called when no more arguments are left to unpack.
     */
    void addLayers() {} // Base case - do nothing

    /**
     * @brief Recursively unpacks the variadic template arguments and adds each Layer pointer to the model.
     * 
     * @tparam First Type of the first argument (should be derived from Layer*).
     * @tparam Rest Variadic template parameter pack for remaining arguments.
     * @param first Pointer to the first Layer to be added.
     * @param rest Remaining Layer pointers to be processed recursively.
     */
    template<typename First, typename... Rest>
    void addLayers(First first, Rest... rest) {
        this->layers.push_back(first);
        this->addLayers(rest...); // Recursive unpacking
    }

public:
    
    /**
     * @brief Variadic template constructor to accept any number of Layer pointers.
     * 
     * This constructor allows initializing the Sequential model with any number of
     * layers directly at the time of object creation.
     * 
     * @tparam Layers Variadic template parameter representing types derived from Layer*.
     * @param args Pointers to Layer objects to be added to the model.
     */
    template<typename... Layers>
    Sequential(Layers... args) {
        this->addLayers(args...);
    }

    /**
     * @brief Initializes weights and biases of Dense layers based on their subsequent ActivationLayers.
     * 
     * Chooses initialization method based on the next activation type:
     * - He for ReLU / Leaky ReLU
     * - Xavier for Sigmoid / Tanh
     * - LeCun for SELU
     * - Xavier as safe fallback
     * 
     * @param seed Random seed for reproducibility.
     * @param a Lower bound for uniform or mean for normal distribution.
     * @param b Upper bound for uniform or variance for normal distribution.
     * @param sparsity Fraction of weights to be set to zero.
     * @param bias_value Constant bias value for biases.
     */
    void initializeParameters(unsigned int seed = MANUAL_SEED, 
                            double a = 0, double b = 1.0, 
                            double sparsity = 0.0, double bias_value = 0.1);

    /**
     * @brief Performs a forward pass through all layers.
     * 
     * @param input Input vector of doubles.
     * @return Output vector after passing through the model.
     */
    std::vector<double> forward(const std::vector<double>& input) const;

    /**
     * @brief Performs a backward pass through all layers.
     * 
     * @param grad_output Gradient of the loss with respect to the model output.
     * @param learning_rate Learning rate for parameter updates.
     * @return Gradient of the loss with respect to the model input.
     */
    std::vector<double> backward(const std::vector<double>& grad_output, 
                                 double learning_rate = 0.01);

    /**
     * @brief Performs a training step over a batch of samples with multi-dimensional labels (per-sample loss).
     * 
     * The actual training loop is delegated to the optimizer via train_batch().
     * 
     * @param X Batch of input samples (2D vector of doubles).
     * @param Y_true Batch of true labels (2D vector of doubles).
     * @param loss_func Pointer to the per-sample loss function.
     * @param loss_derivative Pointer to the per-sample loss derivative function.
     * @param optimizer Pointer to the optimizer that will perform the training loop.
     * @return Average loss over the batch.
     */
    double train(const std::vector<std::vector<double>>& X,
                const std::vector<std::vector<double>>& Y_true,
                double (*loss_func)(const std::vector<double>&, const std::vector<double>&),
                std::vector<double> (*loss_derivative)(const std::vector<double>&, const std::vector<double>&),
                Optimizer* optimizer);

    /**
     * @brief Performs a training step over a batch of samples using batch-aware loss functions.
     * 
     * Supports fully vectorized batch loss and derivative computations for efficiency.
     * 
     * @param X Batch of input samples (2D vector of doubles).
     * @param Y_true Batch of true labels (2D vector of doubles).
     * @param loss_func Pointer to the batch loss function.
     * @param loss_derivative Pointer to the batch loss derivative function.
     * @param optimizer Pointer to the optimizer that will perform the training loop.
     * @return Average loss over the batch.
     */
    double train(const std::vector<std::vector<double>>& X,
                 const std::vector<std::vector<double>>& Y_true,
                 double (*loss_func)(const std::vector<std::vector<double>>&, const std::vector<std::vector<double>>&),
                 std::vector<std::vector<double>> (*loss_derivative)(const std::vector<std::vector<double>>&, const std::vector<std::vector<double>>&),
                 Optimizer* optimizer);

    /**
     * @brief Prints a summary of the model architecture.
     */
    void summary() const;

    /**
     * @brief Destructor to release dynamically allocated layers.
     */
    ~Sequential();
};
