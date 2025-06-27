#pragma once

#include <vector>
#include <memory>
#include <stdexcept>
#include <functional> 
#include "Data/DataLoader.h"
#include "../Layers/Layers.h"
#include "../Optimizers/SGD.h"

// Note : Sequential takes ownership of all layers within it 
//        and these layer pointers or layer shouldn't be used anywhere else
//        otherwise the destructor on destroying the layer may create a dangling pointer.

// Note : Current assumption is that only dense layers are trainable by optimizers.

/**
 * @brief Sequential container for neural network layers.
 * 
 * Allows stacking layers in sequence and performing forward/backward passes.
 */
class Sequential {
private:
    /**
     * @brief Container to hold pointers to layers in the sequence.
     */
    std::vector<std::unique_ptr<BaseLayer>> layers;

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
    void addLayers(First&& first, Rest&&... rest) {
        layers.push_back(std::forward<First>(first));
        addLayers(std::forward<Rest>(rest)...);
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
    Sequential(Layers&&... args) {
        addLayers(std::forward<Layers>(args)...);
    }

    // /**
    //  * @brief Destructor to release dynamically allocated layers.
    //  */
    // ~Sequential();

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
     * @brief Perform forward pass through all layers.
     * @param input Input vector.
     * @return Output vector after processing through all layers.
     */
    std::vector<double> forward(const std::vector<double>& input) const;

    /**
     * @brief Perform backward pass through all layers.
     * @param grad_output Gradient from the loss function.
     * @param lr Learning rate (unused in backward pass).
     * @return Gradient with respect to the input.
     */
    std::vector<double> backward(const std::vector<double>& grad_output);

    /**
     * @brief Print summary of all layers.
     */
    void summary() const;

    /**
     * @brief Get number of layers.
     * @return Number of layers in the sequence.
     */
    size_t size() const {
        return layers.size();
    }

    /**
     * @brief Performs one training pass over the entire dataset.
     * @param X_train Input features dataset.
     * @param y_train Target labels dataset.
     * @param optimizer Optimizer to use for weight updates.
     * @param batch_size Size of each training batch.
     * @param loss_fn Loss function (y_true, y_pred) -> double.
     * @param grad_fn Gradient function (y_true, y_pred) -> vector<double>.
     * @return Total loss over the training set.
     */
    double train(const Dataset& X_train,
                 const Dataset& y_train,
                 BaseOptim& optimizer,
                 size_t batch_size,
                 std::function<double(const std::vector<double>&, 
                                      const std::vector<double>&)> loss_fn,
                 std::function<std::vector<double>(const std::vector<double>&, 
                                                   const std::vector<double>&)> grad_fn);

    double train(
        const Dataset& X_train,
        const Dataset& y_train,
        BaseOptim& optimizer,
        size_t batch_size,
        std::function<double(const std::vector<std::vector<double>>&, 
                            const std::vector<std::vector<double>>&)> batch_loss_fn,
        std::function<std::vector<std::vector<double>>(const std::vector<std::vector<double>>&, 
                                                    const std::vector<std::vector<double>>&)> batch_grad_fn
    );

    
    void clearGradients();
        
    /**
     * @brief Access layer by index.
     * @param index Layer index.
     * @return Pointer to the layer.
     */
    BaseLayer* operator[](size_t index) {
        if (index >= layers.size()) 
            throw std::out_of_range("Index out of range");
        return layers[index].get();
    }

    /**
     * @brief Get all layers in the sequence.
     * @return Vector of pointers to the layers.
     */
    std::vector<BaseLayer*> getLayers() {
        std::vector<BaseLayer*> layer_ptrs;
        for (auto& layer : layers) {
            layer_ptrs.push_back(layer.get());
        }
        return layer_ptrs;
    }

};
