#pragma once
#include "../Layers/Layers.h"
#include <unordered_map>
#include <vector>

/**
 * @brief Abstract base class for optimizers.
 */
class Optimizer {
public:
    virtual ~Optimizer() = default;
    
    /**
     * @brief Update parameters for a single layer.
     * @param layer Pointer to the layer to update.
     * @param batch_size Batch size for gradient normalization.
     */
    virtual void step(std::vector<BaseLayer*> layers, size_t batch_size) = 0;
};

/**
 * @brief Stochastic Gradient Descent optimizer.
 * 
 * Supports momentum and learning rate scheduling.
 */
class SGD {
private:
    double learning_rate;
    double momentum;
    std::unordered_map<BaseLayer*, std::vector<std::vector<double>>> velocity_weights;
    std::unordered_map<BaseLayer*, std::vector<double>> velocity_biases;

    /**
     * @brief Updates parameters for a single layer.
     * @param layer Pointer to the layer to update.
     * @param batch_size Batch size for gradient normalization.
     */
    void updateLayer(BaseLayer* layer, size_t batch_size);

public:
    /**
     * @brief Constructor.
     * @param lr Learning rate (default=0.01).
     * @param momentum Momentum factor (default=0.0).
     */
    SGD(double lr = 0.01, double momentum = 0.0);

    /**
     * @brief Update parameters for all layers in a model.
     * @param layers Vector of layer pointers.
     * @param batch_size Batch size for gradient normalization.
     */
    void step(std::vector<BaseLayer*> layers, size_t batch_size);

    void decayLearningRate(double decay_factor) {
        learning_rate *= decay_factor;
    }

    double getLearningRate() const { return learning_rate; }

    
    /**
     * @brief Set learning rate.
     * @param lr New learning rate.
     */
    void setLearningRate(double lr) { learning_rate = lr; }

    /**
     * @brief Set momentum.
     * @param m New momentum value.
     */
    void setMomentum(double m) { momentum = m; }
};
