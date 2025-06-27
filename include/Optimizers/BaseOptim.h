#pragma once

#include "../Layers/BaseLayer.h"
#include <vector>
#include <memory>

/**
 * @brief Abstract base class for all optimizers.
 */
class BaseOptim {
public:
    virtual ~BaseOptim() = default;
    
    /**
     * @brief Update parameters for a set of layers.
     * @param layers Vector of layer pointers to update.
     * @param batch_size Batch size used in the current step.
     */
    virtual void step(std::vector<BaseLayer*> layers, size_t batch_size) = 0;

    virtual void afterStep() = 0;  // Add this method
    
    /**
     * @brief Set the learning rate.
     * @param lr New learning rate value.
     */
    virtual void setLearningRate(double lr) = 0;
    
    /**
     * @brief Decay the learning rate by a factor.
     * @param decay_factor Factor by which to multiply the learning rate.
     */
    virtual void decayLearningRate(double decay_factor) = 0;
    
    /**
     * @brief Get the current learning rate.
     * @return Current learning rate value.
     */
    virtual double getLearningRate() const = 0;
};
