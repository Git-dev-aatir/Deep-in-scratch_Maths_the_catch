#ifndef BASEOPTIM_H
#define BASEOPTIM_H

#include "../Layers/DenseLayer.h"
#include <vector>

/**
 * @brief Abstract base class for all optimizers.
 * 
 * Defines the interface for optimizers and basic operations on Dense layers.
 */
class Optimizer {
protected:
    double learning_rate; ///< Learning rate for the optimizer.

public:
    /**
     * @brief Constructor for Optimizer.
     * @param lr Learning rate value (default: 0.01).
     */
    explicit Optimizer(double lr = 0.01);

    /**
     * @brief Virtual destructor.
     */
    virtual ~Optimizer();

    /**
     * @brief Perform parameter updates on all Dense layers.
     * @param layers Vector of DenseLayer pointers to update.
     */
    virtual void step(std::vector<DenseLayer*>& layers) = 0;

    /**
     * @brief Hook before each epoch starts (optional).
     * @param layers Vector of DenseLayer pointers.
     */
    virtual void before_epoch(std::vector<DenseLayer*>& layers);

    /**
     * @brief Update step after each sample (for SGD).
     * @param layers Vector of DenseLayer pointers.
     */
    virtual void step_per_sample(std::vector<DenseLayer*>& layers);

    /**
     * @brief Update step after entire batch (for Batch GD).
     * @param layers Vector of DenseLayer pointers.
     */
    virtual void step_after_batch(std::vector<DenseLayer*>& layers);

    /**
     * @brief Clears gradients in all Dense layers.
     * @param layers Vector of DenseLayer pointers.
     */
    virtual void clear_gradients(std::vector<DenseLayer*>& layers);

    /**
     * @brief Hook after each epoch ends (optional).
     * @param layers Vector of DenseLayer pointers.
     */
    virtual void after_epoch(std::vector<DenseLayer*>& layers);
};

#endif // BASEOPTIM_H
