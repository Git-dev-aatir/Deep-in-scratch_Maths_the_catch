#ifndef OPTIMISER_H
#define OPTIMISER_H

#include "../Layers/Dense.h"

/**
 * @brief Abstract base class for all optimizers.
 */
class Optimizer {
protected:
    double learning_rate; ///< Learning rate used by the optimizer.

public:
    /**
     * @brief Constructor for Optimizer.
     * @param lr Learning rate (default 0.01).
     */
    Optimizer(double lr = 0.01) : learning_rate(lr) {}

    /**
     * @brief Performs an optimization step on a vector of Dense layers.
     * @param layers Vector of pointers to Dense layers to update.
     */
    virtual void step(std::vector<Dense*>& layers) = 0;

    virtual ~Optimizer() {}
};

#endif // OPTIMISER_H
