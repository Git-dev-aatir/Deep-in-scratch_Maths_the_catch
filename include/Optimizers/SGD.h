#ifndef SGD_H
#define SGD_H

#include "BaseOptim.h"

/**
 * @brief Stochastic Gradient Descent (SGD) Optimizer.
 * 
 * Performs weight and bias updates after each training sample.
 */
class SGD : public Optimizer {
public:
    /**
     * @brief Constructor for SGD.
     * @param lr Learning rate (default 0.01).
     */
    explicit SGD(double lr = 0.01);

    void step_per_sample(std::vector<DenseLayer*>& layers) override;
    void step_after_batch(std::vector<DenseLayer*>& layers) override;
    void step(std::vector<DenseLayer*>& layers) override;
};

#endif // SGD_H
