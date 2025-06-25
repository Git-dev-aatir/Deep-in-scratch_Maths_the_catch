#ifndef BATCH_GD_H
#define BATCH_GD_H

#include "BaseOptim.h"

/**
 * @brief Batch Gradient Descent Optimizer.
 * 
 * Updates parameters after processing the entire batch.
 */
class BatchGD : public Optimizer {
public:
    /**
     * @brief Constructor for BatchGD.
     * @param lr Learning rate (default 0.01).
     */
    explicit BatchGD(double lr = 0.01);

    void step(std::vector<DenseLayer*>& layers) override;
    void step_per_sample(std::vector<DenseLayer*>& layers) override;
    void step_after_batch(std::vector<DenseLayer*>& layers) override;
};

#endif // BATCH_GD_H
