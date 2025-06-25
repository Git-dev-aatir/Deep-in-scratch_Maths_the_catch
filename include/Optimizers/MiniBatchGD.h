#ifndef MINIBATCH_GD_H
#define MINIBATCH_GD_H

#include "BaseOptim.h"

/**
 * @brief Mini-Batch Gradient Descent Optimizer.
 * 
 * Performs updates after processing a set number of samples (mini-batch size).
 */
class MiniBatchGD : public Optimizer {
private:
    size_t mini_batch_size; ///< Size of each mini-batch.
    size_t sample_count;    ///< Number of samples processed.

public:
    /**
     * @brief Constructor for MiniBatchGD.
     * @param lr Learning rate (default 0.01).
     * @param mb_size Mini-batch size (default 1).
     */
    MiniBatchGD(double lr = 0.01, size_t mb_size = 1);

    void step(std::vector<DenseLayer*>& layers) override;
    void step_per_sample(std::vector<DenseLayer*>& layers) override;
    void step_after_batch(std::vector<DenseLayer*>& layers) override;
};

#endif // MINIBATCH_GD_H
