#include "../../include/Optimizers/BaseOptim.h"

Optimizer::Optimizer(double lr) : learning_rate(lr) {}

Optimizer::~Optimizer() {}

void Optimizer::before_epoch(std::vector<DenseLayer*>& layers) {}

void Optimizer::step_per_sample(std::vector<DenseLayer*>& layers) {}

void Optimizer::step_after_batch(std::vector<DenseLayer*>& layers) {}

void Optimizer::clear_gradients(std::vector<DenseLayer*>& layers) {
    for (auto* layer : layers) {
        layer->clearGradients();
    }
}

void Optimizer::after_epoch(std::vector<DenseLayer*>& layers) {}
