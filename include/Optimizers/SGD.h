#pragma once

#include "BaseOptim.h"
#include <vector>
#include <unordered_map>
#include <functional>

class SGD : public BaseOptim {
private:
    double learning_rate;
    double initial_lr;
    double momentum;
    std::unordered_map<BaseLayer*, std::vector<std::vector<double>>> velocity_weights;
    std::unordered_map<BaseLayer*, std::vector<double>> velocity_biases;

    /**
     * @brief Updates parameters for a single layer.
     * @param layer Pointer to the layer to update.
     * @param batch_size Batch size for gradient normalization.
     */
    void updateLayer(BaseLayer* layer, size_t batch_size);

    // Learning rate scheduler
    std::function<double(double, size_t)> lr_scheduler = nullptr;
    size_t step_count = 0;

public:
    /**
     * @brief Constructor with scheduler support.
     * @param lr Learning rate (default=0.01).
     * @param momentum Momentum factor (default=0.0).
     * @param scheduler Learning rate scheduler function (init_lr, step) -> new_lr.
     */
    SGD(double lr = 0.01, 
        double momentum = 0.0,
        std::function<double(double, size_t)> scheduler = nullptr);
    
    // Implement BaseOptim interface
    void step(std::vector<BaseLayer*> layers, size_t batch_size) override;
    void setLearningRate(double lr) override ;
    void decayLearningRate(double decay_factor) override ; 
    double getLearningRate() const override { return learning_rate; }
    
    /**
     * @brief Set momentum.
     * @param m New momentum value.
     */
    void setMomentum(double m) { momentum = m; }

    // New scheduling features
    void setLRScheduler(std::function<double(double, size_t)> scheduler);
    void resetStepCount() { step_count = 0; }
    void afterStep();  // Call after each batch
};
