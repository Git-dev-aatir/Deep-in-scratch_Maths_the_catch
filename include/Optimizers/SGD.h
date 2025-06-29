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
    size_t batch_size;
    std::unordered_map<BaseLayer*, std::vector<std::vector<double>>> velocity_weights;
    std::unordered_map<BaseLayer*, std::vector<double>> velocity_biases;
    double clip_value_ = 0;  // Add clipping threshold

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
     * @param batch_size Size of mini batch
     * @param scheduler Learning rate scheduler function (init_lr, step) -> new_lr.
     */
    SGD(double lr = 0.01, 
        double momentum = 0.0,
        size_t batch_size = 0,
        std::function<double(double, size_t)> scheduler = nullptr);
    
    // Implement BaseOptim interface
    void step(std::vector<BaseLayer*> layers, size_t batch_size) override;
    void setLearningRate(double lr) override ;
    void decayLearningRate(double decay_factor) override ; 
    double getLearningRate() const override { return learning_rate; }
    size_t getBatchSize() const override { return this->batch_size; }
    
    void setBatchSize(size_t new_batch_size) override { this->batch_size = new_batch_size; }
    
    /**
     * @brief Set momentum.
     * @param m New momentum value.
     */
    void setMomentum(double m) { momentum = m; }

    void setGradientClip(double clip) { clip_value_ = clip; }

    // New scheduling features
    void setLRScheduler(std::function<double(double, size_t)> scheduler);
    void resetStepCount() { step_count = 0; }
    void afterStep();  // Call after each batch
};
