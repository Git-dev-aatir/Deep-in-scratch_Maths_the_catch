#ifndef DENSE_H
#define DENSE_H

#include "Layers.h"

/**
 * @brief Dense (fully connected) layer with gradient storage for optimization.
 */
class Dense : public Layer {
private:
    size_t input_size, output_size;
    vector<vector<double>> weights;
    vector<double> biases;

    // To store gradients for optimization
    vector<vector<double>> grad_weights;
    vector<double> grad_biases;

    // Cache for backpropagation
    vector<double> input_cache;

public:
    Dense(size_t in_features, size_t out_features, bool init_params=true)
        : input_size(in_features), output_size(out_features) 
    {
        if (init_params) {
            this->weights.resize(out_features, vector<double>(in_features, 0));
            this->biases.resize(out_features, 0);
        }

        // Initialize gradients to zero
        grad_weights.resize(this->output_size, vector<double>(this->input_size, 0.0));
        grad_biases.resize(this->output_size, 0.0);
    }


    /**
     * @brief Initializes a 2D weight matrix according to the specified method.
     * 
     * @param method The initialization method to use (Radon, Xavier, He, LeCun, Orthogonal, Bias).
     * @param seed Random seed for reproducibility.
     * @param sparsity Fraction [0,1] of weights to be set to zero (applies to all methods).
     * @param bias_value For BIAS initialization only: constant value to set.
     * @param a lower bound for UNIFORM, or mean for NORMAL distribution.
     * @param b upper bound for UNIFORM, or variance for NORMAL distribution.
     */
    void initializeWeights(
                       InitMethod method,
                       unsigned int seed = MANUAL_SEED,
                       double a = 0,
                       double b = 1.0,
                       double sparsity = 0.0,
                       double bias_value = 0.0) {
    
        this->weights = initializeParameters(this->input_size, this->output_size, method,
                                             seed, a, b, sparsity, bias_value);
        cout << "Weights\n";
        printDimensions(this->weights);
        head(weights, weights.size());
    }

    /**
     * @brief Initializes a 2D biases vector according to the specified method.
     * 
     * @param method The initialization method to use (Radon, Xavier, He, LeCun, Orthogonal, Bias).
     * @param seed Random seed for reproducibility.
     * @param sparsity Fraction [0,1] of biases to be set to zero (applies to all methods).
     * @param bias_value For BIAS initialization only: constant value to set.
     * @param a lower bound for UNIFORM, or mean for NORMAL distribution.
     * @param b upper bound for UNIFORM, or variance for NORMAL distribution.
     */
    void initializeBiases(
                       InitMethod method,
                       unsigned int seed = MANUAL_SEED,
                       double a = 0,
                       double b = 1.0,
                       double sparsity = 0.0,
                       double bias_value = 0.0) {
    
        std::vector<std::vector<double>> kd = initializeParameters(1, this->output_size,
                                            method, seed, a, b, sparsity, bias_value);
        this->biases = kd[0];
        cout << "Biases \n";
        printDimensions(kd);
        head(kd, kd.size());
    }


    vector<double> forward(const vector<double>& input) override {
        assert(input.size() == this->input_size);
        this->input_cache = input;

        vector<double> output(output_size, 0.0);
        for (size_t i = 0; i < output_size; ++i) {
            for (size_t j = 0; j < input_size; ++j) {
                output[i] += this->weights[i][j] * input[j];
            }
            output[i] += this->biases[i];
        }
        return output;
    }

    vector<double> backward(const vector<double>& grad_output, double learning_rate) override {
        vector<double> grad_input(input_size, 0.0);

        // Compute gradients w.r.t. input
        for (size_t i = 0; i < input_size; ++i) {
            for (size_t j = 0; j < output_size; ++j) {
                grad_input[i] += this->weights[j][i] * grad_output[j];
            }
        }

        // Compute gradients w.r.t. weights and biases (store them)
        for (size_t i = 0; i < output_size; ++i) {
            for (size_t j = 0; j < input_size; ++j) {
                this->grad_weights[i][j] = grad_output[i] * input_cache[j];
            }
            this->grad_biases[i] = grad_output[i];
        }

        // Gradient Descent Update (can be removed if using optimizers later)
        for (size_t i = 0; i < this->output_size; ++i) {
            for (size_t j = 0; j < input_size; ++j) {
                this->weights[i][j] += learning_rate * this->grad_weights[i][j];
            }
            this->biases[i] += learning_rate * this->grad_biases[i];
        }

        return grad_input;
    }

    void summary() const override {
        std::cout << "Dense Layer: " << input_size << " -> " << output_size << std::endl;
    }

    // Getters for optimizers (if needed later)
    const vector<vector<double>>& getGradWeights() const { return this->grad_weights; }
    const vector<double>& getGradBiases() const { return this->grad_biases; }

    const vector<vector<double>>& getWeights() const { return this->weights; }
    const vector<double>& getBiases() const { return this->biases; }
};

#endif // DENSE_H
