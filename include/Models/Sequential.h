#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

#include <vector>
#include <iostream>
#include "../Layers/Activation.h"
#include "../Layers/Dense.h"
#include "../Optimizers/Optim.h"
#include "../Optimizers/BatchGD.h"
#include "../Optimizers/SGD.h"

using std::vector;
using std::cout;
using std::endl;

/**
 * @brief Sequential container for stacking layers like in PyTorch.
 */
class Sequential {
private:
    /**
     * @brief Container to hold pointers to layers in the sequence.
     */
    vector<Layer*> layers;

public:
    /**
     * @brief Default constructor for Sequential model.
     */
    Sequential() {}

    /**
     * @brief Adds a new layer to the model.
     * 
     * @param layer Pointer to the layer to add.
     */
    void addLayer(Layer* layer) {
        layers.push_back(layer);
    }

    /**
     * @brief Initializes the weights and biases of Dense layers based on the next ActivationLayer.
     * 
     * Uses sensible defaults:
     * - He for ReLU / Leaky ReLU
     * - Xavier for Sigmoid / Tanh
     * - LeCun for SELU
     * - Xavier as safe fallback
     * 
     * @param seed Random seed for reproducibility.
     * @param a Lower bound for UNIFORM or mean for NORMAL distribution.
     * @param b Upper bound for UNIFORM or variance for NORMAL distribution.
     * @param sparsity Fraction of weights to be set to zero.
     * @param bias_value Constant bias value if using BIAS initialization.
     */
    void initializeParameters(unsigned int seed = MANUAL_SEED, 
                              double a = 0, double b = 1.0,
                              double sparsity = 0.0, double bias_value = 0.0) 
    {
        for (size_t i = 0; i < layers.size(); ++i) {
            Dense* dense_layer = dynamic_cast<Dense*>(layers[i]);
            if (dense_layer) {
                // Default to Xavier
                InitMethod method = InitMethod::XAVIER_NORMAL;

                // Check the next layer if it's an ActivationLayer
                if (i + 1 < layers.size()) {
                    ActivationLayer* act_layer = dynamic_cast<ActivationLayer*>(layers[i + 1]);
                    if (act_layer) {
                        ActivationType act_type = act_layer->getActivationType();

                        switch (act_type) {
                            case ActivationType::RELU:
                            case ActivationType::LEAKY_RELU:
                                method = InitMethod::HE_NORMAL;
                                break;

                            case ActivationType::SIGMOID:
                            case ActivationType::TANH:
                                method = InitMethod::XAVIER_NORMAL;
                                break;

                            case ActivationType::SELU:
                                method = InitMethod::LECUN_NORMAL;
                                break;

                            default:
                                method = InitMethod::XAVIER_NORMAL; // safe fallback
                        }
                    }
                }

                dense_layer->initializeWeights(method, seed, a, b, sparsity, bias_value);
                dense_layer->initializeBiases(InitMethod::BIAS, seed, a, b, sparsity, bias_value); // Biases set separately (default zero)
            }
        }
    }

    /**
     * @brief Performs a forward pass through all layers.
     * 
     * @param input Input vector to the model.
     * @return Output vector after passing through all layers.
     */
    vector<double> forward(const vector<double>& input) {
        vector<double> output = input;
        for (Layer* layer : layers) {
            output = layer->forward(output);
        }
        return output;
    }

    /**
     * @brief Performs a backward pass through all layers.
     * 
     * @param grad_output Gradient of the loss with respect to the model output.
     * @param learning_rate Learning rate for parameter updates.
     * @return Gradient of the loss with respect to the model input.
     */
    vector<double> backward(const vector<double>& grad_output, double learning_rate = 0.01) {
        vector<double> grad = grad_output;
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            grad = (*it)->backward(grad, learning_rate);
        }
        return grad;
    }

/**
 * @brief Performs a training step over a batch of samples.
 * 
 * @param X Batch of input samples (2D vector).
 * @param Y_true Batch of true labels (2D vector).
 * @param loss_func Pointer to the loss function.
 * @param loss_derivative Pointer to the loss derivative function.
 * @param optimizer Pointer to the optimizer.
 * @return Average loss over the batch.
 */
double train(const vector<vector<double>>& X,
             const vector<vector<double>>& Y_true,
             double (*loss_func)(const vector<vector<double>>&, const vector<vector<double>>&),
             vector<vector<double>> (*loss_derivative)(const vector<vector<double>>&, const vector<vector<double>>&),
             Optimizer* optimizer) 
{
    size_t batch_size = X.size();
    vector<vector<double>> outputs;
    for (const auto& sample : X) {
        outputs.push_back(forward(sample));
    }

    double loss = loss_func(Y_true, outputs);
    auto grad_output = loss_derivative(Y_true, outputs);

    // Backward pass (per sample)
    for (size_t i = 0; i < batch_size; ++i) {
        backward(grad_output[i]); // assume backward accepts 1 sample
    }

    // Update weights (global)
    vector<Dense*> dense_layers;
    for (Layer* layer : layers) {
        Dense* dense = dynamic_cast<Dense*>(layer);
        if (dense) dense_layers.push_back(dense);
    }
    optimizer->step(dense_layers); // Passing vector<Dense*>


    return loss;
}


    /**
     * @brief Prints a summary of the model architecture.
     */
    void summary() const {
        cout << "Sequential Model Summary:\n";
        for (const Layer* layer : layers) {
            layer->summary();
        }
    }

    /**
     * @brief Destructor to release dynamically allocated layers.
     */
    ~Sequential() {
        for (Layer* layer : layers) {
            delete layer;
        }
    }
};

#endif // SEQUENTIAL_H
