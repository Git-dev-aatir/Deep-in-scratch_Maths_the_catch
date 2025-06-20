#ifndef LAYERS_H
#define LAYERS_H

#include <vector>
#include <string>
#include "../General/Initialization.h" // For parameter initialization

using std::vector;
using std::string;

/**
 * @brief Abstract base class for all layers.
 */
class Layer {
public:
    virtual vector<double> forward(const vector<double>& input) = 0;
    virtual vector<double> backward(const vector<double>& grad_output, double learning_rate) = 0;
    virtual void summary() const = 0;
    virtual ~Layer() {}
};


#endif // LAYERS_H
