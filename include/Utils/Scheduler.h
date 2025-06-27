#include <functional>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace Schedulers {
    inline std::function<double(double, size_t)> cosine(double total_steps) {
        return [total_steps](double init, size_t step) {
            return init * 0.5 * (1 + cos(M_PI * step / total_steps));
        };
    }
    
    inline std::function<double(double, size_t)> step(size_t step_size, double gamma) {
        return [step_size, gamma](double init, size_t step) {
            return init * pow(gamma, floor(step/step_size));
        };
    }
}
