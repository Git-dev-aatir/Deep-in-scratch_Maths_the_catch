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

    inline std::function<double(double, size_t)> cosine_warmup (
        double min_lr, size_t total_steps, size_t warmup_steps 
    ) {
        if (total_steps <= warmup_steps) {
            throw std::invalid_argument("total_steps must be greater than warmup_steps");
        }
        return [min_lr, warmup_steps, total_steps](double init_lr, size_t step) {
            if (step < warmup_steps) {
                return min_lr + (init_lr - min_lr) * step / warmup_steps;
            } else {
                double progress = std::min(1.0, 
                    double(step - warmup_steps) / (total_steps - warmup_steps));
                double cosine = 0.5 * (1 + cos(M_PI * progress));
                return min_lr + (init_lr - min_lr) * cosine;
            }
        };
    }

    inline std::function<double(double, size_t)> cyclical_lr (
        double min_lr, double max_lr, size_t step_size
    ) {
        return [min_lr, max_lr, step_size](double, size_t step) {
            size_t cycle = floor(1 + step / (2.0 * step_size));
            double denom = std::max(1e-10, double(step_size));
            double x = abs(step / denom - 2 * cycle + 1);
            return min_lr + (max_lr - min_lr) * std::max(0.0, (1 - x));
        };
    }

}
