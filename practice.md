For MSE: $\frac{\partial}{\partial w} \frac{1}{2n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$

The derivative should be: $\frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)$

Per sample: $\frac{(\hat{y}_i - y_i)}{n}$ where $n$ = samples in that batch

Not: $\frac{(\hat{y}_i - y_i)}{\text{total elements}}$