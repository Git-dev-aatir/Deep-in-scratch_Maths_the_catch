#pragma once 

#include <vector>

/**
 * @namespace Losses
 * @brief Contains loss function declarations for MSE, MAE, BCE, Cross Entropy, and Hinge losses.
 */
namespace Losses {

    // ----------------- Mean Squared Error (MSE) -----------------

    /**
     * @brief Computes the Mean Squared Error (MSE) loss for a single sample.
     * @param y_true Ground truth vector.
     * @param y_pred Predicted vector.
     * @return Computed MSE loss.
     */
    double mse_loss(const std::vector<double>& y_true, const std::vector<double>& y_pred);

    /**
     * @brief Computes the derivative of MSE loss for a single sample.
     * @param y_true Ground truth vector.
     * @param y_pred Predicted vector.
     * @return Gradient vector.
     */
    std::vector<double> mse_derivative(const std::vector<double>& y_true, const std::vector<double>& y_pred);

    /**
     * @brief Computes the Mean Squared Error (MSE) loss for a batch of samples.
     * @param y_true Ground truth batch.
     * @param y_pred Predicted batch.
     * @return Computed batch MSE loss.
     */
    double mse_loss_batch(const std::vector<std::vector<double>>& y_true, const std::vector<std::vector<double>>& y_pred);

    /**
     * @brief Computes the derivative of MSE loss for a batch of samples.
     * @param y_true Ground truth batch.
     * @param y_pred Predicted batch.
     * @return Gradient batch.
     */
    std::vector<std::vector<double>> mse_derivative_batch(const std::vector<std::vector<double>>& y_true,
                                                          const std::vector<std::vector<double>>& y_pred);

    // ----------------- Mean Absolute Error (MAE) -----------------

    /**
     * @brief Computes the Mean Absolute Error (MAE) loss for a single sample.
     * @param y_true Ground truth vector.
     * @param y_pred Predicted vector.
     * @return Computed MAE loss.
     */
    double mae_loss(const std::vector<double>& y_true, const std::vector<double>& y_pred);

    /**
     * @brief Computes the derivative of MAE loss for a single sample.
     * @param y_true Ground truth vector.
     * @param y_pred Predicted vector.
     * @return Gradient vector.
     */
    std::vector<double> mae_derivative(const std::vector<double>& y_true, const std::vector<double>& y_pred);

    /**
     * @brief Computes the Mean Absolute Error (MAE) loss for a batch of samples.
     * @param y_true Ground truth batch.
     * @param y_pred Predicted batch.
     * @return Computed batch MAE loss.
     */
    double mae_loss_batch(const std::vector<std::vector<double>>& y_true, const std::vector<std::vector<double>>& y_pred);

    /**
     * @brief Computes the derivative of MAE loss for a batch of samples.
     * @param y_true Ground truth batch.
     * @param y_pred Predicted batch.
     * @return Gradient batch.
     */
    std::vector<std::vector<double>> mae_derivative_batch(const std::vector<std::vector<double>>& y_true,
                                                          const std::vector<std::vector<double>>& y_pred);

    // ----------------- Binary Cross Entropy (BCE) -----------------

    /**
     * @brief Computes the Binary Cross Entropy (BCE) loss for a single sample.
     * 
     * (Suggestion : try to use with_logits=true and output layer as Dense for best results.)
     * 
     * @param y_true Ground truth vector.
     * @param y_pred Predicted vector (probabilities or logits).
     * @param from_logits Set true if predictions are logits and need sigmoid activation.
     * @return Computed BCE loss.
     */
    double bce_loss(const std::vector<double>& y_true, const std::vector<double>& y_pred, bool from_logits = false);

    /**
     * @brief Computes the derivative of BCE loss for a single sample.
     * 
     * (Suggestion : try to use with_logits=true and output layer as Dense for best results.)
     * 
     * @param y_true Ground truth vector.
     * @param y_pred Predicted vector (probabilities or logits).
     * @param from_logits Set true if predictions are logits.
     * @return Gradient vector.
     */
    std::vector<double> bce_derivative(const std::vector<double>& y_true, const std::vector<double>& y_pred, bool from_logits = false);

    /**
     * @brief Computes the Binary Cross Entropy (BCE) loss for a batch of samples.
     * 
     * (Suggestion : try to use with_logits=true and output layer as Dense for best results.)
     * 
     * @param y_true Ground truth batch.
     * @param y_pred Predicted batch (probabilities or logits).
     * @param from_logits Set true if predictions are logits.
     * @return Computed batch BCE loss.
     */
    double bce_loss_batch(const std::vector<std::vector<double>>& y_true, const std::vector<std::vector<double>>& y_pred, bool from_logits = false);

    /**
     * @brief Computes the derivative of BCE loss for a batch of samples.
     * 
     * (Suggestion : try to use with_logits=true and output layer as Dense for best results.)
     * 
     * @param y_true Ground truth batch.
     * @param y_pred Predicted batch (probabilities or logits).
     * @param from_logits Set true if predictions are logits.
     * @return Gradient batch.
     */
    std::vector<std::vector<double>> bce_derivative_batch(const std::vector<std::vector<double>>& y_true,
                                                          const std::vector<std::vector<double>>& y_pred,
                                                          bool from_logits = false);

    // ----------------- Cross Entropy -----------------

    /**
     * @brief Computes the Cross Entropy loss for a single sample.
     * 
     * (Suggestion : try to use with_logits=true and output layer as Dense for best results.)
     * 
     * @param y_true Ground truth vector (expected to be one-hot or probability distribution).
     * @param y_pred Predicted vector (probabilities or logits).
     * @param from_logits Set true if predictions are logits and need softmax activation.
     * @return Computed Cross Entropy loss.
     */
    double cross_entropy_loss(const std::vector<double>& y_true, const std::vector<double>& y_pred, bool from_logits = false);

    /**
     * @brief Computes the derivative of Cross Entropy loss for a single sample.
     * 
     * (Suggestion : try to use with_logits=true and output layer as Dense for best results.)
     * 
     * @param y_true Ground truth vector (expected to be one-hot or probability distribution).
     * @param y_pred Predicted vector (probabilities or logits).
     * @param from_logits Set true if predictions are logits.
     * @return Gradient vector.
     */
    std::vector<double> cross_entropy_derivative(const std::vector<double>& y_true, const std::vector<double>& y_pred, bool from_logits = false);

    /**
     * @brief Computes the Cross Entropy loss for a batch of samples.
     * 
     * (Suggestion : try to use with_logits=true and output layer as Dense for best results.)
     * 
     * @param y_true Ground truth batch (expected to be one-hot or probability distribution).
     * @param y_pred Predicted batch (probabilities or logits).
     * @param from_logits Set true if predictions are logits.
     * @return Computed batch Cross Entropy loss.
     */
    double cross_entropy_loss_batch(const std::vector<std::vector<double>>& y_true, const std::vector<std::vector<double>>& y_pred, bool from_logits = false);

    /**
     * @brief Computes the derivative of Cross Entropy loss for a batch of samples.
     * 
     * (Suggestion : try to use with_logits=true and output layer as Dense for best results.)
     * 
     * @param y_true Ground truth batch (expected to be one-hot or probability distribution).
     * @param y_pred Predicted batch (probabilities or logits).
     * @param from_logits Set true if predictions are logits.
     * @return Gradient batch.
     */
    std::vector<std::vector<double>> cross_entropy_derivative_batch(const std::vector<std::vector<double>>& y_true,
                                                                    const std::vector<std::vector<double>>& y_pred,
                                                                    bool from_logits = false);

    // ----------------- Hinge Loss -----------------

    /**
     * @brief Computes the Hinge loss for a single sample.
     * @param y_true Ground truth vector.
     * @param y_pred Predicted vector.
     * @return Computed Hinge loss.
     */
    double hinge_loss(const std::vector<double>& y_true, const std::vector<double>& y_pred);

    /**
     * @brief Computes the derivative of Hinge loss for a single sample.
     * @param y_true Ground truth vector.
     * @param y_pred Predicted vector.
     * @return Gradient vector.
     */
    std::vector<double> hinge_derivative(const std::vector<double>& y_true, const std::vector<double>& y_pred);

    /**
     * @brief Computes the Hinge loss for a batch of samples.
     * @param y_true Ground truth batch.
     * @param y_pred Predicted batch.
     * @return Computed batch Hinge loss.
     */
    double hinge_loss_batch(const std::vector<std::vector<double>>& y_true, const std::vector<std::vector<double>>& y_pred);

    /**
     * @brief Computes the derivative of Hinge loss for a batch of samples.
     * @param y_true Ground truth batch.
     * @param y_pred Predicted batch.
     * @return Gradient batch.
     */
    std::vector<std::vector<double>> hinge_derivative_batch(const std::vector<std::vector<double>>& y_true,
                                                            const std::vector<std::vector<double>>& y_pred);

} // namespace Losses
