    #include <iostream>
    #include <string>
    #include <vector>
    #include "Data/Dataset.h"
    #include "Data/Preprocessing.h"
    #include "Models/Sequential.h"
    #include "Optimizers/SGD.h"
    #include "Metrics/Losses.h"
    #include "Utils/Scheduler.h"
    #include "Utils/Activations.h"

    #define DATA_PATH "Datasets/test_linear/"

    using namespace std; 

    // testing purely linear data with 3 features and no noise
    int main () {

        Dataset X_train;
        X_train.loadCSV(string(DATA_PATH) + "X_train.csv", ',');
        Dataset y_train;
        y_train.loadCSV(string(DATA_PATH) + "y_train.csv", ',');
        Dataset X_test;
        X_test.loadCSV(string(DATA_PATH) + "X_test.csv", ',');
        Dataset y_test;
        y_test.loadCSV(string(DATA_PATH) + "y_test.csv", ',');

        X_train.printShape();
        y_train.printShape();
        X_test.printShape();
        y_test.printShape();

        Preprocessing::standardize(X_train);
        Preprocessing::standardize(X_test);


        // unsigned int hidden_units = 1;
        Sequential model(
            std::make_unique<DenseLayer>(X_train.cols(), 1)
        );

        model.initializeParameters(21);


        size_t epochs = 50;
        
        // Create optimizer
        const double base_lr = 0.1;
        const size_t base_batch_size = 1;
        const size_t batches_per_epoch = ceil(static_cast<double>(X_train.rows()) / base_batch_size);
        const size_t total_steps = epochs * batches_per_epoch;
        auto scheduler = Schedulers::step(10, 0.9);
        SGD optimizer(
            base_lr,      // learning_rate
            0.9,          // momentum
            base_batch_size,
            scheduler   // LR scheduler
        );
        // optimizer.setGradientClip(0.1);
        
        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            
            double epoch_loss = model.train(
                X_train, 
                y_train,
                optimizer,
                // [](const vector<double>& y_true, const vector<double>& y_pred) {
                //     return Losses::cross_entropy_loss(y_true, y_pred, true);
                // },
                // [](const vector<double>& y_true, const vector<double>& y_pred) {
                //     return Losses::cross_entropy_derivative(y_true, y_pred, true);
                [](const vector<vector<double>>& y_true, const vector<vector<double>>& y_pred) {
                    return Losses::mse_loss_batch(y_true, y_pred);
                },
                [](const vector<vector<double>>& y_true, const vector<vector<double>>& y_pred) {
                    return Losses::mse_derivative_batch(y_true, y_pred);
                },
                21
            );
            
            // Test evaluation
            size_t correct = 0;
            double test_loss = 0;
            for (size_t i = 0; i < X_test.rows(); ++i) {
                vector<double> output = model.forward(X_test[i]);
                test_loss += Losses::mse_loss(y_test[i], output);
            }
            
            // Print every 10 epochs
            if (epoch % 10 == 0 || epoch == epochs-1 || epoch < 10) {
                std::cout << "Epoch " << (epoch + 1) << "/" << epochs
                        << " | LR: " << optimizer.getLearningRate()
                        << " | Loss: " << epoch_loss 
                        << " | Test_loss: " << test_loss / X_test.rows() << std::endl;
            }
        }
        for (size_t i=0; i<X_test.rows(); ++i) {
            std::cout << "Actual : " << y_test[i][0]; 
            std::cout << "\t\t Predicted : " << model.forward(X_test[i])[0] << std::endl;
        }


        return 0;
    }