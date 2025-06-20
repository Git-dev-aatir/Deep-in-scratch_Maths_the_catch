#include <iostream>
#include "../include/Preprocessing/dataset_utils.h"
#include "../include/Layers/Dense.h"
#include "../include/Losses/Losses.h"

#define DATA_PATH "../Datasets/test_linear/"

using namespace std; 

// testing purely linear data with 3 features and no noise
int main () {
    Dataset<double> X_train = loadDataset<double>(string(DATA_PATH) + "X_train.csv", ',');
    Dataset<double> y_train = loadDataset<double>(string(DATA_PATH) + "y_train.csv", ',');
    Dataset<double> X_test = loadDataset<double>(string(DATA_PATH) + "X_test.csv", ',');
    Dataset<double> y_test = loadDataset<double>(string(DATA_PATH) + "y_test.csv", ',');
    
    // printDimensions(X_train);
    // head(X_train, X_train.size());

    // printDimensions(y_train);
    // head(y_train, y_train.size());

    // printDimensions(X_test);
    // head(X_test);

    // printDimensions(y_test);
    // head(y_test);
    // return 0;

    Dense model(X_train[0].size(), 1, false);
    model.initializeWeights(InitMethod::RANDOM_NORMAL, 21, 0, 1);
    model.initializeBiases(InitMethod::RANDOM_NORMAL, 21, 0, 1);
    double lr = 0.01;
    size_t epochs = 100;

    vector <vector<double>> output;
    for (size_t epoch = 0; epoch < epochs; ++epoch) {

        double total_loss = 0.0;

        // Looping through all samples (SGD) 
        for (size_t i = 0; i < X_train.size(); ++i) {

            // forward pass 
            output = unsqueeze<double>(model.forward(X_train[i]), 1);
            vector<vector<double>> true_val = unsqueeze<double>(y_train[i]);
            // printDimensions(output);
            // printDimensions(true_val);
            double loss = Losses::mse_loss(output, true_val);
            total_loss += loss;

            // back pass
            vector<double> grad_output = squeeze<double>((Losses::mse_derivative(output, true_val)));
            model.backward(grad_output, lr);
        }

        //print loss every 10 epochs 
        if ((epoch + 1) % 10 == 0) {
            cout << "Epoch " << epoch + 1 << ", Loss: " << total_loss / X_train.size() << endl;
        }
    }

    // 5. Test the model
    cout << "\nTesting the trained model:" << endl;
    for (size_t i=0; i<X_test.size(); ++i) {
        const vector<double> x = X_test[i];
        vector<double> pred = model.forward(x);
        cout << "Real value: " << y_test[i][0] << " -> Prediction: " << pred[0] << endl;
    }

    // 6. Print learned weights and biases
    cout << "\nLearned Weight: \n";
    head(model.getWeights());
    cout << "Learned Bias: " << model.getBiases()[0] << endl;
    


    return 0;
}