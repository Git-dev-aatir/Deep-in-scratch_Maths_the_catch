#include "../include/dataset_utils.h"
#include "../include/preprocessing.h"
#include <iostream>
#include <string>
#include <typeinfo>
using namespace std;

// #define DATASET_PATH "../Datasets/UCI-Data-Analysis/Boston Housing Dataset/Boston Housing/housing.data"
#define DATASET_PATH_BIN "../Datasets/Boston_housing_dataset/"

int main() {

    // Dataset<double> housing = loadDataset<double>(DATASET_PATH, ' ', true);
    // saveDatasetToBinary(housing, DATASET_PATH_BIN);
    
    Dataset<double> housing = loadDatasetFromBinary<double>(string(DATASET_PATH_BIN)+"data.bin");

    // Attributes : 
    // 1. CRIM      per capita crime rate by town
    // 2. ZN        proportion of residential land zoned for lots over 
    //              25,000 sq.ft.
    // 3. INDUS     proportion of non-retail business acres per town
    // 4. CHAS      Charles River dummy variable (= 1 if tract bounds 
    //              river; 0 otherwise)
    // 5. NOX       nitric oxides concentration (parts per 10 million)
    // 6. RM        average number of rooms per dwelling
    // 7. AGE       proportion of owner-occupied units built prior to 1940
    // 8. DIS       weighted distances to five Boston employment centres
    // 9. RAD       index of accessibility to radial highways
    // 10. TAX      full-value property-tax rate per $10,000
    // 11. PTRATIO  pupil-teacher ratio by town
    // 12. B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks 
    //              by town
    // 13. LSTAT    % lower status of the population
    // 14. MEDV     Median value of owner-occupied homes in $1000's



    // splitting into training and testing set 
    Dataset<double> train_set, test_set;
    auto result = trainTestSplit(housing, 0.2, true);
    train_set = result.first;
    test_set = result.second;

    // splitting into features and labels 
    Dataset<double> X, y;
    auto train_split = splitFeaturesAndLabels(train_set);
    auto X_train = train_split.first;
    auto y_train = train_split.second;
    auto test_split = splitFeaturesAndLabels(test_set);
    auto X_test = test_split.first;
    auto y_test = test_split.second;
    printDimensions<double>(housing);
    cout << endl;
    printDimensions<double>(X_train);
    head(X_train); 
    cout << endl;
    printDimensions<double>(y_train);
    head(y_train); 
    cout << endl;
    printDimensions<double>(X_test);
    head(X_test); 
    cout << endl;
    printDimensions<double>(y_test);
    head(y_test); 
    cout << endl; 
    // auto result = splitFeaturesAndLabels(housing);

    describeDataset(housing);
    
    // Linear Regression     
    

    return 0;
}
