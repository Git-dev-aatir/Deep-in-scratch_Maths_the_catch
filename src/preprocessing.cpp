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
    // 0. CRIM      per capita crime rate by town
    // 1. ZN        proportion of residential land zoned for lots over 
    //              25,000 sq.ft.
    // 2. INDUS     proportion of non-retail business acres per town
    // 3. CHAS      Charles River dummy variable (= 1 if tract bounds 
    //              river; 0 otherwise)
    // 4. NOX       nitric oxides concentration (parts per 10 million)
    // 5. RM        average number of rooms per dwelling
    // 6. AGE       proportion of owner-occupied units built prior to 1940
    // 7. DIS       weighted distances to five Boston employment centres
    // 8. RAD       index of accessibility to radial highways
    // 9. TAX       full-value property-tax rate per $10,000
    // 10. PTRATIO  pupil-teacher ratio by town
    // 11. B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks 
    //              by town
    // 12. LSTAT    % lower status of the population
    // 13. MEDV     Median value of owner-occupied homes in $1000's



// --------------------------------------------------------------------------------------
// PART - 1
    // describeDataset(housing);
    // Dataset<double> corr_mat = computeCorrelationMatrix(housing);
    // cout << endl;
    // printHighlyCorrelatedFeatures(corr_mat, 0.7);

    // vector<double> corr = computeCorrelationWithTarget(housing, housing[0].size()-1);
    // cout << endl;
    // printSortedCorrelations(corr);

    // Result Output :
    // ---------------------------------------------------------------------------------------
    // Column    Mean           Min       25%            Median         75%            Max       
    // ---------------------------------------------------------------------------------------
    // 0         3.61           0.01      0.08           0.26           3.68           88.98     
    // 1         11.36          0.00      0.00           0.00           12.50          100.00    
    // 2         11.14          0.46      5.19           9.69           18.10          27.74     
    // 3         0.07           0.00      0.00           0.00           0.00           1.00      
    // 4         0.55           0.39      0.45           0.54           0.62           0.87      
    // 5         6.28           3.56      5.88           6.21           6.62           8.78      
    // 6         68.57          2.90      45.00          77.50          94.10          100.00    
    // 7         3.80           1.13      2.10           3.21           5.21           12.13     
    // 8         9.55           1.00      4.00           5.00           24.00          24.00
    // 9         408.24         187.00    279.00         330.00         666.00         711.00
    // 10        18.46          12.60     17.40          19.05          20.20          22.00
    // 11        356.67         0.32      375.33         391.44         396.23         396.90
    // 12        12.65          1.73      6.93           11.36          16.96          37.97
    // 13        22.53          5.00      17.00          21.20          25.00          50.00
    // ---------------------------------------------------------------------------------------

    // Highly Correlated Feature Pairs (|correlation| >= 0.70):
    // --------------------------------------------
    // Feature1        Feature2        Correlation
    // --------------------------------------------
    // 8               9               0.9102
    // 4               7               -0.7692
    // 2               4               0.7637
    // 6               7               -0.7479
    // 12              13              -0.7377
    // 4               6               0.7315
    // 2               9               0.7208
    // 2               7               -0.7080
    // --------------------------------------------

    // Correlations sorted by descending absolute value:
    // ------------------------
    // Column  |   Correlation
    // ------------------------
    // 13      |   1.0000
    // 12      |   -0.7377
    // 5       |   0.6954
    // 10      |   -0.5078
    // 2       |   -0.4837
    // 9       |   -0.4685
    // 4       |   -0.4273
    // 0       |   -0.3883
    // 8       |   -0.3816
    // 6       |   -0.3770
    // 1       |   0.3604
    // 11      |   0.3335
    // 7       |   0.2499
    // 3       |   0.1753
    // ------------------------

    // Conclusion  :
    // 1. remove Col8 - Col8 and Col9 are very correlated (corr = 0.91), and 
    //    corr(Col8, Label) < corr(Col9, Label)  and   Col9 has better spread of data as per describe()
    // 2. remove Col4 - corr(Col4, Col7)=-0.7692, corr(Col2, Col7)=0.7637
    // 3. can remove Col6 - corr(Col6, Col7)=-0.7479

// -----------------------------------------------------------------------------------------------

// PART - 2 

    // // removing highly correlated columns 
    // removeColumns(housing, {8, 4});

    // // detecting outliers and removing
    // removeOutliers(housing, OutlierMethod::Z_SCORE, 3);

// ----------------------------------------------------------------------------------------

// Part 3 - standardizing data 

    // standardize(housing);

// ----------------------------------------------------------------------------------------

// Part 4 - splitting the dataset 

    // // Train-test-split
    // auto train_test_split_result = trainTestSplit(housing, 0.2, true);
    // Dataset<double> train_set = train_test_split_result.first;
    // Dataset<double> test_set = train_test_split_result.second;

    // // Feature-label-split
    // auto train_split_result = splitFeaturesAndLabels(train_set);
    // Dataset<double> X_train = train_split_result.first;
    // Dataset<double> y_train = train_split_result.second;

    // auto test_split_result = splitFeaturesAndLabels(test_set);
    // Dataset<double> X_test = test_split_result.first;
    // Dataset<double> y_test = test_split_result.second;

// ----------------------------------------------------------------------------------------

    // cout << "X_train" << endl;
    // printDimensions(X_train);
    // head(X_train);

    // cout << endl << "y_train" << endl;
    // printDimensions(y_train);
    // head(y_train);
    
    // cout << "X_test" << endl;
    // printDimensions(X_test);
    // head(X_test);

    // cout << endl << "y_test" << endl;
    // printDimensions(y_test);
    // head(y_test);

// saving dataset in preprocessed_data.bin

    // saveDatasetToBinary(housing, string(DATASET_PATH_BIN) + "X_train.bin");
    // saveDatasetToBinary(housing, string(DATASET_PATH_BIN) + "y_train.bin");
    // saveDatasetToBinary(housing, string(DATASET_PATH_BIN) + "X_test.bin");
    // saveDatasetToBinary(housing, string(DATASET_PATH_BIN) + "y_test.bin");

    return 0;
}
