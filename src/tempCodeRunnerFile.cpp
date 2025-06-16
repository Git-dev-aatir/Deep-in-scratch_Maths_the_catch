#include "../include/dataset_utils.h"
#include <iostream>
#include <string>
using namespace std;

int main() {

    Dataset housing = loadDataset("../Datasets/UCI-Data-Analysis/Boston Housing Dataset/Boston Housing/housing.data", 
                                  ' ');
    
    head(housing, 8);



    return 0;
}
