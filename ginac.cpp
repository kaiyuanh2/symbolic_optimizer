#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <ginac/ginac.h>

using namespace std;
using namespace GiNaC;
using namespace std::chrono;

matrix load_matrix_from_csv(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        exit(1);
    }
    
    vector<vector<ex>> data;
    string line;
    
    while (getline(file, line)) {
        vector<ex> row;
        istringstream iss(line);
        string value;
        
        while (getline(iss, value, ',')) {
            // Try to convert to numeric first
            try {
                double num_value = stod(value);
                row.push_back(numeric(num_value));
            } catch (...) {
                // If not a number, parse as symbolic expression
                parser reader;
                ex expr = reader(value);
                row.push_back(expr);
            }
        }
        
        data.push_back(row);
    }
    
    int rows = data.size();
    int cols = rows > 0 ? data[0].size() : 0;
    
    matrix result(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result(i, j) = data[i][j];
        }
    }
    
    return result;
}

int main() {
    // Open a file for the output
    ofstream outfile("ginac_output.txt");
    if (!outfile.is_open()) {
        cerr << "Failed to open output file" << endl;
        return 1;
    }
    
    // Load validation set
    matrix X_test = load_matrix_from_csv("X_val_expanded.csv");
    outfile << "X_test: " << X_test.rows() << "x" << X_test.cols() << endl;
    
    matrix y_test = load_matrix_from_csv("y_val.csv");
    outfile << "y_test: " << y_test.rows() << "x" << y_test.cols() << endl;
    
    // Load parameters
    ifstream param_file("param_ginac_ins_lr.txt");
    vector<ex> param_vec;
    string line;
    
    if (!param_file.is_open()) {
        outfile << "Error opening param file" << endl;
        return 1;
    }
    
    while (getline(param_file, line)) {
        parser reader;
        ex expression = reader(line);
        param_vec.push_back(expression);
    }
    
    matrix param(param_vec.size(), 1);
    for (size_t i = 0; i < param_vec.size(); ++i) {
        param(i, 0) = param_vec[i];
    }
    
    outfile << "param: " << param.rows() << "x" << param.cols() << endl;
    
    try {
        // Start timing for preds_diff computation
        auto start_compute = high_resolution_clock::now();

        int n = X_test.rows();
        
        // Calculate test_preds = X_test * param
        matrix test_preds(y_test.rows(), 1);
        for (int i = 0; i < X_test.rows(); i++) {
            ex sum = 0;
            for (int j = 0; j < X_test.cols(); j++) {
                sum += X_test(i, j) * param(j, 0);
            }
            test_preds(i, 0) = sum;
        }
        
        // Calculate diff = test_preds - y_test
        matrix diff(y_test.rows(), 1);
        for (int i = 0; i < y_test.rows(); i++) {
            diff(i, 0) = test_preds(i, 0) - y_test(i, 0);
        }
        
        // Calculate sum of squared differences
        ex sum_squared = 0;
        for (int i = 0; i < diff.rows(); i++) {
            sum_squared += diff(i, 0) * diff(i, 0);
        }
        
        // preds_diff = sum_squared / n
        ex preds_diff = sum_squared / numeric(n);
        // End timing for preds_diff computation
        auto end_compute = high_resolution_clock::now();
        auto duration_compute = duration_cast<milliseconds>(end_compute - start_compute);
        
        // Start timing for linearization
        auto start_linearize = high_resolution_clock::now();
        // Expand the result
        ex linearized_result = preds_diff.expand();
        // End timing for linearization
        auto end_linearize = high_resolution_clock::now();
        auto duration_linearize = duration_cast<milliseconds>(end_linearize - start_linearize);
        
        outfile << "Raw Result: " << preds_diff << endl;
        outfile << "Expanded Result: " << linearized_result << endl;
        outfile << "Time to compute preds_diff: " << duration_compute.count() << " ms" << endl;
        outfile << "Time to expand result: " << duration_linearize.count() << " ms" << endl;

        ofstream result_file("ginac_result.txt");
        if (result_file.is_open()) {
            result_file << linearized_result << endl;
            result_file.close();
        }
        
        cout << "Time to compute preds_diff: " << duration_compute.count() << " ms" << endl;
        cout << "Time to expand result: " << duration_linearize.count() << " ms" << endl;
        cout << "Full output saved to ginac_output.txt" << endl;
        cout << "Expanded result saved to ginac_result.txt" << endl;
        
    } catch (exception& e) {
        outfile << "Exception: " << e.what() << endl;
        cout << "Error occurred. See ginac_output.txt for details." << endl;
    }
    
    outfile.close();
    return 0;
}