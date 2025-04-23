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

int count_nodes(const GiNaC::ex& expr) {
    // Count this node
    int count = 1;
    
    // Add count of all child nodes
    for (size_t i = 0; i < expr.nops(); ++i) {
        count += count_nodes(expr.op(i));
    }
    
    return count;
}

int get_depth(const GiNaC::ex& expr) {
    if (expr.nops() == 0)
        return 0;
    
    int max_depth = 0;
    for (size_t i = 0; i < expr.nops(); ++i) {
        int child_depth = get_depth(expr.op(i));
        if (child_depth > max_depth)
            max_depth = child_depth;
    }
    
    return max_depth + 1;
}

void memory_footprint(const ex& expr, set<void*>& visited, size_t& size) {
    // Get pointer to actual object
    const void* ptr = &expr;
    
    // If we've seen this pointer before, don't count it again
    if (visited.find((void*)ptr) != visited.end())
        return;
    
    // Mark as visited
    visited.insert((void*)ptr);
    
    // Add approximate size for this node (very rough estimate)
    size += sizeof(ex) + 16;  // Base size plus some overhead
    
    // Recurse to children
    for (size_t i = 0; i < expr.nops(); ++i) {
        memory_footprint(expr.op(i), visited, size);
    }
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
        outfile << "diff: " << param.rows() << "x" << param.cols() << endl;
        
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

        outfile << "sum_squared: " << sum_squared << endl;
        outfile << "Number of nodes in sum_squared: " << count_nodes(sum_squared) << endl;
        outfile << "Depth of sum_squared: " << get_depth(sum_squared) << endl;
        set<void*> visited;
        size_t memory = 0;
        memory_footprint(sum_squared, visited, memory);
        outfile << "Approximate memory (bytes): " << memory << endl;
        outfile << endl;

        outfile << "preds_diff: " << preds_diff << endl;
        outfile << "Number of nodes in preds_diff: " << count_nodes(preds_diff) << endl;
        outfile << "Depth of preds_diff: " << get_depth(preds_diff) << endl;
        set<void*> visited_1;
        size_t memory_1 = 0;
        memory_footprint(preds_diff, visited_1, memory_1);
        outfile << "Approximate memory (bytes): " << memory_1 << endl;
        outfile << endl;
        
        // Start timing for linearization
        auto start_linearize = high_resolution_clock::now();
        // Expand the result
        ex linearized_result = preds_diff.expand().normal();
        // End timing for linearization
        auto end_linearize = high_resolution_clock::now();
        auto duration_linearize = duration_cast<milliseconds>(end_linearize - start_linearize);
        
        outfile << "Raw Result: " << preds_diff << endl;
        outfile << "Expanded Result: " << linearized_result << endl;
        outfile << "Time to compute preds_diff: " << duration_compute.count() << " ms" << endl;
        outfile << "Time to expand result: " << duration_linearize.count() << " ms" << endl;
        outfile << "Number of nodes in linearized_result: " << count_nodes(linearized_result) << endl;
        outfile << "Depth of preds_diff: " << get_depth(linearized_result) << endl;
        set<void*> visited_2;
        size_t memory_2 = 0;
        memory_footprint(linearized_result, visited_2, memory_2);
        outfile << "Approximate memory (bytes): " << memory_2 << endl;

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