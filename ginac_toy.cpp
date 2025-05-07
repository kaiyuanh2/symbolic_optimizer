#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <string>
#include <ginac/ginac.h>

#include <cmath>
#include <regex>
#include <climits>
#include <map>
#include <set>

#include <thread>
#include <future>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <memory>
#include <stdexcept>
#include <atomic>
#include <execution>
#include <cstddef>
#include <numeric>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>
#include <cassert>

using namespace std;
using namespace GiNaC;
using namespace std::chrono;
using namespace Eigen;

symbol create_symbol(const string& base, int i) {
    return symbol(base + to_string(i));
}

MatrixXd ginacToEigen(const matrix& m) {
    unsigned rows = m.rows();
    unsigned cols = m.cols();
    MatrixXd eigen_mat(rows, cols);
    
    for (unsigned i = 0; i < rows; ++i) {
        for (unsigned j = 0; j < cols; ++j) {
            // Extract numerical value from GiNaC expression
            ex element = m(i, j);
            eigen_mat(i, j) = ex_to<numeric>(element).to_double();
        }
    }
    
    return eigen_mat;
}

matrix eigenToGinac(const MatrixXd& eigen_mat) {
    unsigned rows = eigen_mat.rows();
    unsigned cols = eigen_mat.cols();
    matrix ginac_mat(rows, cols);
    
    for (unsigned i = 0; i < rows; ++i) {
        for (unsigned j = 0; j < cols; ++j) {
            ginac_mat(i, j) = numeric(eigen_mat(i, j));
        }
    }
    
    return ginac_mat;
}

matrix get_row(const matrix& mat, int row_index) {
    int num_cols = mat.cols();
    matrix row(1, num_cols);
    
    for (int j = 0; j < num_cols; j++) {
        row(0, j) = mat(row_index, j);
    }
    
    return row;
}

string ex_to_string(const ex& expr, bool python_style = true) {
    ostringstream oss;
    
    if (python_style) {
        // Convert to Python-style syntax
        string temp_str;
        ostringstream temp_oss;
        temp_oss << expr;
        temp_str = temp_oss.str();
        
        // Replace GiNaC syntax with Python syntax
        // This is a simple example - you might need more sophisticated replacements
        size_t pos = 0;
        while ((pos = temp_str.find("^", pos)) != string::npos) {
            temp_str.replace(pos, 1, "**");
            pos += 2;
        }
        
        return temp_str;
    } else {
        // Use default GiNaC formatting
        oss << expr;
        return oss.str();
    }
}

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

ex get_numeric_factor(const ex& term) {
    if (is_a<numeric>(term)) {
        // Term is just a numeric value
        return term;
    } else if (is_a<mul>(term)) {
        // Term is a multiplication like numeric * rest
        // Check for specific case like Mul(-1, symbol) or Mul(-1, pow)
        if (term.nops() >= 2 && term.op(0) == -1 && !is_a<numeric>(term.op(1))) {
             return numeric(-1);
        }
        // Assume the first operand IS the numeric factor if it's numeric
        // This directly mirrors the safe arg.args[0] assumption
        else if (term.nops() > 0 && is_a<numeric>(term.op(0))) {
            return term.op(0);
        } else {
            // If no leading numeric factor (e.g., k0*e0), assume factor is 1
            // This case might be covered by the "safe" assumption implicitly
            // meaning such terms might not appear additively on their own
            // or are structured as Mul(1, k0, e0). For safety, return 1.
            return numeric(1);
        }
    } else {
        // Term is not numeric, not mul (e.g., k0, e0, e0^2, etc.)
        // Assume implicit factor is 1, as Python's abs(arg.args[0]) would
        // likely fail or return non-numeric here, but the assumption
        // guarantees safety/numeric output in Python. Mimic by returning 1.
        return numeric(1);
    }
}

int main() {
    // Open a file for the output
    ofstream outfile("ginac_toy_output.txt");
    if (!outfile.is_open()) {
        cerr << "Failed to open output file" << endl;
        return 1;
    }
    
    // Load datasets
    matrix X_test = load_matrix_from_csv("X_test_toy.csv");
    outfile << "X_test: " << X_test.rows() << "x" << X_test.cols() << endl;
    
    matrix y_test = load_matrix_from_csv("y_test_toy.csv");
    outfile << "y_test: " << y_test.rows() << "x" << y_test.cols() << endl;

    matrix X_train = load_matrix_from_csv("X_train_toy.csv");
    outfile << "X_train: " << X_train.rows() << "x" << X_train.cols() << endl;

    matrix y_train = load_matrix_from_csv("y_train_toy.csv");
    outfile << "y_train: " << y_train.rows() << "x" << y_train.cols() << endl;

    matrix X_max = load_matrix_from_csv("X_max_toy.csv");
    outfile << "X_max: " << X_max.rows() << "x" << X_max.cols() << endl;

    matrix X_min = load_matrix_from_csv("X_min_toy.csv");
    outfile << "X_min: " << X_min.rows() << "x" << X_min.cols() << endl;
    
    auto full_start_time = high_resolution_clock::now();
    int X_rows = X_train.rows();
    int X_cols = X_train.cols();
    int curr_id = 0;
    float lr = 0.01;
    float reg = 0.0;

    vector<symbol> e_symbols;
    vector<symbol> k_symbols;
    vector<symbol> ep_symbols;

    matrix XS(X_rows, X_cols);
    matrix XR(X_rows, X_cols);
    matrix yS(X_rows, 1);
    matrix yR(X_rows, 1);
    for (int i = 0; i < X_rows; i++) {
        yS(i, 0) = numeric(0);
        yR(i, 0) = y_train(i, 0);

        for (int j = 0; j < X_cols; j++) {
            float max_val = ex_to<numeric>(X_max(i, j)).to_double();;
            float min_val = ex_to<numeric>(X_min(i, j)).to_double();;

            if (max_val != min_val) {
                cout << "uncertain column: " << j << endl;
                float xmean = (max_val + min_val) / 2;
                float xradius = (max_val - min_val) / 2;
                XR(i, j) = xmean;
                e_symbols.push_back(symbol("e" + to_string(curr_id)));
                XS(i, j) = xradius * e_symbols[curr_id];
                curr_id++;
            } else {
                float xval = ex_to<numeric>(X_train(i, j)).to_double();;
                XR(i, j) = xval;
                XS(i, j) = numeric(0);
            }
        }
    }

    matrix XS_T = XS.transpose();
    matrix XR_T = XR.transpose();

    matrix identity(X_cols, X_cols, lst());
    for (unsigned i = 0; i < X_cols; i++) {
        identity(i, i) = 1;
    }
    matrix common_inv = ex_to<matrix>((XR_T.mul(XR) + reg*X_rows*identity).evalm());
    common_inv = common_inv.inverse();
    
    matrix svd_base = XR_T.mul(XR);
    MatrixXd svd_eigen = ginacToEigen(svd_base);
    cout << svd_eigen << endl;
    BDCSVD<MatrixXd> svd(svd_eigen, ComputeFullU | ComputeFullV);

    MatrixXd Ve = svd.matrixU();
    MatrixXd VTe = svd.matrixV();
    VTe.transposeInPlace();
    cout << "VT" << VTe << endl;
    VectorXd sigmae = svd.singularValues();
    MatrixXd sigmae_row = sigmae.transpose();

    matrix V = eigenToGinac(Ve);
    matrix VT = eigenToGinac(VTe);
    matrix sigma = eigenToGinac(sigmae_row);

    // VT = {{-0.0615807994364942, -0.489945236074280, -0.547114200277674, -0.171422990940765, -0.361240855888811, -0.0980979695107730, -0.536025664716632},
    //     {0.0296825574424337, 0.166552498177049, -0.321665797083011, -0.0986563511031628, 0.590274284982137, -0.711147308935095, -0.0634273126643574},
    //     {-0.347930838576584, -0.179641724515185, 0.146387543365912, -0.832262983360288, -0.0340604633809169, -0.0673912165427683, 0.356202617653624},
    //     {0.922532836719264, 0.0312769981376152, 0.00385682305831768, -0.366523395733000, -0.105620743657351, 0.00286224435498667, 0.0493630015798760},
    //     {-0.0652953957400408, 0.267213899251905, 0.492908010603220, -0.280120193638925, 0.145417705268357, 0.0642531518577407, -0.760021602989970},
    //     {0.137421834042694, -0.790879373477078, 0.379640243642966, 0.155134288231973, 0.432339280034909, -0.0121681510285436, -0.0191409821524003},
    //     {0.00746983868870655, -0.0455091183710457, 0.434359249233338, 0.177048126798802, -0.548362921615305, -0.689797066853192, 0.0365682624115635}};

    matrix wR = ex_to<matrix>((common_inv*XR_T*yR).evalm());
    outfile << "wR: " << wR << endl;
    outfile << endl;
    // wS_data = common_inv*((XS.T*XR + XR.T*XS)*wR - XS.T*yR - XR.T*yS)
    matrix wsd_intermediate = XS_T.mul(XR);
    outfile << "wsd_intermediate: " <<  wsd_intermediate << endl;
    wsd_intermediate = wsd_intermediate.add(XR_T.mul(XS));
    wsd_intermediate = ex_to<matrix>(wsd_intermediate.expand().normal());
    outfile << "wsd_intermediate: " <<  wsd_intermediate << endl;
    wsd_intermediate = wsd_intermediate.mul(wR);
    wsd_intermediate = ex_to<matrix>(wsd_intermediate.expand().normal());
    outfile << "wsd_intermediate: " <<  wsd_intermediate << endl;
    wsd_intermediate = wsd_intermediate.sub(XS_T.mul(yR));
    wsd_intermediate = ex_to<matrix>(wsd_intermediate.expand().normal());
    outfile << "wsd_intermediate: " <<  wsd_intermediate << endl;
    wsd_intermediate = wsd_intermediate.sub(XR_T.mul(yS));
    wsd_intermediate = ex_to<matrix>(wsd_intermediate.expand().normal());
    outfile << "wsd_intermediate: " <<  wsd_intermediate << endl;
    matrix wS_data = common_inv.mul(wsd_intermediate);
    wS_data = ex_to<matrix>(wS_data.expand().normal());


    // matrix wS_data = ex_to<matrix>((common_inv*((XS_T*XR + XR_T*XS)*wR - XS_T*yR - XR_T*yS)).evalm());
    matrix wS_non_data(VT.cols(), 1);
    for (int i = 0; i < X_cols; i++) {
        k_symbols.push_back(symbol("k" + to_string(i)));
        ep_symbols.push_back(symbol("ep" + to_string(i)));
        wS_non_data = ex_to<matrix>((wS_non_data + k_symbols[i] * ep_symbols[i] * get_row(VT, i).transpose()).evalm());
    }
    outfile << "wS_data: " << wS_data << endl;
    outfile << endl;

    matrix eigenvalues(1, sigma.cols());
    for (unsigned j = 0; j < sigma.cols(); j++) {
        eigenvalues(0, j) = 1 - 2*lr*reg - 2*lr*sigma(0, j)/X_rows;
        assert(eigenvalues(0, j) >= 0);
        assert(eigenvalues(0, j) <= 1);
    }

    matrix wS = ex_to<matrix>((wS_non_data + wS_data).evalm());
    outfile << "wS: " << wS << endl;
    outfile << endl;
    matrix w = ex_to<matrix>((wS + wR).evalm());
    outfile << "w: " << w << endl;
    outfile << endl;

    auto start_time = high_resolution_clock::now();
    ex scalar = (numeric(-2.0) * lr) / X_rows;
    // matrix matrix_part = ex_to<matrix>(((((XS_T)*XR + (XR_T)*XS + (XS_T)*XS)*wS + (XS_T)*XS*wR - (XS_T)*yS).expand().normal()).evalm());
    matrix matrix_part = XS_T.mul(XR).add(XR_T.mul(XS));
    matrix_part = matrix_part.add(XS_T.mul(XS));
    matrix_part = ex_to<matrix>(matrix_part.expand().normal());
    matrix_part = matrix_part.mul(wS);
    matrix_part = ex_to<matrix>(matrix_part.expand().normal());
    matrix matrix_part2 = (XS_T).mul(XS);
    matrix_part2 = matrix_part2.mul(wR);
    matrix_part2 = ex_to<matrix>(matrix_part2.expand().normal());
    matrix_part = matrix_part.add(matrix_part2);
    matrix_part = ex_to<matrix>(matrix_part.expand().normal());
    matrix_part = matrix_part.sub(XS_T.mul(yS));
    matrix_part = ex_to<matrix>(matrix_part.expand().normal());
    matrix w_prime = matrix_part.mul_scalar(scalar);
    w_prime = ex_to<matrix>(w_prime.expand().normal());
    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end_time - start_time);
    cout << "W# (abstract gradient descent?) computation time: " << duration.count() << " ms" << endl;
    outfile << "w_prime: " << w_prime << endl;
    outfile << endl;

    start_time = high_resolution_clock::now();
    matrix w_prime_projected = VT.mul(w_prime);
    w_prime_projected = ex_to<matrix>(w_prime_projected.expand().normal());
    end_time = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(end_time - start_time);
    cout << "Projection computation time: " << duration.count() << " ms" << endl;
    outfile << "w_prime_projected[2]: " << w_prime_projected[2] << endl;
    outfile << endl;

    vector<ex> eqs;

    for (int d = 0; d < X_cols; d++) {
        ex eq1 = (1 - GiNaC::abs(eigenvalues(0, d))) * k_symbols[d];
        
        // Use a vector for coefficients
        vector<ex> coef_for_k(X_cols, 0);
        ex const_coef = 0;
        
        // Process all terms in w_prime_projected[d]
        if (is_a<add>(w_prime_projected[d])) {
            for (size_t i = 0; i < w_prime_projected[d].nops(); i++) {
                // if (d == 0 && i <= 5) {
                //     cout << "Term " << i << ": " << w_prime_projected[d].op(i) << endl;
                // }
                ex term = w_prime_projected[d].op(i);
                bool found_k = false;
                
                // Find which k symbol this term contains
                for (int j = 0; j < X_cols; j++) {
                    if (term.has(k_symbols[j])) {
                        // Extract the numeric coefficient
                        ex numeric_coeff = 0;
                        
                        if (is_a<mul>(term)) {
                            // Find the numeric factor
                            for (size_t idx = 0; idx < term.nops(); idx++) {
                                if (is_a<numeric>(term.op(idx))) {
                                    numeric_coeff = term.op(idx);
                                    // if (d == 0 && i <= 5) {
                                    //     cout << "Numeric coefficient: " << numeric_coeff << endl;
                                    // }
                                    break;
                                }
                            }
                        } else if (term == k_symbols[j]) {
                            numeric_coeff = 1;
                        }
                        
                        if (!numeric_coeff.is_zero()) {
                            // Take absolute value immediately (like Python does with GiNaC::abs(arg.args[0]))
                            // and divide by 2
                            coef_for_k[j] += GiNaC::abs(numeric_coeff);
                        }
                        
                        found_k = true;
                        break;
                    }
                }
                
                // If no k symbol found, add to constant term
                if (!found_k) {
                    if (is_a<mul>(term)) {
                        // Find the numeric factor
                        for (size_t idx = 0; idx < term.nops(); idx++) {
                            if (is_a<numeric>(term.op(idx))) {
                                const_coef += GiNaC::abs(term.op(idx));
                                // if (d == 0 && i <= 5) {
                                //     cout << "Numeric coefficient: " << const_coef << endl;
                                // }
                                break;
                            }
                        }
                    }
                }
            }
        }
        
        // Build eq2 directly (no need to take abs again)
        ex eq2 = const_coef;
        for (int i = 0; i < X_cols; i++) {
            eq2 += k_symbols[i] * coef_for_k[i];
        }
        
        // if (d == 0) {
        //     cout << "Equation " << d << " details:" << endl;
        //     cout << "  Left side: " << eq1 << endl;
        //     cout << "  Constant term: " << const_coef << endl;
        //     for (int i = 0; i < X_cols; i++) {
        //         cout << "  k" << i << " coefficient: " << coef_for_k[i] << endl;
        //     }
        // }
        
        eqs.push_back(eq1 == eq2);
    }

    cout << "\nEquations:" << endl;
    for (size_t i = 0; i < eqs.size(); i++) {
        cout << "Equation " << i << ": " << eqs[i] << endl;
    }

    // Solve the linear system of equations
    matrix A(eqs.size(), k_symbols.size());  // Changed dimensions
    matrix b(eqs.size(), 1);                 // Changed dimensions

    // Extract coefficients for linear system Ax = b
    for (unsigned i = 0; i < eqs.size(); i++) {
        ex eq = eqs[i];
        ex lhs = eq.lhs();
        ex rhs = eq.rhs();
        
        // Extract coefficients from left side
        for (unsigned j = 0; j < k_symbols.size(); j++) {
            A(i, j) = lhs.coeff(k_symbols[j], 1);
        }
        
        // Extract coefficients from right side and build constant term
        ex const_term = rhs;
        for (unsigned j = 0; j < k_symbols.size(); j++) {
            A(i, j) = A(i, j) - rhs.coeff(k_symbols[j], 1);
            const_term = const_term.coeff(k_symbols[j], 0);
        }
        b(i, 0) = const_term;
    }

    // Convert to a system of equations and variables for lsolve
    lst equations;
    lst variables;

    // Build equations from the matrix A and vector b
    for (unsigned i = 0; i < eqs.size(); i++) {  // Changed to use eqs.size()
        ex eq = 0;
        for (unsigned j = 0; j < k_symbols.size(); j++) {
            eq += A(i, j) * k_symbols[j];
        }
        equations.append(eq == b(i, 0));
    }

    // Build variables list
    for (unsigned i = 0; i < k_symbols.size(); i++) {
        variables.append(k_symbols[i]);
    }

    // Solve using lsolve
    ex solution = lsolve(equations, variables);

    vector<pair<symbol, ex>> result;

    // Extract solutions from the result
    if (is_a<lst>(solution)) {
        lst sol_list = ex_to<lst>(solution);
        
        for (unsigned i = 0; i < sol_list.nops(); i++) {
            if (is_a<relational>(sol_list.op(i))) {
                relational rel = ex_to<relational>(sol_list.op(i));
                
                if (is_a<symbol>(rel.lhs())) {
                    symbol var = ex_to<symbol>(rel.lhs());
                    result.push_back(make_pair(var, rel.rhs()));
                }
            }
        }
    }

    // Assert all values are non-negative
    for (const auto& kv : result) {
        // Convert to numeric and check
        ex val = kv.second.evalf();
        if (is_a<numeric>(val)) {
            assert(ex_to<numeric>(val).to_double() >= 0);
        } else {
            cerr << "Warning: Could not evaluate " << kv.first << " to numeric value" << endl;
        }
    }

    // Print result
    cout << "Result:" << endl;
    for (const auto& kv : result) {
        cout << kv.first << " = " << kv.second << endl;
    }

    // Create substitution map
    exmap subs_map;
    for (const auto& kv : result) {
        subs_map[kv.first] = kv.second;
    }

    // Substitute into wS and add to wR
    // First, create a copy of wS to avoid modifying original
    matrix wS_substituted(wS.rows(), wS.cols());
    for (unsigned i = 0; i < wS.rows(); i++) {
        for (unsigned j = 0; j < wS.cols(); j++) {
            wS_substituted(i, j) = wS(i, j).subs(subs_map);
        }
    }

    // Now create param matrix by adding wR and substituted wS
    matrix param(wR.rows(), wR.cols());
    for (unsigned i = 0; i < wR.rows(); i++) {
        for (unsigned j = 0; j < wR.cols(); j++) {
            param(i, j) = wR(i, j) + wS_substituted(i, j);
        }
    }

    auto full_end_time = high_resolution_clock::now();
    auto full_duration = duration_cast<milliseconds>(full_end_time - full_start_time);
    cout << "Full pipeline time: " << full_duration.count() << " ms" << endl;

    outfile << param << endl;
    
    outfile.close();
    return 0;
}