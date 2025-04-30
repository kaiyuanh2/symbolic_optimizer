#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
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

namespace py = pybind11;
using namespace std;
using namespace GiNaC;
using namespace std::chrono;

// Custom visitor class to find all symbols
class symbol_finder : public visitor {
    private:
        map<string, ex>& symbol_map;
    
    public:
        symbol_finder(map<string, ex>& m) : symbol_map(m) {}
    
        void visit(const symbol& s) {
            string name = s.get_name();
            symbol_map[name] = s;
        }
        
        void visit(const basic& b) {
            // Do nothing for other types
        }
};
    
// Function to find all symbols in an expression
lst find_all_symbols(const ex& e) {
    map<string, ex> symbol_map;
    symbol_finder sf(symbol_map);
        
    e.traverse(sf);
        
    lst symbols;
    for (const auto& pair : symbol_map) {
        symbols.append(pair.second);
    }
        
    return symbols;
}
    
// Sort product factors in a consistent way
ex sort_product_factors(const ex& e, const map<string, int>& symbol_priority) {
    if (!is_a<mul>(e)) {
        return e;
    }
    
    // Extract factors
    vector<ex> numeric_factors;
    vector<ex> symbol_factors;
    vector<ex> other_factors; // For factors that are neither numeric nor plain symbols
    
    for (size_t i = 0; i < e.nops(); ++i) {
        if (is_a<numeric>(e.op(i))) {
            numeric_factors.push_back(e.op(i));
        } else if (is_a<symbol>(e.op(i))) {
            symbol_factors.push_back(e.op(i));
        } else {
            // Recursively sort nested products if present
            if (is_a<mul>(e.op(i))) {
                other_factors.push_back(sort_product_factors(e.op(i), symbol_priority));
            } else {
                other_factors.push_back(e.op(i));
            }
        }
    }
    
    // Sort symbol factors based on the provided priority map
    sort(symbol_factors.begin(), symbol_factors.end(), [&symbol_priority](const ex& a, const ex& b) {
        string name_a = ex_to<symbol>(a).get_name();
        string name_b = ex_to<symbol>(b).get_name();
        
        int priority_a = symbol_priority.find(name_a) != symbol_priority.end() ? 
                            symbol_priority.at(name_a) : INT_MAX;
        int priority_b = symbol_priority.find(name_b) != symbol_priority.end() ? 
                            symbol_priority.at(name_b) : INT_MAX;
        
        // If priorities are equal, sort alphabetically
        if (priority_a == priority_b) {
            return name_a < name_b;
        }
        
        return priority_a < priority_b;
    });
    
    // Sort other factors
    sort(other_factors.begin(), other_factors.end(), [](const ex& a, const ex& b) {
        ostringstream oss_a, oss_b;
        oss_a << a;
        oss_b << b;
        return oss_a.str() < oss_b.str();
    });
    
    // Reconstruct the product with the sorted factors
    ex result = 1;
    
    // First multiply all numeric factors
    for (const auto& factor : numeric_factors) {
        result = result * factor;
    }
    
    // Then multiply all sorted symbol factors
    for (const auto& factor : symbol_factors) {
        result = result * factor;
    }
    
    // Then multiply all other factors
    for (const auto& factor : other_factors) {
        result = result * factor;
    }
    
    return result;
}

// Generate a dynamic priority map based on the symbols found in the expression
map<string, int> generate_symbol_priority_map(const lst& symbols) {
    map<string, int> priority_map;
    
    // First pass: collect all symbol names and sort them alphabetically
    vector<string> symbol_names;
    for (size_t i = 0; i < symbols.nops(); ++i) {
        if (is_a<symbol>(symbols.op(i))) {
            string name = ex_to<symbol>(symbols.op(i)).get_name();
            symbol_names.push_back(name);
        }
    }
    
    // Sort symbol names alphabetically
    sort(symbol_names.begin(), symbol_names.end());
    
    // Assign priorities (lower number = higher priority)
    for (size_t i = 0; i < symbol_names.size(); ++i) {
        priority_map[symbol_names[i]] = i + 1;
    }
    
    return priority_map;
}

// Create a string-based key for expression comparison with sorted factors
string expr_to_key(const ex& e, const map<string, int>& symbol_priority) {
    // For products, first sort the factors
    ex sorted_expr = e;
    if (is_a<mul>(e)) {
        sorted_expr = sort_product_factors(e, symbol_priority);
    } else if (is_a<add>(e)) {
        // For sums, sort each term
        sorted_expr = 0;
        for (size_t i = 0; i < e.nops(); ++i) {
            sorted_expr = sorted_expr + sort_product_factors(e.op(i), symbol_priority);
        }
    }
    
    // Convert the sorted expression to a string for comparison
    ostringstream oss;
    oss << sorted_expr;
    return oss.str();
}

// Function to collect and combine like terms with string-based comparison
ex combine_like_terms_robust(const ex& e, const map<string, int>& symbol_priority) {
    // If not a sum, just return the sorted expression
    if (!is_a<add>(e)) {
        return sort_product_factors(e, symbol_priority);
    }
    
    // First collect all terms with the same structure using string-based keys
    map<string, ex> coef_map;
    map<string, ex> var_map;
    
    // Process each term in the sum
    for (size_t i = 0; i < e.nops(); ++i) {
        ex term = e.op(i);
        ex coef = 1;
        ex var_part = term;
        
        // Extract numerical coefficient if present
        if (is_a<mul>(term)) {
            bool found_numeric = false;
            
            // Gather all numeric factors and variable factors
            ex numeric_part = 1;
            ex variable_part = 1;
            
            for (size_t j = 0; j < term.nops(); ++j) {
                if (is_a<numeric>(term.op(j))) {
                    numeric_part = numeric_part * term.op(j);
                    found_numeric = true;
                } else {
                    variable_part = variable_part * term.op(j);
                }
            }
            
            coef = numeric_part;
            var_part = variable_part;
        } else if (is_a<numeric>(term)) {
            coef = term;
            var_part = 1;
        }
        
        // Sort the variable part for consistent comparison
        var_part = sort_product_factors(var_part, symbol_priority);
        
        // Generate a string key for this variable part
        string key = expr_to_key(var_part, symbol_priority);
        
        // Combine like terms
        if (coef_map.find(key) != coef_map.end()) {
            coef_map[key] = coef_map[key] + coef;
        } else {
            coef_map[key] = coef;
            var_map[key] = var_part;
        }
    }
    
    // Rebuild the expression
    ex result = 0;
    for (const auto& pair : coef_map) {
        string key = pair.first;
        ex coef = pair.second;
        ex var_part = var_map[key];
        
        // Skip terms with zero coefficient
        if (coef.is_zero())
            continue;
        
        // Handle constant term (var_part is 1)
        if (var_part.is_equal(1)) {
            result = result + coef;
        }
        // Terms with coefficient of 1 don't need to be multiplied
        else if (coef.is_equal(1)) {
            result = result + var_part;
        }
        // General case: coefficient * var_part
        else {
            result = result + coef * var_part;
        }
    }
    
    return result;
}

// Function to extract all symbol names from an expression string
vector<string> extract_symbol_names(const string& expr_str) {
    vector<string> symbol_names;
    set<string> unique_symbols; // To avoid duplicates
    
    // Regex pattern to match variables - assumes variables start with a letter
    // followed by alphanumeric characters or underscore
    // This pattern specifically looks for e\d+ and ep\d+ patterns in your example
    regex symbol_pattern("\\b([a-z][a-z0-9_]*)\\b");
    
    auto words_begin = sregex_iterator(expr_str.begin(), expr_str.end(), symbol_pattern);
    auto words_end = sregex_iterator();
    
    for (auto i = words_begin; i != words_end; ++i) {
        string symbol_name = i->str();
        
        // Skip numeric literals, constants, and function names
        if (unique_symbols.find(symbol_name) == unique_symbols.end() && 
            symbol_name != "e" && // skip mathematical constant e
            symbol_name != "sin" && symbol_name != "cos" && symbol_name != "tan" && 
            symbol_name != "exp" && symbol_name != "log" && 
            symbol_name != "sqrt" && symbol_name != "pow") {
            
            unique_symbols.insert(symbol_name);
            symbol_names.push_back(symbol_name);
        }
    }
    
    return symbol_names;
}

// Function to create a symbol table from an expression string
symtab create_symbol_table_from_expression(const string& expr_str) {
    symtab table;
    vector<string> symbol_names = extract_symbol_names(expr_str);
    
    cout << "Found symbols: ";
    for (const auto& name : symbol_names) {
        // Create the symbol
        symbol sym(name);
        
        // Add to the symbol table
        table[name] = sym;
        
        cout << name << " ";
    }
    cout << endl;
    
    return table;
}

class ExpressionCache {
    private:
        // Cache for expression results
        unordered_map<size_t, ex> cache;
        // Hash function for expressions
        size_t hash_ex(const ex& e) const {
            ostringstream oss;
            oss << e;
            string s = oss.str();
            return hash<string>{}(s);
        }
    
    public:
        // Check if expression is in cache
        bool has(const ex& e) const {
            size_t key = hash_ex(e);
            return cache.find(key) != cache.end();
        }
    
        // Get cached result
        ex get(const ex& e) const {
            size_t key = hash_ex(e);
            return cache.at(key);
        }
    
        // Store result in cache
        void store(const ex& e, const ex& result) {
            size_t key = hash_ex(e);
            cache[key] = result;
        }
    
        // Clear cache
        void clear() {
            cache.clear();
        }
};
    
// Global cache instance
ExpressionCache expr_cache;

ex combine_like_terms_advanced_optimized(const ex& e) {
    // Check if we've already processed this expression
    if (expr_cache.has(e)) {
        return expr_cache.get(e);
    }

    // Find all symbols only once
    auto start_time = high_resolution_clock::now();
    lst symbols = find_all_symbols(e);
    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end_time - start_time).count();
    cout << "Symbol extraction time: " << duration << " ms" << endl;
    
    // Generate symbol priority map
    start_time = high_resolution_clock::now();
    map<string, int> symbol_priority = generate_symbol_priority_map(symbols);
    end_time = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(end_time - start_time).count();
    cout << "Symbol priority generation time: " << duration << " ms" << endl;
    
    // Main processing: combine like terms
    start_time = high_resolution_clock::now();
    ex result = combine_like_terms_robust(e, symbol_priority);
    end_time = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(end_time - start_time).count();
    cout << "combine_like_terms_robust() time: " << duration << " ms" << endl;
    
    expr_cache.store(e, result);
    
    return result;
}

// Enhanced combine function with batched processing for very large expressions
ex combine_like_terms_advanced_large(const ex& e) {
    // For very large expressions, we can split them into batches
    if (is_a<add>(e) && e.nops() > 1000) {
        cout << "Processing large expression with " << e.nops() << " terms using batched approach..." << endl;
        
        // Process in batches of 1000 terms
        const size_t BATCH_SIZE = 1000;
        size_t num_batches = (e.nops() + BATCH_SIZE - 1) / BATCH_SIZE; // Ceiling division
        
        ex result = 0;
        
        for (size_t batch = 0; batch < num_batches; ++batch) {
            size_t start_idx = batch * BATCH_SIZE;
            size_t end_idx = min(start_idx + BATCH_SIZE, e.nops());
            
            // Create a sub-expression with the current batch of terms
            ex batch_expr = 0;
            for (size_t i = start_idx; i < end_idx; ++i) {
                batch_expr = batch_expr + e.op(i);
            }
            
            // Process this batch
            cout << "Processing batch " << (batch + 1) << " of " << num_batches << "..." << endl;
            ex batch_result = combine_like_terms_advanced_optimized(batch_expr);
            
            // Add to the overall result
            result = result + batch_result;
        }
        
        // Final combination of the batched results
        cout << "Combining all batched results..." << endl;
        return combine_like_terms_advanced_optimized(result);
    }
    
    // For smaller expressions, use the optimized version directly
    return combine_like_terms_advanced_optimized(e);
}

// Helper function to print memory usage (if available)
void print_memory_usage() {
#ifdef __linux__
    // On Linux, read memory usage from /proc/self/status
    ifstream status("/proc/self/status");
    string line;
    while (getline(status, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            cout << "Current memory usage: " << line.substr(6) << endl;
            break;
        }
    }
#endif
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

// Efficient term hashing and comparison without strings
class term_hash {
    private:
        // Store factors in a canonical form
        vector<pair<ex, int>> factors; // Symbol and its power
        ex coefficient;
        mutable size_t cached_hash;
        mutable bool hash_valid;
        
        // Private method to extract and normalize factors
        void extract_factors(const ex& e) {
            factors.clear();
            coefficient = 1;
            hash_valid = false;
            
            if (is_a<mul>(e)) {
                for (size_t i = 0; i < e.nops(); ++i) {
                    if (is_a<numeric>(e.op(i))) {
                        coefficient = coefficient * e.op(i);
                    } else if (is_a<power>(e.op(i))) {
                        // Handle powers: x^n
                        ex base = e.op(i).op(0);
                        ex exponent = e.op(i).op(1);
                        
                        if (is_a<symbol>(base) && is_a<numeric>(exponent)) {
                            // Store symbol and its power
                            factors.push_back(make_pair(base, ex_to<numeric>(exponent).to_int()));
                        } else {
                            // Other types of powers
                            factors.push_back(make_pair(e.op(i), 1));
                        }
                    } else if (is_a<symbol>(e.op(i))) {
                        // Plain symbol (power 1)
                        factors.push_back(make_pair(e.op(i), 1));
                    } else {
                        // Other types
                        factors.push_back(make_pair(e.op(i), 1));
                    }
                }
            } else if (is_a<power>(e)) {
                // Single power term
                ex base = e.op(0);
                ex exponent = e.op(1);
                
                if (is_a<symbol>(base) && is_a<numeric>(exponent)) {
                    factors.push_back(make_pair(base, ex_to<numeric>(exponent).to_int()));
                } else {
                    factors.push_back(make_pair(e, 1));
                }
            } else if (is_a<symbol>(e)) {
                // Single symbol
                factors.push_back(make_pair(e, 1));
            } else if (is_a<numeric>(e)) {
                // Just a number
                coefficient = e;
            } else {
                // Other expression type
                factors.push_back(make_pair(e, 1));
            }
            
            // Sort factors by canonical order (symbol name)
            sort(factors.begin(), factors.end(), [](const pair<ex, int>& a, const pair<ex, int>& b) {
                if (is_a<symbol>(a.first) && is_a<symbol>(b.first)) {
                    return ex_to<symbol>(a.first).get_name() < ex_to<symbol>(b.first).get_name();
                } else {
                    // Fall back to string comparison for non-symbols
                    ostringstream oss_a, oss_b;
                    oss_a << a.first;
                    oss_b << b.first;
                    return oss_a.str() < oss_b.str();
                }
            });
        }
        
        // Calculate hash based on normalized factors
        size_t calculate_hash() const {
            size_t h = 0;
            
            // Hash the coefficient
            {
                ostringstream oss;
                oss << coefficient;
                string s = oss.str();
                h = hash<string>{}(s);
            }
            
            // Combine with hash of factors
            for (const auto& factor : factors) {
                size_t factor_hash = 0;
                
                // Hash the symbol part
                {
                    ostringstream oss;
                    oss << factor.first;
                    string s = oss.str();
                    factor_hash = hash<string>{}(s);
                }
                
                // Combine with exponent
                factor_hash ^= (size_t)factor.second;
                
                // Combine into overall hash
                h ^= factor_hash + 0x9e3779b9 + (h << 6) + (h >> 2);
            }
            
            return h;
        }
    
    public:
        term_hash(const ex& e) {
            extract_factors(e);
        }
        
        // Get the hash - make it const correct
        size_t get_hash() const {
            if (!hash_valid) {
                cached_hash = calculate_hash();
                hash_valid = true;
            }
            return cached_hash;
        }
        
        // Compare two terms for equality
        bool equals(const term_hash& other) const {
            // Compare coefficient
            if (!coefficient.is_equal(other.coefficient)) return false;
            
            // Compare number of factors
            if (factors.size() != other.factors.size()) return false;
            
            // Compare each factor
            for (size_t i = 0; i < factors.size(); ++i) {
                if (!factors[i].first.is_equal(other.factors[i].first) || 
                    factors[i].second != other.factors[i].second) {
                    return false;
                }
            }
            
            return true;
        }
        
        // Get the variable part (without coefficient)
        ex get_var_part() const {
            if (factors.empty()) return 1;
            
            ex result = 1;
            for (const auto& factor : factors) {
                if (factor.second == 1) {
                    result = result * factor.first;
                } else {
                    result = result * pow(factor.first, factor.second);
                }
            }
            
            return result;
        }
        
        // Get the coefficient
        ex get_coefficient() const {
            return coefficient;
        }
};
    
// Fast combine like terms using our optimized term hashing
ex fast_combine_like_terms(const ex& e) {
    // If not a sum, return as is
    if (!is_a<add>(e)) return e;
    
    // Use a hash map for fast lookup
    unordered_map<size_t, ex> coef_map;  // Maps hash to coefficient
    unordered_map<size_t, ex> var_map;   // Maps hash to variable part
    vector<term_hash> term_hashes;       // Keep track of all hashes
    
    // Process each term in the sum
    for (size_t i = 0; i < e.nops(); ++i) {
        ex term = e.op(i);
        term_hash th(term);
        
        size_t hash = th.get_hash();
        ex var_part = th.get_var_part();
        ex coef = th.get_coefficient();
        
        // Check if we already have this term pattern
        bool found = false;
        for (const auto& existing_th : term_hashes) {
            if (existing_th.get_hash() == hash && existing_th.equals(th)) {
                // This is a match - we found a like term
                hash = existing_th.get_hash(); // Use the existing hash
                found = true;
                break;
            }
        }
        
        if (!found) {
            // This is a new term pattern
            term_hashes.push_back(th);
            coef_map[hash] = coef;
            var_map[hash] = var_part;
        } else {
            // This matches an existing term pattern
            coef_map[hash] = coef_map[hash] + coef;
        }
    }
    
    // Rebuild the expression
    ex result = 0;
    for (const auto& th : term_hashes) {
        size_t hash = th.get_hash();
        ex coef = coef_map[hash];
        ex var_part = var_map[hash];
        
        // Skip terms with zero coefficient
        if (coef.is_zero())
            continue;
        
        // Handle constant term (var_part is 1)
        if (var_part.is_equal(1)) {
            result = result + coef;
        }
        // Terms with coefficient of 1 don't need to be multiplied
        else if (coef.is_equal(1)) {
            result = result + var_part;
        }
        // General case: coefficient * var_part
        else {
            result = result + coef * var_part;
        }
    }
    
    return result;
}

class ThreadPool {
    public:
        ThreadPool(size_t threads) : stop(false) {
            if (threads == 0) threads = 1; // Ensure at least one thread
            for(size_t i = 0; i < threads; ++i)
                workers.emplace_back([this] {
                    while(true) {
                        std::function<void()> task;
                        {
                            std::unique_lock<std::mutex> lock(this->queue_mutex);
                            this->condition.wait(lock, [this]{ return this->stop || !this->tasks.empty(); });
                            if(this->stop && this->tasks.empty()) return;
                            task = std::move(this->tasks.front());
                            this->tasks.pop();
                        }
                        try {
                            task(); // Execute the task
                        } catch (const std::exception& e) {
                            std::cerr << "!!! Exception caught in thread pool worker: " << e.what() << std::endl;
                        } catch (...) {
                            std::cerr << "!!! Unknown exception caught in thread pool worker." << std::endl;
                        }
                    }
                });
        }
    
        template<class F, class... Args>
        auto enqueue(F&& f, Args&&... args)
            // -> std::future<typename std::invoke_result<F, Args...>::type> { // C++17 version (REMOVE/COMMENT OUT)
            -> std::future<typename std::result_of<F(Args...)>::type> {          // C++11/14 version (USE THIS)

            // using return_type = typename std::invoke_result<F, Args...>::type; // C++17 version (REMOVE/COMMENT OUT)
            using return_type = typename std::result_of<F(Args...)>::type;          // C++11/14 version (USE THIS)

            auto task = std::make_shared< std::packaged_task<return_type()> >(
                std::bind(std::forward<F>(f), std::forward<Args>(args)...)
            );

            std::future<return_type> res = task->get_future();
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                if(stop) throw std::runtime_error("enqueue on stopped ThreadPool");
                tasks.emplace([task](){ (*task)(); });
            }
            condition.notify_one();
            return res;
        }
    
        ~ThreadPool() {
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                stop = true;
            }
            condition.notify_all();
            for(std::thread &worker: workers)
                worker.join();
        }
    
    private:
        std::vector< std::thread > workers;
        std::queue< std::function<void()> > tasks;
        std::mutex queue_mutex;
        std::condition_variable condition;
        bool stop;
};

ex square_and_expand(const matrix& input_matrix) {
    size_t num_rows = input_matrix.rows();
    size_t num_columns = input_matrix.cols();

    if (num_rows * num_columns == 0) {
        return numeric(0); // Return symbolic zero for empty matrix
    }

    ex final_result = numeric(0);
    for (size_t k = 0; k < num_rows; ++k) {
        final_result += (input_matrix(k, 0) * input_matrix(k, 0)).expand();
    }

    return final_result;
}

// Utilities
// Function to multiply X_test and param and return simplified results
py::list multiply_matrices(const py::list& X_test_py, const py::list& param_py) {
    // Convert Python lists to GiNaC matrices
    auto start_time = high_resolution_clock::now();
    int X_rows = py::len(X_test_py);
    int X_cols = py::len(py::list(X_test_py[0]));
    
    matrix X_test(X_rows, X_cols);
    for (int i = 0; i < X_rows; i++) {
        py::list row = X_test_py[i];
        for (int j = 0; j < X_cols; j++) {
            // Parse expression from string
            parser reader;
            string expr_str = py::str(row[j]);
            X_test(i, j) = reader(expr_str);
        }
    }
    
    int param_rows = py::len(param_py);
    matrix param(param_rows, 1);
    for (int i = 0; i < param_rows; i++) {
        parser reader;
        string expr_str = py::str(param_py[i]);
        param(i, 0) = reader(expr_str);
    }
    
    // Perform matrix multiplication
    matrix result(X_rows, 1);
    for (int i = 0; i < X_rows; i++) {
        ex sum = 0;
        for (int j = 0; j < X_cols; j++) {
            sum += X_test(i, j) * param(j, 0);
        }
        
        // Apply GiNaC simplifications
        sum = sum.expand();
        
        result(i, 0) = sum;
    }

    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end_time - start_time);
    cout << "Time for multiplication: " << duration.count() << " ms" << endl;
    
    // Convert result back to Python list
    py::list result_py;
    for (int i = 0; i < result.rows(); i++) {
        ostringstream oss;
        oss << result(i, 0);
        result_py.append(oss.str());
    }
    
    return result_py;
}

py::list abstract_loss(const py::list& X_test_py, const py::list& y_test_py, const py::list& param_py) {
    // Convert Python lists to GiNaC matrices
    auto start_time = high_resolution_clock::now();
    int X_rows = py::len(X_test_py);
    int X_cols = py::len(py::list(X_test_py[0]));
    
    matrix X_test(X_rows, X_cols);
    for (int i = 0; i < X_rows; i++) {
        py::list row = X_test_py[i];
        for (int j = 0; j < X_cols; j++) {
            // Parse expression from string
            parser reader;
            string expr_str = py::str(row[j]);
            X_test(i, j) = reader(expr_str);
        }
    }

    cout << "X: " << X_test.rows() << "*" << X_test.cols() << endl;

    matrix y_test(X_rows, 1);
    for (int i = 0; i < X_rows; i++) {
        parser reader;
        string expr_str = py::str(y_test_py[i]);
        y_test(i, 0) = reader(expr_str);
    }

    cout << "y: " << y_test.rows() << "*" << y_test.cols() << endl;
    
    int param_rows = py::len(param_py);
    matrix param(param_rows, 1);
    for (int i = 0; i < param_rows; i++) {
        parser reader;
        string expr_str = py::str(param_py[i]);
        param(i, 0) = reader(expr_str);
    }
    cout << "param: " << param.rows() << "*" << param.cols() << endl;
    
    // test_preds = X_test * param
    matrix predictions(X_rows, 1);
    for (int i = 0; i < X_rows; i++) {
        ex sum = 0;
        for (int j = 0; j < X_cols; j++) {
            sum += X_test(i, j) * param(j, 0);
        }
        
        predictions(i, 0) = sum;
    }

    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end_time - start_time);
    cout << "Time for multiplication: " << duration.count() << " ms" << endl;

    start_time = high_resolution_clock::now();
    // diff = test_preds - y_test
    matrix diff(X_rows, 1);
    for (int i = 0; i < X_rows; i++) {
        diff(i, 0) = predictions(i, 0) - y_test(i, 0);
    }
    end_time = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(end_time - start_time);
    cout << "Time for difference: " << duration.count() << " ms" << endl;

    start_time = high_resolution_clock::now();
    // Calculate sum of squared differences
    ex sum_squared = square_and_expand(diff);
    // ex temp_result = 0;
    // for (int i = 0; i < diff.rows(); i++) {
    //     sum_squared += (diff(i, 0) * diff(i, 0)).expand();
    // }
    end_time = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(end_time - start_time);
    cout << "Time for square and expand: " << duration.count() << " ms" << endl;

    // start_time = high_resolution_clock::now();
    // expansion
    // sum_squared = sum_squared.expand();
    // end_time = high_resolution_clock::now();
    // duration = duration_cast<milliseconds>(end_time - start_time);
    // cout << "Time for expansion: " << duration.count() << " ms" << endl;

    cout << "Number of nodes in sum_squared: " << count_nodes(sum_squared) << endl;
    cout << "Number of terms in sum_squared: " << sum_squared.nops() << endl;
    cout << "Depth of sum_squared: " << get_depth(sum_squared) << endl;
    print_memory_usage();

    start_time = high_resolution_clock::now();
    sum_squared = combine_like_terms_advanced_optimized(sum_squared);
    end_time = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(end_time - start_time);
    cout << "Time for combining like terms: " << duration.count() << " ms" << endl;

    cout << "Number of nodes in sum_squared after expansion: " << count_nodes(sum_squared) << endl;
    cout << "Number of terms in sum_squared after expansion: " << sum_squared.nops() << endl;
    cout << "Depth of sum_squared after expansion: " << get_depth(sum_squared) << endl;
    print_memory_usage();

    sum_squared = sum_squared / numeric(X_rows);
    
    // Convert result back to Python list
    py::list result_py;
    for (int i = 0; i < 1; i++) {
        ostringstream oss;
        oss << sum_squared;
        result_py.append(oss.str());
    }
    
    return result_py;
}

py::str expand_single(const py::str& expression) {
    // Parse the expression from the string
    parser reader;
    ex expr = reader(expression);
    
    // Expand the expression
    ex expanded_expr = expr.expand();
    
    // Convert back to string and return
    ostringstream oss;
    oss << expanded_expr;
    return py::str(oss.str());
}

py::str expand_chunk(const py::list& expressions) {
    // Parse the expression from the list
    parser reader;
    int rows = py::len(expressions);
    ex chunk_sum = numeric(0);

    std::vector<ex> exprs(rows);
    for (int i = 0; i < rows; i++) {
        exprs[i] = reader(py::str(expressions[i]));
    }
    
    // Expand the expressions
    for (const ex& expr : exprs) {
        try {
            chunk_sum = chunk_sum + expr.expand();

        } catch (const std::exception& e) {
            return "__CPP_CHUNK_ERROR__";
        } catch (...) {
             return "__CPP_CHUNK_UNKNOWN_ERROR__";
        }
    }

    chunk_sum = combine_like_terms_advanced_optimized(chunk_sum);

    std::ostringstream oss;
    oss << chunk_sum;
    return py::str(oss.str());
}

py::list expand_matrix(const py::list& expressions, const py::int_& row, const py::int_& col) {
    // Parse the expression from the string
    parser reader;
    int rows = row.cast<int>();
    int cols = col.cast<int>();

    std::vector<ex> exprs(rows);
    for (int i = 0; i < rows; i++) {
        exprs[i] = reader(py::str(expressions[i]));
    }
    
    // Expand the expressions
    for (ex& expr : exprs) {
        expr = expr.expand();
    }
    
    // Convert back to python list and return
    py::list result_py;
    for (int i = 0; i < rows * cols; i++) {
        ostringstream oss;
        oss << exprs[i];
        result_py.append(oss.str());
    }

    return result_py;
}

py::str combine_like_terms(const py::str& expression) {
    // Parse the expression from the string
    parser reader;
    ex expr = reader(expression);
    
    // Expand the expression
    ex processed_expr = combine_like_terms_advanced_optimized(expr);
    
    // Convert back to string and return
    ostringstream oss;
    oss << processed_expr;
    return py::str(oss.str());
}

PYBIND11_MODULE(ginac_module, m) {
    m.doc() = "GiNaC operations module";
    m.def("multiply_matrices", &multiply_matrices, "Multiply X_test and param");
    m.def("abstract_loss", &abstract_loss, "Full pipeline for abstract loss");
    m.def("expand_single", &expand_single, "Expand one expression");
    m.def("expand_chunk", &expand_chunk, "Expand a chunk of expressions");
    m.def("expand_matrix", &expand_matrix, "Expand a matrix of expressions");
    m.def("combine_like_terms", &combine_like_terms, "Do combine like terms in one expression");
}