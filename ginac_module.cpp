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
#include <unordered_map>
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

namespace py = pybind11;
using namespace std;
using namespace GiNaC;
using namespace std::chrono;
using namespace Eigen;

// Custom visitor class to find all symbols
class symbol_finder : public visitor
{
private:
    map<string, ex> &symbol_map;

public:
    symbol_finder(map<string, ex> &m) : symbol_map(m) {}

    void visit(const symbol &s)
    {
        string name = s.get_name();
        symbol_map[name] = s;
    }

    void visit(const basic &b)
    {
        // Do nothing for other types
    }
};

// Function to find all symbols in an expression
lst find_all_symbols(const ex &e)
{
    map<string, ex> symbol_map;
    symbol_finder sf(symbol_map);

    e.traverse(sf);

    lst symbols;
    for (const auto &pair : symbol_map)
    {
        symbols.append(pair.second);
    }

    return symbols;
}

// Sort product factors in a consistent way
ex sort_product_factors(const ex &e, const map<string, int> &symbol_priority)
{
    if (!is_a<mul>(e))
    {
        return e;
    }

    // Extract factors
    vector<ex> numeric_factors;
    vector<ex> symbol_factors;
    vector<ex> other_factors; // For factors that are neither numeric nor plain symbols

    for (size_t i = 0; i < e.nops(); ++i)
    {
        if (is_a<numeric>(e.op(i)))
        {
            numeric_factors.push_back(e.op(i));
        }
        else if (is_a<symbol>(e.op(i)))
        {
            symbol_factors.push_back(e.op(i));
        }
        else
        {
            // Recursively sort nested products if present
            if (is_a<mul>(e.op(i)))
            {
                other_factors.push_back(sort_product_factors(e.op(i), symbol_priority));
            }
            else
            {
                other_factors.push_back(e.op(i));
            }
        }
    }

    // Sort symbol factors based on the provided priority map
    sort(symbol_factors.begin(), symbol_factors.end(), [&symbol_priority](const ex &a, const ex &b)
         {
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
        
        return priority_a < priority_b; });

    // Sort other factors
    sort(other_factors.begin(), other_factors.end(), [](const ex &a, const ex &b)
         {
        ostringstream oss_a, oss_b;
        oss_a << a;
        oss_b << b;
        return oss_a.str() < oss_b.str(); });

    // Reconstruct the product with the sorted factors
    ex result = 1;

    // First multiply all numeric factors
    for (const auto &factor : numeric_factors)
    {
        result = result * factor;
    }

    // Then multiply all sorted symbol factors
    for (const auto &factor : symbol_factors)
    {
        result = result * factor;
    }

    // Then multiply all other factors
    for (const auto &factor : other_factors)
    {
        result = result * factor;
    }

    return result;
}

// Generate a dynamic priority map based on the symbols found in the expression
map<string, int> generate_symbol_priority_map(const lst &symbols)
{
    map<string, int> priority_map;

    // First pass: collect all symbol names and sort them alphabetically
    vector<string> symbol_names;
    for (size_t i = 0; i < symbols.nops(); ++i)
    {
        if (is_a<symbol>(symbols.op(i)))
        {
            string name = ex_to<symbol>(symbols.op(i)).get_name();
            symbol_names.push_back(name);
        }
    }

    // Sort symbol names alphabetically
    sort(symbol_names.begin(), symbol_names.end());

    // Assign priorities (lower number = higher priority)
    for (size_t i = 0; i < symbol_names.size(); ++i)
    {
        priority_map[symbol_names[i]] = i + 1;
    }

    return priority_map;
}

// Create a string-based key for expression comparison with sorted factors
string expr_to_key(const ex &e, const map<string, int> &symbol_priority)
{
    // For products, first sort the factors
    ex sorted_expr = e;
    if (is_a<mul>(e))
    {
        sorted_expr = sort_product_factors(e, symbol_priority);
    }
    else if (is_a<add>(e))
    {
        // For sums, sort each term
        sorted_expr = 0;
        for (size_t i = 0; i < e.nops(); ++i)
        {
            sorted_expr = sorted_expr + sort_product_factors(e.op(i), symbol_priority);
        }
    }

    // Convert the sorted expression to a string for comparison
    ostringstream oss;
    oss << sorted_expr;
    return oss.str();
}

// Function to collect and combine like terms with string-based comparison
ex combine_like_terms_robust(const ex &e, const map<string, int> &symbol_priority)
{
    // If not a sum, just return the sorted expression
    if (!is_a<add>(e))
    {
        return sort_product_factors(e, symbol_priority);
    }

    // First collect all terms with the same structure using string-based keys
    map<string, ex> coef_map;
    map<string, ex> var_map;

    // Process each term in the sum
    for (size_t i = 0; i < e.nops(); ++i)
    {
        ex term = e.op(i);
        ex coef = 1;
        ex var_part = term;

        // Extract numerical coefficient if present
        if (is_a<mul>(term))
        {
            bool found_numeric = false;

            // Gather all numeric factors and variable factors
            ex numeric_part = 1;
            ex variable_part = 1;

            for (size_t j = 0; j < term.nops(); ++j)
            {
                if (is_a<numeric>(term.op(j)))
                {
                    numeric_part = numeric_part * term.op(j);
                    found_numeric = true;
                }
                else
                {
                    variable_part = variable_part * term.op(j);
                }
            }

            coef = numeric_part;
            var_part = variable_part;
        }
        else if (is_a<numeric>(term))
        {
            coef = term;
            var_part = 1;
        }

        // Sort the variable part for consistent comparison
        var_part = sort_product_factors(var_part, symbol_priority);

        // Generate a string key for this variable part
        string key = expr_to_key(var_part, symbol_priority);

        // Combine like terms
        if (coef_map.find(key) != coef_map.end())
        {
            coef_map[key] = coef_map[key] + coef;
        }
        else
        {
            coef_map[key] = coef;
            var_map[key] = var_part;
        }
    }

    // Rebuild the expression
    ex result = 0;
    for (const auto &pair : coef_map)
    {
        string key = pair.first;
        ex coef = pair.second;
        ex var_part = var_map[key];

        // Skip terms with zero coefficient
        if (coef.is_zero())
            continue;

        // Handle constant term (var_part is 1)
        if (var_part.is_equal(1))
        {
            result = result + coef;
        }
        // Terms with coefficient of 1 don't need to be multiplied
        else if (coef.is_equal(1))
        {
            result = result + var_part;
        }
        // General case: coefficient * var_part
        else
        {
            result = result + coef * var_part;
        }
    }

    return result;
}

// Function to extract all symbol names from an expression string
vector<string> extract_symbol_names(const string &expr_str)
{
    vector<string> symbol_names;
    set<string> unique_symbols; // To avoid duplicates

    // Regex pattern to match variables - assumes variables start with a letter
    // followed by alphanumeric characters or underscore
    // This pattern specifically looks for e\d+ and ep\d+ patterns in your example
    regex symbol_pattern("\\b([a-z][a-z0-9_]*)\\b");

    auto words_begin = sregex_iterator(expr_str.begin(), expr_str.end(), symbol_pattern);
    auto words_end = sregex_iterator();

    for (auto i = words_begin; i != words_end; ++i)
    {
        string symbol_name = i->str();

        // Skip numeric literals, constants, and function names
        if (unique_symbols.find(symbol_name) == unique_symbols.end() &&
            symbol_name != "e" && // skip mathematical constant e
            symbol_name != "sin" && symbol_name != "cos" && symbol_name != "tan" &&
            symbol_name != "exp" && symbol_name != "log" &&
            symbol_name != "sqrt" && symbol_name != "pow")
        {

            unique_symbols.insert(symbol_name);
            symbol_names.push_back(symbol_name);
        }
    }

    return symbol_names;
}

// Function to create a symbol table from an expression string
symtab create_symbol_table_from_expression(const string &expr_str)
{
    symtab table;
    vector<string> symbol_names = extract_symbol_names(expr_str);

    cout << "Found symbols: ";
    for (const auto &name : symbol_names)
    {
        // Create the symbol
        symbol sym(name);

        // Add to the symbol table
        table[name] = sym;

        cout << name << " ";
    }
    cout << endl;

    return table;
}

class ExpressionCache
{
private:
    // Cache for expression results
    unordered_map<size_t, ex> cache;
    // Hash function for expressions
    size_t hash_ex(const ex &e) const
    {
        ostringstream oss;
        oss << e;
        string s = oss.str();
        return hash<string>{}(s);
    }

public:
    // Check if expression is in cache
    bool has(const ex &e) const
    {
        size_t key = hash_ex(e);
        return cache.find(key) != cache.end();
    }

    // Get cached result
    ex get(const ex &e) const
    {
        size_t key = hash_ex(e);
        return cache.at(key);
    }

    // Store result in cache
    void store(const ex &e, const ex &result)
    {
        size_t key = hash_ex(e);
        cache[key] = result;
    }

    // Clear cache
    void clear()
    {
        cache.clear();
    }
};

// Global cache instance
ExpressionCache expr_cache;

ex combine_like_terms_advanced_optimized(const ex &e)
{
    // Check if we've already processed this expression
    if (expr_cache.has(e))
    {
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
ex combine_like_terms_advanced_large(const ex &e)
{
    // For very large expressions, we can split them into batches
    if (is_a<add>(e) && e.nops() > 1000)
    {
        cout << "Processing large expression with " << e.nops() << " terms using batched approach..." << endl;

        // Process in batches of 1000 terms
        const size_t BATCH_SIZE = 1000;
        size_t num_batches = (e.nops() + BATCH_SIZE - 1) / BATCH_SIZE; // Ceiling division

        ex result = 0;

        for (size_t batch = 0; batch < num_batches; ++batch)
        {
            size_t start_idx = batch * BATCH_SIZE;
            size_t end_idx = min(start_idx + BATCH_SIZE, e.nops());

            // Create a sub-expression with the current batch of terms
            ex batch_expr = 0;
            for (size_t i = start_idx; i < end_idx; ++i)
            {
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
void print_memory_usage()
{
#ifdef __linux__
    // On Linux, read memory usage from /proc/self/status
    ifstream status("/proc/self/status");
    string line;
    while (getline(status, line))
    {
        if (line.substr(0, 6) == "VmRSS:")
        {
            cout << "Current memory usage: " << line.substr(6) << endl;
            break;
        }
    }
#endif
}

int count_nodes(const GiNaC::ex &expr)
{
    // Count this node
    int count = 1;

    // Add count of all child nodes
    for (size_t i = 0; i < expr.nops(); ++i)
    {
        count += count_nodes(expr.op(i));
    }

    return count;
}

int get_depth(const GiNaC::ex &expr)
{
    if (expr.nops() == 0)
        return 0;

    int max_depth = 0;
    for (size_t i = 0; i < expr.nops(); ++i)
    {
        int child_depth = get_depth(expr.op(i));
        if (child_depth > max_depth)
            max_depth = child_depth;
    }

    return max_depth + 1;
}

void memory_footprint(const ex &expr, set<void *> &visited, size_t &size)
{
    // Get pointer to actual object
    const void *ptr = &expr;

    // If we've seen this pointer before, don't count it again
    if (visited.find((void *)ptr) != visited.end())
        return;

    // Mark as visited
    visited.insert((void *)ptr);

    // Add approximate size for this node (very rough estimate)
    size += sizeof(ex) + 16; // Base size plus some overhead

    // Recurse to children
    for (size_t i = 0; i < expr.nops(); ++i)
    {
        memory_footprint(expr.op(i), visited, size);
    }
}

// Efficient term hashing and comparison without strings
class term_hash
{
private:
    // Store factors in a canonical form
    vector<pair<ex, int>> factors; // Symbol and its power
    ex coefficient;
    mutable size_t cached_hash;
    mutable bool hash_valid;

    // Private method to extract and normalize factors
    void extract_factors(const ex &e)
    {
        factors.clear();
        coefficient = 1;
        hash_valid = false;

        if (is_a<mul>(e))
        {
            for (size_t i = 0; i < e.nops(); ++i)
            {
                if (is_a<numeric>(e.op(i)))
                {
                    coefficient = coefficient * e.op(i);
                }
                else if (is_a<power>(e.op(i)))
                {
                    // Handle powers: x^n
                    ex base = e.op(i).op(0);
                    ex exponent = e.op(i).op(1);

                    if (is_a<symbol>(base) && is_a<numeric>(exponent))
                    {
                        // Store symbol and its power
                        factors.push_back(make_pair(base, ex_to<numeric>(exponent).to_int()));
                    }
                    else
                    {
                        // Other types of powers
                        factors.push_back(make_pair(e.op(i), 1));
                    }
                }
                else if (is_a<symbol>(e.op(i)))
                {
                    // Plain symbol (power 1)
                    factors.push_back(make_pair(e.op(i), 1));
                }
                else
                {
                    // Other types
                    factors.push_back(make_pair(e.op(i), 1));
                }
            }
        }
        else if (is_a<power>(e))
        {
            // Single power term
            ex base = e.op(0);
            ex exponent = e.op(1);

            if (is_a<symbol>(base) && is_a<numeric>(exponent))
            {
                factors.push_back(make_pair(base, ex_to<numeric>(exponent).to_int()));
            }
            else
            {
                factors.push_back(make_pair(e, 1));
            }
        }
        else if (is_a<symbol>(e))
        {
            // Single symbol
            factors.push_back(make_pair(e, 1));
        }
        else if (is_a<numeric>(e))
        {
            // Just a number
            coefficient = e;
        }
        else
        {
            // Other expression type
            factors.push_back(make_pair(e, 1));
        }

        // Sort factors by canonical order (symbol name)
        sort(factors.begin(), factors.end(), [](const pair<ex, int> &a, const pair<ex, int> &b)
             {
                if (is_a<symbol>(a.first) && is_a<symbol>(b.first)) {
                    return ex_to<symbol>(a.first).get_name() < ex_to<symbol>(b.first).get_name();
                } else {
                    // Fall back to string comparison for non-symbols
                    ostringstream oss_a, oss_b;
                    oss_a << a.first;
                    oss_b << b.first;
                    return oss_a.str() < oss_b.str();
                } });
    }

    // Calculate hash based on normalized factors
    size_t calculate_hash() const
    {
        size_t h = 0;

        // Hash the coefficient
        {
            ostringstream oss;
            oss << coefficient;
            string s = oss.str();
            h = hash<string>{}(s);
        }

        // Combine with hash of factors
        for (const auto &factor : factors)
        {
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
    term_hash(const ex &e)
    {
        extract_factors(e);
    }

    // Get the hash - make it const correct
    size_t get_hash() const
    {
        if (!hash_valid)
        {
            cached_hash = calculate_hash();
            hash_valid = true;
        }
        return cached_hash;
    }

    // Compare two terms for equality
    bool equals(const term_hash &other) const
    {
        // Compare coefficient
        if (!coefficient.is_equal(other.coefficient))
            return false;

        // Compare number of factors
        if (factors.size() != other.factors.size())
            return false;

        // Compare each factor
        for (size_t i = 0; i < factors.size(); ++i)
        {
            if (!factors[i].first.is_equal(other.factors[i].first) ||
                factors[i].second != other.factors[i].second)
            {
                return false;
            }
        }

        return true;
    }

    // Get the variable part (without coefficient)
    ex get_var_part() const
    {
        if (factors.empty())
            return 1;

        ex result = 1;
        for (const auto &factor : factors)
        {
            if (factor.second == 1)
            {
                result = result * factor.first;
            }
            else
            {
                result = result * pow(factor.first, factor.second);
            }
        }

        return result;
    }

    // Get the coefficient
    ex get_coefficient() const
    {
        return coefficient;
    }
};

// Fast combine like terms using our optimized term hashing
ex fast_combine_like_terms(const ex &e)
{
    // If not a sum, return as is
    if (!is_a<add>(e))
        return e;

    // Use a hash map for fast lookup
    unordered_map<size_t, ex> coef_map; // Maps hash to coefficient
    unordered_map<size_t, ex> var_map;  // Maps hash to variable part
    vector<term_hash> term_hashes;      // Keep track of all hashes

    // Process each term in the sum
    for (size_t i = 0; i < e.nops(); ++i)
    {
        ex term = e.op(i);
        term_hash th(term);

        size_t hash = th.get_hash();
        ex var_part = th.get_var_part();
        ex coef = th.get_coefficient();

        // Check if we already have this term pattern
        bool found = false;
        for (const auto &existing_th : term_hashes)
        {
            if (existing_th.get_hash() == hash && existing_th.equals(th))
            {
                // This is a match - we found a like term
                hash = existing_th.get_hash(); // Use the existing hash
                found = true;
                break;
            }
        }

        if (!found)
        {
            // This is a new term pattern
            term_hashes.push_back(th);
            coef_map[hash] = coef;
            var_map[hash] = var_part;
        }
        else
        {
            // This matches an existing term pattern
            coef_map[hash] = coef_map[hash] + coef;
        }
    }

    // Rebuild the expression
    ex result = 0;
    for (const auto &th : term_hashes)
    {
        size_t hash = th.get_hash();
        ex coef = coef_map[hash];
        ex var_part = var_map[hash];

        // Skip terms with zero coefficient
        if (coef.is_zero())
            continue;

        // Handle constant term (var_part is 1)
        if (var_part.is_equal(1))
        {
            result = result + coef;
        }
        // Terms with coefficient of 1 don't need to be multiplied
        else if (coef.is_equal(1))
        {
            result = result + var_part;
        }
        // General case: coefficient * var_part
        else
        {
            result = result + coef * var_part;
        }
    }

    return result;
}

// Helper to extract coefficient from an expression
void extract_coefficient(const ex& expr, double& coeff, ex& sym_part) {
    sym_part = expr;
    coeff = 1.0;
    
    if (is_a<mul>(expr)) {
        double numeric_product = 1.0;
        vector<ex> non_numeric_factors;
        
        // Separate numeric and symbolic factors
        for (size_t i = 0; i < expr.nops(); ++i) {
            if (is_a<numeric>(expr.op(i))) {
                numeric_product *= ex_to<numeric>(expr.op(i)).to_double();
            } else {
                non_numeric_factors.push_back(expr.op(i));
            }
        }
        
        // Set the coefficient
        coeff = numeric_product;
        
        // Reconstruct the symbolic part
        if (!non_numeric_factors.empty()) {
            sym_part = non_numeric_factors[0];
            for (size_t i = 1; i < non_numeric_factors.size(); ++i) {
                sym_part = sym_part * non_numeric_factors[i];
            }
        } else {
            sym_part = 1;  // No symbolic part, just a constant
        }
    }
}

// Check if an expression represents a squared symbol (s_i^2 or s_i*s_i)
bool is_squared_symbol(const ex& expr, string& base_symbol) {
    // Check for explicit power: s_i^2
    if (is_a<power>(expr) && is_a<symbol>(expr.op(0)) && 
        is_a<numeric>(expr.op(1)) && expr.op(1).is_equal(numeric(2))) {
        ostringstream ss;
        ss << expr.op(0);
        base_symbol = ss.str();
        return true;
    }
    
    // Check for product of same symbol: s_i*s_i
    if (is_a<mul>(expr)) {
        map<string, int> symbol_counts;
        
        for (size_t i = 0; i < expr.nops(); ++i) {
            if (is_a<symbol>(expr.op(i))) {
                ostringstream ss;
                ss << expr.op(i);
                symbol_counts[ss.str()]++;
            }
        }
        
        // Check if any symbol appears exactly twice
        for (const auto& entry : symbol_counts) {
            if (entry.second == 2) {
                base_symbol = entry.first;
                return true;
            }
        }
    }
    
    return false;
}

// Get string representation of a symbolic expression
string symbol_to_string(const ex& expr) {
    ostringstream ss;
    ss << expr;
    return ss.str();
}

// Generate a key for the symbol map
string make_key(const ex& expr) {
    string base_symbol;
    
    // Check if it's a squared symbol
    if (is_squared_symbol(expr, base_symbol)) {
        return base_symbol + "^2";  // Use consistent format
    }
    
    // Regular symbol or other expression
    return symbol_to_string(expr);
}

// Main function to expand sum of squares
string optimized_expand_sum_of_squares(const vector<ex>& terms, bool square_the_bases = false, int n = 1) {
    unordered_map<string, double> coefficient_map;
    unordered_map<string, ex> symbol_map;
    double constant_term = 0.0;
    
    // Process each term in the input
    for (const auto& term : terms) {
        ex current_term = square_the_bases ? pow(term, 2) : term;
        
        // Handle numeric terms
        if (is_a<numeric>(current_term)) {
            constant_term += ex_to<numeric>(current_term).to_double();
            continue;
        }
        
        // Handle squared expressions: (a + b + c)^2
        if (is_a<power>(current_term) && is_a<numeric>(current_term.op(1)) && 
            current_term.op(1).is_equal(numeric(2))) {
            
            ex base = current_term.op(0);
            
            // If base is a sum
            if (is_a<add>(base)) {
                ex new_base = combine_like_terms_advanced_optimized(base);
                // cout << "base is a sum: " << new_base << endl;
                // Extract terms from the sum
                vector<pair<double, ex>> symbolic_terms;
                double constant_in_sum = 0.0;
                
                for (size_t i = 0; i < new_base.nops(); ++i) {
                    const ex& term_i = new_base.op(i);
                    
                    if (is_a<numeric>(term_i)) {
                        constant_in_sum += ex_to<numeric>(term_i).to_double();
                    } else {
                        double coeff = 1.0;
                        ex sym_part = term_i;
                        extract_coefficient(term_i, coeff, sym_part);
                        // cout << "Extracted coefficient: " << coeff << ", symbolic part: " << sym_part << endl;
                        symbolic_terms.push_back(make_pair(coeff, sym_part));
                    }
                }
                
                // Add constant^2 term
                constant_term += constant_in_sum * constant_in_sum;
                
                // Process squared terms: (coef*s_i)^2
                for (const auto& term_pair : symbolic_terms) {
                    double coeff = term_pair.first;
                    const ex& sym = term_pair.second;
                    
                    // Add c_i^2 * s_i^2 term
                    string base_symbol = symbol_to_string(sym);
                    string squared_key = base_symbol + "^2";
                    
                    coefficient_map[squared_key] += coeff * coeff;
                    symbol_map[squared_key] = pow(sym, 2);
                    
                    // Add 2*constant*c_i*s_i terms
                    if (fabs(constant_in_sum) > 1e-15) {
                        string linear_key = base_symbol;
                        coefficient_map[linear_key] += 2.0 * constant_in_sum * coeff;
                        symbol_map[linear_key] = sym;
                    }
                }
                
                // Skip cross terms as instructed
            } 
            // Handle single term squared: (term)^2
            else {
                double coeff = 1.0;
                ex sym_part = base;
                extract_coefficient(base, coeff, sym_part);
                
                // Square the coefficient
                double squared_coeff = coeff * coeff;
                
                // Add to the appropriate map
                string base_symbol = symbol_to_string(sym_part);
                string squared_key = base_symbol + "^2";
                
                coefficient_map[squared_key] += squared_coeff;
                symbol_map[squared_key] = pow(sym_part, 2);
            }
        } 
        // Handle sum expressions
        else if (is_a<add>(current_term)) {
            for (size_t i = 0; i < current_term.nops(); ++i) {
                const ex& add_term = current_term.op(i);
                
                if (is_a<numeric>(add_term)) {
                    constant_term += ex_to<numeric>(add_term).to_double();
                } else {
                    double coeff = 1.0;
                    ex sym_part = add_term;
                    extract_coefficient(add_term, coeff, sym_part);
                    
                    // Check if it's a squared term
                    string base_symbol;
                    if (is_squared_symbol(sym_part, base_symbol)) {
                        string squared_key = base_symbol + "^2";
                        coefficient_map[squared_key] += coeff;
                        symbol_map[squared_key] = pow(symbol(base_symbol), 2);
                    } else {
                        string key = symbol_to_string(sym_part);
                        coefficient_map[key] += coeff;
                        symbol_map[key] = sym_part;
                    }
                }
            }
        } 
        // Handle other terms
        else {
            double coeff = 1.0;
            ex sym_part = current_term;
            extract_coefficient(current_term, coeff, sym_part);
            
            // Check if it's a squared term
            string base_symbol;
            if (is_squared_symbol(sym_part, base_symbol)) {
                string squared_key = base_symbol + "^2";
                coefficient_map[squared_key] += coeff;
                symbol_map[squared_key] = pow(symbol(base_symbol), 2);
            } else {
                string key = symbol_to_string(sym_part);
                coefficient_map[key] += coeff;
                symbol_map[key] = sym_part;
            }
        }
    }
    
    // Convert to vector for sorting
    vector<pair<string, double>> sorted_terms;
    
    // Add squared terms first
    for (const auto& entry : coefficient_map) {
        if (fabs(entry.second) > 1e-15 && entry.first.find("^2") != string::npos) {
            sorted_terms.push_back(make_pair(entry.first, entry.second / n));
        }
    }
    
    // Add linear terms
    for (const auto& entry : coefficient_map) {
        if (fabs(entry.second) > 1e-15 && entry.first.find("^2") == string::npos) {
            sorted_terms.push_back(make_pair(entry.first, entry.second / n));
        }
    }
    
    // Sort terms by symbol name
    sort(sorted_terms.begin(), sorted_terms.end(), 
         [](const pair<string, double>& a, const pair<string, double>& b) {
             // Get base symbol names (without ^2)
             string base_a = a.first;
             if (base_a.find("^2") != string::npos) {
                 base_a = base_a.substr(0, base_a.length() - 2);
             }
             
             string base_b = b.first;
             if (base_b.find("^2") != string::npos) {
                 base_b = base_b.substr(0, base_b.length() - 2);
             }
             
             // Sort by base symbol
             if (base_a != base_b) {
                 return base_a < base_b;
             }
             
             // For same base, squared terms come first
             return a.first.find("^2") != string::npos;
         });
    
    // Build output string
    ostringstream oss;
    bool first_term = true;
    
    // Add symbolic terms
    for (const auto& term : sorted_terms) {
        double coeff = term.second;
        string symbol_key = term.first;
        
        if (first_term) {
            if (coeff < 0) {
                oss << "-";
                coeff = -coeff;
            }
            
            if (fabs(coeff - 1.0) < 1e-15) {
                oss << symbol_key;
            } else {
                oss << coeff << "*" << symbol_key;
            }
            first_term = false;
        } else {
            if (coeff < 0) {
                oss << " - ";
                coeff = -coeff;
            } else {
                oss << " + ";
            }
            
            if (fabs(coeff - 1.0) < 1e-15) {
                oss << symbol_key;
            } else {
                oss << coeff << "*" << symbol_key;
            }
        }
    }
    
    // Add constant term
    if (fabs(constant_term) > 1e-15) {
        double scaled_constant = constant_term / n;
        
        if (first_term) {
            oss << scaled_constant;
        } else {
            if (scaled_constant < 0) {
                oss << " - " << -scaled_constant;
            } else {
                oss << " + " << scaled_constant;
            }
        }
    }
    
    // Handle empty result
    if (oss.str().empty()) {
        return "0";
    }
    
    return oss.str();
}

string square_and_expand(const matrix &input_matrix, int n = 1)
{
    size_t num_rows = input_matrix.rows();
    size_t num_columns = input_matrix.cols();

    if (num_rows * num_columns == 0)
    {
        return "0"; // Return zero for empty matrix
    }

    vector<ex> terms;
    for (size_t k = 0; k < num_rows; ++k)
    {
        terms.push_back(input_matrix(k, 0) * input_matrix(k, 0));
    }

    string final_result = optimized_expand_sum_of_squares(terms, false, n);

    return final_result;
}

symbol create_symbol(const string &base, int i)
{
    return symbol(base + to_string(i));
}

MatrixXd ginacToEigen(const matrix &m)
{
    unsigned rows = m.rows();
    unsigned cols = m.cols();
    MatrixXd eigen_mat(rows, cols);

    for (unsigned i = 0; i < rows; ++i)
    {
        for (unsigned j = 0; j < cols; ++j)
        {
            // Extract numerical value from GiNaC expression
            ex element = m(i, j);
            eigen_mat(i, j) = ex_to<numeric>(element).to_double();
        }
    }

    return eigen_mat;
}

matrix eigenToGinac(const MatrixXd &eigen_mat)
{
    unsigned rows = eigen_mat.rows();
    unsigned cols = eigen_mat.cols();
    matrix ginac_mat(rows, cols);

    for (unsigned i = 0; i < rows; ++i)
    {
        for (unsigned j = 0; j < cols; ++j)
        {
            ginac_mat(i, j) = numeric(eigen_mat(i, j));
        }
    }

    return ginac_mat;
}

matrix get_row(const matrix &mat, int row_index)
{
    int num_cols = mat.cols();
    matrix row(1, num_cols);

    for (int j = 0; j < num_cols; j++)
    {
        row(0, j) = mat(row_index, j);
    }

    return row;
}

string ex_to_string(const ex &expr, bool python_style = true)
{
    ostringstream oss;

    if (python_style)
    {
        // Convert to Python-style syntax
        string temp_str;
        ostringstream temp_oss;
        temp_oss << expr;
        temp_str = temp_oss.str();

        // Replace GiNaC syntax with Python syntax
        // This is a simple example - you might need more sophisticated replacements
        size_t pos = 0;
        while ((pos = temp_str.find("^", pos)) != string::npos)
        {
            temp_str.replace(pos, 1, "**");
            pos += 2;
        }

        return temp_str;
    }
    else
    {
        // Use default GiNaC formatting
        oss << expr;
        return oss.str();
    }
}

py::list matrix_to_python_list(const matrix &param)
{
    py::list result;

    for (unsigned i = 0; i < param.rows(); i++)
    {
        py::list row;
        for (unsigned j = 0; j < param.cols(); j++)
        {
            // Convert each expression to string
            string expr_str = ex_to_string(param(i, j));
            row.append(expr_str);
        }
        result.append(row);
    }

    return result;
}

// Utilities
// Function to multiply X_test and param and return simplified results
py::list multiply_matrices(const py::list &X_test_py, const py::list &param_py)
{
    // Convert Python lists to GiNaC matrices
    auto start_time = high_resolution_clock::now();
    int X_rows = py::len(X_test_py);
    int X_cols = py::len(py::list(X_test_py[0]));

    matrix X_test(X_rows, X_cols);
    for (int i = 0; i < X_rows; i++)
    {
        py::list row = X_test_py[i];
        for (int j = 0; j < X_cols; j++)
        {
            // Parse expression from string
            parser reader;
            string expr_str = py::str(row[j]);
            X_test(i, j) = reader(expr_str);
        }
    }

    int param_rows = py::len(param_py);
    matrix param(param_rows, 1);
    for (int i = 0; i < param_rows; i++)
    {
        parser reader;
        string expr_str = py::str(param_py[i]);
        param(i, 0) = reader(expr_str);
    }

    // Perform matrix multiplication
    matrix result(X_rows, 1);
    for (int i = 0; i < X_rows; i++)
    {
        ex sum = 0;
        for (int j = 0; j < X_cols; j++)
        {
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
    for (int i = 0; i < result.rows(); i++)
    {
        ostringstream oss;
        oss << result(i, 0);
        result_py.append(oss.str());
    }

    return result_py;
}

py::list abstract_loss(const py::list &X_test_py, const py::list &y_test_py, const py::list &param_py)
{
    // Convert Python lists to GiNaC matrices
    auto start_time = high_resolution_clock::now();
    int X_rows = py::len(X_test_py);
    int X_cols = py::len(py::list(X_test_py[0]));

    matrix X_test(X_rows, X_cols);
    for (int i = 0; i < X_rows; i++)
    {
        py::list row = X_test_py[i];
        for (int j = 0; j < X_cols; j++)
        {
            // Parse expression from string
            parser reader;
            string expr_str = py::str(row[j]);
            X_test(i, j) = reader(expr_str);
        }
    }

    cout << "X: " << X_test.rows() << "*" << X_test.cols() << endl;

    matrix y_test(X_rows, 1);
    for (int i = 0; i < X_rows; i++)
    {
        parser reader;
        string expr_str = py::str(y_test_py[i]);
        y_test(i, 0) = reader(expr_str);
    }

    cout << "y: " << y_test.rows() << "*" << y_test.cols() << endl;

    int param_rows = py::len(param_py);
    matrix param(param_rows, 1);
    for (int i = 0; i < param_rows; i++)
    {
        parser reader;
        string expr_str = py::str(param_py[i]);
        param(i, 0) = reader(expr_str);
    }
    cout << "param: " << param.rows() << "*" << param.cols() << endl;

    // test_preds = X_test * param
    matrix predictions(X_rows, 1);
    for (int i = 0; i < X_rows; i++)
    {
        ex sum = 0;
        for (int j = 0; j < X_cols; j++)
        {
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
    for (int i = 0; i < X_rows; i++)
    {
        diff(i, 0) = predictions(i, 0) - y_test(i, 0);
    }
    end_time = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(end_time - start_time);
    cout << "Time for difference: " << duration.count() << " ms" << endl;

    start_time = high_resolution_clock::now();
    // Calculate sum of squared differences
    string sum_squared = square_and_expand(diff, X_rows);
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

    // cout << "Number of nodes in sum_squared: " << count_nodes(sum_squared) << endl;
    // cout << "Number of terms in sum_squared: " << sum_squared.nops() << endl;
    // cout << "Depth of sum_squared: " << get_depth(sum_squared) << endl;
    print_memory_usage();

    // start_time = high_resolution_clock::now();
    // sum_squared = combine_like_terms_advanced_optimized(sum_squared);
    // end_time = high_resolution_clock::now();
    // duration = duration_cast<milliseconds>(end_time - start_time);
    // cout << "Time for combining like terms: " << duration.count() << " ms" << endl;

    // cout << "Number of nodes in sum_squared after expansion: " << count_nodes(sum_squared) << endl;
    // cout << "Number of terms in sum_squared after expansion: " << sum_squared.nops() << endl;
    // cout << "Depth of sum_squared after expansion: " << get_depth(sum_squared) << endl;
    // print_memory_usage();

    // sum_squared = sum_squared / numeric(X_rows);

    // Convert result back to Python list
    py::list result_py;
    for (int i = 0; i < 1; i++)
    {
        // ostringstream oss;
        // oss << sum_squared;
        result_py.append(sum_squared);
    }

    return result_py;
}

py::str expand_single(const py::str &expression)
{
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

py::str expand_chunk(const py::list &expressions)
{
    // Parse the expression from the list
    parser reader;
    int rows = py::len(expressions);
    ex chunk_sum = numeric(0);

    vector<ex> exprs(rows);
    for (int i = 0; i < rows; i++)
    {
        exprs[i] = reader(py::str(expressions[i]));
    }

    // Expand the expressions
    for (const ex &expr : exprs)
    {
        try
        {
            chunk_sum = chunk_sum + expr.expand();
        }
        catch (const exception &e)
        {
            return "__CPP_CHUNK_ERROR__";
        }
        catch (...)
        {
            return "__CPP_CHUNK_UNKNOWN_ERROR__";
        }
    }

    chunk_sum = combine_like_terms_advanced_optimized(chunk_sum);

    ostringstream oss;
    oss << chunk_sum;
    return py::str(oss.str());
}

py::list expand_matrix(const py::list &expressions, const py::int_ &row, const py::int_ &col)
{
    // Parse the expression from the string
    parser reader;
    int rows = row.cast<int>();
    int cols = col.cast<int>();

    vector<ex> exprs(rows);
    for (int i = 0; i < rows; i++)
    {
        exprs[i] = reader(py::str(expressions[i]));
    }

    // Expand the expressions
    for (ex &expr : exprs)
    {
        expr = expr.expand();
    }

    // Convert back to python list and return
    py::list result_py;
    for (int i = 0; i < rows * cols; i++)
    {
        ostringstream oss;
        oss << exprs[i];
        result_py.append(oss.str());
    }

    return result_py;
}

py::str combine_like_terms(const py::str &expression)
{
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

py::list zorro(const py::list &X_train, const py::list &y_train, const py::list &X_max, const py::list &X_min, const py::float_ &lr_py, const py::float_ &reg_py)
{
    auto full_start_time = high_resolution_clock::now();
    int X_rows = py::len(X_train);
    int X_cols = py::len(py::list(X_train[0]));
    int curr_id = 0;
    float lr = lr_py.cast<float>();
    float reg = reg_py.cast<float>();

    vector<symbol> e_symbols;
    vector<symbol> k_symbols;
    vector<symbol> ep_symbols;

    matrix XS(X_rows, X_cols);
    matrix XR(X_rows, X_cols);
    matrix yS(X_rows, 1);
    matrix yR(X_rows, 1);
    for (int i = 0; i < X_rows; i++)
    {
        parser reader_y;
        string expr_str_y = py::str(y_train[i]);
        yS(i, 0) = numeric(0);
        yR(i, 0) = reader_y(expr_str_y);

        py::list row_max = X_max[i];
        py::list row_min = X_min[i];
        py::list row = X_train[i];
        for (int j = 0; j < X_cols; j++)
        {
            float max_val = py::cast<float>(row_max[j]);
            float min_val = py::cast<float>(row_min[j]);

            if (max_val != min_val)
            {
                float xmean = (max_val + min_val) / 2;
                float xradius = (max_val - min_val) / 2;
                XR(i, j) = xmean;
                e_symbols.push_back(symbol("e" + to_string(curr_id)));
                XS(i, j) = xradius * e_symbols[curr_id];
                curr_id++;
            }
            else
            {
                float xval = py::cast<float>(row[j]);
                XR(i, j) = xval;
                XS(i, j) = numeric(0);
            }
        }
    }

    matrix XS_T = XS.transpose();
    matrix XR_T = XR.transpose();

    matrix identity(X_cols, X_cols, lst());
    for (unsigned i = 0; i < X_cols; i++)
    {
        identity(i, i) = 1;
    }
    matrix common_inv = ex_to<matrix>((XR_T.mul(XR) + reg * X_rows * identity).evalm());
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

    matrix wR = ex_to<matrix>((common_inv * (XR_T)*yR).evalm());
    matrix wS_data = ex_to<matrix>((common_inv * (((XS_T)*XR + (XR_T)*XS) * wR - (XS_T)*yR - (XR_T)*yS)).evalm());
    matrix wS_non_data(VT.cols(), 1);
    for (int i = 0; i < X_cols; i++)
    {
        k_symbols.push_back(symbol("k" + to_string(i)));
        ep_symbols.push_back(symbol("ep" + to_string(i)));
        wS_non_data = ex_to<matrix>((wS_non_data + k_symbols[i] * ep_symbols[i] * get_row(VT, i).transpose()).evalm());
    }

    matrix eigenvalues(1, sigma.cols());
    for (unsigned j = 0; j < sigma.cols(); j++)
    {
        eigenvalues(0, j) = 1 - 2 * lr * reg - 2 * lr * sigma(0, j) / X_rows;
        assert(eigenvalues(0, j) >= 0);
        assert(eigenvalues(0, j) <= 1);
    }

    matrix wS = ex_to<matrix>((wS_non_data + wS_data).evalm());
    matrix w = ex_to<matrix>((wS + wR).evalm());

    auto start_time = high_resolution_clock::now();
    ex scalar = (numeric(-2.0) * lr) / X_rows;
    matrix matrix_part = ex_to<matrix>((((XS_T)*XR + (XR_T)*XS + (XS_T)*XS) * wS + (XS_T)*XS * wR - (XS_T)*yS).evalm());
    matrix w_prime = ex_to<matrix>(((scalar * matrix_part).evalm().expand()));
    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end_time - start_time);
    cout << "W# (abstract gradient descent?) computation time: " << duration.count() << " ms" << endl;

    start_time = high_resolution_clock::now();
    matrix w_prime_projected = ex_to<matrix>((VT * w_prime).evalm());
    end_time = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(end_time - start_time);
    cout << "Projection computation time: " << duration.count() << " ms" << endl;

    vector<ex> eqs;

    for (int d = 0; d < X_cols; d++)
    {
        ex eq1 = (1 - abs(eigenvalues(0, d))) * k_symbols[d];

        // Use a vector for coefficients
        vector<ex> coef_for_k(X_cols, 0);
        ex const_coef = 0;

        // Process all terms in w_prime_projected[d]
        if (is_a<add>(w_prime_projected[d]))
        {
            for (size_t i = 0; i < w_prime_projected[d].nops(); i++)
            {
                ex term = w_prime_projected[d].op(i);
                bool found_k = false;

                // Find which k symbol this term contains
                for (int j = 0; j < X_cols; j++)
                {
                    if (term.has(k_symbols[j]))
                    {
                        // Extract the numeric coefficient
                        ex numeric_coeff = 0;

                        if (is_a<mul>(term))
                        {
                            // Find the numeric factor
                            for (size_t idx = 0; idx < term.nops(); idx++)
                            {
                                if (is_a<numeric>(term.op(idx)))
                                {
                                    numeric_coeff = term.op(idx);
                                    break;
                                }
                            }
                        }
                        else if (term == k_symbols[j])
                        {
                            numeric_coeff = 1;
                        }

                        if (!numeric_coeff.is_zero())
                        {
                            // Take absolute value immediately (like Python does with abs(arg.args[0]))
                            // and divide by 2
                            coef_for_k[j] += abs(numeric_coeff);
                        }

                        found_k = true;
                        break;
                    }
                }

                // If no k symbol found, add to constant term
                if (!found_k)
                {
                    if (is_a<numeric>(term))
                    {
                        const_coef += abs(term);
                    }
                    else if (is_a<mul>(term))
                    {
                        // Find the numeric factor
                        for (size_t idx = 0; idx < term.nops(); idx++)
                        {
                            if (is_a<numeric>(term.op(idx)))
                            {
                                const_coef += abs(term.op(idx));
                                break;
                            }
                        }
                    }
                }
            }
        }

        // Build eq2 directly (no need to take abs again)
        ex eq2 = const_coef;
        for (int i = 0; i < X_cols; i++)
        {
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
    for (size_t i = 0; i < eqs.size(); i++)
    {
        cout << "Equation " << i << ": " << eqs[i] << endl;
    }

    // Solve the linear system of equations
    matrix A(eqs.size(), k_symbols.size()); // Changed dimensions
    matrix b(eqs.size(), 1);                // Changed dimensions

    // Extract coefficients for linear system Ax = b
    for (unsigned i = 0; i < eqs.size(); i++)
    {
        ex eq = eqs[i];
        ex lhs = eq.lhs();
        ex rhs = eq.rhs();

        // Extract coefficients from left side
        for (unsigned j = 0; j < k_symbols.size(); j++)
        {
            A(i, j) = lhs.coeff(k_symbols[j], 1);
        }

        // Extract coefficients from right side and build constant term
        ex const_term = rhs;
        for (unsigned j = 0; j < k_symbols.size(); j++)
        {
            A(i, j) = A(i, j) - rhs.coeff(k_symbols[j], 1);
            const_term = const_term.coeff(k_symbols[j], 0);
        }
        b(i, 0) = const_term;
    }

    // Convert to a system of equations and variables for lsolve
    lst equations;
    lst variables;

    // Build equations from the matrix A and vector b
    for (unsigned i = 0; i < eqs.size(); i++)
    { // Changed to use eqs.size()
        ex eq = 0;
        for (unsigned j = 0; j < k_symbols.size(); j++)
        {
            eq += A(i, j) * k_symbols[j];
        }
        equations.append(eq == b(i, 0));
    }

    // Build variables list
    for (unsigned i = 0; i < k_symbols.size(); i++)
    {
        variables.append(k_symbols[i]);
    }

    // Solve using lsolve
    ex solution = lsolve(equations, variables);

    vector<pair<symbol, ex>> result;

    // Extract solutions from the result
    if (is_a<lst>(solution))
    {
        lst sol_list = ex_to<lst>(solution);

        for (unsigned i = 0; i < sol_list.nops(); i++)
        {
            if (is_a<relational>(sol_list.op(i)))
            {
                relational rel = ex_to<relational>(sol_list.op(i));

                if (is_a<symbol>(rel.lhs()))
                {
                    symbol var = ex_to<symbol>(rel.lhs());
                    result.push_back(make_pair(var, rel.rhs()));
                }
            }
        }
    }

    // Assert all values are non-negative
    for (const auto &kv : result)
    {
        // Convert to numeric and check
        ex val = kv.second.evalf();
        if (is_a<numeric>(val))
        {
            assert(ex_to<numeric>(val).to_double() >= 0);
        }
        else
        {
            cerr << "Warning: Could not evaluate " << kv.first << " to numeric value" << endl;
        }
    }

    // Print result
    cout << "Result:" << endl;
    for (const auto &kv : result)
    {
        cout << kv.first << " = " << kv.second << endl;
    }

    // Create substitution map
    exmap subs_map;
    for (const auto &kv : result)
    {
        subs_map[kv.first] = kv.second;
    }

    // Substitute into wS and add to wR
    // First, create a copy of wS to avoid modifying original
    matrix wS_substituted(wS.rows(), wS.cols());
    for (unsigned i = 0; i < wS.rows(); i++)
    {
        for (unsigned j = 0; j < wS.cols(); j++)
        {
            wS_substituted(i, j) = wS(i, j).subs(subs_map);
        }
    }

    // Now create param matrix by adding wR and substituted wS
    matrix param(wR.rows(), wR.cols());
    for (unsigned i = 0; i < wR.rows(); i++)
    {
        for (unsigned j = 0; j < wR.cols(); j++)
        {
            param(i, j) = wR(i, j) + wS_substituted(i, j);
        }
    }

    auto full_end_time = high_resolution_clock::now();
    auto full_duration = duration_cast<milliseconds>(full_end_time - full_start_time);
    cout << "Full pipeline time: " << full_duration.count() << " ms" << endl;

    return matrix_to_python_list(param);
}

PYBIND11_MODULE(ginac_module, m)
{
    m.doc() = "GiNaC operations module";
    m.def("multiply_matrices", &multiply_matrices, "Multiply X_test and param");
    m.def("abstract_loss", &abstract_loss, "Full pipeline for abstract loss");
    m.def("expand_single", &expand_single, "Expand one expression");
    m.def("expand_chunk", &expand_chunk, "Expand a chunk of expressions");
    m.def("expand_matrix", &expand_matrix, "Expand a matrix of expressions");
    m.def("combine_like_terms", &combine_like_terms, "Do combine like terms in one expression");
    m.def("zorro", &zorro, "ZORRO training pipeline with GiNaC");
}