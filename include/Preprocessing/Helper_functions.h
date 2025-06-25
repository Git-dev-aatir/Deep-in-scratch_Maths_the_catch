#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <regex>
#include <algorithm>

/**
 * @brief Get the median of a vector within a specified range.
 * 
 * @tparam T Data type.
 * @param v Input vector.
 * @param start Start index.
 * @param end End index.
 * @return T Median value.
 */
template<typename T>
inline double get_median (const std::vector<T>& v, size_t start, size_t end) {
    size_t len = end - start + 1;
    if (len == 0) return T(); // return default-constructed T if empty

    size_t mid = start + len / 2;
    if (len & 1)
        return static_cast<double>(v[mid]);
    else
        return ((static_cast<double>(v[mid - 1] + v[mid])) / 2.0);
};

/**
 * @brief Checks for missing values in floating point types.
 * 
 * @tparam T Floating point type.
 * @param value Value to check.
 * @return true if missing (NaN), false otherwise.
 */
template<typename T>
inline typename std::enable_if<std::is_floating_point<T>::value, bool>::type
isMissing(const T& value) {
    return std::isnan(value);
}

/**
 * @brief Checks for missing values in integral types.
 * 
 * @tparam T Integral type.
 * @param value Value to check.
 * @return true if missing (equal to min limit), false otherwise.
 */
template<typename T>
inline typename std::enable_if<std::is_integral<T>::value, bool>::type
isMissing(const T& value) {
    return value == std::numeric_limits<T>::min();
}


/**
 * @brief Checks for missing values in string type.
 * 
 * @param value String to check.
 * @return true if missing (empty string), false otherwise.
 */
inline bool isMissing(const std::string& value) {
    return value.empty();
}

/**
 * @brief Trim whitespace from the start of a string (in place).
 * @param s The string to trim.
 */
inline void ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(),
        [](unsigned char ch) { return !isspace(ch); }));
}

/**
 * @brief Trim whitespace from the end of a string (in place).
 * @param s The string to trim.
 */
inline void rtrim(std::string &s) {
    s.erase(find_if(s.rbegin(), s.rend(),
        [](unsigned char ch) { return !isspace(ch); }).base(), s.end());
}

/**
 * @brief Trim whitespace from both ends of a string (in place).
 * @param s The string to trim.
 */
inline void trim(std::string &s) {
    ltrim(s);
    rtrim(s);
}

/**
 * @brief Split a string into tokens based on a delimiter.
 * @param line The string to split.
 * @param delimiter The character used as delimiter.
 * @param multiple_spaces If true, treats multiple spaces as one delimiter.
 * @return A vector of string tokens.
 */
inline std::vector<std::string> split(const std::string &line, char delimiter, bool multiple_spaces) {
    std::vector<std::string> tokens;

    if (multiple_spaces && delimiter == ' ') {
        std::regex re(R"(\s+)");
        std::sregex_token_iterator it(line.begin(), line.end(), re, -1);
        std::sregex_token_iterator reg_end;

        for (; it != reg_end; ++it) {
            if (!it->str().empty()) {
                tokens.push_back(it->str());
            }
        }
    } else {
        std::string token;
        std::stringstream tokenStream(line);
        while (getline(tokenStream, token, delimiter)) {
            tokens.push_back(token);
        }
    }

    return tokens;
}

/**
 * @brief Parse a token string to the target type.
 * @tparam T Target type.
 * @param token The string token to parse.
 * @return Parsed value of type T.
 */
template<typename T>
inline T parseToken(const std::string &token);

template<>
inline int parseToken<int>(const std::string &token) {
    try {
        return stoi(token);
    } catch (...) {
        std::cerr << "Warning: Non-int value \"" << token 
        << "\" encountered. Storing 0." << std::endl;
        return 0;
    }
}

template<>
inline double parseToken<double>(const std::string &token) {
    try {
        return stod(token);
    } catch (...) {
        std::cerr << "Warning: Non-double value \"" << token 
        << "\" encountered. Storing 0.0." << std::endl;
        return 0.0;
    }
}

template<>
inline std::string parseToken<std::string>(const std::string &token) {
    return token;
}