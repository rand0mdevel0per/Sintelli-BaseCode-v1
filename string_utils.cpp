/**
 * @file string_utils.cpp
 * @brief Implementation of string utility functions
 * 
 * This file contains the implementation of Python-like string manipulation functions
 * including split, strip, join, and other common operations.
 * 
 * @author String Utils Library Team
 * @version 1.0.0
 * @date 2025
 * @copyright MIT License
 */

#include "string_utils.h"
#include <algorithm>
#include <cctype>
#include <sstream>

namespace StringUtils {

std::vector<std::string> split(const std::string& str, char delimiter, int max_split) {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;
    int split_count = 0;
    
    while (std::getline(ss, token, delimiter)) {
        if (max_split != -1 && split_count >= max_split) {
            // Add the remaining string as the last token
            std::string remaining;
            std::getline(ss, remaining);
            if (!remaining.empty()) {
                token += delimiter + remaining;
            }
            tokens.push_back(token);
            break;
        }
        tokens.push_back(token);
        split_count++;
    }
    
    return tokens;
}

std::vector<std::string> split(const std::string& str, const std::string& delimiter, int max_split) {
    std::vector<std::string> tokens;
    if (delimiter.empty()) {
        // If delimiter is empty, split into individual characters
        for (char c : str) {
            tokens.push_back(std::string(1, c));
        }
        return tokens;
    }
    
    size_t start = 0;
    size_t end = 0;
    int split_count = 0;
    
    while ((end = str.find(delimiter, start)) != std::string::npos) {
        if (max_split != -1 && split_count >= max_split) {
            break;
        }
        
        tokens.push_back(str.substr(start, end - start));
        start = end + delimiter.length();
        split_count++;
    }
    
    // Add the remaining part
    tokens.push_back(str.substr(start));
    
    return tokens;
}

std::string strip(const std::string& str) {
    return lstrip(rstrip(str));
}

std::string lstrip(const std::string& str) {
    auto start = std::find_if(str.begin(), str.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    });
    return std::string(start, str.end());
}

std::string rstrip(const std::string& str) {
    auto end = std::find_if(str.rbegin(), str.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base();
    return std::string(str.begin(), end);
}

std::string strip(const std::string& str, const std::string& chars) {
    auto start = str.find_first_not_of(chars);
    if (start == std::string::npos) {
        return "";
    }
    
    auto end = str.find_last_not_of(chars);
    return str.substr(start, end - start + 1);
}

std::string join(const std::vector<std::string>& strings, const std::string& delimiter) {
    if (strings.empty()) {
        return "";
    }
    
    std::string result = strings[0];
    for (size_t i = 1; i < strings.size(); ++i) {
        result += delimiter + strings[i];
    }
    return result;
}

bool startswith(const std::string& str, const std::string& prefix) {
    if (prefix.length() > str.length()) {
        return false;
    }
    return str.compare(0, prefix.length(), prefix) == 0;
}

bool endswith(const std::string& str, const std::string& suffix) {
    if (suffix.length() > str.length()) {
        return false;
    }
    return str.compare(str.length() - suffix.length(), suffix.length(), suffix) == 0;
}

std::string lower(const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(), 
                   [](unsigned char c) { return std::tolower(c); });
    return result;
}

std::string upper(const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(), 
                   [](unsigned char c) { return std::toupper(c); });
    return result;
}

std::string replace(const std::string& str, const std::string& old_str, const std::string& new_str) {
    if (old_str.empty()) {
        return str;
    }
    
    std::string result = str;
    size_t pos = 0;
    while ((pos = result.find(old_str, pos)) != std::string::npos) {
        result.replace(pos, old_str.length(), new_str);
        pos += new_str.length();
    }
    return result;
}

bool contains(const std::string& str, const std::string& substring) {
    return str.find(substring) != std::string::npos;
}

} // namespace StringUtils