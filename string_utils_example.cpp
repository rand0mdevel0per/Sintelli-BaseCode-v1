/**
 * @file string_utils_example.cpp
 * @brief Example usage of StringUtils functions
 * 
 * This file demonstrates how to use the StringUtils library
 * with examples similar to Python's string operations.
 * 
 * @author String Utils Library Team
 * @version 1.0.0
 * @date 2025
 * @copyright MIT License
 */

#include "string_utils.h"
#include <iostream>
#include <vector>

void demonstrate_split() {
    std::cout << "=== String Split Examples ===" << std::endl;
    
    // Basic split by space
    auto parts1 = StringUtils::split("hello world test");
    std::cout << "split('hello world test'): ";
    for (const auto& part : parts1) {
        std::cout << "['" << part << "'] ";
    }
    std::cout << std::endl;
    
    // Split by comma
    auto parts2 = StringUtils::split("a,b,c,d", ',');
    std::cout << "split('a,b,c,d', ','): ";
    for (const auto& part : parts2) {
        std::cout << "['" << part << "'] ";
    }
    std::cout << std::endl;
    
    // Split with max_split
    auto parts3 = StringUtils::split("a,b,c,d", ',', 2);
    std::cout << "split('a,b,c,d', ',', 2): ";
    for (const auto& part : parts3) {
        std::cout << "['" << part << "'] ";
    }
    std::cout << std::endl;
    
    // Split by string delimiter
    auto parts4 = StringUtils::split("hello::world::test", "::");
    std::cout << "split('hello::world::test', '::'): ";
    for (const auto& part : parts4) {
        std::cout << "['" << part << "'] ";
    }
    std::cout << std::endl;
}

void demonstrate_strip() {
    std::cout << "\n=== String Strip Examples ===" << std::endl;
    
    // Basic strip
    std::cout << "strip('  hello world  '): ['" 
              << StringUtils::strip("  hello world  ") << "']" << std::endl;
    
    // Left strip
    std::cout << "lstrip('  hello world'): ['" 
              << StringUtils::lstrip("  hello world") << "']" << std::endl;
    
    // Right strip
    std::cout << "rstrip('hello world  '): ['" 
              << StringUtils::rstrip("hello world  ") << "']" << std::endl;
    
    // Strip specific characters
    std::cout << "strip('***hello***', '*'): ['" 
              << StringUtils::strip("***hello***", "*") << "']" << std::endl;
}

void demonstrate_join() {
    std::cout << "\n=== String Join Examples ===" << std::endl;
    
    std::vector<std::string> parts = {"apple", "banana", "cherry"};
    
    std::cout << "join({'apple', 'banana', 'cherry'}, ', '): '" 
              << StringUtils::join(parts, ", ") << "'" << std::endl;
    
    std::cout << "join({'apple', 'banana', 'cherry'}, ' -> '): '" 
              << StringUtils::join(parts, " -> ") << "'" << std::endl;
}

void demonstrate_case_operations() {
    std::cout << "\n=== Case Operations ===" << std::endl;
    
    std::cout << "lower('Hello World'): '" 
              << StringUtils::lower("Hello World") << "'" << std::endl;
    
    std::cout << "upper('Hello World'): '" 
              << StringUtils::upper("Hello World") << "'" << std::endl;
}

void demonstrate_other_operations() {
    std::cout << "\n=== Other String Operations ===" << std::endl;
    
    // startsWith
    std::cout << "startswith('hello world', 'hello'): " 
              << (StringUtils::startswith("hello world", "hello") ? "true" : "false") << std::endl;
    
    // endsWith
    std::cout << "endswith('hello world', 'world'): " 
              << (StringUtils::endswith("hello world", "world") ? "true" : "false") << std::endl;
    
    // contains
    std::cout << "contains('hello world', 'world'): " 
              << (StringUtils::contains("hello world", "world") ? "true" : "false") << std::endl;
    
    // replace
    std::cout << "replace('hello world', 'world', 'there'): '" 
              << StringUtils::replace("hello world", "world", "there") << "'" << std::endl;
}

void demonstrate_real_world_example() {
    std::cout << "\n=== Real World Example ===" << std::endl;
    
    // Example: Processing CSV-like data
    std::string data = "  John Doe, 25, Engineer, New York  ";
    
    // Strip whitespace
    std::string cleaned = StringUtils::strip(data);
    std::cout << "Original: ['" << data << "']" << std::endl;
    std::cout << "Stripped: ['" << cleaned << "']" << std::endl;
    
    // Split into fields
    auto fields = StringUtils::split(cleaned, ',');
    std::cout << "Fields: ";
    for (size_t i = 0; i < fields.size(); ++i) {
        std::cout << "Field " << i << ": ['" << StringUtils::strip(fields[i]) << "'] ";
    }
    std::cout << std::endl;
    
    // Convert to uppercase
    std::cout << "Uppercase: '" << StringUtils::upper(cleaned) << "'" << std::endl;
}

int main() {
    std::cout << "StringUtils Library Examples" << std::endl;
    std::cout << "============================" << std::endl;
    
    demonstrate_split();
    demonstrate_strip();
    demonstrate_join();
    demonstrate_case_operations();
    demonstrate_other_operations();
    demonstrate_real_world_example();
    
    std::cout << "\nAll examples completed successfully!" << std::endl;
    return 0;
}