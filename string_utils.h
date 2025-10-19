/**
 * @file string_utils.h
 * @brief String utility functions for C++
 * 
 * This header provides string manipulation utilities similar to Python's string methods,
 * including split, strip, join, and other common string operations.
 * 
 * @author String Utils Library Team
 * @version 1.0.0
 * @date 2025
 * @copyright MIT License
 */

#ifndef STRING_UTILS_H
#define STRING_UTILS_H

#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <locale>

/**
 * @namespace StringUtils
 * @brief Namespace containing string utility functions
 */
namespace StringUtils {

/**
 * @brief Split a string by a delimiter into a vector of strings
 * 
 * Similar to Python's str.split() method. Splits the string by the specified delimiter
 * and returns a vector of substrings.
 * 
 * @param str The input string to split
 * @param delimiter The delimiter character to split on (default: space)
 * @param max_split Maximum number of splits to perform (-1 for unlimited)
 * @return std::vector<std::string> Vector of split substrings
 * 
 * @example
 * auto parts = StringUtils::split("hello world test");
 * // parts = {"hello", "world", "test"}
 * 
 * auto parts = StringUtils::split("a,b,c", ',');
 * // parts = {"a", "b", "c"}
 * 
 * auto parts = StringUtils::split("a,b,c,d", ',', 2);
 * // parts = {"a", "b", "c,d"}
 */
std::vector<std::string> split(const std::string& str, char delimiter = ' ', int max_split = -1);

/**
 * @brief Split a string by a string delimiter
 * 
 * Splits the string by the specified string delimiter.
 * 
 * @param str The input string to split
 * @param delimiter The delimiter string to split on
 * @param max_split Maximum number of splits to perform (-1 for unlimited)
 * @return std::vector<std::string> Vector of split substrings
 * 
 * @example
 * auto parts = StringUtils::split("hello::world::test", "::");
 * // parts = {"hello", "world", "test"}
 */
std::vector<std::string> split(const std::string& str, const std::string& delimiter, int max_split = -1);

/**
 * @brief Strip leading and trailing whitespace from a string
 * 
 * Similar to Python's str.strip() method. Removes leading and trailing whitespace.
 * 
 * @param str The input string to strip
 * @return std::string The stripped string
 * 
 * @example
 * auto result = StringUtils::strip("  hello world  ");
 * // result = "hello world"
 */
std::string strip(const std::string& str);

/**
 * @brief Strip leading whitespace from a string
 * 
 * Similar to Python's str.lstrip() method. Removes leading whitespace.
 * 
 * @param str The input string to strip
 * @return std::string The left-stripped string
 * 
 * @example
 * auto result = StringUtils::lstrip("  hello world");
 * // result = "hello world"
 */
std::string lstrip(const std::string& str);

/**
 * @brief Strip trailing whitespace from a string
 * 
 * Similar to Python's str.rstrip() method. Removes trailing whitespace.
 * 
 * @param str The input string to strip
 * @return std::string The right-stripped string
 * 
 * @example
 * auto result = StringUtils::rstrip("hello world  ");
 * // result = "hello world"
 */
std::string rstrip(const std::string& str);

/**
 * @brief Strip specific characters from the beginning and end of a string
 * 
 * Similar to Python's str.strip(chars) method.
 * 
 * @param str The input string to strip
 * @param chars Characters to remove from both ends
 * @return std::string The stripped string
 * 
 * @example
 * auto result = StringUtils::strip("***hello***", "*");
 * // result = "hello"
 */
std::string strip(const std::string& str, const std::string& chars);

/**
 * @brief Join a vector of strings with a delimiter
 * 
 * Similar to Python's str.join() method. Joins strings with the specified delimiter.
 * 
 * @param strings Vector of strings to join
 * @param delimiter The delimiter string to insert between strings
 * @return std::string The joined string
 * 
 * @example
 * std::vector<std::string> parts = {"a", "b", "c"};
 * auto result = StringUtils::join(parts, ", ");
 * // result = "a, b, c"
 */
std::string join(const std::vector<std::string>& strings, const std::string& delimiter);

/**
 * @brief Check if a string starts with a prefix
 * 
 * Similar to Python's str.startswith() method.
 * 
 * @param str The string to check
 * @param prefix The prefix to check for
 * @return bool True if string starts with prefix
 * 
 * @example
 * bool result = StringUtils::startswith("hello world", "hello");
 * // result = true
 */
bool startswith(const std::string& str, const std::string& prefix);

/**
 * @brief Check if a string ends with a suffix
 * 
 * Similar to Python's str.endswith() method.
 * 
 * @param str The string to check
 * @param suffix The suffix to check for
 * @return bool True if string ends with suffix
 * 
 * @example
 * bool result = StringUtils::endswith("hello world", "world");
 * // result = true
 */
bool endswith(const std::string& str, const std::string& suffix);

/**
 * @brief Convert string to lowercase
 * 
 * Similar to Python's str.lower() method.
 * 
 * @param str The string to convert
 * @return std::string The lowercase string
 * 
 * @example
 * auto result = StringUtils::lower("Hello World");
 * // result = "hello world"
 */
std::string lower(const std::string& str);

/**
 * @brief Convert string to uppercase
 * 
 * Similar to Python's str.upper() method.
 * 
 * @param str The string to convert
 * @return std::string The uppercase string
 * 
 * @example
 * auto result = StringUtils::upper("Hello World");
 * // result = "HELLO WORLD"
 */
std::string upper(const std::string& str);

/**
 * @brief Replace all occurrences of a substring
 * 
 * Similar to Python's str.replace() method.
 * 
 * @param str The input string
 * @param old_str The substring to replace
 * @param new_str The replacement substring
 * @return std::string The string with replacements
 * 
 * @example
 * auto result = StringUtils::replace("hello world", "world", "there");
 * // result = "hello there"
 */
std::string replace(const std::string& str, const std::string& old_str, const std::string& new_str);

/**
 * @brief Check if a string contains a substring
 * 
 * Similar to Python's "in" operator for strings.
 * 
 * @param str The string to search in
 * @param substring The substring to search for
 * @return bool True if substring is found
 * 
 * @example
 * bool result = StringUtils::contains("hello world", "world");
 * // result = true
 */
bool contains(const std::string& str, const std::string& substring);

} // namespace StringUtils

#endif // STRING_UTILS_H