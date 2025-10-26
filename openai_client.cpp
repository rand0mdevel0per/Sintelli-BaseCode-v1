/**
 * @file openai_client.cpp
 * @brief Implementation of OpenAI Compatible API Client using libcurl
 * 
 * This file contains the implementation of the HTTP client for OpenAI-compatible
 * APIs using the libcurl library for HTTP communication.
 * 
 * @author OpenAI Client Library Team
 * @version 1.0.0
 * @date 2025
 * @copyright MIT License
 */

#include "openai_client.h"
#include <curl/curl.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <functional>

using namespace OpenAIClient;

/**
 * @brief Callback function for writing HTTP response data to a string buffer
 * 
 * This function is used by libcurl to accumulate response data from HTTP requests
 * into a std::string buffer.
 * 
 * @param contents Pointer to the data received
 * @param size Size of each data element
 * @param nmemb Number of data elements
 * @param response Pointer to the std::string where data will be appended
 * @return size_t Total number of bytes processed
 */
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* response) {
    size_t total_size = size * nmemb;
    response->append((char*)contents, total_size);
    return total_size;
}

/**
 * @brief Callback function for streaming HTTP response data
 * 
 * This function is used by libcurl to process streaming response data line by line
 * for Server-Sent Events (SSE) streaming.
 * 
 * @param contents Pointer to the data received
 * @param size Size of each data element
 * @param nmemb Number of data elements
 * @param user_data Pointer to user data (streaming callback function)
 * @return size_t Total number of bytes processed
 */
static size_t StreamWriteCallback(void* contents, size_t size, size_t nmemb, void* user_data) {
    size_t total_size = size * nmemb;
    std::string data((char*)contents, total_size);
    
    // Cast user_data to the streaming callback function
    auto* stream_callback = static_cast<std::function<void(const std::string&)>*>(user_data);
    
    // Process data line by line
    std::istringstream stream(data);
    std::string line;
    while (std::getline(stream, line)) {
        // Remove carriage return if present
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        
        // Skip empty lines
        if (!line.empty()) {
            (*stream_callback)(line);
        }
    }
    
    return total_size;
}

/**
 * @class SimpleHttpClient
 * @brief Simple HTTP client implementation using libcurl
 * 
 * This class provides basic HTTP operations (GET, POST, DELETE) using the
 * libcurl library as the underlying HTTP client.
 */
class SimpleHttpClient {
private:
    CURL* curl_;  // libcurl easy handle
    
public:
    /**
     * @brief Constructor - initializes libcurl and creates an easy handle
     * 
     * @throws std::runtime_error if libcurl initialization fails
     */
    SimpleHttpClient() {
        curl_global_init(CURL_GLOBAL_DEFAULT);
        curl_ = curl_easy_init();
        if (!curl_) {
            throw std::runtime_error("Failed to initialize CURL");
        }
    }
    
    /**
     * @brief Destructor - cleans up libcurl resources
     */
    ~SimpleHttpClient() {
        if (curl_) {
            curl_easy_cleanup(curl_);
        }
        curl_global_cleanup();
    }
    
    /**
     * @brief Perform an HTTP POST request
     * 
     * @param url The URL to send the POST request to
     * @param data The data to send in the request body
     * @param headers HTTP headers to include in the request
     * @return std::string The response body
     * @throws std::runtime_error if the HTTP request fails
     */
    std::string post(const std::string& url, 
                     const std::string& data, 
                     const std::map<std::string, std::string>& headers) {
        std::string response;
        
        // Set libcurl options for POST request
        curl_easy_setopt(curl_, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl_, CURLOPT_POSTFIELDS, data.c_str());
        curl_easy_setopt(curl_, CURLOPT_POSTFIELDSIZE, data.length());
        curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl_, CURLOPT_WRITEDATA, &response);
        
        // Set headers
        struct curl_slist* header_list = nullptr;
        for (const auto& header : headers) {
            std::string header_str = header.first + ": " + header.second;
            header_list = curl_slist_append(header_list, header_str.c_str());
        }
        curl_easy_setopt(curl_, CURLOPT_HTTPHEADER, header_list);
        
        // Execute the request
        CURLcode res = curl_easy_perform(curl_);
        curl_slist_free_all(header_list);
        
        if (res != CURLE_OK) {
            throw std::runtime_error("HTTP request failed: " + std::string(curl_easy_strerror(res)));
        }
        
        return response;
    }
    
    /**
     * @brief Perform an HTTP POST request with streaming response
     * 
     * @param url The URL to send the POST request to
     * @param data The data to send in the request body
     * @param headers HTTP headers to include in the request
     * @param stream_callback Callback function to process streaming response data
     * @throws std::runtime_error if the HTTP request fails
     */
    void post_stream(const std::string& url, 
                     const std::string& data, 
                     const std::map<std::string, std::string>& headers,
                     std::function<void(const std::string&)> stream_callback) {
        // Set libcurl options for POST request with streaming
        curl_easy_setopt(curl_, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl_, CURLOPT_POSTFIELDS, data.c_str());
        curl_easy_setopt(curl_, CURLOPT_POSTFIELDSIZE, data.length());
        curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION, StreamWriteCallback);
        curl_easy_setopt(curl_, CURLOPT_WRITEDATA, &stream_callback);
        
        // Set headers
        struct curl_slist* header_list = nullptr;
        for (const auto& header : headers) {
            std::string header_str = header.first + ": " + header.second;
            header_list = curl_slist_append(header_list, header_str.c_str());
        }
        curl_easy_setopt(curl_, CURLOPT_HTTPHEADER, header_list);
        
        // Execute the request
        CURLcode res = curl_easy_perform(curl_);
        curl_slist_free_all(header_list);
        
        if (res != CURLE_OK) {
            throw std::runtime_error("HTTP request failed: " + std::string(curl_easy_strerror(res)));
        }
    }
    
    /**
     * @brief Perform an HTTP GET request
     * 
     * @param url The URL to send the GET request to
     * @param headers HTTP headers to include in the request
     * @return std::string The response body
     * @throws std::runtime_error if the HTTP request fails
     */
    std::string get(const std::string& url, const std::map<std::string, std::string>& headers) {
        std::string response;
        
        // Set libcurl options for GET request
        curl_easy_setopt(curl_, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl_, CURLOPT_HTTPGET, 1L);
        curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl_, CURLOPT_WRITEDATA, &response);
        
        // Set headers
        struct curl_slist* header_list = nullptr;
        for (const auto& header : headers) {
            std::string header_str = header.first + ": " + header.second;
            header_list = curl_slist_append(header_list, header_str.c_str());
        }
        curl_easy_setopt(curl_, CURLOPT_HTTPHEADER, header_list);
        
        // Execute the request
        CURLcode res = curl_easy_perform(curl_);
        curl_slist_free_all(header_list);
        
        if (res != CURLE_OK) {
            throw std::runtime_error("HTTP request failed: " + std::string(curl_easy_strerror(res)));
        }
        
        return response;
    }
    
    /**
     * @brief Perform an HTTP DELETE request
     * 
     * @param url The URL to send the DELETE request to
     * @param headers HTTP headers to include in the request
     * @return bool True if the request was successful
     */
    bool delete_request(const std::string& url, const std::map<std::string, std::string>& headers) {
        std::string response;
        
        // Set libcurl options for DELETE request
        curl_easy_setopt(curl_, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl_, CURLOPT_CUSTOMREQUEST, "DELETE");
        curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl_, CURLOPT_WRITEDATA, &response);
        
        // Set headers
        struct curl_slist* header_list = nullptr;
        for (const auto& header : headers) {
            std::string header_str = header.first + ": " + header.second;
            header_list = curl_slist_append(header_list, header_str.c_str());
        }
        curl_easy_setopt(curl_, CURLOPT_HTTPHEADER, header_list);
        
        // Execute the request
        CURLcode res = curl_easy_perform(curl_);
        curl_slist_free_all(header_list);
        
        return res == CURLE_OK;
    }
};

/**
 * @brief Create a chat completion using the OpenAI-compatible API
 * 
 * Sends a chat completion request to the API and returns the generated response.
 * Supports both regular and streaming responses.
 * 
 * @param request The chat completion request parameters
 * @return ChatCompletionResponse The generated chat completion response
 * @throws std::runtime_error if the request fails or response parsing fails
 */
ChatCompletionResponse HttpClient::createChatCompletion(const ChatCompletionRequest& request) {
    SimpleHttpClient http;
    std::string url = get_base_url() + "/chat/completions";
    
    // Set required headers for authentication and content type
    std::map<std::string, std::string> headers = {
        {"Content-Type", "application/json"},
        {"Authorization", "Bearer " + get_api_key()}
    };
    
    // Add organization header if specified
    if (!get_organization().empty()) {
        headers["OpenAI-Organization"] = get_organization();
    }
    
    // Convert request to JSON
    std::string request_json = request.to_json().dump();
    
    // Handle streaming requests
    if (request.stream) {
        // For streaming, we'll use the dedicated streaming method
        ChatCompletionResponse response;
        response.id = "streamed_response";
        response.object = "chat.completion";
        response.created = std::time(nullptr);
        response.model = request.model;
        
        // Create a temporary callback to collect the streaming responses
        std::string complete_response;
        auto stream_callback = [&response, &complete_response](const std::string& data) {
            // Process Server-Sent Events (SSE) format
            if (data.substr(0, 5) == "data:") {
                std::string json_str = data.substr(5);
                // Remove leading whitespaces
                json_str.erase(0, json_str.find_first_not_of(" \t"));
                
                // Check if it's the end marker
                if (json_str == "[DONE]") {
                    return; // End of stream
                }
                
                try {
                    json chunk = json::parse(json_str);
                    
                    // Parse the chunk and add it to response
                    if (chunk.contains("choices") && chunk["choices"].is_array()) {
                        for (const auto& choice : chunk["choices"]) {
                            ChatCompletionResponse::Choice resp_choice;
                            resp_choice.index = choice["index"];
                            
                            if (choice.contains("delta") && choice["delta"].contains("content")) {
                                resp_choice.delta = choice["delta"]["content"];
                            } else if (choice.contains("text")) {
                                resp_choice.delta = choice["text"];
                            }
                            
                            if (choice.contains("finish_reason") && !choice["finish_reason"].is_null()) {
                                resp_choice.finish_reason = choice["finish_reason"];
                            }
                            
                            response.choices.push_back(resp_choice);
                        }
                    }
                    
                    // Update other fields
                    if (chunk.contains("id")) response.id = chunk["id"];
                    if (chunk.contains("model")) response.model = chunk["model"];
                    if (chunk.contains("created")) response.created = chunk["created"];
                    
                } catch (const std::exception& e) {
                    // Silently ignore malformed JSON chunks
                }
            }
        };
        
        // Send streaming POST request
        http.post_stream(url, request_json, headers, stream_callback);
        
        return response;
    } else {
        // Non-streaming request
        std::string response_json = http.post(url, request_json, headers);
        
        // Parse and return the response
        try {
            json response = json::parse(response_json);
            return ChatCompletionResponse::from_json(response);
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to parse chat completion response: " + std::string(e.what()));
        }
    }
}

/**
 * @brief Create a streaming chat completion using the OpenAI-compatible API
 * 
 * Sends a chat completion request with streaming enabled and processes
 * the Server-Sent Events (SSE) responses as they arrive via callback.
 * 
 * @param request The chat completion request parameters (stream should be true)
 * @param callback Callback function to handle each streaming response
 * @throws std::runtime_error if the request fails
 */
void HttpClient::createChatCompletionStream(const ChatCompletionRequest& request, 
                                           std::function<void(const ChatCompletionResponse&)> callback) {
    SimpleHttpClient http;
    std::string url = get_base_url() + "/chat/completions";
    
    // Set required headers for authentication and content type
    std::map<std::string, std::string> headers = {
        {"Content-Type", "application/json"},
        {"Authorization", "Bearer " + get_api_key()}
    };
    
    // Add organization header if specified
    if (!get_organization().empty()) {
        headers["OpenAI-Organization"] = get_organization();
    }
    
    // Create a mutable copy of request with stream enabled
    ChatCompletionRequest stream_request = request;
    stream_request.stream = true;
    
    // Convert request to JSON
    std::string request_json = stream_request.to_json().dump();
    
    // Define streaming callback to process Server-Sent Events (SSE)
    auto stream_callback = [callback](const std::string& data) {
        // Process Server-Sent Events (SSE) format
        if (data.substr(0, 5) == "data:") {
            std::string json_str = data.substr(5);
            // Remove leading whitespaces
            json_str.erase(0, json_str.find_first_not_of(" \t"));
            
            // Check if it's the end marker
            if (json_str == "[DONE]") {
                return; // End of stream
            }
            
            try {
                json chunk = json::parse(json_str);
                
                // Create a response object from the chunk
                ChatCompletionResponse response;
                if (chunk.contains("id")) response.id = chunk["id"];
                if (chunk.contains("object")) response.object = chunk["object"];
                if (chunk.contains("created")) response.created = chunk["created"];
                if (chunk.contains("model")) response.model = chunk["model"];
                
                // Process choices
                if (chunk.contains("choices") && chunk["choices"].is_array()) {
                    for (const auto& choice : chunk["choices"]) {
                        ChatCompletionResponse::Choice resp_choice;
                        resp_choice.index = choice["index"];
                        
                        if (choice.contains("delta") && choice["delta"].contains("content")) {
                            resp_choice.delta = choice["delta"]["content"];
                        } else if (choice.contains("text")) {
                            resp_choice.delta = choice["text"];
                        }
                        
                        if (choice.contains("finish_reason") && !choice["finish_reason"].is_null()) {
                            resp_choice.finish_reason = choice["finish_reason"];
                        }
                        
                        // Process message if present
                        if (choice.contains("message") && choice["message"].contains("content")) {
                            resp_choice.message.content = choice["message"]["content"];
                            if (choice["message"].contains("role")) {
                                resp_choice.message.role = choice["message"]["role"];
                            }
                        }
                        
                        response.choices.push_back(resp_choice);
                    }
                }
                
                // Process token usage if present
                if (chunk.contains("usage")) {
                    if (chunk["usage"].contains("prompt_tokens")) {
                        response.usage.prompt_tokens = chunk["usage"]["prompt_tokens"];
                    }
                    if (chunk["usage"].contains("completion_tokens")) {
                        response.usage.completion_tokens = chunk["usage"]["completion_tokens"];
                    }
                    if (chunk["usage"].contains("total_tokens")) {
                        response.usage.total_tokens = chunk["usage"]["total_tokens"];
                    }
                }
                
                // Call the provided callback with the processed response
                callback(response);
                
            } catch (const std::exception& e) {
                // Silently ignore malformed JSON chunks
            }
        }
    };
    
    // Send streaming POST request
    http.post_stream(url, request_json, headers, stream_callback);
}

/**
 * @brief Create a fine-tuning job using the OpenAI-compatible API
 * 
 * Creates a new fine-tuning job for training a custom model.
 * 
 * @param request The training job request parameters
 * @return TrainingJobResponse Information about the created training job
 * @throws std::runtime_error if the request fails or response parsing fails
 */
TrainingJobResponse HttpClient::createFineTuneJob(const TrainingJobRequest& request) {
    SimpleHttpClient http;
    std::string url = get_base_url() + "/fine-tunes";
    
    // Set required headers for authentication and content type
    std::map<std::string, std::string> headers = {
        {"Content-Type", "application/json"},
        {"Authorization", "Bearer " + get_api_key()}
    };
    
    // Add organization header if specified
    if (!get_organization().empty()) {
        headers["OpenAI-Organization"] = get_organization();
    }
    
    // Convert request to JSON and send POST request
    std::string request_json = request.to_json().dump();
    std::string response_json = http.post(url, request_json, headers);
    
    // Parse and return the response
    try {
        json response = json::parse(response_json);
        return TrainingJobResponse::from_json(response);
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to parse fine-tune job response: " + std::string(e.what()));
    }
}

/**
 * @brief Get details of a specific fine-tuning job by ID
 * 
 * Retrieves information about a specific fine-tuning job.
 * 
 * @param job_id The ID of the training job to retrieve
 * @return TrainingJobResponse Training job information
 * @throws std::runtime_error if the request fails or response parsing fails
 */
TrainingJobResponse HttpClient::getFineTuneJob(const std::string& job_id) {
    SimpleHttpClient http;
    std::string url = get_base_url() + "/fine-tunes/" + job_id;
    
    // Set authentication headers
    std::map<std::string, std::string> headers = {
        {"Authorization", "Bearer " + get_api_key()}
    };
    
    // Add organization header if specified
    if (!get_organization().empty()) {
        headers["OpenAI-Organization"] = get_organization();
    }
    
    // Send GET request and parse response
    std::string response_json = http.get(url, headers);
    
    try {
        json response = json::parse(response_json);
        return TrainingJobResponse::from_json(response);
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to parse fine-tune job response: " + std::string(e.what()));
    }
}

/**
 * @brief List all fine-tuning jobs associated with the API key
 * 
 * Retrieves a list of all fine-tuning jobs for the current API key.
 * 
 * @return std::vector<TrainingJobResponse> List of training jobs
 * @throws std::runtime_error if the request fails or response parsing fails
 */
std::vector<TrainingJobResponse> HttpClient::listFineTuneJobs() {
    SimpleHttpClient http;
    std::string url = get_base_url() + "/fine-tunes";
    
    // Set authentication headers
    std::map<std::string, std::string> headers = {
        {"Authorization", "Bearer " + get_api_key()}
    };
    
    // Add organization header if specified
    if (!get_organization().empty()) {
        headers["OpenAI-Organization"] = get_organization();
    }
    
    // Send GET request to list jobs
    std::string response_json = http.get(url, headers);
    
    // Parse response and extract job list
    try {
        json response = json::parse(response_json);
        std::vector<TrainingJobResponse> jobs;
        
        // Extract jobs from the "data" array in the response
        if (response.contains("data") && response["data"].is_array()) {
            for (const auto& job_data : response["data"]) {
                jobs.push_back(TrainingJobResponse::from_json(job_data));
            }
        }
        
        return jobs;
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to parse fine-tune jobs list response: " + std::string(e.what()));
    }
}

/**
 * @brief Cancel a fine-tuning job by ID
 * 
 * Attempts to cancel a running fine-tuning job.
 * 
 * @param job_id The ID of the training job to cancel
 * @return bool True if cancellation was successful
 * @throws std::runtime_error if the request fails or response parsing fails
 */
bool HttpClient::cancelFineTuneJob(const std::string& job_id) {
    SimpleHttpClient http;
    std::string url = get_base_url() + "/fine-tunes/" + job_id + "/cancel";
    
    // Set authentication headers
    std::map<std::string, std::string> headers = {
        {"Authorization", "Bearer " + get_api_key()}
    };
    
    // Add organization header if specified
    if (!get_organization().empty()) {
        headers["OpenAI-Organization"] = get_organization();
    }
    
    // Send POST request to cancel the job (empty body)
    std::string response_json = http.post(url, "", headers);
    
    // Check if cancellation was successful
    try {
        json response = json::parse(response_json);
        return response.contains("status") && response["status"] == "cancelled";
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to parse cancel job response: " + std::string(e.what()));
    }
}

/**
 * @brief Upload a file for training (not implemented - requires multipart form data)
 * 
 * This method is currently not implemented and requires proper multipart form data
 * handling for file uploads.
 * 
 * @param file_path Path to the file to upload
 * @param purpose Purpose of the file upload
 * @return std::string ID of the uploaded file
 * @throws std::runtime_error Always throws as this method is not implemented
 */
std::string HttpClient::uploadFile(const std::string& file_path, const std::string& purpose) {
    // Note: File upload requires multipart form data
    // This is a simplified implementation - for production, you'd need proper multipart handling
    throw std::runtime_error("File upload not yet implemented - requires multipart form data handling");
}

/**
 * @brief Delete a file by ID
 * 
 * Deletes a file from the API using its file ID.
 * 
 * @param file_id The ID of the file to delete
 * @return bool True if deletion was successful
 */
bool HttpClient::deleteFile(const std::string& file_id) {
    SimpleHttpClient http;
    std::string url = get_base_url() + "/files/" + file_id;
    
    // Set authentication headers
    std::map<std::string, std::string> headers = {
        {"Authorization", "Bearer " + get_api_key()}
    };
    
    // Add organization header if specified
    if (!get_organization().empty()) {
        headers["OpenAI-Organization"] = get_organization();
    }
    
    // Send DELETE request and return success status
    return http.delete_request(url, headers);
}

/**
 * @brief Create a streaming chat completion using the OpenAI-compatible API
 * 
 * Sends a chat completion request with streaming enabled and processes
 * the Server-Sent Events (SSE) responses as they arrive via callback.
 * 
 * @param request The chat completion request parameters (stream should be true)
 * @param callback Callback function to handle each streaming response
 * @throws std::runtime_error if the request fails
 */
void HttpClient::createChatCompletionStream(const ChatCompletionRequest& request, 
                                           std::function<void(const ChatCompletionResponse&)> callback) {
    SimpleHttpClient http;
    std::string url = get_base_url() + "/chat/completions";
    
    // Set required headers for authentication and content type
    std::map<std::string, std::string> headers = {
        {"Content-Type", "application/json"},
        {"Authorization", "Bearer " + get_api_key()}
    };
    
    // Add organization header if specified
    if (!get_organization().empty()) {
        headers["OpenAI-Organization"] = get_organization();
    }
    
    // Create a mutable copy of request with stream enabled
    ChatCompletionRequest stream_request = request;
    stream_request.stream = true;
    
    // Convert request to JSON
    std::string request_json = stream_request.to_json().dump();
    
    // Define streaming callback to process Server-Sent Events (SSE)
    auto stream_callback = [callback](const std::string& data) {
        // Process Server-Sent Events (SSE) format
        if (data.substr(0, 5) == "data:") {
            std::string json_str = data.substr(5);
            // Remove leading whitespaces
            json_str.erase(0, json_str.find_first_not_of(" \t"));
            
            // Check if it's the end marker
            if (json_str == "[DONE]") {
                return; // End of stream
            }
            
            try {
                json chunk = json::parse(json_str);
                
                // Create a response object from the chunk
                ChatCompletionResponse response;
                if (chunk.contains("id")) response.id = chunk["id"];
                if (chunk.contains("object")) response.object = chunk["object"];
                if (chunk.contains("created")) response.created = chunk["created"];
                if (chunk.contains("model")) response.model = chunk["model"];
                
                // Process choices
                if (chunk.contains("choices") && chunk["choices"].is_array()) {
                    for (const auto& choice : chunk["choices"]) {
                        ChatCompletionResponse::Choice resp_choice;
                        resp_choice.index = choice["index"];
                        
                        if (choice.contains("delta") && choice["delta"].contains("content")) {
                            resp_choice.delta = choice["delta"]["content"];
                        } else if (choice.contains("text")) {
                            resp_choice.delta = choice["text"];
                        }
                        
                        if (choice.contains("finish_reason") && !choice["finish_reason"].is_null()) {
                            resp_choice.finish_reason = choice["finish_reason"];
                        }
                        
                        // Process message if present
                        if (choice.contains("message") && choice["message"].contains("content")) {
                            resp_choice.message.content = choice["message"]["content"];
                            if (choice["message"].contains("role")) {
                                resp_choice.message.role = choice["message"]["role"];
                            }
                        }
                        
                        // Process image data if present
                        if (choice.contains("image_url")) {
                            resp_choice.image_url = choice["image_url"];
                        }
                        if (choice.contains("image_b64")) {
                            resp_choice.image_b64 = choice["image_b64"];
                        }
                        
                        response.choices.push_back(resp_choice);
                    }
                }
                
                // Process token usage if present
                if (chunk.contains("usage")) {
                    if (chunk["usage"].contains("prompt_tokens")) {
                        response.usage.prompt_tokens = chunk["usage"]["prompt_tokens"];
                    }
                    if (chunk["usage"].contains("completion_tokens")) {
                        response.usage.completion_tokens = chunk["usage"]["completion_tokens"];
                    }
                    if (chunk["usage"].contains("total_tokens")) {
                        response.usage.total_tokens = chunk["usage"]["total_tokens"];
                    }
                }
                
                // Call the provided callback with the processed response
                callback(response);
                
            } catch (const std::exception& e) {
                // Silently ignore malformed JSON chunks
            }
        }
    };
    
    // Send streaming POST request
    http.post_stream(url, request_json, headers, stream_callback);
}

/**
 * @brief Create an image using the image generation API
 * 
 * Sends an image generation request and returns the generated image URL or base64 data.
 * 
 * @param prompt Text description of the image to generate
 * @param n Number of images to generate (default: 1)
 * @param size Size of the generated image (default: "1024x1024")
 * @param response_format Format of the response ("url" or "b64_json", default: "url")
 * @return ImageGenerationResponse Generated image data
 * @throws std::runtime_error if the request fails
 */
ImageGenerationResponse HttpClient::createImage(const std::string& prompt, int n, const std::string& size, const std::string& response_format) {
    SimpleHttpClient http;
    std::string url = get_base_url() + "/images/generations";
    
    // Set required headers for authentication and content type
    std::map<std::string, std::string> headers = {
        {"Content-Type", "application/json"},
        {"Authorization", "Bearer " + get_api_key()}
    };
    
    // Add organization header if specified
    if (!get_organization().empty()) {
        headers["OpenAI-Organization"] = get_organization();
    }
    
    // Create request JSON
    json request_json;
    request_json["prompt"] = prompt;
    request_json["n"] = n;
    request_json["size"] = size;
    request_json["response_format"] = response_format;
    
    // Convert to string
    std::string request_data = request_json.dump();
    
    // Send POST request
    std::string response_json = http.post(url, request_data, headers);
    
    // Parse and return the response
    try {
        json response = json::parse(response_json);
        return ImageGenerationResponse::from_json(response);
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to parse image generation response: " + std::string(e.what()));
    }
}