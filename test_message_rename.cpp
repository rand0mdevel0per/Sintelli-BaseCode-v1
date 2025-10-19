/**
 * @file test_message_rename.cpp
 * @brief Test file to verify Message -> ChatMessage rename
 */

#include "openai_client.h"
#include <iostream>

int main() {
    // Test that ChatMessage works correctly
    OpenAIClient::ChatMessage system_msg("system", "You are a helpful assistant.");
    OpenAIClient::ChatMessage user_msg("user", "Hello!");
    
    // Test ChatCompletionRequest
    OpenAIClient::ChatCompletionRequest request;
    request.model = "test-model";
    request.messages.push_back(system_msg);
    request.messages.push_back(user_msg);
    
    std::cout << "ChatMessage rename test passed!" << std::endl;
    std::cout << "Number of messages: " << request.messages.size() << std::endl;
    std::cout << "First message role: " << request.messages[0].role << std::endl;
    std::cout << "Second message content: " << request.messages[1].content << std::endl;
    
    return 0;
}