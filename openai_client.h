/**
 * @file openai_client.h
 * @brief OpenAI Compatible API Client for OpenRouter and similar services
 * 
 * This header provides a C++ client implementation for OpenAI-compatible APIs,
 * supporting chat completions and fine-tuning operations.
 * 
 * @author OpenAI Client Library Team
 * @version 1.0.0
 * @date 2025
 * @copyright MIT License
 */

#ifndef OPENAI_CLIENT_H
#define OPENAI_CLIENT_H

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace OpenAIClient {
    /**

     * @struct ChatMessageContentPart

     * @brief Represents a part of a message content (text or image)

     */

    struct ChatMessageContentPart {
        std::string type; ///< Type of content: "text" or "image_url"

        std::string text; ///< Text content (when type is "text")

        std::string image_url; ///< Image URL (when type is "image_url")


        /**

         * @brief Construct a text content part

         *

         * @param text_content The text content

         */

        ChatMessageContentPart(const std::string &text_content)

            : type("text"), text(text_content) {
        }


        /**

         * @brief Construct an image content part

         *

         * @param img_url The image URL

         */

        ChatMessageContentPart(const std::string &img_url, bool is_image)

            : type(is_image ? "image_url" : "text"), text(is_image ? "" : "content"),
              image_url(is_image ? "content" : "") {
        }


        /**

         * @brief Construct an image content part from base64 data

         *

         * @param base64_data The base64 encoded image data

         * @param mime_type The MIME type of the image (e.g., "image/jpeg", "image/png")

         */

        ChatMessageContentPart(const std::string &base64_data, const std::string &mime_type)

            : type("image_url"), image_url("data:" + mime_type + ";base64," + base64_data) {
        }


        /**

         * @brief Convert the content part to JSON format

         *

         * @return json JSON representation of the content part

         */

        json to_json() const {
            json j;

            j["type"] = type;

            if (type == "text") {
                j["text"] = text;
            } else if (type == "image_url") {
                json image_obj;

                image_obj["url"] = image_url;

                j["image_url"] = image_obj;
            }

            return j;
        }
    };

    /**
     * @struct ChatMessage
     * @brief Represents a message in a chat conversation
     *
     * This structure defines the format for messages exchanged between
     * the user, system, and AI assistant in chat completion requests.
     */
    struct ChatMessage {
        std::string role; ///< Role of the message sender: "system", "user", "assistant"
        std::string content; ///< The actual content of the message (legacy text-only)
        std::vector<ChatMessageContentPart> content_parts; ///< Content parts for multimodal support
        std::string name; ///< Optional: name of the participant

        /**
         * @brief Construct a new Message object with text content
         *
         * @param r Role of the message sender
         * @param c Content of the message
         * @param n Optional name of the participant
         */
        ChatMessage(const std::string &r, const std::string &c, const std::string &n = "")
            : role(r), content(c), name(n) {
        }

        /**
         * @brief Construct a new Message object with content parts
         *
         * @param r Role of the message sender
         * @param parts Content parts for multimodal messages
         * @param n Optional name of the participant
         */
        ChatMessage(const std::string &r, const std::vector<ChatMessageContentPart> &parts, const std::string &n = "")
            : role(r), content(""), content_parts(parts), name(n) {
        }

        /**
         * @brief Convert the message to JSON format
         *
         * @return json JSON representation of the message
         */
        json to_json() const {
            json j;
            j["role"] = role;

            // If content_parts is provided, use it (multimodal format)
            if (!content_parts.empty()) {
                j["content"] = json::array();
                for (const auto &part: content_parts) {
                    j["content"].push_back(part.to_json());
                }
            } else {
                // Otherwise, use legacy text-only format
                j["content"] = content;
            }

            if (!name.empty()) {
                j["name"] = name;
            }
            return j;
        }
    };

    /**
     * @struct ChatCompletionRequest
     * @brief Request structure for chat completion API
     *
     * This structure contains all parameters needed to make a chat completion
     * request to the OpenAI-compatible API.
     */
    struct ChatCompletionRequest {
        std::string model; ///< Model identifier to use for completion
        std::vector<ChatMessage> messages; ///< Conversation history messages
        double temperature = 1.0; ///< Sampling temperature (0.0 to 2.0)
        double top_p = 1.0; ///< Nucleus sampling parameter
        int max_tokens = 2048; ///< Maximum tokens to generate
        bool stream = false; ///< Whether to stream the response
        std::vector<std::string> stop; ///< Stop sequences to end generation

        /**
         * @brief Convert the request to JSON format
         *
         * @return json JSON representation of the chat completion request
         */
        json to_json() const {
            json j;
            j["model"] = model;
            j["messages"] = json::array();
            for (const auto &msg: messages) {
                j["messages"].push_back(msg.to_json());
            }
            j["temperature"] = temperature;
            j["top_p"] = top_p;
            j["max_tokens"] = max_tokens;
            j["stream"] = stream;
            if (!stop.empty()) {
                j["stop"] = stop;
            }
            return j;
        }
    };

    /**
     * @struct ChatCompletionResponse
     * @brief Response structure for chat completion API
     *
     * This structure contains the response data from a chat completion request.
     */
    struct ChatCompletionResponse {
        /**

         * @struct Choice

         * @brief Individual choice in the completion response

         */

        struct Choice {
            int index; ///< Index of the choice in the response

            ChatMessage message; ///< Generated message content

            std::string delta; ///< Delta content for streaming responses

            std::string finish_reason; ///< Reason why generation stopped

            std::string image_url; ///< URL of generated image (for image generation models)

            std::string image_b64; ///< Base64 encoded image data (for image generation models)
        };

        /**
         * @struct Usage
         * @brief Token usage information
         */
        struct Usage {
            int prompt_tokens; ///< Number of tokens in the prompt
            int completion_tokens; ///< Number of tokens in the completion
            int total_tokens; ///< Total tokens used
        };

        std::string id; ///< Unique identifier for the completion
        std::string object; ///< Object type (e.g., "chat.completion")
        long created; ///< Unix timestamp of creation
        std::string model; ///< Model used for generation
        std::vector<Choice> choices; ///< Array of generated choices
        Usage usage; ///< Token usage information

        /**

         * @brief Create a ChatCompletionResponse from JSON

         *

         * @param j JSON object containing chat completion response data

         * @return ChatCompletionResponse Parsed response object

         */

        static ChatCompletionResponse from_json(const json &j) {
            ChatCompletionResponse response;

            if (j.contains("id")) response.id = j["id"];

            if (j.contains("object")) response.object = j["object"];

            if (j.contains("created")) response.created = j["created"];

            if (j.contains("model")) response.model = j["model"];


            if (j.contains("choices") && j["choices"].is_array()) {
                for (const auto &choice: j["choices"]) {
                    Choice c{0, ChatMessage{"", ""}};

                    if (choice.contains("index")) c.index = choice["index"];


                    if (choice.contains("delta")) {
                        // Streaming response

                        if (choice["delta"].contains("content")) {
                            c.delta = choice["delta"]["content"];
                        }
                    } else if (choice.contains("message")) {
                        // Non-streaming response

                        c.message.role = choice["message"]["role"];

                        if (choice["message"].contains("content")) {
                            c.message.content = choice["message"]["content"];
                        }
                    }


                    if (choice.contains("finish_reason")) {
                        c.finish_reason = choice["finish_reason"];
                    }


                    // 解析图片数据（如果有）

                    if (choice.contains("image_url")) {
                        c.image_url = choice["image_url"];
                    }

                    if (choice.contains("image_b64")) {
                        c.image_b64 = choice["image_b64"];
                    }


                    response.choices.push_back(c);
                }
            }


            if (j.contains("usage")) {
                if (j["usage"].contains("prompt_tokens")) response.usage.prompt_tokens = j["usage"]["prompt_tokens"];

                if (j["usage"].contains("completion_tokens"))
                    response.usage.completion_tokens = j["usage"]["completion_tokens"];

                if (j["usage"].contains("total_tokens")) response.usage.total_tokens = j["usage"]["total_tokens"];
            }


            return response;
        }
    };


    /**

     * @struct ImageGenerationResponse

     * @brief Response structure for image generation API

     *

     * This structure contains the response data from an image generation request.

     */

    struct ImageGenerationResponse {
        std::string id; ///< Unique identifier for the generation

        std::string object; ///< Object type (e.g., "image")

        long created; ///< Unix timestamp of creation

        std::vector<ChatMessageContentPart> data; ///< Array of generated images

        std::string model; ///< Model used for generation


        /**

         * @brief Create an ImageGenerationResponse from JSON

         *

         * @param j JSON object containing image generation response data

         * @return ImageGenerationResponse Parsed response object

         */

        static ImageGenerationResponse from_json(const json &j) {
            ImageGenerationResponse response;

            if (j.contains("id")) response.id = j["id"];

            if (j.contains("object")) response.object = j["object"];

            if (j.contains("created")) response.created = j["created"];

            if (j.contains("model")) response.model = j["model"];


            if (j.contains("data") && j["data"].is_array()) {
                for (const auto &item: j["data"]) {
                    if (item.contains("b64_json")) {
                        // Base64 encoded image data

                        response.data.emplace_back(item["b64_json"], "image/png");
                    } else if (item.contains("url")) {
                        // Image URL

                        response.data.emplace_back(item["url"]);
                    }
                }
            }


            return response;
        }
    };


    /**

     * @struct TrainingJobRequest

     * @brief Request structure for creating a fine-tuning job

     *

     * This structure contains all parameters needed to create a fine-tuning job

     * for training a custom model.

     */
    struct TrainingJobRequest {
        std::string model; ///< Base model to fine-tune
        std::string training_file; ///< Training data file ID
        std::string validation_file; ///< Optional validation file ID
        int n_epochs = 4; ///< Number of training epochs
        double learning_rate_multiplier = 1.0; ///< Learning rate multiplier
        std::string suffix; ///< Optional suffix for the model name

        /**
         * @brief Convert the training job request to JSON format
         *
         * @return json JSON representation of the training job request
         */
        json to_json() const {
            json j;
            j["model"] = model;
            j["training_file"] = training_file;
            if (!validation_file.empty()) {
                j["validation_file"] = validation_file;
            }
            j["n_epochs"] = n_epochs;
            j["learning_rate_multiplier"] = learning_rate_multiplier;
            if (!suffix.empty()) {
                j["suffix"] = suffix;
            }
            return j;
        }
    };

    /**
     * @struct TrainingJobResponse
     * @brief Response structure for training job operations
     *
     * This structure contains information about a fine-tuning job.
     */
    struct TrainingJobResponse {
        std::string id; ///< Unique identifier for the training job
        std::string object; ///< Object type (e.g., "fine-tune")
        std::string model; ///< Base model used for fine-tuning
        long created_at; ///< Unix timestamp of job creation
        std::string status; ///< Current status of the job
        std::string fine_tuned_model; ///< Name of the resulting fine-tuned model

        /**
         * @brief Create a TrainingJobResponse from JSON
         *
         * @param j JSON object containing training job response data
         * @return TrainingJobResponse Parsed response object
         */
        static TrainingJobResponse from_json(const json &j) {
            TrainingJobResponse response;
            response.id = j["id"];
            response.object = j["object"];
            response.model = j["model"];
            response.created_at = j["created_at"];
            response.status = j["status"];
            if (j.contains("fine_tuned_model") && !j["fine_tuned_model"].is_null()) {
                response.fine_tuned_model = j["fine_tuned_model"];
            }
            return response;
        }
    };

    /**
     * @class OpenAIClient
     * @brief Abstract base class for OpenAI-compatible API clients
     *
     * This class defines the interface for interacting with OpenAI-compatible APIs.
     * It provides methods for chat completions and fine-tuning operations.
     */
    class OpenAIClient {
    private:
        std::string api_key_; ///< API key for authentication
        std::string base_url_; ///< Base URL for API endpoints
        std::string organization_; ///< Optional organization ID

        // HTTP client implementation would go here
        // For now, we'll provide a simple interface

    public:
        /**
         * @brief Construct a new OpenAIClient object
         *
         * @param api_key API key for authentication
         * @param base_url Base URL for API endpoints (default: "https://openrouter.ai/api/v1")
         * @param organization Optional organization ID
         */
        OpenAIClient(const std::string &api_key,
                     const std::string &base_url = "https://openrouter.ai/api/v1",
                     const std::string &organization = "")
            : api_key_(api_key), base_url_(base_url), organization_(organization) {
        }

        /**
         * @brief Create a chat completion
         *
         * @param request Chat completion request parameters
         * @return ChatCompletionResponse Generated chat completion response
         */
        virtual ChatCompletionResponse createChatCompletion(const ChatCompletionRequest &request) = 0;

        /**
         * @brief Create a fine-tuning job
         *
         * @param request Training job request parameters
         * @return TrainingJobResponse Created training job information
         */
        virtual TrainingJobResponse createFineTuneJob(const TrainingJobRequest &request) = 0;

        /**
         * @brief Get details of a specific fine-tuning job
         *
         * @param job_id ID of the training job to retrieve
         * @return TrainingJobResponse Training job information
         */
        virtual TrainingJobResponse getFineTuneJob(const std::string &job_id) = 0;

        /**
         * @brief List all fine-tuning jobs
         *
         * @return std::vector<TrainingJobResponse> List of training jobs
         */
        virtual std::vector<TrainingJobResponse> listFineTuneJobs() = 0;

        /**
         * @brief Cancel a fine-tuning job
         *
         * @param job_id ID of the training job to cancel
         * @return bool True if cancellation was successful
         */
        virtual bool cancelFineTuneJob(const std::string &job_id) = 0;

        /**
         * @brief Upload a file for training
         *
         * @param file_path Path to the file to upload
         * @param purpose Purpose of the file upload
         * @return std::string ID of the uploaded file
         */
        virtual std::string uploadFile(const std::string &file_path, const std::string &purpose) = 0;

        /**
         * @brief Delete a file
         *
         * @param file_id ID of the file to delete
         * @return bool True if deletion was successful
         */
        virtual bool deleteFile(const std::string &file_id) = 0;

        // Getters

        /**
         * @brief Get the API key
         *
         * @return std::string API key
         */
        std::string get_api_key() const { return api_key_; }

        /**
         * @brief Get the base URL
         *
         * @return std::string Base URL
         */
        std::string get_base_url() const { return base_url_; }

        /**
         * @brief Get the organization ID
         *
         * @return std::string Organization ID
         */
        std::string get_organization() const { return organization_; }
    };

    /**
     * @class HttpClient
     * @brief HTTP implementation of the OpenAIClient interface using libcurl
     *
     * This class provides a concrete implementation of the OpenAIClient interface
     * using libcurl for HTTP communication with OpenAI-compatible APIs.
     */
    class HttpClient : public OpenAIClient {
    private:
        // HTTP client implementation details

    public:
        /**
         * @brief Construct a new HttpClient object
         *
         * @param api_key API key for authentication
         * @param base_url Base URL for API endpoints (default: "https://openrouter.ai/api/v1")
         * @param organization Optional organization ID
         */
        HttpClient(const std::string &api_key,
                   const std::string &base_url = "https://openrouter.ai/api/v1",
                   const std::string &organization = "")
            : OpenAIClient(api_key, base_url, organization) {
            // Initialize HTTP client
        }

        /**

         * @copydoc OpenAIClient::createChatCompletion

         */

        ChatCompletionResponse createChatCompletion(const ChatCompletionRequest &request) override;


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

        void createChatCompletionStream(const ChatCompletionRequest &request,

                                        std::function<void(const ChatCompletionResponse &)> callback);

        /**
         * @copydoc OpenAIClient::createFineTuneJob
         */
        TrainingJobResponse createFineTuneJob(const TrainingJobRequest &request) override;

        /**
         * @copydoc OpenAIClient::getFineTuneJob
         */
        TrainingJobResponse getFineTuneJob(const std::string &job_id) override;

        /**
         * @copydoc OpenAIClient::listFineTuneJobs
         */
        std::vector<TrainingJobResponse> listFineTuneJobs() override;

        /**
         * @copydoc OpenAIClient::cancelFineTuneJob
         */
        bool cancelFineTuneJob(const std::string &job_id) override;

        /**
         * @copydoc OpenAIClient::uploadFile
         */
        std::string uploadFile(const std::string &file_path, const std::string &purpose) override;

        /**
         * @copydoc OpenAIClient::deleteFile
         */
        bool deleteFile(const std::string &file_id) override;
    };
} // namespace OpenAIClient

#endif // OPENAI_CLIENT_H
