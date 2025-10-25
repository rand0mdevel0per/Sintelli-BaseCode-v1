// Model Process for Neural Network Inference with WebSocket Support
// Dealer socket connecting to central router

#include <zmq.h>
#include <iostream>
#include <thread>
#include <random>
#include <chrono>
#include <string>
#include <cstring>
#include <csignal>
#include <map>
#include <atomic>
#include <functional>

// Configuration structure for model process with shared memory support
struct ModelProcessConfig {
    std::string process_id;
    std::string connection_address = "inproc://model_process"; // Default to shared memory
    int heartbeat_interval_ms = 5000;
    std::map<std::string, std::string> custom_params;
    
    // Streaming callback for long-running processing
    std::function<void(
        const std::function<bool(const std::string&)>& send_callback, // Send function that returns success status
        const std::function<std::string()>& receive_callback,         // Receive function for incoming messages
        const std::map<std::string, std::string>& params             // Custom parameters
    )> streaming_callback;
    
    // Callback for processing received commands (optional)
    std::function<void(const std::string&, const std::map<std::string, std::string>&)> command_callback;
    bool enable_websocket_keepalive;
};

class ModelProcess {
private:
    void* context;
    void* dealer_socket;
    ModelProcessConfig config;
    std::atomic<bool> running;
    std::atomic<bool> inference_in_progress;
    std::thread heartbeat_thread;
    bool inference_callback() {};

public:
    ModelProcess(const ModelProcessConfig& cfg) : 
        config(cfg),
        running(true),
        inference_in_progress(false) {
        
        // Initialize ZMQ context
        context = zmq_ctx_new();
        if (!context) {
            throw std::runtime_error("Failed to create ZMQ context");
        }
        
        // Create dealer socket
        dealer_socket = zmq_socket(context, ZMQ_DEALER);
        if (!dealer_socket) {
            zmq_ctx_destroy(context);
            throw std::runtime_error("Failed to create ZMQ dealer socket");
        }
        
        // Set identity for this dealer
        zmq_setsockopt(dealer_socket, ZMQ_IDENTITY, config.process_id.c_str(), config.process_id.size());
        
        // Set socket options for better reliability
        int linger = 0; // Don't linger on close
        zmq_setsockopt(dealer_socket, ZMQ_LINGER, &linger, sizeof(linger));
        
        int reconnect_ivl = 100; // Reconnect interval in ms
        zmq_setsockopt(dealer_socket, ZMQ_RECONNECT_IVL, &reconnect_ivl, sizeof(reconnect_ivl));
        
        // Connect using shared memory (inproc) or TCP
        int rc = zmq_connect(dealer_socket, config.connection_address.c_str());
        if (rc != 0) {
            zmq_close(dealer_socket);
            zmq_ctx_destroy(context);
            throw std::runtime_error("Failed to connect to: " + config.connection_address);
        }
        
        std::cout << "Model process " << config.process_id << " started, using connection: " 
                  << config.connection_address << "..." << std::endl;
    }

    ~ModelProcess() {
        stop();
        
        if (heartbeat_thread.joinable()) {
            heartbeat_thread.join();
        }
        
        if (dealer_socket) {
            zmq_close(dealer_socket);
        }
        if (context) {
            zmq_ctx_destroy(context);
        }
    }

    void start() {
        // Start heartbeat thread for WebSocket keepalive
        if (config.enable_websocket_keepalive) {
            heartbeat_thread = std::thread(&ModelProcess::heartbeatLoop, this);
        }
        
        std::thread send_thread(&ModelProcess::sendInferenceResults, this);
        std::thread recv_thread(&ModelProcess::receiveCommands, this);
        
        send_thread.join();
        recv_thread.join();
    }

    void stop() {
        running = false;
        inference_in_progress = false;
    }
    
    // Method to update configuration at runtime
    void updateConfig(const ModelProcessConfig& new_config) {
        config = new_config;
        std::cout << "[Config] Configuration updated for process " << config.process_id << std::endl;
    }
    
    // Method to send custom messages
    bool sendMessage(const std::string& message) {
        zmq_msg_t msg;
        zmq_msg_init_size(&msg, message.size());
        memcpy(zmq_msg_data(&msg), message.c_str(), message.size());
        
        int rc = zmq_msg_send(&msg, dealer_socket, 0);
        zmq_msg_close(&msg);
        
        if (rc >= 0) {
            std::cout << "[" << config.process_id << "] Sent custom message: " << message << std::endl;
            return true;
        }
        return false;
    }

private:
    // Heartbeat loop for connection keepalive
    void heartbeatLoop() {
        while (running) {
            std::string heartbeat_msg = "HEARTBEAT:" + config.process_id + ":" + std::to_string(std::time(nullptr));
            sendMessage(heartbeat_msg);
            std::this_thread::sleep_for(std::chrono::milliseconds(config.heartbeat_interval_ms));
        }
    }
    

    
    // Receive function callback for streaming
    std::string receiveCallback() {
        zmq_msg_t message;
        zmq_msg_init(&message);
        
        int rc = zmq_msg_recv(&message, dealer_socket, ZMQ_DONTWAIT);
        
        if (rc >= 0) {
            std::string msg(static_cast<char*>(zmq_msg_data(&message)), zmq_msg_size(&message));
            zmq_msg_close(&message);
            
            // Process command if callback is provided
            if (config.command_callback) {
                config.command_callback(msg, config.custom_params);
            }
            
            return msg;
        }
        
        zmq_msg_close(&message);
        return ""; // Empty string if no message received
    }
    
    // Send function callback for streaming
    bool sendCallback(const std::string& message) {
        return sendMessage("STREAM:" + message);
    }
    

    
    void sendInferenceResults() {
        // If streaming callback is provided, use it for long-running processing
        if (config.streaming_callback) {
            std::cout << "[" << config.process_id << "] Starting streaming processing..." << std::endl;
            
            try {
                // Pass send and receive callbacks to the streaming function
                config.streaming_callback(
                    [this](const std::string& msg) { return this->sendCallback(msg); },
                    [this]() { return this->receiveCallback(); },
                    config.custom_params
                );
            } catch (const std::exception& e) {
                std::cerr << "[ERROR] Streaming callback failed: " << e.what() << std::endl;
                sendMessage("STREAM_ERROR:" + std::string(e.what()));
            }
            
            std::cout << "[" << config.process_id << "] Streaming processing finished." << std::endl;
        } else {
            // Fallback to regular inference mode
            int message_count = 0;
            
            while (running) {
                // Fallback inference logic
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_real_distribution<float> result_dis(0.0f, 1.0f);
                
                float inference_result = result_dis(gen);
                std::string message = "FALLBACK_INFERENCE:MSG_ID:" + std::to_string(message_count) + 
                                     ":confidence=" + std::to_string(inference_result);
                
                if (sendMessage(message)) {
                    message_count++;
                }
                
                std::this_thread::sleep_for(std::chrono::milliseconds(2000)); // 2 second interval
            }
        }
    }

    void receiveCommands() {
        while (running) {
            zmq_msg_t message;
            zmq_msg_init(&message);
            
            // Try to receive message without blocking
            int rc = zmq_msg_recv(&message, dealer_socket, ZMQ_DONTWAIT);
            
            if (rc >= 0) {
                std::string msg(static_cast<char*>(zmq_msg_data(&message)), zmq_msg_size(&message));
                std::cout << "[" << config.process_id << "] Received command: " << msg << std::endl;
                
                // Process command using custom callback if available
                if (config.command_callback) {
                    try {
                        config.command_callback(msg, config.custom_params);
                    } catch (const std::exception& e) {
                        std::cerr << "[ERROR] Command callback failed: " << e.what() << std::endl;
                    }
                }
                
                // Handle built-in commands
                if (msg.find("stop") != std::string::npos) {
                    std::cout << "[" << config.process_id << "] Stopping process..." << std::endl;
                    stop();
                    zmq_msg_close(&message);
                    break;
                } else if (msg.find("update_config:") == 0) {
                    // Handle configuration updates
                    handleConfigUpdate(msg);
                } else if (msg.find("custom_inference:") == 0) {
                    // Handle custom inference requests
                    handleCustomInference(msg);
                } else if (msg.find("ping") != std::string::npos) {
                    // Respond to ping requests
                    sendMessage("pong:" + config.process_id);
                }
            }
            
            zmq_msg_close(&message);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    
    void handleConfigUpdate(const std::string& msg) {
        // Parse and update configuration from message
        // Format: update_config:param1=value1:param2=value2
        size_t pos = msg.find(':');
        if (pos != std::string::npos) {
            std::string config_str = msg.substr(pos + 1);
            std::istringstream iss(config_str);
            std::string token;
            
            while (std::getline(iss, token, ':')) {
                size_t eq_pos = token.find('=');
                if (eq_pos != std::string::npos) {
                    std::string key = token.substr(0, eq_pos);
                    std::string value = token.substr(eq_pos + 1);
                    config.custom_params[key] = value;
                }
            }
            std::cout << "[Config] Updated custom parameters" << std::endl;
        }
    }
    
    void handleCustomInference(const std::string& msg) {
        // Handle custom inference request
        // Format: custom_inference:param1=value1:param2=value2
        if (!inference_in_progress) {
            size_t pos = msg.find(':');
            if (pos != std::string::npos) {
                std::string params_str = msg.substr(pos + 1);
                std::istringstream iss(params_str);
                std::string token;
                
                std::map<std::string, std::string> temp_params = config.custom_params;
                
                while (std::getline(iss, token, ':')) {
                    size_t eq_pos = token.find('=');
                    if (eq_pos != std::string::npos) {
                        std::string key = token.substr(0, eq_pos);
                        std::string value = token.substr(eq_pos + 1);
                        temp_params[key] = value;
                    }
                }
                
                // Perform inference with custom parameters
                if (config.inference_callback) {
                    try {
                        std::string result = config.inference_callback(temp_params);
                        sendMessage("CUSTOM_INFERENCE_RESULT:" + result);
                    } catch (const std::exception& e) {
                        sendMessage("CUSTOM_INFERENCE_ERROR:" + std::string(e.what()));
                    }
                }
            }
        }
    }
};

// Example streaming callback for long-running processing
void exampleStreamingCallback(
    const std::function<bool(const std::string&)>& send_func,
    const std::function<std::string()>& receive_func,
    const std::map<std::string, std::string>& params
) {
    std::cout << "Starting streaming processing with params: " << params.size() << " parameters" << std::endl;
    
    int chunk_count = 0;
    const int max_chunks = params.count("max_chunks") ? std::stoi(params.at("max_chunks")) : 100;
    
    while (chunk_count < max_chunks) {
        // Send a chunk of data
        std::string chunk = "CHUNK_" + std::to_string(chunk_count) + 
                           ":DATA_SIZE=" + std::to_string(chunk_count * 1024) + 
                           ":TIMESTAMP=" + std::to_string(std::time(nullptr));
        
        if (!send_func(chunk)) {
            std::cerr << "Failed to send chunk " << chunk_count << std::endl;
            break;
        }
        
        // Check for incoming messages
        std::string received = receive_func();
        if (!received.empty()) {
            std::cout << "Received during streaming: " << received << std::endl;
            
            if (received.find("stop") != std::string::npos) {
                std::cout << "Received stop signal, ending streaming" << std::endl;
                break;
            }
        }
        
        chunk_count++;
        std::this_thread::sleep_for(std::chrono::milliseconds(500)); // Simulate processing time
    }
    
    send_func("STREAMING_COMPLETE:chunks=" + std::to_string(chunk_count));
}

// Simple example of a streaming function that can be used
void exampleStreamingFunction(
    const std::function<bool(const std::string&)>& send,
    const std::function<std::string()>& receive,
    const std::map<std::string, std::string>& params
) {
    std::cout << "[Streaming] Starting streaming processing" << std::endl;
    
    int step = 0;
    while (step < 50) { // Run for 50 steps
        // Send progress update
        std::string message = "PROGRESS:step=" + std::to_string(step) + 
                             ":timestamp=" + std::to_string(std::time(nullptr));
        
        if (!send(message)) {
            std::cout << "[Streaming] Failed to send message, stopping" << std::endl;
            break;
        }
        
        // Check for incoming commands
        std::string cmd = receive();
        if (!cmd.empty()) {
            std::cout << "[Streaming] Received command: " << cmd << std::endl;
            
            if (cmd.find("stop") != std::string::npos) {
                send("Streaming stopped by command");
                break;
            }
        }
        
        step++;
        std::this_thread::sleep_for(std::chrono::milliseconds(200)); // Simulate work
    }
    
    send("STREAMING_COMPLETE");
}

// Example function showing how to use ModelProcess in shared memory mode internally
void exampleSharedMemoryUsage() {
    std::cout << "=== Shared Memory Usage Example ===" << std::endl;
    
    // Create configuration for shared memory communication
    ModelProcessConfig config;
    config.process_id = "internal_model";
    config.connection_address = "inproc://neural_network";
    
    // Set up streaming callback for long-running inference
    config.streaming_callback = [](const auto& send, const auto& receive, const auto& params) {
        std::cout << "[Internal Streaming] Starting neural network inference" << std::endl;
        
        // Simulate neural network processing with shared memory optimization
        for (int i = 0; i < 20; i++) {
            // Send inference results
            std::string result = "INFERENCE:batch=" + std::to_string(i) + 
                               ":confidence=" + std::to_string(0.8 + (i * 0.01)) +
                               ":latency=5ms";
            
            if (!send(result)) {
                std::cout << "[Internal] Failed to send result" << std::endl;
                break;
            }
            
            // Check for control messages
            std::string cmd = receive();
            if (!cmd.empty() && cmd.find("stop") != std::string::npos) {
                send("INFERENCE_STOPPED");
                break;
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        send("INFERENCE_COMPLETE");
    };
    
    // Set up command callback for receiving instructions
    config.command_callback = [](const std::string& cmd, const auto& params) {
        std::cout << "[Command Handler] Received: " << cmd << std::endl;
        
        if (cmd.find("update_model") != std::string::npos) {
            std::cout << "[Command Handler] Model update requested" << std::endl;
        }
    };
    
    try {
        // Create and start the model process
        ModelProcess process(config);
        
        // Start in a separate thread for internal usage
        std::thread process_thread([&process]() {
            process.start();
        });
        
        std::cout << "Model process started in shared memory mode" << std::endl;
        std::cout << "Press Ctrl+C to stop" << std::endl;
        
        // Wait for the process thread
        process_thread.join();
        
    } catch (const std::exception& e) {
        std::cerr << "Error in shared memory example: " << e.what() << std::endl;
    }
}

// Simple main function for testing - can be removed in production
int main(int argc, char* argv[]) {
    if (argc == 1) {
        // Demo internal shared memory usage
        exampleSharedMemoryUsage();
        return 0;
    }
    
    if (argc < 2) {
        std::cout << "Usage examples:" << std::endl;
        std::cout << "  Demo: " << argv[0] << "           (runs shared memory example)" << std::endl;
        std::cout << "  Custom: " << argv[0] << " <process_id> [options]" << std::endl;
        std::cout << "Options:" << std::endl;
        std::cout << "  --stream     Enable streaming mode" << std::endl;
        std::cout << "  --inproc=ADDR Set inproc address (default: inproc://model_process)" << std::endl;
        std::cout << "  --tcp=ADDR    Set TCP address" << std::endl;
        return 1;
    }
    
    try {
        ModelProcessConfig config;
        config.process_id = argv[1];
        
        // Parse command line arguments
        for (int i = 2; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--stream") {
                config.streaming_callback = exampleStreamingFunction;
                config.custom_params["mode"] = "streaming";
                std::cout << "[Config] Using streaming mode" << std::endl;
            } else if (arg.find("--inproc=") == 0) {
                config.connection_address = arg.substr(9);
            } else if (arg.find("--tcp=") == 0) {
                config.connection_address = arg.substr(6);
            }
        }
        
        ModelProcess process(config);
        
        // Handle Ctrl+C gracefully
        signal(SIGINT, [](int) { 
            std::cout << "\nReceived interrupt signal, shutting down..." << std::endl;
            exit(0);
        });
        
        std::cout << "Starting model process with ID: " << config.process_id << std::endl;
        std::cout << "Connection address: " << config.connection_address << std::endl;
        
        process.start();
        
    } catch (const std::exception& e) {
        std::cerr << "Model process error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}