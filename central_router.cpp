// Central Router Process for Multi-process Communication
// Router-Dealer pattern: Multiple model processes -> Central router

#include <zmq.h>
#include <iostream>
#include <thread>
#include <unordered_map>
#include <vector>
#include <string>
#include <cstring>

class CentralRouter {
private:
    void* context;
    void* router_socket;  // For model processes (ROUTER)
    void* dealer_socket;  // For Rust server (DEALER)
    std::vector<std::string> connected_workers;
    std::atomic<bool> running;

public:
    CentralRouter() : running(true) {
        // Initialize ZMQ context
        context = zmq_ctx_new();
        if (!context) {
            throw std::runtime_error("Failed to create ZMQ context");
        }

        // Create ROUTER socket for model processes
        router_socket = zmq_socket(context, ZMQ_ROUTER);
        if (!router_socket) {
            zmq_ctx_destroy(context);
            throw std::runtime_error("Failed to create ROUTER socket: " + std::to_string(zmq_errno()));
        }

        // Create DEALER socket for Rust server
        dealer_socket = zmq_socket(context, ZMQ_DEALER);
        if (!dealer_socket) {
            zmq_close(router_socket);
            zmq_ctx_destroy(context);
            throw std::runtime_error("Failed to create DEALER socket: " + std::to_string(zmq_errno()));
        }

        // Bind sockets
        if (zmq_bind(router_socket, "tcp://*:5555") != 0) {
            zmq_close(dealer_socket);
            zmq_close(router_socket);
            zmq_ctx_destroy(context);
            throw std::runtime_error("Failed to bind ROUTER socket: " + std::to_string(zmq_errno()));
        }

        if (zmq_bind(dealer_socket, "tcp://*:5556") != 0) {
            zmq_close(dealer_socket);
            zmq_close(router_socket);
            zmq_ctx_destroy(context);
            throw std::runtime_error("Failed to bind DEALER socket: " + std::to_string(zmq_errno()));
        }

        std::cout << "Central Router started on ports 5555 (models) and 5556 (Rust)" << std::endl;
    }

    ~CentralRouter() {
        stop();
        if (dealer_socket) zmq_close(dealer_socket);
        if (router_socket) zmq_close(router_socket);
        if (context) zmq_ctx_destroy(context);
    }

    void start() {
        std::thread router_thread(&CentralRouter::handleModelProcesses, this);
        std::thread dealer_thread(&CentralRouter::handleRustServer, this);
        
        router_thread.join();
        dealer_thread.join();
    }

    void stop() {
        running = false;
    }

private:
    void handleModelProcesses() {
        zmq_pollitem_t items[] = {
            { router_socket, 0, ZMQ_POLLIN, 0 },
            { dealer_socket, 0, ZMQ_POLLIN, 0 }
        };

        while (running) {
            // Poll for incoming messages
            int rc = zmq_poll(items, 2, 100); // 100ms timeout
            if (rc == -1) {
                if (running) {
                    std::cerr << "Poll error: " << zmq_strerror(zmq_errno()) << std::endl;
                }
                break;
            }

            // Handle messages from model processes
            if (items[0].revents & ZMQ_POLLIN) {
                handleModelMessage();
            }

            // Handle messages from Rust server
            if (items[1].revents & ZMQ_POLLIN) {
                handleServerMessage();
            }
        }
    }

    void handleModelMessage() {
        // Receive identity frame
        zmq_msg_t identity;
        zmq_msg_init(&identity);
        if (zmq_msg_recv(&identity, router_socket, 0) == -1) {
            zmq_msg_close(&identity);
            return;
        }

        // Receive message frame
        zmq_msg_t message;
        zmq_msg_init(&message);
        if (zmq_msg_recv(&message, router_socket, 0) == -1) {
            zmq_msg_close(&identity);
            zmq_msg_close(&message);
            return;
        }

        std::string worker_id(static_cast<char*>(zmq_msg_data(&identity)), zmq_msg_size(&identity));
        std::string msg(static_cast<char*>(zmq_msg_data(&message)), zmq_msg_size(&message));

        // Track connected workers
        if (std::find(connected_workers.begin(), connected_workers.end(), worker_id) == connected_workers.end()) {
            connected_workers.push_back(worker_id);
            std::cout << "New model process connected: " << worker_id << std::endl;
        }

        std::cout << "From model " << worker_id << ": " << msg << std::endl;

        // Forward to Rust server with worker ID prefix
        std::string forward_msg = "[" + worker_id + "] " + msg;
        sendToServer(forward_msg);

        zmq_msg_close(&identity);
        zmq_msg_close(&message);
    }

    void handleServerMessage() {
        zmq_msg_t message;
        zmq_msg_init(&message);
        if (zmq_msg_recv(&message, dealer_socket, 0) == -1) {
            zmq_msg_close(&message);
            return;
        }

        std::string msg(static_cast<char*>(zmq_msg_data(&message)), zmq_msg_size(&message));
        std::cout << "From server: " << msg << std::endl;

        // Broadcast to all connected model processes
        if (!connected_workers.empty()) {
            for (const auto& worker_id : connected_workers) {
                sendToModel(worker_id, msg);
            }
            std::cout << "Broadcasted to " << connected_workers.size() << " model processes" << std::endl;
        }

        zmq_msg_close(&message);
    }

    void sendToServer(const std::string& message) {
        zmq_msg_t msg;
        zmq_msg_init_size(&msg, message.size());
        memcpy(zmq_msg_data(&msg), message.c_str(), message.size());
        
        if (zmq_msg_send(&msg, dealer_socket, 0) == -1) {
            std::cerr << "Failed to send to server: " << zmq_strerror(zmq_errno()) << std::endl;
        }
        
        zmq_msg_close(&msg);
    }

    void sendToModel(const std::string& worker_id, const std::string& message) {
        // Send identity frame
        zmq_msg_t identity;
        zmq_msg_init_size(&identity, worker_id.size());
        memcpy(zmq_msg_data(&identity), worker_id.c_str(), worker_id.size());
        
        if (zmq_msg_send(&identity, router_socket, ZMQ_SNDMORE) == -1) {
            zmq_msg_close(&identity);
            std::cerr << "Failed to send identity: " << zmq_strerror(zmq_errno()) << std::endl;
            return;
        }

        // Send message frame
        zmq_msg_t msg;
        zmq_msg_init_size(&msg, message.size());
        memcpy(zmq_msg_data(&msg), message.c_str(), message.size());
        
        if (zmq_msg_send(&msg, router_socket, 0) == -1) {
            zmq_msg_close(&msg);
            std::cerr << "Failed to send message: " << zmq_strerror(zmq_errno()) << std::endl;
        }

        zmq_msg_close(&identity);
        zmq_msg_close(&msg);
    }

    void handleRustServer() {
        // This method is kept for backward compatibility
        handleModelProcesses();
    }
};

/*
int main() {
    try {
        CentralRouter router;
        std::cout << "Starting central router..." << std::endl;
        std::cout << "Press Ctrl+C to stop" << std::endl;
        
        router.start();
    } catch (const std::exception& e) {
        std::cerr << "Router error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
*/