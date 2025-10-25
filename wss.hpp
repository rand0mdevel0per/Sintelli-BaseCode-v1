#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <thread>
#include <future>
#include <utility>
#include <chrono>
#include <algorithm>
#include <map>
#include <memory>
#include <iterator>
#include "queue.cpp"

using namespace std;


// 修复：在Crow之前定义这些宏来避免模板冲突
#define CROW_MAIN
#define CROW_STATIC_DIRECTORY "static/"
#define CROW_STATIC_ENDPOINT "/static/<path>"

// 修复：禁用Crow的一些可能冲突的特性
#define CROW_ENABLE_COMPRESSION
#define CROW_ENABLE_SSL

// 修复：WebSocket宏必须在crow_all.h之前定义
#define CROW_ENABLE_WEBSOCKETS
#include "crow_all.h"

// --- 类型定义 ---
using DataPacket = string;
using RecvCallback = function<DataPacket(void)>;
using SendCallback = function<void(const DataPacket&)>;
using HandlerFunction = function<void(RecvCallback, SendCallback)>;

// --- 修复：改进的线程安全队列 ---

// --- 修复：改进的异步调用器 ---

/**
 * @brief 异步全双工 Handler 调用器（修复future问题）
 */
class AsyncDuplexHandlerInvoker {
private:
    RecvCallback _recv_func;
    SendCallback _send_func;

public:
    AsyncDuplexHandlerInvoker(RecvCallback recv_cb, SendCallback send_cb)
        : _recv_func(std::move(recv_cb)), _send_func(std::move(send_cb)) {}

    // 修复：返回shared_future避免移动问题
    void invoke_handler(HandlerFunction handler) {
        // 直接在新线程中运行，不返回 future
        thread([this, handler = std::move(handler)]() {
            std::cout << "[Invoker] 异步 Handler 任务启动..." << std::endl;
            handler(_recv_func, _send_func);
            std::cout << "[Invoker] 异步 Handler 任务结束" << std::endl;
        }).detach();
    }
};

// --- 用户自定义的 Handler 逻辑 ---

/**
 * @brief 用户自定义的业务 Handler 函数（修复字符串操作）
 */
void user_custom_handler(RecvCallback recv, SendCallback send) {
    std::cout << "\n[Handler] 业务逻辑线程启动，等待客户端连接..." << std::endl;

    // 修复：使用std::string构造而不是字符串字面量拼接
    send(std::string("Server_Ready: WebSocket通道已建立。"));

    int counter = 0;
    while (true) {
        DataPacket incoming;
        if (!recv) {
            std::cout << "[Handler] 接收函数无效，退出。" << std::endl;
            break;
        }

        incoming = recv();

        if (incoming.empty()) {
            std::cout << "[Handler] 接收通道已关闭 (收到空包)。Handler 退出。" << std::endl;
            break;
        }

        std::cout << "[Handler] 收到客户端消息: '" << incoming << "'" << std::endl;

        // 修复：改进字符串构造
        std::string response = "ECHO " + std::to_string(++counter) + ": " + incoming;
        send(response);

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    std::cout << "[Handler] 业务逻辑执行完毕。" << std::endl;
}