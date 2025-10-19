//
// Created by ASUS on 10/19/2025.
//
// ========== python_binding.cu ==========
#include <Python.h>
#include "NeuronModel.cu"
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>

// 全局模型和输出队列
static NeuronModel<32> *g_model = nullptr;
static std::queue<std::string> g_output_queue;
static std::mutex g_output_mutex;
static std::condition_variable g_output_cv;
static bool g_model_running = false;

// 输出收集线程
static std::thread g_collector_thread;
static bool g_collector_running = false;

// 输出收集函数（后台持续收集）
void output_collector_loop() {
    while (g_collector_running) {
        if (g_model && g_model_running) {
            auto output = g_model->getoutput();

            if (output.has_text && !output.text.empty()) {
                std::lock_guard<std::mutex> lock(g_output_mutex);
                g_output_queue.push(output.text);
                g_output_cv.notify_one();
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

// ========== Python接口函数 ==========

// 1. 创建模型
static PyObject* py_create_model(PyObject* self, PyObject* args) {
    int grid_size = 32;

    if (!PyArg_ParseTuple(args, "|i", &grid_size)) {
        std::cout << "WARN: Using default grid size 32" << std::endl;
        grid_size = 32;
    }

    try {
        // 清理旧模型
        if (g_model) {
            g_model->stop();
            delete g_model;
        }

        // 创建新模型
        g_model = new NeuronModel(grid_size);

        Py_RETURN_TRUE;

    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

// 2. 启动模型
static PyObject* py_start_model(PyObject* self, PyObject* args) {
    if (!g_model) {
        PyErr_SetString(PyExc_RuntimeError, "Model not created");
        return NULL;
    }

    try {
        // 启动模型
        bool success = g_model->run();
        if (!success) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to start model");
            return NULL;
        }

        g_model_running = true;

        // 启动输出收集线程
        if (!g_collector_running) {
            g_collector_running = true;
            g_collector_thread = std::thread(output_collector_loop);
        }

        Py_RETURN_TRUE;

    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

// 3. 输入文本
static PyObject* py_input(PyObject* self, PyObject* args) {
    try {
        const char* text;
        const char* img_base64 = nullptr;
        InputMessage msg;
        msg.has_text = false;
        msg.has_img = false;
        if (PyArg_ParseTuple(args, "s", &text)) {
            msg.has_text = true;
            msg.text = std::string(text);
        }

        if (PyArg_ParseTuple(args, "s", &img_base64)) {
            msg.has_img = true;
            msg.base64_image = std::string(img_base64);
        }

        if (!g_model || !g_model_running) {
            PyErr_SetString(PyExc_RuntimeError, "Model not running");
            return NULL;
        }

        bool success = g_model->input(msg);

        return PyBool_FromLong(success);

    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

// 4. 流式获取输出（非阻塞）
static PyObject* py_get_next_output(PyObject* self, PyObject* args) {
    double timeout_sec = 1.0;

    if (!PyArg_ParseTuple(args, "|d", &timeout_sec)) {
        return NULL;
    }

    std::unique_lock<std::mutex> lock(g_output_mutex);

    // 等待输出或超时
    if (g_output_cv.wait_for(lock,
                             std::chrono::milliseconds((int)(timeout_sec * 1000)),
                             []{ return !g_output_queue.empty(); })) {
        // 有输出
        std::string output = g_output_queue.front();
        g_output_queue.pop();

        return PyUnicode_FromString(output.c_str());
    }

    // 超时，返回None
    Py_RETURN_NONE;
}

// 5. 检查是否有输出可用
static PyObject* py_has_output(PyObject* self, PyObject* args) {
    std::lock_guard<std::mutex> lock(g_output_mutex);
    return PyBool_FromLong(!g_output_queue.empty());
}

// 6. 清空输出队列
static PyObject* py_clear_outputs(PyObject* self, PyObject* args) {
    std::lock_guard<std::mutex> lock(g_output_mutex);
    while (!g_output_queue.empty()) {
        g_output_queue.pop();
    }
    Py_RETURN_NONE;
}

// 7. 停止模型
static PyObject* py_stop_model(PyObject* self, PyObject* args) {
    if (g_model) {
        g_model->stop();
        g_model_running = false;
    }

    // 停止收集线程
    if (g_collector_running) {
        g_collector_running = false;
        if (g_collector_thread.joinable()) {
            g_collector_thread.join();
        }
    }

    Py_RETURN_NONE;
}

// 8. 销毁模型
static PyObject* py_destroy_model(PyObject* self, PyObject* args) {
    py_stop_model(self, args);

    if (g_model) {
        delete g_model;
        g_model = nullptr;
    }

    Py_RETURN_NONE;
}

// ========== 模块定义 ==========

static PyMethodDef NeuronMethods[] = {
    {"create_model32", py_create_model32, METH_VARARGS,
     "Create neuron model\n\nArgs:\n    grid_size (int): Grid size (default 32)"},

    {"start_model", py_start_model, METH_VARARGS,
     "Start the model"},

    {"input", py_input, METH_VARARGS,
     "Input content to model\n\nArgs:\n    text (str): Input text [optional] , image (str): Input image(base64)[optional]"},

    {"get_next_output", py_get_next_output, METH_VARARGS,
     "Get next output (blocking with timeout)\n\nArgs:\n    timeout (float): Timeout in seconds (default 1.0)\n\nReturns:\n    str or None"},

    {"has_output", py_has_output, METH_VARARGS,
     "Check if output is available\n\nReturns:\n    bool"},

    {"clear_outputs", py_clear_outputs, METH_VARARGS,
     "Clear output queue"},

    {"stop_model", py_stop_model, METH_VARARGS,
     "Stop the model"},

    {"destroy_model", py_destroy_model, METH_VARARGS,
     "Destroy the model"},

    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef neuronmodule = {
    PyModuleDef_HEAD_INIT,
    "neuron_model",
    "Distributed Neuron Network with streaming output",
    -1,
    NeuronMethods
};

PyMODINIT_FUNC PyInit_neuron_model(void) {
    return PyModule_Create(&neuronmodule);
}