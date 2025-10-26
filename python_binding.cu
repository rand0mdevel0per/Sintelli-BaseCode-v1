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
#include "training_interval.cu"

// 全局模型和输出队列
static NeuronModel *g_model = nullptr;
static std::queue<std::string> g_output_queue;
static std::queue<std::string> g_output_img_queue;
static std::mutex g_output_mutex;
static std::mutex g_output_img_mut;
static std::condition_variable g_output_cv;
static std::condition_variable g_output_img_cv;
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

            if (output.has_img && !output.base64_image.empty()) {
                std::lock_guard<std::mutex> lock(g_output_img_mut);
                g_output_img_queue.push(output.base64_image);
                g_output_img_cv.notify_one();
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

// ========== Python接口函数 ==========

// 1. 创建模型
static PyObject *py_create_model(PyObject *self, PyObject *args) {
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
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

static PyObject *py_load_model(PyObject *self, PyObject *args) {
    const char *path;
    if (!PyArg_ParseTuple(args, "|s", &path)) {
        std::cout << "WARN: Using default path..." << std::endl;
        path = "";
    }
    try {
        // 清理旧模型
        if (g_model) {
            g_model->stop();
            delete g_model;
        }

        // 创建新模型
        g_model = new NeuronModel(std::string(path));

        Py_RETURN_TRUE;
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

// 2. Start model
static PyObject *py_start_model(PyObject *self, PyObject *args) {
    if (!g_model) {
        PyErr_SetString(PyExc_RuntimeError, "Model not created");
        return nullptr;
    }

    try {
        // Start the neuron model
        bool success = g_model->run();
        if (!success) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to start model");
            return nullptr;
        }

        g_model_running = true;

        // Start output collector thread
        if (!g_collector_running) {
            g_collector_running = true;
            g_collector_thread = std::thread(output_collector_loop);
        }

        Py_RETURN_TRUE;
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

// 3. Input
static PyObject *py_input(PyObject *self, PyObject *args) {
    try {
        const char *text = nullptr;
        const char *img_base64 = nullptr;
        const char *role = "user";

        // Parse arguments: text, image, role are all optional
        if (!PyArg_ParseTuple(args, "|sss", &text, &img_base64, &role)) {
            return nullptr; // Argument parsing failed
        }

        InputMessage msg;
        msg.has_text = (text != nullptr && strlen(text) > 0);
        msg.has_img = (img_base64 != nullptr && strlen(img_base64) > 0);

        if (msg.has_text) {
            msg.text = std::string(text);
        }

        if (msg.has_img) {
            msg.base64_image = std::string(img_base64);
        }

        if (!g_model || !g_model_running) {
            PyErr_SetString(PyExc_RuntimeError, "Model not running");
            return nullptr;
        }

        bool success = g_model->input(msg, std::string(role));

        return PyBool_FromLong(success);
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

// 4. Get next output (streaming, non-blocking)
static PyObject *py_get_next_output(PyObject *self, PyObject *args) {
    double timeout_sec = 1.0;

    if (!PyArg_ParseTuple(args, "|d", &timeout_sec)) {
        return nullptr;
    }

    std::unique_lock<std::mutex> lock(g_output_mutex);

    // Wait for output or timeout
    if (g_output_cv.wait_for(lock,
                             std::chrono::milliseconds((int) (timeout_sec * 1000)),
                             [] { return !g_output_queue.empty(); })) {
        // Output available
        std::string output = g_output_queue.front();
        g_output_queue.pop();

        return PyUnicode_FromString(output.c_str());
    }

    // Timeout, return None
    Py_RETURN_NONE;
}

// 5. Set score for training
static PyObject *py_set_score(PyObject *self, PyObject *args) {
    double score = 1.0;

    if (!PyArg_ParseTuple(args, "|d", &score)) {
        return nullptr;
    }

    try {
        return PyBool_FromLong(g_model->update_score(score));
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

// 6. Enable training mode
static PyObject *py_enable_training_mode(PyObject *self, PyObject *args) {
    try {
        g_model->enable_training_mode();
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
    return PyBool_FromLong(true);
}

// 7. Disable training mode
static PyObject *py_disable_training_mode(PyObject *self, PyObject *args) {
    try {
        g_model->disable_training_mode();
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
    return PyBool_FromLong(true);
}

// 8. Check if output is available
static PyObject *py_has_output(PyObject *self, PyObject *args) {
    std::lock_guard<std::mutex> lock(g_output_mutex);
    return PyBool_FromLong(!g_output_queue.empty());
}

// 9. Clear output queue
static PyObject *py_clear_outputs(PyObject *self, PyObject *args) {
    std::lock_guard<std::mutex> lock(g_output_mutex);
    while (!g_output_queue.empty()) {
        g_output_queue.pop();
    }
    Py_RETURN_NONE;
}

// 10. Get next image output (streaming, non-blocking)
static PyObject *py_get_next_output_img(PyObject *self, PyObject *args) {
    double timeout_sec = 1.0;

    if (!PyArg_ParseTuple(args, "|d", &timeout_sec)) {
        return nullptr;
    }

    std::unique_lock<std::mutex> lock(g_output_img_mut);

    // Wait for output or timeout
    if (g_output_img_cv.wait_for(lock,
                                 std::chrono::milliseconds((int) (timeout_sec * 1000)),
                                 [] { return !g_output_img_queue.empty(); })) {
        // Output available
        std::string output = g_output_img_queue.front();
        g_output_img_queue.pop();

        return PyUnicode_FromString(output.c_str());
    }

    // Timeout, return None
    Py_RETURN_NONE;
}

// 11. Check if image output is available
static PyObject *py_has_output_img(PyObject *self, PyObject *args) {
    std::lock_guard<std::mutex> lock(g_output_img_mut);
    return PyBool_FromLong(!g_output_img_queue.empty());
}

// 12. Clear image output queue
static PyObject *py_clear_outputs_img(PyObject *self, PyObject *args) {
    std::lock_guard<std::mutex> lock(g_output_img_mut);
    while (!g_output_img_queue.empty()) {
        g_output_img_queue.pop();
    }
    Py_RETURN_NONE;
}

// 13. Stop model
static PyObject *py_stop_model(PyObject *self, PyObject *args) {
    if (g_model) {
        g_model->stop();
        g_model_running = false;
    }

    // Stop collector thread
    if (g_collector_running) {
        g_collector_running = false;
        if (g_collector_thread.joinable()) {
            g_collector_thread.join();
        }
    }

    Py_RETURN_NONE;
}

// 14. Destroy model
static PyObject *py_destroy_model(PyObject *self, PyObject *args) {
    py_stop_model(self, args);

    if (g_model) {
        delete g_model;
        g_model = nullptr;
    }

    Py_RETURN_NONE;
}

//15. Save Model
static PyObject *py_save_model(PyObject *self, PyObject *args) {
    const char *path = "";

    if (!PyArg_ParseTuple(args, "|s", &path)) {
        cout << "WARN: Using default path" << std::endl;
        path = "";
    }

    try {
        if (path != "") {
            return PyBool_FromLong(g_model->save(std::string(path)));
        }
        return PyBool_FromLong(g_model->save());
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

std::thread thd_training;
bool stop_training = true;

static PyObject *py_run_native_training(PyObject *self, PyObject *args) {
    const char *bing_api = "";
    const char *google_api = "";
    const char *google_eng = "";
    const char *openrouter_api = "";
    const char *name = "Sydney";

    if (!stop_training) Py_RETURN_FALSE;

    if (!PyArg_ParseTuple(args, "|sssss", bing_api, google_api, google_eng, openrouter_api, name)) {
        if (bing_api == "") {
            cout << "WARN:Missing bing api key" << endl;
        }
        if (google_api == "" || google_eng == "") {
            cout << "WARN:Missing google api key" << endl;
        }
        if (openrouter_api == "") {
            cerr << "Error:Missing openrouter api key" << endl;
            PyErr_SetString(PyExc_RuntimeError, "Missing OpenRouter API Key");
            return nullptr;
        }
    }

    try{
        thd_training = std::thread([bing_api, google_api, google_eng, openrouter_api, name]() {
           stop_training = false;
           run_training(g_model, bing_api, name, google_api, google_eng, openrouter_api, &stop_training);
       });
        //thd_training.detach();
    } catch (...) {
        Py_RETURN_FALSE;
    }
    Py_RETURN_TRUE;
}

static PyObject *py_stop_native_training(PyObject *self, PyObject *args) {
    if (!stop_training) {
        try{
            stop_training = true;
            thd_training.join();
        } catch (...) {
            Py_RETURN_FALSE;
        }
    } else Py_RETURN_FALSE;
    Py_RETURN_TRUE;
}

// ========== 模块定义 ==========

static PyMethodDef NeuronMethods[] = {
    {
        "create_model", py_create_model, METH_VARARGS,
        "Create neuron model\n"
        "\n"
        "Args:\n"
        "    grid_size (int): Grid size (default 32)"
    },

    {
        "start_model", py_start_model, METH_VARARGS,
        "Start the model"
    },

    {
        "input", py_input, METH_VARARGS,
        "Input content to model\n"
        "\n"
        "Args:\n"
        "    text (str, optional): Input text\n"
        "    image (str, optional): Input image (base64)\n"
        "    role (str, optional): User role\n"
        "\n"
        "Returns:\n"
        "    bool: True if success"
    },

    {
        "get_next_output", py_get_next_output, METH_VARARGS,
        "Get next text output (blocking with timeout)\n"
        "\n"
        "Args:\n"
        "    timeout (float): Timeout in seconds (default 1.0)\n"
        "\n"
        "Returns:\n"
        "    str or None"
    },

    {
        "has_output", py_has_output, METH_VARARGS,
        "Check if text output is available\n"
        "\n"
        "Returns:\n"
        "    bool: True if available"
    },

    {
        "clear_outputs", py_clear_outputs, METH_VARARGS,
        "Clear text output queue"
    },

    {
        "get_next_output_img", py_get_next_output_img, METH_VARARGS,
        "Get next image output (blocking with timeout)\n"
        "\n"
        "Args:\n"
        "    timeout (float): Timeout in seconds (default 1.0)\n"
        "\n"
        "Returns:\n"
        "    str or None"
    },

    {
        "has_output_img", py_has_output_img, METH_VARARGS,
        "Check if image output is available\n"
        "\n"
        "Returns:\n"
        "    bool: True if available"
    },

    {
        "clear_outputs_img", py_clear_outputs_img, METH_VARARGS,
        "Clear image output queue"
    },

    {
        "stop_model", py_stop_model, METH_VARARGS,
        "Stop the model"
    },

    {
        "destroy_model", py_destroy_model, METH_VARARGS,
        "Destroy the model"
    },

    {
        "set_score", py_set_score, METH_VARARGS,
        "Set output scores when training\n"
        "\n"
        "Args:\n"
        "    score (float): Current score for the output\n"
        "\n"
        "Returns:\n"
        "    bool: True if success"
    },

    {
        "enable_training_mode", py_enable_training_mode, METH_VARARGS,
        "Enable training mode for the model\n"
        "\n"
        "Returns:\n"
        "    bool: True if success"
    },

    {
        "disable_training_mode", py_disable_training_mode, METH_VARARGS,
        "Disable training mode for the model\n"
        "\n"
        "Returns:\n"
        "    bool: True if success"
    },

    {
        "load_model_from_file", py_load_model, METH_VARARGS,
        "Load model from a .nm2 file\n\n"
        "Args:\n"
        "   path(str):The path of .nm2 model file \n"
        "\n"
        "Returns:\n"
        "   bool: True if success"
    },

    {
        "save_model_from_file", py_save_model, METH_VARARGS,
        "Save model to a .nm2 file\n"
        "\n"
        "Args:\n"
        "   path(str): The path of .nm2 model file \n"
        "\n"
        "Returns:\n"
        "   bool: True if success"
    },

    {
        "run_native_training", py_run_native_training, METH_VARARGS,
        "Run CUDA C++ side builtin native training\n"
        "\n"
        "Args:\n"
        "   bing_api_key(str): Bing Custom Search API Key\n"
        "   google_api_key(str): Google Custom Search API Key\n"
        "   google_engine_id(str): Google Cloud Engine Id\n"
        "   openrouter_api_key(str):OpenRouter API Key\n"
        "\n"
        "Returns:\n"
        "   bool: True if success"
    },

    {"stop_native_training", py_stop_native_training, METH_VARARGS,
        "Stop CUDA C++ side builtin native training\n"
        "\n"
        "Returns:\n"
        "   bool: True if success"
    },

    {nullptr, nullptr, 0, nullptr}
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
