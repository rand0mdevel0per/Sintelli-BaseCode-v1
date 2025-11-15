//
// Created by ASUS on 11/11/2025.
//

#include "NeuronModel.cu"
#include <cstring>

extern "C" {

// Model management
void* ffi_create_model(const ull grid_size) {
    try {
        return new NeuronModel(grid_size);
    } catch (...) {
        return nullptr;
    }
}

void* ffi_load_model(const char* path) {
    try {
        return new NeuronModel(std::string(path));
    } catch (...) {
        return nullptr;
    }
}

void ffi_destroy_model(void* model) {
    if (model) {
        auto* m = static_cast<NeuronModel*>(model);
        m->stop();
        delete m;
    }
}

bool ffi_save_model(void* model, const char* path) {
    if (!model) return false;
    auto* m = static_cast<NeuronModel*>(model);
    try {
        return path ? m->save(std::string(path)) : m->save();
    } catch (...) {
        return false;
    }
}

// Model operations
bool ffi_start_model(void* model) {
    if (!model) return false;
    auto* m = static_cast<NeuronModel*>(model);
    return m->run();
}

void ffi_stop_model(void* model) {
    if (model) {
        static_cast<NeuronModel*>(model)->stop();
    }
}

bool ffi_input(void* model, const InputMessage* msg, const char* role) {
    if (!model || !msg) return false;
    auto* m = static_cast<NeuronModel*>(model);
    
    InputMessage cpp_msg;
    cpp_msg.has_text = msg->has_text;
    cpp_msg.has_img = msg->has_img;
    
    if (msg->has_text) {
        cpp_msg.text = std::string(msg->text);
    }
    if (msg->has_img) {
        cpp_msg.base64_image = std::string(msg->base64_image);
    }
    
    return m->input(cpp_msg, role ? std::string(role) : "user");
}

// Output handling
InputMessage* ffi_get_output(void* model) {
    if (!model) return nullptr;
    auto* m = static_cast<NeuronModel*>(model);
    
    auto output = m->getoutput();
    auto* result = new InputMessage();
    
    result->has_text = output.has_text;
    result->has_img = output.has_img;
    
    if (output.has_text) {
        result->text = _strdup(output.text.c_str());
    } else {
        result->text = nullptr;
    }
    
    if (output.has_img) {
        result->base64_image = _strdup(output.base64_image.c_str());
    } else {
        result->base64_image = nullptr;
    }
    
    return result;
}

void ffi_free_output(const InputMessage* output) {
    if (output) {
        if (output->has_text) free(const_cast<char*>(output->text.c_str()));
        if (output->has_img) free(const_cast<char*>(output->base64_image.c_str()));
        delete output;
    }
}

bool ffi_has_output(void* model) {
    // Implement based on your queue logic
    return false;
}

// Training
void ffi_enable_training(void* model) {
    if (model) {
        static_cast<NeuronModel*>(model)->enable_training_mode();
    }
}

void ffi_disable_training(void* model) {
    if (model) {
        static_cast<NeuronModel*>(model)->disable_training_mode();
    }
}

bool ffi_set_score(void* model, double score) {
    if (!model) return false;
    return static_cast<NeuronModel*>(model)->update_score(score);
}

// Stats
const char* ffi_get_neuron_stats(void* model, unsigned long long neuron_id) {
    if (!model) return nullptr;
    auto* m = static_cast<NeuronModel*>(model);
    try {
        auto stats = m->get_n_stats(neuron_id);
        const auto json_str = stats.to_json().dump();
        return _strdup(json_str.c_str());
    } catch (...) {
        return nullptr;
    }
}

void ffi_free_string(const char* s) {
    if (s) free(const_cast<char*>(s));
}

} // extern "C"