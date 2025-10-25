//
// Created by ASUS on 10/3/2025.
//
// NeuronModel template class for managing 3D grid of neurons
// This class handles neuron allocation and inter-neuron connectivity

#ifndef SRC_NEURONMODEL_CUH
#define SRC_NEURONMODEL_CUH

#include "Neuron.cu"
#include <cuda_runtime.h>
#include <thread>
#include <mutex>
#include "isw.hpp"
#include "sct.hpp"
#include "dslzma.h"
#include "smry.cpp"
#include "feature_extractor.h"
#include "semantic_matcher.h"
#include "semantic_query_interface.h"
#include "semantic_matcher.cpp"
#include "structs.h"  // Contains KFE_STM_Slot definition
#include "deviceQueue.cpp"  // Contains DeviceQueue template definition
#include "KFEManager.h"    // KFE storage manager
#include "memory_slot.h"
#include <Windows.h>
#include "rag_knowledge_loader.cpp"
#include "huggingface_rag_integration.cpp"
#include <queue>

#define ll long long
#define ull unsigned ll

// Forward declaration of NeuronModel class
class NeuronModel;


/**
 * @brief Template class for managing a 3D grid of neurons.
 *
 * This class handles:
 * - Allocation and initialization of neurons.
 * - Inter-neuron connectivity.
 * - Integration with semantic matching systems.
 * - Serialization and deserialization of the model state.
 *
 */
class NeuronModel {
public:
    /**
     * @brief Default constructor - initializes the neuron grid with default 32x32x32 size.
     *
     * Allocates memory for neurons and sets up semantic matching components.
     * Initializes all required subsystems including E5 model, feature extractors, and storage.
     */
    NeuronModel() : processor(e5), logic_processor(e5), memory_processor(e5), sct_processor(e5), cache_processor(e5) {
        NeuronModel(32);
    }

    /**
     * @brief Constructor with custom grid size.
     *
     * Allocates memory for neurons and sets up semantic matching components.
     * Initializes all required subsystems including E5 model, feature extractors, and storage.
     * @param grid_size Size of the 3D neuron grid (grid_size x grid_size x grid_size)
     */
    NeuronModel(ull grid_size) : processor(e5), logic_processor(e5), memory_processor(e5), sct_processor(e5),
                                 cache_processor(e5) {
        if (grid_size != 0) {
            GRID_SIZE = grid_size;
        }

        // Initialize computation-related constants
        NEURON_COUNT = GRID_SIZE * GRID_SIZE * GRID_SIZE;
        NUM_BLOCKS = (NEURON_COUNT + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        // Initialize queue array
        queues.resize(GRID_SIZE);
        for (ull i = 0; i < GRID_SIZE; i++) {
            queues[i].resize(GRID_SIZE);
            for (ull j = 0; j < GRID_SIZE; j++) {
                queues[i][j].resize(GRID_SIZE);
            }
        }

        cudaMalloc(&d_active_flags, NEURON_COUNT * sizeof(bool));
        for (int i = 0; i < 4; i++) {
            cudaStreamCreate(&streams[i]);
        }

        // Initialize: all activated
        cudaMemset(d_active_flags, 1, NEURON_COUNT * sizeof(bool));
        static E5LargeModel e5_instance("/models/e5/e5_large.onnx", "/models/vocab.json", "/models/merges.txt",
                                        "/models/special_tokens.json");
        e5 = &e5_instance;
        processor = UnifiedInputProcessor(e5);
        logic_processor = UnifiedInputProcessor(e5);
        memory_processor = UnifiedInputProcessor(e5);
        sct_processor = UnifiedInputProcessor(e5);
        cache_processor = UnifiedInputProcessor(e5);
        cudaMallocManaged(&d_neurons, GRID_SIZE * GRID_SIZE * GRID_SIZE * sizeof(Neuron));

        // Initialize semantic matching system
        feature_extractor = std::make_unique<FeatureExtractor>(
            "/models/e5/e5_large.onnx", "/models/vocab.json",
            "/models/merges.txt", "/models/special_tokens.json");
        semantic_matcher = std::make_unique<SemanticMatcher>(
            "/models/e5/e5_large.onnx", "/models/vocab.json",
            "/models/merges.txt", "/models/special_tokens.json");

        // Create external storage for semantic matching
        static ExternalStorage<SemanticMatch> semantic_storage(1024, 100.0, 1.0);
        integrated_matcher = std::make_unique<IntegratedSemanticMatcher>(&semantic_storage);

        // Initialize Logic semantic matching system
        logic_matcher = std::make_unique<LogicSemanticMatcher>(
            "/models/e5/e5_large.onnx", "/models/vocab.json",
            "/models/merges.txt", "/models/special_tokens.json");

        static ExternalStorage<LogicDescriptor> logic_storage(512, 50.0, 0.5);
        logic_injector = std::make_unique<LogicInjector>(&logic_storage,
                                                         "/models/e5/e5_large.onnx");

        static ExternalStorage<LogicDescriptor> memory_logic_storage(512, 50.0, 0.5);
        memory_injector = std::make_unique<LogicInjector>(&memory_logic_storage,
                                                          "/models/e5/e5_large.onnx");

        // Create KFE storage queues (shared by all neurons)
        static DeviceQueue<KFE_STM_Slot, 32> kfe_storage_queue;
        static DeviceQueue<std::string, 32> kfe_query_queue;
        static DeviceQueue<KFE_STM_Slot, 32> kfe_result_queue;

        // Start KFE manager
        static KFEManager kfe_manager(&kfe_storage_queue, &kfe_query_queue, &kfe_result_queue);

        for (int idx = 0; idx < GRID_SIZE * GRID_SIZE * GRID_SIZE; idx++) {
            ll x = idx / (GRID_SIZE * GRID_SIZE);
            ll y = (idx / GRID_SIZE) % GRID_SIZE;
            ll z = idx % GRID_SIZE;

            ll coord[3] = {x, y, z};
            // Set up neighbor queue connections
            DeviceQueue<Message, 32> *neighbour_q[6] = {nullptr};
            ull seed = x * 1000000 + y * 1000 + z;

            // +X direction
            if (x < GRID_SIZE - 1) {
                neighbour_q[0] = &queues[x + 1][y][z]; // 指向邻居的-X队列
            } else {
                neighbour_q[0] = &queues[x - 1][y][z]; // 指向邻居的+X队列
            }
            // -X direction
            if (x > 0) {
                neighbour_q[1] = &queues[x - 1][y][z]; // 指向邻居的+X队列
            } else {
                neighbour_q[1] = &queues[x + 1][y][z]; // 指向邻居的-X队列
            }
            // +Y direction
            if (y < GRID_SIZE - 1) {
                neighbour_q[2] = &queues[x][y + 1][z];
            } else {
                neighbour_q[2] = &queues[x][y - 1][z];
            }
            // -Y direction
            if (y > 0) {
                neighbour_q[3] = &queues[x][y - 1][z];
            } else {
                neighbour_q[3] = &queues[x][y + 1][z];
            }
            // +Z direction
            if (z < GRID_SIZE - 1) {
                neighbour_q[4] = &queues[x][y][z + 1];
            } else {
                neighbour_q[4] = &queues[x][y][z - 1];
            }
            // -Z direction
            if (z > 0) {
                neighbour_q[5] = &queues[x][y][z - 1];
            } else {
                neighbour_q[5] = &queues[x][y][z + 1];
            }

            new(&d_neurons[idx]) Neuron(neighbour_q, coord, seed, &queues[x][y][z],
                                        &kfe_storage_queue, &kfe_query_queue, &kfe_result_queue);
            // placement new
        }

        std::cout << "Distributed neuron system initialization completed!" << std::endl;
    }

    explicit NeuronModel(const std::string& path) : processor(e5), logic_processor(e5), memory_processor(e5),
                                             sct_processor(e5), cache_processor(e5) {
        // Use delegating constructor
        NeuronModel(2);
        path_default = path;
        if (!load(path)) {
            throw std::runtime_error("Failed to load model from path: " + path);
        }
    }

    /**
     * @brief Start the neuron network execution.
     *
     * Launches the main processing loop in a separate thread.
     * Initializes the event loop and starts neural processing.
     * @return true if successfully started, false otherwise
     */
    bool run() {
        try {
            is_running = true;
            eventloop = std::thread([this]() {
                this->loop();
            });
            eventloop.detach();
            std::cout << "Neuron network is running..." << std::endl;
            std::thread(loop).detach();
            return true;
        } catch (...) {
            return false;
        }
    }

    bool save(std::string path = "") {
        if (path.empty()) path = path_default;
        try {
            // Reset pointers before saving
            resetPointersForSerialization();

            std::vector<NeuronData> ndata;

            for (ull i = 0; i < GRID_SIZE ^ 3; i++) {
                ndata.push_back(d_neurons[i].save());
            }

            std::pair<NeuronModel, std::vector<NeuronData> > nmdata;
            nmdata.first.copyFrom(*this);
            nmdata.second = ndata;

            bool result = Serializer<std::pair<NeuronModel, std::vector<NeuronData> > >::save(nmdata, path);

            // Restore pointer connections after saving
            rebuildPointerConnections();

            return result;
        } catch (...) {
            return false;
        }
    }

    bool load(std::string path = "") {
        if (path.empty()) path = path_default;
        try {
            std::pair<NeuronModel, std::vector<NeuronData> > nmdata;
            if (!Serializer<std::pair<NeuronModel, std::vector<NeuronData> > >::load(nmdata, path)) {
                return false;
            }
            this->copyFrom(nmdata.first);
            for (ull i = 0; i < nmdata.second.size(); i++) {
                d_neurons[i].load(nmdata.second[i]);
            }
            resetPointersForSerialization();
            rebuildPointerConnections();
            return true;
        } catch (...) {
            return false;
        }
    }

    // ===== External Storage Integration =====
    // This section manages integration with external storage systems
    // Provides methods for data persistence and retrieval from tiered storage
    // This section provides interfaces for semantic text processing and matching
    // It integrates with the E5 language model for text embedding and similarity computation

    // Register text to semantic matching system
    uint64_t registerTextForMatching(const std::string &text, const std::string &category = "") {
        if (!integrated_matcher) return 0;
        return integrated_matcher->registerAndStore(text, category);
    }

    // Batch register texts
    std::vector<uint64_t> batchRegisterTexts(const std::vector<std::string> &texts,
                                             const std::string &category = "") {
        if (!semantic_matcher) return {};
        return semantic_matcher->batchRegisterTexts(texts, category);
    }

    // Semantic search
    std::vector<SemanticMatch> semanticSearch(const std::string &query_text,
                                              int top_k = 10,
                                              double similarity_threshold = 0.5,
                                              const std::string &metric = "cosine") {
        if (!semantic_matcher) return {};
        return semantic_matcher->findSimilarTexts(query_text, top_k, similarity_threshold, metric);
    }

    // Semantic search by category
    std::vector<SemanticMatch> semanticSearchByCategory(const std::string &query_text,
                                                        const std::string &category,
                                                        int top_k = 10,
                                                        double similarity_threshold = 0.5) {
        if (!semantic_matcher) return {};
        return semantic_matcher->findSimilarTextsByCategory(query_text, category, top_k, similarity_threshold);
    }

    // Get semantic matching statistics
    SemanticMatcher::Stats getSemanticMatchingStats() {
        if (!semantic_matcher) return SemanticMatcher::Stats();
        return semantic_matcher->getStats();
    }

    // Get hottest semantic matching results
    std::vector<SemanticMatch> getHottestSemanticMatches(int k = 10) {
        if (!integrated_matcher) return {};
        return integrated_matcher->getHottestMatches(k);
    }

    // Extract text feature vector
    FeatureVector<float> extractTextFeature(const std::string &text) {
        if (!feature_extractor) return FeatureVector<float>();
        return feature_extractor->extractTextFeature(text);
    }

    // Calculate text similarity
    double calculateTextSimilarity(const std::string &text1, const std::string &text2,
                                   const std::string &metric = "cosine") {
        if (!feature_extractor) return -1.0;
        auto feature1 = feature_extractor->extractTextFeature(text1);
        auto feature2 = feature_extractor->extractTextFeature(text2);
        return feature_extractor->calculateSimilarity(feature1, feature2, metric);
    }

    // ===== Logic Semantic Matching Interface =====
    // This section handles Logic-based semantic matching and injection
    // Logic descriptors are registered and matched against input text for neural processing

    // Register Logic to matching system
    bool registerLogic(const std::string &logic_id,
                       const std::string &description,
                       const std::string &category = "",
                       double activation_threshold = 0.5,
                       std::function<void(const std::string &, NeuronInput &)> generate_input_callback = nullptr) {
        if (!logic_matcher) return false;

        LogicDescriptor logic_desc(logic_id, description, category, activation_threshold);
        logic_desc.generate_input_callback = generate_input_callback;

        return logic_injector->registerLogicWithStorage(logic_desc);
    }

    // Find matching Logic based on text query
    std::vector<std::pair<LogicDescriptor, double> > findMatchingLogics(
        const std::string &query_text,
        int top_k = 5,
        double similarity_threshold = 0.3,
        const std::string &category = "") {
        if (!logic_matcher) return {};
        return logic_matcher->findMatchingLogics(query_text, top_k, similarity_threshold, category);
    }

    // Inject matching Logic (execute callback and return NeuronInput list)
    std::vector<std::pair<std::string, NeuronInput> > injectMatchingLogics(
        const std::string &query_text,
        int top_k = 3,
        double similarity_threshold = 0.4) {
        if (!logic_injector) return {};
        return logic_injector->injectMatchingLogics(query_text, top_k, similarity_threshold);
    }

    // Process user input and automatically inject matching Logic to specified neuron
    // This function finds matching Logic based on input text and injects it into the specified neuron
    bool processInputWithLogicInjection(const std::string &input_text,
                                        int neuron_index = 0,
                                        int port = 0,
                                        int max_logics = 3,
                                        double similarity_threshold = 0.4) {
        if (!logic_injector || input_text.empty()) return false;

        auto activated_logics = logic_injector->injectMatchingLogics(
            input_text, max_logics, similarity_threshold);

        // Inject matching Logic into neuron
        for (const auto &[logic_id, neuron_input]: activated_logics) {
            if (neuron_index >= 0 && neuron_index < GRID_SIZE * GRID_SIZE * GRID_SIZE) {
                d_neurons[neuron_index].inject(neuron_input, port);
                std::cout << "Injecting Logic: " << logic_id << " to neuron " << neuron_index
                        << " port " << port << std::endl;
            }
        }

        return !activated_logics.empty();
    }

    // Get Logic statistics
    LogicSemanticMatcher::LogicStats getLogicStats() {
        if (!logic_matcher) return {};
        return logic_matcher->getLogicStats();
    }

    // Set Logic callback function
    bool setLogicCallback(const std::string &logic_id,
                          std::function<void(const std::string &, NeuronInput &)> callback) {
        if (!logic_matcher) return false;
        return logic_matcher->setLogicCallback(logic_id, callback);
    }

    // Set simple Logic callback function (version without parameters)
    bool setSimpleLogicCallback(const std::string &logic_id, std::function<void()> callback) {
        if (!logic_matcher) return false;
        return logic_matcher->setSimpleLogicCallback(logic_id, callback);
    }

    // Remove Logic
    bool removeLogic(const std::string &logic_id) {
        if (!logic_matcher) return false;
        return logic_matcher->removeLogic(logic_id);
    }

    bool input(const InputMessage &msg, const std::string &role) {
        try {
            if (msg.has_img) {
                ImageData img = base64_to_image(msg.base64_image);
                double img_data[256][256];
                memcpy(img_data, ImageProcessor::encode(&img)->data, sizeof(img_data));
                NeuronInput img_inp{};
                memcpy(img_inp.array, img_data, sizeof(img_data));
                img_inp.activity = 1.0;
                img_inp.from_coord[0] = 0;
                img_inp.from_coord[1] = 0;
                img_inp.from_coord[2] = 0;
                img_inp.weight = 1.0;
                d_neurons[0].inject(img_inp, 1);
            }
            if (msg.has_text) {
                processTextString(&processor, "<User:" + role + "> " + msg.text);
                std::vector<float> txt_d = feature_extractor->extractTextFeature(msg.text).data;
                auto *text_data = new float[txt_d.size()];
                memcpy(text_data, txt_d.data(), txt_d.size() * sizeof(float));
                auto text_data_d = new double[txt_d.size()];
                for (size_t i = 0; i < txt_d.size(); i++) {
                    text_data_d[i] = static_cast<double>(txt_d[i]);
                }
                double query_emb[128];
                std::vector<float> emb_flt = feature_extractor->extractTextFeature(msg.text).data;
                for (size_t i = 0; i < 128; i++) {
                    query_emb[i] = static_cast<double>(emb_flt[i]);
                }
                std::thread([this, &query_emb, msg, text_data_d, role]() {
                    // Recall top-10 related texts
                    auto indices = sct.recallTopK(query_emb, 10);
                    auto contents = sct.getRecalledTexts(indices);
                    for (const auto &ct: contents) {
                        processTextString(&sct_processor, ct);
                    }
                    sct.addTurn("<User:" + role + "> " + msg.text, text_data_d);
                }).detach();
                auto input_emb = feature_extractor->extractTextFeature(msg.text);
                auto matched_logics = logic_injector->findMatchingLogicIds(msg.text);
                if (matched_logics.size() <= 5) {
                    std::thread([this, msg, input_emb]() {
                        try {
                            rag_loader.autoFetchAndRegisterLogic(logic_injector.get(), &logic_tree, msg.text);
                            rag_loader.autoFetchAndRegisterLogic(logic_injector.get(), &logic_tree, msg.text, 10,
                                                                 "HuggingFaceFW/finepdfs", "eng_Latn",
                                                                 "general");
                            const std::vector<std::string> all_subsets_maths = {
                                "Deepseek-Math-RL-7B",
                                "Deepseek-Math-RL-7B-T=1.1",
                                "Deepseek-Math-RL-7B-T=1.3",
                                "InternLM2-Math-Plus-7B",
                                "InternLM2-Math-Plus-7B-T=1.1",
                                "InternLM2-Math-Plus-7B-T=1.3",
                                "InternLM2-Math-Plus-1.8B",
                                "InternLM2-Math-Plus-1.8B-T=1.1"
                            };
                            if (input_emb.cosineSimilarity(feature_extractor->extractTextFeature("Maths")) > 0.6) {
                                for (const auto &ss: all_subsets_maths) {
                                    rag_loader.autoFetchAndRegisterLogic(
                                        logic_injector.get(), &logic_tree, msg.text, 10, "WNJXYK/MATH-Reasoning-Paths",
                                        ss, "maths");
                                }
                            }
                            if (input_emb.cosineSimilarity(feature_extractor->extractTextFeature("education")) > 0.6) {
                                rag_loader.autoFetchAndRegisterLogic(logic_injector.get(), &logic_tree, msg.text, 10,
                                                                     "karpathy/fineweb-edu-100b-shuffle", "",
                                                                     "education");
                            }
                            if (input_emb.cosineSimilarity(feature_extractor->extractTextFeature("coding")) > 0.6) {
                                rag_loader.autoFetchAndRegisterLogic(logic_injector.get(), &logic_tree, msg.text, 10,
                                                                     "nick007x/github-code-2025", "above-2-stars",
                                                                     "coding");
                            }
                            if (input_emb.cosineSimilarity(feature_extractor->extractTextFeature("cybersecurity")) >
                                0.6) {
                                rag_loader.autoFetchAndRegisterLogic(logic_injector.get(), &logic_tree, msg.text, 10,
                                                                     "ethanolivertroy/nist-cybersecurity-training", "",
                                                                     "cybersecurity");
                            }
                        } catch (...) {
                            std::cerr << "WARN: RAG AutoLoading Failed" << std::endl;
                        }
                    }).detach();
                }
                std::thread([this, matched_logics]() {
                    for (const auto &logic_id: matched_logics) {
                        Logic curr_logic{};
                        logic_tree.fetchByHash(logic_id.first, curr_logic);
                        char *curr_logic_str = wideCharToMultiByte(curr_logic.content);
                        processTextString(&logic_processor, std::string(curr_logic_str));
                        delete[] curr_logic_str;
                    }
                }).detach();
                std::thread([this, msg]() {
                    auto matched_memories = memory_injector->findMatchingLogicIds(msg.text);
                    for (const auto &key: matched_memories | views::keys) {
                        MemorySlot curr_memory;
                        memory_tree.fetchByHash(key, curr_memory);
                        const char *curr_memory_str = reinterpret_cast<const char *>(curr_memory.content.c_str);
                        processTextString(&memory_processor, std::string(curr_memory_str));
                        delete[] curr_memory_str;
                    }
                }).detach();
            }
            return true;
        } catch (...) {
            return false;
        }
    }

    InputMessage getoutput() {
        std::lock_guard lock(msg_mutex);
        if (!output_msgs.empty()) {
            InputMessage msg = output_msgs.back();
            output_msgs.pop_back();
            return msg;
        }
        return InputMessage{false, false, "", ""};
    }

    bool stop() {
        try {
            is_running = false;
            // Wait for all streams to complete
            // Synchronize all CUDA streams to ensure all GPU operations are finished
            for (auto & stream : streams) {
                cudaStreamSynchronize(stream);
            }
            eventloop.join();
            return true;
        } catch (...) {
            return false;
        }
    }

    ~NeuronModel() {
        std::cout << "Shutting down neuron network..." << std::endl;
        this->stop();
        // 等待所有流完成
        for (const auto &stream: streams) {
            cudaStreamSynchronize(stream);
        }
        if (d_neurons) {
            // Call destructor for each neuron
            // Explicitly call destructors for all neurons in the 3D grid before freeing memory
            ull total_neurons = GRID_SIZE * GRID_SIZE * GRID_SIZE;
            for (ull i = 0; i < total_neurons; i++) {
                d_neurons[i].~Neuron();
            }
            cudaFree(d_neurons);
            d_neurons = nullptr;
        }
    }

    ull get_size() const { return GRID_SIZE; }

    void enable_training_mode() {
        training = true;
    }

    void disable_training_mode() {
        training = false;
    }

    bool update_score(double score_) {
        if (!training) return false;
        try{
            ull neuron_count = GRID_SIZE ^ 3;
            ull threads_per_block = 256;
            ull blocks = (neuron_count + threads_per_block - 1) / threads_per_block;
            reset_trace<<<blocks, threads_per_block>>>(d_trace);
            score = score_;
        } catch (...) {
            return false;
        }
        return true;
    }

    std::queue<std::string> get_cache() {
        try {
            return {this->cache_queue};
        } catch (...) {
            return {};
        }
    }

private:
    ull GRID_SIZE = 32;
    cudaStream_t streams[4];
    __managed__ bool *d_active_flags;
    __managed__ double *d_trace;
    std::thread eventloop;
    ull NEURON_COUNT = 0; // Will be initialized in constructor
    ull THREADS_PER_BLOCK = 256;
    ull NUM_BLOCKS = 0; // Will be initialized in constructor
    std::vector<std::vector<std::vector<DeviceQueue<Message, 32> > > > queues; // 使用vector代替固定数组
    Neuron *d_neurons{};
    ExternalStorage<KFE_STM_Slot> ext_kfe{};
    SemanticConversationTree sct{};
    ExternalStorage<Logic> logic_tree{};
    ExternalStorage<MemorySlot> memory_tree{};
    std::string path_default = "./models/Si/model.nm2";
    UnifiedInputProcessor processor;
    E5LargeModel *e5;
    std::vector<InputMessage> output_msgs;
    bool is_running = false;
    std::mutex msg_mutex;
    std::vector<std::string> current_logic;
    std::queue<std::string> cache_queue;
    std::string output_cache = "<You> ";

    // 语义匹配相关成员
    std::unique_ptr<FeatureExtractor> feature_extractor;
    std::unique_ptr<SemanticMatcher> semantic_matcher;
    std::unique_ptr<IntegratedSemanticMatcher> integrated_matcher;
    std::unique_ptr<LogicSemanticMatcher> logic_matcher;
    std::unique_ptr<LogicInjector> logic_injector;
    std::unique_ptr<LogicInjector> memory_injector;

    UnifiedInputProcessor logic_processor;
    UnifiedInputProcessor memory_processor;
    UnifiedInputProcessor sct_processor;
    UnifiedInputProcessor cache_processor;

    RAGKnowledgeBaseLoader rag_loader;

    __managed__ double score;
    __managed__ bool training;

    /**
     * @brief Main processing loop for the neuron network.
     *
     * This function runs continuously while the network is active.
     * It processes neuron inputs, manages data flow between components,
     * handles semantic matching, Logic injection, and output generation.
     *
     * The loop performs the following operations:
     * 1. Processes input blocks from various processors
     * 2. Executes neuron computations in parallel streams
     * 3. Manages matrix data flow between neurons
     * 4. Handles semantic matching and Logic injection
     * 5. Processes memory and cache operations
     * 6. Generates output messages
     */
    __host__ void loop() {
        double next_block[256][256]{};
        Matrix256 previous_matrix_0{};
        Matrix256 current_matrix_0{};
        Matrix256 previous_matrix{};
        Matrix256 current_matrix{};
        Matrix256 curr_logic_mat{};
        Matrix256 prev_logic_mat{};
        Matrix256 prev_mem_mat{};
        Matrix256 curr_mem_mat{};
        Matrix256 prev_find_mem_mat{};
        Matrix256 curr_find_mem_mat{};
        Matrix256 prev_find_logic_mat{};
        Matrix256 curr_find_logic_mat{};
        Matrix256 prev_cache_mat{};
        Matrix256 curr_cache_mat{};
        Matrix256 prev_find_cache_mat{};
        Matrix256 curr_find_cache_mat{};
        Matrix256 prev_find_sct_mat{};
        Matrix256 curr_find_sct_mat{};
        InputMessage cache_msg{};
        std::string find_cache;
        std::string find_mem;
        std::string mem;
        std::string find_logic;
        std::string cache;
        std::string find_sct;
        Matrix256 matrix_cache_img[16];
        int curr_img_mat = 0;
        while (!is_running) {
            std::this_thread::sleep_for(std::chrono::nanoseconds(100));
        }
        while (is_running) {
            const Matrix256 *next_matrix = processor.getNextBlock();
            if (next_matrix != nullptr) {
                NeuronInput next_inp{};
                memcpy(next_block, next_matrix->data, sizeof(next_block));
                memcpy(next_inp.array, next_block, sizeof(next_block));
                next_inp.activity = 1.0;
                next_inp.from_coord[0] = 0;
                next_inp.from_coord[1] = 0;
                next_inp.from_coord[2] = 0;
                next_inp.weight = 1.0;
                d_neurons[0].inject(next_inp, 0);
                delete next_matrix;
                next_matrix = nullptr;
            }
            Matrix256 *next_inject_mt = nullptr;
            for (int i = 0; i < 4; i++) {
                if (logic_processor.hasMoreBlocks()) {
                    next_inject_mt = logic_processor.getNextBlock();
                    if (next_inject_mt != nullptr) {
                        NeuronInput logic_inp{};
                        memcpy(next_block, next_inject_mt->data, sizeof(next_block));
                        memcpy(logic_inp.array, next_block, sizeof(next_block));
                        logic_inp.activity = 1.0;
                        logic_inp.from_coord[0] = 0;
                        logic_inp.from_coord[1] = 0;
                        logic_inp.from_coord[2] = 0;
                        logic_inp.weight = 1.0;
                        d_neurons[4].inject(logic_inp, i);
                    }
                }
            }
            for (int i = 0; i < 4; i++) {
                if (sct_processor.hasMoreBlocks()) {
                    next_inject_mt = sct_processor.getNextBlock();
                    if (next_inject_mt != nullptr) {
                        NeuronInput sct_inp{};
                        memcpy(next_block, next_inject_mt->data, sizeof(next_block));
                        memcpy(sct_inp.array, next_block, sizeof(next_block));
                        sct_inp.activity = 1.0;
                        sct_inp.from_coord[0] = 0;
                        sct_inp.from_coord[1] = 0;
                        sct_inp.from_coord[2] = 0;
                        sct_inp.weight = 1.0;
                        d_neurons[5].inject(sct_inp, i);
                    }
                }
            }
            for (int i = 0; i < 4; i++) {
                if (memory_processor.hasMoreBlocks()) {
                    next_inject_mt = memory_processor.getNextBlock();
                    if (next_inject_mt != nullptr) {
                        NeuronInput memory_inp{};
                        memcpy(next_block, next_inject_mt->data, sizeof(next_block));
                        memcpy(memory_inp.array, next_block, sizeof(next_block));
                        memory_inp.activity = 1.0;
                        memory_inp.from_coord[0] = 0;
                        memory_inp.from_coord[1] = 0;
                        memory_inp.from_coord[2] = 0;
                        memory_inp.weight = 1.0;
                        d_neurons[6].inject(memory_inp, i);
                    }
                }
            }
            for (int i = 0; i < 4; i++) {
                if (cache_processor.hasMoreBlocks()) {
                    next_inject_mt = cache_processor.getNextBlock();
                    if (next_inject_mt != nullptr) {
                        NeuronInput cache_inp{};
                        memcpy(next_block, next_inject_mt->data, sizeof(next_block));
                        memcpy(cache_inp.array, next_block, sizeof(next_block));
                        cache_inp.activity = 1.0;
                        cache_inp.from_coord[0] = 0;
                        cache_inp.from_coord[1] = 0;
                        cache_inp.from_coord[2] = 0;
                        cache_inp.weight = 1.0;
                        d_neurons[11].inject(cache_inp, i);
                    }
                }
            }
            if (next_inject_mt != nullptr) {
                delete next_inject_mt;
                next_inject_mt = nullptr;
            }
            // Divide 32768 neurons into 4 groups, each using one stream
            ull neurons_per_stream = NEURON_COUNT / 4;

            for (int i = 0; i < 4; i++) {
                ull offset = i * neurons_per_stream;
                ull count = (i == 3) ? (NEURON_COUNT - offset) : neurons_per_stream;

                // Asynchronous launch (no waiting)
                all_neurons_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK, 0, streams[i]>>>(
                    d_neurons + offset,
                    d_active_flags + offset,
                    count
                );
            }
            memcpy(previous_matrix.data, current_matrix.data, sizeof(previous_matrix.data));
            memcpy(current_matrix.data, d_neurons[1].detach(0).array, sizeof(current_matrix.data));
            memcpy(previous_matrix_0.data, current_matrix_0.data, sizeof(previous_matrix_0.data));
            memcpy(current_matrix_0.data, d_neurons[1].detach(1).array, sizeof(current_matrix_0.data));
            memcpy(prev_cache_mat.data, curr_cache_mat.data, sizeof(prev_cache_mat.data));
            memcpy(curr_cache_mat.data, d_neurons[5].detach(0).array, sizeof(curr_cache_mat.data));
            memcpy(prev_find_logic_mat.data, curr_find_logic_mat.data, sizeof(prev_find_logic_mat.data));
            memcpy(curr_find_logic_mat.data, d_neurons[6].detach(1).array, sizeof(curr_find_logic_mat.data));
            memcpy(prev_find_mem_mat.data, curr_find_mem_mat.data, sizeof(prev_find_mem_mat.data));
            memcpy(curr_find_mem_mat.data, d_neurons[7].detach(1).array, sizeof(curr_find_mem_mat.data));
            memcpy(prev_find_sct_mat.data, curr_find_sct_mat.data, sizeof(prev_find_sct_mat.data));
            memcpy(curr_find_sct_mat.data, d_neurons[8].detach(0).array, sizeof(curr_find_sct_mat.data));
            memcpy(prev_mem_mat.data, curr_mem_mat.data, sizeof(prev_mem_mat.data));
            memcpy(curr_mem_mat.data, d_neurons[9].detach(0).array, sizeof(curr_mem_mat.data));
            memcpy(prev_find_cache_mat.data, curr_find_cache_mat.data, sizeof(prev_find_cache_mat.data));
            memcpy(curr_find_cache_mat.data, d_neurons[10].detach(1).array, sizeof(curr_find_cache_mat.data));
            cache_msg = InputMessage{false, false, "", ""};
            if (ImageProcessor::isValidFrame(&current_matrix, &previous_matrix)) {
                std::string text = matrix_to_string(&current_matrix);
                cache_msg.has_text = true;
                cache_msg.text = text;
                output_cache.append(text);
            } else if (!output_cache.empty()) {
                auto output_ch = output_cache;
                std::thread([this, output_ch]() {
                    const auto fea = feature_extractor->extractTextFeature(output_ch).data;
                    double txt_emb[128]{};
                    for (int i = 0; i < 128; i++) {
                        txt_emb[i] = static_cast<double>(fea[i]);
                    }
                    sct.addTurn(output_cache, txt_emb);
                }).detach();
                output_cache.clear();
                output_cache = "<You> ";
            }
            if (ImageProcessor::isValidFrame(&current_matrix_0, &previous_matrix_0)) {
                matrix_cache_img[curr_img_mat] = current_matrix_0;
                curr_img_mat++;
                if (curr_img_mat == 16) {
                    curr_img_mat = 0;
                    // Convert to pointer array
                    Matrix256 *frame_ptrs[16];
                    for (int i = 0; i < 16; i++) {
                        frame_ptrs[i] = &matrix_cache_img[i];
                    }
                    ImageData *img_ptr = ImageProcessor::mergeFrames(frame_ptrs, 16);
                    if (img_ptr) {
                        std::string base64_img = image_to_base64(*img_ptr);
                        cache_msg.has_img = true;
                        cache_msg.base64_image = base64_img;
                        delete img_ptr;
                    }
                }
            }
            if (cache_msg.has_img || cache_msg.has_text) {
                std::lock_guard<std::mutex> lock(msg_mutex);
                output_msgs.push_back(cache_msg);
            }
            if (next_matrix != nullptr) {
                delete next_matrix;
                next_matrix = nullptr;
            }
            memcpy(prev_logic_mat.data, curr_logic_mat.data, sizeof(prev_logic_mat.data));
            memcpy(curr_logic_mat.data, d_neurons[2].detach(0).array, sizeof(curr_logic_mat.data));
            if (ImageProcessor::isValidFrame(&curr_logic_mat, &prev_logic_mat)) {
                std::string logic_str = matrix_to_string(&curr_logic_mat);
                current_logic.push_back(logic_str);
            } else if ((!current_logic.empty()) || current_logic.size() > 16) {
                std::string query_logic;
                for (const auto &i: current_logic) {
                    query_logic.append(i);
                }

                current_logic.clear();
            }
            if (ImageProcessor::isValidFrame(&curr_find_logic_mat, &prev_find_logic_mat)) {
                std::string logic_str = matrix_to_string(&curr_find_logic_mat);
                find_logic.append(logic_str);
            } else if (!find_logic.empty()) {
                auto input_emb = feature_extractor->extractTextFeature(find_logic);
                auto matched_logics = logic_injector->findMatchingLogicIds(find_logic);
                if (matched_logics.size() <= 5) {
                    std::thread([this, find_logic, input_emb]() {
                        try {
                            rag_loader.autoFetchAndRegisterLogic(logic_injector.get(), &logic_tree, find_logic);
                            rag_loader.autoFetchAndRegisterLogic(logic_injector.get(), &logic_tree, find_logic, 10,
                                                                 "HuggingFaceFW/finepdfs", "eng_Latn",
                                                                 "general");
                            const std::vector<std::string> all_subsets_maths = {
                                "Deepseek-Math-RL-7B",
                                "Deepseek-Math-RL-7B-T=1.1",
                                "Deepseek-Math-RL-7B-T=1.3",
                                "InternLM2-Math-Plus-7B",
                                "InternLM2-Math-Plus-7B-T=1.1",
                                "InternLM2-Math-Plus-7B-T=1.3",
                                "InternLM2-Math-Plus-1.8B",
                                "InternLM2-Math-Plus-1.8B-T=1.1"
                            };
                            if (input_emb.cosineSimilarity(feature_extractor->extractTextFeature("Maths")) > 0.6) {
                                for (const auto &ss: all_subsets_maths) {
                                    rag_loader.autoFetchAndRegisterLogic(
                                        logic_injector.get(), &logic_tree, find_logic, 10,
                                        "WNJXYK/MATH-Reasoning-Paths",
                                        ss, "maths");
                                }
                            }
                            if (input_emb.cosineSimilarity(feature_extractor->extractTextFeature("education")) > 0.6) {
                                rag_loader.autoFetchAndRegisterLogic(logic_injector.get(), &logic_tree, find_logic, 10,
                                                                     "karpathy/fineweb-edu-100b-shuffle", "",
                                                                     "education");
                            }
                            if (input_emb.cosineSimilarity(feature_extractor->extractTextFeature("coding")) > 0.6) {
                                rag_loader.autoFetchAndRegisterLogic(logic_injector.get(), &logic_tree, find_logic, 10,
                                                                     "nick007x/github-code-2025", "above-2-stars",
                                                                     "coding");
                            }
                            if (input_emb.cosineSimilarity(feature_extractor->extractTextFeature("cybersecurity")) >
                                0.6) {
                                rag_loader.autoFetchAndRegisterLogic(logic_injector.get(), &logic_tree, find_logic, 10,
                                                                     "ethanolivertroy/nist-cybersecurity-training", "",
                                                                     "cybersecurity");
                            }
                        } catch (...) {
                            std::cerr << "WARN: RAG AutoLoading Failed" << std::endl;
                        }
                    }).detach();
                }
                std::thread([this, matched_logics]() {
                    for (const auto &logic_id: matched_logics) {
                        Logic curr_logic{};
                        logic_tree.fetchByHash(logic_id.first, curr_logic);
                        char *curr_logic_str = wideCharToMultiByte(curr_logic.content);
                        processTextString(&logic_processor, std::string(curr_logic_str));
                        delete[] curr_logic_str;
                    }
                }).detach();
                find_logic.clear();
            }
            if (ImageProcessor::isValidFrame(&curr_find_mem_mat, &prev_find_mem_mat)) {
                std::string mem_str = matrix_to_string(&curr_find_mem_mat);
                find_mem.append(mem_str);
            } else if (!find_mem.empty()) {
                auto input_emb = feature_extractor->extractTextFeature(find_mem);
                auto matched_memories = memory_injector->findMatchingLogicIds(find_mem);
                for (const auto &memory_id: matched_memories) {
                    MemorySlot curr_memory;
                    memory_tree.fetchByHash(memory_id.first, curr_memory);
                    const char *curr_memory_str = reinterpret_cast<const char *>(curr_memory.content.c_str);
                    processTextString(&memory_processor, std::string(curr_memory_str));
                    delete[] curr_memory_str;
                }
                find_mem.clear();
            }
            if (ImageProcessor::isValidFrame(&curr_mem_mat, &prev_mem_mat)) {
                std::string mem_str = matrix_to_string(&curr_mem_mat);
                mem.append(mem_str);
            } else if (!mem.empty()) {
                MemorySlot new_memory;
                new_memory.content = mem;
                memory_tree.storeWithFeature(new_memory, feature_extractor->extractTextFeature(mem));
                LogicDescriptor mem_desc;
                mem_desc.feature = feature_extractor->extractTextFeature(mem);
                mem_desc.logic_id = sha256_hash<MemorySlot>(new_memory);
                mem_desc.activation_threshold = 0.5;
                mem_desc.category = "memory";
                mem_desc.description = "AutoInjectedMemory";
                memory_injector->registerLogicWithStorage(mem_desc);
                mem.clear();
            }
            if (ImageProcessor::isValidFrame(&curr_find_sct_mat, &prev_find_sct_mat)) {
                std::string sct_str = matrix_to_string(&curr_find_sct_mat);
                find_sct.append(sct_str);
            } else if (!find_sct.empty()) {
                double query_emb[128];
                std::vector<float> emb_flt = feature_extractor->extractTextFeature(find_sct).data;
                for (size_t i = 0; i < 128; i++) {
                    query_emb[i] = static_cast<double>(emb_flt[i]);
                }
                // Recall top-10 related texts
                auto indices = sct.recallTopK(query_emb, 10);
                auto contents = sct.getRecalledTexts(indices);
                for (const auto &ct: contents) {
                    processTextString(&sct_processor, ct);
                }
                find_sct.clear();
            }
            if (ImageProcessor::isValidFrame(&curr_cache_mat, &prev_cache_mat)) {
                std::string cache_str = matrix_to_string(&curr_cache_mat);
                cache.append(cache_str);
            } else if (!cache.empty()) {
                cache_queue.push(cache);
                cache.clear();
                if (cache_queue.size() > 128) {
                    cache_queue.pop();
                }
            }
            if (ImageProcessor::isValidFrame(&curr_find_cache_mat, &prev_find_cache_mat)) {
                std::string cache_query_str = matrix_to_string(&curr_find_cache_mat);
                find_cache.append(cache_query_str);
            } else if (!find_cache.empty()) {
                std::queue<std::string> temp_queue = cache_queue;
                auto query_feature = feature_extractor->extractTextFeature(find_cache);
                while (!temp_queue.empty()) {
                    std::string cached_str = temp_queue.front();
                    temp_queue.pop();
                    double sim = feature_extractor->calculateSimilarity(
                        query_feature,
                        feature_extractor->extractTextFeature(cached_str), "cosine");
                    if (sim > 0.5) {
                        processTextString(&cache_processor, cached_str);
                    }
                }
                find_cache.clear();
            }
            ull neuron_count = GRID_SIZE ^ 3;
            int threads_per_block = 256;
            int blocks = (neuron_count + threads_per_block - 1) / threads_per_block;

            // 启动kernel
            update_activity<<<blocks, threads_per_block>>>(d_neurons, d_active_flags, d_trace, score);
            std::this_thread::sleep_for(std::chrono::nanoseconds(100));
            // 等待所有流完成
            for (int i = 0; i < 4; i++) {
                cudaStreamSynchronize(streams[i]);
            }
        }
    }

    KFE_STM_Slot get_ext_kfe(const std::string &hash) {
        KFE_STM_Slot cache_kfe{};
        ext_kfe.fetchByHash(hash, cache_kfe);
        return cache_kfe;
    }

    bool store_kfe(KFE_STM_Slot &slot) {
        return ext_kfe.store(slot) != 0;
    }

    [[nodiscard]] double get_avg_activity() const {
        double avg = 0.0;
        ull total_neurons = GRID_SIZE * GRID_SIZE * GRID_SIZE;
        for (ull i = 0; i < total_neurons; i++) {
            avg += d_neurons[i].get_activity();
        }
        return avg / total_neurons;
    }

    bool exp_kfe(KFE_STM_Slot kfe) {
        if (get_avg_activity() > 0.75) {
            ll retry = 20;
            while (retry > 0) {
                try {
                    if (store_kfe(kfe)) break;
                    retry--;
                } catch (...) {
                }
            }
            return true;
        }
        return false;
    }

    void resetPointersForSerialization() const {
        // Reset pointers for all neurons
        ull total_neurons = GRID_SIZE * GRID_SIZE * GRID_SIZE;
        for (ull i = 0; i < total_neurons; i++) {
            d_neurons[i].resetPointersForSerialization();
        }
    }

    void copyFrom(const NeuronModel &other) {
        path_default = other.path_default;
        ext_kfe = other.ext_kfe;
        sct = other.sct;
        logic_tree = other.logic_tree;
        memory_tree = other.memory_tree;
        cache_queue = other.cache_queue;
        memcpy(d_active_flags, other.d_active_flags, sizeof(d_active_flags));
        ull total_neurons = GRID_SIZE * GRID_SIZE * GRID_SIZE;
        for (ull i = 0; i < total_neurons; i++) {
            d_neurons[i].load(other.d_neurons[i].save());
        }
        resetPointersForSerialization();
        rebuildPointerConnections();
    }

    void rebuildPointerConnections() {
        // 重新构建所有神经元的指针连接
        for (int idx = 0; idx < GRID_SIZE * GRID_SIZE * GRID_SIZE; idx++) {
            ull x = idx / (GRID_SIZE * GRID_SIZE);
            ull y = (idx / GRID_SIZE) % GRID_SIZE;
            ull z = idx % GRID_SIZE;

            // 重设队列指针
            d_neurons[idx].setQueuePointer(&queues[x][y][z]);

            // 重设邻居队列指针
            DeviceQueue<Message, 32> *neighbour_q[6] = {nullptr};

            if (x < GRID_SIZE - 1) {
                neighbour_q[0] = &queues[x + 1][y][z];
            }
            if (x > 0) {
                neighbour_q[1] = &queues[x - 1][y][z];
            }
            if (y < GRID_SIZE - 1) {
                neighbour_q[2] = &queues[x][y + 1][z];
            }
            if (y > 0) {
                neighbour_q[3] = &queues[x][y - 1][z];
            }
            if (z < GRID_SIZE - 1) {
                neighbour_q[4] = &queues[x][y][z + 1];
            }
            if (z > 0) {
                neighbour_q[5] = &queues[x][y][z - 1];
            }

            // 重新设置邻居指针
            d_neurons[idx].setNeighbourQueuePointers(neighbour_q);
        }
    }

    /**
     * @brief Validate pointer validity.
     * 
     * Verifies that all neuron pointers are correctly set up and consistent.
     * Checks both main queue pointers and neighbor queue pointers for all neurons.
     * 
     * This function is essential for ensuring the integrity of the 3D neuron network
     * and preventing segmentation faults due to invalid pointer access.
     * 
     * @return true if all pointers are valid, false otherwise
     */
    bool validatePointers() {
        ull total_neurons = GRID_SIZE * GRID_SIZE * GRID_SIZE;

        for (ull idx = 0; idx < total_neurons; idx++) {
            ull x = idx / (GRID_SIZE * GRID_SIZE);
            ull y = (idx / GRID_SIZE) % GRID_SIZE;
            ull z = idx % GRID_SIZE;

            // 检查主队列指针
            if (d_neurons[idx].getQueue() != &queues[x][y][z]) {
                return false;
            }

            // 检查邻居队列指针
            for (int i = 0; i < 6; i++) {
                DeviceQueue<Message, 32> *expected = nullptr;

                switch (i) {
                    case 0: if (x < GRID_SIZE - 1) expected = &queues[x + 1][y][z];
                        break;
                    case 1: if (x > 0) expected = &queues[x - 1][y][z];
                        break;
                    case 2: if (y < GRID_SIZE - 1) expected = &queues[x][y + 1][z];
                        break;
                    case 3: if (y > 0) expected = &queues[x][y - 1][z];
                        break;
                    case 4: if (z < GRID_SIZE - 1) expected = &queues[x][y][z + 1];
                        break;
                    case 5: if (z > 0) expected = &queues[x][y][z - 1];
                        break;
                }

                if (d_neurons[idx].getNeighbourQueue(i) != expected) {
                    return false;
                }
            }
        }

        return true;
    }
};

#endif //SRC_NEURONMODEL_CUH
