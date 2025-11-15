#include "rag_knowledge_loader.cuh"
#include "isw.cuh"
#include "semantic_query_engine.cuh"
#include <iostream>
#include <memory>

using namespace std;

int main() {
    cout << "🚀 HuggingFace RAG Integration Demo" << endl;
    cout << "========================" << endl;

    // 1. Create RAG knowledge base loader
    RAGKnowledgeBaseLoader loader;

    // 2. Create external storage
    auto external_storage = make_shared<ExternalStorage<KnowledgeEntry>>();
    loader.setExternalStorage(external_storage);

    // 3. Set maximum storage size
    loader.setMaxStorageSize(5000); // Limit to 5000 entries

    // 4. Demonstrate HuggingFace dataset streaming parsing
    cout << "\n1️⃣ Demonstrate HuggingFace Dataset Streaming Parsing:" << endl;
    bool stream_success = loader.streamHuggingFaceDataset(
        "HuggingFaceFW/fineweb",
        "sample-10BT",
        "train",
        10,  // 获取10个条目
        "web_content"
    );

    if (stream_success) {
        cout << "✅ Streaming parsing successful" << endl;
    } else {
        cout << "❌ Streaming parsing failed" << endl;
    }

    // 5. Demonstrate HuggingFace dataset query
    cout << "Demonstrate HuggingFace Dataset Query:" << endl;
    bool query_success = loader.queryAndLoadFromHFDataset(
        "artifical intelligence",
        "HuggingFaceFW/fineweb",
        "sample-10BT",
        5,  // 获取5个相关条目
        "ai_research"
    );

    if (query_success) {
        cout << "✅ Dataset query successful" << endl;
    } else {
        cout << "❌ Dataset query failed" << endl;
    }

    // 6. Demonstrate automatic data fetching when Logic matching is insufficient
    cout << "\n3️⃣ Demonstrate Automatic Data Fetching When Logic Matching Is Insufficient:" << endl;
    bool auto_fetch_success = loader.autoFetchDataWhenLogicInsufficient(
        "deep learning",
        20,  // 需要至少20个匹配项
        "HuggingFaceFW/fineweb",
        "sample-10BT"
    );

    if (auto_fetch_success) {
        cout << "✅ Automatic data fetching successful" << endl;
    } else {
        cout << "❌ Automatic data fetching failed" << endl;
    }

    // 7. Get knowledge base statistics
    auto stats = loader.getKnowledgeStats();
    cout << "\n4️⃣ Knowledge Base Statistics:" << endl;
    cout << "   Total entries: " << stats.total_entries << endl;
    cout << "   Number of categories: " << stats.unique_categories << endl;
    cout << "   Average relevance score: " << stats.avg_relevance_score << endl;

    // Display entry count for each category
    cout << "   Entry count for each category:" << endl;
    for (const auto& pair : stats.category_counts) {
        cout << "     " << pair.first << ": " << pair.second << endl;
    }

    // 8. Demonstrate external storage functionality
    cout << "Demonstrate External Storage Functionality:" << endl;
    auto entries = loader.getAllEntries();
    if (!entries.empty()) {
        cout << "   Inserting first 10 entries into external storage..." << endl;
        int count = 0;
        vector<KnowledgeEntry> first_entries;
        for (const auto& entry : entries) {
            if (count++ >= 10) break;
            first_entries.push_back(entry);
        }

        if (loader.insertToExternalStorage(first_entries)) {
            cout << "✅ Successfully inserted " << first_entries.size() << " entries into external storage" << endl;
        } else {
            cout << "❌ Failed to insert into external storage" << endl;
        }
    }

    // 9. Check and clean up storage
    cout << "Check and Clean Up Storage:" << endl;
    bool cleanup_triggered = loader.checkAndCleanupStorage();
    if (cleanup_triggered) {
        cout << "✅ Storage cleanup triggered" << endl;
    } else {
        cout << "ℹ️  Storage size is within limits, no cleanup needed" << endl;
    }

    // 10. Get external storage statistics
    if (external_storage) {
        auto storage_stats = external_storage->getStatistics();
        cout << " External Storage Statistics:" << endl;
        cout << "   L2 memory pool size: " << storage_stats.l2_size << endl;
        cout << "   L3 disk storage size: " << storage_stats.l3_size << endl;
        cout << "   Total size: " << storage_stats.total_size << endl;
        cout << "   Average heat: " << storage_stats.avg_heat << endl;
        cout << "   Maximum heat: " << storage_stats.max_heat << endl;
        cout << "   Minimum heat: " << storage_stats.min_heat << endl;
    }

    // 11. Demonstrate semantic search functionality
    cout << "Demonstrate Semantic Search Functionality:" << endl;
    if (!entries.empty()) {
        // 创建语义查询引擎
        SemanticQueryEngine semantic_engine("/models/e5/e5_large.onnx");

        if (semantic_engine.isInitialized()) {
            cout << "✅ Semantic query engine initialized successfully" << endl;

            // Perform semantic search on the first few entries
            vector<string> candidates;
            int candidate_count = 0;
            for (const auto& entry : entries) {
                if (candidate_count++ >= 5) break;
                candidates.push_back(entry.content);
            }

            // 执行语义搜索
            auto search_results = semantic_engine.semanticSearch(
                "machine learning algorithm",
                candidates,
                3,  // 返回前3个最相似的
                0.1 // 相似度阈值
            );

            cout << "   Semantic search results:" << endl;
            for (const auto& result : search_results) {
                cout << "     similarity: " << result.second
                     << ", category index: " << result.first << endl;
            }
        } else {
            cout << "⚠️  Semantic query engine initialization failed, using mock implementation" << endl;
        }
    }

    cout << "\n🎉 HuggingFace RAG Integration Demo Completed!" << endl;
    cout << "\n💡 Key Features:" << endl;
    cout << "   • 🌊 HuggingFace Dataset Streaming Parsing" << endl;
    cout << "   • 🔍 HuggingFace Dataset Semantic Query" << endl;
    cout << "   • 🤖 Automatic Data Fetching When Logic Matching Is Insufficient" << endl;
    cout << "   • 💾 External Storage Integration (L2 Memory + L3 Disk)" << endl;
    cout << "   • 🧹 Intelligent Storage Management (Automatic Cleanup)" << endl;
    cout << "   • 🧠 Semantic Query Support (E5 Model)" << endl;
    cout << "   • 📊 Real-time Statistics and Monitoring" << endl;

    return 0;
}