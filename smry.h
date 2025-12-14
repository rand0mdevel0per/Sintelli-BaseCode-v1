#pragma once

// Lightweight header providing forward declarations for the smry (E5) model API
// so that headers can depend on types without including the full implementation

#ifdef __cplusplus
extern "C" {
#endif

// Forward-declare C++ classes used across headers
class E5LargeModel;
class UnifiedInputProcessor;

// C-compatible factory functions (implemented in smry.cpp/.cu)
UnifiedInputProcessor* createTextProcessor();
void destroyTextProcessor(UnifiedInputProcessor* processor);

// Initialize / shutdown
bool initUnifiedSystem(const char* model_path, const char* vocab_path, const char* merges_path, const char* special_tokens_path);

#ifdef __cplusplus
}
#endif

