#pragma once

// Forward declarations for ISW types used in headers to avoid heavy includes
// Use this in headers that only need pointers/references to these types.

template<typename T>
class ExternalStorage;

template<typename T>
struct FeatureVector; // incomplete type; full definition lives in isw.hpp

// Forward-declare other commonly used types if needed
class SemanticQueryEngine; // typically declared elsewhere


