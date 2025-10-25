#ifndef SRC_VARIANCE_AWARE_FILTER_H
#define SRC_VARIANCE_AWARE_FILTER_H

#include "text_quality_filter.h"
#include <vector>
#include <deque>
#include <cmath>
#include <algorithm>
#include <functional>

// 方差级别
enum class VarianceLevel {
    LOW_VARIANCE,      // 低方差 - 严格过滤
    MEDIUM_VARIANCE,   // 中等方差 - 中等过滤
    HIGH_VARIANCE,     // 高方差 - 宽松过滤
    VERY_HIGH_VARIANCE // 极高方差 - 几乎不过滤
};

// 过滤级别
enum class FilterLevel {
    STRICT,     // 严格：只接受高质量自然语言
    MODERATE,   // 中等：接受高质量和部分低质量
    RELAXED,    // 宽松：接受大部分文本
    VERY_RELAXED // 极宽松：几乎接受所有文本
};

class VarianceAwareTextFilter {
private:
    TextQualityFilter quality_filter;
    
    // 系统方差跟踪
    std::deque<double> recent_variances;
    size_t variance_window_size;
    
    // 当前方差级别
    VarianceLevel current_variance_level;
    
    // 过滤阈值配置
    struct FilterThresholds {
        double min_natural_score;      // 最小自然语言分数
        double max_symbol_density;     // 最大符号密度
        double max_repetition_ratio;   // 最大重复率
        bool accept_math;              // 是否接受数学表达式
        bool accept_code;              // 是否接受代码片段
        bool accept_neural_terms;      // 是否接受神经网络术语
        bool accept_custom_symbols;    // 是否接受自定义符号
    };
    
    std::unordered_map<VarianceLevel, FilterThresholds> variance_thresholds;
    std::unordered_map<FilterLevel, FilterThresholds> filter_thresholds;
    
public:
    VarianceAwareTextFilter(size_t window_size = 100) 
        : variance_window_size(window_size), current_variance_level(VarianceLevel::MEDIUM_VARIANCE) {
        
        // 初始化方差级别阈值
        initializeVarianceThresholds();
        
        // 初始化过滤级别阈值
        initializeFilterThresholds();
    }
    
    // 更新系统方差并调整过滤级别
    void updateSystemVariance(double current_variance) {
        // 添加新的方差值
        recent_variances.push_back(current_variance);
        
        // 保持窗口大小
        if (recent_variances.size() > variance_window_size) {
            recent_variances.pop_front();
        }
        
        // 计算平均方差
        double avg_variance = calculateAverageVariance();
        
        // 根据平均方差调整过滤级别
        updateFilterLevel(avg_variance);
    }
    
    // 检查是否接受文本
    bool shouldAcceptText(const std::string& text, double confidence_score = 1.0) {
        // 获取当前过滤阈值
        FilterThresholds thresholds = getCurrentThresholds();
        
        // 质量评估
        TextQuality quality = quality_filter.evaluateQuality(text);
        SpecialLanguageType lang_type = quality_filter.detectSpecialLanguageType(text);
        
        // 计算详细指标
        double natural_score = quality_filter.calculateNaturalLanguageScore(text);
        double symbol_density = quality_filter.calculateSymbolDensity(text);
        double repetition_ratio = quality_filter.calculateRepetitionRatio(text);
        
        // 应用置信度调整
        natural_score *= confidence_score;
        
        // 检查基本质量要求
        if (natural_score < thresholds.min_natural_score) {
            return false;
        }
        
        if (symbol_density > thresholds.max_symbol_density) {
            return false;
        }
        
        if (repetition_ratio > thresholds.max_repetition_ratio) {
            return false;
        }
        
        // 检查专有语言类型限制
        if (!shouldAcceptLanguageType(lang_type, thresholds)) {
            return false;
        }
        
        return true;
    }
    
    // 获取当前过滤级别
    FilterLevel getCurrentFilterLevel() const {
        switch (current_variance_level) {
            case VarianceLevel::LOW_VARIANCE:
                return FilterLevel::STRICT;
            case VarianceLevel::MEDIUM_VARIANCE:
                return FilterLevel::MODERATE;
            case VarianceLevel::HIGH_VARIANCE:
                return FilterLevel::RELAXED;
            case VarianceLevel::VERY_HIGH_VARIANCE:
                return FilterLevel::VERY_RELAXED;
            default:
                return FilterLevel::MODERATE;
        }
    }
    
    // 获取当前方差级别
    VarianceLevel getCurrentVarianceLevel() const {
        return current_variance_level;
    }
    
    // 手动设置过滤级别（用于测试或特殊情况）
    void setFilterLevel(FilterLevel level) {
        switch (level) {
            case FilterLevel::STRICT:
                current_variance_level = VarianceLevel::LOW_VARIANCE;
                break;
            case FilterLevel::MODERATE:
                current_variance_level = VarianceLevel::MEDIUM_VARIANCE;
                break;
            case FilterLevel::RELAXED:
                current_variance_level = VarianceLevel::HIGH_VARIANCE;
                break;
            case FilterLevel::VERY_RELAXED:
                current_variance_level = VarianceLevel::VERY_HIGH_VARIANCE;
                break;
        }
    }
    
    // 获取文本的详细评估报告
    struct TextEvaluation {
        bool accepted;
        TextQuality quality;
        SpecialLanguageType language_type;
        double natural_score;
        double symbol_density;
        double repetition_ratio;
        std::string rejection_reason;
    };
    
    TextEvaluation evaluateTextWithDetails(const std::string& text, double confidence_score = 1.0) {
        TextEvaluation result;
        
        result.quality = quality_filter.evaluateQuality(text);
        result.language_type = quality_filter.detectSpecialLanguageType(text);
        result.natural_score = quality_filter.calculateNaturalLanguageScore(text);
        result.symbol_density = quality_filter.calculateSymbolDensity(text);
        result.repetition_ratio = quality_filter.calculateRepetitionRatio(text);
        
        // 应用置信度调整
        result.natural_score *= confidence_score;
        
        // 获取当前过滤阈值
        FilterThresholds thresholds = getCurrentThresholds();
        
        // 检查接受条件
        result.accepted = true;
        result.rejection_reason = "";
        
        if (result.natural_score < thresholds.min_natural_score) {
            result.accepted = false;
            result.rejection_reason = "自然语言分数过低";
        } else if (result.symbol_density > thresholds.max_symbol_density) {
            result.accepted = false;
            result.rejection_reason = "符号密度过高";
        } else if (result.repetition_ratio > thresholds.max_repetition_ratio) {
            result.accepted = false;
            result.rejection_reason = "重复率过高";
        } else if (!shouldAcceptLanguageType(result.language_type, thresholds)) {
            result.accepted = false;
            result.rejection_reason = "不支持的专有语言类型";
        }
        
        return result;
    }
    
private:
    // 初始化方差级别阈值
    void initializeVarianceThresholds() {
        // 低方差 - 严格过滤
        variance_thresholds[VarianceLevel::LOW_VARIANCE] = {
            .min_natural_score = 0.7,
            .max_symbol_density = 0.15,
            .max_repetition_ratio = 0.2,
            .accept_math = false,
            .accept_code = false,
            .accept_neural_terms = true,
            .accept_custom_symbols = false
        };
        
        // 中等方差 - 中等过滤
        variance_thresholds[VarianceLevel::MEDIUM_VARIANCE] = {
            .min_natural_score = 0.5,
            .max_symbol_density = 0.25,
            .max_repetition_ratio = 0.4,
            .accept_math = true,
            .accept_code = false,
            .accept_neural_terms = true,
            .accept_custom_symbols = true
        };
        
        // 高方差 - 宽松过滤
        variance_thresholds[VarianceLevel::HIGH_VARIANCE] = {
            .min_natural_score = 0.3,
            .max_symbol_density = 0.35,
            .max_repetition_ratio = 0.6,
            .accept_math = true,
            .accept_code = true,
            .accept_neural_terms = true,
            .accept_custom_symbols = true
        };
        
        // 极高方差 - 几乎不过滤
        variance_thresholds[VarianceLevel::VERY_HIGH_VARIANCE] = {
            .min_natural_score = 0.1,
            .max_symbol_density = 0.5,
            .max_repetition_ratio = 0.8,
            .accept_math = true,
            .accept_code = true,
            .accept_neural_terms = true,
            .accept_custom_symbols = true
        };
    }
    
    // 初始化过滤级别阈值
    void initializeFilterThresholds() {
        filter_thresholds[FilterLevel::STRICT] = variance_thresholds[VarianceLevel::LOW_VARIANCE];
        filter_thresholds[FilterLevel::MODERATE] = variance_thresholds[VarianceLevel::MEDIUM_VARIANCE];
        filter_thresholds[FilterLevel::RELAXED] = variance_thresholds[VarianceLevel::HIGH_VARIANCE];
        filter_thresholds[FilterLevel::VERY_RELAXED] = variance_thresholds[VarianceLevel::VERY_HIGH_VARIANCE];
    }
    
    // 计算平均方差
    double calculateAverageVariance() {
        if (recent_variances.empty()) return 0.0;
        
        double sum = 0.0;
        for (double var : recent_variances) {
            sum += var;
        }
        return sum / recent_variances.size();
    }
    
    // 根据平均方差更新过滤级别
    void updateFilterLevel(double avg_variance) {
        if (avg_variance < 0.1) {
            current_variance_level = VarianceLevel::LOW_VARIANCE;
        } else if (avg_variance < 0.3) {
            current_variance_level = VarianceLevel::MEDIUM_VARIANCE;
        } else if (avg_variance < 0.6) {
            current_variance_level = VarianceLevel::HIGH_VARIANCE;
        } else {
            current_variance_level = VarianceLevel::VERY_HIGH_VARIANCE;
        }
    }
    
    // 获取当前阈值
    FilterThresholds getCurrentThresholds() const {
        return variance_thresholds.at(current_variance_level);
    }
    
    // 检查是否接受特定语言类型
    bool shouldAcceptLanguageType(SpecialLanguageType lang_type, const FilterThresholds& thresholds) const {
        switch (lang_type) {
            case SpecialLanguageType::MATH_EXPRESSION:
                return thresholds.accept_math;
            case SpecialLanguageType::CODE_SNIPPET:
                return thresholds.accept_code;
            case SpecialLanguageType::NEURAL_NETWORK_TERM:
                return thresholds.accept_neural_terms;
            case SpecialLanguageType::CUSTOM_SYMBOL:
                return thresholds.accept_custom_symbols;
            case SpecialLanguageType::NATURAL_LANGUAGE:
                return true; // 自然语言总是接受
            default:
                return false;
        }
    }
};

#endif // SRC_VARIANCE_AWARE_FILTER_H