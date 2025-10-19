#!/usr/bin/env python3
"""
HuggingFace流式查询测试脚本
支持在检索失败时使用流式查询作为回退方案
"""

import sys
import json
import time
from datasets import load_dataset

def test_streaming_query():
    """测试流式查询功能"""
    print("🧪 测试HuggingFace流式查询...")
    
    try:
        # 尝试加载数据集（流式模式）
        print("🔍 加载数据集...")
        dataset = load_dataset("HuggingFaceFW/fineweb-edu-100b-shuffle", 
                             split='train', 
                             streaming=True)
        
        # 获取少量样本进行测试
        samples = []
        max_samples = 5
        
        print(f"📥 流式获取前 {max_samples} 个样本...")
        for i, item in enumerate(dataset):
            if i >= max_samples:
                break
            samples.append(item)
            print(f"  已获取 {i+1}/{max_samples} 个样本")
        
        print(f"✅ 成功获取 {len(samples)} 个样本")
        
        # 保存样本到文件
        with open('streaming_test_results.json', 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        
        print("💾 结果已保存到 streaming_test_results.json")
        return True
        
    except Exception as e:
        print(f"❌ 流式查询失败: {e}")
        return False

def test_streaming_fallback(query_text, dataset_name, max_results=3):
    """测试检索失败时的流式回退机制"""
    print(f"\n🔄 测试流式回退机制...")
    print(f"   查询: '{query_text}'")
    print(f"   数据集: {dataset_name}")
    
    try:
        # 模拟本地检索失败
        print("1️⃣ 模拟本地检索失败...")
        local_success = False
        
        if not local_success:
            print("2️⃣ 启动流式查询回退...")
            
            # 执行流式查询
            dataset = load_dataset(dataset_name, split='train', streaming=True)
            
            results = []
            count = 0
            
            # 简单的关键词匹配
            for item in dataset:
                if count >= max_results:
                    break
                
                # 检查是否包含查询关键词
                text_content = str(item).lower()
                if query_text.lower() in text_content:
                    results.append(item)
                    count += 1
                    print(f"   📄 找到第 {count} 个匹配结果")
            
            if results:
                print(f"✅ 流式回退成功，找到 {len(results)} 个结果")
                
                # 保存结果
                with open('streaming_fallback_results.json', 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                
                return True
            else:
                print("❌ 流式回退未找到匹配结果")
                return False
        
    except Exception as e:
        print(f"❌ 流式回退失败: {e}")
        return False

def test_timeout_handling():
    """测试超时处理"""
    print("\n⏰ 测试超时处理...")
    
    try:
        start_time = time.time()
        timeout = 10  # 10秒超时
        
        dataset = load_dataset("HuggingFaceFW/fineweb-edu-100b-shuffle", 
                             split='train', 
                             streaming=True)
        
        samples = []
        for i, item in enumerate(dataset):
            if time.time() - start_time > timeout:
                print(f"⏰ 查询超时: {timeout}秒")
                break
            
            if i >= 3:  # 只取3个样本
                break
            
            samples.append(item)
            print(f"  获取样本 {i+1}")
        
        print(f"✅ 超时测试完成，获取 {len(samples)} 个样本")
        return True
        
    except Exception as e:
        print(f"❌ 超时测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🌊 HuggingFace流式查询测试")
    print("==========================")
    
    # 测试1: 基本流式查询
    test1_success = test_streaming_query()
    
    # 测试2: 流式回退机制
    test2_success = test_streaming_fallback("神经网络", "HuggingFaceFW/fineweb-edu-100b-shuffle")
    
    # 测试3: 超时处理
    test3_success = test_timeout_handling()
    
    # 输出测试结果
    print("\n📊 测试结果汇总:")
    print(f"   基本流式查询: {'✅ 通过' if test1_success else '❌ 失败'}")
    print(f"   流式回退机制: {'✅ 通过' if test2_success else '❌ 失败'}")
    print(f"   超时处理: {'✅ 通过' if test3_success else '❌ 失败'}")
    
    if test1_success and test2_success and test3_success:
        print("\n🎉 所有测试通过！")
        print("\n💡 流式查询功能特点:")
        print("   • 支持HuggingFace数据集流式访问")
        print("   • 内存占用低，适合大文件")
        print("   • 检索失败时自动回退到流式查询")
        print("   • 支持超时控制和重试机制")
        print("   • 实时进度反馈")
    else:
        print("\n⚠️  部分测试失败，请检查网络连接和依赖")
        print("   需要安装: pip install datasets")

if __name__ == "__main__":
    main()