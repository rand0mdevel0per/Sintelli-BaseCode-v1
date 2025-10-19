#!/usr/bin/env python3
"""
HuggingFace数据集流式解析器 - C++接口版本
提供与C++代码交互的API
"""

import sys
import json
import argparse
from datasets import load_dataset
import traceback

def stream_huggingface_dataset(dataset_name, subset, split, max_entries, category, output_file):
    """
    流式解析HuggingFace数据集并输出到文件
    
    Args:
        dataset_name: 数据集名称
        subset: 子集名称
        split: 数据分割
        max_entries: 最大条目数
        category: 类别
        output_file: 输出文件路径
    """
    try:
        print(f"🔍 加载数据集: {dataset_name}")
        if subset:
            dataset = load_dataset(dataset_name, subset, split=split, streaming=True)
        else:
            dataset = load_dataset(dataset_name, split=split, streaming=True)
        
        entries = []
        count = 0
        
        print(f"📥 流式获取数据...")
        for item in dataset:
            if count >= max_entries:
                break
            
            # 将数据转换为KnowledgeEntry格式
            entry = {
                "title": f"HF数据集条目 {count+1}",
                "content": str(item),
                "category": category,
                "source": f"huggingface://{dataset_name}/{subset}/{split}",
                "relevance_score": 0.8,
                "tags": []
            }
            
            entries.append(entry)
            count += 1
            
            if count % 10 == 0:
                print(f"   已处理 {count} 个条目")
        
        # 保存到JSON文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 成功处理 {count} 个条目，结果保存到 {output_file}")
        return True
        
    except Exception as e:
        print(f"❌ 处理数据集时出错: {e}")
        traceback.print_exc()
        return False

def query_huggingface_dataset(dataset_name, subset, query, max_results, category, output_file):
    """
    查询HuggingFace数据集并输出结果
    
    Args:
        dataset_name: 数据集名称
        subset: 子集名称
        query: 查询关键词
        max_results: 最大结果数
        category: 类别
        output_file: 输出文件路径
    """
    try:
        print(f"🔍 查询数据集: {dataset_name}")
        if subset:
            dataset = load_dataset(dataset_name, subset, split='train', streaming=True)
        else:
            dataset = load_dataset(dataset_name, split='train', streaming=True)
        
        results = []
        count = 0
        
        print(f"📥 查询相关条目...")
        for item in dataset:
            if count >= max_results:
                break
            
            # 简单的文本匹配
            item_text = str(item).lower()
            if query.lower() in item_text:
                entry = {
                    "title": f"查询结果 {count+1}: {query}",
                    "content": str(item),
                    "category": category,
                    "source": f"huggingface-query://{dataset_name}/{subset}",
                    "relevance_score": 0.7 + (0.2 * count / max_results),
                    "tags": []
                }
                
                results.append(entry)
                count += 1
                
                if count % 5 == 0:
                    print(f"   已找到 {count} 个匹配结果")
        
        # 保存到JSON文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 成功找到 {count} 个匹配结果，结果保存到 {output_file}")
        return True
        
    except Exception as e:
        print(f"❌ 查询数据集时出错: {e}")
        traceback.print_exc()
        return False

def main():
    """主函数 - 命令行接口"""
    parser = argparse.ArgumentParser(description='HuggingFace数据集流式解析器')
    parser.add_argument('action', choices=['stream', 'query'], help='操作类型')
    parser.add_argument('--dataset', required=True, help='数据集名称')
    parser.add_argument('--subset', default='', help='数据集子集')
    parser.add_argument('--split', default='train', help='数据分割')
    parser.add_argument('--max-entries', type=int, default=100, help='最大条目数')
    parser.add_argument('--query', default='', help='查询关键词')
    parser.add_argument('--category', default='huggingface', help='类别')
    parser.add_argument('--output', required=True, help='输出文件路径')
    
    args = parser.parse_args()
    
    print("🌊 HuggingFace数据集流式解析器")
    print("=" * 40)
    
    if args.action == 'stream':
        success = stream_huggingface_dataset(
            args.dataset, args.subset, args.split, 
            args.max_entries, args.category, args.output
        )
    elif args.action == 'query':
        if not args.query:
            print("❌ 查询操作需要提供查询关键词")
            return False
        success = query_huggingface_dataset(
            args.dataset, args.subset, args.query,
            args.max_entries, args.category, args.output
        )
    else:
        print("❌ 未知操作类型")
        return False
    
    if success:
        print("\n🎉 操作成功完成！")
        return True
    else:
        print("\n❌ 操作失败！")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)