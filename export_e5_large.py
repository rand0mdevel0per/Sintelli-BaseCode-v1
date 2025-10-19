from sentence_transformers import SentenceTransformer
import torch
import json
import os

# 下载 e5-large 模型
model = SentenceTransformer('intfloat/multilingual-e5-large')

os.makedirs('models', exist_ok=True)

# 导出ONNX
dummy_input = {
    'input_ids': torch.randint(0, 250000, (1, 512)),
    'attention_mask': torch.ones(1, 512, dtype=torch.long)
}

torch.onnx.export(
    model[0].auto_model,
    (dummy_input['input_ids'], dummy_input['attention_mask']),
    'models/e5/e5_large.onnx',
    input_names=['input_ids', 'attention_mask'],
    output_names=['last_hidden_state'],
    dynamic_axes={
        'input_ids': {0: 'batch', 1: 'seq'},
        'attention_mask': {0: 'batch', 1: 'seq'},
        'last_hidden_state': {0: 'batch', 1: 'seq'}
    },
    opset_version=14
)

print("✅ ONNX模型导出: models/e5/e5_large.onnx")

# 导出完整的tokenizer配置
tokenizer = model.tokenizer

# 1. 词汇表（BPE merges）
vocab = tokenizer.get_vocab()
with open('models/vocab.json', 'w', encoding='utf-8') as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)

# 2. BPE merges（关键！）
if hasattr(tokenizer, 'get_merges'):
    merges = tokenizer.get_merges()
    with open('models/merges.txt', 'w', encoding='utf-8') as f:
        for merge in merges:
            f.write(f"{merge}\n")

# 3. 特殊token配置
special_tokens = {
    'pad_token': tokenizer.pad_token,
    'unk_token': tokenizer.unk_token,
    'cls_token': tokenizer.cls_token,
    'sep_token': tokenizer.sep_token,
    'mask_token': tokenizer.mask_token,
    'pad_token_id': tokenizer.pad_token_id,
    'unk_token_id': tokenizer.unk_token_id,
    'cls_token_id': tokenizer.cls_token_id,
    'sep_token_id': tokenizer.sep_token_id,
    'mask_token_id': tokenizer.mask_token_id,
}

with open('models/special_tokens.json', 'w', encoding='utf-8') as f:
    json.dump(special_tokens, f, indent=2)

print("✅ Tokenizer配置导出:")
print(f"   - vocab.json ({len(vocab)} tokens)")
print(f"   - merges.txt")
print(f"   - special_tokens.json")

# 测试
test_texts = [
    "你好世界",
    "Hello World",
    "مرحبا بالعالم"
]

print("\n✅ 测试编码:")
for text in test_texts:
    tokens = tokenizer.encode(text)
    print(f"   '{text}' -> {len(tokens)} tokens")