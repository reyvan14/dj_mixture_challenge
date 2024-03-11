import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json
import os
from tqdm import tqdm

# 检测是否有可用的 GPU，如果没有，则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型和分词器
model = BertForSequenceClassification.from_pretrained('reyvan/bert_large_maths')
tokenizer = BertTokenizer.from_pretrained('reyvan/bert_large_maths')
model.to(device)

def predict_math_question(question, tokenizer, model, max_len=512):
    encoding = tokenizer.encode_plus(
        question,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        prediction = torch.max(outputs.logits, dim=1)[1].item()

    return prediction

# 读取 JSONL 文件
def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file.readlines()]

# 处理数据并保存结果到同一个文件
def process_and_save(input_dir, output_file):
    with open(output_file, 'w', encoding='utf-8') as out_file:
        for filename in os.listdir(input_dir):
            if filename.endswith('.jsonl'):
                input_file = os.path.join(input_dir, filename)
                data = read_jsonl(input_file)
                for item in tqdm(data, desc=f"Processing {filename}"):
                    combined_text = item['instruction'] + " " + item['input']
                    if predict_math_question(combined_text, tokenizer, model) == 1:
                        json.dump(item, out_file)
                        out_file.write('\n')

# 调用函数处理数据
input_dir = 'input'  # 指定输入目录
output_file = 'sorted_math_questions_bert_large.jsonl'  # 指定输出文件
process_and_save(input_dir, output_file)
