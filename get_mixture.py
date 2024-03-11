import os
import json
import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# 设置环境变量以确保transformers库在离线环境中使用
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 定义随机种子
seed = 42
random_seed = 66

# 定义基础路径
base_dir = Path(__file__).resolve().parent.parent
input_dir = base_dir / "input"
ratio_path = base_dir / "output" / "sft_data" / "ratio.json"
first_mixture_path = base_dir / "output" / "sft_data" / "first_mixture.jsonl"
final_mixture_path = base_dir / "output" / "sft_data" / "mixture.jsonl"

def sequential_sampling(file_path, num_samples, seed=None):
    if seed is not None:
        random.seed(seed)
    sample = []
    with open(file_path, "r") as f:
        lines = f.readlines()
    if num_samples < len(lines):
        sample = random.sample(lines, num_samples)
    else:
        sample = lines
    return sample

def process_file(file_path, num_samples):
    sampled_data = sequential_sampling(file_path, num_samples)
    return sampled_data, len(sampled_data)

def generate_mixture(input_dir, ratio_path, first_mixture_path, sample_counts):
    file_list = list(input_dir.glob("*.jsonl"))
    file_to_sample_count = {file.name: sample_counts.get(file.name, 0) for file in file_list}

    with ProcessPoolExecutor() as executor:
        future_to_file = {executor.submit(process_file, file, file_to_sample_count[file.name]): file for file in file_list}

    with open(first_mixture_path, "w") as mixture_file:
        for future in future_to_file:
            file_path = future_to_file[future]
            result, num_samples = future.result()
            print(f"{file_path.name}: 抽取了 {num_samples} 个样本")
            mixture_file.writelines(result)

def shuffle_jsonl(input_jsonl_path, final_mixture_path):
    random.seed(random_seed)
    with open(input_jsonl_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    data = [json.loads(line) for line in lines]
    random.shuffle(data)
    with open(final_mixture_path, 'w', encoding='utf-8') as file:
        for item in data:
            file.write(json.dumps(item) + '\n')

# 指定每个文件的样本数量
sample_counts = {'sorted_TruthfulQA_0229.jsonl': 40000, 'sorted_math_questions_bert_large.jsonl': 23000}

# 生成混合文件
generate_mixture(input_dir, ratio_path, first_mixture_path, sample_counts)

# 打乱生成的混合文件
shuffle_jsonl(first_mixture_path, final_mixture_path)
