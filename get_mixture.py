import os
import json
import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# 设置环境变量以确保transformers库在离线环境中使用
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 定义随机种子
seed = 42
random.seed(seed)

# 定义基础路径
base_dir = Path(__file__).resolve().parent.parent
input_dir = base_dir / "input"
ratio_path = base_dir / "output" / "sft_data" / "ratio.json"
mixture_path = base_dir / "output" / "sft_data" / "mixture.jsonl"

def reservoir_sampling(file_path, num_samples):
    """
    使用水库抽样算法从文件中选取样本
    """
    sample = []
    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            if i < num_samples:
                sample.append(line)
            else:
                s = random.randint(0, i)
                if s < num_samples:
                    sample[s] = line
    return sample

def process_file(file_path, prob, sample_count):
    """
    处理单个文件，返回抽样后的数据及其数量
    """
    num_samples = int(prob * sample_count)
    sampled_data = reservoir_sampling(file_path, num_samples)
    return sampled_data, len(sampled_data)

def generate_mixture(input_dir, ratio_path, mixture_path):
    sample_count = 200000

    file_list = list(input_dir.glob("*.jsonl"))
    num_files_to_select = random.randint(1, len(file_list))
    selected_files = random.sample(file_list, k=num_files_to_select)

    probabilities = [random.random() for _ in selected_files]
    total_probability = sum(probabilities)
    probabilities = [p / total_probability for p in probabilities]

    ratio = {f.name: prob for f, prob in zip(selected_files, probabilities)}
    with open(ratio_path, "w") as ratio_file:
        json.dump(ratio, ratio_file, indent=4)

    with ProcessPoolExecutor() as executor:
        future_to_file = {executor.submit(process_file, file, prob, sample_count): file for file, prob in zip(selected_files, probabilities)}
    
    with open(mixture_path, "w") as mixture_file:
        for future in future_to_file:
            file_path = future_to_file[future]
            result, num_samples = future.result()
            print(f"{file_path.name}: 抽取了 {num_samples} 个样本")
            mixture_file.writelines(result)

generate_mixture(input_dir, ratio_path, mixture_path)
