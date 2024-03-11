# # =============================================================================
# # 以下是一个采用随机采样的示例，参赛者可自由编写文件内容。
# # =============================================================================
# import os
# os.environ["TRANSFORMERS_OFFLINE"] = "1"

# # 然后是其他的导入
# import transformers
# import torch
# # 其他代码

# import json
# import random
# from pathlib import Path

# # 设置随机种子，保证结果可复现
# seed = 42
# random.seed(seed)

# # 如果使用 NumPy
# # import numpy as np
# # np.random.seed(seed)

# # 如果使用 PyTorch
# # import torch
# # torch.manual_seed(seed)

# # 开发套件根目录
# base_dir = Path(__file__).resolve().parent.parent

# # 输入输出路径
# input_dir = base_dir / "input"
# ratio_path = base_dir / "output" / "sft_data" / "ratio.json"
# mixture_path = base_dir / "output" / "sft_data" / "mixture.jsonl"


# # 函数仅为示意，可自由改写
# def generate_mixture(input_dir, ratio_path, mixture_path):
#     # 假设采样 60k 条样本
#     sample_count = 100000

#     # 获取所有 jsonl 文件的路径列表
#     file_list = list(input_dir.glob("*.jsonl"))

#     # 随机选择随机数量的文件
#     num_files_to_select = random.randint(1, len(file_list))
#     selected_files = random.sample(file_list, k=num_files_to_select)

#     # 随机分配采样概率
#     probabilities = [random.random() for _ in selected_files]
#     total_probability = sum(probabilities)
#     probabilities = [p / total_probability for p in probabilities]

#     # 保存采样概率到 ratio.json
#     ratio = {f.name: prob for f, prob in zip(selected_files, probabilities)}
#     with open(ratio_path, "w") as ratio_file:
#         json.dump(ratio, ratio_file, indent=4)

#     # 按照采样概率抽取样本并写入到 mixture.jsonl
#     with open(mixture_path, "w") as mixture_file:
#         for file_path, prob in zip(selected_files, probabilities):
#             with open(file_path, "r") as f:
#                 samples = f.readlines()
#                 num_samples = int(prob * sample_count)
#                 selected_samples = random.sample(
#                     samples, min(num_samples, len(samples))
#                 )
#                 mixture_file.writelines(selected_samples)


# # 执行函数
# generate_mixture(input_dir, ratio_path, mixture_path)


# import os
# import json
# import random
# from pathlib import Path
# from concurrent.futures import ProcessPoolExecutor

# # 设置环境变量以确保transformers库在离线环境中使用
# os.environ["TRANSFORMERS_OFFLINE"] = "1"

# # 定义随机种子
# seed = 42
# random.seed(seed)

# # 定义基础路径
# base_dir = Path(__file__).resolve().parent.parent
# input_dir = base_dir / "input"
# ratio_path = base_dir / "output" / "sft_data" / "ratio.json"
# mixture_path = base_dir / "output" / "sft_data" / "mixture.jsonl"

# def reservoir_sampling(file_path, num_samples):
#     """
#     使用水库抽样算法从文件中选取样本
#     """
#     sample = []
#     with open(file_path, "r") as f:
#         for i, line in enumerate(f):
#             if i < num_samples:
#                 sample.append(line)
#             else:
#                 s = random.randint(0, i)
#                 if s < num_samples:
#                     sample[s] = line
#     return sample

# def process_file(file_path, weight, total_weight, sample_count):
#     """
#     处理单个文件，返回抽样后的数据及其数量
#     根据文件权重调整抽样数量
#     """
#     num_samples = int(weight / total_weight * sample_count)
#     sampled_data = reservoir_sampling(file_path, num_samples)
#     return sampled_data, len(sampled_data)

# def get_file_size(file_path):
#     """
#     返回文件的行数作为大小的指标
#     """
#     with open(file_path, 'r') as f:
#         return sum(1 for line in f)

# def generate_mixture(input_dir, ratio_path, mixture_path):
#     sample_count = 350000

#     file_list = list(input_dir.glob("*.jsonl"))
#     file_weights = [get_file_size(f) for f in file_list]
#     total_weight = sum(file_weights)

#     with ProcessPoolExecutor() as executor:
#         future_to_file = {executor.submit(process_file, file, weight, total_weight, sample_count): file for file, weight in zip(file_list, file_weights)}
    
#     with open(mixture_path, "w") as mixture_file:
#         for future in future_to_file:
#             file_path = future_to_file[future]
#             result, num_samples = future.result()
#             print(f"{file_path.name}: 抽取了 {num_samples} 个样本")
#             mixture_file.writelines(result)

# generate_mixture(input_dir, ratio_path, mixture_path)

# 2024-02-18修改

# import os
# import json
# import random
# from pathlib import Path
# from concurrent.futures import ProcessPoolExecutor

# # 设置环境变量以确保transformers库在离线环境中使用
# os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 定义随机种子
# seed = 42
# random.seed(seed)

# # 定义基础路径
# base_dir = Path(__file__).resolve().parent.parent
# input_dir = base_dir / "input"
# ratio_path = base_dir / "output" / "sft_data" / "ratio.json"
# mixture_path = base_dir / "output" / "sft_data" / "mixture.jsonl"

# def reservoir_sampling(file_path, num_samples):
#     """
#     使用水库抽样算法从文件中选取样本
#     """
#     sample = []
#     with open(file_path, "r") as f:
#         for i, line in enumerate(f):
#             if i < num_samples:
#                 sample.append(line)
#             else:
#                 s = random.randint(0, i)
#                 if s < num_samples:
#                     sample[s] = line
#     return sample

# def process_file(file_path, num_samples):
#     """
#     处理单个文件，返回抽样后的数据及其数量
#     """
#     sampled_data = reservoir_sampling(file_path, num_samples)
#     return sampled_data, len(sampled_data)

# def generate_mixture(input_dir, ratio_path, mixture_path, sample_counts):
#     file_list = list(input_dir.glob("*.jsonl"))
#     file_to_sample_count = {file.name: sample_counts.get(file.name, 0) for file in file_list}

#     with ProcessPoolExecutor() as executor:
#         future_to_file = {executor.submit(process_file, file, file_to_sample_count[file.name]): file for file in file_list}
    
#     with open(mixture_path, "w") as mixture_file:
#         for future in future_to_file:
#             file_path = future_to_file[future]
#             result, num_samples = future.result()
#             print(f"{file_path.name}: 抽取了 {num_samples} 个样本")
#             mixture_file.writelines(result)

# # 指定每个文件的样本数量，示例: {'dataset1.jsonl': 50000, 'dataset2.jsonl': 80000}
# sample_counts = {'instruct.jsonl': 0,'dolly.jsonl': 10002,'alpaca_data.jsonl': 10002,'sorted_TruthfulQA.jsonl': 10000,'sorted_math_questions_bert_large.jsonl': 23000}

# # sample_counts = {'instruct.jsonl': 0,'ZZZgpt4all.jsonl': 30000,'alpaca_data.jsonl': 3002,'sorted_TruthfulQA_2.jsonl': 3002,'sorted_math_questions_bert_large.jsonl': 4000}

# generate_mixture(input_dir, ratio_path, mixture_path, sample_counts)







# import os
# import json
# import random
# from pathlib import Path
# from concurrent.futures import ProcessPoolExecutor, as_completed

# # 设置环境变量以确保transformers库在离线环境中使用
# os.environ["TRANSFORMERS_OFFLINE"] = "1"

# # 定义随机种子
# seed = 66
# random.seed(seed)

# # 定义基础路径
# base_dir = Path(__file__).resolve().parent.parent
# input_dir = base_dir / "input"
# ratio_path = base_dir / "output" / "sft_data" / "ratio.json"
# mixture_path = base_dir / "output" / "sft_data" / "mixture.jsonl"

# def reservoir_sampling(file_path, num_samples):
#     """
#     使用水库抽样算法从文件中选取样本
#     """
#     sample = []
#     with open(file_path, "r") as f:
#         for i, line in enumerate(f):
#             if i < num_samples:
#                 sample.append(line)
#             else:
#                 s = random.randint(0, i)
#                 if s < num_samples:
#                     sample[s] = line
#     return sample

# def process_file(file_path, num_samples):
#     """
#     处理单个文件，返回抽样后的数据及其数量
#     """
#     sampled_data = reservoir_sampling(file_path, num_samples)
#     return sampled_data, len(sampled_data)

# def generate_mixture(input_dir, ratio_path, mixture_path, sample_counts):
#     file_list = sorted(list(input_dir.glob("*.jsonl")), key=lambda x: x.name)
#     file_to_sample_count = {file.name: sample_counts.get(file.name, 0) for file in file_list}

#     results = {}
#     with ProcessPoolExecutor() as executor:
#         future_to_file = {executor.submit(process_file, file, file_to_sample_count[file.name]): file for file in file_list}
#         for future in as_completed(future_to_file):
#             file = future_to_file[future]
#             results[file.name] = future.result()

#     with open(mixture_path, "w") as mixture_file:
#         for file_name in sorted(results.keys()):
#             result, num_samples = results[file_name]
#             print(f"{file_name}: 抽取了 {num_samples} 个样本")
#             mixture_file.writelines(result)

# # 指定每个文件的样本数量，示例: {'dataset1.jsonl': 50000, 'dataset2.jsonl': 80000}
# sample_counts = {'instruct.jsonl': 12002, 'dolly.jsonl': 20002, 'alpaca_data.jsonl': 10002, 'sorted_math_questions_bert_large.jsonl': 22000}

# generate_mixture(input_dir, ratio_path, mixture_path, sample_counts)


# 2024-03-03最优
import os
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# 设置环境变量以确保transformers库在离线环境中使用
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 定义随机种子
seed = 42

# 定义基础路径
base_dir = Path(__file__).resolve().parent.parent
input_dir = base_dir / "input"
ratio_path = base_dir / "output" / "sft_data" / "ratio.json"
mixture_path = base_dir / "output" / "sft_data" / "mixture.jsonl"

def sequential_sampling(file_path, num_samples):
    """
    按顺序从文件中选取指定数量的样本
    """
    sample = []
    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            if i < num_samples:
                sample.append(line)
            else:
                break
    return sample

def process_file(file_path, num_samples):
    """
    处理单个文件，返回抽样后的数据及其数量
    """
    sampled_data = sequential_sampling(file_path, num_samples)
    return sampled_data, len(sampled_data)

def generate_mixture(input_dir, ratio_path, mixture_path, sample_counts):
    file_list = list(input_dir.glob("*.jsonl"))
    file_to_sample_count = {file.name: sample_counts.get(file.name, 0) for file in file_list}

    with ProcessPoolExecutor() as executor:
        future_to_file = {executor.submit(process_file, file, file_to_sample_count[file.name]): file for file in file_list}
    
    with open(mixture_path, "w") as mixture_file:
        for future in future_to_file:
            file_path = future_to_file[future]
            result, num_samples = future.result()
            print(f"{file_path.name}: 抽取了 {num_samples} 个样本")
            mixture_file.writelines(result)

# 指定每个文件的样本数量，示例: {'dataset1.jsonl': 50000, 'dataset2.jsonl': 80000}
sample_counts = {'instruct.jsonl': 0,'dolly.jsonl': 0,'alpaca_data.jsonl': 0,'sorted_TruthfulQA_0229.jsonl': 40000,'sorted_math_questions_bert_large.jsonl': 23000}

generate_mixture(input_dir, ratio_path, mixture_path, sample_counts)

# import os
# import json
# from pathlib import Path
# from concurrent.futures import ProcessPoolExecutor

# # 设置环境变量以确保transformers库在离线环境中使用
# os.environ["TRANSFORMERS_OFFLINE"] = "1"

# # 定义随机种子
# seed = 42

# # 定义基础路径
# base_dir = Path(__file__).resolve().parent.parent
# input_dir = base_dir / "input"
# ratio_path = base_dir / "output" / "sft_data" / "ratio.json"
# mixture_path = base_dir / "output" / "sft_data" / "mixture.jsonl"

# def sequential_sampling(file_path, num_samples):
#     """
#     按顺序从文件中选取指定数量的样本
#     """
#     sample = []
#     with open(file_path, "r") as f:
#         for i, line in enumerate(f):
#             if i < num_samples:
#                 sample.append(line)
#             else:
#                 break
#     return sample

# def process_file(file_path, num_samples):
#     """
#     处理单个文件，返回抽样后的数据及其数量
#     """
#     sampled_data = sequential_sampling(file_path, num_samples)
#     return sampled_data, len(sampled_data)

# def generate_mixture(input_dir, ratio_path, mixture_path, sample_counts):
#     file_list = list(input_dir.glob("*.jsonl"))
#     file_to_sample_count = {file.name: sample_counts.get(file.name, 0) for file in file_list}

#     with ProcessPoolExecutor() as executor:
#         future_to_file = {executor.submit(process_file, file, file_to_sample_count[file.name]): file for file in file_list}
    
#     with open(mixture_path, "w") as mixture_file:
#         for future in future_to_file:
#             file_path = future_to_file[future]
#             result, num_samples = future.result()
#             print(f"{file_path.name}: 抽取了 {num_samples} 个样本")
#             mixture_file.writelines(result)

# # 指定每个文件的样本数量，示例: {'dataset1.jsonl': 50000, 'dataset2.jsonl': 80000}
# sample_counts = {'instruct.jsonl': 0,'dolly.jsonl': 0,'alpaca_data.jsonl': 0,'sorted_20240304.jsonl': 40000,'sorted_math_questions_bert_large.jsonl': 23000}

# generate_mixture(input_dir, ratio_path, mixture_path, sample_counts)