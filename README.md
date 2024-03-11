# Better Mixture 竞赛开发套件

## 1. 安装依赖并下载数据

```bash
# 建议在 conda 环境中安装
conda create -n mixture python=3.10
conda activate mixture

bash install.sh
```

## 2. 根据可用资源设置环境变量

本方案使用的环境A100-SXM4-80GB * 1卡，所以MICRO_BATCH_SIZE可以设大点，另外经过多次测试LEARNING_RATE设为1e-3效果是比较好

```bash
# 修改 entry.env
```

## 3. 实现配比和采样算法
Baichuan2-7B 模型没进过 PEFT（LoRA）训练的score得分大概在0.97左右。gsm8k和scrolls_summscreenfd的单项得分非常低。
前期经过多次试验测试观察到，针对性提升解决truthfulqa_mc、gsm8k和scrolls_summscreenfd的任务能力的效果较为明显。

因次本方案使用了两个自训练的算子模型，分为：

1、筛选数学相关语料算子模型（reyvan/bert_large_maths）

2、筛选逻辑性，总结性和真实性较强的语料（reyvan/bert_large_TruthfulAndSumm）

具体训练方法是：利用gpt4生成一些语料作为训练数据。筛选数学相关语料的效果非常好的，接近100%的准确率，但是后者的准确大概在85-90%之间，后续可以针对这个算子模型进行优化。

```bash
# python sort_TruthfulQA.py
# python sort_Maths.py
```

```bash
# 自定义 solution/get_mixture.py
```



## 4. 执行算法、训练、评测流程
```bash
bash entry.sh
```

