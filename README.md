# Better Mixture 竞赛开发套件

## 1. 安装依赖并下载数据

```bash
# 建议在 conda 环境中安装
conda create -n mixture python=3.10
conda activate mixture

bash install.sh
```

## 2. 根据可用资源设置环境变量

```bash
# 修改 entry.env
```

## 3. 实现配比和采样算法
```bash
# 自定义 solution/get_mixture.py
```

## 4. 执行算法、训练、评测流程
```bash
bash entry.sh
```

## 5. 向天池提交

将所需文件打包为 zip 格式，提交到天池评测。

```bash
zip -r submit.zip entry.env solution output
```
