# Physics Knowledge Embedding

面向物理场智能建模的领域知识嵌入研究代码。仓库包含数据准备、模型训练、知识约束、实验调度和结果评估等功能，支持热传导、流体、固体力学等典型物理场景。

数据集、模型权重、运行日志和报告材料不进入仓库。

## 方法范围

代码围绕“数据与任务定义—模型初始化—训练约束—预测校验”四个环节组织，主要包括：

- PDE 与本构残差；
- 软/硬边界条件；
- 镜像与周期对称性；
- 动态损失权重及离散残差匹配；
- 小样本数据设置；
- 预测后对称投影和物理一致性评估。

## 安装

需要 Python 3.10 及以上版本，以及支持所用 PyTorch 版本的 CUDA 环境。

```bash
git clone --recurse-submodules https://github.com/Syunjyan/knowledge_embed_q3.git
cd knowledge_embed_q3
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,viz,experiments]"
```

如果仓库已克隆但未拉取外部依赖：

```bash
git submodule update --init --recursive
```

## 实验运行

下载公开基准数据：

```bash
python scripts/download_official.py
```

运行一个最小示例：

```bash
python scripts/official/exp_darcy.py \
  --gpu 0 \
  --group A \
  --epochs 2 \
  --data_path ./data/official \
  --ckpt_dir ./runs/darcy_A
```

`scripts/run_official_seq.sh` 和 `scripts/run_phase2.sh` 提供顺序运行示例，所有路径均可通过环境变量覆盖。

## 数据与训练工具

```bash
physics-embed-data \
  --equation heat \
  --out data/heat.npz \
  --spatial-resolution 32

physics-embed-train \
  --equation heat \
  --dataset data/heat.npz \
  --output-dir runs/heat \
  --epochs 2000

physics-embed-evaluate \
  --run-dir runs/heat \
  --dataset data/heat.npz
```

## 验证

```bash
pytest
python -m compileall -q embed src scripts tests
```
