# Physics Knowledge Embedding

面向 PINN 与算子网络的物理领域知识嵌入研究代码。仓库包含两条相互独立的实验路径：

1. `src/physics_embed`：二维 PDE 制造解、基础 PINN、边界条件、对称性、守恒关系和降阶模型约束；
2. `scripts/official`：基于 Transolver 主干的 Darcy、Elasticity、Navier–Stokes 和 Plasticity 实验入口。

数据集、模型权重、运行日志和报告材料不进入仓库。

## 方法范围

代码覆盖“数据与任务定义—模型初始化—训练约束—预测校验”四个环节中的可复用组件，包括：

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
pip install -e ".[dev,viz,transolver]"
```

如果仓库已克隆但未拉取 Transolver：

```bash
git submodule update --init --recursive
```

Transolver 作为上游 Git 子模块引用，本仓库不复制其源码。上游项目：
https://github.com/thuml/Transolver

## Transolver 实验

下载公开基准数据：

```bash
python scripts/download_official.py
```

运行一个最小 Darcy 数据基线：

```bash
python scripts/official/exp_darcy.py \
  --gpu 0 \
  --group A \
  --epochs 2 \
  --data_path ./data/official \
  --ckpt_dir ./runs/darcy_A
```

主要实验分组：

- `A`：数据监督基线；
- `C` / `D`：固定或分阶段加入物理残差；
- `HARD`：将零 Dirichlet 边界作为输出结构；
- `MATCH`：匹配标签与预测的离散残差；
- `DYN`：动态限制物理损失权重；
- `S100*`：稀疏训练样本设置；
- `POS` / `CONS`：材料或守恒关系的探索性约束。

`scripts/run_official_seq.sh` 和 `scripts/run_phase2.sh` 提供顺序运行示例，所有路径均可通过环境变量覆盖。

## PINN 工具

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
