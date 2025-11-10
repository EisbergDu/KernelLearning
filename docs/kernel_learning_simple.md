# Kernel Learning 第一模型复现说明

## 目的
复现文中最简单的合成模型 $Y = \frac{1}{2} X + \frac{1}{2} U$，并用核权重估计条件分布，再将估计的均值与理论值对齐。

## 环境依赖
在本仓库根目录已经列出的 `requirements.txt` 中规定了运行所需的版本，建议执行：

```
pip install -r requirements.txt
```

## 运行示例模型
执行如下命令即可运行复现脚本：

```
python repro/kernel_learning_simple.py --bandwidth 0.05 --sample-count 2000 --queries 0.1 0.5 0.9
```

可选参数说明：
- `--bandwidth`：核的带宽，越小越强调邻近样本。
- `--sample-count`：生成训练样本数量。
- `--queries`：用于评估的特征点。
- `--draw-samples`：用于近似条件分布采样的数量，默认 500。

## 核心验证点
1. 输出应包含每个查询点的核估计均值、理论均值（$0.5x + 0.25$）以及误差，误差应处于 0.05 左右。
2. 输出的抽样估计方差应接近理论值 $\mathrm{Var}(Y|X=x) = 0.5^2 / 12$。

## 可选进一步探索
1. 调整 `--bandwidth` 观察均值估计的偏移与方差变化。
2. 增加样本数检查估计是否更稳定。
