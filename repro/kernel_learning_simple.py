"""复现《Kernel Learning》论文中最简单模型的脚本。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class DatasetConfig:
    sample_count: int = 1000
    random_seed: int = 0


def generate_samples(config: DatasetConfig) -> tuple[np.ndarray, np.ndarray]:
    """生成简单模型的训练样本。

    特征 X 和噪声 U 均为 [0, 1] 上的均匀分布，目标变量 Y = 0.5 * X + 0.5 * U。
    """

    rng = np.random.default_rng(config.random_seed)
    x_samples = rng.uniform(0, 1, size=config.sample_count)
    noise = rng.uniform(0, 1, size=config.sample_count)
    y_samples = 0.5 * x_samples + 0.5 * noise
    return x_samples, y_samples


def gaussian_kernel_weights(
    query: float, x_samples: np.ndarray, bandwidth: float
) -> np.ndarray:
    """根据查询点与样本之间的欧式距离计算高斯核权重。"""

    scaled = (x_samples - query) / bandwidth
    raw_weights = np.exp(-0.5 * scaled**2)
    normalizer = raw_weights.sum()
    if normalizer == 0:
        return np.full_like(raw_weights, 1 / len(raw_weights))
    return raw_weights / normalizer


def estimate_conditional_mean(
    query: float, x_samples: np.ndarray, y_samples: np.ndarray, bandwidth: float
) -> float:
    """通过加权平均估计条件分布的期望。"""

    weights = gaussian_kernel_weights(query, x_samples, bandwidth)
    return float(np.dot(weights, y_samples))


def sample_estimated_distribution(
    query: float,
    x_samples: np.ndarray,
    y_samples: np.ndarray,
    bandwidth: float,
    sample_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """根据核权重在样本点上采样近似条件分布。"""

    weights = gaussian_kernel_weights(query, x_samples, bandwidth)
    return rng.choice(y_samples, size=sample_size, p=weights)


def true_conditional_statistics(query: float) -> tuple[float, float]:
    """返回该模型的真实条件期望与方差。"""

    # 由于 U ~ Uniform(0,1) 且 Y = 0.5 X + 0.5 U，得 Y|X=x 是 Uniform([x/2 + 0, x/2 + 0.5])
    # 期望为 x / 2 + 0.25，方差为 (0.5)^2 / 12。
    expected = 0.5 * query + 0.25
    variance = (0.5**2) / 12
    return expected, variance


def main() -> None:
    parser = argparse.ArgumentParser(description="复现最简单的核学习模型")
    parser.add_argument(
        "--bandwidth", type=float, default=0.05, help="核带宽，越大越平滑"
    )
    parser.add_argument("--sample-count", type=int, default=2000)
    parser.add_argument(
        "--queries",
        type=float,
        nargs="+",
        default=[0.1, 0.5, 0.9],
        help="用于评估的查询特征点",
    )
    parser.add_argument(
        "--draw-samples", type=int, default=500, help="用于近似分布的采样数量"
    )
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    x_samples, y_samples = generate_samples(
        DatasetConfig(sample_count=args.sample_count, random_seed=args.random_seed)
    )
    rng = np.random.default_rng(args.random_seed)

    print("Kernel Learning 最简单模型复现")
    print(f"样本数：{args.sample_count}，带宽：{args.bandwidth}")
    true_variance = (0.5**2) / 12
    for query in args.queries:
        mean_estimate = estimate_conditional_mean(
            query, x_samples, y_samples, args.bandwidth
        )
        true_mean, _ = true_conditional_statistics(query)
        residual = abs(mean_estimate - true_mean)
        sampled = sample_estimated_distribution(
            query, x_samples, y_samples, args.bandwidth, args.draw_samples, rng
        )
        sample_std = float(np.std(sampled))

        print("\n查询点", query)
        print(f"  核估计平均值：{mean_estimate:.4f}，真实均值：{true_mean:.4f}，误差 {residual:.4f}")
        print(f"  抽样估计方差：{sample_std**2:.5f}，理论方差：{true_variance:.5f}")


if __name__ == "__main__":
    main()
