import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def parse_args():
    parser = argparse.ArgumentParser(
        description="Show random samples from Breast Cancer dataset as feature charts."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datapic",
        help="Output directory for sample visualization images."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Number of random samples to visualize."
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed."
    )
    return parser.parse_args()


def load_dataset():
    data = load_breast_cancer()

    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")

    label_map = {
        0: "malignant",
        1: "benign"
    }

    return X, y, label_map


def save_sample_feature_chart(sample_id, features, label, label_name, output_dir):
    plt.figure(figsize=(12, 5))

    feature_names = features.index
    feature_values = features.values

    plt.bar(range(len(feature_values)), feature_values)

    plt.xticks(
        ticks=range(len(feature_names)),
        labels=feature_names,
        rotation=90,
        fontsize=7
    )

    plt.ylabel("Feature Value")
    plt.title(f"Sample {sample_id} - Label: {label_name} ({label})")
    plt.tight_layout()

    save_path = output_dir / f"sample_{sample_id:03d}_{label_name}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    return save_path


def save_normalized_sample_feature_chart(sample_id, features, label, label_name, output_dir):
    plt.figure(figsize=(12, 5))

    feature_names = features.index
    feature_values = features.values

    plt.bar(range(len(feature_values)), feature_values)

    plt.axhline(0, linestyle="--", linewidth=1)

    plt.xticks(
        ticks=range(len(feature_names)),
        labels=feature_names,
        rotation=90,
        fontsize=7
    )

    plt.ylabel("Standardized Feature Value")
    plt.title(f"Standardized Sample {sample_id} - Label: {label_name} ({label})")
    plt.tight_layout()

    save_path = output_dir / f"sample_{sample_id:03d}_{label_name}_standardized.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    return save_path


def save_pca_overview(X, y, label_map, selected_indices, output_dir):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))

    for label in [0, 1]:
        mask = y == label
        plt.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            alpha=0.5,
            label=label_map[label]
        )

    plt.scatter(
        X_pca[selected_indices, 0],
        X_pca[selected_indices, 1],
        s=80,
        marker="x",
        label="selected 50 samples"
    )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Overview of Breast Cancer Dataset")
    plt.legend()
    plt.tight_layout()

    save_path = output_dir / "pca_overview_selected_50.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    return save_path


def save_random_sample_table(X, y, label_map, selected_indices, output_dir):
    selected_data = X.iloc[selected_indices].copy()
    selected_data.insert(0, "sample_id", selected_indices)
    selected_data["target"] = y.iloc[selected_indices].values
    selected_data["label_name"] = selected_data["target"].map(label_map)

    save_path = output_dir / "selected_50_samples.csv"
    selected_data.to_csv(save_path, index=False, encoding="utf-8-sig")

    return save_path


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw_feature_charts"
    standardized_dir = output_dir / "standardized_feature_charts"

    raw_dir.mkdir(parents=True, exist_ok=True)
    standardized_dir.mkdir(parents=True, exist_ok=True)

    X, y, label_map = load_dataset()

    rng = np.random.default_rng(args.random_state)

    num_samples = min(args.num_samples, len(X))
    selected_indices = rng.choice(len(X), size=num_samples, replace=False)
    selected_indices = sorted(selected_indices)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns
    )

    print("=" * 60)
    print("数据集说明")
    print("=" * 60)
    print("sklearn 的 Breast Cancer 数据集不是原始图片数据。")
    print("它是从医学图像中提取出的 30 个数值特征组成的表格数据。")
    print(f"样本数量: {X.shape[0]}")
    print(f"特征数量: {X.shape[1]}")
    print("标签含义: 0 = malignant 恶性, 1 = benign 良性")

    print("=" * 60)
    print(f"开始生成随机 {num_samples} 个样本的特征图")
    print("=" * 60)

    for sample_id in selected_indices:
        label = int(y.iloc[sample_id])
        label_name = label_map[label]

        save_sample_feature_chart(
            sample_id=sample_id,
            features=X.iloc[sample_id],
            label=label,
            label_name=label_name,
            output_dir=raw_dir
        )

        save_normalized_sample_feature_chart(
            sample_id=sample_id,
            features=X_scaled.iloc[sample_id],
            label=label,
            label_name=label_name,
            output_dir=standardized_dir
        )

    pca_path = save_pca_overview(
        X=X,
        y=y,
        label_map=label_map,
        selected_indices=selected_indices,
        output_dir=output_dir
    )

    table_path = save_random_sample_table(
        X=X,
        y=y,
        label_map=label_map,
        selected_indices=selected_indices,
        output_dir=output_dir
    )

    print("可视化生成完成。")
    print(f"原始特征柱状图文件夹: {raw_dir}")
    print(f"标准化特征柱状图文件夹: {standardized_dir}")
    print(f"PCA 总览图: {pca_path}")
    print(f"随机样本表格: {table_path}")


if __name__ == "__main__":
    main()