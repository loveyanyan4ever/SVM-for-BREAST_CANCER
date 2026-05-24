import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare breast cancer dataset for SVM experiment.")
    parser.add_argument("--output_dir", type=str, default="datasets", help="Dataset output directory.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set ratio.")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    data = load_breast_cancer()

    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y
    )

    X_train.to_csv(train_dir / "X_train.csv", index=False, encoding="utf-8-sig")
    y_train.to_csv(train_dir / "y_train.csv", index=False, encoding="utf-8-sig")

    X_test.to_csv(test_dir / "X_test.csv", index=False, encoding="utf-8-sig")
    y_test.to_csv(test_dir / "y_test.csv", index=False, encoding="utf-8-sig")

    dataset_info = {
        "dataset_name": "Breast Cancer Wisconsin Diagnostic Dataset",
        "total_samples": int(X.shape[0]),
        "feature_count": int(X.shape[1]),
        "target_names": list(data.target_names),
        "label_mapping": {
            "0": "malignant",
            "1": "benign"
        },
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "test_size": args.test_size,
        "random_state": args.random_state,
        "feature_names": list(data.feature_names),
        "train_label_distribution": y_train.value_counts().to_dict(),
        "test_label_distribution": y_test.value_counts().to_dict()
    }

    with open(output_dir / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=4)

    print("数据准备完成。")
    print(f"训练集保存路径: {train_dir}")
    print(f"测试集保存路径: {test_dir}")
    print(f"训练集样本数: {X_train.shape[0]}")
    print(f"测试集样本数: {X_test.shape[0]}")
    print("标签含义: 0 = malignant 恶性, 1 = benign 良性")


if __name__ == "__main__":
    main()