import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    RocCurveDisplay
)



def parse_args():
    parser = argparse.ArgumentParser(description="Test SVM model.")
    parser.add_argument("--test_dir", type=str, default="datasets/test", help="Test dataset directory.")
    parser.add_argument("--model_path", type=str, default="models/best_model.joblib", help="Model path.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Test result output directory.")
    return parser.parse_args()


def load_test_data(test_dir):
    test_dir = Path(test_dir)

    X_test_path = test_dir / "X_test.csv"
    y_test_path = test_dir / "y_test.csv"

    if not X_test_path.exists() or not y_test_path.exists():
        raise FileNotFoundError("测试集文件不存在，请先运行 data.py。")

    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)["target"]

    return X_test, y_test


def load_model(model_path):
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}，请先运行 train.py，或用 --model_path 指定其他模型。")

    model = joblib.load(model_path)
    return model


def get_prediction_scores(model, X_test):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]

    if hasattr(model, "decision_function"):
        return model.decision_function(X_test)

    return None


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_score = get_prediction_scores(model, X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred))
    }

    if y_score is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_score))
    else:
        metrics["roc_auc"] = None

    return metrics, y_pred, y_score


def save_predictions(X_test, y_test, y_pred, y_score, output_dir):
    prediction_df = X_test.copy()
    prediction_df["true_label"] = y_test.values
    prediction_df["predicted_label"] = y_pred

    prediction_df["true_label_name"] = prediction_df["true_label"].map({
        0: "malignant",
        1: "benign"
    })

    prediction_df["predicted_label_name"] = prediction_df["predicted_label"].map({
        0: "malignant",
        1: "benign"
    })

    prediction_df["is_correct"] = prediction_df["true_label"] == prediction_df["predicted_label"]

    if y_score is not None:
        prediction_df["score_for_benign_class"] = y_score

    save_path = output_dir / "test_predictions.csv"
    prediction_df.to_csv(save_path, index=False, encoding="utf-8-sig")

    return save_path


def plot_confusion_matrix(y_test, y_pred, output_dir):
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["malignant 恶性", "benign 良性"],
        yticklabels=["malignant 恶性", "benign 良性"]
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    save_path = output_dir / "confusion_matrix.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    return save_path


def plot_roc_curve(model, X_test, y_test, output_dir):
    save_path = output_dir / "roc_curve.png"

    plt.figure(figsize=(6, 5))
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return save_path


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_test, y_test = load_test_data(args.test_dir)
    model = load_model(args.model_path)

    print("测试数据和模型加载完成。")
    print(f"测试样本数: {X_test.shape[0]}")
    print(f"特征数量: {X_test.shape[1]}")
    print(f"使用模型: {args.model_path}")

    metrics, y_pred, y_score = evaluate_model(model, X_test, y_test)

    print("=" * 60)
    print("测试集评价指标")
    print("=" * 60)
    for key, value in metrics.items():
        if value is None:
            print(f"{key}: None")
        else:
            print(f"{key}: {value:.4f}")

    print()
    print("分类报告:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["malignant 恶性", "benign 良性"]
    ))

    metrics_path = output_dir / "test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    predictions_path = save_predictions(
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
        y_score=y_score,
        output_dir=output_dir
    )

    confusion_matrix_path = plot_confusion_matrix(
        y_test=y_test,
        y_pred=y_pred,
        output_dir=output_dir
    )

    roc_curve_path = plot_roc_curve(
        model=model,
        X_test=X_test,
        y_test=y_test,
        output_dir=output_dir
    )

    print("=" * 60)
    print("测试完成")
    print("=" * 60)
    print(f"测试指标保存路径: {metrics_path}")
    print(f"预测结果保存路径: {predictions_path}")
    print(f"混淆矩阵保存路径: {confusion_matrix_path}")
    print(f"ROC 曲线保存路径: {roc_curve_path}")


if __name__ == "__main__":
    main()