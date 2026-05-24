import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, GridSearchCV


def parse_args():
    parser = argparse.ArgumentParser(description="Train SVM models.")
    parser.add_argument("--train_dir", type=str, default="datasets/train", help="Training dataset directory.")
    parser.add_argument("--model_dir", type=str, default="models", help="Model output directory.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Training result output directory.")
    parser.add_argument("--cv", type=int, default=5, help="Cross validation folds.")
    return parser.parse_args()


def load_train_data(train_dir):
    train_dir = Path(train_dir)

    X_train_path = train_dir / "X_train.csv"
    y_train_path = train_dir / "y_train.csv"

    if not X_train_path.exists() or not y_train_path.exists():
        raise FileNotFoundError("训练集文件不存在，请先运行 data.py。")

    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)["target"]

    return X_train, y_train


def build_candidate_models():
    models = {
        "svm_linear_C1": Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="linear", C=1, probability=True, random_state=42))
        ]),
        "svm_rbf_C1_gamma_scale": Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", C=1, gamma="scale", probability=True, random_state=42))
        ]),
        "svm_poly_C1_degree3": Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="poly", C=1, degree=3, gamma="scale", probability=True, random_state=42))
        ])
    }

    return models


def train_candidate_models(models, X_train, y_train, checkpoint_dir, cv):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }

    results = []

    for model_name, model in models.items():
        print("=" * 60)
        print(f"正在训练候选模型: {model_name}")
        print("=" * 60)

        cv_result = cross_validate(
            model,
            X_train,
            y_train,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            return_train_score=True
        )

        model.fit(X_train, y_train)

        checkpoint_path = checkpoint_dir / f"{model_name}.joblib"
        joblib.dump(model, checkpoint_path)

        result = {
            "model_name": model_name,
            "checkpoint_path": str(checkpoint_path),
            "train_accuracy_mean": cv_result["train_accuracy"].mean(),
            "valid_accuracy_mean": cv_result["test_accuracy"].mean(),
            "valid_precision_mean": cv_result["test_precision"].mean(),
            "valid_recall_mean": cv_result["test_recall"].mean(),
            "valid_f1_mean": cv_result["test_f1"].mean(),
            "valid_roc_auc_mean": cv_result["test_roc_auc"].mean()
        }

        results.append(result)

        print(f"模型已保存: {checkpoint_path}")
        print(f"CV Accuracy: {result['valid_accuracy_mean']:.4f}")
        print(f"CV F1: {result['valid_f1_mean']:.4f}")
        print(f"CV ROC-AUC: {result['valid_roc_auc_mean']:.4f}")

    return pd.DataFrame(results)


def grid_search_best_model(X_train, y_train, cv):
    print("=" * 60)
    print("开始 GridSearchCV 搜索最佳模型")
    print("=" * 60)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(probability=True, random_state=42))
    ])

    param_grid = [
        {
            "svc__kernel": ["linear"],
            "svc__C": [0.01, 0.1, 1, 10, 100]
        },
        {
            "svc__kernel": ["rbf"],
            "svc__C": [0.1, 1, 10, 100],
            "svc__gamma": [0.001, 0.01, 0.1, 1, "scale"]
        },
        {
            "svc__kernel": ["poly"],
            "svc__C": [0.1, 1, 10],
            "svc__degree": [2, 3],
            "svc__gamma": ["scale", "auto"]
        }
    ]

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )

    grid_search.fit(X_train, y_train)

    print(f"最佳参数: {grid_search.best_params_}")
    print(f"交叉验证最佳 F1: {grid_search.best_score_:.4f}")

    return grid_search


def main():
    args = parse_args()

    model_dir = Path(args.model_dir)
    checkpoint_dir = model_dir / "checkpoints"
    output_dir = Path(args.output_dir)

    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train = load_train_data(args.train_dir)

    print("训练数据加载完成。")
    print(f"训练样本数: {X_train.shape[0]}")
    print(f"特征数量: {X_train.shape[1]}")

    candidate_models = build_candidate_models()

    results_df = train_candidate_models(
        models=candidate_models,
        X_train=X_train,
        y_train=y_train,
        checkpoint_dir=checkpoint_dir,
        cv=args.cv
    )

    train_results_path = output_dir / "train_results.csv"
    results_df.to_csv(train_results_path, index=False, encoding="utf-8-sig")

    grid_search = grid_search_best_model(X_train, y_train, cv=args.cv)

    best_model_path = model_dir / "best_model.joblib"
    joblib.dump(grid_search.best_estimator_, best_model_path)

    grid_search_result_path = output_dir / "grid_search_results.csv"
    pd.DataFrame(grid_search.cv_results_).to_csv(
        grid_search_result_path,
        index=False,
        encoding="utf-8-sig"
    )

    best_info = {
        "best_model_path": str(best_model_path),
        "best_params": grid_search.best_params_,
        "best_cv_f1": float(grid_search.best_score_),
        "selection_metric": "f1"
    }

    with open(output_dir / "best_model_info.json", "w", encoding="utf-8") as f:
        json.dump(best_info, f, ensure_ascii=False, indent=4)

    print("=" * 60)
    print("训练完成")
    print("=" * 60)
    print(f"候选模型训练结果: {train_results_path}")
    print(f"GridSearch 详细结果: {grid_search_result_path}")
    print(f"最佳模型已保存: {best_model_path}")


if __name__ == "__main__":
    main()