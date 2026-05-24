# SVM 乳腺肿瘤分类

使用支持向量机（SVM）完成乳腺肿瘤良恶性分类。数据集来自 `sklearn.datasets.load_breast_cancer`，输入不是原始图片，而是 30 个数值特征。

## 项目结构

```text
.
├── data.py          # 生成训练集和测试集
├── data_show.py     # 数据可视化
├── train.py         # 训练模型
├── test.py          # 测试模型
├── environment.yml  # conda 环境
└── README.md
```

运行后会自动生成：

```text
datasets/   # 数据集文件
models/     # 模型文件
outputs/    # 训练和测试结果
datapic/    # 数据可视化图片
```

## 安装

```bash
git clone https://github.com/your-name/svm-breast-cancer.git
cd svm-breast-cancer
```

```bash
conda env create -f environment.yml
conda activate svm-course-project
```

## 快速运行

```bash
python data.py
python data_show.py
python train.py
python test.py
```

## 各文件用法

### 1. 生成数据集

```bash
python data.py
```

可选参数：

```bash
python data.py --output_dir datasets --test_size 0.2 --random_state 42
```

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--output_dir` | `datasets` | 数据输出目录 |
| `--test_size` | `0.2` | 测试集比例 |
| `--random_state` | `42` | 随机种子 |

输出：

```text
datasets/train/X_train.csv
datasets/train/y_train.csv
datasets/test/X_test.csv
datasets/test/y_test.csv
datasets/dataset_info.json
```

### 2. 数据可视化

```bash
python data_show.py
```

可选参数：

```bash
python data_show.py --output_dir datapic --num_samples 50 --random_state 42
```

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--output_dir` | `datapic` | 图片输出目录 |
| `--num_samples` | `50` | 随机展示样本数 |
| `--random_state` | `42` | 随机种子 |

输出：

```text
datapic/raw_feature_charts/
datapic/standardized_feature_charts/
datapic/pca_overview_selected_50.png
```

### 3. 训练模型

```bash
python train.py
```

可选参数：

```bash
python train.py --train_dir datasets/train --model_dir models --output_dir outputs --cv 5
```

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--train_dir` | `datasets/train` | 训练集目录 |
| `--model_dir` | `models` | 模型输出目录 |
| `--output_dir` | `outputs` | 结果输出目录 |
| `--cv` | `5` | 交叉验证折数 |

训练内容：

```text
linear SVM
rbf SVM
poly SVM
GridSearchCV 最优模型搜索
```

输出：

```text
models/best_model.joblib
models/checkpoints/
outputs/train_results.csv
outputs/grid_search_results.csv
outputs/best_model_info.json
```

### 4. 测试模型

```bash
python test.py
```

默认使用：

```text
models/best_model.joblib
```

指定其他模型：

```bash
python test.py --model_path models/checkpoints/svm_linear_C1.joblib
```

可选参数：

```bash
python test.py --test_dir datasets/test --model_path models/best_model.joblib --output_dir outputs
```

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--test_dir` | `datasets/test` | 测试集目录 |
| `--model_path` | `models/best_model.joblib` | 模型路径 |
| `--output_dir` | `outputs` | 测试结果目录 |

输出：

```text
outputs/test_metrics.json
outputs/test_predictions.csv
outputs/confusion_matrix.png
outputs/roc_curve.png
```

## 评价指标

测试阶段输出：

```text
accuracy
precision
recall
f1
roc_auc
```

其中 `accuracy` 是正确率。医疗分类任务中还应重点看恶性样本的召回率，避免漏诊。

## 环境

`environment.yml`：

```yaml
name: svm-course-project
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - numpy
  - pandas
  - scikit-learn
  - matplotlib
  - seaborn
  - joblib
```

## 运行顺序

```bash
conda env create -f environment.yml
conda activate svm-course-project

python data.py
python data_show.py
python train.py
python test.py
```