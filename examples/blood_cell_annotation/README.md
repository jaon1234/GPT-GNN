# 血细胞镜检图片预训练模型检测与标注示例

这个示例会自动下载少量公开血细胞镜检图片，然后使用预训练目标检测模型定位并标注细胞。

默认使用两个 Hugging Face 上公开的 BCCD 血细胞检测模型：

- `keremberke/yolov8n-blood-cell-detection`：YOLOv8n，支持 `Platelets`、`RBC`、`WBC`
- `keremberke/yolov8s-blood-cell-detection`：YOLOv8s，支持 `Platelets`、`RBC`、`WBC`

图片来源包括：

- Hugging Face 数据集 `keremberke/blood-cell-object-detection` 的测试集样本
- Wikimedia Commons 上的公开血细胞图片，例如 `Peripheral blood smear.jpg`、`Redbloodcells.jpg`

> 注意：这是科研/教学用途示例，不应用作临床诊断结论。不同来源图片的染色方式、显微镜类型和尺度差异较大，通用标注结果需要人工复核。

## 安装

建议使用单独虚拟环境，避免影响仓库原有 GPT-GNN 旧版依赖：

```bash
python3 -m venv .venv-blood-cell
source .venv-blood-cell/bin/activate
python -m pip install --upgrade pip
python -m pip install -r examples/blood_cell_annotation/requirements.txt
```

## 运行

```bash
python examples/blood_cell_annotation/annotate_blood_cells.py \
  --output-dir examples/blood_cell_annotation/runs/demo \
  --max-dataset-images 5 \
  --max-images 8
```

输出内容：

- `images/`：下载到本地的原始图片
- `annotations/<model_name>/`：每个预训练模型各自生成的带框标注图片
- `detections.json`：每张图、每个模型的结构化检测框、类别和置信度
- `sources.json`：图片来源、许可与原始 URL

## 可选：只运行一个模型

```bash
python examples/blood_cell_annotation/annotate_blood_cells.py \
  --models yolov8s_bccd \
  --output-dir examples/blood_cell_annotation/runs/yolov8s
```

## 可选：Cellpose 细胞分割

脚本还提供 `--run-cellpose` 开关，用于在安装 `cellpose` 后额外运行通用细胞分割预训练模型，并把分割实例转换成定位框：

```bash
python -m pip install cellpose
python examples/blood_cell_annotation/annotate_blood_cells.py \
  --run-cellpose \
  --output-dir examples/blood_cell_annotation/runs/with-cellpose
```

Cellpose 输出的类别统一记为 `cellpose_cell`，适合作为细胞候选区域定位结果；它不会区分红细胞、白细胞和血小板。
