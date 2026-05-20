# FineGrainedLayerCAM

`FineGrainedLayerCAM` 是一个独立的 PyTorch 可解释性类，不会替换现有 Grad-CAM 代码。它实现的是多层 Layer-CAM，并可选用输入梯度进一步细化边缘和纹理响应，适合 ResNet18 二分类场景中需要比 Grad-CAM 更丰富细节的热力图。

## 基本用法

```python
from explainability import FineGrainedLayerCAM

final_net.eval()

explainer = FineGrainedLayerCAM(
    final_net=final_net,
    target_layers=[
        final_net.layer2[-1],
        final_net.layer3[-1],
        final_net.layer4[-1],
    ],
    use_input_gradient_refine=True,
    refine_strength=0.5,
)

# input_tensor: [B, C, H, W]
# 两个 logit 的二分类模型：target_class=1 表示解释正类
heatmap = explainer.generate(input_tensor, target_class=1)

# heatmap: [B, H, W]，数值范围 [0, 1]
```

## 单输出二分类

如果模型输出形状是 `[B, 1]`：

- `target_class=1`：解释正类；
- `target_class=0`：解释负类，内部会对单输出取相反数作为目标。

```python
heatmap_positive = explainer(input_tensor, target_class=1)
heatmap_negative = explainer(input_tensor, target_class=0)
```

## 自动选择 ResNet 层

对 ResNet18 这类有 `layer2`、`layer3`、`layer4` 的模型，可以省略 `target_layers`：

```python
explainer = FineGrainedLayerCAM(final_net)
heatmap = explainer(input_tensor, target_class=1)
```

默认会使用：

```python
[
    final_net.layer2[-1],
    final_net.layer3[-1],
    final_net.layer4[-1],
]
```

如果你只想解释某几层，也可以传层名：

```python
explainer = FineGrainedLayerCAM(
    final_net,
    target_layers=["layer3.1", "layer4.1"],
)
```

使用完毕后可以移除 hooks：

```python
explainer.remove_hooks()
```
