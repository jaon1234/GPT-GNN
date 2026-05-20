import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from explainability import FineGrainedLayerCAM


class SmallBinaryCNN(nn.Module):
    def __init__(self, outputs=2):
        super(SmallBinaryCNN, self).__init__()
        self.layer2 = nn.Sequential(nn.Conv2d(3, 4, kernel_size=3, padding=1), nn.ReLU(inplace=False))
        self.layer3 = nn.Sequential(nn.Conv2d(4, 6, kernel_size=3, padding=1), nn.ReLU(inplace=False))
        self.layer4 = nn.Sequential(nn.Conv2d(6, 8, kernel_size=3, padding=1), nn.ReLU(inplace=False))
        self.classifier = nn.Linear(8, outputs)

    def forward(self, x):
        x = self.layer2(x)
        x = F.max_pool2d(x, 2)
        x = self.layer3(x)
        x = F.max_pool2d(x, 2)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        return self.classifier(x)


class FineGrainedLayerCAMTest(unittest.TestCase):
    def test_generates_normalized_heatmap_for_two_output_binary_model(self):
        torch.manual_seed(7)
        model = SmallBinaryCNN(outputs=2)
        input_tensor = torch.randn(2, 3, 32, 32)

        explainer = FineGrainedLayerCAM(model, use_input_gradient_refine=True)
        heatmap = explainer.generate(input_tensor, target_class=1)
        explainer.remove_hooks()

        self.assertEqual(tuple(heatmap.shape), (2, 32, 32))
        self.assertGreaterEqual(float(heatmap.min()), 0.0)
        self.assertLessEqual(float(heatmap.max()), 1.0)

    def test_supports_single_output_binary_targets(self):
        torch.manual_seed(11)
        model = SmallBinaryCNN(outputs=1)
        input_tensor = torch.randn(1, 3, 24, 24)

        explainer = FineGrainedLayerCAM(
            model,
            target_layers=["layer2.1", "layer3.1", "layer4.1"],
            use_input_gradient_refine=False,
        )
        positive_heatmap = explainer(input_tensor, target_class=1)
        negative_heatmap = explainer(input_tensor, target_class=0)
        explainer.remove_hooks()

        self.assertEqual(tuple(positive_heatmap.shape), (1, 24, 24))
        self.assertEqual(tuple(negative_heatmap.shape), (1, 24, 24))
        self.assertGreaterEqual(float(positive_heatmap.min()), 0.0)
        self.assertLessEqual(float(positive_heatmap.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
