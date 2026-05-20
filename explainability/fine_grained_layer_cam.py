"""Fine-grained CAM utilities for image classifiers.

The implementation is intentionally dependency-free beyond PyTorch so it can
be dropped into existing training code that already owns the model instance.
"""

from __future__ import print_function

import torch
import torch.nn.functional as F


class FineGrainedLayerCAM(object):
    """Generate detailed class activation maps with multi-layer Layer-CAM.

    This class keeps Grad-CAM untouched and implements a separate algorithm
    based on Layer-CAM. Compared with vanilla Grad-CAM, it uses spatially
    resolved positive gradients instead of one global gradient weight per
    channel, then fuses several convolutional layers. For ResNet18 this means
    shallower layers can contribute edge/texture detail while deeper layers
    keep class semantics.

    Args:
        final_net: Trained PyTorch classifier, for example a ResNet18 model.
        target_layers: Optional list of layer modules or layer-name strings.
            If omitted, ResNet-style ``layer2[-1]``, ``layer3[-1]`` and
            ``layer4[-1]`` are used when available.
        layer_weights: Optional fusion weights with the same length as
            ``target_layers``. When omitted, all layers are averaged.
        device: Optional torch device. If omitted, the device of ``final_net``
            parameters is used when possible.
        use_input_gradient_refine: If True, multiply the fused CAM by a
            normalized input-gradient saliency map to sharpen local detail.
        refine_strength: Strength of input-gradient refinement. ``0`` disables
            the effect; ``1`` means ``cam * (1 + saliency)``.
        eps: Small value used in min-max normalization.
    """

    def __init__(
        self,
        final_net,
        target_layers=None,
        layer_weights=None,
        device=None,
        use_input_gradient_refine=True,
        refine_strength=0.5,
        eps=1e-7,
    ):
        self.final_net = final_net
        self.device = device or self._infer_device(final_net)
        self.target_layers = self._resolve_target_layers(final_net, target_layers)
        self.layer_weights = self._prepare_layer_weights(layer_weights, len(self.target_layers))
        self.use_input_gradient_refine = use_input_gradient_refine
        self.refine_strength = float(refine_strength)
        self.eps = eps

        self._activations = {}
        self._gradients = {}
        self._handles = []
        self._register_hooks()

    def generate(
        self,
        input_tensor,
        target_class=None,
        resize_to=None,
        retain_graph=False,
        create_graph=False,
        return_logits=False,
    ):
        """Compute a normalized CAM for each input image.

        Args:
            input_tensor: Tensor of shape ``[B, C, H, W]``.
            target_class: Class to explain. For two-logit binary classifiers,
                pass ``0`` or ``1``. For one-logit binary classifiers,
                ``1`` explains the positive class and ``0`` explains the
                negative class by negating the scalar output. If omitted, the
                predicted class is used.
            resize_to: Optional ``(height, width)`` output size. Defaults to
                the input spatial size.
            retain_graph: Forwarded to ``backward``.
            create_graph: Forwarded to ``backward``.
            return_logits: If True, return ``(cam, logits)``.

        Returns:
            A tensor of shape ``[B, H, W]`` with values in ``[0, 1]``.
        """
        if input_tensor.dim() != 4:
            raise ValueError("input_tensor must have shape [B, C, H, W].")

        was_training = self.final_net.training
        self.final_net.eval()
        try:
            self._activations = {}
            self._gradients = {}

            input_tensor = input_tensor.to(self.device)
            model_input = input_tensor.detach().clone()
            model_input.requires_grad_(True)

            self.final_net.zero_grad()
            logits = self.final_net(model_input)
            scores = self._select_scores(logits, target_class)
            objective = scores.sum()
            objective.backward(retain_graph=retain_graph, create_graph=create_graph)

            output_size = resize_to or tuple(model_input.shape[-2:])
            cams = []
            for layer_index in range(len(self.target_layers)):
                if layer_index not in self._activations or layer_index not in self._gradients:
                    raise RuntimeError(
                        "Missing activation or gradient for target layer index %d. "
                        "Make sure the selected layer participates in the forward pass."
                        % layer_index
                    )

                activation = self._activations[layer_index]
                gradient = self._gradients[layer_index]
                layer_cam = self._layer_cam(activation, gradient)
                layer_cam = F.interpolate(layer_cam, size=output_size, mode="bilinear", align_corners=False)
                cams.append(self._normalize(layer_cam))

            fused_cam = self._fuse_cams(cams)

            if self.use_input_gradient_refine and self.refine_strength > 0:
                fused_cam = self._refine_with_input_gradient(fused_cam, model_input.grad)

            fused_cam = self._normalize(fused_cam).squeeze(1)

            if return_logits:
                return fused_cam.detach(), logits.detach()
            return fused_cam.detach()
        finally:
            if was_training:
                self.final_net.train()

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    def remove_hooks(self):
        """Remove registered hooks when the explainer is no longer needed."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def close(self):
        self.remove_hooks()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.remove_hooks()

    def _register_hooks(self):
        for layer_index, layer in enumerate(self.target_layers):
            self._handles.append(layer.register_forward_hook(self._make_forward_hook(layer_index)))
            self._handles.append(layer.register_backward_hook(self._make_backward_hook(layer_index)))

    def _make_forward_hook(self, layer_index):
        def hook(module, inputs, output):
            self._activations[layer_index] = output

        return hook

    def _make_backward_hook(self, layer_index):
        def hook(module, grad_input, grad_output):
            self._gradients[layer_index] = grad_output[0]

        return hook

    def _layer_cam(self, activation, gradient):
        positive_gradient = F.relu(gradient)
        cam = (activation * positive_gradient).sum(dim=1, keepdim=True)
        return F.relu(cam)

    def _fuse_cams(self, cams):
        fused = None
        for cam, weight in zip(cams, self.layer_weights):
            weighted_cam = cam * weight
            fused = weighted_cam if fused is None else fused + weighted_cam
        return fused

    def _refine_with_input_gradient(self, cam, input_gradient):
        if input_gradient is None:
            return cam

        saliency = input_gradient.detach().abs().mean(dim=1, keepdim=True)
        saliency = self._normalize(saliency)
        return cam * (1.0 + self.refine_strength * saliency)

    def _select_scores(self, logits, target_class):
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if logits.dim() == 1:
            logits = logits.view(-1, 1)
        if logits.dim() > 2:
            logits = logits.view(logits.size(0), -1)

        batch_size = logits.size(0)
        num_outputs = logits.size(1)

        if target_class is None:
            if num_outputs == 1:
                class_indices = (logits[:, 0] >= 0).long()
            else:
                class_indices = logits.argmax(dim=1)
        elif isinstance(target_class, int):
            class_indices = torch.LongTensor([target_class] * batch_size).to(logits.device)
        elif torch.is_tensor(target_class):
            class_indices = target_class.long().to(logits.device)
            if class_indices.dim() == 0:
                class_indices = class_indices.view(1).repeat(batch_size)
        else:
            class_indices = torch.LongTensor(list(target_class)).to(logits.device)

        if class_indices.numel() != batch_size:
            raise ValueError("target_class must be a scalar or contain one class per input sample.")

        if num_outputs == 1:
            scalar_output = logits[:, 0]
            positive_mask = class_indices > 0
            return torch.where(positive_mask, scalar_output, -scalar_output)

        if class_indices.min().item() < 0 or class_indices.max().item() >= num_outputs:
            raise ValueError("target_class is outside the model output dimension.")
        return logits.gather(1, class_indices.view(-1, 1)).squeeze(1)

    def _normalize(self, tensor):
        flat = tensor.contiguous().view(tensor.size(0), -1)
        min_value = flat.min(dim=1)[0].view(tensor.size(0), 1, 1, 1)
        max_value = flat.max(dim=1)[0].view(tensor.size(0), 1, 1, 1)
        return (tensor - min_value) / (max_value - min_value + self.eps)

    def _resolve_target_layers(self, final_net, target_layers):
        if target_layers is None:
            layers = self._default_resnet_layers(final_net)
        else:
            named_modules = dict(final_net.named_modules())
            layers = []
            for layer in target_layers:
                if isinstance(layer, str):
                    if layer not in named_modules:
                        raise ValueError("Cannot find target layer named '%s'." % layer)
                    layers.append(named_modules[layer])
                else:
                    layers.append(layer)

        if not layers:
            raise ValueError(
                "No target layers were provided or inferred. Pass layers such as "
                "[final_net.layer2[-1], final_net.layer3[-1], final_net.layer4[-1]]."
            )
        return layers

    def _default_resnet_layers(self, final_net):
        layers = []
        for layer_name in ("layer2", "layer3", "layer4"):
            if hasattr(final_net, layer_name):
                layer = getattr(final_net, layer_name)
                try:
                    layers.append(layer[-1])
                except TypeError:
                    layers.append(layer)
        return layers

    def _prepare_layer_weights(self, layer_weights, num_layers):
        if layer_weights is None:
            return [1.0 / float(num_layers)] * num_layers

        if len(layer_weights) != num_layers:
            raise ValueError("layer_weights must have the same length as target_layers.")

        total = float(sum(layer_weights))
        if total <= 0:
            raise ValueError("layer_weights must sum to a positive value.")
        return [float(weight) / total for weight in layer_weights]

    def _infer_device(self, final_net):
        try:
            return next(final_net.parameters()).device
        except StopIteration:
            return torch.device("cpu")
