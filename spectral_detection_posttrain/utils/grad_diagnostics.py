from __future__ import annotations

from collections.abc import Iterable, Mapping
from math import sqrt

import torch


DEFAULT_GRAD_COMPONENTS = (
    "bbox_adapter",
    "cls_adapter",
    "cls_score",
    "bbox_pred",
    "box_head",
    "rpn",
    "backbone",
    "other",
)


def parameter_component(name: str) -> str:
    if "bbox_adapter" in name:
        return "bbox_adapter"
    if "cls_adapter" in name:
        return "cls_adapter"
    if "box_predictor.base_predictor.cls_score" in name or "box_predictor.cls_score" in name:
        return "cls_score"
    if "box_predictor.base_predictor.bbox_pred" in name or "box_predictor.bbox_pred" in name:
        return "bbox_pred"
    if "roi_heads.box_head" in name:
        return "box_head"
    if name.startswith("rpn.") or ".rpn." in name:
        return "rpn"
    if name.startswith("backbone.") or ".backbone." in name:
        return "backbone"
    return "other"


def _empty_component_squares(components: Iterable[str]) -> dict[str, float]:
    return {component: 0.0 for component in components}


def _l2_metrics_from_squares(prefix: str, squares: Mapping[str, float]) -> dict[str, float]:
    metrics = {f"{prefix}_{component}_l2": sqrt(max(0.0, float(value))) for component, value in squares.items()}
    metrics[f"{prefix}_total_l2"] = sqrt(max(0.0, sum(float(value) for value in squares.values())))
    return metrics


def summarize_current_parameter_gradients(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    *,
    components: Iterable[str] = DEFAULT_GRAD_COMPONENTS,
    prefix: str = "grad_total",
) -> dict[str, float]:
    component_list = tuple(components)
    squares = _empty_component_squares(component_list)
    max_abs = {component: 0.0 for component in component_list}
    elem_count = {component: 0 for component in component_list}
    for name, parameter in named_parameters:
        grad = parameter.grad
        if grad is None:
            continue
        component = parameter_component(name)
        if component not in squares:
            component = "other"
        grad_float = grad.detach().float()
        squares[component] += float(grad_float.pow(2).sum().cpu().item())
        max_abs[component] = max(max_abs[component], float(grad_float.abs().max().cpu().item()))
        elem_count[component] += int(grad.numel())
    metrics = _l2_metrics_from_squares(prefix, squares)
    for component in component_list:
        metrics[f"{prefix}_{component}_max_abs"] = float(max_abs[component])
        metrics[f"{prefix}_{component}_elem_count"] = int(elem_count[component])
    return metrics


def summarize_loss_component_gradients(
    losses: Mapping[str, torch.Tensor],
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    *,
    components: Iterable[str] = DEFAULT_GRAD_COMPONENTS,
    retain_graph: bool = True,
) -> dict[str, float]:
    component_list = tuple(components)
    named_parameter_list = [(name, parameter) for name, parameter in named_parameters if parameter.requires_grad]
    parameters = [parameter for _, parameter in named_parameter_list]
    metrics: dict[str, float] = {}
    if not parameters:
        return metrics

    for loss_name, loss in losses.items():
        squares = _empty_component_squares(component_list)
        if torch.is_tensor(loss) and loss.requires_grad:
            grads = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
            for (name, _), grad in zip(named_parameter_list, grads):
                if grad is None:
                    continue
                component = parameter_component(name)
                if component not in squares:
                    component = "other"
                squares[component] += float(grad.detach().float().pow(2).sum().cpu().item())
        metrics.update(_l2_metrics_from_squares(f"grad_{loss_name}", squares))
    return metrics
