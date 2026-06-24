r"""Retrieval evaluation utilities for image-text alignment.

Provides standard image-to-text and text-to-image recall metrics computed
from a shared embedding space.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _normalise(x: torch.Tensor) -> torch.Tensor:
    """L2-normalise a tensor along the last dimension with a small epsilon."""
    return F.normalize(x, p=2, dim=-1, eps=1e-12)


def compute_similarity_matrix(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
) -> torch.Tensor:
    r"""Compute the cosine similarity matrix between image and text features.

    Args:
        image_features: tensor of shape ``(N, D)``.
        text_features: tensor of shape ``(M, D)``.

    Returns:
        Similarity matrix of shape ``(N, M)`` where ``[i, j]`` is the cosine
        similarity between image ``i`` and text ``j``.
    """
    image_norm = _normalise(image_features)
    text_norm = _normalise(text_features)
    return torch.matmul(image_norm, text_norm.t())


def recall_at_k(
    similarity: torch.Tensor,
    gt_indices: torch.Tensor | None = None,
    k_values: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    r"""Compute Recall@K from an image-to-text similarity matrix.

    The ground-truth index for image ``i`` is assumed to be ``i`` unless
    ``gt_indices`` is provided.  A query is counted as correct if its ground
    truth appears in the top-``k`` retrieved texts.

    Args:
        similarity: tensor of shape ``(N, M)``.
        gt_indices: optional tensor of shape ``(N,)`` giving the ground-truth
            text index for each image. Defaults to ``torch.arange(N)``.
        k_values: tuple of ``k`` values at which to report recall.

    Returns:
        Dictionary mapping ``"R@k"`` to recall values in ``[0, 1]``.
    """
    num_queries = similarity.size(0)
    if gt_indices is None:
        gt_indices = torch.arange(num_queries, device=similarity.device)
    else:
        gt_indices = gt_indices.to(similarity.device)

    results: dict[str, float] = {}
    sorted_indices = torch.argsort(similarity, dim=-1, descending=True)
    for k in k_values:
        top_k = sorted_indices[:, :k]
        hits = (top_k == gt_indices.unsqueeze(1)).any(dim=-1)
        results[f"R@{k}"] = hits.float().mean().item()
    return results


def evaluate_image_text_retrieval(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    gt_indices: torch.Tensor | None = None,
    k_values: tuple[int, ...] = (1, 5, 10),
) -> dict[str, dict[str, float]]:
    r"""Compute image-to-text and text-to-image retrieval recalls.

    Args:
        image_features: tensor of shape ``(N, D)``.
        text_features: tensor of shape ``(N, D)`` (one caption per image).
        gt_indices: optional tensor of shape ``(N,)`` giving the ground-truth
            text index for each image. Defaults to ``torch.arange(N)``.
        k_values: tuple of ``k`` values for Recall@K reporting.

    Returns:
        Nested dictionary with keys ``"image_to_text"`` and
        ``"text_to_image"``, each mapping to a ``recall_at_k`` result dict.
    """
    similarity = compute_similarity_matrix(image_features, text_features)
    image_to_text = recall_at_k(similarity, gt_indices=gt_indices, k_values=k_values)
    text_to_image = recall_at_k(similarity.t(), gt_indices=gt_indices, k_values=k_values)
    return {"image_to_text": image_to_text, "text_to_image": text_to_image}
