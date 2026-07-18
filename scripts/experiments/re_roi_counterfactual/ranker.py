"""Minimal DeepSets ranker for the re-ROI counterfactual evidence protocol.

All arms share the same architecture.  The only controlled difference is the
input feature set:

- Arm A (family_prior): zero-residual predictor (equivalent to predicting mu_f).
- Arm B (static_roi): residual target using only pre-action ROI features.
- Arm C (re_roi): residual target using h_pre + h_post.
- Arm D (re_roi_bundle_shuffle): same as C but h_post is shuffled per image.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from scripts.experiments.re_roi_counterfactual.action_family import get_action_family
from scripts.experiments.re_roi_counterfactual.teacher import fit_q_teacher_stats, standardize_q_teacher


_ARM_NAMES = ("A", "B", "C", "D")


def _family_index_map() -> dict[str, int]:
    return {spec.family: index for index, spec in enumerate(get_action_family())}


def _fit_q_teacher_stats_from_records(records: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    values = [
        float(row["q_teacher"])
        for record in records
        for row in record["actions"]
        if row["family"] != "identity_permutation"
    ]
    return fit_q_teacher_stats(torch.tensor(values))


def fit_family_prior(
    records: list[dict[str, Any]],
    q_stats: dict[str, torch.Tensor] | None = None,
    identity_family: str = "identity_permutation",
) -> dict[str, float]:
    """Compute mu_f on fit records using standardized Q_teacher.

    Identity family is forced to zero and is excluded from fitting.
    """
    if q_stats is None:
        q_stats = _fit_q_teacher_stats_from_records(records)
    family_values: dict[str, list[float]] = {}
    for record in records:
        for row in record["actions"]:
            family = row["family"]
            if family == identity_family:
                continue
            q_std = float(standardize_q_teacher(torch.tensor(row["q_teacher"]), q_stats).item())
            family_values.setdefault(family, []).append(q_std)
    prior: dict[str, float] = {identity_family: 0.0}
    for family, values in family_values.items():
        prior[family] = float(sum(values) / len(values)) if values else 0.0
    return prior


def _max_label_in_records(records: list[dict[str, Any]]) -> int:
    max_label = 0
    for record in records:
        labels = record["baseline"]["labels"]
        if labels.numel():
            max_label = max(max_label, int(labels.max().item()))
        for row in record["actions"]:
            max_label = max(max_label, int(row.get("label", 0)))
    return max_label


def _shuffle_h_post_for_record(
    record: dict[str, Any],
    seed: int,
) -> dict[int, dict[str, torch.Tensor]]:
    """Return a map candidate_index -> family -> shuffled h_post tensor."""
    rows_by_candidate: dict[int, list[dict[str, Any]]] = {}
    for row in record["actions"]:
        rows_by_candidate.setdefault(int(row["candidate_index"]), []).append(row)

    shuffled: dict[int, dict[str, torch.Tensor]] = {}
    for candidate_index, rows in rows_by_candidate.items():
        families = [row["family"] for row in rows]
        posts = [row["h_post"] for row in rows]
        generator = torch.Generator().manual_seed(seed + candidate_index)
        offset = int(torch.randint(1, len(posts), (1,), generator=generator).item())
        perm = [(index + offset) % len(posts) for index in range(len(posts))]
        shuffled[candidate_index] = {families[i]: posts[perm[i]] for i in range(len(rows))}
    return shuffled


@dataclass(frozen=True)
class RankerRow:
    """One training/example row for the ranker."""

    image_id: int
    family: str
    family_idx: int
    candidate_index: int
    q_teacher_raw: float
    q_teacher_std: float
    residual: float
    h_pre: torch.Tensor
    h_post: torch.Tensor
    baseline_roi_features: torch.Tensor
    baseline_boxes: torch.Tensor
    baseline_scores: torch.Tensor
    baseline_labels: torch.Tensor
    acted_mask: torch.Tensor
    image_size: torch.Tensor
    mask: torch.Tensor


class ReROIRankerDataset(Dataset):
    """Torch dataset over re-ROI cache records for one arm."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        arm: str,
        prior: dict[str, float] | None = None,
        q_stats: dict[str, torch.Tensor] | None = None,
        max_label: int | None = None,
        shuffle_seed: int = 42,
        identity_family: str = "identity_permutation",
    ):
        if arm not in _ARM_NAMES:
            raise ValueError(f"arm must be one of {_ARM_NAMES}, got {arm}")
        self.arm = arm
        self.family_to_index = _family_index_map()
        self.identity_idx = self.family_to_index[identity_family]

        if q_stats is None and records:
            q_stats = _fit_q_teacher_stats_from_records(records)
        self.q_stats = q_stats

        if prior is None and records:
            prior = fit_family_prior(records, q_stats=q_stats, identity_family=identity_family)
        self.prior = prior or {}

        if max_label is None and records:
            max_label = _max_label_in_records(records)
        self.max_label = max(max_label or 0, 1)

        self.rows: list[RankerRow] = []
        for record in records:
            self.rows.extend(self._build_rows(record, shuffle_seed, identity_family))

    def _build_rows(
        self,
        record: dict[str, Any],
        shuffle_seed: int,
        identity_family: str,
    ) -> list[RankerRow]:
        baseline = record["baseline"]
        roi_features = baseline["roi_features"]
        boxes = baseline["boxes"]
        scores = baseline["scores"]
        labels = baseline["labels"]
        image_size = torch.tensor(record["image_size"], dtype=torch.float32)
        num_detections = int(roi_features.shape[0])
        mask = torch.ones(num_detections, dtype=torch.bool)

        shuffled_posts: dict[int, dict[str, torch.Tensor]] | None = None
        if self.arm == "D":
            shuffled_posts = _shuffle_h_post_for_record(record, int(record["image_id"]) + shuffle_seed)

        rows: list[RankerRow] = []
        for row in record["actions"]:
            family = row["family"]
            if family == identity_family:
                # Identity is the exact no-op; standardization artifacts are forced to zero.
                q_std = 0.0
                residual = 0.0
            else:
                q_std = float(
                    standardize_q_teacher(torch.tensor(float(row["q_teacher"])), self.q_stats).item()
                )
                mu_f = self.prior.get(family, 0.0)
                residual = q_std - mu_f

            candidate_index = int(row["candidate_index"])
            acted_mask = torch.zeros(num_detections, dtype=torch.bool)
            if candidate_index < num_detections:
                acted_mask[candidate_index] = True

            if self.arm in ("A", "B"):
                h_post = torch.zeros_like(row["h_pre"])
            elif self.arm == "D" and shuffled_posts is not None:
                h_post = shuffled_posts[candidate_index][family]
            else:
                h_post = row["h_post"]

            rows.append(
                RankerRow(
                    image_id=int(record["image_id"]),
                    family=family,
                    family_idx=self.family_to_index[family],
                    candidate_index=candidate_index,
                    q_teacher_raw=float(row["q_teacher"]),
                    q_teacher_std=q_std,
                    residual=residual,
                    h_pre=row["h_pre"].detach().cpu().float(),
                    h_post=h_post.detach().cpu().float(),
                    baseline_roi_features=roi_features.detach().cpu().float(),
                    baseline_boxes=boxes.detach().cpu().float(),
                    baseline_scores=scores.detach().cpu().float(),
                    baseline_labels=labels.detach().cpu().long(),
                    acted_mask=acted_mask,
                    image_size=image_size,
                    mask=mask,
                )
            )
        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return {
            "image_id": row.image_id,
            "family": row.family,
            "family_idx": row.family_idx,
            "candidate_index": row.candidate_index,
            "q_teacher_raw": row.q_teacher_raw,
            "q_teacher_std": row.q_teacher_std,
            "residual": row.residual,
            "h_pre": row.h_pre,
            "h_post": row.h_post,
            "baseline_roi_features": row.baseline_roi_features,
            "baseline_boxes": row.baseline_boxes,
            "baseline_scores": row.baseline_scores,
            "baseline_labels": row.baseline_labels,
            "acted_mask": row.acted_mask,
            "image_size": row.image_size,
            "mask": row.mask,
        }

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        max_n = max(int(item["baseline_roi_features"].shape[0]) for item in batch)
        feature_dim = int(batch[0]["h_pre"].shape[0])

        def _pad_tensor(tensor: torch.Tensor, target_len: int, pad_value: float) -> torch.Tensor:
            pad_size = target_len - tensor.shape[0]
            if pad_size <= 0:
                return tensor
            if tensor.ndim == 1:
                return F.pad(tensor, (0, pad_size), value=pad_value)
            if tensor.ndim == 2:
                return F.pad(tensor, (0, 0, 0, pad_size), value=pad_value)
            raise ValueError(f"unsupported per-detection tensor rank: {tensor.ndim}")

        fields = {
            "baseline_roi_features": (feature_dim, 0.0),
            "baseline_boxes": (4, 0.0),
            "baseline_scores": (1, 0.0),
            "baseline_labels": (1, 0),
        }
        padded: dict[str, list[torch.Tensor]] = {key: [] for key in fields}
        masks: list[torch.Tensor] = []
        acted_masks: list[torch.Tensor] = []

        for item in batch:
            n = int(item["baseline_roi_features"].shape[0])
            masks.append(F.pad(torch.ones(n, dtype=torch.bool), (0, max_n - n), value=False))
            acted_masks.append(_pad_tensor(item["acted_mask"].float(), max_n, 0.0).bool())
            for key, (feat_dim, pad_value) in fields.items():
                padded[key].append(_pad_tensor(item[key], max_n, pad_value))

        return {
            "image_id": torch.tensor([item["image_id"] for item in batch], dtype=torch.long),
            "family_idx": torch.tensor([item["family_idx"] for item in batch], dtype=torch.long),
            "candidate_index": torch.tensor([item["candidate_index"] for item in batch], dtype=torch.long),
            "q_teacher_raw": torch.tensor(
                [item["q_teacher_raw"] for item in batch], dtype=torch.float32
            ),
            "q_teacher_std": torch.tensor([item["q_teacher_std"] for item in batch], dtype=torch.float32),
            "residual": torch.tensor([item["residual"] for item in batch], dtype=torch.float32),
            "h_pre": torch.stack([item["h_pre"] for item in batch]),
            "h_post": torch.stack([item["h_post"] for item in batch]),
            "baseline_roi_features": torch.stack(padded["baseline_roi_features"]),
            "baseline_boxes": torch.stack(padded["baseline_boxes"]),
            "baseline_scores": torch.stack(padded["baseline_scores"]),
            "baseline_labels": torch.stack(padded["baseline_labels"]),
            "mask": torch.stack(masks),
            "acted_mask": torch.stack(acted_masks),
            "image_size": torch.stack([item["image_size"] for item in batch]),
        }


class ZeroResidualModel(nn.Module):
    """Arm A: always predicts zero residual."""

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.zeros(batch["residual"].shape[0], dtype=torch.float32, device=batch["residual"].device)


class DeepSetResidualRanker(nn.Module):
    """Minimal permutation-equivariant residual ranker."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        label_embed_dim: int = 8,
        num_families: int = 10,
        max_label: int = 128,
        use_post: bool = True,
        identity_idx: int = 0,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.use_post = use_post
        self.register_buffer("identity_idx", torch.tensor(identity_idx, dtype=torch.long))
        self.label_embed = nn.Embedding(max_label + 1, label_embed_dim)

        node_in_dim = feature_dim + 4 + 1 + label_embed_dim + 1
        self.u_net = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        acted_in_dim = feature_dim + (feature_dim if use_post else 0)
        self.acted_net = nn.Sequential(
            nn.Linear(acted_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.v_net = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.family_embed = nn.Embedding(num_families, label_embed_dim)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3 + label_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        roi_features = batch["baseline_roi_features"]
        boxes = batch["baseline_boxes"]
        scores = batch["baseline_scores"]
        labels = batch["baseline_labels"]
        mask = batch["mask"]
        acted_mask = batch["acted_mask"]
        image_size = batch["image_size"]
        h_pre = batch["h_pre"]
        h_post = batch["h_post"]
        family_idx = batch["family_idx"]

        # image_size is [height, width], while boxes are [x1, y1, x2, y2].
        box_scale = torch.stack(
            [image_size[:, 1], image_size[:, 0], image_size[:, 1], image_size[:, 0]],
            dim=-1,
        )[:, None, :]
        boxes_norm = boxes / (box_scale + 1e-6)
        label_emb = self.label_embed(labels.clamp(0, self.label_embed.num_embeddings - 1))

        node_input = torch.cat(
            [
                roi_features,
                boxes_norm,
                scores.unsqueeze(-1),
                label_emb,
                acted_mask.unsqueeze(-1).float(),
            ],
            dim=-1,
        )
        node_out = self.u_net(node_input)
        # Masked mean pool over the detection set.
        node_out_sum = (node_out * mask.unsqueeze(-1).float()).sum(dim=1)
        pooled = node_out_sum / (mask.sum(dim=1, keepdim=True).clamp_min(1.0))

        widths = (boxes_norm[..., 2] - boxes_norm[..., 0]).clamp_min(1e-6)
        heights = (boxes_norm[..., 3] - boxes_norm[..., 1]).clamp_min(1e-6)
        geometry = torch.stack(
            [
                (boxes_norm[..., 0] + boxes_norm[..., 2]) * 0.5,
                (boxes_norm[..., 1] + boxes_norm[..., 3]) * 0.5,
                widths.log(),
                heights.log(),
            ],
            dim=-1,
        )
        acted_weights = acted_mask.unsqueeze(-1).float()
        acted_geometry = (geometry * acted_weights).sum(dim=1, keepdim=True)
        acted_boxes = (boxes_norm * acted_weights).sum(dim=1, keepdim=True)
        acted_scores = (scores * acted_mask.float()).sum(dim=1, keepdim=True)
        acted_labels = (labels * acted_mask.long()).sum(dim=1, keepdim=True)

        intersection_lt = torch.maximum(boxes_norm[..., :2], acted_boxes[..., :2])
        intersection_rb = torch.minimum(boxes_norm[..., 2:], acted_boxes[..., 2:])
        intersection_wh = (intersection_rb - intersection_lt).clamp_min(0.0)
        intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
        node_area = widths * heights
        acted_area = (
            (acted_boxes[..., 2] - acted_boxes[..., 0]).clamp_min(1e-6)
            * (acted_boxes[..., 3] - acted_boxes[..., 1]).clamp_min(1e-6)
        )
        iou = intersection / (node_area + acted_area - intersection + 1e-6)
        edge_input = torch.cat(
            [
                geometry - acted_geometry,
                iou.unsqueeze(-1),
                (scores - acted_scores).unsqueeze(-1),
                (labels == acted_labels).float().unsqueeze(-1),
            ],
            dim=-1,
        )
        edge_out = self.v_net(edge_input)
        edge_sum = (edge_out * mask.unsqueeze(-1).float()).sum(dim=1)
        pair_pooled = edge_sum / mask.sum(dim=1, keepdim=True).clamp_min(1.0)

        acted_input = torch.cat([h_pre, h_post], dim=-1) if self.use_post else h_pre
        acted_out = self.acted_net(acted_input)

        family_vec = self.family_embed(family_idx)
        combined = torch.cat([pooled, pair_pooled, acted_out, family_vec], dim=-1)
        output = self.head(combined).squeeze(-1)
        non_identity = (family_idx != self.identity_idx).float()
        return output * non_identity


def make_model_for_arm(
    arm: str,
    feature_dim: int,
    max_label: int,
    num_families: int = 10,
    identity_idx: int = 0,
    hidden_dim: int = 64,
) -> nn.Module:
    """Factory matching the controlled arm input set."""
    if arm == "A":
        return ZeroResidualModel()
    if arm == "B":
        return DeepSetResidualRanker(
            feature_dim,
            hidden_dim=hidden_dim,
            use_post=False,
            max_label=max_label,
            num_families=num_families,
            identity_idx=identity_idx,
        )
    if arm in ("C", "D"):
        return DeepSetResidualRanker(
            feature_dim,
            hidden_dim=hidden_dim,
            use_post=True,
            max_label=max_label,
            num_families=num_families,
            identity_idx=identity_idx,
        )
    raise ValueError(f"unknown arm {arm}")


def ranker_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    family_idx: torch.Tensor,
    image_id: torch.Tensor,
    identity_idx: int,
    lambda_rank: float = 0.1,
    tie_eps: float = 1e-6,
) -> torch.Tensor:
    """Smooth L1 on residuals plus a same-image same-family ranking term."""
    non_identity_rows = family_idx != identity_idx
    if non_identity_rows.any():
        regression = F.smooth_l1_loss(
            pred[non_identity_rows], target[non_identity_rows], beta=0.05
        )
    else:
        regression = pred.sum() * 0.0

    if lambda_rank <= 0.0:
        return regression

    n = pred.shape[0]
    idx = torch.arange(n, device=pred.device)
    i = idx.view(-1, 1)
    j = idx.view(1, -1)
    upper = i < j
    same_family = family_idx.unsqueeze(1) == family_idx.unsqueeze(0)
    same_image = image_id.unsqueeze(1) == image_id.unsqueeze(0)
    non_identity = (family_idx.unsqueeze(1) != identity_idx) & (family_idx.unsqueeze(0) != identity_idx)
    diff_target = target.unsqueeze(1) - target.unsqueeze(0)
    valid = (
        upper
        & same_family
        & same_image
        & non_identity
        & (diff_target.abs() > tie_eps)
    )

    if not valid.any():
        return regression

    sign = torch.sign(diff_target)
    pair_losses = F.softplus(-sign * (pred.unsqueeze(1) - pred.unsqueeze(0)))
    rank_term = pair_losses[valid].mean()
    return regression + lambda_rank * rank_term


def _batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def build_grouped_batches(
    dataset: ReROIRankerDataset,
    *,
    batch_size: int,
    seed: int,
    epoch: int,
) -> list[list[int]]:
    """Build deterministic batches without splitting image-family rank groups."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    groups: dict[tuple[int, int], list[int]] = {}
    for index, row in enumerate(dataset.rows):
        if row.family_idx == dataset.identity_idx:
            continue
        groups.setdefault((row.image_id, row.family_idx), []).append(index)
    if any(len(indices) > batch_size for indices in groups.values()):
        raise ValueError("batch_size is smaller than an image-family rank group")

    keys = sorted(groups)
    generator = torch.Generator().manual_seed(int(seed) * 1_000_003 + int(epoch))
    order = torch.randperm(len(keys), generator=generator).tolist()
    batches: list[list[int]] = []
    current: list[int] = []
    for offset in order:
        group = groups[keys[offset]]
        if current and len(current) + len(group) > batch_size:
            batches.append(current)
            current = []
        current.extend(group)
    if current:
        batches.append(current)
    return batches


def train_ranker(
    model: nn.Module,
    dataset: ReROIRankerDataset,
    *,
    epochs: int = 20,
    lr: float = 1e-3,
    lambda_rank: float = 0.1,
    batch_size: int = 32,
    device: torch.device | None = None,
    seed: int = 42,
) -> nn.Module:
    """Train a ranker in-place and return it."""
    torch.manual_seed(seed)
    if device is None:
        device = torch.device("cpu")
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        loader = DataLoader(
            dataset,
            batch_sampler=build_grouped_batches(
                dataset, batch_size=batch_size, seed=seed, epoch=epoch
            ),
            collate_fn=ReROIRankerDataset.collate_fn,
        )
        for batch in loader:
            batch = _batch_to_device(batch, device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = ranker_loss(
                pred,
                batch["residual"],
                batch["family_idx"],
                batch["image_id"],
                dataset.identity_idx,
                lambda_rank=lambda_rank,
            )
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def _gather_predictions(
    model: nn.Module,
    dataset: ReROIRankerDataset,
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=ReROIRankerDataset.collate_fn,
    )
    all_pred: list[torch.Tensor] = []
    all_target: list[torch.Tensor] = []
    all_family: list[torch.Tensor] = []
    all_image: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            batch = _batch_to_device(batch, device)
            pred = model(batch)
            all_pred.append(pred.cpu())
            all_target.append(batch["residual"].cpu())
            all_family.append(batch["family_idx"].cpu())
            all_image.append(batch["image_id"].cpu())
    return {
        "pred": torch.cat(all_pred),
        "target": torch.cat(all_target),
        "family_idx": torch.cat(all_family),
        "image_id": torch.cat(all_image),
    }


def _pairwise_accuracy_per_image(
    pred: torch.Tensor,
    target: torch.Tensor,
    family_idx: torch.Tensor,
    image_id: torch.Tensor,
    identity_idx: int,
    tie_eps: float = 1e-6,
    require_same_family: bool = True,
) -> dict[int, dict[int, float]]:
    """Return {image_id: {family_idx: accuracy}} for same-image same-family pairs."""
    n = pred.shape[0]
    idx = torch.arange(n, device=pred.device)
    i = idx.view(-1, 1)
    j = idx.view(1, -1)
    upper = i < j
    same_family = family_idx.unsqueeze(1) == family_idx.unsqueeze(0)
    family_compatible = same_family if require_same_family else torch.ones_like(same_family)
    same_image = image_id.unsqueeze(1) == image_id.unsqueeze(0)
    non_identity = (family_idx.unsqueeze(1) != identity_idx) & (family_idx.unsqueeze(0) != identity_idx)
    diff_target = target.unsqueeze(1) - target.unsqueeze(0)
    valid = (
        upper
        & family_compatible
        & same_image
        & non_identity
        & (diff_target.abs() > tie_eps)
    )

    if not valid.any():
        return {}

    i_idx, j_idx = valid.nonzero(as_tuple=True)
    correct = ((pred[i_idx] - pred[j_idx]) * (target[i_idx] - target[j_idx])) > 0
    pair_image = image_id[i_idx]
    pair_family = family_idx[i_idx] if require_same_family else torch.zeros_like(pair_image)
    pair_correct = correct.float()

    per_image: dict[int, dict[int, float]] = {}
    for img in torch.unique(pair_image):
        img_mask = pair_image == img
        fams = pair_family[img_mask]
        corr = pair_correct[img_mask]
        per_image[int(img.item())] = {}
        for fam in torch.unique(fams):
            fam_mask = fams == fam
            per_image[int(img.item())][int(fam.item())] = float(corr[fam_mask].mean().item())
    return per_image


def _mean_pairwise_accuracy(per_image: dict[int, dict[int, float]]) -> float:
    values = [acc for fams in per_image.values() for acc in fams.values()]
    return float(sum(values) / len(values)) if values else 0.0


def _sign_auroc(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mann-Whitney U estimate of AUROC for residual sign."""
    pos_mask = target > 0
    neg_mask = target < 0
    pos_scores = pred[pos_mask]
    neg_scores = pred[neg_mask]
    if pos_scores.numel() == 0 or neg_scores.numel() == 0:
        return 0.5
    # Compute P(pred_pos > pred_neg) + 0.5 P(pred_pos == pred_neg).
    pairwise = (pos_scores[:, None] > neg_scores[None, :]).float()
    ties = (pos_scores[:, None] == neg_scores[None, :]).float()
    return float((pairwise.sum() + 0.5 * ties.sum()).item() / (pos_scores.numel() * neg_scores.numel()))


@dataclass(frozen=True)
class RankerMetrics:
    mae: float
    relative_mae_gain: float
    pairwise_accuracy: float
    sign_auroc: float
    per_image_pairwise: dict[int, dict[int, float]]
    num_rows: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "mae": self.mae,
            "relative_mae_gain": self.relative_mae_gain,
            "pairwise_accuracy": self.pairwise_accuracy,
            "sign_auroc": self.sign_auroc,
            "num_rows": self.num_rows,
        }


def _metrics_from_predictions(
    pred: torch.Tensor,
    target: torch.Tensor,
    family_idx: torch.Tensor,
    image_id: torch.Tensor,
    identity_idx: int,
    *,
    require_same_family: bool,
) -> RankerMetrics:
    mae = float(torch.abs(pred - target).mean().item())
    zero_mae = float(torch.abs(target).mean().item())
    relative_mae_gain = (zero_mae - mae) / (zero_mae + 1e-12)
    per_image = _pairwise_accuracy_per_image(
        pred,
        target,
        family_idx,
        image_id,
        identity_idx,
        require_same_family=require_same_family,
    )
    return RankerMetrics(
        mae=mae,
        relative_mae_gain=relative_mae_gain,
        pairwise_accuracy=_mean_pairwise_accuracy(per_image),
        sign_auroc=_sign_auroc(pred, target),
        per_image_pairwise=per_image,
        num_rows=int(pred.shape[0]),
    )


def _non_identity_predictions(
    model: nn.Module,
    dataset: ReROIRankerDataset,
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    out = _gather_predictions(model, dataset, batch_size, device)
    mask = out["family_idx"] != dataset.identity_idx
    return {key: value[mask] for key, value in out.items()}


def _prior_for_family_indices(
    dataset: ReROIRankerDataset, family_idx: torch.Tensor
) -> torch.Tensor:
    prior_by_index = torch.zeros(len(dataset.family_to_index), dtype=torch.float32)
    for family, index in dataset.family_to_index.items():
        prior_by_index[index] = float(dataset.prior.get(family, 0.0))
    return prior_by_index[family_idx]


def evaluate_ranker(
    model: nn.Module,
    dataset: ReROIRankerDataset,
    *,
    batch_size: int = 32,
    device: torch.device | None = None,
) -> RankerMetrics:
    """Evaluate a trained ranker on a dataset."""
    if device is None:
        device = torch.device("cpu")
    model.to(device)
    model.eval()
    out = _non_identity_predictions(model, dataset, batch_size, device)
    return _metrics_from_predictions(
        out["pred"],
        out["target"],
        out["family_idx"],
        out["image_id"],
        dataset.identity_idx,
        require_same_family=True,
    )


def evaluate_reconstructed_ranker(
    model: nn.Module,
    dataset: ReROIRankerDataset,
    *,
    batch_size: int = 32,
    device: torch.device | None = None,
) -> RankerMetrics:
    """Evaluate reconstructed standardized total utility across action families."""
    if device is None:
        device = torch.device("cpu")
    model.to(device)
    model.eval()
    out = _non_identity_predictions(model, dataset, batch_size, device)
    prior = _prior_for_family_indices(dataset, out["family_idx"])
    total_pred = out["pred"] + prior
    total_target = out["target"] + prior
    baseline_error = out["target"]
    metrics = _metrics_from_predictions(
        total_pred,
        total_target,
        out["family_idx"],
        out["image_id"],
        dataset.identity_idx,
        require_same_family=False,
    )
    baseline_mae = float(torch.abs(baseline_error).mean().item())
    relative_gain = (baseline_mae - metrics.mae) / (baseline_mae + 1e-12)
    return RankerMetrics(
        mae=metrics.mae,
        relative_mae_gain=relative_gain,
        pairwise_accuracy=metrics.pairwise_accuracy,
        sign_auroc=metrics.sign_auroc,
        per_image_pairwise=metrics.per_image_pairwise,
        num_rows=metrics.num_rows,
    )


def evaluate_utility_shuffle_metrics(
    dataset: ReROIRankerDataset, *, seed: int = 42
) -> dict[str, RankerMetrics]:
    """Evaluate a deterministic within-family utility-permutation null."""
    rows = [row for row in dataset.rows if row.family_idx != dataset.identity_idx]
    target = torch.tensor([row.residual for row in rows], dtype=torch.float32)
    family_idx = torch.tensor([row.family_idx for row in rows], dtype=torch.long)
    image_id = torch.tensor([row.image_id for row in rows], dtype=torch.long)
    pred = torch.empty_like(target)
    generator = torch.Generator().manual_seed(seed)
    for family in torch.unique(family_idx):
        indices = (family_idx == family).nonzero(as_tuple=True)[0]
        permutation = torch.randperm(len(indices), generator=generator)
        pred[indices] = target[indices[permutation]]
    residual = _metrics_from_predictions(
        pred,
        target,
        family_idx,
        image_id,
        dataset.identity_idx,
        require_same_family=True,
    )
    prior = _prior_for_family_indices(dataset, family_idx)
    reconstructed = _metrics_from_predictions(
        pred + prior,
        target + prior,
        family_idx,
        image_id,
        dataset.identity_idx,
        require_same_family=False,
    )
    return {"residual": residual, "reconstructed_total": reconstructed}


def evaluate_utility_shuffle(
    dataset: ReROIRankerDataset, *, seed: int = 42
) -> dict[str, dict[str, Any]]:
    return {
        name: metrics.to_dict()
        for name, metrics in evaluate_utility_shuffle_metrics(
            dataset, seed=seed
        ).items()
    }


def action_selection_controls(
    dataset: ReROIRankerDataset, *, seed: int = 42
) -> dict[str, dict[str, float | int]]:
    """Return oracle-rate random and no-op action-selection diagnostics."""
    rows_by_image: dict[int, list[RankerRow]] = {}
    for row in dataset.rows:
        if row.family_idx != dataset.identity_idx:
            rows_by_image.setdefault(row.image_id, []).append(row)
    image_ids = sorted(rows_by_image)
    oracle_values = {
        image_id: max(row.q_teacher_raw for row in rows_by_image[image_id])
        for image_id in image_ids
    }
    oracle_action_ids = [image_id for image_id in image_ids if oracle_values[image_id] > 0.0]
    generator = torch.Generator().manual_seed(seed)
    image_order = torch.randperm(len(image_ids), generator=generator).tolist()
    random_action_ids = {
        image_ids[index] for index in image_order[: len(oracle_action_ids)]
    }
    random_values: dict[int, float] = {}
    for image_id in sorted(random_action_ids):
        choices = rows_by_image[image_id]
        choice = int(torch.randint(0, len(choices), (1,), generator=generator).item())
        random_values[image_id] = choices[choice].q_teacher_raw

    def summarize(selected: dict[int, float]) -> dict[str, float | int]:
        values = list(selected.values())
        total_images = len(image_ids)
        return {
            "selected_count": len(values),
            "action_image_rate": len(values) / total_images if total_images else 0.0,
            "positive_precision": sum(value > 0.0 for value in values) / len(values) if values else 0.0,
            "selected_mean_raw_utility": sum(values) / len(values) if values else 0.0,
            "mean_raw_utility": sum(values) / total_images if total_images else 0.0,
        }

    return {
        "oracle_top1": summarize(
            {image_id: oracle_values[image_id] for image_id in oracle_action_ids}
        ),
        "matched_rate_random": summarize(random_values),
        "always_noop": summarize({}),
    }


def paired_bootstrap_lcb(
    per_image_a: dict[int, dict[int, float]],
    per_image_b: dict[int, dict[int, float]],
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> float:
    """Bootstrap LCB of mean(A - B) over images with paired family averages."""
    return float(
        paired_bootstrap_interval(
            per_image_a, per_image_b, n_bootstrap=n_bootstrap, seed=seed
        )["lower"]
    )


def paired_bootstrap_interval(
    per_image_a: dict[int, dict[int, float]],
    per_image_b: dict[int, dict[int, float]],
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired image-bootstrap interval on common image-family support."""
    common_images = sorted(set(per_image_a.keys()) & set(per_image_b.keys()))
    if not common_images:
        return {
            "point_delta": float("nan"),
            "lower": float("-inf"),
            "upper": float("inf"),
            "common_image_count": 0,
            "common_image_ids": [],
        }

    diffs: list[float] = []
    supporting_images: list[int] = []
    for img in common_images:
        families = set(per_image_a[img].keys()) & set(per_image_b[img].keys())
        if not families:
            continue
        a_mean = sum(per_image_a[img][f] for f in families) / len(families)
        b_mean = sum(per_image_b[img][f] for f in families) / len(families)
        diffs.append(a_mean - b_mean)
        supporting_images.append(img)

    if not diffs:
        return {
            "point_delta": float("nan"),
            "lower": float("-inf"),
            "upper": float("inf"),
            "common_image_count": 0,
            "common_image_ids": [],
        }

    values = torch.tensor(diffs, dtype=torch.float32)
    n = values.shape[0]
    generator = torch.Generator().manual_seed(seed)
    resampled_means: list[float] = []
    for _ in range(n_bootstrap):
        idx = torch.randint(0, n, (n,), generator=generator)
        resampled_means.append(float(values[idx].mean().item()))
    bootstrap = torch.tensor(resampled_means)
    return {
        "point_delta": float(values.mean().item()),
        "lower": float(torch.quantile(bootstrap, 0.025).item()),
        "upper": float(torch.quantile(bootstrap, 0.975).item()),
        "common_image_count": len(supporting_images),
        "common_image_ids": supporting_images,
    }
