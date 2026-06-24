from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.matching.box_iou import box_iou
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.models.bbox_adapter import (
    freeze_adapters_and_predictor,
    freeze_selected_adapters,
    install_residual_bbox_adapter,
)
from spectral_detection_posttrain.rlvr.action_verifier import (
    ActionVerifierConfig,
    build_action_batch,
    build_dpo_pairs,
    compute_fft_action_quality,
    compute_manifold_action_quality,
    decode_box_actions,
    dpo_loss_from_log_probs,
)
from spectral_detection_posttrain.rlvr.confidence_rescue import (
    BestCheckpointConfig,
    ConfidenceRescueConfig,
    ManifoldGateConfig,
    ManifoldGateReference,
    bbox_localization_rescue_loss,
    build_pairwise_rescue_ranking_loss,
    build_verifier_guided_ranking_loss,
    build_manifold_gate_reference,
    calibrate_classwise_thresholds,
    combine_verifier_scores,
    confidence_rescue_increment_loss,
    confidence_rescue_loss,
    confidence_threshold_crossing_loss,
    evaluate_verifier_offline,
    manifold_soft_rescue_weights,
    match_boxes_to_target_boxes,
    match_boxes_to_targets,
    project_manifold_features,
    select_best_checkpoint_update,
    score_shift_budget_loss,
    score_manifold_gate,
    summarize_confidence_rescue_effect,
    summarize_confidence_iou_regions,
    summarize_verifier_gate,
)
from spectral_detection_posttrain.eval.rescue_oracle import unmatched_gt_candidate_mask
from spectral_detection_posttrain.rlvr.roi_policy_loss import (
    baseline_kl_loss,
    extract_roi_head_outputs_for_boxes,
    resize_boxes_to_image,
)
from spectral_detection_posttrain.analysis.raw_ifft_features import (
    crop_and_resize_boxes,
    penn_fudan_legacy_ifft_metric_bank,
)
from spectral_detection_posttrain.analysis.raw_ifft_verifier import (
    calibrate_precision_threshold,
    fit_train_effect_scorer,
    parse_legacy_ifft_feature_specs,
    score_legacy_ifft_metric_bank,
    score_scene_legacy_ifft_metric_bank,
)
from spectral_detection_posttrain.train.action_verifier_posttrain import (
    _match_action_iou,
    _person_box_deltas,
)
from spectral_detection_posttrain.utils.io import save_checkpoint, save_json
from spectral_detection_posttrain.utils.seed import set_seed
from spectral_detection_posttrain.utils.grad_diagnostics import (
    summarize_current_parameter_gradients,
    summarize_loss_component_gradients,
)


DATA = Path("data/NWPU VHR-10 dataset")
ANNOT = Path("data/NWPU_VHR10_coco.json")
CHECKPOINT = Path("runs/round2100_nwpu_baseline/checkpoint_best.pth")
NUM_CLASSES = 11
MAX_SIZE = 480

SCENE_RAW_IFFT_PRESETS = {
    "maritime": {
        "classes": [2, 8],
        "features": ["fft_edge_truncation@64", "phase_edge@64", "phase_abs_low@64", "center_surround@64"],
    },
    "vehicle": {
        "classes": [10],
        "features": ["hp015_edge@21", "high_edge@7", "high_energy_ratio@7", "high_low_energy_ratio@7"],
    },
    "compact": {
        "classes": [3, 4],
        "features": ["center_surround@64", "center_surround@21", "fft_edge_truncation@15"],
    },
    "sports": {
        "classes": [5, 6, 7],
        "features": [],
    },
}


def resolve_scene_raw_ifft_groups(group_names: list[str]) -> list[dict]:
    names = list(group_names)
    if "all" in names:
        names = list(SCENE_RAW_IFFT_PRESETS.keys())
    groups = []
    for name in names:
        if name not in SCENE_RAW_IFFT_PRESETS:
            raise ValueError(f"Unknown raw-iFFT scene group: {name}")
        preset = SCENE_RAW_IFFT_PRESETS[name]
        groups.append(
            {
                "name": name,
                "classes": list(preset["classes"]),
                "features": list(preset["features"]),
            }
        )
    return groups


def mask_for_class_ids(labels: torch.Tensor, class_ids: list[int]) -> torch.Tensor:
    mask = torch.zeros_like(labels, dtype=torch.bool)
    for class_id in class_ids:
        mask |= labels == int(class_id)
    return mask


class NWPUDataset(Dataset):
    def __init__(self, root: Path, coco_json: Path, img_ids: set[int], max_size: int):
        self.root = Path(root)
        self.max_size = int(max_size)
        self.coco = json.loads(Path(coco_json).read_text(encoding="utf-8"))
        self.img_infos = {img["id"]: img for img in self.coco["images"] if img["id"] in img_ids}
        self.img_ids = list(self.img_infos.keys())
        anns: dict[int, list[dict]] = {}
        for ann in self.coco["annotations"]:
            if ann["image_id"] in img_ids:
                anns.setdefault(ann["image_id"], []).append(ann)
        self.anns = anns

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        info = self.img_infos[img_id]
        img_path = self.root / "positive image set" / info["file_name"]
        if not img_path.exists():
            img_path = self.root / "negative image set" / info["file_name"]
        image = Image.open(str(img_path)).convert("RGB")
        image_t = TF.to_tensor(image)
        boxes, labels = [], []
        for ann in self.anns.get(img_id, []):
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
        }
        _, height, width = image_t.shape
        if max(height, width) > self.max_size:
            scale = self.max_size / float(max(height, width))
            new_h, new_w = int(height * scale), int(width * scale)
            image_t = F.interpolate(
                image_t.unsqueeze(0),
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            target["boxes"] = target["boxes"] * scale
        return image_t, target


def collate(batch):
    return tuple(zip(*batch))


def split_ids() -> tuple[set[int], set[int]]:
    coco = json.loads(ANNOT.read_text(encoding="utf-8"))
    all_ids = list(
        set(
            img["id"]
            for img in coco["images"]
            if (DATA / "positive image set" / img["file_name"]).exists()
        )
    )
    np.random.seed(42)
    np.random.shuffle(all_ids)
    n_train = int(0.7 * len(all_ids))
    return set(all_ids[:n_train]), set(all_ids[n_train:])


def limited_ids(ids: set[int], limit: int | None) -> set[int]:
    ordered = sorted(ids)
    if limit is not None:
        ordered = ordered[: max(0, int(limit))]
    return set(ordered)


def build_loaders(args):
    train_ids, val_ids = split_ids()
    train_ids = limited_ids(train_ids, args.limit_train)
    val_ids = limited_ids(val_ids, args.limit_val)
    train_ds = NWPUDataset(DATA, ANNOT, train_ids, MAX_SIZE)
    val_ds = NWPUDataset(DATA, ANNOT, val_ids, MAX_SIZE)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate, num_workers=0)
    return train_loader, val_loader


def build_nwpu_model(
    device: torch.device,
    *,
    checkpoint_path: Path = CHECKPOINT,
    install_adapter: bool = False,
    enable_cls_adapter: bool = False,
    cls_scale: float = 1.0,
):
    model = build_detector(
        {
            "model": {
                "name": "fasterrcnn_mobilenet_v3_large_320_fpn",
                "model_name": "fasterrcnn_mobilenet_v3_large_320_fpn",
                "pretrained": False,
                "num_classes": NUM_CLASSES,
                "min_size": MAX_SIZE,
                "max_size": MAX_SIZE,
            }
        }
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"]

    # Some checkpoints were saved after the residual bbox adapter was installed.
    # Install the adapter first so the keys line up, then load state_dict.
    has_adapter_keys = any("bbox_adapter" in k for k in state_dict.keys())
    has_cls_adapter_keys = any("cls_adapter" in k for k in state_dict.keys())
    if has_adapter_keys:
        install_residual_bbox_adapter(
            model,
            hidden_dim=128,
            scale=1.0,
            enable_cls_adapter=enable_cls_adapter or has_cls_adapter_keys,
            cls_scale=cls_scale,
        )
    model.load_state_dict(state_dict)

    if install_adapter and not has_adapter_keys:
        install_residual_bbox_adapter(
            model,
            hidden_dim=128,
            scale=1.0,
            enable_cls_adapter=enable_cls_adapter,
            cls_scale=cls_scale,
        )
    return model


def configure_detector_rollout(model, *, score_threshold: float, detections_per_img: int) -> None:
    roi_heads = getattr(model, "roi_heads", None)
    if roi_heads is None:
        return
    if hasattr(roi_heads, "score_thresh"):
        roi_heads.score_thresh = float(score_threshold)
    if hasattr(roi_heads, "detections_per_img"):
        roi_heads.detections_per_img = max(int(getattr(roi_heads, "detections_per_img")), int(detections_per_img))


def configure_detector_eval(model, *, score_threshold: float, detections_per_img: int) -> dict[str, object]:
    roi_heads = getattr(model, "roi_heads", None)
    if roi_heads is None:
        return {}
    previous = {}
    if hasattr(roi_heads, "score_thresh"):
        previous["score_thresh"] = roi_heads.score_thresh
        roi_heads.score_thresh = float(score_threshold)
    if hasattr(roi_heads, "detections_per_img"):
        previous["detections_per_img"] = roi_heads.detections_per_img
        roi_heads.detections_per_img = int(detections_per_img)
    return previous


def restore_detector_eval(model, previous: dict[str, object]) -> None:
    roi_heads = getattr(model, "roi_heads", None)
    if roi_heads is None:
        return
    for name, value in previous.items():
        setattr(roi_heads, name, value)


def evaluate_clean_detector(
    model,
    val_loader,
    device: torch.device,
    *,
    score_threshold: float,
    detections_per_img: int,
):
    previous = configure_detector_eval(
        model,
        score_threshold=float(score_threshold),
        detections_per_img=int(detections_per_img),
    )
    try:
        return evaluate(model, val_loader, device)
    finally:
        restore_detector_eval(model, previous)


def extract_roi_outputs_and_features_for_boxes(model, images: list[torch.Tensor], boxes: list[torch.Tensor]):
    original_sizes = [tuple(img.shape[-2:]) for img in images]
    transformed, _ = model.transform(images, None)
    features = model.backbone(transformed.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])
    scaled_boxes = [
        resize_boxes_to_image(b.to(transformed.tensors.device), original, new)
        for b, original, new in zip(boxes, original_sizes, transformed.image_sizes)
    ]
    roi_features = model.roi_heads.box_roi_pool(features, scaled_boxes, transformed.image_sizes)
    box_features = model.roi_heads.box_head(roi_features)
    class_logits, box_regression = model.roi_heads.box_predictor(box_features)
    return class_logits, box_regression, box_features, scaled_boxes, transformed.image_sizes


def select_manifold_feature_source(
    box_features: torch.Tensor,
    boxes: torch.Tensor,
    image_size: tuple[int, int],
    args,
) -> torch.Tensor:
    source = str(getattr(args, "rescue_manifold_feature_source", "box_features"))
    if source == "box_features":
        return box_features
    if source == "box_features_l2":
        return F.normalize(box_features, p=2, dim=1)
    if source == "geometry":
        if boxes.numel() == 0:
            return box_features.new_empty((0, 6))
        boxes = boxes.to(box_features.device).float()
        height, width = image_size
        width = max(float(width), 1.0)
        height = max(float(height), 1.0)
        x1, y1, x2, y2 = boxes.unbind(dim=1)
        bw = (x2 - x1).clamp_min(0.0)
        bh = (y2 - y1).clamp_min(0.0)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        area = (bw * bh) / max(width * height, 1.0)
        aspect = torch.log((bw + 1.0) / (bh + 1.0))
        return torch.stack([cx / width, cy / height, bw / width, bh / height, area, aspect], dim=1)
    raise ValueError(f"Unknown rescue manifold feature source: {source}")


@torch.no_grad()
def evaluate(model, val_loader, device: torch.device):
    model.eval()
    predictions, targets = [], []
    for images, batch_targets in val_loader:
        outputs = model([image.to(device) for image in images])
        predictions.extend([{k: v.detach().cpu() for k, v in output.items()} for output in outputs])
        targets.extend([{k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in target.items()} for target in batch_targets])
    return evaluate_detection_predictions(predictions, targets, iou_threshold=0.5, score_threshold=0.05)


def proposal_iou_for_scores(proposals: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    if proposals.numel() == 0 or gt_boxes.numel() == 0:
        return proposals.new_zeros((proposals.shape[0],))
    return box_iou(proposals, gt_boxes.to(proposals.device)).max(dim=1).values


def class_box_deltas(box_regression: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    box_regression_4d = box_regression.reshape(box_regression.shape[0], int(num_classes), 4)
    labels = labels.to(box_regression.device).long().clamp(min=0, max=int(num_classes) - 1)
    row = torch.arange(box_regression.shape[0], device=box_regression.device)
    return box_regression_4d[row, labels]


def match_pre_nms_decoded_boxes_to_targets(
    proposals: torch.Tensor,
    box_regression: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    *,
    image_size: tuple[int, int],
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = box_regression.device
    proposals = proposals.to(device).float()
    gt_boxes = gt_boxes.to(device).float()
    gt_labels = gt_labels.to(device).long()
    proposal_count = int(proposals.shape[0])
    best_iou = box_regression.new_zeros((proposal_count,))
    best_labels = torch.zeros((proposal_count,), dtype=torch.long, device=device)
    best_gt_indices = torch.zeros((proposal_count,), dtype=torch.long, device=device)
    target_boxes = box_regression.new_zeros((proposal_count, 4))
    decoded_best_boxes = box_regression.new_zeros((proposal_count, 4))
    if proposal_count == 0 or gt_boxes.numel() == 0:
        return best_iou, best_labels, target_boxes, best_gt_indices, decoded_best_boxes

    box_regression_4d = box_regression.reshape(proposal_count, int(num_classes), 4)
    for gt_idx, gt_label in enumerate(gt_labels.tolist()):
        label = int(gt_label)
        if label <= 0 or label >= int(num_classes):
            continue
        decoded_for_label = decode_box_actions(
            proposals,
            box_regression_4d[:, label, :].unsqueeze(1),
            image_size,
        ).squeeze(1)
        ious = box_iou(decoded_for_label, gt_boxes[gt_idx : gt_idx + 1]).squeeze(1)
        update = ious > best_iou
        best_iou[update] = ious[update]
        best_labels[update] = label
        best_gt_indices[update] = int(gt_idx)
        target_boxes[update] = gt_boxes[gt_idx]
        decoded_best_boxes[update] = decoded_for_label[update]
    return best_iou, best_labels, target_boxes, best_gt_indices, decoded_best_boxes


def pre_nms_score_rescue_loss(
    class_logits: torch.Tensor,
    baseline_logits: torch.Tensor,
    labels: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    score_target: float,
    score_threshold: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if class_logits.numel() == 0:
        zero = class_logits.sum() * 0.0
        return zero, {
            "pre_nms_rescue_count": 0,
            "pre_nms_rescue_active_count": 0,
            "pre_nms_rescue_loss": 0.0,
            "pre_nms_rescue_prob_delta_mean": 0.0,
            "pre_nms_rescue_score_cross_count": 0,
        }
    device = class_logits.device
    labels = labels.to(device).long().clamp(min=0, max=class_logits.shape[1] - 1)
    selected = candidate_mask.to(device).bool() & (labels > 0)
    if not selected.any():
        zero = class_logits.sum() * 0.0
        return zero, {
            "pre_nms_rescue_count": 0,
            "pre_nms_rescue_active_count": 0,
            "pre_nms_rescue_loss": 0.0,
            "pre_nms_rescue_prob_delta_mean": 0.0,
            "pre_nms_rescue_score_cross_count": 0,
        }
    row = torch.arange(labels.numel(), device=device)
    probs = F.softmax(class_logits, dim=1)[row, labels]
    baseline_probs = F.softmax(baseline_logits.to(device), dim=1)[row, labels].detach()
    target = class_logits.new_full(probs[selected].shape, float(score_target)).clamp(
        min=float(score_threshold),
        max=1.0,
    )
    penalties = F.relu(target - probs[selected])
    loss = penalties.pow(2).mean()
    delta = probs[selected] - baseline_probs[selected]
    score_cross = (baseline_probs[selected] < float(score_threshold)) & (probs[selected] >= float(score_threshold))
    return loss, {
        "pre_nms_rescue_count": int(selected.sum().item()),
        "pre_nms_rescue_active_count": int((penalties > 0).sum().item()),
        "pre_nms_rescue_loss": float(loss.detach().cpu().item()),
        "pre_nms_rescue_prob_delta_mean": float(delta.mean().detach().cpu().item()),
        "pre_nms_rescue_score_cross_count": int(score_cross.sum().item()),
    }


def mine_pre_nms_local_dpo_pairs(
    labels: torch.Tensor,
    best_gt_indices: torch.Tensor,
    best_iou: torch.Tensor,
    baseline_probs: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    min_iou_gap: float,
    require_rejected_score_ge_chosen: bool,
    max_pairs_per_gt: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    device = best_iou.device
    labels = labels.to(device).long()
    best_gt_indices = best_gt_indices.to(device).long()
    best_iou = best_iou.to(device).float()
    baseline_probs = baseline_probs.to(device).float()
    candidate_mask = candidate_mask.to(device).bool() & (labels > 0)
    chosen_indices: list[int] = []
    rejected_indices: list[int] = []
    iou_gaps: list[torch.Tensor] = []
    if candidate_mask.any():
        for gt_idx in best_gt_indices[candidate_mask].unique(sorted=True).tolist():
            gt_idx = int(gt_idx)
            chosen_for_gt = torch.nonzero(candidate_mask & (best_gt_indices == gt_idx), as_tuple=False).flatten()
            if chosen_for_gt.numel() == 0:
                continue
            pair_candidates: list[tuple[float, int, int, torch.Tensor]] = []
            for chosen_idx_tensor in chosen_for_gt:
                chosen_idx = int(chosen_idx_tensor.item())
                same_local = (
                    (best_gt_indices == gt_idx)
                    & (labels == labels[chosen_idx])
                    & (torch.arange(labels.numel(), device=device) != chosen_idx)
                )
                iou_gap = best_iou[chosen_idx] - best_iou
                valid_rejected = same_local & (iou_gap >= float(min_iou_gap))
                if bool(require_rejected_score_ge_chosen):
                    valid_rejected = valid_rejected & (
                        baseline_probs >= baseline_probs[chosen_idx].detach()
                    )
                rejected_for_chosen = torch.nonzero(valid_rejected, as_tuple=False).flatten()
                for rejected_idx_tensor in rejected_for_chosen:
                    rejected_idx = int(rejected_idx_tensor.item())
                    gap = iou_gap[rejected_idx]
                    score_priority = float(gap.detach().cpu().item())
                    pair_candidates.append((score_priority, chosen_idx, rejected_idx, gap))
            if not pair_candidates:
                continue
            pair_candidates.sort(key=lambda item: item[0], reverse=True)
            for _, chosen_idx, rejected_idx, gap in pair_candidates[: max(1, int(max_pairs_per_gt))]:
                chosen_indices.append(chosen_idx)
                rejected_indices.append(rejected_idx)
                iou_gaps.append(gap.detach())
    if not chosen_indices:
        empty = torch.empty((0,), dtype=torch.long, device=device)
        return empty, empty, {
            "pre_nms_dpo_pair_count": 0,
            "pre_nms_dpo_mean_iou_gap": 0.0,
        }
    chosen = torch.tensor(chosen_indices, dtype=torch.long, device=device)
    rejected = torch.tensor(rejected_indices, dtype=torch.long, device=device)
    mean_gap = torch.stack(iou_gaps).mean() if iou_gaps else best_iou.new_tensor(0.0)
    return chosen, rejected, {
        "pre_nms_dpo_pair_count": int(chosen.numel()),
        "pre_nms_dpo_mean_iou_gap": float(mean_gap.detach().cpu().item()),
    }


def local_pre_nms_dpo_loss(
    current_logits: torch.Tensor,
    baseline_logits: torch.Tensor,
    labels: torch.Tensor,
    chosen_indices: torch.Tensor,
    rejected_indices: torch.Tensor,
    *,
    beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if current_logits.numel() == 0 or chosen_indices.numel() == 0:
        zero = current_logits.sum() * 0.0
        return zero, {
            "pre_nms_dpo_pair_count": 0,
            "pre_nms_dpo_loss": 0.0,
            "pre_nms_dpo_preference_margin_mean": 0.0,
            "pre_nms_dpo_win_count": 0,
        }
    device = current_logits.device
    labels = labels.to(device).long().clamp(min=0, max=current_logits.shape[1] - 1)
    chosen = chosen_indices.to(device).long()
    rejected = rejected_indices.to(device).long()
    current_row = torch.arange(labels.numel(), device=device)
    current_label_logits = current_logits[current_row, labels]
    baseline_label_logits = baseline_logits.to(device)[current_row, labels].detach()
    policy_margin = current_label_logits[chosen] - current_label_logits[rejected]
    reference_margin = baseline_label_logits[chosen] - baseline_label_logits[rejected]
    preference_margin = policy_margin - reference_margin
    loss = -F.logsigmoid(float(beta) * preference_margin).mean()
    return loss, {
        "pre_nms_dpo_pair_count": int(chosen.numel()),
        "pre_nms_dpo_loss": float(loss.detach().cpu().item()),
        "pre_nms_dpo_preference_margin_mean": float(preference_margin.detach().mean().cpu().item()),
        "pre_nms_dpo_win_count": int((policy_margin > 0.0).sum().item()),
    }


def class_threshold_tensor(labels: torch.Tensor, reference_stats: dict[str, float], fallback: float) -> torch.Tensor:
    thresholds = reference_stats.get("class_thresholds", {})
    if not thresholds:
        return labels.new_full(labels.shape, float(fallback), dtype=torch.float32)
    values = [float(thresholds.get(str(int(label)), float(fallback))) for label in labels.detach().cpu().long().tolist()]
    return torch.tensor(values, dtype=torch.float32, device=labels.device)


def configure_trainable_parts(model, args) -> list[str]:
    mode = str(args.trainable_mode)
    if mode == "adapter":
        return freeze_selected_adapters(
            model,
            train_bbox_adapter=True,
            train_cls_adapter=bool(args.rescue_mode),
        )
    if mode == "predictor":
        return freeze_adapters_and_predictor(
            model,
            train_cls_adapter=bool(args.rescue_mode),
            train_cls_score=True,
            train_bbox_pred=True,
        )
    if mode == "cls_adapter":
        return freeze_selected_adapters(
            model,
            train_bbox_adapter=False,
            train_cls_adapter=True,
        )
    if mode == "bbox_adapter":
        return freeze_selected_adapters(
            model,
            train_bbox_adapter=True,
            train_cls_adapter=False,
        )
    if mode == "cls_predictor":
        return freeze_adapters_and_predictor(
            model,
            train_bbox_adapter=False,
            train_cls_adapter=bool(args.rescue_mode),
            train_cls_score=True,
            train_bbox_pred=False,
        )
    if mode == "cls_score":
        return freeze_adapters_and_predictor(
            model,
            train_bbox_adapter=False,
            train_cls_adapter=False,
            train_cls_score=True,
            train_bbox_pred=False,
        )
    if mode == "bbox_predictor":
        return freeze_adapters_and_predictor(
            model,
            train_bbox_adapter=True,
            train_cls_adapter=False,
            train_cls_score=False,
            train_bbox_pred=True,
        )
    if mode == "bbox_predictor_cls_adapter":
        return freeze_adapters_and_predictor(
            model,
            train_bbox_adapter=True,
            train_cls_adapter=True,
            train_cls_score=False,
            train_bbox_pred=True,
        )
    raise ValueError(f"Unknown trainable mode: {mode}")


def build_optimizer(model, args):
    predictor = model.roi_heads.box_predictor
    adapter_params = []
    predictor_cls_params = []
    predictor_box_params = []
    other_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "bbox_adapter" in name or "cls_adapter" in name:
            adapter_params.append(parameter)
        elif "base_predictor.cls_score" in name:
            predictor_cls_params.append(parameter)
        elif "base_predictor.bbox_pred" in name:
            predictor_box_params.append(parameter)
        else:
            other_params.append(parameter)

    groups = []
    adapter_lr = float(args.lr if args.adapter_lr is None else args.adapter_lr)
    predictor_lr = float(args.lr if args.predictor_lr is None else args.predictor_lr)
    cls_score_lr = float(args.lr if args.cls_score_lr is None else args.cls_score_lr)
    if adapter_params:
        groups.append({"params": adapter_params, "lr": adapter_lr})
    if predictor_cls_params:
        groups.append({"params": predictor_cls_params, "lr": cls_score_lr})
    if predictor_box_params:
        groups.append({"params": predictor_box_params, "lr": predictor_lr})
    if other_params:
        groups.append({"params": other_params, "lr": float(args.lr)})
    if not groups:
        raise RuntimeError("No trainable parameters.")
    return torch.optim.AdamW(groups, lr=float(args.lr), weight_decay=float(args.weight_decay))


def accumulate_metric_sums(metric_sums: dict[str, float], metrics: dict[str, float]) -> None:
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            metric_sums[key] = metric_sums.get(key, 0.0) + float(value)


def average_metric_sums(metric_sums: dict[str, float], count: int) -> dict[str, float]:
    if count <= 0:
        return {}
    return {key: value / float(count) for key, value in metric_sums.items()}


def rescue_config_from_args(args) -> ConfidenceRescueConfig:
    return ConfidenceRescueConfig(
        low_conf_max=float(args.rescue_low_conf_max),
        high_conf_min=float(args.rescue_high_conf_min),
        high_iou_min=float(args.rescue_high_iou_min),
        low_iou_max=float(args.rescue_low_iou_max),
        positive_weight=float(args.rescue_positive_weight),
        negative_weight=float(args.rescue_negative_weight),
        include_low_conf_negatives=bool(args.rescue_include_low_conf_negatives),
        verifier_positive_min=(
            float(args.rescue_verifier_gate)
            if bool(args.rescue_mode) and args.rescue_verifier_mode != "none"
            else None
        ),
        verifier_hard_negative_min=(
            float(args.rescue_hard_negative_verifier_gate)
            if bool(args.rescue_use_hard_negative_mining)
            else None
        ),
        verifier_weight_mode=str(args.rescue_verifier_weight_mode),
        verifier_weight_temperature=float(args.rescue_verifier_weight_temperature),
    )


def select_rollout_proposals(rollout: dict, args) -> tuple[torch.Tensor, torch.Tensor]:
    threshold = float(args.rollout_score_threshold if args.rescue_mode else args.score_threshold)
    keep = rollout["scores"].detach().cpu() >= threshold
    proposals = rollout["boxes"].detach().cpu()[keep][: int(args.max_proposals)]
    scores = rollout["scores"].detach().cpu()[keep][: int(args.max_proposals)]
    return proposals, scores


def select_top_rpn_proposals(
    proposals: torch.Tensor,
    *,
    transformed_size: tuple[int, int],
    original_size: tuple[int, int],
    max_proposals: int,
) -> torch.Tensor:
    proposals = proposals.detach().cpu().float()[: int(max_proposals)].clone()
    if proposals.numel() == 0:
        return proposals.reshape(0, 4)
    return resize_boxes_to_image(proposals, transformed_size, original_size)


@torch.no_grad()
def select_rpn_proposals_for_images(model, images: list[torch.Tensor], args, device: torch.device) -> list[torch.Tensor]:
    transformed, _ = model.transform([image.to(device) for image in images], None)
    features = model.backbone(transformed.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])
    proposals, _ = model.rpn(transformed, features, None)
    return [
        select_top_rpn_proposals(
            proposal,
            transformed_size=tuple(transformed_size),
            original_size=tuple(image.shape[-2:]),
            max_proposals=int(args.max_proposals),
        )
        for proposal, transformed_size, image in zip(proposals, transformed.image_sizes, images)
    ]


def matched_label_probabilities(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    labels = labels.to(logits.device).long()
    valid = (labels > 0) & (labels < probs.shape[1])
    gathered = probs.new_zeros((labels.shape[0],))
    if valid.any():
        gathered[valid] = probs[valid].gather(1, labels[valid].unsqueeze(1)).squeeze(1)
    return gathered


def _fit_logistic_scores(train_features: np.ndarray, train_labels: np.ndarray, query_features: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    model = LogisticRegression(C=0.1, class_weight="balanced", solver="liblinear", max_iter=1000, random_state=42)
    model.fit(np.asarray(train_features, dtype=np.float64), np.asarray(train_labels, dtype=np.int32))
    train_scores = model.decision_function(train_features)
    query_scores = model.decision_function(query_features)
    return train_scores, query_scores, {
        "coef": model.coef_.astype(float).tolist(),
        "intercept": model.intercept_.astype(float).tolist(),
    }


def _fit_center_scores(train_features: np.ndarray, train_labels: np.ndarray, query_features: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    train_features = np.asarray(train_features, dtype=np.float64)
    query_features = np.asarray(query_features, dtype=np.float64)
    labels = np.asarray(train_labels, dtype=bool)
    positive_center = train_features[labels].mean(axis=0)
    negative_center = train_features[~labels].mean(axis=0)

    def score(features: np.ndarray) -> np.ndarray:
        pos_dist = np.linalg.norm(features - positive_center.reshape(1, -1), axis=1)
        neg_dist = np.linalg.norm(features - negative_center.reshape(1, -1), axis=1)
        return neg_dist - pos_dist

    return score(train_features), score(query_features), {
        "positive_center": positive_center.astype(float).tolist(),
        "negative_center": negative_center.astype(float).tolist(),
    }


def _fit_fusion_scorer(train_columns: list[np.ndarray], train_labels: np.ndarray, *, method: str) -> tuple[np.ndarray, dict[str, object]]:
    means = [float(np.asarray(column, dtype=np.float64).mean()) for column in train_columns]
    stds = [max(float(np.asarray(column, dtype=np.float64).std()), 1e-6) for column in train_columns]
    matrix = np.stack(
        [
            (np.asarray(column, dtype=np.float64) - mean) / std
            for column, mean, std in zip(train_columns, means, stds)
        ],
        axis=1,
    )
    if method == "train_effect":
        scorer = fit_train_effect_scorer(matrix, np.asarray(train_labels, dtype=bool), method="train_effect_sum")
        return scorer.score(matrix), {
            "method": "train_effect",
            "column_means": means,
            "column_stds": stds,
            "scaler_mean": scorer.scaler.mean_.astype(float).tolist(),
            "scaler_scale": scorer.scaler.scale_.astype(float).tolist(),
            "weights": scorer.weights.astype(float).tolist(),
        }
    if method == "logistic":
        model = LogisticRegression(C=0.1, class_weight="balanced", solver="liblinear", max_iter=1000, random_state=42)
        model.fit(matrix, np.asarray(train_labels, dtype=np.int32))
        return model.decision_function(matrix), {
            "method": "logistic",
            "column_means": means,
            "column_stds": stds,
            "coef": model.coef_.astype(float).tolist(),
            "intercept": model.intercept_.astype(float).tolist(),
        }
    raise ValueError(f"Unknown fusion scorer method: {method}")


def _apply_linear_scorer(features: torch.Tensor, stats: dict[str, object], *, prefix: str) -> torch.Tensor:
    device = features.device
    dtype = features.dtype
    if str(stats[f"{prefix}_scorer"]) == "logistic":
        coef = torch.tensor(stats[f"{prefix}_coef"], dtype=dtype, device=device).reshape(1, -1)
        intercept = torch.tensor(stats[f"{prefix}_intercept"], dtype=dtype, device=device).reshape(())
        return features.matmul(coef.t()).reshape(-1) + intercept
    if str(stats[f"{prefix}_scorer"]) == "center":
        positive_center = torch.tensor(stats[f"{prefix}_positive_center"], dtype=dtype, device=device).reshape(1, -1)
        negative_center = torch.tensor(stats[f"{prefix}_negative_center"], dtype=dtype, device=device).reshape(1, -1)
        pos_dist = torch.linalg.norm(features - positive_center, dim=1)
        neg_dist = torch.linalg.norm(features - negative_center, dim=1)
        return neg_dist - pos_dist
    raise ValueError(f"Unknown {prefix} scorer: {stats[f'{prefix}_scorer']}")


def _apply_hd_projection(features: torch.Tensor, stats: dict[str, object]) -> torch.Tensor:
    device = features.device
    dtype = features.dtype
    projected = F.normalize(features.float(), p=2, dim=1) if bool(stats.get("hd_use_l2", True)) else features.float()
    scaler_mean = torch.tensor(stats["hd_scaler_mean"], dtype=projected.dtype, device=device).reshape(1, -1)
    scaler_scale = torch.tensor(stats["hd_scaler_scale"], dtype=projected.dtype, device=device).reshape(1, -1).clamp_min(1e-6)
    projected = (projected - scaler_mean) / scaler_scale
    if int(stats.get("hd_pca_components", 0)) > 0:
        pca_mean = torch.tensor(stats["hd_pca_mean"], dtype=projected.dtype, device=device).reshape(1, -1)
        pca_components = torch.tensor(stats["hd_pca_components_matrix"], dtype=projected.dtype, device=device)
        projected = (projected - pca_mean).matmul(pca_components.t())
        if bool(stats.get("hd_pca_whiten", True)):
            pca_scale = torch.tensor(stats["hd_pca_whiten_scale"], dtype=projected.dtype, device=device).reshape(1, -1).clamp_min(1e-6)
            projected = projected / pca_scale
    return projected.to(dtype=dtype)


def _apply_fusion_scorer(columns: list[torch.Tensor], stats: dict[str, object]) -> torch.Tensor:
    device = columns[0].device
    dtype = columns[0].dtype
    means = torch.tensor(stats["fusion_column_means"], dtype=dtype, device=device).reshape(1, -1)
    stds = torch.tensor(stats["fusion_column_stds"], dtype=dtype, device=device).reshape(1, -1).clamp_min(1e-6)
    matrix = torch.stack([column.to(device=device, dtype=dtype) for column in columns], dim=1)
    matrix = (matrix - means) / stds
    method = str(stats["fusion_method"])
    if method == "train_effect":
        scaler_mean = torch.tensor(stats["fusion_scaler_mean"], dtype=dtype, device=device).reshape(1, -1)
        scaler_scale = torch.tensor(stats["fusion_scaler_scale"], dtype=dtype, device=device).reshape(1, -1).clamp_min(1e-6)
        weights = torch.tensor(stats["fusion_weights"], dtype=dtype, device=device).reshape(1, -1)
        return ((matrix - scaler_mean) / scaler_scale * weights).sum(dim=1)
    if method == "logistic":
        coef = torch.tensor(stats["fusion_coef"], dtype=dtype, device=device).reshape(1, -1)
        intercept = torch.tensor(stats["fusion_intercept"], dtype=dtype, device=device).reshape(())
        return matrix.matmul(coef.t()).reshape(-1) + intercept
    raise ValueError(f"Unknown fusion method: {method}")


def class_margin_rescue_loss(
    class_logits: torch.Tensor,
    best_iou: torch.Tensor,
    best_labels: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    margin: float,
    include_background: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    if class_logits.numel() == 0 or class_logits.shape[1] <= 1:
        zero = class_logits.sum() * 0.0
        return zero, {"class_margin_count": 0, "class_margin_active_count": 0, "class_margin_loss": 0.0}
    device = class_logits.device
    labels = best_labels.to(device).long()
    selected = (
        candidate_mask.to(device).bool()
        & (best_iou.to(device).float() > 0.0)
        & (labels > 0)
        & (labels < class_logits.shape[1])
    )
    if not selected.any():
        zero = class_logits.sum() * 0.0
        return zero, {"class_margin_count": 0, "class_margin_active_count": 0, "class_margin_loss": 0.0}
    rows = class_logits[selected]
    target_labels = labels[selected]
    row_indices = torch.arange(rows.shape[0], device=device)
    target_logits = rows[row_indices, target_labels]
    masked = rows.clone()
    if not bool(include_background):
        masked[:, 0] = -torch.inf
    masked[row_indices, target_labels] = -torch.inf
    top_other_logits = masked.max(dim=1).values
    penalties = F.relu(top_other_logits + float(margin) - target_logits)
    loss = penalties.pow(2).mean()
    return loss, {
        "class_margin_count": int(selected.sum().item()),
        "class_margin_active_count": int((penalties > 0).sum().item()),
        "class_margin_loss": float(loss.detach().cpu().item()),
    }


def build_chain_rescue_candidate_mask(
    best_iou: torch.Tensor,
    best_labels: torch.Tensor,
    best_gt_indices: torch.Tensor,
    low_conf_scores: torch.Tensor,
    unmatched_gt_mask: torch.Tensor,
    *,
    low_conf_max: float,
    high_iou_min: float,
    topk_per_gt: int,
) -> torch.Tensor:
    device = best_iou.device
    selected = (
        unmatched_gt_mask.to(device).bool()
        & (low_conf_scores.to(device).float() <= float(low_conf_max))
        & (best_iou.to(device).float() >= float(high_iou_min))
        & (best_labels.to(device).long() > 0)
    )
    if not selected.any():
        return selected
    topk = max(1, int(topk_per_gt))
    out = torch.zeros_like(selected)
    gt_indices = best_gt_indices.to(device).long()
    for gt_idx in gt_indices[selected].unique(sorted=True):
        candidates = torch.nonzero(selected & (gt_indices == gt_idx), as_tuple=False).flatten()
        if candidates.numel() == 0:
            continue
        scores = best_iou[candidates].float()
        keep_count = min(topk, int(candidates.numel()))
        keep = candidates[torch.topk(scores, k=keep_count).indices]
        out[keep] = True
    return out


def chain_rescue_ranking_loss(
    class_logits: torch.Tensor,
    best_labels: torch.Tensor,
    positive_mask: torch.Tensor,
    dangerous_negative_mask: torch.Tensor,
    *,
    margin: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if class_logits.numel() == 0:
        zero = class_logits.sum() * 0.0
        return zero, {
            "chain_ranking_pair_count": 0,
            "chain_ranking_active_count": 0,
            "chain_ranking_loss": 0.0,
        }
    device = class_logits.device
    labels = best_labels.to(device).long().clamp(min=0, max=class_logits.shape[1] - 1)
    row = torch.arange(labels.numel(), device=device)
    label_logits = class_logits[row, labels]
    positives = positive_mask.to(device).bool() & (labels > 0)
    negatives = dangerous_negative_mask.to(device).bool() & (labels > 0)
    if not positives.any() or not negatives.any():
        zero = class_logits.sum() * 0.0
        return zero, {
            "chain_ranking_pair_count": 0,
            "chain_ranking_active_count": 0,
            "chain_ranking_loss": 0.0,
        }
    positive_scores = label_logits[positives]
    negative_scores = label_logits[negatives]
    penalties = F.relu(negative_scores.unsqueeze(0) + float(margin) - positive_scores.unsqueeze(1))
    loss = penalties.mean()
    return loss, {
        "chain_ranking_pair_count": int(penalties.numel()),
        "chain_ranking_active_count": int((penalties > 0).sum().item()),
        "chain_ranking_loss": float(loss.detach().cpu().item()),
    }


def nms_aware_rescue_ranking_loss(
    class_logits: torch.Tensor,
    decoded_boxes: torch.Tensor,
    labels: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    nms_iou_threshold: float,
    margin: float,
    require_suppressor_score_ge_candidate: bool = True,
    ranking_mode: str = "joint",
) -> tuple[torch.Tensor, dict[str, float]]:
    if class_logits.numel() == 0 or decoded_boxes.numel() == 0:
        zero = class_logits.sum() * 0.0
        return zero, {
            "nms_aware_pair_count": 0,
            "nms_aware_active_count": 0,
            "nms_aware_ranking_loss": 0.0,
        }
    device = class_logits.device
    labels = labels.to(device).long().clamp(min=0, max=class_logits.shape[1] - 1)
    decoded_boxes = decoded_boxes.to(device).float()
    candidate_mask = candidate_mask.to(device).bool()
    row = torch.arange(labels.numel(), device=device)
    label_logits = class_logits[row, labels]
    ranking_mode = str(ranking_mode)
    valid_candidates = torch.nonzero(candidate_mask & (labels > 0), as_tuple=False).flatten()
    penalties = []
    for positive_idx in valid_candidates.tolist():
        same_class = (labels == labels[positive_idx]) & (row != int(positive_idx))
        if not same_class.any():
            continue
        negative_indices = torch.nonzero(same_class, as_tuple=False).flatten()
        overlaps = box_iou(decoded_boxes[positive_idx].unsqueeze(0), decoded_boxes[negative_indices]).squeeze(0)
        valid_negative = overlaps >= float(nms_iou_threshold)
        if bool(require_suppressor_score_ge_candidate):
            valid_negative = valid_negative & (label_logits[negative_indices] >= label_logits[positive_idx].detach())
        if not valid_negative.any():
            continue
        valid_negative_indices = negative_indices[valid_negative]
        suppressor_score = label_logits[valid_negative_indices].max()
        if ranking_mode == "detached_suppressor":
            suppressor_score = suppressor_score.detach()
        elif ranking_mode != "joint":
            raise ValueError(f"Unknown nms-aware ranking mode: {ranking_mode}")
        penalties.append(F.relu(suppressor_score + float(margin) - label_logits[positive_idx]))
    if not penalties:
        zero = class_logits.sum() * 0.0
        return zero, {
            "nms_aware_pair_count": 0,
            "nms_aware_active_count": 0,
            "nms_aware_ranking_loss": 0.0,
        }
    penalty_tensor = torch.stack(penalties)
    loss = penalty_tensor.mean()
    return loss, {
        "nms_aware_pair_count": int(penalty_tensor.numel()),
        "nms_aware_active_count": int((penalty_tensor > 0).sum().item()),
        "nms_aware_ranking_loss": float(loss.detach().cpu().item()),
    }


def blocked_nms_crossing_rescue_loss(
    class_logits: torch.Tensor,
    decoded_boxes: torch.Tensor,
    labels: torch.Tensor,
    best_iou: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    score_threshold: float,
    score_epsilon: float,
    nms_iou_threshold: float,
    base_margin: float,
    iou_margin_scale: float,
    max_margin: float,
    rank_weight: float,
    crossing_weight: float,
    require_suppressor_score_ge_candidate: bool = True,
    ranking_mode: str = "joint",
    baseline_logits: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if class_logits.numel() == 0 or decoded_boxes.numel() == 0:
        zero = class_logits.sum() * 0.0
        return zero, {
            "blocked_nms_pair_count": 0,
            "blocked_nms_active_rank_count": 0,
            "blocked_nms_active_crossing_count": 0,
            "blocked_nms_rank_loss": 0.0,
            "blocked_nms_crossing_loss": 0.0,
            "blocked_nms_loss": 0.0,
            "blocked_nms_candidate_delta_mean": 0.0,
            "blocked_nms_suppressor_delta_mean": 0.0,
            "blocked_nms_relative_delta_mean": 0.0,
        }
    device = class_logits.device
    labels = labels.to(device).long().clamp(min=0, max=class_logits.shape[1] - 1)
    decoded_boxes = decoded_boxes.to(device).float()
    best_iou = best_iou.to(device).float()
    candidate_mask = candidate_mask.to(device).bool()
    probs = F.softmax(class_logits, dim=1)
    row = torch.arange(labels.numel(), device=device)
    label_probs = probs[row, labels]
    ranking_mode = str(ranking_mode)
    baseline_label_probs = None
    if ranking_mode == "delta":
        if baseline_logits is None:
            raise ValueError("baseline_logits is required when ranking_mode='delta'")
        baseline_probs = F.softmax(baseline_logits.to(device), dim=1)
        baseline_label_probs = baseline_probs[row, labels]
    elif ranking_mode not in {"joint", "detached_suppressor"}:
        raise ValueError(f"Unknown blocked-nms ranking mode: {ranking_mode}")
    valid_candidates = torch.nonzero(candidate_mask & (labels > 0), as_tuple=False).flatten()
    rank_penalties = []
    crossing_penalties = []
    candidate_deltas = []
    suppressor_deltas = []
    for positive_idx in valid_candidates.tolist():
        positive_idx = int(positive_idx)
        same_class = (labels == labels[positive_idx]) & (row != positive_idx)
        if not same_class.any():
            continue
        negative_indices = torch.nonzero(same_class, as_tuple=False).flatten()
        overlaps = box_iou(decoded_boxes[positive_idx].unsqueeze(0), decoded_boxes[negative_indices]).squeeze(0)
        valid_negative = overlaps >= float(nms_iou_threshold)
        if bool(require_suppressor_score_ge_candidate):
            valid_negative = valid_negative & (label_probs[negative_indices].detach() >= label_probs[positive_idx].detach())
        if not valid_negative.any():
            continue
        valid_negative_indices = negative_indices[valid_negative]
        suppressor_probs = label_probs[valid_negative_indices]
        suppressor_best_iou = best_iou[valid_negative_indices]
        max_suppressor_prob, max_idx = suppressor_probs.max(dim=0)
        chosen_suppressor_idx = valid_negative_indices[int(max_idx.item())]
        chosen_suppressor_iou = suppressor_best_iou[int(max_idx.item())]
        margin = float(base_margin) + float(iou_margin_scale) * torch.clamp(
            best_iou[positive_idx].detach() - chosen_suppressor_iou.detach(),
            min=0.0,
        )
        margin = torch.clamp(margin, min=0.0, max=float(max_margin))
        if ranking_mode == "delta":
            assert baseline_label_probs is not None
            candidate_delta = label_probs[positive_idx] - baseline_label_probs[positive_idx].detach()
            suppressor_delta = max_suppressor_prob - baseline_label_probs[chosen_suppressor_idx].detach()
            rank_penalties.append(F.relu(suppressor_delta + margin - candidate_delta))
            candidate_deltas.append(candidate_delta.detach())
            suppressor_deltas.append(suppressor_delta.detach())
        else:
            suppressor_for_rank = max_suppressor_prob.detach() if ranking_mode == "detached_suppressor" else max_suppressor_prob
            rank_penalties.append(F.relu(suppressor_for_rank + margin - label_probs[positive_idx]))
            if baseline_label_probs is not None:
                candidate_deltas.append((label_probs[positive_idx] - baseline_label_probs[positive_idx]).detach())
                suppressor_deltas.append((max_suppressor_prob - baseline_label_probs[chosen_suppressor_idx]).detach())
        crossing_penalties.append(
            F.relu(float(score_threshold) + float(score_epsilon) - label_probs[positive_idx])
        )
    if not rank_penalties:
        zero = class_logits.sum() * 0.0
        return zero, {
            "blocked_nms_pair_count": 0,
            "blocked_nms_active_rank_count": 0,
            "blocked_nms_active_crossing_count": 0,
            "blocked_nms_rank_loss": 0.0,
            "blocked_nms_crossing_loss": 0.0,
            "blocked_nms_loss": 0.0,
            "blocked_nms_candidate_delta_mean": 0.0,
            "blocked_nms_suppressor_delta_mean": 0.0,
            "blocked_nms_relative_delta_mean": 0.0,
        }
    rank_tensor = torch.stack(rank_penalties)
    crossing_tensor = torch.stack(crossing_penalties)
    rank_loss = rank_tensor.mean()
    crossing_loss = crossing_tensor.mean()
    loss = float(rank_weight) * rank_loss + float(crossing_weight) * crossing_loss
    candidate_delta_mean = torch.stack(candidate_deltas).mean() if candidate_deltas else rank_tensor.new_tensor(0.0)
    suppressor_delta_mean = torch.stack(suppressor_deltas).mean() if suppressor_deltas else rank_tensor.new_tensor(0.0)
    return loss, {
        "blocked_nms_pair_count": int(rank_tensor.numel()),
        "blocked_nms_active_rank_count": int((rank_tensor > 0).sum().item()),
        "blocked_nms_active_crossing_count": int((crossing_tensor > 0).sum().item()),
        "blocked_nms_rank_loss": float(rank_loss.detach().cpu().item()),
        "blocked_nms_crossing_loss": float(crossing_loss.detach().cpu().item()),
        "blocked_nms_loss": float(loss.detach().cpu().item()),
        "blocked_nms_candidate_delta_mean": float(candidate_delta_mean.detach().cpu().item()),
        "blocked_nms_suppressor_delta_mean": float(suppressor_delta_mean.detach().cpu().item()),
        "blocked_nms_relative_delta_mean": float((candidate_delta_mean - suppressor_delta_mean).detach().cpu().item()),
    }


def find_same_gt_worse_duplicate_pairs(
    candidate_boxes: torch.Tensor,
    candidate_labels: torch.Tensor,
    candidate_gt_indices: torch.Tensor,
    target: dict,
    final_prediction: dict,
    *,
    score_threshold: float,
    nms_iou_threshold: float,
    min_iou_gap: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    candidate_boxes = candidate_boxes.detach().cpu().float()
    candidate_labels = candidate_labels.detach().cpu().long()
    candidate_gt_indices = candidate_gt_indices.detach().cpu().long()
    final_boxes = final_prediction.get("boxes", torch.empty((0, 4))).detach().cpu().float()
    final_labels = final_prediction.get("labels", torch.empty((0,), dtype=torch.long)).detach().cpu().long()
    final_scores = final_prediction.get("scores", torch.empty((0,))).detach().cpu().float()
    gt_boxes = target.get("boxes", torch.empty((0, 4))).detach().cpu().float()
    gt_labels = target.get("labels", torch.empty((0,), dtype=torch.long)).detach().cpu().long()
    if candidate_boxes.numel() == 0 or final_boxes.numel() == 0 or gt_boxes.numel() == 0:
        return (
            torch.empty((0,), dtype=torch.long),
            torch.empty((0, 4), dtype=torch.float32),
            {"same_gt_duplicate_pair_count": 0},
        )
    candidate_indices = []
    suppressor_boxes = []
    final_keep = final_scores >= float(score_threshold)
    if not final_keep.any():
        return (
            torch.empty((0,), dtype=torch.long),
            torch.empty((0, 4), dtype=torch.float32),
            {"same_gt_duplicate_pair_count": 0},
        )
    kept_boxes = final_boxes[final_keep]
    kept_labels = final_labels[final_keep]
    for idx in range(int(candidate_boxes.shape[0])):
        gt_idx = int(candidate_gt_indices[idx].item())
        if gt_idx < 0 or gt_idx >= int(gt_boxes.shape[0]):
            continue
        if int(candidate_labels[idx].item()) != int(gt_labels[gt_idx].item()):
            continue
        overlaps = box_iou(candidate_boxes[idx].unsqueeze(0), kept_boxes).squeeze(0)
        same_class = kept_labels == candidate_labels[idx]
        overlaps = torch.where(same_class, overlaps, torch.full_like(overlaps, -1.0))
        best_overlap, suppressor_idx = overlaps.max(dim=0)
        if float(best_overlap.item()) < float(nms_iou_threshold):
            continue
        candidate_gt_iou = float(box_iou(candidate_boxes[idx].unsqueeze(0), gt_boxes[gt_idx].unsqueeze(0)).item())
        suppressor_gt_iou = float(
            box_iou(kept_boxes[int(suppressor_idx.item())].unsqueeze(0), gt_boxes[gt_idx].unsqueeze(0)).item()
        )
        if candidate_gt_iou <= suppressor_gt_iou + float(min_iou_gap):
            continue
        candidate_indices.append(idx)
        suppressor_boxes.append(kept_boxes[int(suppressor_idx.item())])
    if not candidate_indices:
        return (
            torch.empty((0,), dtype=torch.long),
            torch.empty((0, 4), dtype=torch.float32),
            {"same_gt_duplicate_pair_count": 0},
        )
    return (
        torch.tensor(candidate_indices, dtype=torch.long),
        torch.stack(suppressor_boxes, dim=0).float(),
        {"same_gt_duplicate_pair_count": len(candidate_indices)},
    )


def find_same_gt_worse_duplicate_proposal_pairs(
    decoded_boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    gt_indices: torch.Tensor,
    candidate_mask: torch.Tensor,
    target: dict,
    *,
    nms_iou_threshold: float,
    min_iou_gap: float,
    require_suppressor_score_ge_candidate: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    device = decoded_boxes.device
    decoded_boxes_cpu = decoded_boxes.detach().cpu().float()
    scores_cpu = scores.detach().cpu().float()
    labels_cpu = labels.detach().cpu().long()
    gt_indices_cpu = gt_indices.detach().cpu().long()
    candidate_mask_cpu = candidate_mask.detach().cpu().bool()
    gt_boxes = target.get("boxes", torch.empty((0, 4))).detach().cpu().float()
    gt_labels = target.get("labels", torch.empty((0,), dtype=torch.long)).detach().cpu().long()
    if decoded_boxes_cpu.numel() == 0 or gt_boxes.numel() == 0 or not candidate_mask_cpu.any():
        empty = torch.empty((0,), dtype=torch.long, device=device)
        return empty, empty, {"same_gt_proposal_pair_count": 0}

    positive_indices = []
    negative_indices = []
    all_indices = torch.arange(decoded_boxes_cpu.shape[0])
    for positive_idx in torch.nonzero(candidate_mask_cpu, as_tuple=False).flatten().tolist():
        gt_idx = int(gt_indices_cpu[positive_idx].item())
        if gt_idx < 0 or gt_idx >= int(gt_boxes.shape[0]):
            continue
        label = int(labels_cpu[positive_idx].item())
        if label <= 0 or label != int(gt_labels[gt_idx].item()):
            continue

        same_target = (
            (all_indices != int(positive_idx))
            & (gt_indices_cpu == gt_idx)
            & (labels_cpu == label)
        )
        if not same_target.any():
            continue

        duplicate_indices = torch.nonzero(same_target, as_tuple=False).flatten()
        proposal_overlaps = box_iou(
            decoded_boxes_cpu[positive_idx].unsqueeze(0),
            decoded_boxes_cpu[duplicate_indices],
        ).squeeze(0)
        positive_gt_iou = box_iou(
            decoded_boxes_cpu[positive_idx].unsqueeze(0),
            gt_boxes[gt_idx].unsqueeze(0),
        ).squeeze(0)
        duplicate_gt_iou = box_iou(decoded_boxes_cpu[duplicate_indices], gt_boxes[gt_idx].unsqueeze(0)).squeeze(1)
        valid = (proposal_overlaps >= float(nms_iou_threshold)) & (
            positive_gt_iou > duplicate_gt_iou + float(min_iou_gap)
        )
        if bool(require_suppressor_score_ge_candidate):
            valid = valid & (scores_cpu[duplicate_indices] >= scores_cpu[positive_idx])
        if not valid.any():
            continue

        valid_duplicate_indices = duplicate_indices[valid]
        valid_scores = scores_cpu[valid_duplicate_indices]
        best_duplicate = valid_duplicate_indices[int(torch.argmax(valid_scores).item())]
        positive_indices.append(int(positive_idx))
        negative_indices.append(int(best_duplicate.item()))

    if not positive_indices:
        empty = torch.empty((0,), dtype=torch.long, device=device)
        return empty, empty, {"same_gt_proposal_pair_count": 0}
    return (
        torch.tensor(positive_indices, dtype=torch.long, device=device),
        torch.tensor(negative_indices, dtype=torch.long, device=device),
        {"same_gt_proposal_pair_count": len(positive_indices)},
    )


def same_gt_duplicate_ranking_loss(
    candidate_logits: torch.Tensor,
    suppressor_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    margin: float,
    detach_suppressor: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    if candidate_logits.numel() == 0 or suppressor_logits.numel() == 0:
        zero = candidate_logits.sum() * 0.0 + suppressor_logits.sum() * 0.0
        return zero, {
            "same_gt_duplicate_ranking_pair_count": 0,
            "same_gt_duplicate_ranking_active_count": 0,
            "same_gt_duplicate_ranking_loss": 0.0,
        }
    labels = labels.to(candidate_logits.device).long().clamp(min=0, max=candidate_logits.shape[1] - 1)
    row = torch.arange(labels.numel(), device=candidate_logits.device)
    candidate_scores = candidate_logits[row, labels]
    suppressor_scores = suppressor_logits.to(candidate_logits.device)[row, labels]
    if bool(detach_suppressor):
        suppressor_scores = suppressor_scores.detach()
    penalties = F.relu(suppressor_scores + float(margin) - candidate_scores)
    loss = penalties.mean()
    return loss, {
        "same_gt_duplicate_ranking_pair_count": int(penalties.numel()),
        "same_gt_duplicate_ranking_active_count": int((penalties > 0).sum().item()),
        "same_gt_duplicate_ranking_loss": float(loss.detach().cpu().item()),
    }


def _json_safe_args_dict(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, Path):
            out[k] = str(v)
        else:
            out[k] = v
    return out


def _compute_and_store_interpretable_model(
    *,
    reference_stats: dict,
    n_total: int,
    all_proposals: np.ndarray,
    all_low_conf: np.ndarray,
    all_best_iou: np.ndarray,
    all_best_labels: np.ndarray,
    all_image_ids: np.ndarray,
    interpretable_images: list[torch.Tensor],
    interpretable_proposals: list[torch.Tensor],
    image_infos: dict,
    coco: dict,
    aspect_stats: dict,
    args,
    signal_device: torch.device,
    data_root: Path,
    max_size: int,
) -> None:
    import scripts.diagnose_interpretable_reward_signals as _diag

    arrays = {
        "proposal_boxes": all_proposals,
        "label_probs": all_low_conf,
        "best_iou": all_best_iou,
        "labels": all_best_labels,
        "image_ids": all_image_ids,
        "class_ids": all_best_labels,
    }
    train_signals = _diag.compute_image_signals(
        arrays=arrays,
        coco_infos=image_infos,
        data_root=data_root,
        max_size=max_size,
        device=signal_device,
    )
    train_signals["aspect_ratio_plausibility"] = _diag.aspect_plausibility(
        all_proposals, all_best_labels, aspect_stats
    )
    train_signals["nms_survivor_density"] = _diag.nms_support_density(
        all_proposals, all_best_labels, all_image_ids
    )

    new7_names = [
        "boundary_phase_coherence",
        "interior_exterior_texture_contrast",
        "aspect_ratio_plausibility",
        "multi_scale_saliency_consistency",
        "score_edge_alignment",
        "nms_survivor_density",
        "activation_centroid_consistency",
    ]

    fusion = str(args.interpretable_signal_fusion)
    signal_names: list[str] = []

    if fusion == "new7":
        mat = np.stack([train_signals[n] for n in new7_names], axis=1)
        signal_names = list(new7_names)
    elif fusion == "new7_raw_ifft_recipe":
        raw3 = _diag.load_legacy_feature_matrix(
            np.load(str(Path("runs/round2150_raw_ifft_legacy_full_ap75/candidate_raw_ifft_features.npz"))),
            "train",
            _diag.RAW_IFFT_REFERENCE_FEATURES,
            np.arange(n_total),
        )
        raw_scorer = _diag.fit_train_effect_scorer(
            raw3, all_best_labels.astype(bool), method="train_effect_sum"
        )
        raw_score = raw_scorer.score(raw3)
        mat = np.column_stack([
            np.stack([train_signals[n] for n in new7_names], axis=1),
            raw_score,
        ])
        signal_names = list(new7_names) + ["reference_raw_ifft_recipe"]
    elif fusion == "new7_raw_ifft_individual3":
        raw3 = _diag.load_legacy_feature_matrix(
            np.load(str(Path("runs/round2150_raw_ifft_legacy_full_ap75/candidate_raw_ifft_features.npz"))),
            "train",
            _diag.RAW_IFFT_REFERENCE_FEATURES,
            np.arange(n_total),
        )
        mat = np.column_stack([
            np.stack([train_signals[n] for n in new7_names], axis=1),
            raw3,
        ])
        signal_names = list(new7_names) + list(_diag.RAW_IFFT_REFERENCE_FEATURES)
    else:
        raw3 = _diag.load_legacy_feature_matrix(
            np.load(str(Path("runs/round2150_raw_ifft_legacy_full_ap75/candidate_raw_ifft_features.npz"))),
            "train",
            _diag.RAW_IFFT_REFERENCE_FEATURES,
            np.arange(n_total),
        )
        legacy_all, legacy_names = _diag.load_all_legacy_feature_matrix(
            np.load(str(Path("runs/round2150_raw_ifft_legacy_full_ap75/candidate_raw_ifft_features.npz"))),
            "train",
            np.arange(n_total),
        )
        mat = np.column_stack([
            np.stack([train_signals[n] for n in new7_names], axis=1),
            raw3,
            legacy_all,
        ])
        signal_names = list(new7_names) + list(_diag.RAW_IFFT_REFERENCE_FEATURES) + legacy_names

    scaler = StandardScaler()
    mat_z = scaler.fit_transform(mat)
    nan_col_mask = ~np.isfinite(mat_z).any(axis=0)
    if nan_col_mask.any():
        mat_z[:, nan_col_mask] = 0.0

    target = str(args.interpretable_residual_target)
    if target == "rank_diff":
        rank_best = _diag.rankdata_simple(all_best_iou)
        rank_conf = _diag.rankdata_simple(all_low_conf)
        y = rank_best - rank_conf
        model = Ridge(alpha=1.0, random_state=42).fit(mat_z, y)
        model_type = "ridge"
    elif target == "iou_residual":
        y = all_best_iou - all_low_conf
        model = Ridge(alpha=1.0, random_state=42).fit(mat_z, y)
        model_type = "ridge"
    elif target == "rank_underestimation":
        rank_best = _diag.rankdata_simple(all_best_iou)
        rank_conf = _diag.rankdata_simple(all_low_conf)
        y = (rank_best > rank_conf).astype(np.int32)
        model = LogisticRegression(class_weight="balanced", C=0.25, solver="liblinear", random_state=42, max_iter=1000).fit(mat_z, y)
        model_type = "logistic"
    else:
        raise ValueError(f"Unknown interpretable residual target: {target}")

    reference_stats["interpretable_signal_names"] = signal_names
    reference_stats["interpretable_scaler_mean"] = scaler.mean_.astype(float).tolist()
    reference_stats["interpretable_scaler_scale"] = scaler.scale_.astype(float).tolist()
    reference_stats["interpretable_model_type"] = model_type
    if model_type == "ridge":
        reference_stats["interpretable_coef"] = model.coef_.astype(float).tolist()
        reference_stats["interpretable_intercept"] = float(model.intercept_)
    else:
        reference_stats["interpretable_coef"] = model.coef_.astype(float).tolist()
        reference_stats["interpretable_intercept"] = float(model.intercept_[0])
    reference_stats["interpretable_target"] = target
    reference_stats["interpretable_n_train"] = n_total


def compute_rescue_verifier_scores(
    image: torch.Tensor,
    proposals: torch.Tensor,
    box_features: torch.Tensor,
    reference_features: torch.Tensor | ManifoldGateReference,
    reference_stats: dict[str, float],
    args,
    proposal_labels: torch.Tensor | None = None,
    proposal_label_probs: torch.Tensor | None = None,
) -> torch.Tensor | None:
    mode = str(args.rescue_verifier_mode)
    if mode == "none" or proposals.numel() == 0:
        return None
    device = box_features.device
    if mode in {"raw_ifft", "raw_ifft_hd_fusion"}:
        parsed_specs = [
            (int(item["feature_index"]), int(item["crop_size"]), str(item["spec"]))
            for item in reference_stats.get("raw_ifft_parsed_features", [])
        ]
        metric_bank = {}
        for crop_size in sorted({crop_size for _, crop_size, _ in parsed_specs}):
            crops = crop_and_resize_boxes(image.to(device), proposals.to(device), crop_size=int(crop_size))
            metric_bank[int(crop_size)] = penn_fudan_legacy_ifft_metric_bank(crops)
        raw_scores = score_legacy_ifft_metric_bank(
            metric_bank,
            parsed_specs,
            mean=torch.tensor(reference_stats["raw_ifft_scaler_mean"], dtype=box_features.dtype, device=device),
            scale=torch.tensor(reference_stats["raw_ifft_scaler_scale"], dtype=box_features.dtype, device=device),
            weights=torch.tensor(reference_stats["raw_ifft_weights"], dtype=box_features.dtype, device=device),
            threshold=0.0 if mode == "raw_ifft_hd_fusion" else float(reference_stats["raw_ifft_threshold"]),
        )
        if mode == "raw_ifft":
            return raw_scores
        if proposal_label_probs is None:
            raise ValueError("proposal_label_probs is required for raw_ifft_hd_fusion verifier")
        hd_features = _apply_hd_projection(box_features.detach(), reference_stats)
        hd_scores = _apply_linear_scorer(hd_features, reference_stats, prefix="hd")
        fused_scores = _apply_fusion_scorer(
            [raw_scores, hd_scores, proposal_label_probs.to(device=device, dtype=box_features.dtype)],
            reference_stats,
        )
        return fused_scores - float(reference_stats["fusion_threshold"])
    if mode == "raw_ifft_scene":
        if proposal_labels is None:
            raise ValueError("proposal_labels is required for raw_ifft_scene verifier")
        scene_groups = list(reference_stats.get("raw_ifft_scene_groups", []))
        crop_sizes = sorted(
            {
                int(item["crop_size"])
                for group in scene_groups
                for item in group.get("parsed_features", [])
            }
        )
        if not crop_sizes:
            return torch.full((proposals.shape[0],), -1.0e6, dtype=box_features.dtype, device=device)
        metric_bank = {}
        for crop_size in crop_sizes:
            crops = crop_and_resize_boxes(image.to(device), proposals.to(device), crop_size=int(crop_size))
            metric_bank[int(crop_size)] = penn_fudan_legacy_ifft_metric_bank(crops)
        return score_scene_legacy_ifft_metric_bank(
            metric_bank,
            proposal_labels.to(device),
            scene_groups,
            fallback_score=-1.0e6,
        ).to(device=device, dtype=box_features.dtype)
    if mode == "interpretable_residual":
        signal_names = list(reference_stats.get("interpretable_signal_names", []))
        if not signal_names:
            return torch.zeros((proposals.shape[0],), dtype=box_features.dtype, device=device)
        scaler_mean = np.asarray(reference_stats["interpretable_scaler_mean"], dtype=np.float64)
        scaler_scale = np.asarray(reference_stats["interpretable_scaler_scale"], dtype=np.float64)
        model_type = str(reference_stats.get("interpretable_model_type", "ridge"))
        coef = np.asarray(reference_stats["interpretable_coef"], dtype=np.float64)
        intercept = float(reference_stats.get("interpretable_intercept", 0.0))

        import scripts.diagnose_interpretable_reward_signals as _diag

        signal_device = torch.device(str(args.interpretable_signal_device))
        image_t = image.to(signal_device)
        gray = image_t.mean(dim=0)
        edge = _diag.robust_normalize_map(_diag.sobel_map(gray))
        phase_edge = _diag.robust_normalize_map(_diag.phase_only_edge_map(gray))
        saliency = _diag.robust_normalize_map(edge + phase_edge)
        multiscale_edge_maps = _diag.build_multiscale_edge_maps(gray)

        # aspect stats from reference_stats or compute on the fly
        coco = json.loads(Path(ANNOT).read_text(encoding="utf-8"))
        train_image_ids = set(
            int(img["id"])
            for img in coco["images"]
            if (DATA / "positive image set" / img["file_name"]).exists()
        )
        aspect_stats = _diag.compute_gt_aspect_stats(coco, train_image_ids)

        prop = proposals.to(signal_device)
        prop_cpu = prop.cpu().numpy()
        n = int(prop.shape[0])

        signals_dict: dict[str, np.ndarray] = {}
        needed = set(signal_names)
        for name in needed:
            signals_dict[name] = np.zeros((n,), dtype=np.float64)

        for i in range(n):
            box = prop_cpu[i]
            if "boundary_phase_coherence" in needed:
                pb, pi = _diag.boundary_and_interior_means(phase_edge, box)
                signals_dict["boundary_phase_coherence"][i] = pb / (pi + _diag.EPS)
            if "interior_exterior_texture_contrast" in needed:
                inside_edge = _diag.crop_mean(edge, box)
                outside_edge = _diag.ring_mean(edge, box)
                signals_dict["interior_exterior_texture_contrast"][i] = abs(inside_edge - outside_edge) / (inside_edge + outside_edge + _diag.EPS)
            if "multi_scale_saliency_consistency" in needed:
                signals_dict["multi_scale_saliency_consistency"][i] = _diag.multiscale_saliency_score_from_maps(multiscale_edge_maps, box)
            if "activation_centroid_consistency" in needed:
                signals_dict["activation_centroid_consistency"][i] = _diag.centroid_consistency(saliency, box)

        if "score_edge_alignment" in needed:
            if proposal_label_probs is not None:
                label_probs_np = proposal_label_probs.detach().cpu().numpy()
            else:
                label_probs_np = np.zeros((n,), dtype=np.float64)
            for i in range(n):
                box = prop_cpu[i]
                eb, ei = _diag.boundary_and_interior_means(edge, box)
                signals_dict["score_edge_alignment"][i] = (eb / (ei + _diag.EPS)) * (1.0 - label_probs_np[i])

        if "aspect_ratio_plausibility" in needed:
            if proposal_labels is not None:
                class_ids_np = proposal_labels.detach().cpu().numpy()
            else:
                class_ids_np = np.ones((n,), dtype=np.int64)
            signals_dict["aspect_ratio_plausibility"] = _diag.aspect_plausibility(prop_cpu, class_ids_np, aspect_stats)

        if "nms_survivor_density" in needed:
            signals_dict["nms_survivor_density"] = _diag.nms_support_density(
                prop_cpu, np.ones((n,), dtype=np.int64), np.arange(n, dtype=np.int64)
            )

        mat = np.column_stack([signals_dict[name] for name in signal_names])
        mat_z = (mat - scaler_mean) / np.maximum(scaler_scale, 1e-12)

        if model_type == "ridge":
            raw_scores = mat_z @ coef + intercept
        else:
            logits = mat_z @ coef + intercept
            raw_scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -100, 100)))

        return torch.tensor(raw_scores, dtype=box_features.dtype, device=device)
    if mode in {"fft", "fft_manifold"}:
        fft_scores = compute_fft_action_quality(
            image,
            proposals.to(device).unsqueeze(1),
            crop_size=int(args.rescue_fft_crop_size),
        ).squeeze(1)
    else:
        fft_scores = torch.zeros((proposals.shape[0],), dtype=box_features.dtype, device=device)
    if mode in {"manifold", "fft_manifold"}:
        if isinstance(reference_features, ManifoldGateReference):
            if proposal_labels is None:
                proposal_labels = torch.ones((proposals.shape[0],), dtype=torch.long, device=device)
            manifold_query_features = select_manifold_feature_source(
                box_features.detach(),
                proposals.to(device),
                tuple(image.shape[-2:]),
                args,
            )
            manifold_scores = score_manifold_gate(
                reference_features,
                manifold_query_features,
                proposal_labels.to(device).long(),
                proposals.to(device),
                image_size=tuple(image.shape[-2:]),
                cfg=ManifoldGateConfig(
                    mode=str(args.rescue_manifold_score_mode),
                    k=int(args.rescue_manifold_k),
                    fp_weight=float(args.rescue_manifold_fp_weight),
                    hard_negative_weight=float(args.rescue_hard_negative_weight),
                    margin_weight=float(args.rescue_margin_weight),
                    use_bucket_thresholds=bool(args.rescue_use_bucket_thresholds),
                ),
            )
        else:
            manifold_query_features = project_manifold_features(
                select_manifold_feature_source(
                    box_features.detach(),
                    proposals.to(device),
                    tuple(image.shape[-2:]),
                    args,
                ),
                str(args.rescue_manifold_feature_projection),
            )
            manifold_scores = compute_manifold_action_quality(
                manifold_query_features.unsqueeze(1),
                reference_features,
                k=int(args.rescue_manifold_k),
            ).squeeze(1)
    else:
        manifold_scores = torch.zeros((proposals.shape[0],), dtype=box_features.dtype, device=device)

    if mode == "fft":
        fft_weight, manifold_weight = 1.0, 0.0
    elif mode == "manifold":
        fft_weight, manifold_weight = 0.0, 1.0
    else:
        fft_weight, manifold_weight = float(args.rescue_fft_weight), float(args.rescue_manifold_weight)
    combined_scores = combine_verifier_scores(
        fft_scores.detach(),
        manifold_scores.detach(),
        reference_stats,
        fft_weight=fft_weight,
        manifold_weight=manifold_weight,
    )
    if bool(getattr(args, "rescue_use_class_thresholds", False)) and proposal_labels is not None:
        thresholds = reference_stats.get("class_thresholds", {})
        if thresholds:
            label_list = proposal_labels.detach().cpu().long().tolist()
            threshold_values = combined_scores.new_tensor(
                [float(thresholds.get(str(int(label)), float(args.rescue_verifier_gate))) for label in label_list]
            )
            combined_scores = combined_scores - threshold_values
    return combined_scores


@torch.no_grad()
def build_rescue_reference(
    baseline_model,
    train_loader,
    device: torch.device,
    args,
) -> tuple[torch.Tensor | ManifoldGateReference, dict[str, float]]:
    rescue_cfg = rescue_config_from_args(args)
    baseline_model.eval()
    positive_features = []
    fallback_features = []
    reference_bank_features = []
    reference_bank_labels = []
    reference_bank_positive = []
    reference_bank_boxes = []
    reference_bank_low_conf_scores = []
    fft_values = []
    raw_feature_values = []
    raw_ifft_features = []
    raw_ifft_hd_features = []
    raw_ifft_label_probs = []
    raw_ifft_labels = []
    raw_ifft_scene_features: dict[str, list[np.ndarray]] = {}
    raw_ifft_scene_labels: dict[str, list[np.ndarray]] = {}
    hd_fusion_features = []
    hd_fusion_labels = []
    interpretable_images: list[torch.Tensor] = []
    interpretable_proposals: list[torch.Tensor] = []
    interpretable_label_probs: list[torch.Tensor] = []
    interpretable_best_iou: list[torch.Tensor] = []
    interpretable_best_labels: list[torch.Tensor] = []
    interpretable_image_ids: list[torch.Tensor] = []
    scene_group_cfgs = (
        resolve_scene_raw_ifft_groups(list(args.rescue_raw_ifft_scene_groups))
        if str(args.rescue_verifier_mode) == "raw_ifft_scene"
        else []
    )
    for group in scene_group_cfgs:
        raw_ifft_scene_features[str(group["name"])] = []
        raw_ifft_scene_labels[str(group["name"])] = []

    for images, targets in train_loader:
        device_images = [image.to(device) for image in images]
        rollout_outputs = baseline_model(device_images)
        for image, target, rollout in zip(images, targets, rollout_outputs):
            proposals, _ = select_rollout_proposals(rollout, args)
            if proposals.numel() == 0:
                continue
            class_logits, _, box_features, _, _ = extract_roi_outputs_and_features_for_boxes(
                baseline_model,
                [image.to(device)],
                [proposals],
            )
            manifold_features = select_manifold_feature_source(
                box_features.detach(),
                proposals.to(device),
                tuple(image.shape[-2:]),
                args,
            )
            projected_manifold_features = project_manifold_features(
                manifold_features,
                str(args.rescue_manifold_feature_projection),
            )
            gt_boxes = target.get("boxes", torch.empty((0, 4))).to(device)
            gt_labels = target.get("labels", torch.empty((0,), dtype=torch.long)).to(device)
            best_iou, best_labels = match_boxes_to_targets(proposals.to(device), gt_boxes, gt_labels)
            low_conf_scores = matched_label_probabilities(class_logits, best_labels)
            positive_mask = (best_iou >= float(rescue_cfg.high_iou_min)) & (best_labels > 0)
            negative_mask = (best_iou <= float(rescue_cfg.low_iou_max)) & (best_labels > 0)
            if str(args.rescue_verifier_mode) in {"raw_ifft", "raw_ifft_hd_fusion", "raw_ifft_scene"}:
                candidate_mask = (
                    (low_conf_scores <= float(rescue_cfg.low_conf_max))
                    & (best_labels > 0)
                    & (positive_mask | negative_mask)
                )
                if str(args.rescue_verifier_mode) == "raw_ifft_hd_fusion":
                    hd_candidate_mask = (
                        (low_conf_scores <= float(rescue_cfg.low_conf_max))
                        & (positive_mask | negative_mask)
                    )
                    hd_candidate_mask = hd_candidate_mask | (
                        (low_conf_scores <= float(rescue_cfg.low_conf_max))
                        & (best_iou <= float(rescue_cfg.low_iou_max))
                    )
                    if hd_candidate_mask.any():
                        hd_fusion_features.append(
                            F.normalize(box_features.detach(), p=2, dim=1)[hd_candidate_mask].detach().cpu().numpy()
                        )
                        hd_fusion_labels.append(positive_mask[hd_candidate_mask].detach().cpu().numpy().astype(bool))
                if str(args.rescue_verifier_mode) == "raw_ifft_scene" and candidate_mask.any():
                    for group in scene_group_cfgs:
                        group_name = str(group["name"])
                        group_features = list(group["features"])
                        group_mask = candidate_mask & mask_for_class_ids(best_labels, [int(c) for c in group["classes"]])
                        if not group_features or not group_mask.any():
                            continue
                        parsed_specs = parse_legacy_ifft_feature_specs(group_features)
                        metric_bank = {}
                        for crop_size in sorted({crop_size for _, crop_size, _ in parsed_specs}):
                            raw_crops = crop_and_resize_boxes(
                                image.to(device),
                                proposals.to(device),
                                crop_size=int(crop_size),
                            )
                            metric_bank[int(crop_size)] = penn_fudan_legacy_ifft_metric_bank(raw_crops)
                        raw_columns = [
                            metric_bank[int(crop_size)][:, int(feature_index)]
                            for feature_index, crop_size, _ in parsed_specs
                        ]
                        raw_matrix = torch.stack(raw_columns, dim=1)
                        raw_ifft_scene_features[group_name].append(raw_matrix[group_mask].detach().cpu().numpy())
                        raw_ifft_scene_labels[group_name].append(
                            positive_mask[group_mask].detach().cpu().numpy().astype(bool)
                        )
                if str(args.rescue_verifier_mode) in {"raw_ifft", "raw_ifft_hd_fusion"} and candidate_mask.any():
                    parsed_specs = parse_legacy_ifft_feature_specs(list(args.rescue_raw_ifft_features))
                    metric_bank = {}
                    for crop_size in sorted({crop_size for _, crop_size, _ in parsed_specs}):
                        raw_crops = crop_and_resize_boxes(image.to(device), proposals.to(device), crop_size=int(crop_size))
                        metric_bank[int(crop_size)] = penn_fudan_legacy_ifft_metric_bank(raw_crops)
                    raw_columns = [
                        metric_bank[int(crop_size)][:, int(feature_index)]
                        for feature_index, crop_size, _ in parsed_specs
                    ]
                    raw_matrix = torch.stack(raw_columns, dim=1)
                    raw_ifft_features.append(raw_matrix[candidate_mask].detach().cpu().numpy())
                    if str(args.rescue_verifier_mode) == "raw_ifft_hd_fusion":
                        raw_ifft_hd_features.append(
                            F.normalize(box_features.detach(), p=2, dim=1)[candidate_mask].detach().cpu().numpy()
                        )
                        raw_ifft_label_probs.append(low_conf_scores[candidate_mask].detach().cpu().numpy())
                    raw_ifft_labels.append(positive_mask[candidate_mask].detach().cpu().numpy().astype(bool))
            if str(args.rescue_verifier_mode) == "interpretable_residual":
                interpretable_images.append(image.detach().cpu())
                interpretable_proposals.append(proposals.detach().cpu())
                interpretable_label_probs.append(low_conf_scores.detach().cpu())
                interpretable_best_iou.append(best_iou.detach().cpu())
                interpretable_best_labels.append(best_labels.detach().cpu())
                interpretable_image_ids.append(target["image_id"].detach().cpu())
            if positive_mask.any():
                positive_features.append(projected_manifold_features[positive_mask].detach())
            reference_mask = positive_mask | negative_mask
            if reference_mask.any():
                reference_bank_features.append(manifold_features[reference_mask].detach())
                reference_bank_labels.append(best_labels[reference_mask].detach())
                reference_bank_positive.append(positive_mask[reference_mask].detach())
                reference_bank_boxes.append(proposals.to(device)[reference_mask].detach())
                reference_bank_low_conf_scores.append(low_conf_scores[reference_mask].detach())
            fallback_features.append(projected_manifold_features.detach())
            raw_feature_values.append(projected_manifold_features.detach())
            fft_values.append(
                compute_fft_action_quality(
                    image,
                    proposals.to(device).unsqueeze(1),
                    crop_size=int(args.rescue_fft_crop_size),
                ).squeeze(1)
            )

    if positive_features:
        reference_features = torch.cat(positive_features, dim=0)
    elif fallback_features:
        reference_features = torch.cat(fallback_features, dim=0)
    else:
        feature_dim = int(baseline_model.roi_heads.box_predictor.bbox_pred.in_features)
        reference_features = torch.zeros((1, feature_dim), device=device)

    if raw_feature_values:
        all_features = torch.cat(raw_feature_values, dim=0)
        manifold_values = compute_manifold_action_quality(
            all_features.unsqueeze(1),
            reference_features,
            k=int(args.rescue_manifold_k),
        ).squeeze(1)
    else:
        manifold_values = torch.zeros((1,), device=device)
    all_fft = torch.cat(fft_values, dim=0) if fft_values else torch.zeros((1,), device=device)
    reference_stats = {
        "reference_feature_count": int(reference_features.shape[0]),
        "fft_mean": float(all_fft.mean().item()),
        "fft_std": float(all_fft.std(unbiased=False).clamp_min(1e-6).item()),
        "manifold_mean": float(manifold_values.mean().item()),
        "manifold_std": float(manifold_values.std(unbiased=False).clamp_min(1e-6).item()),
    }
    if str(args.rescue_verifier_mode) == "raw_ifft_scene":
        scene_group_stats = []
        for group in scene_group_cfgs:
            group_name = str(group["name"])
            group_features = raw_ifft_scene_features.get(group_name, [])
            group_labels = raw_ifft_scene_labels.get(group_name, [])
            parsed_specs = parse_legacy_ifft_feature_specs(list(group["features"]))
            if not parsed_specs or not group_features:
                scene_group_stats.append(
                    {
                        "name": group_name,
                        "classes": [int(c) for c in group["classes"]],
                        "features": list(group["features"]),
                        "parsed_features": [],
                        "enabled": False,
                        "reason": "no_features_or_candidates",
                    }
                )
                continue
            raw_train = np.concatenate(group_features, axis=0)
            raw_labels = np.concatenate(group_labels, axis=0).astype(bool)
            positive_count = int(raw_labels.sum())
            negative_count = int((~raw_labels).sum())
            if positive_count < int(args.rescue_raw_ifft_scene_min_positives) or negative_count <= 0:
                scene_group_stats.append(
                    {
                        "name": group_name,
                        "classes": [int(c) for c in group["classes"]],
                        "features": list(group["features"]),
                        "parsed_features": [
                            {"feature_index": int(feature_index), "crop_size": int(crop_size), "spec": str(spec)}
                            for feature_index, crop_size, spec in parsed_specs
                        ],
                        "candidate_count": int(raw_labels.shape[0]),
                        "positive_count": positive_count,
                        "negative_count": negative_count,
                        "enabled": False,
                        "reason": "insufficient_positive_or_negative_candidates",
                    }
                )
                continue
            scorer = fit_train_effect_scorer(raw_train, raw_labels, method=str(args.rescue_raw_ifft_score_method))
            raw_scores = scorer.score(raw_train)
            margin = float(np.std(raw_scores) * float(args.rescue_raw_ifft_margin_std_frac))
            calibration = calibrate_precision_threshold(
                raw_scores,
                raw_labels,
                target_precision=float(args.rescue_raw_ifft_scene_target_precision),
                margin=margin,
            )
            scene_group_stats.append(
                {
                    "name": group_name,
                    "classes": [int(c) for c in group["classes"]],
                    "features": list(group["features"]),
                    "parsed_features": [
                        {"feature_index": int(feature_index), "crop_size": int(crop_size), "spec": str(spec)}
                        for feature_index, crop_size, spec in parsed_specs
                    ],
                    "candidate_count": int(raw_labels.shape[0]),
                    "positive_count": positive_count,
                    "negative_count": negative_count,
                    "score_method": str(args.rescue_raw_ifft_score_method),
                    "target_precision": float(args.rescue_raw_ifft_scene_target_precision),
                    "margin": margin,
                    "threshold": float(calibration.threshold),
                    "calibration": {
                        "selected_prefix": int(calibration.selected_prefix),
                        "tp_prefix": int(calibration.tp_prefix),
                        "fp_prefix": int(calibration.fp_prefix),
                        "precision_prefix": float(calibration.precision_prefix),
                        "recall_prefix": float(calibration.recall_prefix),
                        "reason": str(calibration.reason),
                    },
                    "scaler_mean": scorer.scaler.mean_.astype(float).tolist(),
                    "scaler_scale": scorer.scaler.scale_.astype(float).tolist(),
                    "weights": scorer.weights.astype(float).tolist(),
                    "enabled": bool(np.isfinite(float(calibration.threshold))),
                    "reason": str(calibration.reason),
                }
            )
        reference_stats.update(
            {
                "raw_ifft_scene_groups": scene_group_stats,
                "raw_ifft_scene_group_names": list(args.rescue_raw_ifft_scene_groups),
                "raw_ifft_scene_target_precision": float(args.rescue_raw_ifft_scene_target_precision),
                "raw_ifft_scene_min_positives": int(args.rescue_raw_ifft_scene_min_positives),
                "fft_mean": 0.0,
                "fft_std": 1.0,
                "manifold_mean": 0.0,
                "manifold_std": 1.0,
                "manifold_gate_mode": "raw_ifft_scene",
            }
        )
        return reference_features.detach(), reference_stats
    if str(args.rescue_verifier_mode) in {"raw_ifft", "raw_ifft_hd_fusion"}:
        parsed_specs = parse_legacy_ifft_feature_specs(list(args.rescue_raw_ifft_features))
        if not raw_ifft_features:
            raise RuntimeError("raw_ifft verifier calibration found no LC-HI/LC-LI train candidates")
        raw_train = np.concatenate(raw_ifft_features, axis=0)
        raw_labels = np.concatenate(raw_ifft_labels, axis=0).astype(bool)
        scorer = fit_train_effect_scorer(raw_train, raw_labels, method=str(args.rescue_raw_ifft_score_method))
        raw_scores = scorer.score(raw_train)
        margin = float(np.std(raw_scores) * float(args.rescue_raw_ifft_margin_std_frac))
        calibration = calibrate_precision_threshold(
            raw_scores,
            raw_labels,
            target_precision=float(args.rescue_raw_ifft_target_precision),
            margin=margin,
        )
        reference_stats.update(
            {
                "raw_ifft_feature_specs": list(args.rescue_raw_ifft_features),
                "raw_ifft_parsed_features": [
                    {"feature_index": int(feature_index), "crop_size": int(crop_size), "spec": str(spec)}
                    for feature_index, crop_size, spec in parsed_specs
                ],
                "raw_ifft_candidate_count": int(raw_labels.shape[0]),
                "raw_ifft_positive_count": int(raw_labels.sum()),
                "raw_ifft_negative_count": int((~raw_labels).sum()),
                "raw_ifft_score_method": str(args.rescue_raw_ifft_score_method),
                "raw_ifft_target_precision": float(args.rescue_raw_ifft_target_precision),
                "raw_ifft_margin_std_frac": float(args.rescue_raw_ifft_margin_std_frac),
                "raw_ifft_margin": margin,
                "raw_ifft_threshold": float(calibration.threshold),
                "raw_ifft_calibration": {
                    "selected_prefix": int(calibration.selected_prefix),
                    "tp_prefix": int(calibration.tp_prefix),
                    "fp_prefix": int(calibration.fp_prefix),
                    "precision_prefix": float(calibration.precision_prefix),
                    "recall_prefix": float(calibration.recall_prefix),
                    "reason": str(calibration.reason),
                },
                "raw_ifft_scaler_mean": scorer.scaler.mean_.astype(float).tolist(),
                "raw_ifft_scaler_scale": scorer.scaler.scale_.astype(float).tolist(),
                "raw_ifft_weights": scorer.weights.astype(float).tolist(),
                "fft_mean": 0.0,
                "fft_std": 1.0,
                "manifold_mean": 0.0,
                "manifold_std": 1.0,
                "manifold_gate_mode": "raw_ifft",
            }
        )
        if str(args.rescue_verifier_mode) == "raw_ifft_hd_fusion":
            if not raw_ifft_hd_features or not raw_ifft_label_probs:
                raise RuntimeError("raw_ifft_hd_fusion verifier calibration found no high-dimensional train candidates")
            if not hd_fusion_features:
                raise RuntimeError("raw_ifft_hd_fusion verifier calibration found no full low-conf HD train candidates")
            hd_train = np.concatenate(hd_fusion_features, axis=0).astype(np.float64)
            hd_labels = np.concatenate(hd_fusion_labels, axis=0).astype(bool)
            hd_query = np.concatenate(raw_ifft_hd_features, axis=0).astype(np.float64)
            label_prob_train = np.concatenate(raw_ifft_label_probs, axis=0).astype(np.float64)
            hd_scaler = StandardScaler().fit(hd_train)
            hd_train_z = hd_scaler.transform(hd_train)
            hd_query_z = hd_scaler.transform(hd_query)
            pca_components = min(int(args.rescue_hd_fusion_pca_components), hd_train_z.shape[0] - 1, hd_train_z.shape[1])
            if pca_components > 0:
                hd_pca = PCA(n_components=pca_components, whiten=True, random_state=42).fit(hd_train_z)
                hd_projected = hd_pca.transform(hd_train_z)
                hd_query_projected = hd_pca.transform(hd_query_z)
                pca_components_matrix = hd_pca.components_.astype(float).tolist()
                pca_whiten_scale = np.sqrt(hd_pca.explained_variance_).astype(float).tolist()
                pca_explained = float(hd_pca.explained_variance_ratio_.sum())
                pca_mean = hd_pca.mean_.astype(float).tolist()
            else:
                hd_pca = None
                hd_projected = hd_train_z
                hd_query_projected = hd_query_z
                pca_components_matrix = []
                pca_whiten_scale = []
                pca_explained = 0.0
                pca_mean = []
            if str(args.rescue_hd_fusion_hd_scorer) == "logistic":
                _, hd_scores, hd_model = _fit_logistic_scores(hd_projected, hd_labels, hd_query_projected)
                hd_stats = {
                    "hd_scorer": "logistic",
                    "hd_coef": hd_model["coef"][0],
                    "hd_intercept": hd_model["intercept"][0],
                }
            elif str(args.rescue_hd_fusion_hd_scorer) == "center":
                _, hd_scores, hd_model = _fit_center_scores(hd_projected, hd_labels, hd_query_projected)
                hd_stats = {
                    "hd_scorer": "center",
                    "hd_positive_center": hd_model["positive_center"],
                    "hd_negative_center": hd_model["negative_center"],
                }
            else:
                raise ValueError(f"Unknown rescue_hd_fusion_hd_scorer: {args.rescue_hd_fusion_hd_scorer}")
            raw_scores_no_threshold = scorer.score(raw_train)
            fusion_columns = [raw_scores_no_threshold, hd_scores, label_prob_train]
            fusion_scores, fusion_model = _fit_fusion_scorer(
                fusion_columns,
                raw_labels,
                method=str(args.rescue_hd_fusion_method),
            )
            fusion_margin = float(np.std(fusion_scores) * float(args.rescue_raw_ifft_margin_std_frac))
            fusion_calibration = calibrate_precision_threshold(
                fusion_scores,
                raw_labels,
                target_precision=float(args.rescue_raw_ifft_target_precision),
                margin=fusion_margin,
            )
            reference_stats.update(
                {
                    "manifold_gate_mode": "raw_ifft_hd_fusion",
                    "hd_use_l2": True,
                    "hd_pca_components": int(pca_components),
                    "hd_pca_whiten": True,
                    "hd_pca_explained_variance": pca_explained,
                    "hd_candidate_count": int(hd_labels.shape[0]),
                    "hd_positive_count": int(hd_labels.sum()),
                    "hd_negative_count": int((~hd_labels).sum()),
                    "hd_scaler_mean": hd_scaler.mean_.astype(float).tolist(),
                    "hd_scaler_scale": hd_scaler.scale_.astype(float).tolist(),
                    "hd_pca_mean": pca_mean,
                    "hd_pca_components_matrix": pca_components_matrix,
                    "hd_pca_whiten_scale": pca_whiten_scale,
                    "fusion_method": str(fusion_model["method"]),
                    "fusion_column_means": fusion_model["column_means"],
                    "fusion_column_stds": fusion_model["column_stds"],
                    "fusion_threshold": float(fusion_calibration.threshold),
                    "fusion_target_precision": float(args.rescue_raw_ifft_target_precision),
                    "fusion_margin": fusion_margin,
                    "fusion_calibration": {
                        "selected_prefix": int(fusion_calibration.selected_prefix),
                        "tp_prefix": int(fusion_calibration.tp_prefix),
                        "fp_prefix": int(fusion_calibration.fp_prefix),
                        "precision_prefix": float(fusion_calibration.precision_prefix),
                        "recall_prefix": float(fusion_calibration.recall_prefix),
                        "reason": str(fusion_calibration.reason),
                    },
                    **hd_stats,
                }
            )
            if str(fusion_model["method"]) == "train_effect":
                reference_stats.update(
                    {
                        "fusion_scaler_mean": fusion_model["scaler_mean"],
                        "fusion_scaler_scale": fusion_model["scaler_scale"],
                        "fusion_weights": fusion_model["weights"],
                    }
                )
            elif str(fusion_model["method"]) == "logistic":
                reference_stats.update(
                    {
                        "fusion_coef": fusion_model["coef"][0],
                        "fusion_intercept": fusion_model["intercept"][0],
                    }
                )
        return reference_features.detach(), reference_stats
    if str(args.rescue_verifier_mode) == "interpretable_residual" and interpretable_images:
        signal_device = torch.device(str(args.interpretable_signal_device))
        _diag = sys.modules.get("scripts.diagnose_interpretable_reward_signals")
        if _diag is None:
            import scripts.diagnose_interpretable_reward_signals as _diag

        coco = json.loads(Path(ANNOT).read_text(encoding="utf-8"))
        image_infos = {int(info["id"]): info for info in coco["images"]}

        all_image_ids_np = torch.cat(interpretable_image_ids, dim=0).numpy()
        train_image_ids_set = set(int(v) for v in np.unique(all_image_ids_np))
        aspect_stats = _diag.compute_gt_aspect_stats(coco, train_image_ids_set)

        all_proposals_np = torch.cat(interpretable_proposals, dim=0).numpy()
        all_low_conf_np = torch.cat(interpretable_label_probs, dim=0).numpy()
        all_best_iou_np = torch.cat(interpretable_best_iou, dim=0).numpy()
        all_best_labels_np = torch.cat(interpretable_best_labels, dim=0).numpy()
        n_total = int(all_proposals_np.shape[0])

        _compute_and_store_interpretable_model(
            reference_stats=reference_stats,
            n_total=n_total,
            all_proposals=all_proposals_np,
            all_low_conf=all_low_conf_np,
            all_best_iou=all_best_iou_np,
            all_best_labels=all_best_labels_np,
            all_image_ids=all_image_ids_np,
            interpretable_images=interpretable_images,
            interpretable_proposals=interpretable_proposals,
            image_infos=image_infos,
            coco=coco,
            aspect_stats=aspect_stats,
            args=args,
            signal_device=signal_device,
            data_root=DATA,
            max_size=int(args.interpretable_signal_max_size),
        )
        return reference_features.detach(), reference_stats
        bank_features = torch.cat(reference_bank_features, dim=0)
        bank_labels = torch.cat(reference_bank_labels, dim=0)
        bank_positive = torch.cat(reference_bank_positive, dim=0)
        bank_boxes = torch.cat(reference_bank_boxes, dim=0)
        bank_low_conf_scores = torch.cat(reference_bank_low_conf_scores, dim=0)
        improved_reference = build_manifold_gate_reference(
            bank_features,
            bank_labels,
            bank_positive,
            bank_boxes,
            image_size=(MAX_SIZE, MAX_SIZE),
            num_classes=NUM_CLASSES,
            feature_projection=str(args.rescue_manifold_feature_projection),
        )
        improved_scores = score_manifold_gate(
            improved_reference,
            bank_features,
            bank_labels,
            bank_boxes,
            image_size=(MAX_SIZE, MAX_SIZE),
            cfg=ManifoldGateConfig(
                mode=str(args.rescue_manifold_score_mode),
                k=int(args.rescue_manifold_k),
                fp_weight=float(args.rescue_manifold_fp_weight),
                hard_negative_weight=float(args.rescue_hard_negative_weight),
                margin_weight=float(args.rescue_margin_weight),
                use_bucket_thresholds=bool(args.rescue_use_bucket_thresholds),
            ),
        )
        improved_mean = float(improved_scores.mean().item())
        improved_std = float(improved_scores.std(unbiased=False).clamp_min(1e-6).item())
        standardized_scores = (improved_scores - improved_mean) / improved_std
        class_thresholds, class_threshold_diagnostics = calibrate_classwise_thresholds(
            standardized_scores,
            bank_labels,
            torch.zeros_like(standardized_scores),
            torch.where(
                bank_positive,
                torch.full_like(standardized_scores, float(rescue_cfg.high_iou_min)),
                torch.zeros_like(standardized_scores),
            ),
            rescue_cfg,
            min_precision=float(args.rescue_class_threshold_min_precision),
            fallback_threshold=float(args.rescue_verifier_gate),
            min_positives=int(args.rescue_class_threshold_min_positives),
            min_threshold=(
                float(args.rescue_class_threshold_min_threshold)
                if args.rescue_class_threshold_min_threshold is not None
                else None
            ),
            low_conf_scores=bank_low_conf_scores,
        )
        reference_stats.update(
            {
                "manifold_gate_mode": "improved",
                "manifold_feature_source": str(args.rescue_manifold_feature_source),
                "manifold_feature_projection": str(args.rescue_manifold_feature_projection),
                "manifold_reference_count": int(bank_features.shape[0]),
                "manifold_reference_positive_count": int(bank_positive.sum().item()),
                "manifold_threshold_count": len(improved_reference.thresholds),
                "manifold_mean": improved_mean,
                "manifold_std": improved_std,
                "class_thresholds": {str(key): float(value) for key, value in class_thresholds.items()},
                "class_threshold_diagnostics": {
                    str(key): {metric: float(metric_value) for metric, metric_value in value.items()}
                    for key, value in class_threshold_diagnostics.items()
                },
            }
        )
        return improved_reference, reference_stats
    reference_stats["manifold_gate_mode"] = "legacy"
    reference_stats["manifold_feature_source"] = str(args.rescue_manifold_feature_source)
    reference_stats["manifold_feature_projection"] = str(args.rescue_manifold_feature_projection)
    return reference_features.detach(), reference_stats


@torch.no_grad()
def collect_rescue_diagnostics(
    model,
    baseline_model,
    loader,
    device: torch.device,
    args,
    reference_features: torch.Tensor,
    reference_stats: dict[str, float],
) -> dict[str, float]:
    rescue_cfg = rescue_config_from_args(args)
    model.eval()
    baseline_model.eval()
    totals: dict[str, float] = {
        "proposal_count": 0,
        "low_conf_high_iou_count": 0,
        "high_conf_low_iou_count": 0,
        "low_conf_low_iou_count": 0,
        "high_conf_high_iou_count": 0,
        "verifier_positive_lchi_count": 0,
        "verifier_positive_lcli_count": 0,
        "verifier_positive_low_conf_count": 0,
        "lchi_prob_delta_sum": 0.0,
        "verifier_positive_lchi_prob_delta_sum": 0.0,
        "lchi_verifier_sum": 0.0,
        "lchi_baseline_label_prob_sum": 0.0,
        "lchi_current_label_prob_sum": 0.0,
    }

    for images, targets in loader:
        device_images = [image.to(device) for image in images]
        rollout_outputs = baseline_model(device_images)
        for image, target, rollout in zip(images, targets, rollout_outputs):
            proposals, proposal_scores = select_rollout_proposals(rollout, args)
            if proposals.numel() == 0:
                continue
            class_logits, _, box_features, _, _ = extract_roi_outputs_and_features_for_boxes(
                model,
                [image.to(device)],
                [proposals],
            )
            baseline_logits, _, _, _, _ = extract_roi_outputs_and_features_for_boxes(
                baseline_model,
                [image.to(device)],
                [proposals],
            )
            gt_boxes = target.get("boxes", torch.empty((0, 4))).to(device)
            gt_labels = target.get("labels", torch.empty((0,), dtype=torch.long)).to(device)
            best_iou, best_labels = match_boxes_to_targets(proposals.to(device), gt_boxes, gt_labels)
            baseline_label_probs = matched_label_probabilities(baseline_logits, best_labels)
            summary = summarize_confidence_iou_regions(
                proposal_scores.to(device),
                best_iou,
                rescue_cfg,
                low_conf_scores=baseline_label_probs,
            )
            for key in [
                "proposal_count",
                "low_conf_high_iou_count",
                "high_conf_low_iou_count",
                "low_conf_low_iou_count",
                "high_conf_high_iou_count",
            ]:
                totals[key] += float(summary[key])

            lchi = (
                (baseline_label_probs <= float(rescue_cfg.low_conf_max))
                & (best_iou >= float(rescue_cfg.high_iou_min))
                & (best_labels > 0)
            )
            verifier_scores = compute_rescue_verifier_scores(
                image,
                proposals,
                box_features,
                reference_features,
                reference_stats,
                args,
                proposal_labels=best_labels,
                proposal_label_probs=baseline_label_probs,
            )
            verifier_positive = None
            if verifier_scores is not None:
                verifier_positive = lchi & (verifier_scores.to(device) >= float(args.rescue_verifier_gate))
                totals["verifier_positive_lchi_count"] += int(verifier_positive.sum().item())
                gate_summary = summarize_verifier_gate(
                    proposal_scores.to(device),
                    best_iou,
                    verifier_scores.to(device),
                    rescue_cfg,
                    threshold=float(args.rescue_verifier_gate),
                    low_conf_scores=baseline_label_probs,
                )
                totals["verifier_positive_lcli_count"] += float(gate_summary["gate_low_conf_low_iou_count"])
                totals["verifier_positive_low_conf_count"] += float(gate_summary["gate_low_conf_total_count"])
                if lchi.any():
                    totals["lchi_verifier_sum"] += float(verifier_scores.to(device)[lchi].sum().item())

            if lchi.any():
                labels = best_labels[lchi].long()
                row = torch.arange(labels.numel(), device=device)
                current_probs = F.softmax(class_logits[lchi], dim=1)[row, labels]
                baseline_probs = baseline_label_probs[lchi]
                totals["lchi_prob_delta_sum"] += float((current_probs - baseline_probs).sum().item())
                totals["lchi_baseline_label_prob_sum"] += float(baseline_probs.sum().item())
                totals["lchi_current_label_prob_sum"] += float(current_probs.sum().item())
                if verifier_positive is not None and verifier_positive.any():
                    gated_labels = best_labels[verifier_positive].long()
                    gated_row = torch.arange(gated_labels.numel(), device=device)
                    gated_current = F.softmax(class_logits[verifier_positive], dim=1)[gated_row, gated_labels]
                    gated_baseline = baseline_label_probs[verifier_positive]
                    totals["verifier_positive_lchi_prob_delta_sum"] += float((gated_current - gated_baseline).sum().item())

    lchi_count = max(1.0, totals["low_conf_high_iou_count"])
    gated_lchi_count = max(1.0, totals["verifier_positive_lchi_count"])
    totals["low_conf_high_iou_rate"] = totals["low_conf_high_iou_count"] / max(1.0, totals["proposal_count"])
    totals["high_conf_low_iou_rate"] = totals["high_conf_low_iou_count"] / max(1.0, totals["proposal_count"])
    totals["verifier_positive_lchi_rate"] = totals["verifier_positive_lchi_count"] / lchi_count
    totals["verifier_positive_lcli_rate"] = totals["verifier_positive_lcli_count"] / max(
        1.0,
        totals["low_conf_low_iou_count"],
    )
    totals["verifier_positive_low_conf_precision"] = totals["verifier_positive_lchi_count"] / max(
        1.0,
        totals["verifier_positive_lchi_count"] + totals["verifier_positive_lcli_count"],
    )
    totals["verifier_positive_low_conf_false_rescue_rate"] = totals["verifier_positive_lcli_count"] / max(
        1.0,
        totals["verifier_positive_low_conf_count"],
    )
    totals["lchi_prob_delta_mean"] = totals["lchi_prob_delta_sum"] / lchi_count
    totals["lchi_baseline_label_prob_mean"] = totals["lchi_baseline_label_prob_sum"] / lchi_count
    totals["lchi_current_label_prob_mean"] = totals["lchi_current_label_prob_sum"] / lchi_count
    totals["verifier_positive_lchi_prob_delta_mean"] = (
        totals["verifier_positive_lchi_prob_delta_sum"] / gated_lchi_count
    )
    totals["lchi_verifier_mean"] = totals["lchi_verifier_sum"] / lchi_count
    return totals


@torch.no_grad()
def collect_offline_verifier_report(
    model,
    loader,
    device: torch.device,
    args,
    reference_features: torch.Tensor | ManifoldGateReference,
    reference_stats: dict[str, float],
) -> dict[str, float]:
    rescue_cfg = rescue_config_from_args(args)
    model.eval()
    all_verifier_scores = []
    all_scores = []
    all_iou = []
    all_labels = []
    all_low_conf_scores = []
    for images, targets in loader:
        device_images = [image.to(device) for image in images]
        rollout_outputs = model(device_images)
        for image, target, rollout in zip(images, targets, rollout_outputs):
            proposals, proposal_scores = select_rollout_proposals(rollout, args)
            if proposals.numel() == 0:
                continue
            class_logits, _, box_features, _, _ = extract_roi_outputs_and_features_for_boxes(
                model,
                [image.to(device)],
                [proposals],
            )
            gt_boxes = target.get("boxes", torch.empty((0, 4))).to(device)
            gt_labels = target.get("labels", torch.empty((0,), dtype=torch.long)).to(device)
            best_iou, best_labels = match_boxes_to_targets(proposals.to(device), gt_boxes, gt_labels)
            low_conf_scores = matched_label_probabilities(class_logits, best_labels)
            verifier_scores = compute_rescue_verifier_scores(
                image,
                proposals,
                box_features,
                reference_features,
                reference_stats,
                args,
                proposal_labels=best_labels,
                proposal_label_probs=low_conf_scores,
            )
            if verifier_scores is None:
                continue
            all_verifier_scores.append(verifier_scores.detach().cpu())
            all_scores.append(proposal_scores.detach().cpu())
            all_iou.append(best_iou.detach().cpu())
            all_labels.append(best_labels.detach().cpu())
            all_low_conf_scores.append(low_conf_scores.detach().cpu())
    if not all_verifier_scores:
        return {
            "candidate_count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "auc": 0.0,
        }
    return evaluate_verifier_offline(
        torch.cat(all_verifier_scores, dim=0),
        torch.cat(all_labels, dim=0),
        torch.cat(all_scores, dim=0),
        torch.cat(all_iou, dim=0),
        rescue_cfg,
        threshold=float(args.rescue_verifier_gate),
        precision_targets=(0.7, 0.8, 0.9),
        low_conf_scores=torch.cat(all_low_conf_scores, dim=0),
    )


def train_one_epoch(
    model,
    baseline_model,
    train_loader,
    device: torch.device,
    args,
    reference_features: torch.Tensor,
    reference_stats: dict[str, float],
) -> dict:
    action_cfg = ActionVerifierConfig(
        num_samples=2,
        sigma=float(args.sigma),
        seed=42,
        include_identity_action=True,
    )
    rescue_cfg = rescue_config_from_args(args)
    optimizer = build_optimizer(model, args)
    total_loss = total_policy = total_kl = total_det = total_rescue = 0.0
    total_kl_cls = 0.0
    total_kl_box = 0.0
    total_pairwise_rescue = 0.0
    total_score_budget = 0.0
    total_bbox_rescue = 0.0
    total_confidence_crossing = 0.0
    total_class_margin = 0.0
    total_chain_bbox = 0.0
    total_chain_cls_margin = 0.0
    total_chain_ranking = 0.0
    total_verifier_ranking = 0.0
    total_nms_aware_ranking = 0.0
    total_blocked_nms = 0.0
    total_pre_nms_rescue = 0.0
    total_pre_nms_dpo = 0.0
    total_same_gt_duplicate_ranking = 0.0
    total_valid = total_actions = total_batches = 0
    total_rescue_positive = total_rescue_negative = 0
    total_pairwise_pairs = 0
    total_chain_candidate_count = 0
    total_chain_ranking_pairs = 0
    total_chain_ranking_active = 0
    total_verifier_ranking_pairs = 0
    total_verifier_ranking_active = 0
    total_verifier_ranking_positive = 0
    total_verifier_ranking_negative = 0
    total_nms_aware_pairs = 0
    total_nms_aware_active = 0
    total_blocked_nms_pairs = 0
    total_blocked_nms_active_rank = 0
    total_blocked_nms_active_crossing = 0
    total_blocked_nms_candidate_delta = 0.0
    total_blocked_nms_suppressor_delta = 0.0
    total_blocked_nms_relative_delta = 0.0
    total_pre_nms_rescue_count = 0
    total_pre_nms_rescue_active = 0
    total_pre_nms_rescue_prob_delta = 0.0
    total_pre_nms_rescue_score_cross = 0
    total_pre_nms_dpo_pairs = 0
    total_pre_nms_dpo_win = 0
    total_pre_nms_dpo_preference_margin = 0.0
    total_pre_nms_dpo_iou_gap = 0.0
    total_same_gt_duplicate_pairs = 0
    total_same_gt_duplicate_active = 0
    total_chain_score_threshold_cross = 0
    total_chain_low_conf_max_cross = 0
    total_chain_baseline_prob_sum = 0.0
    total_chain_current_prob_sum = 0.0
    total_chain_baseline_iou_sum = 0.0
    total_chain_current_iou_sum = 0.0
    total_score_budget_count = total_score_budget_violations = 0
    total_bbox_rescue_count = 0
    total_bbox_rescue_weight = 0.0
    total_confidence_crossing_count = 0
    total_confidence_crossing_active_count = 0
    total_class_margin_count = 0
    total_class_margin_active_count = 0
    total_lchi = total_hcli = 0
    confidence_rescue_effect_sums: dict[str, float] = {}
    grad_metric_sums: dict[str, float] = {}
    grad_metric_batches = 0
    num_classes = NUM_CLASSES

    for images, targets in train_loader:
        device_images = [image.to(device) for image in images]
        device_targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in target.items()} for target in targets]

        model.train()
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
        det_loss_dict = model(device_images, device_targets)
        det_loss = sum(det_loss_dict.values())
        model.eval()

        with torch.no_grad():
            rollout_outputs = baseline_model(device_images)
            rpn_proposals_by_image = (
                select_rpn_proposals_for_images(baseline_model, images, args, device)
                if str(args.proposal_source) == "rpn"
                else None
            )

        policy_batch_loss = None
        kl_batch_loss = None
        kl_cls_batch_loss = None
        kl_box_batch_loss = None
        rescue_batch_loss = None
        pairwise_batch_loss = None
        score_budget_batch_loss = None
        bbox_rescue_batch_loss = None
        confidence_crossing_batch_loss = None
        class_margin_batch_loss = None
        chain_bbox_batch_loss = None
        chain_cls_margin_batch_loss = None
        chain_ranking_batch_loss = None
        verifier_ranking_batch_loss = None
        nms_aware_batch_loss = None
        blocked_nms_batch_loss = None
        pre_nms_rescue_batch_loss = None
        pre_nms_dpo_batch_loss = None
        same_gt_duplicate_batch_loss = None
        pairwise_batch_metric = 0.0
        score_budget_batch_metric = 0.0
        bbox_rescue_batch_metric = 0.0
        confidence_crossing_batch_metric = 0.0
        class_margin_batch_metric = 0.0
        chain_bbox_batch_metric = 0.0
        chain_cls_margin_batch_metric = 0.0
        chain_ranking_batch_metric = 0.0
        verifier_ranking_batch_metric = 0.0
        nms_aware_batch_metric = 0.0
        blocked_nms_batch_metric = 0.0
        pre_nms_rescue_batch_metric = 0.0
        pre_nms_dpo_batch_metric = 0.0
        same_gt_duplicate_batch_metric = 0.0
        batch_valid = 0
        batch_actions = 0
        for image_idx, (image, target, rollout) in enumerate(zip(images, targets, rollout_outputs)):
            if str(args.proposal_source) == "rpn":
                proposals = rpn_proposals_by_image[image_idx] if rpn_proposals_by_image is not None else torch.empty((0, 4))
                proposal_scores = torch.zeros((proposals.shape[0],), dtype=torch.float32)
            else:
                proposals, proposal_scores = select_rollout_proposals(rollout, args)
            if proposals.numel() == 0:
                continue
            class_logits, box_regression, box_features, scaled_boxes, transformed_sizes = extract_roi_outputs_and_features_for_boxes(
                model,
                [image.to(device)],
                [proposals],
            )
            baseline_logits, baseline_box_regression, _, _ = extract_roi_head_outputs_for_boxes(
                baseline_model,
                [image.to(device)],
                [proposals],
            )
            mu = _person_box_deltas(box_regression, num_classes)
            baseline_mu = _person_box_deltas(baseline_box_regression, num_classes)
            action_batch = build_action_batch(proposals.to(device), mu, tuple(image.shape[-2:]), action_cfg)
            baseline_actions = build_action_batch(proposals.to(device), baseline_mu, tuple(image.shape[-2:]), action_cfg)
            gt_boxes = target.get("boxes", torch.empty((0, 4))).to(device)
            action_iou, _ = _match_action_iou(action_batch.decoded_boxes, gt_boxes)
            prop_iou = proposal_iou_for_scores(proposals.to(device), gt_boxes).unsqueeze(1)
            quality = torch.maximum(action_iou, prop_iou.expand_as(action_iou))
            pairs = build_dpo_pairs(quality, margin=0.0)
            policy_loss = dpo_loss_from_log_probs(
                action_batch.log_probs,
                baseline_actions.log_probs.detach(),
                pairs,
                beta=0.5,
            )
            kl_cls_loss = baseline_kl_loss(class_logits, baseline_logits)
            kl_box_loss = F.smooth_l1_loss(
                box_regression,
                baseline_box_regression.to(box_regression.device),
            )
            kl_loss = float(args.kl_cls_weight) * kl_cls_loss + float(args.kl_box_weight) * kl_box_loss
            rescue_loss = class_logits.sum() * 0.0
            pairwise_rescue_loss = class_logits.sum() * 0.0
            score_budget_loss = class_logits.sum() * 0.0
            bbox_rescue_loss = class_logits.sum() * 0.0
            confidence_crossing_loss = class_logits.sum() * 0.0
            class_margin_loss = class_logits.sum() * 0.0
            chain_bbox_loss = class_logits.sum() * 0.0
            chain_cls_margin_loss = class_logits.sum() * 0.0
            chain_ranking_loss = class_logits.sum() * 0.0
            verifier_ranking_loss = class_logits.sum() * 0.0
            nms_aware_loss = class_logits.sum() * 0.0
            blocked_nms_loss = class_logits.sum() * 0.0
            pre_nms_rescue_loss = class_logits.sum() * 0.0
            pre_nms_dpo_loss = class_logits.sum() * 0.0
            same_gt_duplicate_loss = class_logits.sum() * 0.0
            if bool(args.rescue_mode) and (
                float(args.rescue_loss_weight) > 0.0
                or float(args.rescue_pairwise_loss_weight) > 0.0
                or float(args.score_budget_loss_weight) > 0.0
                or float(args.bbox_rescue_loss_weight) > 0.0
                or float(args.confidence_crossing_loss_weight) > 0.0
                or float(args.class_margin_loss_weight) > 0.0
                or float(args.chain_bbox_loss_weight) > 0.0
                or float(args.chain_cls_margin_loss_weight) > 0.0
                or float(args.chain_ranking_loss_weight) > 0.0
                or float(args.verifier_ranking_loss_weight) > 0.0
                or float(args.nms_aware_ranking_loss_weight) > 0.0
                or float(args.blocked_nms_loss_weight) > 0.0
                or float(args.pre_nms_rescue_loss_weight) > 0.0
                or float(args.pre_nms_dpo_loss_weight) > 0.0
                or float(args.same_gt_duplicate_ranking_loss_weight) > 0.0
            ):
                gt_labels = target.get("labels", torch.empty((0,), dtype=torch.long)).to(device)
                scaled_gt_boxes = resize_boxes_to_image(
                    gt_boxes,
                    tuple(image.shape[-2:]),
                    tuple(transformed_sizes[0]),
                )
                best_iou, best_labels, target_boxes = match_boxes_to_target_boxes(
                    scaled_boxes[0],
                    scaled_gt_boxes,
                    gt_labels,
                )
                baseline_label_probs = matched_label_probabilities(baseline_logits, best_labels)
                if str(args.proposal_source) == "rpn":
                    proposal_scores = baseline_label_probs.detach().cpu()
                low_conf_scores_for_rescue = (
                    proposal_scores.to(device)
                    if str(args.rescue_low_conf_source) == "final_score"
                    else baseline_label_probs
                )
                crossing_baseline_scores = (
                    proposal_scores.to(device)
                    if str(args.rescue_low_conf_source) == "final_score"
                    else None
                )
                positive_candidate_mask = torch.ones_like(best_labels, dtype=torch.bool, device=device)
                best_gt_indices = torch.zeros_like(best_labels, dtype=torch.long, device=device)
                if gt_boxes.numel() > 0 and proposals.numel() > 0:
                    original_iou = box_iou(proposals.to(device), gt_boxes.float())
                    _, best_gt_indices = original_iou.max(dim=1)
                if str(args.rescue_positive_filter) == "ap75_misses":
                    if gt_boxes.numel() == 0:
                        positive_candidate_mask = torch.zeros_like(positive_candidate_mask)
                    else:
                        rollout_cpu = {key: value.detach().cpu() for key, value in rollout.items()}
                        target_cpu = {
                            key: value.detach().cpu() if torch.is_tensor(value) else value
                            for key, value in target.items()
                        }
                        positive_candidate_mask = unmatched_gt_candidate_mask(
                            rollout_cpu,
                            target_cpu,
                            best_gt_indices.detach().cpu(),
                            torch.ones((proposals.shape[0],), dtype=torch.bool),
                            iou_threshold=0.75,
                            score_threshold=float(args.score_threshold),
                        ).to(device)
                pre_nms_candidate_mask = None
                pre_nms_best_iou = None
                pre_nms_best_labels = None
                pre_nms_target_boxes = None
                pre_nms_best_gt_indices = None
                pre_nms_decoded_boxes = None
                if float(args.pre_nms_rescue_loss_weight) > 0.0 or float(args.pre_nms_dpo_loss_weight) > 0.0:
                    (
                        pre_nms_best_iou,
                        pre_nms_best_labels,
                        pre_nms_target_boxes,
                        pre_nms_best_gt_indices,
                        pre_nms_decoded_boxes,
                    ) = match_pre_nms_decoded_boxes_to_targets(
                        scaled_boxes[0],
                        box_regression,
                        scaled_gt_boxes,
                        gt_labels,
                        image_size=tuple(transformed_sizes[0]),
                        num_classes=num_classes,
                    )
                    if gt_boxes.numel() == 0:
                        pre_nms_positive_mask = torch.zeros_like(pre_nms_best_labels, dtype=torch.bool, device=device)
                    elif str(args.rescue_positive_filter) == "ap75_misses":
                        rollout_cpu = {key: value.detach().cpu() for key, value in rollout.items()}
                        target_cpu = {
                            key: value.detach().cpu() if torch.is_tensor(value) else value
                            for key, value in target.items()
                        }
                        pre_nms_positive_mask = unmatched_gt_candidate_mask(
                            rollout_cpu,
                            target_cpu,
                            pre_nms_best_gt_indices.detach().cpu(),
                            torch.ones((proposals.shape[0],), dtype=torch.bool),
                            iou_threshold=0.75,
                            score_threshold=float(args.score_threshold),
                        ).to(device)
                    else:
                        pre_nms_positive_mask = torch.ones_like(pre_nms_best_labels, dtype=torch.bool, device=device)
                    pre_nms_baseline_probs = matched_label_probabilities(baseline_logits, pre_nms_best_labels)
                    pre_nms_candidate_mask = build_chain_rescue_candidate_mask(
                        pre_nms_best_iou,
                        pre_nms_best_labels,
                        pre_nms_best_gt_indices,
                        pre_nms_baseline_probs,
                        pre_nms_positive_mask,
                        low_conf_max=float(args.pre_nms_low_conf_max),
                        high_iou_min=float(args.pre_nms_high_iou_min),
                        topk_per_gt=int(args.pre_nms_topk_per_gt),
                    )
                    if float(args.pre_nms_rescue_loss_weight) > 0.0:
                        pre_nms_rescue_loss, pre_nms_diag = pre_nms_score_rescue_loss(
                            class_logits,
                            baseline_logits,
                            pre_nms_best_labels,
                            pre_nms_candidate_mask,
                            score_target=float(args.pre_nms_score_target),
                            score_threshold=float(args.score_threshold),
                        )
                        pre_nms_rescue_batch_metric += float(pre_nms_diag["pre_nms_rescue_loss"])
                        pre_nms_count = int(pre_nms_diag["pre_nms_rescue_count"])
                        total_pre_nms_rescue_count += pre_nms_count
                        total_pre_nms_rescue_active += int(pre_nms_diag["pre_nms_rescue_active_count"])
                        total_pre_nms_rescue_prob_delta += (
                            float(pre_nms_diag["pre_nms_rescue_prob_delta_mean"]) * pre_nms_count
                        )
                        total_pre_nms_rescue_score_cross += int(pre_nms_diag["pre_nms_rescue_score_cross_count"])
                    if float(args.pre_nms_dpo_loss_weight) > 0.0:
                        chosen_indices, rejected_indices, dpo_pair_diag = mine_pre_nms_local_dpo_pairs(
                            pre_nms_best_labels,
                            pre_nms_best_gt_indices,
                            pre_nms_best_iou,
                            pre_nms_baseline_probs,
                            pre_nms_candidate_mask,
                            min_iou_gap=float(args.pre_nms_dpo_min_iou_gap),
                            require_rejected_score_ge_chosen=bool(args.pre_nms_dpo_require_rejected_score_ge_chosen),
                            max_pairs_per_gt=int(args.pre_nms_dpo_max_pairs_per_gt),
                        )
                        pre_nms_dpo_loss, pre_nms_dpo_diag = local_pre_nms_dpo_loss(
                            class_logits,
                            baseline_logits,
                            pre_nms_best_labels,
                            chosen_indices,
                            rejected_indices,
                            beta=float(args.pre_nms_dpo_beta),
                        )
                        pre_nms_dpo_batch_metric += float(pre_nms_dpo_diag["pre_nms_dpo_loss"])
                        pair_count = int(pre_nms_dpo_diag["pre_nms_dpo_pair_count"])
                        total_pre_nms_dpo_pairs += pair_count
                        total_pre_nms_dpo_win += int(pre_nms_dpo_diag["pre_nms_dpo_win_count"])
                        total_pre_nms_dpo_preference_margin += (
                            float(pre_nms_dpo_diag["pre_nms_dpo_preference_margin_mean"]) * pair_count
                        )
                        total_pre_nms_dpo_iou_gap += (
                            float(dpo_pair_diag["pre_nms_dpo_mean_iou_gap"]) * pair_count
                        )
                chain_candidate_mask = build_chain_rescue_candidate_mask(
                    best_iou,
                    best_labels,
                    best_gt_indices,
                    low_conf_scores_for_rescue,
                    positive_candidate_mask,
                    low_conf_max=float(rescue_cfg.low_conf_max),
                    high_iou_min=float(rescue_cfg.high_iou_min),
                    topk_per_gt=int(args.chain_topk_per_gt),
                )
                verifier_scores = compute_rescue_verifier_scores(
                    image,
                    proposals,
                    box_features,
                    reference_features,
                    reference_stats,
                    args,
                    proposal_labels=best_labels,
                    proposal_label_probs=baseline_label_probs,
                )
                if float(args.bbox_rescue_loss_weight) > 0.0:
                    thresholds = class_threshold_tensor(
                        best_labels,
                        reference_stats,
                        fallback=float(args.rescue_verifier_gate),
                    )
                    if verifier_scores is None:
                        bbox_weights = (
                            (low_conf_scores_for_rescue <= float(rescue_cfg.low_conf_max))
                            & (best_iou >= float(rescue_cfg.high_iou_min))
                            & (best_labels > 0)
                        ).float()
                    else:
                        bbox_weights, _ = manifold_soft_rescue_weights(
                            proposal_scores.to(device),
                            best_iou,
                            best_labels,
                            verifier_scores.to(device),
                            rescue_cfg,
                            thresholds=thresholds,
                            temperature=float(args.bbox_rescue_weight_temperature),
                            low_conf_scores=low_conf_scores_for_rescue,
                        )
                    bbox_weights = bbox_weights * positive_candidate_mask.float()
                    decoded_current = decode_box_actions(
                        scaled_boxes[0],
                        class_box_deltas(box_regression, best_labels, num_classes).unsqueeze(1),
                        tuple(transformed_sizes[0]),
                    ).squeeze(1)
                    bbox_rescue_loss, bbox_rescue_diag = bbox_localization_rescue_loss(
                        decoded_current,
                        target_boxes,
                        proposal_scores.to(device),
                        best_iou,
                        best_labels,
                        rescue_cfg,
                        rescue_weights=bbox_weights,
                        low_conf_scores=low_conf_scores_for_rescue,
                        loss_mode=str(args.bbox_localization_loss),
                    )
                    total_bbox_rescue_count += int(bbox_rescue_diag["bbox_rescue_count"])
                    total_bbox_rescue_weight += float(bbox_rescue_diag["bbox_rescue_weight_sum"])
                    bbox_rescue_batch_metric += float(bbox_rescue_loss.detach().item())
                decoded_current_for_chain = decode_box_actions(
                    scaled_boxes[0],
                    class_box_deltas(box_regression, best_labels, num_classes).unsqueeze(1),
                    tuple(transformed_sizes[0]),
                ).squeeze(1)
                if float(args.chain_bbox_loss_weight) > 0.0:
                    chain_bbox_weights = chain_candidate_mask.float()
                    chain_bbox_loss, bbox_rescue_diag_chain = bbox_localization_rescue_loss(
                        decoded_current_for_chain,
                        target_boxes,
                        proposal_scores.to(device),
                        best_iou,
                        best_labels,
                        rescue_cfg,
                        rescue_weights=chain_bbox_weights,
                        low_conf_scores=low_conf_scores_for_rescue,
                        loss_mode=str(args.bbox_localization_loss),
                    )
                    total_bbox_rescue_count += int(bbox_rescue_diag_chain["bbox_rescue_count"])
                    total_bbox_rescue_weight += float(bbox_rescue_diag_chain["bbox_rescue_weight_sum"])
                    chain_bbox_batch_metric += float(chain_bbox_loss.detach().item())
                if str(args.rescue_target_mode) == "increment":
                    rescue_loss, rescue_diag = confidence_rescue_increment_loss(
                        class_logits,
                        baseline_logits,
                        proposal_scores.to(device),
                        best_iou,
                        best_labels,
                        rescue_cfg,
                        verifier_scores=verifier_scores,
                        low_conf_scores=low_conf_scores_for_rescue,
                        positive_candidate_mask=positive_candidate_mask,
                        target_delta=float(args.rescue_increment_delta),
                        target_cap=float(args.rescue_increment_cap),
                    )
                else:
                    rescue_loss, rescue_diag = confidence_rescue_loss(
                        class_logits,
                        proposal_scores.to(device),
                        best_iou,
                        best_labels,
                        rescue_cfg,
                        verifier_scores=verifier_scores,
                        low_conf_scores=low_conf_scores_for_rescue,
                        positive_candidate_mask=positive_candidate_mask,
                    )
                if float(args.rescue_pairwise_loss_weight) > 0.0:
                    pairwise_rescue_loss, pairwise_diag = build_pairwise_rescue_ranking_loss(
                        class_logits,
                        proposal_scores.to(device),
                        best_iou,
                        best_labels,
                        rescue_cfg,
                        verifier_scores=verifier_scores,
                        margin=float(args.rescue_pairwise_margin),
                        negative_mode=str(args.rescue_pairwise_negative_mode),
                        low_conf_scores=low_conf_scores_for_rescue,
                    )
                    total_pairwise_pairs += int(pairwise_diag["pairwise_rescue_pair_count"])
                    pairwise_batch_metric += float(pairwise_rescue_loss.detach().item())
                if float(args.score_budget_loss_weight) > 0.0:
                    score_budget_loss, score_budget_diag = score_shift_budget_loss(
                        class_logits,
                        baseline_logits,
                        proposal_scores.to(device),
                        best_iou,
                        best_labels,
                        rescue_cfg,
                        verifier_scores=verifier_scores,
                        low_conf_scores=low_conf_scores_for_rescue,
                        delta=float(args.score_budget_delta),
                    )
                    total_score_budget_count += int(score_budget_diag["score_budget_count"])
                    total_score_budget_violations += int(score_budget_diag["score_budget_violation_count"])
                    score_budget_batch_metric += float(score_budget_loss.detach().item())
                if float(args.confidence_crossing_loss_weight) > 0.0:
                    confidence_crossing_loss, confidence_crossing_diag = confidence_threshold_crossing_loss(
                        class_logits,
                        baseline_logits,
                        proposal_scores.to(device),
                        best_iou,
                        best_labels,
                        rescue_cfg,
                        verifier_scores=verifier_scores,
                        low_conf_scores=low_conf_scores_for_rescue,
                        crossing_baseline_scores=crossing_baseline_scores,
                        positive_candidate_mask=positive_candidate_mask,
                        score_threshold=float(args.score_threshold),
                        margin=float(args.confidence_crossing_margin),
                    )
                    confidence_crossing_batch_metric += float(confidence_crossing_diag["confidence_crossing_loss"])
                    total_confidence_crossing_count += int(confidence_crossing_diag["confidence_crossing_count"])
                    total_confidence_crossing_active_count += int(
                        confidence_crossing_diag["confidence_crossing_active_count"]
                    )
                if float(args.class_margin_loss_weight) > 0.0:
                    if verifier_scores is None:
                        verifier_positive_for_margin = torch.ones_like(positive_candidate_mask)
                    else:
                        thresholds = class_threshold_tensor(
                            best_labels,
                            reference_stats,
                            fallback=float(args.rescue_verifier_gate),
                        )
                        verifier_positive_for_margin = verifier_scores.to(device) >= thresholds
                    margin_candidate_mask = (
                        positive_candidate_mask
                        & verifier_positive_for_margin
                        & (low_conf_scores_for_rescue <= float(rescue_cfg.low_conf_max))
                        & (best_iou >= float(rescue_cfg.high_iou_min))
                        & (best_labels > 0)
                    )
                    class_margin_loss, class_margin_diag = class_margin_rescue_loss(
                        class_logits,
                        best_iou,
                        best_labels,
                        margin_candidate_mask,
                        margin=float(args.class_margin_margin),
                    )
                    class_margin_batch_metric += float(class_margin_diag["class_margin_loss"])
                    total_class_margin_count += int(class_margin_diag["class_margin_count"])
                    total_class_margin_active_count += int(class_margin_diag["class_margin_active_count"])
                if float(args.chain_cls_margin_loss_weight) > 0.0:
                    chain_cls_margin_loss, chain_class_margin_diag = class_margin_rescue_loss(
                        class_logits,
                        best_iou,
                        best_labels,
                        chain_candidate_mask,
                        margin=float(args.chain_cls_margin_margin),
                        include_background=bool(args.chain_cls_margin_include_background),
                    )
                    chain_cls_margin_batch_metric += float(chain_class_margin_diag["class_margin_loss"])
                    total_class_margin_count += int(chain_class_margin_diag["class_margin_count"])
                    total_class_margin_active_count += int(chain_class_margin_diag["class_margin_active_count"])
                if float(args.chain_ranking_loss_weight) > 0.0:
                    dangerous_negative_mask = (
                        (best_labels > 0)
                        & (best_iou <= float(rescue_cfg.low_iou_max))
                        & (low_conf_scores_for_rescue >= float(args.chain_dangerous_negative_min_score))
                    )
                    chain_ranking_loss, chain_ranking_diag = chain_rescue_ranking_loss(
                        class_logits,
                        best_labels,
                        chain_candidate_mask,
                        dangerous_negative_mask,
                        margin=float(args.chain_ranking_margin),
                    )
                    chain_ranking_batch_metric += float(chain_ranking_diag["chain_ranking_loss"])
                    total_chain_ranking_pairs += int(chain_ranking_diag["chain_ranking_pair_count"])
                    total_chain_ranking_active += int(chain_ranking_diag["chain_ranking_active_count"])
                if float(args.verifier_ranking_loss_weight) > 0.0 and verifier_scores is not None:
                    verifier_ranking_loss, verifier_ranking_diag = build_verifier_guided_ranking_loss(
                        class_logits,
                        best_labels,
                        best_iou,
                        verifier_scores,
                        positive_iou_min=float(args.verifier_ranking_positive_iou_min),
                        negative_iou_max=float(args.verifier_ranking_negative_iou_max),
                        positive_score_min=float(args.verifier_ranking_positive_score_min),
                        negative_score_max=float(args.verifier_ranking_negative_score_max),
                        margin=float(args.verifier_ranking_margin),
                        max_pairs=int(args.verifier_ranking_max_pairs),
                    )
                    verifier_ranking_batch_metric += float(verifier_ranking_diag["verifier_ranking_loss"])
                    total_verifier_ranking_pairs += int(verifier_ranking_diag["verifier_ranking_pair_count"])
                    total_verifier_ranking_active += int(verifier_ranking_diag["verifier_ranking_active_count"])
                    total_verifier_ranking_positive += int(verifier_ranking_diag["verifier_ranking_positive_count"])
                    total_verifier_ranking_negative += int(verifier_ranking_diag["verifier_ranking_negative_count"])
                if float(args.nms_aware_ranking_loss_weight) > 0.0:
                    nms_aware_loss, nms_aware_diag = nms_aware_rescue_ranking_loss(
                        class_logits,
                        decoded_current_for_chain,
                        best_labels,
                        chain_candidate_mask,
                        nms_iou_threshold=float(args.nms_aware_nms_iou),
                        margin=float(args.nms_aware_ranking_margin),
                        require_suppressor_score_ge_candidate=bool(
                            args.nms_aware_require_suppressor_score_ge_candidate
                        ),
                        ranking_mode=str(args.nms_aware_ranking_mode),
                    )
                    nms_aware_batch_metric += float(nms_aware_diag["nms_aware_ranking_loss"])
                    total_nms_aware_pairs += int(nms_aware_diag["nms_aware_pair_count"])
                    total_nms_aware_active += int(nms_aware_diag["nms_aware_active_count"])
                if float(args.blocked_nms_loss_weight) > 0.0:
                    blocked_nms_loss, blocked_nms_diag = blocked_nms_crossing_rescue_loss(
                        class_logits,
                        decoded_current_for_chain,
                        best_labels,
                        best_iou,
                        chain_candidate_mask,
                        score_threshold=float(args.score_threshold),
                        score_epsilon=float(args.blocked_nms_score_epsilon),
                        nms_iou_threshold=float(args.blocked_nms_iou),
                        base_margin=float(args.blocked_nms_base_margin),
                        iou_margin_scale=float(args.blocked_nms_iou_margin_scale),
                        max_margin=float(args.blocked_nms_max_margin),
                        rank_weight=float(args.blocked_nms_rank_weight),
                        crossing_weight=float(args.blocked_nms_crossing_weight),
                        require_suppressor_score_ge_candidate=bool(
                            args.blocked_nms_require_suppressor_score_ge_candidate
                        ),
                        ranking_mode=str(args.blocked_nms_ranking_mode),
                        baseline_logits=baseline_logits,
                    )
                    blocked_nms_batch_metric += float(blocked_nms_diag["blocked_nms_loss"])
                    total_blocked_nms_pairs += int(blocked_nms_diag["blocked_nms_pair_count"])
                    total_blocked_nms_active_rank += int(blocked_nms_diag["blocked_nms_active_rank_count"])
                    total_blocked_nms_active_crossing += int(blocked_nms_diag["blocked_nms_active_crossing_count"])
                    pair_count_for_delta = int(blocked_nms_diag["blocked_nms_pair_count"])
                    total_blocked_nms_candidate_delta += (
                        float(blocked_nms_diag["blocked_nms_candidate_delta_mean"]) * pair_count_for_delta
                    )
                    total_blocked_nms_suppressor_delta += (
                        float(blocked_nms_diag["blocked_nms_suppressor_delta_mean"]) * pair_count_for_delta
                    )
                    total_blocked_nms_relative_delta += (
                        float(blocked_nms_diag["blocked_nms_relative_delta_mean"]) * pair_count_for_delta
                    )
                if float(args.same_gt_duplicate_ranking_loss_weight) > 0.0 and chain_candidate_mask.any():
                    if str(args.same_gt_duplicate_pair_source) == "proposal":
                        labels_for_duplicate_scores = best_labels.to(device).long().clamp(
                            min=0,
                            max=class_logits.shape[1] - 1,
                        )
                        row_for_duplicate_scores = torch.arange(labels_for_duplicate_scores.numel(), device=device)
                        duplicate_label_probs = F.softmax(class_logits, dim=1)[
                            row_for_duplicate_scores,
                            labels_for_duplicate_scores,
                        ]
                        positive_indices, negative_indices, duplicate_diag = find_same_gt_worse_duplicate_proposal_pairs(
                            decoded_current_for_chain,
                            duplicate_label_probs,
                            best_labels,
                            best_gt_indices,
                            chain_candidate_mask,
                            {
                                "boxes": scaled_gt_boxes.detach(),
                                "labels": gt_labels.detach(),
                            },
                            nms_iou_threshold=float(args.same_gt_duplicate_nms_iou),
                            min_iou_gap=float(args.same_gt_duplicate_min_iou_gap),
                            require_suppressor_score_ge_candidate=bool(
                                args.same_gt_duplicate_require_suppressor_score_ge_candidate
                            ),
                        )
                        if positive_indices.numel() > 0:
                            same_gt_duplicate_loss, same_gt_diag = same_gt_duplicate_ranking_loss(
                                class_logits[positive_indices],
                                class_logits[negative_indices],
                                best_labels[positive_indices],
                                margin=float(args.same_gt_duplicate_ranking_margin),
                                detach_suppressor=bool(args.same_gt_duplicate_detach_suppressor),
                            )
                            same_gt_duplicate_batch_metric += float(same_gt_diag["same_gt_duplicate_ranking_loss"])
                            total_same_gt_duplicate_pairs += int(same_gt_diag["same_gt_duplicate_ranking_pair_count"])
                            total_same_gt_duplicate_active += int(same_gt_diag["same_gt_duplicate_ranking_active_count"])
                        else:
                            total_same_gt_duplicate_pairs += int(duplicate_diag["same_gt_proposal_pair_count"])
                    else:
                        decoded_chain_original = resize_boxes_to_image(
                            decoded_current_for_chain,
                            tuple(transformed_sizes[0]),
                            tuple(image.shape[-2:]),
                        )
                        candidate_indices, suppressor_boxes, duplicate_diag = find_same_gt_worse_duplicate_pairs(
                            decoded_chain_original[chain_candidate_mask].detach(),
                            best_labels[chain_candidate_mask].detach().cpu(),
                            best_gt_indices[chain_candidate_mask].detach().cpu(),
                            {
                                key: value.detach().cpu() if torch.is_tensor(value) else value
                                for key, value in target.items()
                            },
                            {
                                key: value.detach().cpu() if torch.is_tensor(value) else value
                                for key, value in rollout.items()
                            },
                            score_threshold=float(args.score_threshold),
                            nms_iou_threshold=float(args.same_gt_duplicate_nms_iou),
                            min_iou_gap=float(args.same_gt_duplicate_min_iou_gap),
                        )
                        if candidate_indices.numel() > 0:
                            chain_all_indices = torch.nonzero(chain_candidate_mask, as_tuple=False).flatten()
                            selected_candidate_indices = chain_all_indices[candidate_indices.to(chain_all_indices.device)]
                            suppressor_logits, _, _, _ = extract_roi_head_outputs_for_boxes(
                                model,
                                [image.to(device)],
                                [suppressor_boxes],
                            )
                            same_gt_duplicate_loss, same_gt_diag = same_gt_duplicate_ranking_loss(
                                class_logits[selected_candidate_indices],
                                suppressor_logits,
                                best_labels[selected_candidate_indices],
                                margin=float(args.same_gt_duplicate_ranking_margin),
                                detach_suppressor=bool(args.same_gt_duplicate_detach_suppressor),
                            )
                            same_gt_duplicate_batch_metric += float(same_gt_diag["same_gt_duplicate_ranking_loss"])
                            total_same_gt_duplicate_pairs += int(same_gt_diag["same_gt_duplicate_ranking_pair_count"])
                            total_same_gt_duplicate_active += int(same_gt_diag["same_gt_duplicate_ranking_active_count"])
                        else:
                            total_same_gt_duplicate_pairs += int(duplicate_diag["same_gt_duplicate_pair_count"])
                region_summary = summarize_confidence_iou_regions(
                    proposal_scores.to(device),
                    best_iou,
                    rescue_cfg,
                    low_conf_scores=low_conf_scores_for_rescue,
                )
                labels_for_probs = best_labels.to(device).long().clamp(min=0, max=class_logits.shape[1] - 1)
                all_row = torch.arange(labels_for_probs.numel(), device=device)
                current_label_probs = F.softmax(class_logits, dim=1)[all_row, labels_for_probs]
                baseline_decoded_for_chain = decode_box_actions(
                    scaled_boxes[0],
                    class_box_deltas(baseline_box_regression.to(device), best_labels, num_classes).unsqueeze(1),
                    tuple(transformed_sizes[0]),
                ).squeeze(1)
                baseline_chain_iou = proposal_iou_for_scores(baseline_decoded_for_chain, scaled_gt_boxes)
                current_chain_iou = proposal_iou_for_scores(decoded_current_for_chain, scaled_gt_boxes)
                chain_count = int(chain_candidate_mask.sum().item())
                total_chain_candidate_count += chain_count
                if chain_count > 0:
                    total_chain_baseline_prob_sum += float(baseline_label_probs[chain_candidate_mask].sum().item())
                    total_chain_current_prob_sum += float(current_label_probs[chain_candidate_mask].sum().item())
                    total_chain_score_threshold_cross += int(
                        (
                            (baseline_label_probs[chain_candidate_mask] < float(args.score_threshold))
                            & (current_label_probs[chain_candidate_mask] >= float(args.score_threshold))
                        ).sum().item()
                    )
                    total_chain_low_conf_max_cross += int(
                        (
                            (baseline_label_probs[chain_candidate_mask] <= float(rescue_cfg.low_conf_max))
                            & (current_label_probs[chain_candidate_mask] > float(rescue_cfg.low_conf_max))
                        ).sum().item()
                    )
                    total_chain_baseline_iou_sum += float(baseline_chain_iou[chain_candidate_mask].sum().item())
                    total_chain_current_iou_sum += float(current_chain_iou[chain_candidate_mask].sum().item())
                lchi_mask = (
                    (low_conf_scores_for_rescue <= float(rescue_cfg.low_conf_max))
                    & (best_iou >= float(rescue_cfg.high_iou_min))
                    & (best_labels > 0)
                )
                if verifier_scores is None:
                    verifier_positive_mask = torch.zeros_like(lchi_mask)
                else:
                    thresholds = class_threshold_tensor(
                        best_labels,
                        reference_stats,
                        fallback=float(args.rescue_verifier_gate),
                    )
                    verifier_positive_mask = verifier_scores.to(device) >= thresholds
                confidence_effect = summarize_confidence_rescue_effect(
                    baseline_label_probs,
                    current_label_probs,
                    lchi_mask,
                    verifier_positive_mask=verifier_positive_mask,
                    score_threshold=float(args.score_threshold),
                    low_conf_max=float(rescue_cfg.low_conf_max),
                )
                accumulate_metric_sums(confidence_rescue_effect_sums, confidence_effect)
                total_rescue_positive += int(rescue_diag["rescue_positive_count"])
                total_rescue_negative += int(rescue_diag["rescue_negative_count"])
                total_lchi += int(region_summary["low_conf_high_iou_count"])
                total_hcli += int(region_summary["high_conf_low_iou_count"])
            policy_batch_loss = policy_loss if policy_batch_loss is None else policy_batch_loss + policy_loss
            kl_batch_loss = kl_loss if kl_batch_loss is None else kl_batch_loss + kl_loss
            kl_cls_batch_loss = kl_cls_loss if kl_cls_batch_loss is None else kl_cls_batch_loss + kl_cls_loss
            kl_box_batch_loss = kl_box_loss if kl_box_batch_loss is None else kl_box_batch_loss + kl_box_loss
            rescue_batch_loss = rescue_loss if rescue_batch_loss is None else rescue_batch_loss + rescue_loss
            pairwise_batch_loss = (
                pairwise_rescue_loss
                if pairwise_batch_loss is None
                else pairwise_batch_loss + pairwise_rescue_loss
            )
            score_budget_batch_loss = (
                score_budget_loss
                if score_budget_batch_loss is None
                else score_budget_batch_loss + score_budget_loss
            )
            bbox_rescue_batch_loss = (
                bbox_rescue_loss
                if bbox_rescue_batch_loss is None
                else bbox_rescue_batch_loss + bbox_rescue_loss
            )
            confidence_crossing_batch_loss = (
                confidence_crossing_loss
                if confidence_crossing_batch_loss is None
                else confidence_crossing_batch_loss + confidence_crossing_loss
            )
            class_margin_batch_loss = (
                class_margin_loss
                if class_margin_batch_loss is None
                else class_margin_batch_loss + class_margin_loss
            )
            chain_bbox_batch_loss = (
                chain_bbox_loss
                if chain_bbox_batch_loss is None
                else chain_bbox_batch_loss + chain_bbox_loss
            )
            chain_cls_margin_batch_loss = (
                chain_cls_margin_loss
                if chain_cls_margin_batch_loss is None
                else chain_cls_margin_batch_loss + chain_cls_margin_loss
            )
            chain_ranking_batch_loss = (
                chain_ranking_loss
                if chain_ranking_batch_loss is None
                else chain_ranking_batch_loss + chain_ranking_loss
            )
            verifier_ranking_batch_loss = (
                verifier_ranking_loss
                if verifier_ranking_batch_loss is None
                else verifier_ranking_batch_loss + verifier_ranking_loss
            )
            nms_aware_batch_loss = (
                nms_aware_loss
                if nms_aware_batch_loss is None
                else nms_aware_batch_loss + nms_aware_loss
            )
            blocked_nms_batch_loss = (
                blocked_nms_loss
                if blocked_nms_batch_loss is None
                else blocked_nms_batch_loss + blocked_nms_loss
            )
            pre_nms_rescue_batch_loss = (
                pre_nms_rescue_loss
                if pre_nms_rescue_batch_loss is None
                else pre_nms_rescue_batch_loss + pre_nms_rescue_loss
            )
            pre_nms_dpo_batch_loss = (
                pre_nms_dpo_loss
                if pre_nms_dpo_batch_loss is None
                else pre_nms_dpo_batch_loss + pre_nms_dpo_loss
            )
            same_gt_duplicate_batch_loss = (
                same_gt_duplicate_loss
                if same_gt_duplicate_batch_loss is None
                else same_gt_duplicate_batch_loss + same_gt_duplicate_loss
            )
            batch_valid += int(pairs.valid.sum().item())
            batch_actions += int(action_batch.log_probs.numel())

        if policy_batch_loss is None:
            policy_batch_loss = det_loss * 0.0
            kl_batch_loss = det_loss * 0.0
            kl_cls_batch_loss = det_loss * 0.0
            kl_box_batch_loss = det_loss * 0.0
            rescue_batch_loss = det_loss * 0.0
            pairwise_batch_loss = det_loss * 0.0
            score_budget_batch_loss = det_loss * 0.0
            bbox_rescue_batch_loss = det_loss * 0.0
            confidence_crossing_batch_loss = det_loss * 0.0
            class_margin_batch_loss = det_loss * 0.0
            chain_bbox_batch_loss = det_loss * 0.0
            chain_cls_margin_batch_loss = det_loss * 0.0
            chain_ranking_batch_loss = det_loss * 0.0
            verifier_ranking_batch_loss = det_loss * 0.0
            nms_aware_batch_loss = det_loss * 0.0
            blocked_nms_batch_loss = det_loss * 0.0
            pre_nms_rescue_batch_loss = det_loss * 0.0
            pre_nms_dpo_batch_loss = det_loss * 0.0
            same_gt_duplicate_batch_loss = det_loss * 0.0
        if rescue_batch_loss is None:
            rescue_batch_loss = policy_batch_loss * 0.0
        if pairwise_batch_loss is None:
            pairwise_batch_loss = policy_batch_loss * 0.0
        if score_budget_batch_loss is None:
            score_budget_batch_loss = policy_batch_loss * 0.0
        if bbox_rescue_batch_loss is None:
            bbox_rescue_batch_loss = policy_batch_loss * 0.0
        if confidence_crossing_batch_loss is None:
            confidence_crossing_batch_loss = policy_batch_loss * 0.0
        if class_margin_batch_loss is None:
            class_margin_batch_loss = policy_batch_loss * 0.0
        if chain_bbox_batch_loss is None:
            chain_bbox_batch_loss = policy_batch_loss * 0.0
        if chain_cls_margin_batch_loss is None:
            chain_cls_margin_batch_loss = policy_batch_loss * 0.0
        if chain_ranking_batch_loss is None:
            chain_ranking_batch_loss = policy_batch_loss * 0.0
        if verifier_ranking_batch_loss is None:
            verifier_ranking_batch_loss = policy_batch_loss * 0.0
        if nms_aware_batch_loss is None:
            nms_aware_batch_loss = policy_batch_loss * 0.0
        if blocked_nms_batch_loss is None:
            blocked_nms_batch_loss = policy_batch_loss * 0.0
        if pre_nms_rescue_batch_loss is None:
            pre_nms_rescue_batch_loss = policy_batch_loss * 0.0
        if pre_nms_dpo_batch_loss is None:
            pre_nms_dpo_batch_loss = policy_batch_loss * 0.0
        if same_gt_duplicate_batch_loss is None:
            same_gt_duplicate_batch_loss = policy_batch_loss * 0.0
        batch_loss = (
            float(args.det_loss_weight) * det_loss
            + float(args.policy_loss_weight) * policy_batch_loss
            + float(args.kl_weight) * kl_batch_loss
            + float(args.rescue_loss_weight) * rescue_batch_loss
            + float(args.rescue_pairwise_loss_weight) * pairwise_batch_loss
            + float(args.score_budget_loss_weight) * score_budget_batch_loss
            + float(args.bbox_rescue_loss_weight) * bbox_rescue_batch_loss
            + float(args.confidence_crossing_loss_weight) * confidence_crossing_batch_loss
            + float(args.class_margin_loss_weight) * class_margin_batch_loss
            + float(args.chain_bbox_loss_weight) * chain_bbox_batch_loss
            + float(args.chain_cls_margin_loss_weight) * chain_cls_margin_batch_loss
            + float(args.chain_ranking_loss_weight) * chain_ranking_batch_loss
            + float(args.verifier_ranking_loss_weight) * verifier_ranking_batch_loss
            + float(args.nms_aware_ranking_loss_weight) * nms_aware_batch_loss
            + float(args.blocked_nms_loss_weight) * blocked_nms_batch_loss
            + float(args.pre_nms_rescue_loss_weight) * pre_nms_rescue_batch_loss
            + float(args.pre_nms_dpo_loss_weight) * pre_nms_dpo_batch_loss
            + float(args.same_gt_duplicate_ranking_loss_weight) * same_gt_duplicate_batch_loss
        )
        if bool(args.record_grad_diagnostics):
            grad_components = {
                "det": float(args.det_loss_weight) * det_loss,
                "policy": float(args.policy_loss_weight) * policy_batch_loss,
                "kl": float(args.kl_weight) * kl_batch_loss,
                "confidence": float(args.rescue_loss_weight) * rescue_batch_loss,
                "pairwise": float(args.rescue_pairwise_loss_weight) * pairwise_batch_loss,
                "score_budget": float(args.score_budget_loss_weight) * score_budget_batch_loss,
                "bbox_rescue": float(args.bbox_rescue_loss_weight) * bbox_rescue_batch_loss,
                "confidence_crossing": float(args.confidence_crossing_loss_weight) * confidence_crossing_batch_loss,
                "class_margin": float(args.class_margin_loss_weight) * class_margin_batch_loss,
                "chain_bbox": float(args.chain_bbox_loss_weight) * chain_bbox_batch_loss,
                "chain_cls_margin": float(args.chain_cls_margin_loss_weight) * chain_cls_margin_batch_loss,
                "chain_ranking": float(args.chain_ranking_loss_weight) * chain_ranking_batch_loss,
                "verifier_ranking": float(args.verifier_ranking_loss_weight) * verifier_ranking_batch_loss,
                "nms_aware": float(args.nms_aware_ranking_loss_weight) * nms_aware_batch_loss,
                "blocked_nms": float(args.blocked_nms_loss_weight) * blocked_nms_batch_loss,
                "pre_nms_rescue": float(args.pre_nms_rescue_loss_weight) * pre_nms_rescue_batch_loss,
                "pre_nms_dpo": float(args.pre_nms_dpo_loss_weight) * pre_nms_dpo_batch_loss,
                "same_gt_duplicate": float(args.same_gt_duplicate_ranking_loss_weight) * same_gt_duplicate_batch_loss,
            }
            grad_metrics = summarize_loss_component_gradients(
                grad_components,
                model.named_parameters(),
                retain_graph=True,
            )
            accumulate_metric_sums(grad_metric_sums, grad_metrics)
        optimizer.zero_grad(set_to_none=True)
        batch_loss.backward()
        if bool(args.record_grad_diagnostics):
            total_grad_metrics = summarize_current_parameter_gradients(model.named_parameters())
            accumulate_metric_sums(grad_metric_sums, total_grad_metrics)
            grad_metric_batches += 1
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=2.0)
        optimizer.step()

        total_loss += float(batch_loss.item())
        total_policy += float(policy_batch_loss.item())
        total_kl += float(kl_batch_loss.item())
        total_kl_cls += float(kl_cls_batch_loss.item())
        total_kl_box += float(kl_box_batch_loss.item())
        total_det += float(det_loss.item())
        total_rescue += float(rescue_batch_loss.item())
        total_pairwise_rescue += pairwise_batch_metric
        total_score_budget += score_budget_batch_metric
        total_bbox_rescue += bbox_rescue_batch_metric
        total_confidence_crossing += confidence_crossing_batch_metric
        total_class_margin += class_margin_batch_metric
        total_chain_bbox += chain_bbox_batch_metric
        total_chain_cls_margin += chain_cls_margin_batch_metric
        total_chain_ranking += chain_ranking_batch_metric
        total_verifier_ranking += verifier_ranking_batch_metric
        total_nms_aware_ranking += nms_aware_batch_metric
        total_blocked_nms += blocked_nms_batch_metric
        total_pre_nms_rescue += pre_nms_rescue_batch_metric
        total_pre_nms_dpo += pre_nms_dpo_batch_metric
        total_same_gt_duplicate_ranking += same_gt_duplicate_batch_metric
        total_valid += batch_valid
        total_actions += batch_actions
        total_batches += 1

    row = {
        "loss": total_loss / max(1, total_batches),
        "policy_loss": total_policy / max(1, total_batches),
        "kl_loss": total_kl / max(1, total_batches),
        "kl_cls_loss": total_kl_cls / max(1, total_batches),
        "kl_box_loss": total_kl_box / max(1, total_batches),
        "det_loss": total_det / max(1, total_batches),
        "rescue_loss": total_rescue / max(1, total_batches),
        "pairwise_rescue_loss": total_pairwise_rescue / max(1, total_batches),
        "score_budget_loss": total_score_budget / max(1, total_batches),
        "bbox_rescue_loss": total_bbox_rescue / max(1, total_batches),
        "confidence_crossing_loss": total_confidence_crossing / max(1, total_batches),
        "class_margin_loss": total_class_margin / max(1, total_batches),
        "chain_bbox_loss": total_chain_bbox / max(1, total_batches),
        "chain_cls_margin_loss": total_chain_cls_margin / max(1, total_batches),
        "chain_ranking_loss": total_chain_ranking / max(1, total_batches),
        "verifier_ranking_loss": total_verifier_ranking / max(1, total_batches),
        "nms_aware_ranking_loss": total_nms_aware_ranking / max(1, total_batches),
        "blocked_nms_loss": total_blocked_nms / max(1, total_batches),
        "pre_nms_rescue_loss": total_pre_nms_rescue / max(1, total_batches),
        "pre_nms_dpo_loss": total_pre_nms_dpo / max(1, total_batches),
        "same_gt_duplicate_ranking_loss": total_same_gt_duplicate_ranking / max(1, total_batches),
        "valid_count": total_valid,
        "action_count": total_actions,
        "valid_rate": total_valid / max(1, total_actions),
        "rescue_positive_count": total_rescue_positive,
        "rescue_negative_count": total_rescue_negative,
        "pairwise_rescue_pair_count": total_pairwise_pairs,
        "score_budget_count": total_score_budget_count,
        "score_budget_violation_count": total_score_budget_violations,
        "bbox_rescue_count": total_bbox_rescue_count,
        "bbox_rescue_weight_sum": total_bbox_rescue_weight,
        "confidence_crossing_count": total_confidence_crossing_count,
        "confidence_crossing_active_count": total_confidence_crossing_active_count,
        "class_margin_count": total_class_margin_count,
        "class_margin_active_count": total_class_margin_active_count,
        "chain_candidate_count": total_chain_candidate_count,
        "chain_ranking_pair_count": total_chain_ranking_pairs,
        "chain_ranking_active_count": total_chain_ranking_active,
        "verifier_ranking_pair_count": total_verifier_ranking_pairs,
        "verifier_ranking_active_count": total_verifier_ranking_active,
        "verifier_ranking_positive_count": total_verifier_ranking_positive,
        "verifier_ranking_negative_count": total_verifier_ranking_negative,
        "nms_aware_ranking_pair_count": total_nms_aware_pairs,
        "nms_aware_ranking_active_count": total_nms_aware_active,
        "blocked_nms_pair_count": total_blocked_nms_pairs,
        "blocked_nms_active_rank_count": total_blocked_nms_active_rank,
        "blocked_nms_active_crossing_count": total_blocked_nms_active_crossing,
        "blocked_nms_candidate_delta_mean": total_blocked_nms_candidate_delta / max(1, total_blocked_nms_pairs),
        "blocked_nms_suppressor_delta_mean": total_blocked_nms_suppressor_delta / max(1, total_blocked_nms_pairs),
        "blocked_nms_relative_delta_mean": total_blocked_nms_relative_delta / max(1, total_blocked_nms_pairs),
        "pre_nms_rescue_count": total_pre_nms_rescue_count,
        "pre_nms_rescue_active_count": total_pre_nms_rescue_active,
        "pre_nms_rescue_prob_delta_mean": total_pre_nms_rescue_prob_delta / max(1, total_pre_nms_rescue_count),
        "pre_nms_rescue_score_cross_count": total_pre_nms_rescue_score_cross,
        "pre_nms_dpo_pair_count": total_pre_nms_dpo_pairs,
        "pre_nms_dpo_win_count": total_pre_nms_dpo_win,
        "pre_nms_dpo_win_rate": total_pre_nms_dpo_win / max(1, total_pre_nms_dpo_pairs),
        "pre_nms_dpo_preference_margin_mean": total_pre_nms_dpo_preference_margin
        / max(1, total_pre_nms_dpo_pairs),
        "pre_nms_dpo_iou_gap_mean": total_pre_nms_dpo_iou_gap / max(1, total_pre_nms_dpo_pairs),
        "same_gt_duplicate_ranking_pair_count": total_same_gt_duplicate_pairs,
        "same_gt_duplicate_ranking_active_count": total_same_gt_duplicate_active,
        "chain_score_threshold_cross_count": total_chain_score_threshold_cross,
        "chain_low_conf_max_cross_count": total_chain_low_conf_max_cross,
        "low_conf_high_iou_count": total_lchi,
        "high_conf_low_iou_count": total_hcli,
        "rescue_positive_rate": total_rescue_positive / max(1, total_lchi),
    }
    chain_count = max(1.0, float(total_chain_candidate_count))
    row["chain_baseline_prob_mean"] = total_chain_baseline_prob_sum / chain_count
    row["chain_current_prob_mean"] = total_chain_current_prob_sum / chain_count
    row["chain_prob_delta_mean"] = (total_chain_current_prob_sum - total_chain_baseline_prob_sum) / chain_count
    row["chain_baseline_iou_mean"] = total_chain_baseline_iou_sum / chain_count
    row["chain_current_iou_mean"] = total_chain_current_iou_sum / chain_count
    row["chain_iou_delta_mean"] = (total_chain_current_iou_sum - total_chain_baseline_iou_sum) / chain_count
    row["chain_score_threshold_cross_rate"] = total_chain_score_threshold_cross / chain_count
    row["chain_low_conf_max_cross_rate"] = total_chain_low_conf_max_cross / chain_count
    row.update(confidence_rescue_effect_sums)
    lchi_conf_count = max(1.0, float(row.get("lchi_conf_count", 0.0)))
    verifier_lchi_conf_count = max(1.0, float(row.get("verifier_positive_lchi_conf_count", 0.0)))
    row["lchi_conf_baseline_prob_mean"] = float(row.get("lchi_conf_baseline_prob_sum", 0.0)) / lchi_conf_count
    row["lchi_conf_current_prob_mean"] = float(row.get("lchi_conf_current_prob_sum", 0.0)) / lchi_conf_count
    row["lchi_conf_delta_mean"] = float(row.get("lchi_conf_delta_sum", 0.0)) / lchi_conf_count
    row["lchi_conf_cross_score_threshold_rate"] = float(
        row.get("lchi_conf_cross_score_threshold_count", 0.0)
    ) / lchi_conf_count
    row["lchi_conf_cross_low_conf_max_rate"] = float(row.get("lchi_conf_cross_low_conf_max_count", 0.0)) / lchi_conf_count
    row["verifier_positive_lchi_conf_baseline_prob_mean"] = float(
        row.get("verifier_positive_lchi_conf_baseline_prob_sum", 0.0)
    ) / verifier_lchi_conf_count
    row["verifier_positive_lchi_conf_current_prob_mean"] = float(
        row.get("verifier_positive_lchi_conf_current_prob_sum", 0.0)
    ) / verifier_lchi_conf_count
    row["verifier_positive_lchi_conf_delta_mean"] = float(
        row.get("verifier_positive_lchi_conf_delta_sum", 0.0)
    ) / verifier_lchi_conf_count
    row["verifier_positive_lchi_conf_cross_score_threshold_rate"] = float(
        row.get("verifier_positive_lchi_conf_cross_score_threshold_count", 0.0)
    ) / verifier_lchi_conf_count
    row["verifier_positive_lchi_conf_cross_low_conf_max_rate"] = float(
        row.get("verifier_positive_lchi_conf_cross_low_conf_max_count", 0.0)
    ) / verifier_lchi_conf_count
    row.update(average_metric_sums(grad_metric_sums, grad_metric_batches))
    return row


def parse_args():
    parser = argparse.ArgumentParser(description="Round 2.129 NWPU posttrain smoke.")
    parser.add_argument("--run-name", default="round2129_nwpu_posttrain_smoke")
    parser.add_argument("--limit-train", type=int, default=16)
    parser.add_argument("--limit-val", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-proposals", type=int, default=8)
    parser.add_argument("--proposal-source", choices=["rollout", "rpn"], default="rollout")
    parser.add_argument("--rollout-score-threshold", type=float, default=0.01)
    parser.add_argument("--rollout-detections-per-img", type=int, default=300)
    parser.add_argument("--eval-detections-per-img", type=int, default=100)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=7e-5)
    parser.add_argument("--adapter-lr", type=float, default=None)
    parser.add_argument("--predictor-lr", type=float, default=None)
    parser.add_argument("--cls-score-lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--trainable-mode",
        choices=[
            "adapter",
            "predictor",
            "cls_adapter",
            "bbox_adapter",
            "cls_predictor",
            "cls_score",
            "bbox_predictor",
            "bbox_predictor_cls_adapter",
        ],
        default="adapter",
    )
    parser.add_argument("--record-grad-diagnostics", action="store_true")
    parser.add_argument("--det-loss-weight", type=float, default=0.1)
    parser.add_argument("--policy-loss-weight", type=float, default=0.003)
    parser.add_argument("--kl-weight", type=float, default=1.0)
    parser.add_argument("--kl-cls-weight", type=float, default=1.0)
    parser.add_argument("--kl-box-weight", type=float, default=1.0)
    parser.add_argument("--rescue-mode", action="store_true")
    parser.add_argument("--rescue-loss-weight", type=float, default=0.02)
    parser.add_argument("--rescue-low-conf-max", type=float, default=0.5)
    parser.add_argument("--rescue-high-conf-min", type=float, default=0.7)
    parser.add_argument("--rescue-high-iou-min", type=float, default=0.75)
    parser.add_argument("--rescue-low-iou-max", type=float, default=0.3)
    parser.add_argument("--rescue-positive-weight", type=float, default=1.0)
    parser.add_argument("--rescue-negative-weight", type=float, default=0.25)
    parser.add_argument("--rescue-include-low-conf-negatives", action="store_true")
    parser.add_argument("--rescue-use-hard-negative-mining", action="store_true")
    parser.add_argument("--rescue-hard-negative-verifier-gate", type=float, default=0.0)
    parser.add_argument("--rescue-pairwise-loss-weight", type=float, default=0.0)
    parser.add_argument("--rescue-pairwise-margin", type=float, default=0.1)
    parser.add_argument(
        "--rescue-pairwise-negative-mode",
        choices=["all_low_iou", "dangerous"],
        default="all_low_iou",
    )
    parser.add_argument("--rescue-target-mode", choices=["ce", "increment"], default="ce")
    parser.add_argument("--rescue-increment-delta", type=float, default=0.05)
    parser.add_argument("--rescue-increment-cap", type=float, default=0.6)
    parser.add_argument("--score-budget-loss-weight", type=float, default=0.0)
    parser.add_argument("--score-budget-delta", type=float, default=0.05)
    parser.add_argument("--confidence-crossing-loss-weight", type=float, default=0.0)
    parser.add_argument("--confidence-crossing-margin", type=float, default=0.02)
    parser.add_argument(
        "--rescue-low-conf-source",
        choices=["roi_label_prob", "final_score"],
        default="roi_label_prob",
    )
    parser.add_argument(
        "--rescue-positive-filter",
        choices=["all", "ap75_misses"],
        default="all",
    )
    parser.add_argument("--class-margin-loss-weight", type=float, default=0.0)
    parser.add_argument("--class-margin-margin", type=float, default=0.05)
    parser.add_argument("--bbox-rescue-loss-weight", type=float, default=0.0)
    parser.add_argument("--bbox-rescue-weight-temperature", type=float, default=0.2)
    parser.add_argument(
        "--bbox-localization-loss",
        choices=["smooth_l1", "giou", "diou", "ciou"],
        default="giou",
    )
    parser.add_argument("--chain-topk-per-gt", type=int, default=1)
    parser.add_argument("--chain-bbox-loss-weight", type=float, default=0.0)
    parser.add_argument("--chain-cls-margin-loss-weight", type=float, default=0.0)
    parser.add_argument("--chain-cls-margin-margin", type=float, default=0.1)
    parser.add_argument("--chain-cls-margin-include-background", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--chain-ranking-loss-weight", type=float, default=0.0)
    parser.add_argument("--chain-ranking-margin", type=float, default=0.1)
    parser.add_argument("--chain-dangerous-negative-min-score", type=float, default=0.5)
    parser.add_argument("--verifier-ranking-loss-weight", type=float, default=0.0)
    parser.add_argument("--verifier-ranking-margin", type=float, default=0.1)
    parser.add_argument("--verifier-ranking-positive-iou-min", type=float, default=0.75)
    parser.add_argument("--verifier-ranking-negative-iou-max", type=float, default=0.3)
    parser.add_argument("--verifier-ranking-positive-score-min", type=float, default=0.5)
    parser.add_argument("--verifier-ranking-negative-score-max", type=float, default=0.0)
    parser.add_argument("--verifier-ranking-max-pairs", type=int, default=32)
    parser.add_argument("--nms-aware-ranking-loss-weight", type=float, default=0.0)
    parser.add_argument("--nms-aware-ranking-margin", type=float, default=0.1)
    parser.add_argument("--nms-aware-nms-iou", type=float, default=0.5)
    parser.add_argument(
        "--nms-aware-ranking-mode",
        choices=["joint", "detached_suppressor"],
        default="joint",
    )
    parser.add_argument(
        "--nms-aware-require-suppressor-score-ge-candidate",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--blocked-nms-loss-weight", type=float, default=0.0)
    parser.add_argument("--blocked-nms-score-epsilon", type=float, default=0.02)
    parser.add_argument("--blocked-nms-iou", type=float, default=0.5)
    parser.add_argument("--blocked-nms-base-margin", type=float, default=0.05)
    parser.add_argument("--blocked-nms-iou-margin-scale", type=float, default=0.5)
    parser.add_argument("--blocked-nms-max-margin", type=float, default=0.3)
    parser.add_argument("--blocked-nms-rank-weight", type=float, default=1.0)
    parser.add_argument("--blocked-nms-crossing-weight", type=float, default=1.0)
    parser.add_argument(
        "--blocked-nms-ranking-mode",
        choices=["joint", "detached_suppressor", "delta"],
        default="joint",
    )
    parser.add_argument(
        "--blocked-nms-require-suppressor-score-ge-candidate",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--pre-nms-rescue-loss-weight", type=float, default=0.0)
    parser.add_argument("--pre-nms-score-target", type=float, default=0.2)
    parser.add_argument("--pre-nms-low-conf-max", type=float, default=0.5)
    parser.add_argument("--pre-nms-high-iou-min", type=float, default=0.75)
    parser.add_argument("--pre-nms-topk-per-gt", type=int, default=1)
    parser.add_argument("--pre-nms-dpo-loss-weight", type=float, default=0.0)
    parser.add_argument("--pre-nms-dpo-beta", type=float, default=1.0)
    parser.add_argument("--pre-nms-dpo-min-iou-gap", type=float, default=0.05)
    parser.add_argument("--pre-nms-dpo-max-pairs-per-gt", type=int, default=1)
    parser.add_argument(
        "--pre-nms-dpo-require-rejected-score-ge-chosen",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--same-gt-duplicate-ranking-loss-weight", type=float, default=0.0)
    parser.add_argument("--same-gt-duplicate-ranking-margin", type=float, default=0.1)
    parser.add_argument("--same-gt-duplicate-pair-source", choices=["final", "proposal"], default="final")
    parser.add_argument("--same-gt-duplicate-detach-suppressor", action="store_true")
    parser.add_argument(
        "--same-gt-duplicate-require-suppressor-score-ge-candidate",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--same-gt-duplicate-nms-iou", type=float, default=0.5)
    parser.add_argument("--same-gt-duplicate-min-iou-gap", type=float, default=0.05)
    parser.add_argument(
        "--rescue-verifier-mode",
        choices=["none", "fft", "manifold", "fft_manifold", "raw_ifft", "raw_ifft_hd_fusion", "raw_ifft_scene", "interpretable_residual"],
        default="fft_manifold",
    )
    parser.add_argument("--rescue-verifier-gate", type=float, default=-0.5)
    parser.add_argument("--rescue-verifier-weight-mode", choices=["hard", "sigmoid"], default="hard")
    parser.add_argument("--rescue-verifier-weight-temperature", type=float, default=1.0)
    parser.add_argument("--rescue-fft-weight", type=float, default=0.25)
    parser.add_argument("--rescue-manifold-weight", type=float, default=0.75)
    parser.add_argument("--checkpoint-path", type=str, default=str(CHECKPOINT))
    parser.add_argument("--interpretable-signal-max-size", type=int, default=480)
    parser.add_argument(
        "--interpretable-residual-target",
        choices=["rank_diff", "iou_residual", "rank_underestimation"],
        default="rank_diff",
    )
    parser.add_argument(
        "--interpretable-signal-fusion",
        choices=["new7", "new7_raw_ifft_recipe", "new7_raw_ifft_individual3", "all_signals"],
        default="new7_raw_ifft_recipe",
    )
    parser.add_argument("--interpretable-signal-device", default="cpu")
    parser.add_argument("--interpretable-score-target-precision", type=float, default=0.7)
    parser.add_argument("--rescue-fft-crop-size", type=int, default=32)
    parser.add_argument(
        "--rescue-raw-ifft-features",
        nargs="+",
        default=["fft_edge_truncation@64", "phase_edge@64", "phase_abs_high@11"],
    )
    parser.add_argument("--rescue-raw-ifft-target-precision", type=float, default=0.8)
    parser.add_argument("--rescue-raw-ifft-margin-std-frac", type=float, default=0.0)
    parser.add_argument("--rescue-raw-ifft-score-method", choices=["train_effect_sum", "rank_sum"], default="train_effect_sum")
    parser.add_argument(
        "--rescue-raw-ifft-scene-groups",
        nargs="+",
        default=["maritime", "vehicle", "compact"],
        choices=["all", *SCENE_RAW_IFFT_PRESETS.keys()],
    )
    parser.add_argument("--rescue-raw-ifft-scene-target-precision", type=float, default=0.7)
    parser.add_argument("--rescue-raw-ifft-scene-min-positives", type=int, default=2)
    parser.add_argument("--rescue-hd-fusion-pca-components", type=int, default=96)
    parser.add_argument("--rescue-hd-fusion-hd-scorer", choices=["logistic", "center"], default="logistic")
    parser.add_argument("--rescue-hd-fusion-method", choices=["train_effect", "logistic"], default="train_effect")
    parser.add_argument("--rescue-manifold-k", type=int, default=5)
    parser.add_argument("--rescue-manifold-gate-mode", choices=["legacy", "improved"], default="legacy")
    parser.add_argument("--rescue-manifold-score-mode", choices=["density_ratio", "margin"], default="density_ratio")
    parser.add_argument("--rescue-manifold-fp-weight", type=float, default=1.0)
    parser.add_argument("--rescue-hard-negative-weight", type=float, default=0.0)
    parser.add_argument("--rescue-margin-weight", type=float, default=0.0)
    parser.add_argument(
        "--rescue-manifold-feature-source",
        choices=["box_features", "box_features_l2", "geometry"],
        default="box_features",
    )
    parser.add_argument(
        "--rescue-manifold-feature-projection",
        choices=["identity", "first_half", "second_half", "l2"],
        default="identity",
    )
    parser.add_argument("--rescue-use-bucket-thresholds", action="store_true")
    parser.add_argument("--rescue-use-class-thresholds", action="store_true")
    parser.add_argument("--rescue-class-threshold-min-precision", type=float, default=0.7)
    parser.add_argument("--rescue-class-threshold-min-positives", type=int, default=1)
    parser.add_argument("--rescue-class-threshold-min-threshold", type=float, default=None)
    parser.add_argument("--skip-initial-rescue-diagnostics", action="store_true")
    parser.add_argument("--skip-epoch-rescue-diagnostics", action="store_true")
    parser.add_argument("--skip-final-rescue-diagnostics", action="store_true")
    parser.add_argument("--skip-offline-verifier-report", action="store_true")
    parser.add_argument("--rescue-reference-refresh-epochs", type=int, default=0)
    parser.add_argument("--selection-metric", default="ap75")
    parser.add_argument("--selection-lower-is-better", action="store_true")
    parser.add_argument("--selection-min-delta", type=float, default=0.0)
    parser.add_argument("--safety-max-prediction-ratio", type=float, default=2.0)
    parser.add_argument("--safety-max-prediction-delta", type=int, default=None)
    parser.add_argument("--safety-max-fp-rate-delta", type=float, default=0.02)
    parser.add_argument("--safety-max-high-conf-fp-rate-delta", type=float, default=0.03)
    parser.add_argument("--safety-max-ece-delta", type=float, default=0.03)
    parser.add_argument("--cls-adapter-scale", type=float, default=0.25)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path("runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    args_serializable = _json_safe_args_dict(vars(args))
    save_json(args_serializable, run_dir / "round_config.json")

    train_loader, val_loader = build_loaders(args)
    baseline_model = build_nwpu_model(
        device,
        checkpoint_path=Path(args.checkpoint_path),
        enable_cls_adapter=bool(args.rescue_mode),
        cls_scale=float(args.cls_adapter_scale),
    )
    configure_detector_rollout(
        baseline_model,
        score_threshold=float(args.rollout_score_threshold if args.rescue_mode else args.score_threshold),
        detections_per_img=int(args.rollout_detections_per_img),
    )
    baseline_model.eval()
    for parameter in baseline_model.parameters():
        parameter.requires_grad = False

    model = build_nwpu_model(
        device,
        checkpoint_path=Path(args.checkpoint_path),
        install_adapter=True,
        enable_cls_adapter=bool(args.rescue_mode),
        cls_scale=float(args.cls_adapter_scale),
    )
    configure_detector_rollout(
        model,
        score_threshold=float(args.rollout_score_threshold if args.rescue_mode else args.score_threshold),
        detections_per_img=int(args.rollout_detections_per_img),
    )
    install_residual_bbox_adapter(
        model,
        hidden_dim=128,
        scale=1.0,
        enable_cls_adapter=bool(args.rescue_mode),
        cls_scale=float(args.cls_adapter_scale),
    )
    trainable_names = configure_trainable_parts(model, args)
    save_json({"trainable_parameters": trainable_names}, run_dir / "trainable_parameters.json")

    reference_features, reference_stats = build_rescue_reference(baseline_model, train_loader, device, args)
    save_json(reference_stats, run_dir / "rescue_reference_stats.json")

    baseline_metrics = evaluate_clean_detector(
        baseline_model,
        val_loader,
        device,
        score_threshold=float(args.score_threshold),
        detections_per_img=int(args.eval_detections_per_img),
    )
    save_json(baseline_metrics, run_dir / "baseline_eval_metrics.json")
    if args.rescue_mode and not args.skip_offline_verifier_report:
        verifier_report = collect_offline_verifier_report(
            baseline_model,
            train_loader,
            device,
            args,
            reference_features,
            reference_stats,
        )
        save_json(verifier_report, run_dir / "verifier_offline_report.json")
    if args.rescue_mode and not args.skip_initial_rescue_diagnostics:
        baseline_rescue_diag = collect_rescue_diagnostics(
            baseline_model,
            baseline_model,
            val_loader,
            device,
            args,
            reference_features,
            reference_stats,
        )
        save_json(baseline_rescue_diag, run_dir / "baseline_rescue_diagnostics.json")

    rows = []
    best_metrics = None
    best_decision = None
    best_epoch = None
    checkpoint_cfg = BestCheckpointConfig(
        selection_metric=str(args.selection_metric),
        higher_is_better=not bool(args.selection_lower_is_better),
        min_delta=float(args.selection_min_delta),
        max_prediction_ratio=(
            float(args.safety_max_prediction_ratio)
            if args.safety_max_prediction_ratio is not None
            else None
        ),
        max_prediction_delta=(
            int(args.safety_max_prediction_delta)
            if args.safety_max_prediction_delta is not None
            else None
        ),
        max_fp_rate_delta=(
            float(args.safety_max_fp_rate_delta)
            if args.safety_max_fp_rate_delta is not None
            else None
        ),
        max_high_conf_fp_rate_delta=(
            float(args.safety_max_high_conf_fp_rate_delta)
            if args.safety_max_high_conf_fp_rate_delta is not None
            else None
        ),
        max_ece_delta=(
            float(args.safety_max_ece_delta)
            if args.safety_max_ece_delta is not None
            else None
        ),
    )
    for epoch in range(1, int(args.epochs) + 1):
        row = train_one_epoch(model, baseline_model, train_loader, device, args, reference_features, reference_stats)
        metrics = evaluate_clean_detector(
            model,
            val_loader,
            device,
            score_threshold=float(args.score_threshold),
            detections_per_img=int(args.eval_detections_per_img),
        )
        if args.rescue_mode and not args.skip_epoch_rescue_diagnostics:
            rescue_diag = collect_rescue_diagnostics(
                model,
                baseline_model,
                val_loader,
                device,
                args,
                reference_features,
                reference_stats,
            )
            save_json(rescue_diag, run_dir / f"rescue_diagnostics_epoch_{epoch}.json")
        else:
            rescue_diag = {}
        row.update(
            {
                "epoch": epoch,
                "ap50": metrics["ap50"],
                "ap75": metrics["ap75"],
                "ece": metrics.get("ece", 0.0),
                "num_predictions": metrics.get("num_predictions", 0),
                "false_positive_rate": metrics.get("false_positive_rate", 0.0),
                "lchi_prob_delta_mean": rescue_diag.get("lchi_prob_delta_mean", 0.0),
                "lchi_baseline_label_prob_mean": rescue_diag.get("lchi_baseline_label_prob_mean", 0.0),
                "lchi_current_label_prob_mean": rescue_diag.get("lchi_current_label_prob_mean", 0.0),
                "verifier_positive_lchi_prob_delta_mean": rescue_diag.get(
                    "verifier_positive_lchi_prob_delta_mean",
                    0.0,
                ),
                "verifier_positive_lchi_rate": rescue_diag.get("verifier_positive_lchi_rate", 0.0),
            }
        )
        decision = select_best_checkpoint_update(metrics, baseline_metrics, best_metrics, checkpoint_cfg)
        row.update(
            {
                "best_metric_decision": decision,
                "saved_as_best": bool(decision["should_update_best"]),
            }
        )
        save_checkpoint(
            model,
            run_dir / "checkpoint_last.pth",
            {"run_name": args.run_name, "epoch": epoch, "row": row, "reference_stats": reference_stats},
        )
        if bool(decision["should_update_best"]):
            best_metrics = dict(metrics)
            best_epoch = epoch
            best_decision = dict(decision)
            save_checkpoint(
                model,
                run_dir / "checkpoint_best.pth",
                {
                    "run_name": args.run_name,
                    "epoch": epoch,
                    "metrics": metrics,
                    "decision": decision,
                    "reference_stats": reference_stats,
                },
            )
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if (
            int(args.rescue_reference_refresh_epochs) > 0
            and epoch < int(args.epochs)
            and epoch % int(args.rescue_reference_refresh_epochs) == 0
        ):
            reference_features, reference_stats = build_rescue_reference(model, train_loader, device, args)
            reference_stats["refreshed_after_epoch"] = epoch
            save_json(reference_stats, run_dir / f"rescue_reference_stats_epoch_{epoch}.json")

    final_metrics = evaluate_clean_detector(
        model,
        val_loader,
        device,
        score_threshold=float(args.score_threshold),
        detections_per_img=int(args.eval_detections_per_img),
    )
    save_json(final_metrics, run_dir / "eval_metrics.json")
    if args.rescue_mode and not args.skip_final_rescue_diagnostics:
        final_rescue_diag = collect_rescue_diagnostics(
            model,
            baseline_model,
            val_loader,
            device,
            args,
            reference_features,
            reference_stats,
        )
        save_json(final_rescue_diag, run_dir / "rescue_diagnostics.json")
    save_json(
        {
            "history": rows,
            "best_epoch": best_epoch,
            "best_metrics": best_metrics or {},
            "best_decision": best_decision or {},
        },
        run_dir / "metrics_train.json",
    )

    print(
        json.dumps(
            {
                "run": args.run_name,
                "baseline_ap50": baseline_metrics["ap50"],
                "baseline_ap75": baseline_metrics["ap75"],
                "ap50": final_metrics["ap50"],
                "ap75": final_metrics["ap75"],
                "delta_ap75": final_metrics["ap75"] - baseline_metrics["ap75"],
                "best_epoch": best_epoch,
                "best_ap75": (best_metrics or {}).get("ap75"),
                "pred": final_metrics["num_predictions"],
                "fp_rate": final_metrics["false_positive_rate"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
