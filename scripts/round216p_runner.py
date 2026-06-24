"""Plan 2.16+: mid06 + frozen, 3 epoch x 3 seed, clean + object_edge eval."""
import subprocess
import sys
import json
from pathlib import Path

import torch
import torch.nn.functional as F

PYTHON = sys.executable
SCRIPT = "scripts/round28_train_eval.py"
GIT_HASH = subprocess.run(
    ["git", "rev-parse", "HEAD"], capture_output=True, text=True
).stdout.strip()

CONFIGS = [
    ("round216p_mid06", "mplseg_mid"),
    ("round216p_frozen", "mplseg_frozen"),
]
SEEDS = [42, 123, 456]


def train_and_eval_clean(run_name, afm_type, seed):
    cmd = [
        PYTHON, SCRIPT,
        "--run-name", run_name, "--afm-type", afm_type,
        "--trainable-mode", "full", "--epochs", "3", "--seed", str(seed),
    ]
    print(f"  TRAIN {run_name} ...", end=" ", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("CRASH")
        return None
    m = Path(f"runs/{run_name}/eval_metrics.json")
    if not m.exists():
        print("NO_METRICS")
        return None
    d = json.loads(m.read_text())
    d["git_hash"] = GIT_HASH
    d["cli_train"] = " ".join(cmd)
    m.write_text(json.dumps(d, indent=2, ensure_ascii=False))
    print(f"AP50={d['ap50']:.4f} AP75={d['ap75']:.4f}")
    return d


def eval_edge(run_name, seed, afm_type):
    """Evaluate trained model on object_edge checkerboard patches."""
    from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
    from spectral_detection_posttrain.datasets.patch_transform import add_detection_patch
    from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
    from spectral_detection_posttrain.models import build_detector
    from spectral_detection_posttrain.utils.io import load_checkpoint
    from spectral_detection_posttrain.utils.seed import resolve_device, set_seed

    set_seed(seed)
    model_cfg = {"name": "fasterrcnn_mobilenet_v3_large_320_fpn", "pretrained": True,
                 "num_classes": 2, "min_size": 320, "max_size": 320,
                 "afm_channels": 256, "afm_type": afm_type}
    config = {
        "seed": seed, "device": "auto",
        "data": {"root": "./data", "download": True, "max_size": 320,
                 "train_fraction": 0.8, "num_workers": 0},
        "model": model_cfg,
        "train": {"batch_size": 2},
    }
    device = resolve_device(config)
    _, val_loader = build_penn_fudan_loaders(config)

    ckpt = f"runs/{run_name}/checkpoint_last.pth"
    model = build_detector(config).to(device)
    load_checkpoint(model, ckpt, device)
    model.eval()

    preds, targets_list = [], []
    for images, batch_targets in val_loader:
        stressed = [add_detection_patch(img, tgt, placement="object_edge",
                     patch_type="checkerboard", patch_size=48)
                    for img, tgt in zip(images, batch_targets)]
        outputs = model([s.to(device) for s in stressed])
        preds.extend([{k: v.detach().cpu() for k, v in o.items()} for o in outputs])
        targets_list.extend([{k: v.detach().cpu() if torch.is_tensor(v) else v
                              for k, v in t.items()} for t in batch_targets])

    m = evaluate_detection_predictions(preds, targets_list, iou_threshold=0.5, score_threshold=0.05)
    return {"edge_ap50": m["ap50"], "edge_ap75": m["ap75"],
            "edge_precision": m["precision"], "edge_recall": m["recall"],
            "edge_ece": m["ece"]}


def main():
    print(f"Plan 2.16+: mid06 + frozen, 3 epoch x 3 seed, clean + edge")
    print(f"git: {GIT_HASH}\n")

    all_results = []
    for cfg_name, afm_type in CONFIGS:
        label = "mid06" if "mid06" in cfg_name else "frozen"
        print(f"\n-- {label} ({afm_type}) 3 epochs --")
        for seed in SEEDS:
            run_name = f"{cfg_name}_s{seed}"
            d = train_and_eval_clean(run_name, afm_type, seed)
            if d is None:
                all_results.append({"run": run_name, "status": "CRASH"})
                continue

            print(f"  EDGE {run_name} ...", end=" ", flush=True)
            edge = eval_edge(run_name, seed, afm_type)
            d.update(edge)
            Path(f"runs/{run_name}/eval_metrics.json").write_text(
                json.dumps(d, indent=2, ensure_ascii=False))
            print(f"edge_AP50={edge['edge_ap50']:.4f} edge_AP75={edge['edge_ap75']:.4f}")

            h = d.get("history", [{}])
            last_epoch = h[-1] if h else {}
            result = {
                "run": run_name, "afm": label, "seed": seed, "status": "OK",
                "ap50": d["ap50"], "ap75": d["ap75"],
                "prec": d["precision"], "rec": d["recall"], "ece": d["ece"],
                "edge_ap50": edge["edge_ap50"], "edge_ap75": edge["edge_ap75"],
                "edge_prec": edge["edge_precision"], "edge_rec": edge["edge_recall"],
                "edge_ece": edge["edge_ece"],
                "r_scale": last_epoch.get("residual_scale", "N/A"),
                "train_loss": last_epoch.get("train_loss", "N/A"),
            }
            all_results.append(result)

    # Format tables
    lines = ["## Plan 2.16+: 3 epoch, clean + object_edge", ""]
    lines.append("### Clean (3 epoch)")
    lines.append("| Run | AP50 | AP75 | Prec | Recall | ECE | r_scale |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for r in all_results:
        if r["status"] != "OK": continue
        rs = f"{r['r_scale']:.4f}" if isinstance(r['r_scale'], float) else "-"
        lines.append(f'| {r["run"]} | {r["ap50"]:.4f} | {r["ap75"]:.4f} | {r["prec"]:.4f} | {r["rec"]:.4f} | {r["ece"]:.4f} | {rs} |')

    lines.append("")
    lines.append("### Object-edge stress")
    lines.append("| Run | AP50 | AP75 | Prec | Recall | ECE |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for r in all_results:
        if r["status"] != "OK": continue
        lines.append(f'| {r["run"]} | {r["edge_ap50"]:.4f} | {r["edge_ap75"]:.4f} | {r["edge_prec"]:.4f} | {r["edge_rec"]:.4f} | {r["edge_ece"]:.4f} |')

    # Summary
    lines.append("")
    lines.append("### Mean over 3 seeds")
    lines.append("| Config | AP50 | AP75 | ECE | edge_AP50 | edge_AP75 | edge_ECE |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for label in ["mid06", "frozen"]:
        ok = [r for r in all_results if r["afm"] == label and r["status"] == "OK"]
        if len(ok) < 3:
            continue
        def mean(k):
            vv = [r[k] for r in ok if r.get(k) is not None]
            return sum(vv) / len(vv) if vv else 0
        lines.append(f'| {label} | {mean("ap50"):.4f} | {mean("ap75"):.4f} | {mean("ece"):.4f} | {mean("edge_ap50"):.4f} | {mean("edge_ap75"):.4f} | {mean("edge_ece"):.4f} |')

    msg = "\n".join(lines)
    print(f"\n{msg}")

    subprocess.run([PYTHON, "scripts/notify_feishu.py",
                    f"Plan 2.16+ done: {sum(1 for r in all_results if r['status']=='OK')}/{len(all_results)} OK. git={GIT_HASH[:8]}"],
                   capture_output=True)
    subprocess.run([PYTHON, "scripts/notify_feishu.py", msg[:800]], capture_output=True)
    print("\nSent to Feishu.")


if __name__ == "__main__":
    main()
