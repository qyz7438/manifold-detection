"""Plan 2.19: Run all 12 groups, send results to Feishu."""
import subprocess, sys, json
from pathlib import Path

PY = sys.executable
SCRIPT = "scripts/round28_train_eval.py"
GIT = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
SEEDS = [42, 123, 456]

GROUPS = [
    # (run_name, afm_type, dataset, model_name, epochs)
    # Penn-Fudan + ResNet50
    ("round219_r50_pf_baseline_s{}", "none", "penn_fudan", "fasterrcnn_resnet50_fpn", 3),
    ("round219_r50_pf_mid06_s{}", "mplseg_mid", "penn_fudan", "fasterrcnn_resnet50_fpn", 3),
    # VOC + MobileNetV3
    ("round219_voc_mob_baseline_s{}", "none", "voc", "fasterrcnn_mobilenet_v3_large_320_fpn", 3),
    ("round219_voc_mob_mid06_s{}", "mplseg_mid", "voc", "fasterrcnn_mobilenet_v3_large_320_fpn", 3),
]


def run_one(run_name, afm_type, dataset, model_name, epochs, seed):
    cmd = [
        PY, SCRIPT,
        "--run-name", run_name,
        "--afm-type", afm_type,
        "--dataset", dataset,
        "--model-name", model_name,
        "--trainable-mode", "full",
        "--epochs", str(epochs),
        "--seed", str(seed),
    ]
    if dataset == "voc":
        cmd.extend(["--limit-train", "300", "--limit-val", "150"])
    print(f"  {run_name} ...", end=" ", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"CRASH: {r.stderr[-150:]}")
        return {"run": run_name, "status": "CRASH"}
    m = Path(f"runs/{run_name}/eval_metrics.json")
    if m.exists():
        d = json.loads(m.read_text())
        d["git_hash"] = GIT
        d["model_name"] = model_name
        d["dataset"] = dataset
        m.write_text(json.dumps(d, indent=2, ensure_ascii=False))
        print(f"AP50={d['ap50']:.4f} AP75={d['ap75']:.4f}")
        return {"run": run_name, "status": "OK", "ap50": d["ap50"], "ap75": d["ap75"],
                "prec": d["precision"], "ece": d["ece"]}
    print("no_metrics")
    return {"run": run_name, "status": "NO_METRICS"}


def main():
    all_r = []
    for name_template, afm_type, dataset, model_name, epochs in GROUPS:
        label = name_template.replace("_s{}", "").replace("round219_", "")
        print(f"\n-- {label} ({model_name}, {dataset}) --")
        for seed in SEEDS:
            run_name = name_template.format(seed)
            r = run_one(run_name, afm_type, dataset, model_name, epochs, seed)
            all_r.append(r)

    # Summary
    lines = ["## Plan 2.19 Results", ""]
    lines.append("| Run | AP50 | AP75 | Prec | ECE |")
    lines.append("|---:|---:|---:|---:|---:|")
    for r in all_r:
        if r["status"] != "OK":
            lines.append(f"| {r['run']} | {r['status']} | | | |")
        else:
            lines.append(f"| {r['run']} | {r['ap50']:.4f} | {r['ap75']:.4f} | {r['prec']:.4f} | {r['ece']:.4f} |")
    msg = "\n".join(lines)
    print(f"\n{msg}")
    subprocess.run([PY, "scripts/notify_feishu.py", f"Plan 2.19: {sum(1 for r in all_r if r['status']=='OK')}/{len(all_r)} OK"], capture_output=True)
    subprocess.run([PY, "scripts/notify_feishu.py", msg[:800]], capture_output=True)


if __name__ == "__main__":
    main()
