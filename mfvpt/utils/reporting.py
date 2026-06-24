from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


METHOD_NAMES = {
    "eval_baseline_ft": "Baseline FT",
    "eval_standard_aug_ft": "Standard Aug FT",
    "eval_fourier_aug_ft": "Fourier Aug FT",
    "eval_mfvpt_posttrain": "MFVPT Post-train",
}


def collect_eval_results(runs_dir: str | Path = "runs") -> pd.DataFrame:
    root = Path(runs_dir)
    rows = []
    for metrics_path in root.glob("eval_*/eval_metrics.json"):
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        run_name = metrics_path.parent.name
        metrics["run_name"] = run_name
        metrics["method"] = METHOD_NAMES.get(run_name, run_name)
        rows.append(metrics)
    return pd.DataFrame(rows)


def write_report(runs_dir: str | Path = "runs") -> None:
    root = Path(runs_dir)
    df = collect_eval_results(root)
    if df.empty:
        return
    root.mkdir(parents=True, exist_ok=True)
    df.to_csv(root / "report.csv", index=False)
    columns = [
        "method",
        "clean_acc",
        "low_acc",
        "high_acc",
        "patch_acc",
        "cons_low",
        "cons_high",
        "cons_patch",
        "hce_patch",
        "ece_patch",
    ]
    present = [col for col in columns if col in df.columns]
    markdown_df = df[present]
    header = "| " + " | ".join(markdown_df.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(markdown_df.columns)) + " |"
    rows = []
    for _, row in markdown_df.iterrows():
        cells = []
        for value in row.tolist():
            if isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        rows.append("| " + " | ".join(cells) + " |")
    markdown = "\n".join([header, separator, *rows])
    note = (
        "\n\n结论判断规则：如果 MFVPT 的 patch_acc、high_acc、cons_patch 高于 baseline，"
        "且 clean_acc 下降不超过 2 个百分点，则第一版假设得到初步支持。"
        "如果 HCE/ECE 不下降，则不能声称校准改善。\n"
    )
    (root / "report.md").write_text(markdown + note, encoding="utf-8")
