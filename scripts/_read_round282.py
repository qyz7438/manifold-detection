import json, numpy as np
from pathlib import Path

for cfg in ['det_only_unf', 'c_gated', 'c_gated_shuffle']:
    for seed in [42, 123, 456]:
        f = Path(f'runs/round282_{cfg}_s{seed}/eval_metrics.json')
        if not f.exists(): continue
        d = json.loads(f.read_text())
        print(f'--- {cfg} seed={seed} ---')
        for row in d['history']:
            ap75 = row['val_ap75']
            rs = row.get('reward_std', 0)
            en_gap = row.get('energy_gap', float('nan'))
            ng = row.get('n_gated', 0)
            print(f'  e{row["epoch"]:2d}: AP75={ap75:.4f} r_std={rs:.4f} en_gap={en_gap:.4f} n_gated={ng}')
        print()
