# Results aggregation

## Per-run results

| run | dataset | backbone | method | seed | epochs | best_ap50 | ap50 | ap75 | ece |
|---|---|---|---|---|---|---|---|---|---|
| voc_mob_baseline_s42_12ep | voc | mob | baseline | 42 | 12 | 0.705259667077811 | 0.6969988243537477 | 0.46197073437517744 | 0.06905207993946294 |
| voc_resnet_baseline_s42_12ep_bs8 | voc | resnet | baseline | 42 | 12 | 0.7316358526708033 | 0.704461120116678 | 0.3570078993866569 | 0.15787991553913983 |

## Group statistics (over seeds)

| dataset | backbone | method | epochs | n | best_ap50_mean | best_ap50_std | ap50_mean | ap50_std | ap75_mean | ap75_std | ece_mean | ece_std |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| voc | mob | baseline | 12 | 1 | 0.7053 | 0.0000 | 0.6970 | 0.0000 | 0.4620 | 0.0000 | 0.0691 | 0.0000 |
| voc | resnet | baseline | 12 | 1 | 0.7316 | 0.0000 | 0.7045 | 0.0000 | 0.3570 | 0.0000 | 0.1579 | 0.0000 |
