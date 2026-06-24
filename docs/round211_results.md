## VOC Clean Metrics
| Group | Mode | AP50 | AP75 | Prec | Recall | ECE | hiFP | #Pred |
|---|---|---|---|---|---|---|---|---|
| round211_voc_v1_baseline_eval | eval_only | N/A | N/A | 0.3191 | 0.8000 | N/A | N/A | 865 |
| round211_voc_v2_posttrain_detection_only | detection_only | N/A | N/A | 0.3067 | 0.8580 | N/A | N/A | 965 |
| round211_voc_v3_posttrain_spatial | spatial | N/A | N/A | 0.3060 | 0.8551 | N/A | N/A | 964 |
| round211_voc_v4_posttrain_spatial_spectral_loggate | spatial_spectral_loggate | N/A | N/A | 0.3054 | 0.8551 | N/A | N/A | 966 |
| round211_voc_v5_posttrain_spatial_shuffled_spectral | spatial_shuffled_spectral | N/A | N/A | 0.3075 | 0.8609 | N/A | N/A | 966 |

## VOC Stress Metrics (AP50 only)
| Group | clean | object_edge | background_texture | near_object |
|---|---|---|---|---|
| round211_voc_v1_baseline_eval | N/A | N/A | N/A | N/A |
| round211_voc_v2_posttrain_detection_only | N/A | N/A | N/A | N/A |
| round211_voc_v3_posttrain_spatial | N/A | N/A | N/A | N/A |
| round211_voc_v4_posttrain_spatial_spectral_loggate | N/A | N/A | N/A | N/A |
| round211_voc_v5_posttrain_spatial_shuffled_spectral | N/A | N/A | N/A | N/A |

## Decision Verdict

**Verdict**: V4 did not beat V3. Spectral evidence is still not useful on a harder detection subset.