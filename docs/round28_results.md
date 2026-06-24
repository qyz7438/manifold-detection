# Round 2.8 AFM Diagnostics Results

| group | AP50 | AP75 | prec | recall | ECE | high_FP | pred | IoU_m | ctr_err | sz_err | dup | mag_s | pha_s | res_s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| round28_g01_baseline_full | 0.8086 | 0.4713 | 0.6944 | 0.8242 | 0.0705 | 3 | 108 | NA | NA | NA | NA | NA | NA | NA |
| round28_g02_old_afm_full | 0.8390 | 0.4470 | 0.3620 | 0.8791 | 0.1282 | 12 | 221 | 0.7404 | 6.3968 | 9.3195 | 15 | NA | NA | NA |
| round28_g03_identity_current_full | 0.8802 | 0.5252 | 0.5541 | 0.9011 | 0.0660 | 6 | 148 | 0.7702 | 6.0233 | 6.8074 | 6 | 0.000000 | 0.000000 | -0.061927 |
| round28_g04_identity_delta_full | 0.8976 | 0.5658 | 0.5185 | 0.9231 | 0.0734 | 6 | 162 | 0.7645 | 5.8178 | 7.3200 | 7 | 0.000000 | 0.000000 | -0.000000 |
| round28_g05_identity_norm_delta_full | 0.8749 | 0.4275 | 0.5000 | 0.9121 | 0.0779 | 8 | 166 | 0.7505 | 5.5955 | 7.9410 | 10 | 0.000000 | 0.000000 | 0.000004 |
| round28_g06_baseline_box_head_only | 0.8628 | 0.6016 | 0.6838 | 0.8791 | 0.0561 | 3 | 117 | 0.8046 | 5.5701 | 6.5812 | 5 | NA | NA | NA |
| round28_g07_identity_current_afm_only | 0.0515 | 0.0152 | 0.0482 | 0.8901 | 0.5116 | 3 | 1682 | 0.6783 | 12.6173 | 14.3458 | 39 | 0.000000 | 0.000000 | -0.381106 |
| round28_g08_identity_current_afm_box_head | 0.8653 | 0.7378 | 0.6154 | 0.8791 | 0.0283 | 4 | 130 | 0.8134 | 5.1481 | 6.1197 | 6 | 0.000000 | 0.000000 | 0.028726 |
| round28_g09_identity_delta_afm_box_head | 0.8650 | 0.7374 | 0.6015 | 0.8791 | 0.0298 | 4 | 133 | 0.8138 | 5.1164 | 6.1022 | 6 | 0.000000 | 0.000000 | 0.000002 |

## Verdict Checklist

- [ ] Frozen parity passed: identity AFM is detector-level no-op before training.
- [ ] Identity delta residual improves AP75/precision over identity current.
- [ ] Old AFM gain, if present, is separated from prediction-count inflation.
- [ ] AFM-only and AFM+box-head training scopes explain whether AFM itself or head adaptation causes drift.
- [ ] Threshold curves identify whether the AP75 drop is localization error or score calibration.
