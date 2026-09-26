# Calibration robustness (road test set, 2,417 frames)

MinMax rows are the paper's existing models. Retention = mAP@0.5:0.95 / FP32.

| detector | scope | calibration | mAP@0.5 | mAP@0.5:0.95 | retention (%) | output-concat / box scales |
|---|---|---|---:|---:|---:|---|
| v3 | full | minmax | 0.0000 | 0.0000 | 0.0 |  |
| v3 | full | percentile | 0.0000 | 0.0000 | 0.0 | Reshape_1_output_0=0.04637, Concat_2_output_0=0.3129, Mul_2_output_0=2.504, output0_QuantizeLinear_Input=2.504 |
| v3 | full | entropy | 0.0000 | 0.0000 | 0.0 | Reshape_1_output_0=0.04687, Concat_2_output_0=0.3142, Mul_2_output_0=2.513, output0_QuantizeLinear_Input=2.513 |
| v3 | head_excl | minmax | 0.5585 | 0.2772 | 98.7 |  |
| v3 | head_excl | percentile | 0.5596 | 0.2779 | 99.0 |  |
| v3 | head_excl | entropy | 0.5250 | 0.2607 | 92.9 |  |
| v4 | full | minmax | 0.0000 | 0.0000 | 0.0 |  |
| v4 | full | percentile | 0.0000 | 0.0000 | 0.0 | Concat_output_0=0.04143, Concat_2_output_0=0.3329, Mul_2_output_0=3.611, Concat_3_output_0=3.581, output0_QuantizeLinear_Input=2.536 |
| v4 | full | entropy | 0.0000 | 0.0000 | 0.0 | Concat_output_0=0.04041, Concat_2_output_0=0.3347, Mul_2_output_0=3.533, Concat_3_output_0=3.34, output0_QuantizeLinear_Input=2.538 |
| v4 | head_excl | minmax | 0.4826 | 0.2350 | 97.0 |  |
| v4 | head_excl | percentile | 0.4883 | 0.2382 | 98.4 |  |
| v4 | head_excl | entropy | 0.4727 | 0.2315 | 95.6 |  |
