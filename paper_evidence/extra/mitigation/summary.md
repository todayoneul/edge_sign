# Box-coordinate normalization baseline (road test set, 2,417 frames)

Retention = mAP@0.5:0.95 / FP32 of the same detector. Existing rows come from paper_evidence/runtime/matrix.

| detector | variant | file (MB) | mAP@0.5 | mAP@0.5:0.95 | retention (%) | box/concat scales |
|---|---|---:|---:|---:|---:|---|
| v3 | fp32 | 44.75 | 0.5653 | 0.2808 | 100.0 |  |
| v3 | int8_full | 11.66 | 0.0000 | 0.0000 | 0.0 |  |
| v3 | int8_head_excl | 18.00 | 0.5585 | 0.2772 | 98.7 |  |
| v3 | int8_decode_excl | 11.72 | 0.5399 | 0.2606 | 92.8 |  |
| v3 | post_norm fp32 | 44.75 | 0.5653 | 0.2808 | 100.0 |  |
| v3 | post_norm int8_full | 11.66 | 0.2821 | 0.0958 | 34.1 | Reshape_1_output_0=0.04687, Concat_2_output_0=0.3142, Mul_2_output_0=2.513, briq_boxes_norm=0.003927, output0_QuantizeLinear_Input=0.003927 |
| v3 | pre_norm fp32 | 44.92 | 0.5653 | 0.2808 | 100.0 |  |
| v3 | pre_norm int8_full | 11.66 | 0.3867 | 0.1545 | 55.0 | Reshape_1_output_0=0.04687, briq_dist_norm=0.001172, Concat_2_output_0=0.003927, output0_QuantizeLinear_Input=0.003927 |
| v4 | fp32 | 9.81 | 0.4999 | 0.2422 | 100.0 |  |
| v4 | int8_full | 3.10 | 0.0000 | 0.0000 | 0.0 |  |
| v4 | int8_head_excl | 3.42 | 0.4826 | 0.2350 | 97.0 |  |
| v4 | int8_decode_excl | 3.16 | 0.4755 | 0.2284 | 94.3 |  |
| v4 | post_norm fp32 | 9.81 | 0.4999 | 0.2422 | 100.0 |  |
| v4 | post_norm int8_full | 3.10 | 0.3223 | 0.1027 | 42.4 | Concat_output_0=0.04526, Concat_2_output_0=0.3347, Mul_2_output_0=3.716, briq_boxes_norm=0.005807, Concat_3_output_0=0.005807, output0_QuantizeLinear_Input=0.003966 |
| v4 | pre_norm fp32 | 9.97 | 0.4999 | 0.2422 | 100.0 |  |
| v4 | pre_norm int8_full | 3.10 | 0.3469 | 0.1233 | 50.9 | Concat_output_0=0.04526, briq_dist_norm=0.001842, Concat_2_output_0=0.005807, Concat_3_output_0=0.005807, output0_QuantizeLinear_Input=0.003966 |
| v4 | fold_norm fp32 | 9.94 | 0.4999 | 0.2422 | 100.0 |  |
| v4 | fold_norm int8_full | 3.09 | 0.3464 | 0.1235 | 51.0 | Concat_output_0=0.001842, Concat_2_output_0=0.005807, Concat_3_output_0=0.005807, output0_QuantizeLinear_Input=0.003966 |
