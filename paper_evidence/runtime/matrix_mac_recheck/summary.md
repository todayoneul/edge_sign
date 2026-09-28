## WASM FP32 vs INT8 head-excluded, repeated fresh launches

| models | WASM threads (seen) | rounds | FP32 mean: median | INT8 mean: median | FP32/INT8 per round | median [min-max] | rounds INT8 faster |
|---|---|---:|---:|---:|---|---:|---:|
| v4_fp32 vs v4_int8_head_excl | 4 (4) | 5 | 43.6 | 52.2 | 0.84, 0.84, 0.84, 0.84, 0.84 | 0.84 [0.84-0.84] | 0/5 |
| v3_fp32 vs v3_int8_head_excl | 4 (4) | 3 | 187.2 | 196.8 | 0.95, 0.95, 0.95 | 0.95 [0.95-0.95] | 0/3 |

## Browser pipeline: FP32 vs INT8 on WASM 4T, repeated launches

| configuration | launches | det | rec | mean: median [min-max] | p90: median [min-max] | A 30 FPS | B 15 FPS | pooled p90 (check) |
|---|---:|---:|---:|---:|---:|:-:|:-:|---:|
| v4_fp16@webgpu+rec_fp32@wasm | 5 | 12.53 | 0.19 | 15.53 [15.47-15.59] | 16.15 [15.98-16.27] | Y (5/5) | Y (5/5) | 16.14 (same verdict) |
| v4_fp16@webgpu+rec_fp16@webgpu | 5 | 12.59 | 0.67 | 16.04 [15.95-16.33] | 16.89 [16.74-17.15] | Y (5/5) | Y (5/5) | 16.95 (same verdict) |
| v4_int8_head_excl@wasm+rec_int8_full@wasm | 5 | 52.14 | 0.21 | 53.69 [53.68-53.81] | 54.46 [54.43-54.50] | N (0/5) | Y (5/5) | 54.47 (same verdict) |
| v4_fp32@wasm+rec_fp32@wasm | 5 | 43.32 | 0.16 | 44.78 [44.69-45.15] | 45.31 [45.17-45.58] | N (0/5) | Y (5/5) | 45.38 (same verdict) |

| paired comparison (a - b, per round) | rounds | mean diff ms | [min, max] | rounds a faster |
|---|---:|---:|---:|---:|
| INT8 head-excluded instead of FP32 detector on WASM | 5 | +8.86 | [+8.54, +9.01] | 0/5 |

## Quiet gate (see quiet.log)

measurements: 36, not quiet after 5 min: 0

## Browser numeric parity (vs ORT CPU, same frames and decoder)

| model | runtime | mAP50 | delta | mAP50-95 | delta |
|---|---|---:|---:|---:|---:|
| v4_fp32 | wasmt4_1300 | 0.499865 | -0.000000 | 0.242181 | +0.000000 |
| v4_int8_head_excl | wasmt4_1300 | 0.483577 | +0.000977 | 0.235839 | +0.000811 |
| v4_int8_head_excl_fbias | wasmt4_1300 | 0.482775 | +0.000339 | 0.235393 | +0.000227 |
