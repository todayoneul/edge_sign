## Accuracy (ORT CPU, independent test)

| model | MB | mAP50 or top-1 | mAP50-95 | retention | >=95% | >=99% |
|---|---:|---:|---:|---:|:-:|:-:|
| v4_fp32 | 9.806 | 0.4999 | 0.2422 | 1.000 | Y | Y |
| v4_fp16 | 4.972 | 0.4986 | 0.2418 | 0.998 | Y | Y |
| v4_int8_full | 3.099 | 0.0000 | 0.0000 | 0.000 | N | N |
| v4_int8_head_excl | 3.416 | 0.4826 | 0.2350 | 0.970 | Y | N |
| v4_int8_full_fbias | 2.994 | 0.0000 | 0.0000 | 0.000 | N | N |
| v4_int8_head_excl_fbias | 3.330 | 0.4824 | 0.2352 | 0.971 | Y | N |
| v3_fp32 | 44.747 | 0.5653 | 0.2808 | 1.000 | Y | Y |
| v3_fp16 | 22.382 | 0.5656 | 0.2805 | 0.999 | Y | Y |
| v3_int8_full | 11.656 | 0.0000 | 0.0000 | 0.000 | N | N |
| v3_int8_head_excl | 17.998 | 0.5585 | 0.2772 | 0.987 | Y | N |
| v3_int8_full_fbias | 11.552 | 0.0000 | 0.0000 | 0.000 | N | N |
| v3_int8_head_excl_fbias | 17.912 | 0.5584 | 0.2771 | 0.987 | Y | N |
| rec_fp32 | 0.117 | 0.8082 | - | 1.000 | Y | Y |
| rec_fp16 | 0.060 | 0.8080 | - | 1.000 | Y | Y |
| rec_int8_full | 0.042 | 0.8084 | - | 1.000 | Y | Y |
| rec_int8_full_fbias | 0.039 | 0.8084 | - | 1.000 | Y | Y |
| rec_int8_head_excl | 0.043 | 0.8058 | - | 0.997 | Y | Y |
| rec_int8_head_excl_fbias | 0.041 | 0.8058 | - | 0.997 | Y | Y |
| coco_fp32 | 101.744 | 0.7091 | 0.5401 | 1.000 | Y | Y |
| coco_fp16 | 50.896 | 0.7091 | 0.5404 | 1.000 | Y | Y |
| coco_int8_full | 26.724 | 0.0000 | 0.0000 | 0.000 | N | N |
| coco_int8_full_fbias | 26.408 | 0.0000 | 0.0000 | 0.000 | N | N |
| coco_int8_head_excl | 31.053 | 0.7010 | 0.5356 | 0.992 | Y | Y |
| coco_int8_head_excl_fbias | 30.781 | 0.7014 | 0.5351 | 0.991 | Y | Y |

## Latency mean / p90 ms, batch 1 (detector: A = p90<=33.3, B = p90<=66.7)

| model | cpu_t1 | cpu_t4 | webgpu_1220 | wasm_1300 | wasmt4_1300 | webgpu_1300 |
|---|---:|---:|---:|---:|---:|---:|
| v4_fp32 | 80.4 / 81.3 | 29.7 / 30.6 A | 14.8 / 15.3 A | 156.2 / 159.7 | 47.9 / 52.9 B | 15.4 / 15.7 A |
| v4_fp16 | 83.7 / 84.6 | 31.9 / 32.7 A | 23.6 / 30.3 A | 164.1 / 168.3 | 48.6 / 50.6 B | 12.9 / 13.3 A |
| v4_int8_full | - | - | - | - | - | - |
| v4_int8_head_excl | 40.0 / 40.5 B | 17.7 / 17.9 A | - | 171.4 / 174.6 | 53.7 / 55.1 B | 215.5 / 222.7 |
| v4_int8_full_fbias | - | - | - | - | - | - |
| v4_int8_head_excl_fbias | 40.7 / 41.3 B | 17.7 / 18.0 A | - | 173.1 / 178.1 | 55.5 / 58.1 B | 214.5 / 220.1 |
| v3_fp32 | 331.6 / 334.0 | 104.8 / 105.9 | - | 710.0 / 730.0 | 193.4 / 201.8 | 30.9 / 30.8 A |
| v3_fp16 | 336.4 / 338.5 | 107.4 / 110.0 | - | 698.3 / 708.0 | 197.2 / 209.4 | 24.9 / 24.7 A |
| v3_int8_full | - | - | - | - | - | - |
| v3_int8_head_excl | 166.7 / 167.6 | 53.3 / 53.7 B | - | 703.4 / 707.0 | 203.0 / 210.6 | 233.5 / 245.5 |
| v3_int8_full_fbias | - | - | - | - | - | - |
| v3_int8_head_excl_fbias | 166.5 / 167.5 | 53.3 / 53.7 B | - | 712.3 / 724.7 | 199.2 / 202.5 | 233.7 / 250.7 |
| rec_fp32 | 0.10 / 0.11 | 0.07 / 0.07 | - | 0.21 / 0.22 | 0.13 / 0.20 | 1.26 / 1.36 |
| rec_fp16 | 0.10 / 0.11 | 0.07 / 0.07 | - | 0.22 / 0.23 | 0.16 / 0.21 | 1.41 / 1.57 |
| rec_int8_full | 0.06 / 0.06 | 0.04 / 0.05 | - | 0.27 / 0.29 | 0.17 / 0.23 | 5.64 / 6.14 |
| rec_int8_full_fbias | 0.06 / 0.06 | 0.05 / 0.05 | - | 0.27 / 0.29 | 0.21 / 0.29 | 5.42 / 5.88 |
| rec_int8_head_excl | - | - | - | - | - | - |
| rec_int8_head_excl_fbias | - | - | - | - | - | - |
| coco_fp32 | - | 320.1 / 323.2 | - | - | 564.8 / 566.8 | 80.6 / 81.0 |
| coco_fp16 | - | - | - | - | - | 61.9 / 63.5 B |
| coco_int8_full | - | - | - | - | - | - |
| coco_int8_full_fbias | - | - | - | - | - | - |
| coco_int8_head_excl | - | 117.3 / 118.2 | - | - | 583.1 / 584.9 | 721.8 / 742.1 |
| coco_int8_head_excl_fbias | - | - | - | - | - | - |

## WebGPU node placement (WebGPU nodes / CPU nodes)

| model | ORT 1220 | CPU op types (1220) | ORT 1300 | CPU op types (1300) |
|---|---:|---|---:|---|
| v4_fp32 | 399/11 | TopKx2, Unsqueezex2, Tilex2, Flattenx1, Divx1, Modx1, Gatherx1, Castx1 | 226/9 | Unsqueezex2, Tilex2, Flattenx1, Divx1, Modx1, Gatherx1, Castx1 |
| v4_fp16 | 427/29 | Splitx12, Castx5, TopKx2, Unsqueezex2, Tilex2, GatherElementsx2, Flattenx1, Divx1, Modx1, Gatherx1 | 228/9 | Unsqueezex2, Tilex2, Flattenx1, Divx1, Modx1, Gatherx1, Castx1 |
| v4_int8_full | - | - | - | - |
| v4_int8_head_excl | - | - | 1571/325 | QuantizeLinearx312, Transposex4, Unsqueezex2, Tilex2, Flattenx1, Divx1, Modx1, Gatherx1, Castx1 |
| v4_int8_full_fbias | - | - | - | - |
| v4_int8_head_excl_fbias | - | - | 1571/325 | QuantizeLinearx312, Transposex4, Unsqueezex2, Tilex2, Flattenx1, Divx1, Modx1, Gatherx1, Castx1 |
| v3_fp32 | - | - | 133/0 | - |
| v3_fp16 | - | - | 139/0 | - |
| v3_int8_full | - | - | - | - |
| v3_int8_head_excl | - | - | 913/181 | QuantizeLinearx177, Transposex4 |
| v3_int8_full_fbias | - | - | - | - |
| v3_int8_head_excl_fbias | - | - | 913/181 | QuantizeLinearx177, Transposex4 |
| rec_fp32 | - | - | 12/0 | - |
| rec_fp16 | - | - | 14/0 | - |
| rec_int8_full | - | - | 57/12 | QuantizeLinearx12 |
| rec_int8_full_fbias | - | - | 57/12 | QuantizeLinearx12 |
| rec_int8_head_excl | - | - | - | - |
| rec_int8_head_excl_fbias | - | - | - | - |
| coco_fp32 | - | - | - | - |
| coco_fp16 | - | - | - | - |
| coco_int8_full | - | - | - | - |
| coco_int8_full_fbias | - | - | - | - |
| coco_int8_head_excl | - | - | - | - |
| coco_int8_head_excl_fbias | - | - | 2791/560 | QuantizeLinearx556, Transposex4 |

## Browser numeric parity (vs ORT CPU, same frames and decoder)

| model | runtime | mAP50 | delta | mAP50-95 | delta |
|---|---|---:|---:|---:|---:|
| v4_fp32 | webgpu_1300 | 0.499865 | -0.000000 | 0.242181 | +0.000000 |
| v4_fp16 | webgpu_1300 | 0.499071 | +0.000466 | 0.242121 | +0.000330 |
| v3_fp32 | webgpu_1300 | 0.565269 | -0.000001 | 0.280751 | -0.000000 |
| v3_fp16 | webgpu_1300 | 0.564541 | -0.001083 | 0.279280 | -0.001184 |

| configuration | launches | det | rec | mean: median [min-max] | p90: median [min-max] | A 30 FPS | B 15 FPS | pooled p90 (check) |
|---|---:|---:|---:|---:|---:|:-:|:-:|---:|
| v4_fp32@webgpu+rec_fp32@wasm | 5 | 15.15 | 0.20 | 16.69 [16.64-16.69] | 17.49 [17.41-17.65] | Y (5/5) | Y (5/5) | 17.51 (same verdict) |
| v4_fp16@webgpu+rec_fp32@wasm | 5 | 12.70 | 0.20 | 15.77 [15.75-15.84] | 16.56 [16.48-16.71] | Y (5/5) | Y (5/5) | 16.59 (same verdict) |
| v4_fp16@webgpu+rec_int8_full@wasm | 1 | 12.75 | 0.27 | 15.91 [15.91-15.91] | 16.80 [16.80-16.80] | Y (1/1) | Y (1/1) | 16.80 (same verdict) |
| v4_fp32@webgpu+rec_fp32@webgpu | 5 | 15.16 | 0.67 | 17.20 [17.15-17.54] | 18.24 [18.19-18.44] | Y (5/5) | Y (5/5) | 18.26 (same verdict) |
| v4_fp16@webgpu+rec_fp16@webgpu | 1 | 12.89 | 0.93 | 16.79 [16.79-16.79] | 17.84 [17.84-17.84] | Y (1/1) | Y (1/1) | 17.84 (same verdict) |
| v4_int8_head_excl@wasm+rec_int8_full@wasm | 5 | 52.91 | 0.24 | 54.57 [54.40-54.77] | 55.39 [55.18-55.59] | N (0/5) | Y (5/5) | 55.38 (same verdict) |
| v4_fp32@wasm+rec_fp32@wasm | 1 | 44.06 | 0.17 | 45.56 [45.56-45.56] | 46.40 [46.40-46.40] | N (0/1) | Y (1/1) | 46.40 (same verdict) |

| paired comparison (a - b, per round) | rounds | mean diff ms | [min, max] | rounds a faster |
|---|---:|---:|---:|---:|
| recognizer on WASM instead of WebGPU (FP32 detector on WebGPU) | 5 | -0.59 | [-0.85, -0.48] | 5/5 |
| FP16 instead of FP32 detector on WebGPU (recognizer FP32 on WASM) | 5 | -0.89 | [-0.94, -0.81] | 5/5 |
| INT8 instead of FP32 recognizer on WASM (FP16 detector on WebGPU) | 1 | +0.12 | [+0.12, +0.12] | 0/1 |
| all FP16 instead of all FP32 on WebGPU | 1 | -0.76 | [-0.76, -0.76] | 1/1 |
| INT8 head-excluded instead of FP32 detector on WASM | 1 | +8.94 | [+8.94, +8.94] | 0/1 |
