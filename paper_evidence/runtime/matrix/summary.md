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
| v4_fp32 | 34.1 / 37.3 B | 19.2 / 20.4 A | 14.7 / 16.7 A | 157.1 / 164.6 | 49.8 / 55.7 B | 14.3 / 16.8 A |
| v4_fp16 | 33.5 / 36.6 B | 20.0 / 20.9 A | 54.3 / 58.1 B | 164.3 / 172.0 | 52.0 / 57.9 B | 13.6 / 15.6 A |
| v4_int8_full | 26.6 / 28.3 A | 19.2 / 19.8 A | - | 92.2 / 101.6 | 35.4 / 40.4 B | 1184.6 / 1233.8 |
| v4_int8_head_excl | 28.2 / 29.6 A | 20.2 / 20.8 A | - | 90.8 / 98.1 | 37.2 / 41.8 B | 943.5 / 970.3 |
| v4_int8_full_fbias | 27.3 / 30.5 A | 19.2 / 19.8 A | - | 88.2 / 94.8 | 35.1 / 39.7 B | 1184.1 / 1232.2 |
| v4_int8_head_excl_fbias | 28.3 / 29.8 A | 20.7 / 22.5 A | - | 97.9 / 106.7 | 35.8 / 40.3 B | 917.8 / 961.1 |
| v3_fp32 | 118.1 / 121.2 | 60.4 / 63.1 B | 11.6 / 12.7 A | 671.3 / 702.6 | 203.0 / 214.3 | 11.9 / 13.0 A |
| v3_fp16 | 116.4 / 120.2 | 60.5 / 62.7 B | 35.2 / 37.4 B | 679.9 / 702.6 | 204.6 / 215.7 | 12.1 / 13.2 A |
| v3_int8_full | 54.5 / 56.9 B | 33.2 / 35.3 B | - | 294.3 / 314.4 | 94.6 / 103.0 | 792.3 / 821.2 |
| v3_int8_head_excl | 75.6 / 79.1 | 44.7 / 46.7 B | - | 397.0 / 415.2 | 125.1 / 134.7 | 583.3 / 597.1 |
| v3_int8_full_fbias | 55.2 / 58.7 B | 33.8 / 35.6 B | - | 289.3 / 307.1 | 93.1 / 101.7 | 784.8 / 803.0 |
| v3_int8_head_excl_fbias | 76.2 / 81.2 | 42.8 / 47.6 B | - | 410.6 / 434.3 | 125.4 / 134.6 | 580.5 / 593.5 |
| rec_fp32 | 0.03 / 0.03 | 0.03 / 0.03 | 3.25 / 3.96 | 0.25 / 0.32 | - | 3.48 / 4.09 |
| rec_fp16 | 0.03 / 0.03 | 0.03 / 0.03 | 3.22 / 3.93 | 0.25 / 0.34 | - | 3.42 / 4.05 |
| rec_int8_full | 0.06 / 0.06 | 0.04 / 0.05 | - | 0.19 / 0.28 | - | 31.0 / 35.7 |
| rec_int8_full_fbias | 0.05 / 0.05 | 0.04 / 0.05 | 29.9 / 35.0 | 0.24 / 0.33 | - | 30.8 / 35.5 |
| rec_int8_head_excl | 0.05 / 0.05 | 0.04 / 0.03 | - | 0.19 / 0.30 | - | 26.1 / 30.5 |
| rec_int8_head_excl_fbias | 0.06 / 0.07 | 0.04 / 0.04 | 25.3 / 30.1 | 0.29 / 0.38 | - | 25.6 / 30.1 |
| coco_fp32 | 370.4 / 382.9 | 180.9 / 191.9 | 25.2 / 26.9 A | - | 762.7 / 777.3 | 25.2 / 26.9 A |
| coco_fp16 | 365.1 / 375.0 | 191.7 / 194.9 | 55.7 / 58.9 B | - | 676.5 / 731.9 | 24.0 / 25.3 A |
| coco_int8_full | 140.5 / 145.4 | 85.7 / 87.6 | - | - | 319.1 / 338.0 | 2238.2 / 2274.0 |
| coco_int8_full_fbias | - | 74.4 / 86.4 | 2056.5 / 2148.8 | - | - | - |
| coco_int8_head_excl | 163.0 / 171.5 | 90.5 / 97.9 | - | - | 330.3 / 365.9 | 1897.5 / 1986.2 |
| coco_int8_head_excl_fbias | - | 81.8 / 92.6 | 1812.4 / 1839.6 | - | - | - |

## WebGPU node placement (WebGPU nodes / CPU nodes)

| model | ORT 1220 | CPU op types (1220) | ORT 1300 | CPU op types (1300) |
|---|---:|---|---:|---|
| v4_fp32 | 399/11 | TopKx2, Unsqueezex2, Tilex2, Flattenx1, Divx1, Modx1, Gatherx1, Castx1 | 226/9 | Unsqueezex2, Tilex2, Flattenx1, Divx1, Modx1, Gatherx1, Castx1 |
| v4_fp16 | 427/29 | Splitx12, Castx5, TopKx2, Unsqueezex2, Tilex2, GatherElementsx2, Flattenx1, Divx1, Modx1, Gatherx1 | 228/9 | Unsqueezex2, Tilex2, Flattenx1, Divx1, Modx1, Gatherx1, Castx1 |
| v4_int8_full | run failed (unsupported) | 8944496 | 1951/417 | QuantizeLinearx404, Transposex4, Unsqueezex2, Tilex2, Flattenx1, Divx1, Modx1, Gatherx1, Castx1 |
| v4_int8_head_excl | run failed (unsupported) | 14662232 | 1571/325 | QuantizeLinearx312, Transposex4, Unsqueezex2, Tilex2, Flattenx1, Divx1, Modx1, Gatherx1, Castx1 |
| v4_int8_full_fbias | 1948/419 | QuantizeLinearx404, Transposex4, TopKx2, Unsqueezex2, Tilex2, Flattenx1, Divx1, Modx1, Gatherx1, Castx1 | 1951/417 | QuantizeLinearx404, Transposex4, Unsqueezex2, Tilex2, Flattenx1, Divx1, Modx1, Gatherx1, Castx1 |
| v4_int8_head_excl_fbias | 1606/327 | QuantizeLinearx312, Transposex4, TopKx2, Unsqueezex2, Tilex2, Flattenx1, Divx1, Modx1, Gatherx1, Castx1 | 1571/325 | QuantizeLinearx312, Transposex4, Unsqueezex2, Tilex2, Flattenx1, Divx1, Modx1, Gatherx1, Castx1 |
| v3_fp32 | 245/0 | - | 133/0 | - |
| v3_fp16 | 267/8 | Splitx8 | 139/0 | - |
| v3_int8_full | run failed (unsupported) | 24794648 | 1210/254 | QuantizeLinearx250, Transposex4 |
| v3_int8_head_excl | run failed (unsupported) | 8944728 | 913/181 | QuantizeLinearx177, Transposex4 |
| v3_int8_full_fbias | 1210/254 | QuantizeLinearx250, Transposex4 | 1210/254 | QuantizeLinearx250, Transposex4 |
| v3_int8_head_excl_fbias | 937/181 | QuantizeLinearx177, Transposex4 | 913/181 | QuantizeLinearx177, Transposex4 |
| rec_fp32 | 12/0 | - | 12/0 | - |
| rec_fp16 | 14/0 | - | 14/0 | - |
| rec_int8_full | run failed (unsupported) | 10017880 | 57/12 | QuantizeLinearx12 |
| rec_int8_full_fbias | 57/12 | QuantizeLinearx12 | 57/12 | QuantizeLinearx12 |
| rec_int8_head_excl | run failed (unsupported) | 8944672 | 49/10 | QuantizeLinearx10 |
| rec_int8_head_excl_fbias | 49/10 | QuantizeLinearx10 | 49/10 | QuantizeLinearx10 |
| coco_fp32 | 637/0 | - | 319/0 | - |
| coco_fp16 | 667/11 | Splitx11 | 325/0 | - |
| coco_int8_full | - | - | - | - |
| coco_int8_full_fbias | 3172/651 | QuantizeLinearx647, Transposex4 | 3172/651 | QuantizeLinearx647, Transposex4 |
| coco_int8_head_excl | - | - | - | - |
| coco_int8_head_excl_fbias | 2827/560 | QuantizeLinearx556, Transposex4 | 2791/560 | QuantizeLinearx556, Transposex4 |

## Browser numeric parity (vs ORT CPU, same frames and decoder)

| model | runtime | mAP50 | delta | mAP50-95 | delta |
|---|---|---:|---:|---:|---:|
| v4_fp32 | webgpu_1300 | 0.499865 | -0.000000 | 0.242181 | +0.000000 |
| v4_fp32 | wasm_1300 | 0.499865 | -0.000000 | 0.242181 | +0.000000 |
| v4_fp16 | webgpu_1300 | 0.498247 | -0.000358 | 0.241837 | +0.000046 |
| v4_int8_head_excl_fbias | wasm_1300 | 0.482775 | +0.000339 | 0.235393 | +0.000227 |
| v3_fp32 | webgpu_1300 | 0.565269 | -0.000000 | 0.280751 | -0.000000 |
| v3_fp16 | webgpu_1300 | 0.565025 | -0.000599 | 0.279343 | -0.001120 |

## Browser pipeline: five launches per key configuration

| configuration | launches | det | rec | mean: median [min-max] | p90: median [min-max] | A 30 FPS | B 15 FPS | pooled p90 (check) |
|---|---:|---:|---:|---:|---:|:-:|:-:|---:|
| v4_fp32@webgpu+rec_fp32@wasm | 5 | 15.57 | 0.24 | 17.25 [16.21-18.20] | 19.91 [18.00-21.02] | Y (5/5) | Y (5/5) | 20.15 (same verdict) |
| v4_fp16@webgpu+rec_fp32@wasm | 5 | 13.74 | 0.24 | 16.72 [16.20-17.39] | 19.37 [18.45-19.92] | Y (5/5) | Y (5/5) | 19.41 (same verdict) |
| v4_fp16@webgpu+rec_int8_full@wasm | 5 | 13.88 | 0.29 | 16.96 [16.04-17.39] | 19.71 [18.34-20.23] | Y (5/5) | Y (5/5) | 19.75 (same verdict) |
| v4_fp32@webgpu+rec_fp32@webgpu | 5 | 15.40 | 2.34 | 19.32 [17.75-20.24] | 22.95 [20.38-23.18] | Y (5/5) | Y (5/5) | 22.68 (same verdict) |
| v4_fp16@webgpu+rec_fp16@webgpu | 5 | 13.93 | 2.38 | 18.90 [17.47-19.69] | 22.35 [20.34-23.78] | Y (5/5) | Y (5/5) | 22.34 (same verdict) |
| v4_int8_head_excl@wasm+rec_int8_full@wasm | 5 | 46.65 | 0.27 | 48.61 [37.19-50.22] | 52.91 [41.71-53.64] | N (0/5) | Y (5/5) | 52.74 (same verdict) |
| v4_fp32@wasm+rec_fp32@wasm | 5 | 62.95 | 0.23 | 64.85 [51.37-67.15] | 67.36 [57.45-85.49] | N (0/5) | N (1/5) | 68.17 (same verdict) |

| paired comparison (a - b, per round) | rounds | mean diff ms | [min, max] | rounds a faster |
|---|---:|---:|---:|---:|
| recognizer on WASM instead of WebGPU (FP32 detector on WebGPU) | 5 | -1.84 | [-2.74, -1.31] | 5/5 |
| FP16 instead of FP32 detector on WebGPU (recognizer FP32 on WASM) | 5 | -0.50 | [-0.81, -0.01] | 5/5 |
| INT8 instead of FP32 recognizer on WASM (FP16 detector on WebGPU) | 5 | +0.11 | [-0.16, +0.55] | 2/5 |
| all FP16 instead of all FP32 on WebGPU | 5 | -0.35 | [-0.67, +0.38] | 4/5 |
| INT8 head-excluded instead of FP32 detector on WASM | 5 | -16.06 | [-18.54, -14.19] | 5/5 |
