# COCO latency runs that overlapped background load

These four results are the first measurements. The CPU-load log (`../coco_latency_cpu_load.log`) shows a busy background process during them:
- `vgtray`, 17:25–17:32: `cpu_t4_coco_{fp32,fp16,int8_full}`
- iCloud Drive, 17:58–18:06: `speed_wasmt4_ort1300_coco_fp32`

They were re-measured after the load cleared (`../coco_latency_rerun_cpu_load.log`). The selection used only the load log, not the results. The re-measured files replace them in `../`, and these originals are kept for comparison (differences within 10%).
