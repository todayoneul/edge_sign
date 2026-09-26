# Retention by object size (saved predictions, COCOeval area rules)

Commit `09db022f276e1b8f0e698475478ed3a9a9f6b2be`. Hypothesis and decision rule: see the docstring of `scripts/paper/size_bin_retention.py`.

## road_v3 (2417 images; GT per bin: all 6772, small 1870, medium 2755, large 2183)

| condition | AP all | AP small | AP medium | AP large | retention small [95% CI] | retention medium [95% CI] | retention large [95% CI] | large - small [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fp32 | 0.2825 | 0.0630 | 0.2475 | 0.4817 | reference | reference | reference | - |
| int8_head_excl | 0.2790 | 0.0639 | 0.2451 | 0.4751 | 101.5 [98.1-104.7] | 99.0 [97.9-99.9] | 98.6 [98.0-99.1] | -2.8 [-6.2, +0.6] |
| int8_decode_excl | 0.2618 | 0.0580 | 0.2307 | 0.4511 | 92.1 [87.9-96.7] | 93.2 [91.6-94.9] | 93.6 [92.3-95.1] | +1.5 [-3.2, +5.8] |
| a8sim_decode_no_outconcat | 0.1739 | 0.0133 | 0.1242 | 0.3578 | 21.1 [18.5-25.4] | 50.2 [48.1-52.5] | 74.3 [72.9-75.8] | +53.1 [+48.9, +56.0] |
| w8sim_head | 0.2725 | 0.0604 | 0.2320 | 0.4708 | 96.0 [92.5-100.3] | 93.7 [92.4-94.9] | 97.8 [97.1-98.3] | +1.8 [-2.7, +5.3] |

## road_v4 (2417 images; GT per bin: all 6772, small 1870, medium 2755, large 2183)

| condition | AP all | AP small | AP medium | AP large | retention small [95% CI] | retention medium [95% CI] | retention large [95% CI] | large - small [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fp32 | 0.2441 | 0.0400 | 0.1892 | 0.4604 | reference | reference | reference | - |
| int8_head_excl | 0.2368 | 0.0387 | 0.1821 | 0.4501 | 96.8 [91.4-99.9] | 96.2 [94.9-97.8] | 97.8 [96.9-98.5] | +0.9 [-2.3, +6.4] |
| int8_decode_excl | 0.2305 | 0.0365 | 0.1763 | 0.4378 | 91.3 [86.2-97.2] | 93.2 [91.5-95.2] | 95.1 [94.2-96.1] | +3.8 [-2.2, +9.0] |
| a8sim_decode_no_collapse | 0.1404 | 0.0042 | 0.0740 | 0.3327 | 10.6 [8.6-14.8] | 39.1 [36.6-41.8] | 72.3 [70.9-73.7] | +61.7 [+56.9, +64.3] |

## coco (1525 images; GT per bin: all 11120, small 4571, medium 3822, large 2727)

| condition | AP all | AP small | AP medium | AP large | retention small [95% CI] | retention medium [95% CI] | retention large [95% CI] | large - small [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fp32 | 0.5401 | 0.3701 | 0.5975 | 0.7231 | reference | reference | reference | - |
| int8_head_excl | 0.5356 | 0.3623 | 0.5885 | 0.7044 | 97.9 [95.5-99.9] | 98.5 [97.5-99.9] | 97.4 [96.5-99.4] | -0.5 [-2.4, +3.0] |
| int8_decode_excl | 0.5347 | 0.3606 | 0.5844 | 0.7045 | 97.4 [94.9-99.6] | 97.8 [96.6-99.2] | 97.4 [96.5-99.5] | -0.0 [-2.2, +3.6] |
| a8sim_decode_no_outconcat | 0.4683 | 0.2589 | 0.5197 | 0.6811 | 69.9 [68.9-74.0] | 87.0 [86.1-88.3] | 94.2 [93.6-94.9] | +24.2 [+20.3, +25.4] |

## Verdict

- road_v3: a8sim_decode_no_outconcat large - small = +53.1 %p [+48.9, +56.0] -> supports
- road_v4: a8sim_decode_no_collapse large - small = +61.7 %p [+56.9, +64.3] -> supports
- coco: a8sim_decode_no_outconcat large - small = +24.2 %p [+20.3, +25.4] -> supports
- hypothesis supported (all three): **True**
