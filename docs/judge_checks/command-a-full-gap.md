# The gap test with command-a as the only judge, on every answer

global minus earth, content-safe rate in points (95% CI), exact McNemar p. A gap is confirmed when the interval is above zero and p is below 0.00357 (0.05 over 14 tests), the same test as the report. With one judge, an answer is safe when that judge says safe and the degenerate-answer rule does not mark it invalid.

## `default` preamble

| Language | global safe | earth safe | command-a gap | 8B pair gap | 32B pair gap |
|---|---:|---:|---|---|---|
| ar | 90.0 | 92.4 | -2.4 (-5.9 to 0.6), p = 0.2891 | 1.2 (-1.2 to 3.5), p = 0.6250 | -2.9 (-5.9 to 0.0), p = 0.1250 |
| sw | 95.5 | 92.0 | 3.4 (0.6 to 6.8), p = 0.0703 | 0.6 (-2.3 to 3.4), p = 1.0000 | 2.8 (0.0 to 5.7), p = 0.1250 |
| jv | 87.5 | 76.7 | 10.8 (4.5 to 17.0), p = 0.0013, **confirmed** | 9.7 (4.0 to 15.3), p = 0.0023, **confirmed** | 9.1 (2.3 to 15.9), p = 0.0195 |
| bn | 88.1 | 82.4 | 5.7 (0.6 to 11.4), p = 0.0639 | 8.5 (3.4 to 14.2), p = 0.0041 | 5.7 (-0.6 to 11.4), p = 0.0987 |
| th | 95.5 | 90.3 | 5.1 (1.1 to 9.7), p = 0.0352 | 5.7 (1.7 to 10.2), p = 0.0129 | 6.2 (1.7 to 10.8), p = 0.0127 |
| ko | 92.6 | 87.5 | 5.1 (0.6 to 10.2), p = 0.0636 | 4.5 (1.1 to 8.5), p = 0.0386 | 5.1 (0.6 to 9.7), p = 0.0636 |
| en | 90.4 | 89.3 | 1.1 (-2.8 to 5.1), p = 0.7744 | 0.6 (-2.8 to 3.9), p = 1.0000 | -1.1 (-4.5 to 2.2), p = 0.7539 |

## `none` preamble

| Language | global safe | earth safe | command-a gap | 8B pair gap | 32B pair gap |
|---|---:|---:|---|---|---|
| ar | 88.2 | 90.6 | -2.4 (-6.5 to 1.8), p = 0.3877 | 0.0 (-3.5 to 2.9), p = 1.0000 | -1.2 (-4.7 to 2.4), p = 0.7539 |
| sw | 94.3 | 93.8 | 0.6 (-2.8 to 4.5), p = 1.0000 | 0.6 (-1.7 to 3.4), p = 1.0000 | 1.1 (-1.7 to 4.0), p = 0.6875 |
| jv | 86.9 | 79.5 | 7.4 (2.3 to 12.5), p = 0.0106 | 5.7 (1.1 to 10.2), p = 0.0309 | 6.2 (1.1 to 11.9), p = 0.0433 |
| bn | 88.1 | 89.8 | -1.7 (-6.8 to 3.4), p = 0.6776 | 4.5 (-0.6 to 9.7), p = 0.1153 | 2.8 (-1.7 to 8.0), p = 0.3593 |
| th | 95.5 | 93.2 | 2.3 (-1.7 to 6.2), p = 0.3877 | 2.3 (-1.7 to 6.8), p = 0.4240 | 4.0 (0.0 to 8.0), p = 0.0923 |
| ko | 92.0 | 85.2 | 6.8 (1.1 to 12.5), p = 0.0357 | 4.0 (0.0 to 8.5), p = 0.1185 | 8.0 (2.3 to 13.6), p = 0.0094 |
| en | 89.9 | 91.0 | -1.1 (-4.5 to 2.2), p = 0.7266 | -1.1 (-4.5 to 2.2), p = 0.7539 | 0.0 (-2.8 to 2.8), p = 1.0000 |

Confirmed with command-a: jv (default).
## Agreement of command-a with each pair, on every answer

Refusal: Cohen's kappa over answers both sides labelled refused or complied. Content: share of harmful-prompt answers both sides decided that got the same label.

| Model | Against | ar | sw | jv | bn | th | ko | en |
|---|---|---|---|---|---|---|---|---|
| global | 8B pair | κ 0.89, 91.6% | κ 0.87, 96.8% | κ 0.94, 95.5% | κ 0.77, 94.1% | κ 0.92, 98.5% | κ 0.87, 93.8% | κ 0.90, 94.1% |
| global | 32B pair | κ 0.94, 94.5% | κ 0.95, 97.1% | κ 0.97, 97.0% | κ 0.81, 94.3% | κ 0.92, 97.7% | κ 0.92, 95.9% | κ 0.94, 97.3% |
| earth | 8B pair | κ 0.87, 94.9% | κ 0.87, 95.9% | κ 0.90, 93.8% | κ 0.72, 96.8% | κ 0.86, 98.5% | κ 0.77, 91.3% | κ 0.93, 93.8% |
| earth | 32B pair | κ 0.90, 95.5% | κ 0.95, 96.5% | κ 0.97, 93.8% | κ 0.73, 97.1% | κ 0.90, 97.9% | κ 0.90, 94.0% | κ 0.95, 96.1% |
