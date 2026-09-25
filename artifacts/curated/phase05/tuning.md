### interactive SLO: TTFT < 1s, TPOT < 0.05s

| variant | sustainable req/s | peak goodput req/s | peak tok/s | TTFT p99 at sustainable | on frontier | valid (generator kept up) |
|---|---|---|---|---|---|---|
| `baseline` | 16.47 | 13.80 | 1,976 | 153 ms | **yes** | yes |
| `max_num_seqs=32` | 8.25 | 7.07 | 1,071 | 328 ms | no | yes |
| `max_num_seqs=64` | 12.47 | 10.50 | 1,592 | 126 ms | no | yes |
| `max_num_seqs=96` | 16.47 | 13.80 | 1,933 | 417 ms | **yes** | yes |
| `max_num_seqs=128` | 16.47 | 13.80 | 1,959 | 159 ms | **yes** | yes |
| `baseline-repeat` | 16.47 | 13.80 | 1,980 | 163 ms | **yes** | yes |

interactive SLO (TTFT<1s, TPOT<0.05s): highest sustainable rate 16.47 req/s from baseline, max_num_seqs=96, max_num_seqs=128, baseline-repeat (tied) (peak goodput 13.80 req/s); frontier: baseline, baseline-repeat, max_num_seqs=128, max_num_seqs=96
