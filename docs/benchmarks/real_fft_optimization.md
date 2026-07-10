# Real-FFT optimization benchmark

This benchmark compares commit `fe62e9e` with the real-FFT implementation.
Each workload ran in a fresh process on 2026-07-10 with Python 3.12.2 and
NumPy 1.26.4. Values are a single measurement on the development machine;
compare runs made on the same machine rather than treating them as universal
throughput figures.

```sh
PYTHONPATH=src /usr/bin/time -f 'user=%U system=%S rss_kib=%M elapsed=%e' \
  python benchmarks/benchmark_generators.py selfaffine
PYTHONPATH=src /usr/bin/time -f 'user=%U system=%S rss_kib=%M elapsed=%e' \
  python benchmarks/benchmark_generators.py iaaft
```

Both cases use a 3D `128**3` grid and seed 42. The self-affine workload uses
`Hurst=0.8`; IAAFT uses `1 / (1 + k**2)`, a uniform inverse CDF, and ten
iterations.

| Workload | Version | User CPU (s) | System CPU (s) | CPU total (s) | Elapsed (s) | Peak RSS (MiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Self-affine | `fe62e9e` | 2.23 | 0.96 | 3.19 | 0.51 | 244.97 |
| Self-affine | real FFT | 2.17 | 0.96 | 3.13 | 0.39 | 135.01 |
| IAAFT | `fe62e9e` | 7.02 | 1.29 | 8.31 | 5.61 | 590.21 |
| IAAFT | real FFT | 5.84 | 1.11 | 6.95 | 4.21 | 231.11 |

Peak memory fell by 45% for self-affine generation and 61% for IAAFT. The
total CPU time fell by 2% and 16%, respectively; elapsed time fell by 24% and
25%. Real FFTs remove the redundant Hermitian half-spectrum, while in-place
normalisation/filtering and the exact Parseval variance calculation remove
additional full-size temporaries in IAAFT.
