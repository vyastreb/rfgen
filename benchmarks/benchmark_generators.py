"""Reproducible CPU/RAM workloads for the spectral generators.

Run each workload in a fresh process under ``/usr/bin/time`` so that peak RSS
includes all NumPy allocations::

    PYTHONPATH=src /usr/bin/time -f 'user=%U system=%S rss_kib=%M elapsed=%e' \
        python benchmarks/benchmark_generators.py selfaffine
"""

from __future__ import annotations

import argparse

import numpy as np

from rfgen import arbitrary_pdf_psd_field, selfaffine_field


def selfaffine_workload() -> None:
    """Generate a representative 3D self-affine field."""
    selfaffine_field(dim=3, N=128, Hurst=0.8, rng=np.random.default_rng(42))


def iaaft_workload() -> None:
    """Generate a representative 3D field with PSD/PDF control."""
    arbitrary_pdf_psd_field(
        dim=3,
        N=128,
        psd_func=lambda k: 1.0 / (1.0 + k**2),
        icdf_func=lambda u: u,
        n_iters=10,
        rng=np.random.default_rng(42),
    )


def main() -> None:
    """Run the selected benchmark workload."""
    parser = argparse.ArgumentParser()
    parser.add_argument("workload", choices=("selfaffine", "iaaft"))
    args = parser.parse_args()

    if args.workload == "selfaffine":
        selfaffine_workload()
    else:
        iaaft_workload()


if __name__ == "__main__":
    main()
