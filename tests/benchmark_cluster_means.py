
#!/usr/bin/env python3
# This script benchmarks two methods for computing group means of a large matrix:
# 1) np.add.at accumulation (_cluster_means_nd)
# 2) Sparse-accumulation with CSR and a dense matmul (cool_mean)
#
# It generates a dummy dataset using NumPy memmaps so you can benchmark arrays
# as large as 2_000_000 x 1000 without needing all that RAM at once.
#
# Usage examples:
#   python benchmark_cluster_means.py --n 2_000_000 --f 1000 --k 1000 --dtype float32 --repeats 1
#   python benchmark_cluster_means.py --n 200_000 --f 100 --k 100 --dtype float32 --verify
#
# Notes:
# - Disk space: with float32 data, the memmap will use ~4 * n * f bytes (e.g., ~8 GB for 2e6 x 1000).
# - Time and peak RSS are measured inside isolated child processes to capture peak memory accurately.
# - Labels are generated uniformly in [0, k).
# - Both methods assume labels are contiguous 0..k-1; that's satisfied by our generator.
#
# Save this file locally as "benchmark_cluster_means.py" and run from a terminal.

import argparse
import gc
import os
import platform
import resource
import sys
import time
import multiprocessing as mp
from typing import Tuple

import numpy as np
import scipy as sc
import scipy.sparse as sp


def _cluster_means_nd(X: np.ndarray, labels: np.ndarray):
    labels = np.asarray(labels)
    uniq, inv = np.unique(labels, return_inverse=True)
    counts = np.bincount(inv, minlength=len(uniq))
    means = np.zeros((len(uniq), X.shape[1]), dtype=X.dtype)
    np.add.at(means, inv, X)
    means = means / counts[:, None]
    return uniq, counts, means



def cool_mean(data: np.ndarray, partition: np.ndarray, return_counts=False):
    """Efficiently calculate the mean of all rows of a matrix M over a partition u. The number of classes in the
    partition is implicitly defined from the values of u.

    
    Parameters
    ----------
        data: Matrix of dimensions (n, f) with n data points of f features each.

        partition: Partition of the data points in the form of an (n, ) array with k different integer values.

    Returns
    -------
        group_mean: A (k, f) matrix with the vectors averaged over the k partition values.
    
    General version: works for arbitrary integer labels (gaps, unsorted, etc.).
    Returns (unique_labels_sorted, counts_per_label, means_in_that_order).
    """
    n = data.shape[0]
    un, inv, nf = np.unique(partition, return_inverse=True, return_counts=True)
    umat = sp.csr_matrix(
        (np.ones(n, dtype=data.dtype), (np.arange(n, dtype=np.int64), inv)),
        shape=(n, len(un)),
    )
    means = (umat.T @ data) / nf[:, None]
    if return_counts:
        return un, nf, means
    else:
        return means

def _peak_kb_from_ru() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux gives KB, macOS gives bytes
    if platform.system() == "Darwin":
        return ru / 1024.0
    return float(ru)


def _run_child(method: str, data_path: str, labels_path: str, n: int, f: int, dtype_str: str, q: mp.Queue):
    """Child process: reopen memmaps, run the method once, return timing + peak RSS."""
    try:
        dtype = np.dtype(dtype_str)
        X = np.memmap(data_path, dtype=dtype, mode="r", shape=(n, f))
        y = np.memmap(labels_path, dtype=np.int32, mode="r", shape=(n,))

        start = time.perf_counter()
        if method == "method1":
            _, _, means = _cluster_means_nd(X, y)
        elif method == "method2":
            means = cool_mean(X, y)
        else:
            raise ValueError(f"Unknown method: {method}")
        elapsed = time.perf_counter() - start
        checksum = float(means.mean())  # tiny check to avoid optimizing away work
        peak_kb = _peak_kb_from_ru()
        q.put({"ok": True, "elapsed_s": elapsed, "peak_rss_kb": peak_kb, "checksum": checksum})
    except Exception as e:
        q.put({"ok": False, "error": repr(e)})
    finally:
        # Best-effort cleanup before child exits
        try:
            del X, y, means
        except Exception:
            pass
        gc.collect()


def run_and_measure(method: str, data_path: str, labels_path: str, n: int, f: int, dtype: np.dtype, repeats: int = 1):
    ctx = mp.get_context("spawn")
    results = []
    for r in range(repeats):
        q = ctx.Queue()
        p = ctx.Process(
            target=_run_child,
            args=(method, data_path, labels_path, n, f, str(dtype), q),
        )
        p.start()
        res = q.get()  # wait for result
        p.join()
        if not res.get("ok", False):
            raise RuntimeError(f"{method} run failed: {res.get('error')}")
        results.append(res)
    # Aggregate
    times = [x["elapsed_s"] for x in results]
    peaks = [x["peak_rss_kb"] for x in results]
    checks = [x["checksum"] for x in results]
    return {
        "method": method,
        "runs": repeats,
        "time_s_mean": float(np.mean(times)),
        "time_s_std": float(np.std(times)),
        "peak_rss_mb_mean": float(np.mean(peaks) / 1024.0),
        "peak_rss_mb_std": float(np.std(peaks) / 1024.0),
        "checksum_mean": float(np.mean(checks)),
    }


def create_memmap_dataset(
    directory: str,
    n: int,
    f: int,
    k: int,
    dtype: np.dtype = np.float32,
    rows_per_chunk: int = 20_000,
    seed: int = 0,
    force_data: bool = True,
) -> Tuple[str, str]:
    """Create memmap files for X (n,f) and labels y (n,).
    Returns (data_path, labels_path)."""
    os.makedirs(directory, exist_ok=True)
    data_path = os.path.join(directory, f"X_{n}x{f}_{dtype.name}.mmap")
    labels_path = os.path.join(directory, f"y_{n}_int32.mmap")

    # If files already exist with correct size, optionally reuse
    need_make_data = True
    need_make_labels = True

    if not force_data and os.path.exists(data_path):
        try:
            X = np.memmap(data_path, dtype=dtype, mode="r", shape=(n, f))
            _ = X[0, 0]
            need_make_data = False
        except Exception:
            need_make_data = True

    if not force_data and os.path.exists(labels_path):
        try:
            y = np.memmap(labels_path, dtype=np.int32, mode="r", shape=(n,))
            # Validate contiguity and full coverage of labels 0..k-1
            un = np.unique(y)
            if un.size == k and un[0] == 0 and un[-1] == k - 1:
                need_make_labels = False
                print(f"[dataset] Reusing labels (validated contiguous 0..{k-1})")
            else:
                print(f"[dataset] Existing labels invalid (min={un[0] if un.size>0 else 'NA'}, max={un[-1] if un.size>0 else 'NA'}, unique={un.size}). Will rebuild.")
                need_make_labels = True
        except Exception:
            need_make_labels = True

    if not need_make_data and not need_make_labels:
        print(f"[dataset] Reusing existing memmaps in {directory}")
        return data_path, labels_path

    print(f"[dataset] Creating/updating memmaps in {directory}")
    rng = np.random.default_rng(seed)

    # Create data memmap and fill in chunks (only if needed)
    if need_make_data:
        X = np.memmap(data_path, dtype=dtype, mode="w+", shape=(n, f))
        t0 = time.perf_counter()
        for start in range(0, n, rows_per_chunk):
            end = min(start + rows_per_chunk, n)
            rows = end - start
            chunk = rng.standard_normal((rows, f)).astype(dtype, copy=False)
            X[start:end] = chunk
            if (start // rows_per_chunk) % 10 == 0:
                print(f"[dataset] Filled rows {start:,}..{end:,} / {n:,}")
        X.flush()
        t1 = time.perf_counter()
        print(f"[dataset] Data memmap written in {t1 - t0:.2f}s -> {data_path}")
    else:
        print(f"[dataset] Using existing data memmap -> {data_path}")

    # Create/overwrite labels memmap with guaranteed contiguous coverage 0..k-1 (if needed)
    if n < k:
        raise ValueError(f"n={n} must be >= k={k} to guarantee all labels appear at least once.")
    if need_make_labels:
        y = np.memmap(labels_path, dtype=np.int32, mode="w+", shape=(n,))
        base = np.arange(k, dtype=np.int32)
        remaining = n - k
        if remaining > 0:
            extra = rng.integers(0, k, size=remaining, dtype=np.int32)
            labels_full = np.concatenate([base, extra])
        else:
            labels_full = base
        rng.shuffle(labels_full)
        y[:] = labels_full
        y.flush()
        print(f"[dataset] Labels memmap written with all {k} labels present -> {labels_path}")
    else:
        print(f"[dataset] Keeping existing labels -> {labels_path}")
    return data_path, labels_path


def verify_small(n: int = 20_000, f: int = 128, k: int = 50, dtype: np.dtype = np.float32):
    print(f"[verify] Running a small correctness check with n={n}, f={f}, k={k}, dtype={dtype}")
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, f), dtype=dtype)
    y = rng.integers(0, k, size=n, dtype=np.int32)
    _, _, m1 = _cluster_means_nd(X, y)
    m2 = cool_mean(X, y)
    max_abs = np.max(np.abs(m1 - m2))
    print(f"[verify] max|m1 - m2| = {max_abs:.6g}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark group-mean methods (np.add.at vs CSR @ dense).")
    parser.add_argument("--n", type=int, default=2_000_000, help="Number of rows (data points).")
    parser.add_argument("--f", type=int, default=1_000, help="Number of features (columns).")
    parser.add_argument("--k", type=int, default=1_000, help="Number of label classes (0..k-1).")
    parser.add_argument("--dtype", type=str, choices=["float32", "float64"], default="float32", help="Data dtype.")
    parser.add_argument("--repeats", type=int, default=1, help="Number of times to run each method.")
    parser.add_argument("--memdir", type=str, default="memmaps", help="Directory to store memmap files.")
    parser.add_argument("--rows-per-chunk", type=int, default=20_000, help="Rows per chunk when generating memmaps.")
    parser.add_argument("--verify", action="store_true", help="Run a small correctness check first.")
    parser.add_argument("--force-data", action="store_true", help="Recreate memmaps even if they exist.")
    args = parser.parse_args()

    print(f"Python {sys.version.split()[0]} | NumPy {np.__version__} | SciPy {sc.__version__}")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Config: n={args.n:,}, f={args.f:,}, k={args.k:,}, dtype={args.dtype}, repeats={args.repeats}")

    if args.verify:
        verify_small(dtype=np.dtype(args.dtype))

    # Create or reuse dataset memmaps
    data_path, labels_path = create_memmap_dataset(
        args.memdir, args.n, args.f, args.k, dtype=np.dtype(args.dtype), rows_per_chunk=args.rows_per_chunk, force_data=args.force_data
    )

    # Run benchmarks
    res1 = run_and_measure("method1", data_path, labels_path, args.n, args.f, np.dtype(args.dtype), repeats=args.repeats)
    print("\n[method1] np.add.at (_cluster_means_nd)")
    print(f"  time: {res1['time_s_mean']:.3f} s ± {res1['time_s_std']:.3f}")
    print(f"  peak RSS: {res1['peak_rss_mb_mean']:.1f} MB ± {res1['peak_rss_mb_std']:.1f}")
    print(f"  checksum: {res1['checksum_mean']:.6g}")

    res2 = run_and_measure("method2", data_path, labels_path, args.n, args.f, np.dtype(args.dtype), repeats=args.repeats)
    print("\n[method2] CSR @ dense (cool_mean)")
    print(f"  time: {res2['time_s_mean']:.3f} s ± {res2['time_s_std']:.3f}")
    print(f"  peak RSS: {res2['peak_rss_mb_mean']:.1f} MB ± {res2['peak_rss_mb_std']:.1f}")
    print(f"  checksum: {res2['checksum_mean']:.6g}")

    # Simple summary
    speedup = res1["time_s_mean"] / res2["time_s_mean"] if res2["time_s_mean"] > 0 else float("nan")
    print("\n[summary] speedup (method1 / method2): "
          f"{speedup:.2f}x  (values >1 mean method2 is faster)")


if __name__ == "__main__":
    main()
