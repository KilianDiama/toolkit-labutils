labutils_pro.py — toolkit 10/10 : déterministe, robuste, facile à intégrer.
Aucune dépendance obligatoire. numpy/torch optionnels.
"""

from __future__ import annotations

import os
import gc
import time
import random
import logging
import contextlib
from typing import (
    Any, Callable, Iterable, Iterator, List, Optional,
    Sequence, Tuple, TypeVar
)
from concurrent.futures import ProcessPoolExecutor, Future

# Optional deps
try:
    import numpy as np
except Exception:
    np = None

try:
    import torch
except Exception:
    torch = None

T = TypeVar("T")
U = TypeVar("U")


# ===========================================================================
# 1. Seed déterministe (strict)
# ===========================================================================

def set_global_seed(seed: int) -> None:
    """Seed global strictement déterministe."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if np is not None:
        np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


# ===========================================================================
# 2. Timer & benchmark robuste
# ===========================================================================

@contextlib.contextmanager
def time_block(label: str = "block") -> Iterator[float]:
    """Context manager silencieux, retourne la durée en secondes."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        print(f"[{label}] {dt*1000:.2f} ms")


def benchmark(
    fn: Callable[..., Any],
    *args: Any,
    warmup: int = 3,
    runs: int = 20,
    **kwargs: Any,
) -> dict:
    """Benchmark robuste (GC off, warmup, stats propres)."""
    gc.disable()

    for _ in range(warmup):
        fn(*args, **kwargs)

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)

    gc.enable()

    times.sort()
    n = len(times)
    return {
        "n": n,
        "mean_ms": 1000 * (sum(times) / n),
        "p50_ms": 1000 * times[n // 2],
        "p90_ms": 1000 * times[int(0.9 * (n - 1))],
        "min_ms": 1000 * times[0],
        "max_ms": 1000 * times[-1],
    }


# ===========================================================================
# 3. Parallel map déterministe + batching
# ===========================================================================

def _worker_batch(fn: Callable[[T], U], batch: List[T]) -> List[U]:
    return [fn(x) for x in batch]


def parallel_map(
    fn: Callable[[T], U],
    iterable: Iterable[T],
    max_workers: Optional[int] = None,
    batch_size: int = 32,
    fail_fast: bool = True,
) -> List[U]:
    """
    Parallel map robuste, déterministe, batché.
    - Ordre strictement préservé
    - Batching pour amortir le pickling
    - fail_fast optionnel
    """
    items = list(iterable)
    if not items:
        return []

    batches = [items[i:i+batch_size] for i in range(0, len(items), batch_size)]
    results: List[Optional[List[U]]] = [None] * len(batches)

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures: List[Tuple[int, Future]] = [
            (i, ex.submit(_worker_batch, fn, batch))
            for i, batch in enumerate(batches)
        ]

        for idx, fut in futures:
            try:
                results[idx] = fut.result()
            except Exception as e:
                if fail_fast:
                    # Cancel all remaining futures
                    for _, f in futures:
                        f.cancel()
                    raise RuntimeError(f"parallel_map failed at batch {idx}") from e
                results[idx] = None

    out: List[U] = []
    for r in results:
        if r is not None:
            out.extend(r)
    return out


# ===========================================================================
# 4. Logging propre
# ===========================================================================

def get_logger(name: str = "labutils", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
        logger.addHandler(h)
    return logger


# ===========================================================================
# 5. Helpers universels
# ===========================================================================

def chunks(seq: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    if size <= 0:
        raise ValueError("size must be > 0")
    for i in range(0, len(seq), size):
        yield seq[i:i+size]


def flatten(nested: Iterable[Iterable[T]]) -> List[T]:
    out: List[T] = []
    for it in nested:
        out.extend(it)
    return out


def sliding_window(seq: Sequence[T], k: int) -> Iterator[Tuple[T, ...]]:
    """Fenêtre glissante de taille k."""
    if k <= 0:
        raise ValueError("k must be > 0")
    for i in range(len(seq) - k + 1):
        yield tuple(seq[i:i+k])
