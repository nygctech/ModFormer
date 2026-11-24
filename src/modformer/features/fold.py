"""RNA folding with ViennaRNA: base-pair probabilities and the minimum-free-energy structure.

Each tile is folded as its full 1022-nt window, ``N`` padding included: padding stays in
the folded sequence (an ``N`` cannot pair) rather than being stripped and re-added, so the
fold sees exactly what the model was trained on.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Optional

import numpy as np

TEMPERATURE = 37.0
DANGLES = 2
SALT = 0.15


def _configured_md():
    """A ViennaRNA model-details object, or an error if this build cannot set the salt.

    The salt setting changes the energy model. A build that ignored it would silently
    produce different folds, so it is refused.
    """
    import RNA

    from . import VIENNARNA_VERSION

    md = RNA.md()
    md.temperature = TEMPERATURE
    md.dangles = DANGLES
    try:
        md.salt = SALT
        ok = abs(float(md.salt) - SALT) < 1e-9
    except AttributeError:
        ok = False
    if not ok:
        raise RuntimeError(
            f"ViennaRNA {getattr(RNA, '__version__', '?')} cannot set the salt "
            f"concentration to {SALT} M. Run: pip install ViennaRNA=={VIENNARNA_VERSION}")
    return md


def fold_window(window_seq: str) -> tuple[np.ndarray, str]:
    """Fold one window.

    Returns ``(bpp, dot_bracket)``: the symmetric base-pair probability matrix as float32
    with a zero diagonal, after a round trip through float16 (the precision the model was
    trained on), and the dot-bracket string of the minimum-free-energy structure.
    """
    import RNA

    seq = window_seq.replace("T", "U")
    n = len(seq)
    fc = RNA.fold_compound(seq, _configured_md())
    structure, energy = fc.mfe()
    fc.exp_params_rescale(energy)
    fc.pf()
    table = np.asarray(fc.bpp(), dtype=np.float32)[1:n + 1, 1:n + 1]
    upper = np.triu(table, k=1)
    bpp = (upper + upper.T).astype(np.float16).astype(np.float32)
    return bpp, structure


def _worker_init() -> None:
    """Make a worker exit when its parent dies, so an interrupted run leaves nothing behind.

    Workers ignore Ctrl-C: only the parent reacts to it and then drops the pool. The parent
    blocks the signal while it starts them (see :func:`fold_windows`), so none can arrive
    before this line; it is unblocked here.

    The worker also gets standard output and error objects of its own. It is forked from a
    process whose other threads may be writing (the session draws its progress line from one),
    and a fork copies each lock as it is at that instant: a copy of the error stream's lock
    taken mid-write is never released in the worker. Multiprocessing flushes both streams when
    the worker exits, so the worker would hang there and the pool's shutdown with it.
    """
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None:
            continue
        try:
            fresh = open(stream.fileno(), "w", buffering=1, encoding=stream.encoding,
                         errors=stream.errors, closefd=False)
        except Exception:  # not backed by a descriptor
            fresh = open(os.devnull, "w")
        setattr(sys, name, fresh)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if hasattr(signal, "pthread_sigmask"):
        signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT})
    if sys.platform.startswith("linux"):
        try:
            import ctypes

            ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
        except OSError:
            pass


def _fold_worker(window_seq: str) -> tuple[np.ndarray, str]:
    bpp, structure = fold_window(window_seq)
    return bpp.astype(np.float16), structure


def _stop_workers(pool) -> None:
    """Shut ``pool`` down without waiting and end its worker processes now. Without this a
    worker finishes the fold it is in (seconds), and the interpreter waits for it at exit, so
    leaving after a stop is slow. The processes are taken before the shutdown, which forgets them."""
    workers = list((getattr(pool, "_processes", None) or {}).values())
    pool.shutdown(wait=False, cancel_futures=True)
    for process in workers:
        try:
            process.terminate()
        except Exception:
            pass


def fold_windows(windows: list[str], workers: Optional[int] = None,
                 progress: Optional[Callable[[str, int, int], None]] = None
                 ) -> list[tuple[np.ndarray, str]]:
    """Fold many windows in a process pool; results keep the order of ``windows``.

    ``workers`` defaults to the number of available CPUs. With one worker, or one window,
    folding runs in this process.
    """
    total = len(windows)
    _configured_md()
    if workers is None:
        workers = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") \
            else (os.cpu_count() or 1)
    workers = max(1, min(int(workers), total))

    results: list[tuple[np.ndarray, str]] = []
    if workers == 1:
        for done, w in enumerate(windows, 1):
            results.append(fold_window(w))
            if progress:
                progress("fold", done, total)
        return results

    method = "fork" if sys.platform.startswith("linux") else "spawn"
    pool = None
    try:
        # A Ctrl-C that reaches a worker before it ignores the signal would print a traceback
        # from inside the worker's start-up; the parent takes it once the workers exist.
        before = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT}) \
            if hasattr(signal, "pthread_sigmask") else None
        try:
            pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context(method),
                                       initializer=_worker_init)
            pieces = pool.map(_fold_worker, windows)  # the workers start here
        finally:
            if before is not None:
                signal.pthread_sigmask(signal.SIG_SETMASK, before)
        for done, (bpp, structure) in enumerate(pieces, 1):
            results.append((bpp.astype(np.float32), structure))
            if progress:
                progress("fold", done, total)
    except BaseException:
        # An interrupt must not wait for the folds still running: drop the queue and return.
        if pool is not None:
            _stop_workers(pool)
        raise
    pool.shutdown()
    return results
