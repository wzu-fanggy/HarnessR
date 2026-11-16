"""Serialise request/response exchanges with one resident process-backed resource.

Why this exists
---------------

Every managed process resource owns exactly one stdin/stdout pipe, and Python's
text streams are not thread-safe.  LangGraph executes the *synchronous* tools of a
single ToolNode step concurrently in a thread pool, so when the agent batches two
calls that need the same resident worker -- for example two
``predict_formulation_conductivity`` calls at two temperatures -- two threads write
to and read from that one pipe at the same time.

The failure this produces is not a wrong answer but a hang.  Observed twice during
the four-resource development collection, both times as a ToolNode batch of two
``predict_formulation_conductivity`` calls that never completed and an episode that
burned its full 900 s wall-clock budget:

* ``NC2_A08#r1`` (family A) -- batch 4, two predictions at -20 C and 25 C.
* ``NC2_B01#r2`` (family B) -- batch 7, the same shape after two identical batches
  in the same episode had completed normally.

Holding one lock across the write *and* the read restores the worker's real
concurrency: one worker process serves one request at a time.  This changes
neither how long the resource stays resident nor how the agent batches its calls
nor which idle candidates are eligible for eviction, so it does not change the
contention profile a collection run is measuring.
"""

from __future__ import annotations

import threading


def exchange_lock() -> threading.RLock:
    """Return a reentrant lock that guards one worker's write/read pair.

    Reentrant because an exchange can be started from a path that already holds the
    lock, such as a startup handshake reached from ``promote``.
    """

    return threading.RLock()


def run_exchange(lock: threading.RLock, send, read):
    """Run one complete request/response exchange while holding ``lock``."""

    with lock:
        send()
        return read()
