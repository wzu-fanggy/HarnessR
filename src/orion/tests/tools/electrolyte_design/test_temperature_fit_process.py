from __future__ import annotations

import threading

from tools.electrolyte_design.temperature_fit_process import TemperatureFitProcessAdapter

from .worker_pipe_fake import FakeWorkerStdio


def test_concurrent_fits_do_not_race_on_the_worker_pipe() -> None:
    """One worker process, therefore one exchange at a time.

    The same defect that hung two collection episodes also lets a racing caller
    read another caller's answer, so the guarantee is asserted at the adapter.
    """

    adapter = TemperatureFitProcessAdapter()
    worker = FakeWorkerStdio()
    adapter._process = worker  # type: ignore[assignment]

    requests = [
        {"formulation_id": f"measured-formulation-{index:02d}", "measurements": []}
        for index in range(4)
    ]
    barrier = threading.Barrier(len(requests))
    results: dict[str, dict] = {}
    failures: list[BaseException] = []

    def fit(request: dict) -> None:
        try:
            barrier.wait(timeout=5.0)
            results[request["formulation_id"]] = adapter.fit_temperature_laws(request)
        except BaseException as error:  # noqa: BLE001 - reported through the assertion
            failures.append(error)

    threads = [threading.Thread(target=fit, args=(request,)) for request in requests]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)

    assert not failures, failures
    assert all(not thread.is_alive() for thread in threads), "a caller never returned"
    assert worker.max_concurrent_reads == 1
    for request in requests:
        assert results[request["formulation_id"]]["echo"]["request"] == request
