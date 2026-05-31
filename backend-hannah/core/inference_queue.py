"""
core/inference_queue.py
-----------------------
Cola de inferencia para serializar requests a los modelos GPU.
Evita OOM cuando múltiples usuarios piden respuesta simultáneamente.
Usa asyncio.Queue + asyncio.Semaphore para control de concurrencia.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable


@dataclass
class _InferenceJob:
    payload: dict
    future: asyncio.Future = field(default_factory=lambda: asyncio.get_event_loop().create_future())


class InferenceQueue:
    """
    Cola FIFO para serializar llamadas a un modelo.

    Uso:
        queue = InferenceQueue(worker_fn=_call_fast_model, max_concurrent=1)
        result = await queue.submit(payload)
    """

    def __init__(
        self,
        worker_fn: Callable[[dict], Awaitable[str]],
        max_concurrent: int = 1,
        max_queue_size: int = 50,
    ) -> None:
        self._worker_fn = worker_fn
        self._queue: asyncio.Queue[_InferenceJob] = asyncio.Queue(maxsize=max_queue_size)
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._started = False

    def start(self) -> None:
        """Lanzar el worker loop. Llamar una vez en el startup de FastAPI."""
        if not self._started:
            asyncio.create_task(self._worker_loop())
            self._started = True

    async def submit(self, payload: dict) -> str:
        """Encolar un job y esperar su resultado."""
        loop = asyncio.get_event_loop()
        job = _InferenceJob(payload=payload, future=loop.create_future())
        try:
            self._queue.put_nowait(job)
        except asyncio.QueueFull:
            raise RuntimeError("Cola de inferencia llena. Demasiadas solicitudes simultáneas.")
        return await job.future

    async def _worker_loop(self) -> None:
        while True:
            job = await self._queue.get()
            async with self._semaphore:
                try:
                    result = await self._worker_fn(job.payload)
                    if not job.future.done():
                        job.future.set_result(result)
                except Exception as e:
                    if not job.future.done():
                        job.future.set_exception(e)
                finally:
                    self._queue.task_done()
