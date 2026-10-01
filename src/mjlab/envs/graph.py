"""Capture a whole fixed-shape control step in one torch-owned CUDA graph.

mjlab already captures MuJoCo Warp's ``step``/``forward``/``reset`` in Warp-owned
graphs, but the rest of a control step (action processing, managers, rewards,
observations, resets) is hundreds of small torch kernels launched from Python. This
module records that whole program, physics included, in a single ``torch.cuda``
graph so a control step is one launch.

How the pieces fit:

* The graph is owned by torch, so every torch allocation made during capture lives
  in the graph's private pool and keeps its address across replays. A Warp-owned
  capture would let torch's allocator reuse that memory between replays.
* Warp is told a capture is running with ``wp.capture_begin(external=True)`` on the
  same stream. MuJoCo Warp's ``capture_while``/``capture_if`` then emit device-side
  conditional nodes instead of host loops, and its kernels are recorded in this
  graph. ``Simulation`` launches them directly while ``in_outer_capture``.
* Randomness is device-side Philox. The default CUDA generator and any generator
  passed in ``generators`` (the environment's ``reset_generator``) are registered, so
  each replay advances their offsets exactly as the eager run would.

The program must be fixed-shape and synchronization-free: no host reads, no
data-dependent shapes, no host state that changes between steps, and no tensor that
is rebound between steps and read before it is written (a replay would keep reading
the capture-time object). Eagerly running the same program first (warm-up) is the
caller's job; see ``CudaGraphProgram``.
"""

from __future__ import annotations

import gc
from collections.abc import Callable, Iterable
from typing import Generic, TypeVar

import torch
import warp as wp

from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

T = TypeVar("T")


class CudaGraphProgram(Generic[T]):
  """Capture ``program()`` once and replay it.

  ``program`` reads its inputs from tensors that the caller overwrites in place
  before each ``replay()`` and returns the tensors it produced. Those outputs
  are static: the next replay overwrites them, so clone anything kept across steps.
  """

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    program: Callable[[], T],
    *,
    generators: Iterable[torch.Generator] = (),
  ) -> None:
    if not env.sim.wp_device.is_cuda:
      raise RuntimeError("CUDA-graph capture requires a CUDA device")
    self.env = env
    self._program = program
    self._generators = tuple(generators)
    self._graph: torch.cuda.CUDAGraph | None = None
    self._epoch = -1
    self.outputs: T | None = None

  @property
  def captured(self) -> bool:
    return self._graph is not None

  def capture(self) -> T:
    """Record one run of the program; the first replay performs the first step."""
    if self._graph is not None:
      raise RuntimeError("program is already captured")
    sim = self.env.sim
    graph = torch.cuda.CUDAGraph()
    for generator in self._generators:
      graph.register_generator_state(generator)
    stream = torch.cuda.Stream(device=self.env.device)
    wp_stream = wp.stream_from_torch(stream)
    torch.cuda.synchronize(self.env.device)
    gc.collect()
    # A collected Warp graph would record its destructor into this capture.
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
      with wp.ScopedDevice(sim.wp_device), wp.ScopedStream(wp_stream, sync_enter=False):
        with torch.cuda.graph(graph, stream=stream, capture_error_mode="thread_local"):
          wp.capture_begin(stream=wp_stream, external=True)
          try:
            outputs = self._program()
          finally:
            wp.capture_end(stream=wp_stream)
    finally:
      if gc_was_enabled:
        gc.enable()
    torch.cuda.synchronize(self.env.device)
    self._graph = graph
    self._epoch = sim.graph_epoch
    self.outputs = outputs
    return outputs

  def replay(self) -> T:
    """Launch the captured step and return its static outputs."""
    if self._graph is None or self.outputs is None:
      raise RuntimeError("program has not been captured")
    if self.env.sim.graph_epoch != self._epoch:
      raise RuntimeError(
        "simulation arrays were replaced after capture (graph_epoch changed); "
        "the captured graph would read the old arrays"
      )
    self._graph.replay()
    return self.outputs
