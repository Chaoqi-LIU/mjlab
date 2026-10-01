"""Fixed-shape world masks for resets that must not synchronize the host.

A scoped reset normally receives ``env_ids``: an index vector whose length is
data dependent, so building it synchronizes the host and its shape cannot be
captured in a CUDA graph. A masked reset receives a boolean ``[num_envs]`` tensor
instead and every write becomes ``where(mask, new, old)``, so the work and the
shapes are the same whether zero or all worlds reset.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class EnvMask:
  """Select worlds with a boolean ``[num_envs]`` tensor instead of indices.

  Pass it wherever an API accepts ``env_ids`` and declares mask support.
  Values written through a mask are always full-batch ``[num_envs, ...]``; rows
  outside the mask keep their old contents.
  """

  mask: torch.Tensor

  def __post_init__(self) -> None:
    if self.mask.ndim != 1 or self.mask.dtype != torch.bool:
      raise ValueError("EnvMask requires a boolean [num_envs] tensor")


def expand_rows(mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
  """View ``mask`` ``[N]`` so it broadcasts over the trailing axes of ``like``."""
  return mask.reshape(mask.shape[0], *([1] * (like.ndim - 1)))


def where_rows(
  mask: torch.Tensor, new: torch.Tensor, old: torch.Tensor
) -> torch.Tensor:
  """Row-wise select: ``new`` where ``mask`` else ``old``; ``new`` broadcasts."""
  return torch.where(expand_rows(mask, old), new, old)


def assign_rows_(dst: torch.Tensor, mask: torch.Tensor, new: torch.Tensor) -> None:
  """In-place ``dst[mask] = new[mask]`` with full-batch ``new`` and no host sync."""
  dst.copy_(where_rows(mask, new, dst))


def fill_rows_(
  dst: torch.Tensor, mask: torch.Tensor, value: float | bool | int
) -> None:
  """In-place ``dst[mask] = value`` without a host sync."""
  dst.masked_fill_(expand_rows(mask, dst), value)


def mean_log(
  values: torch.Tensor, mask: torch.Tensor, divisor: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
  """Device mean of ``values[mask]`` and whether any world is masked.

  The legacy episode log is ``torch.mean(values[env_ids])`` and is simply absent
  on steps without a reset. The value here is 0 when no world is masked, so it
  can be accumulated unconditionally and weighted by the returned validity flag.
  """
  values = values.to(torch.float32)
  count = mask.sum()
  total = torch.where(expand_rows(mask, values), values, torch.zeros_like(values)).sum()
  elements = max(values[0].numel(), 1)
  return total / (count.clamp_min(1) * elements) / divisor, count > 0
