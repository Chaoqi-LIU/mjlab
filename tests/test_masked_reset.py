"""Fixed-shape (masked) resets: same state as a scoped reset, no host index vector."""

import pytest
import torch
from conftest import get_test_device

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import EventTermCfg
from mjlab.tasks.cartpole.cartpole_env_cfg import cartpole_balance_env_cfg
from mjlab.utils.masking import (
  EnvMask,
  assign_rows_,
  fill_rows_,
  mean_log,
  where_rows,
)


@pytest.fixture(scope="module")
def device():
  return get_test_device()


def _env(device, *, events: bool = False) -> ManagerBasedRlEnv:
  cfg = cartpole_balance_env_cfg()
  cfg.episode_length_s = 0.5
  cfg.scene.num_envs = 6
  cfg.auto_reset = False
  if not events:
    cfg.events = {}
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  env.reset()
  env.prepare_static_program()
  return env


def _advance(env: ManagerBasedRlEnv, steps: int = 3) -> None:
  action = torch.full((env.num_envs, 1), 0.3, device=env.device)
  for _ in range(steps):
    env.step(action)
    env._manual_reset_pending.zero_()


def _state(env: ManagerBasedRlEnv) -> dict[str, torch.Tensor]:
  data = env.sim.data
  return {
    "qpos": data.qpos.clone(),
    "qvel": data.qvel.clone(),
    "episode_length": env.episode_length_buf.clone(),
    "action": env.action_manager.action.clone(),
    "prev_action": env.action_manager.prev_action.clone(),
  }


def test_masked_reset_matches_the_scoped_reset(device) -> None:
  env = _env(device)
  _advance(env)
  start = _state(env)
  mask = torch.tensor([True, False, True, False, False, True], device=env.device)
  ids = mask.nonzero().flatten()

  obs_scoped, _ = env.reset(env_ids=ids)
  scoped = _state(env)
  scoped_obs = {key: value.clone() for key, value in obs_scoped.items()}

  env.sim.data.qpos[:] = start["qpos"]
  env.sim.data.qvel[:] = start["qvel"]
  env.episode_length_buf.copy_(start["episode_length"])
  env.action_manager._action.copy_(start["action"])
  env.action_manager._prev_action.copy_(start["prev_action"])
  # A step's observations are what the masked reset keeps for unmasked rows.
  env.obs_buf = env.observation_manager.compute(update_history=False)
  with env.program():
    obs_masked, extras = env.reset_masked(mask)
  masked = _state(env)

  for key in scoped:
    assert torch.equal(scoped[key], masked[key]), key
  for key, value in obs_masked.items():
    assert torch.equal(value, scoped_obs[key]), key
  assert not bool(env._manual_reset_pending.any())
  assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in extras["log"].values())
  env.close()


def test_zero_mask_changes_nothing(device) -> None:
  env = _env(device)
  _advance(env)
  before = _state(env)
  with env.program():
    env.reset_masked(torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
  after = _state(env)
  assert all(torch.equal(before[key], after[key]) for key in before)
  env.close()


def test_program_mode_guards(device) -> None:
  env = _env(device)
  with pytest.raises(RuntimeError, match="device clock"):
    with env.program():
      _ = env.common_step_counter
  env.common_step_counter = 7
  assert int(env.step_clock) == 7
  env.advance_host_clock(2)
  assert env.common_step_counter == 9
  with env.program():
    env.step(torch.zeros((env.num_envs, 1), device=env.device))
  assert int(env.step_clock) == 8  # the program advanced only the device clock
  env.close()


def test_program_refuses_what_it_cannot_mask(device) -> None:
  cfg = cartpole_balance_env_cfg()
  cfg.scene.num_envs = 2
  cfg.auto_reset = False
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  env.reset()
  env.prepare_static_program()
  assert env.event_manager.available_modes
  with pytest.raises(NotImplementedError, match="supports_env_mask|masked"):
    with env.program():
      env.reset_masked(torch.ones(2, dtype=torch.bool, device=env.device))
  env.close()
  cfg = cartpole_balance_env_cfg()
  cfg.scene.num_envs = 2
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  with pytest.raises(RuntimeError, match="auto_reset"):
    env.prepare_static_program()
  env.close()


def test_a_mask_aware_event_receives_an_env_mask(device) -> None:
  seen = []

  def mark(env, env_ids):
    seen.append(env_ids)

  mark.supports_env_mask = True
  cfg = cartpole_balance_env_cfg()
  cfg.scene.num_envs = 3
  cfg.auto_reset = False
  cfg.events = {"mark": EventTermCfg(func=mark, mode="reset")}
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  env.reset()
  env.prepare_static_program()
  mask = torch.tensor([True, False, True], device=env.device)
  with env.program():
    env.reset_masked(mask)
  assert isinstance(seen[-1], EnvMask) and torch.equal(seen[-1].mask, mask)
  env.close()


def test_mask_helpers() -> None:
  mask = torch.tensor([True, False, True])
  old = torch.zeros(3, 2)
  new = torch.arange(6.0).reshape(3, 2)
  assert torch.equal(where_rows(mask, new, old), torch.tensor([[0.0, 1.0], [0.0, 0.0], [4.0, 5.0]]))
  target = old.clone()
  assign_rows_(target, mask, new)
  fill_rows_(old, mask, 9.0)
  assert torch.equal(target[mask], new[mask]) and not target[~mask].any()
  assert torch.equal(old[:, 0], torch.tensor([9.0, 0.0, 9.0]))
  mean, valid = mean_log(torch.tensor([1.0, 100.0, 3.0]), mask)
  assert float(mean) == 2.0 and bool(valid)
  mean, valid = mean_log(torch.tensor([1.0, 100.0, 3.0]), torch.zeros(3, dtype=torch.bool))
  assert float(mean) == 0.0 and not bool(valid)
  # -inf in an unmasked world does not poison the mean.
  mean, _ = mean_log(torch.tensor([1.0, float("-inf"), 3.0]), mask)
  assert float(mean) == 2.0
  with pytest.raises(ValueError):
    EnvMask(torch.zeros(3, dtype=torch.long))
