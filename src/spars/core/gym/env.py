from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import torch as T

from spars.core.gym.components import (
    Reward,
    action_translator,
    feature_extraction,
    reset_component_state,
)
from spars.core.logging import log_trace
from spars.core.simulator.simulator import EVENT_DRIVEN_RL_TYPES

if TYPE_CHECKING:
    from spars.core.simulator.simulator import Simulator


CPU_DEVICE = T.device("cpu")


class HPCGymEnv(gym.Env):
    """Gym wrapper that advances the simulator between RL decision points."""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        simulator: Simulator,
        training: bool,
        device: T.device = CPU_DEVICE,
    ) -> None:
        super().__init__()

        self.simulator = simulator
        self.device = device
        self.training = training
        self._reset_feature_state()

    @staticmethod
    def _reset_feature_state() -> None:
        reset_component_state(feature_extraction)

    def _push_rl_events(
        self,
        rl_events: list[dict[str, Any]],
    ) -> None:
        for rl_event in rl_events:
            self.simulator.push_event(
                timestamp=rl_event["time"],
                event=rl_event["event"],
            )

    def _schedule_current_time(self) -> None:
        self.simulator.planner.update(
            current_time=self.simulator.current_time,
            push_event_buckets=self.simulator._push_event_buckets,
        )

    def start_episode(self) -> None:
        self.simulator.start_simulator()
        self._schedule_current_time()
        if self.simulator.rl_type in EVENT_DRIVEN_RL_TYPES:
            self._advance_until_next_rl_call()

    def _settle_and_advance(self) -> None:
        """Run every event at the current time, replan, then go to the next call.

        Never advances a future event before the current timestamp is drained,
        so scheduler allocations made now are applied before planning again.
        """
        while (
            self.simulator.is_running
            and self.simulator.events
            and self.simulator.events[0]["timestamp"] == self.simulator.current_time
        ):
            self.simulator.proceed()
        if self.simulator.is_running:
            self._schedule_current_time()
            self._advance_until_next_rl_call()

    def _advance_until_next_rl_call(self) -> None:
        need_rl = False

        while not need_rl and self.simulator.is_running:
            proceeded_events = self.simulator.proceed()

            for event_list in proceeded_events["event_list"]:
                for event in event_list["events"]:
                    if event["type"] == "CALL_RL":
                        need_rl = True
                        break

                if need_rl:
                    break

            if self.simulator.is_running:
                self._schedule_current_time()
            if need_rl:
                break

    def step(
        self,
        obs: Any,
        logits: T.Tensor,
    ) -> tuple[Any, dict[str, Any], bool]:
        log_trace("========== RL POLICY DECISION ==========")
        log_trace(f"Current Time: {self.simulator.current_time}")

        reward_function = Reward()
        # A reward may declare the pre-action state it reads; deep-copying the
        # whole simulator remains the fallback for rewards that do not.
        snapshot = getattr(reward_function, "snapshot", None)
        pre_action_simulator = (
            snapshot(self.simulator)
            if snapshot is not None
            else copy.deepcopy(self.simulator)
        )

        action_time = self.simulator.current_time
        rl_events, actions, logprob = action_translator(logits, self.simulator)
        self._push_rl_events(rl_events)
        log_trace("Translated power events: %s", rl_events)

        self._settle_and_advance()

        next_obs = self.get_observation()

        reward = reward_function.calculate_reward(
            obs,
            next_obs,
            pre_action_simulator,
            self.simulator,
            actions,
        )

        done = not self.simulator.is_running
        self.simulator.monitor.flush_state_hist_if_safe(
            force=done,
        )

        ppo_info = {
            "reward": reward,
            "actions": actions,
            "logprob": logprob,
        }

        return next_obs, ppo_info, done

    def reset(self, simulator: Simulator) -> None:
        self.simulator = simulator
        self._reset_feature_state()

    def get_observation(self) -> Any:
        return feature_extraction(
            simulator=self.simulator,
            training=self.training,
        )
