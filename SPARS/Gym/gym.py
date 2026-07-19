# gym_env.py
import copy
import numpy as np
import gymnasium as gym
from SPARS.Logger import log_info, log_trace
import torch as T

from SPARS.Gym.utils import Reward, action_translator

from SPARS.Gym.utils import feature_extraction

CPU_DEVICE = T.device("cpu")

class HPCGymEnv(gym.Env):
    """
    Gymnasium environment that ONLY wraps Simulator + RJMS.
    Responsibilities:
      - advance_system(): run sim -> rjms -> apply rjms events -> return features (pre-action)
      - apply_action(action): translate agent action -> apply to sim -> compute reward (prev vs next)
      - step(action): helper for Gym compatibility -> advance_system + apply_action
    No agent/critic/memory inside.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, simulator, training, device=CPU_DEVICE):
        super().__init__()

        self.simulator = simulator
        self.device = device
        self.training=training

    def step(self, obs, logits):
        log_trace("============= CALL RL ================")
        log_trace(
            f"Current Time: {self.simulator.current_time}"
        )

        pre_action_simulator = copy.deepcopy(
            self.simulator
        )

        # 1. Translate the selected action into simulator events.
        rl_events, actions, logprob = action_translator(
            logits,
            self.simulator,
        )

        # 2. Add the RL events to the simulator.
        for rl_event in rl_events:
            self.simulator.push_event(
                timestamp=rl_event["time"],
                event=rl_event["event"],
            )

        # 3. Process the RL switching events.
        # If the action does not cause any switching,
        # there is nothing to process here.
        if rl_events:
            self.simulator.proceed()

        # 4. Run the scheduler after applying the RL action.
        scheduler_message = (
            self.simulator.scheduler.schedule(
                self.simulator.current_time
            )
        )

        # 5. Add the scheduler-generated events.
        for data in scheduler_message:
            timestamp = data["timestamp"]
            events = data["events"]

            for event in events:
                self.simulator.push_event(
                    timestamp,
                    event,
                )

        # 6. Continue the simulation until the next CALL_RL.
        need_rl = False

        while (
            not need_rl
            and self.simulator.is_running
        ):
            proceeded_events = self.simulator.proceed()

            for event_list in proceeded_events["event_list"]:
                for event in event_list["events"]:
                    if event["type"] == "CALL_RL":
                        need_rl = True
                        break

                if need_rl:
                    break

            # CALL_RL was found, so skip the scheduler
            # and leave this loop.
            
            # 7. If no RL call then call scheduler, if there's then break out and call RL immediately
            if need_rl:
                break

            scheduler_message = (
                self.simulator.scheduler.schedule(
                    self.simulator.current_time
                )
            )

            for data in scheduler_message:
                timestamp = data["timestamp"]
                events = data["events"]

                for event in events:
                    self.simulator.push_event(
                        timestamp,
                        event,
                    )

        # 7. Obtain the state at the next RL decision point.
        next_obs = self.get_observation()

        # 8. Calculate the reward over the transition.
        reward_function = Reward()

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


    def reset(self, simulator):
        self.simulator = simulator

    def get_observation(self):
        obs = feature_extraction(simulator=self.simulator,training=self.training)
        return obs
