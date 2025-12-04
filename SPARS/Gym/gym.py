# gym_env.py
import copy
import numpy as np
import gymnasium as gym
from SPARS.Logger import log_info, log_trace
import torch as T

# import your real Simulator and RJMS
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

    def __init__(self, simulator, device=CPU_DEVICE):
        super().__init__()

        self.simulator = simulator
        self.device = device

    def step(self, actions):
        log_trace('============= CALL RL ================')
        state = self.simulator.PlatformControl.get_state()
        log_info(f"Current Time: {self.simulator.current_time}")

        s = repr(actions.detach().cpu()).replace("\n", " ")
        log_info(f"Action taken: {s}")
        
        

        """"Action translator for Scalar Active Target"""
        rl_events = action_translator(
            self.simulator.Monitor.num_nodes, state, actions, self.simulator.current_time)


        monitor = {
            'energy': copy.deepcopy(self.simulator.Monitor.energy),
            'ecr': copy.deepcopy(self.simulator.Monitor.ecr),
            'nodes_state': copy.deepcopy(self.simulator.Monitor.nodes_state),
        }


        for _rl_event in rl_events:
            self.simulator.push_event(
                timestamp=_rl_event['time'], event=_rl_event['event'])

        need_rl = False

        prev_current_time = self.simulator.current_time
        skipped = False
        
        while not need_rl and self.simulator.is_running:
            if rl_events or skipped:
                events = self.simulator.proceed()

                for event_list in events['event_list']:
                    for event in event_list['events']:
                        if event['type'] == 'CALL_RL':
                            need_rl = True
                            break
                    if need_rl:
                        break
                if need_rl:
                    break

            if not rl_events and not skipped:
                skipped = True

            scheduler_message = self.simulator.scheduler.schedule(
                self.simulator.current_time)

            for _data in scheduler_message:
                timestamp = _data['timestamp']
                _events = _data['events']
                for event in _events:
                    self.simulator.push_event(timestamp, event)

        next_current_time = self.simulator.current_time
        reward_function = Reward()
        future_monitor = self.simulator.Monitor

        """SPARS Calculate Reward"""
        reward = reward_function.calculate_reward(
            self.simulator.Monitor, future_monitor, self.simulator.current_time, next_current_time)



        done = not self.simulator.is_running
        observation = self.get_observation()


        return observation, reward, done

    def reset(self, simulator):
        self.simulator = simulator

    def get_observation(self):
        features = feature_extraction(self.simulator)
        features_ = T.from_numpy(features).to(self.device).float()
        observation = (features_)

        return observation
