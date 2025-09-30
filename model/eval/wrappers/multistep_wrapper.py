# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict, deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces


def stack_repeated(x, n, loc):
    return np.repeat(np.expand_dims(x, axis=loc), n, axis=loc)


def repeated_box(box_space, n, loc):
    return spaces.Box(
        low=stack_repeated(box_space.low, n, loc),
        high=stack_repeated(box_space.high, n, loc),
        shape=box_space.shape[:loc] + (n,) + box_space.shape[loc:],
        dtype=box_space.dtype,
    )


def repeated_space(space, n, loc=0):
    if isinstance(space, spaces.Box):
        return repeated_box(space, n, loc)
    elif isinstance(space, spaces.Dict):
        result_space = spaces.Dict()
        for key, value in space.items():
            result_space[key] = repeated_space(value, n, loc)
        return result_space
    elif isinstance(space, spaces.Discrete):
        return spaces.MultiDiscrete([[space.n] for _ in range(n)])
    elif isinstance(space, spaces.Text):  # For language, we don't repeat and only keep the last one
        return space
    else:
        raise RuntimeError(f"Unsupported space type {type(space)}")


def take_last_n(x, n):
    x = list(x)
    n = min(len(x), n)
    return np.array(x[-n:])


def dict_take_last_n(x, n):
    result = dict()
    for key, value in x.items():
        result[key] = take_last_n(value, n)
    return result


def aggregate(data, method="max"):
    if method == "max":
        # equivalent to any
        return np.max(data)
    elif method == "min":
        # equivalent to all
        return np.min(data)
    elif method == "mean":
        return np.mean(data)
    elif method == "sum":
        return np.sum(data)
    else:
        raise NotImplementedError()


class MultiStepWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        video_delta_indices,
        state_delta_indices,
        n_action_steps,
        max_episode_steps=None,
        reward_agg_method="max",
    ):
        """
        video_delta_indices: np.ndarray[int], please check `assert_delta_indices` to see the requirements
        state_delta_indices: np.ndarray[int] | None, please check `assert_delta_indices` to see the requirements
          if None, it means the model is vision-only
        """
        super().__init__(env)
        # Assign action space
        self._action_space = repeated_space(env.action_space, n_action_steps)

        # Assign delta indices and horizons
        self.video_delta_indices = video_delta_indices
        self.video_horizon = len(video_delta_indices)
        self.assert_delta_indices(self.video_delta_indices, self.video_horizon)
        if state_delta_indices is not None:
            self.state_delta_indices = state_delta_indices
            self.state_horizon = len(state_delta_indices)
            self.assert_delta_indices(self.state_delta_indices, self.state_horizon)
        else:
            self.state_horizon = None
            self.state_delta_indices = None

        # Assign observation space
        self._observation_space = self.convert_observation_space(
            self.observation_space,
            self.video_horizon,
            self.state_horizon,
        )

        # Assign other attributes
        self.max_episode_steps = max_episode_steps
        self.n_action_steps = n_action_steps
        self.reward_agg_method = reward_agg_method
        self.max_steps_needed = self.get_max_steps_needed()

        self.obs = deque(maxlen=self.max_steps_needed + 1)
        self.reward = list()
        self.done = list()
        self.info = defaultdict(lambda: deque(maxlen=self.max_steps_needed + 1))

    def convert_observation_space(self, observation_space, video_horizon, state_horizon):
        """
        For video, the observation space will be (video_horizon,) + original shape
        For state (if not None), the observation space will be (state_horizon,) + original shape
        """
        new_observation_space = {}
        for k in observation_space.keys():
            if k.startswith("video"):
                box = observation_space[k]
                horizon = video_horizon
                new_observation_space[k] = repeated_space(box, horizon)
            elif k.startswith("state"):
                box = observation_space[k]
                if state_horizon is not None:
                    horizon = state_horizon
                else:
                    # Don't include the state in the observation space
                    continue
                new_observation_space[k] = repeated_space(box, horizon)
            elif k.startswith("annotation"):
                text = observation_space[k]
                new_observation_space[k] = text
            else:
                raise ValueError(f"Unknown key: {k}")  # NOTE: We might add "language" in the future

        return spaces.Dict(new_observation_space)

    def get_max_steps_needed(self):
        """
        Get the maximum number of steps that we need to cache.
        """
        video_max_steps_needed = (
            np.max(self.video_delta_indices) - np.min(self.video_delta_indices) + 1
        )
        if self.state_delta_indices is not None:
            state_max_steps_needed = (
                np.max(self.state_delta_indices) - np.min(self.state_delta_indices) + 1
            )
        else:
            state_max_steps_needed = 0
        return int(max(video_max_steps_needed, state_max_steps_needed))

    def assert_delta_indices(self, delta_indices: np.ndarray, horizon: int):
        # Check the length
        # (In this wrapper, this seems redundant because we get the horizon from the delta indices. But in the policy, the horizon is not derived from the delta indices but we need to make it consistent. To make the function consistent, we keep the check here.)
        assert len(delta_indices) == horizon, f"{delta_indices=}, {horizon=}"
        # All delta indices should be non-positive because there's no way to get the future observations
        assert np.all(delta_indices <= 0), f"{delta_indices=}"
        # The last delta index should be 0 because it doesn't make sense to not use the latest observation
        assert delta_indices[-1] == 0, f"{delta_indices=}"
        if len(delta_indices) > 1:
            # The step is consistent (because in real robot experiments, we actually use the dt to get the observations, which requires the step to be consistent)
            assert np.all(
                np.diff(delta_indices) == delta_indices[1] - delta_indices[0]
            ), f"{delta_indices=}"
            # And the step is positive
            assert (delta_indices[1] - delta_indices[0]) > 0, f"{delta_indices=}"

    def reset(self, seed=None, options=None):
        """Resets the environment using kwargs."""
        obs, info = super().reset(seed=seed, options=options)

        self.obs = deque([obs] * (self.max_steps_needed + 1), maxlen=self.max_steps_needed + 1)
        self.reward = list()
        self.done = list()
        self.info = defaultdict(lambda: deque(maxlen=self.max_steps_needed + 1))

        obs = self._get_obs(self.video_delta_indices, self.state_delta_indices)
        info = {k: [v] for k, v in info.items()}
        return obs, info

    def step(self, action):
        """
        action: dict: key-value pairs where the values are of shape (n_action_steps,) + action_shape
        """
        states = []
        rewards = []
        dones = []
        for step in range(self.n_action_steps):
            act = {}
            for key, value in action.items():
                act[key] = value[step, :]
            if len(self.done) > 0 and self.done[-1]:
                # termination
                break
            observation, reward, done, truncated, info = super().step(act)
            env_state = {"states": [], "model": []}
            states.append(env_state["states"])
            rewards.append(reward)
            dones.append(done)
            self.obs.append(observation)
            self.reward.append(reward)
            if (self.max_episode_steps is not None) and (
                len(self.reward) >= self.max_episode_steps
            ):
                # truncation
                done = True
            self.done.append(done)
            self._add_info(info)

        observation = self._get_obs(self.video_delta_indices, self.state_delta_indices)
        reward = aggregate(self.reward, self.reward_agg_method)
        done = aggregate(self.done, "max")
        info = dict_take_last_n(self.info, self.max_steps_needed)
        states = np.array(states)
        rewards = np.array(rewards)
        dones = np.array(dones)
        info["states"] = states
        info["rewards"] = rewards
        info["model"] = env_state["model"]
        info["actions"] = action
        info["dones"] = dones
        return observation, reward, done, truncated, info

    def _get_obs(self, video_delta_indices, state_delta_indices):
        """
        Output:
        For video: (video_horizon,) + obs_shape
        For state (if not None): (state_horizon,) + obs_shape
        """
        assert len(self.obs) > 0
        if isinstance(self.observation_space, spaces.Dict):
            result = dict()
            for key in self.observation_space.keys():
                if key.startswith("video"):
                    """
                    NOTE:
                      We need to subtract 1 because video_delta_indices is 0-indexed.
                      E.g., video_delta_indices = np.array([-4, -3, -2, -1, 0])
                      Then when we select the observation,
                        it should be [obs[-5], obs[-4], obs[-3], obs[-2], obs[-1]]
                      (i.e., the latest observation is at the last index)
                    """
                    delta_indices = video_delta_indices - 1
                    this_obs = [self.obs[i][key] for i in delta_indices]
                    result[key] = np.stack(this_obs, axis=0)
                elif key.startswith("state"):
                    if state_delta_indices is not None:
                        delta_indices = state_delta_indices - 1
                    else:
                        raise ValueError(
                            f"state_delta_indices is None but `state` is still in the {self.observation_space=}"
                        )
                    this_obs = [self.obs[i][key] for i in delta_indices]
                    result[key] = np.stack(this_obs, axis=0)
                elif key.startswith("annotation"):
                    result[key] = self.obs[-1][key]
                else:
                    raise ValueError(f"Unknown key: {key}")
            return result
        else:
            raise RuntimeError(f"Unsupported space type: {type(self.observation_space)=}")

    def _add_info(self, info):
        for key, value in info.items():
            self.info[key].append(value)

    def get_rewards(self):
        return self.reward

    def get_attr(self, name):
        return getattr(self, name)

    def get_infos(self):
        result = dict()
        for k, v in self.info.items():
            result[k] = list(v)
        return result
