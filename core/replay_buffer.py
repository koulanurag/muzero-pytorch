import numpy as np
import ray
import torch


@ray.remote
class ReplayBuffer(object):
    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.batch_size = batch_size
        self.buffer = []
        self._eps_collected = 0

    def save_game(self, game):
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)
        self.buffer.append(game)
        self._eps_collected += 1

    def sample_batch(self, num_unroll_steps: int, td_steps: int):

        obs_batch, action_batch, reward_batch, value_batch, policy_batch = [], [], [], [], []

        # these are just place holders for now
        weights = torch.ones(self.batch_size)
        indices = torch.ones(self.batch_size)

        for _ in range(self.batch_size):
            game = self.sample_game()
            game_pos = self.sample_position(game)
            _actions = game.history[game_pos:game_pos + num_unroll_steps]
            # random action selection to complete num_unroll_steps
            _actions += [np.random.randint(0, game.action_space_size)
                         for _ in range(num_unroll_steps - len(_actions) + 1)]

            obs_batch.append(game.obs(game_pos))
            action_batch.append(_actions)
            value, reward, policy = game.make_target(game_pos, num_unroll_steps, td_steps)
            reward_batch.append(reward)
            value_batch.append(value)
            policy_batch.append(policy)

        obs_batch = torch.tensor(obs_batch).float()
        action_batch = torch.tensor(action_batch).long()
        reward_batch = torch.tensor(reward_batch).float()
        value_batch = torch.tensor(value_batch).float()
        policy_batch = torch.tensor(policy_batch).float()

        return obs_batch, action_batch, reward_batch, value_batch, policy_batch, indices, weights

    def sample_game(self):
        # Sample game from buffer either uniformly or according to some priority.
        return self.buffer[np.random.choice(range(self.size()))]

    def sample_position(self, game) -> int:
        # Sample position from game either uniformly or according to some priority.
        return np.random.choice(range(len(game)))

    def size(self):
        return len(self.buffer)

    def episodes_collected(self):
        return self._eps_collected
