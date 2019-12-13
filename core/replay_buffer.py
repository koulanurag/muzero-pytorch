import numpy as np
import ray
import torch


@ray.remote
class ReplayBuffer(object):
    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.batch_size = batch_size
        self.buffer = []
        self._eps_colleted = 0

    def save_game(self, game):
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)
        self.buffer.append(game)
        self._eps_colleted += 1

    def sample_batch(self, num_unroll_steps: int, td_steps: int):

        obs_batch, action_batch, reward_batch, value_batch, policy_batch = [], [], [], [], []

        # these are just place holders for now
        weights = torch.ones(self.batch_size)
        indices = torch.ones(self.batch_size)

        for _ in range(self.batch_size):
            game = self.sample_game()
            game_pos = self.sample_position(game)
            obs_batch.append(game.obs(game_pos))
            action_batch.append(game.history[game_pos:game_pos + num_unroll_steps])
            value, reward, policy = game.make_target(game_pos, num_unroll_steps, td_steps)
            reward_batch.append(reward)
            value_batch.append(value)
            policy_batch.append(policy)

        _zip = zip(obs_batch, action_batch, reward_batch, value_batch, policy_batch, weights, indices)
        batch = zip(*sorted(_zip, key=lambda x: len(x[1]), reverse=True))
        obs_batch, action_batch, reward_batch, value_batch, policy_batch, weights, indices = batch

        obs_batch = torch.tensor(obs_batch).float()
        reward_batch = torch.tensor(reward_batch).float()
        value_batch = torch.tensor(value_batch).float()
        policy_batch = torch.tensor(policy_batch).float()

        # pack sequences
        action_lens = torch.tensor([len(x) for x in action_batch]).float()
        action_batch = [torch.tensor(x).long() for x in action_batch]
        action_batch = torch.nn.utils.rnn.pack_sequence(action_batch)

        indices = torch.tensor(indices)
        weights = torch.tensor(weights)

        return obs_batch, action_batch, reward_batch, value_batch, policy_batch, indices, weights, action_lens

    def sample_game(self):
        # Sample game from buffer either uniformly or according to some priority.
        return self.buffer[np.random.choice(range(self.size()))]

    def sample_position(self, game) -> int:
        # Sample position from game either uniformly or according to some priority.
        return np.random.choice(range(len(game)))

    def size(self):
        return len(self.buffer)

    def episodes_collected(self):
        return self._eps_colleted
