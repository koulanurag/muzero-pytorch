import gym
from torch.nn import MSELoss

from core.config import MuZeroConfig as MC
from .env_wrapper import ClassicControlWrapper
from .model import MuZeroNet


class ClassicControlConfig(MC):
    def __init__(self):
        super(ClassicControlConfig, self).__init__(
            training_steps=10000,
            test_interval=20,
            test_episodes=2,
            checkpoint_interval=10,
            max_moves=1000,
            discount=0.997,
            dirichlet_alpha=0.25,
            num_simulations=50,
            batch_size=128,
            td_steps=10,
            num_actors=10,
            lr_init=0.05,
            lr_decay_steps=3500,
            window_size=10000)

    def visit_softmax_temperature_fn(self, num_moves, trained_steps):
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25

    def set_game(self, env_name):
        self.env_name = env_name
        game = self.new_game()
        self.obs_shape = game.reset().shape[0]
        self.action_space_size = game.action_space_size

    def get_uniform_network(self):
        return MuZeroNet(self.obs_shape, self.action_space_size)

    def new_game(self, seed=None):
        env = gym.make(self.env_name)
        if seed is not None:
            env.seed(seed)
        return ClassicControlWrapper(env, discount=self.discount, k=1)

    def scalar_loss(self, prediction, target):
        return MSELoss(reduction='none')(prediction, target)


muzero_config = ClassicControlConfig()
