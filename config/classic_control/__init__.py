import gym
from torch.nn import MSELoss

from core.config import BaseMuZeroConfig
from .env_wrapper import ClassicControlWrapper
from .model import MuZeroNet


class ClassicControlConfig(BaseMuZeroConfig):
    def __init__(self):
        super(ClassicControlConfig, self).__init__(
            training_steps=10000,
            test_interval=100,
            test_episodes=5,
            checkpoint_interval=20,
            max_moves=1000,
            discount=0.997,
            dirichlet_alpha=0.25,
            num_simulations=50,
            batch_size=128,
            td_steps=5,
            num_actors=32,
            lr_init=0.01,
            lr_decay_rate=0.01,
            lr_decay_steps=10000,
            window_size=1000,
            value_loss_coeff=1)

    def visit_softmax_temperature_fn(self, num_moves, trained_steps):
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25

    def set_game(self, env_name, save_video=False, save_path=None, video_callable=None):
        self.env_name = env_name
        game = self.new_game()
        self.obs_shape = game.reset().shape[0]
        self.action_space_size = game.action_space_size

    def get_uniform_network(self):
        return MuZeroNet(self.obs_shape, self.action_space_size)

    def new_game(self, seed=None, save_video=False, save_path=None, video_callable=None):
        env = gym.make(self.env_name)
        if seed is not None:
            env.seed(seed)

        if save_video:
            from gym.wrappers import Monitor
            env = Monitor(env, directory=save_path, force=True, video_callable=video_callable)
        return ClassicControlWrapper(env, discount=self.discount, k=1)

    def scalar_reward_loss(self, prediction, target):
        return MSELoss(reduction='none')(prediction, target)

    def scalar_value_loss(self, prediction, target):
        return MSELoss(reduction='none')(prediction, target)

    def scalar_loss(self, prediction, target):
        return MSELoss(reduction='none')(prediction, target)


muzero_config = ClassicControlConfig()
