import os

from .game import Game


class BaseMuZeroConfig(object):

    def __init__(self,
                 training_steps: int,
                 test_interval: int,
                 test_episodes: int,
                 checkpoint_interval: int,
                 max_moves: int,
                 discount: float,
                 dirichlet_alpha: float,
                 num_simulations: int,
                 batch_size: int,
                 td_steps: int,
                 num_actors: int,
                 lr_init: float,
                 lr_decay_rate: float,
                 lr_decay_steps: float,
                 window_size: int = int(1e6),
                 value_loss_coeff: float = 1, ):
        # Self-Play
        self.action_space_size = None
        self.num_actors = num_actors

        self.max_moves = max_moves
        self.num_simulations = num_simulations
        self.discount = discount
        self.max_grad_norm = 10

        # testing arguments
        self.test_interval = test_interval
        self.test_episodes = test_episodes

        # Root prior exploration noise.
        self.root_dirichlet_alpha = dirichlet_alpha
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # If we already have some information about which values occur in the environment, we can use them to
        # initialize the rescaling. This is not strictly necessary, but establishes identical behaviour to
        # AlphaZero in board games.
        self.max_value_bound = None
        self.min_value_bound = None

        # Training
        self.training_steps = training_steps
        self.checkpoint_interval = checkpoint_interval
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_unroll_steps = 5
        self.td_steps = td_steps
        self.value_loss_coeff = value_loss_coeff

        self.weight_decay = 1e-4
        self.momentum = 0.9

        # Exponential learning rate schedule
        self.lr_init = lr_init
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_steps = lr_decay_steps

        self.device, self.exp_path = 'cpu', None
        self.debug = False
        self.priority_prob_alpha = 1
        self.use_target_model = True
        self.model_path = None
        self.seed = None
        self.revisit_policy_search_rate = 0
        self.use_max_priority = None

    def visit_softmax_temperature_fn(self, num_moves, trained_steps):
        raise NotImplementedError

    def set_game(self, env_name):
        raise NotImplementedError

    def new_game(self, seed=None, save_video=False, save_path=None, video_callable=None) -> Game:
        """ returns a new instance of the game"""
        raise NotImplementedError

    def get_uniform_network(self):
        raise NotImplementedError

    def scalar_loss(self, prediction, target):
        raise NotImplementedError

    def get_hparams(self):
        hparams = {}
        for k, v in self.__dict__.items():
            if 'path' not in k and (v is not None):
                hparams[k] = v
        return hparams

    def disable_priority(self):
        self.priority_prob_alpha = 0

    def disable_target_model(self):
        self.use_target_model = False

    def set_config(self, args):
        self.set_game(args.env)
        self.seed = args.seed
        self.priority_prob_alpha = 0 if args.no_priority else 1
        self.use_target_model = not args.no_target_model
        self.debug = args.debug
        self.device = args.device
        self.use_max_priority = args.use_max_priority

        if args.value_loss_coeff is not None:
            self.value_loss_coeff = args.value_loss_coeff

        if args.revisit_policy_search_rate is not None:
            self.revisit_policy_search_rate = args.revisit_policy_search_rate

        self.exp_path = os.path.join(args.result_dir, args.env,
                                     'revisit_rate_{}'.format(self.revisit_policy_search_rate),
                                     'val_coeff_{}'.format(self.value_loss_coeff),
                                     'no_target' if self.use_target_model else 'with_target',
                                     'no_prio' if args.no_priority else 'with_prio',
                                     'max_prio' if args.use_max_priority else 'no_max_prio',
                                     'seed_{}'.format(args.seed))

        self.model_path = os.path.join(self.exp_path, 'model.p')
        return self.exp_path
