import argparse
import logging.config
import os

import numpy as np
import ray
import torch
from torch.utils.tensorboard import SummaryWriter

from core.test import test
from core.train import train
from core.utils import init_logger, make_results_dir

ray.init()
if __name__ == '__main__':
    # Lets gather arguments
    parser = argparse.ArgumentParser(description='MuZero Pytorch Implementation')
    parser.add_argument('--env', required=True, help='Name of the environment')
    parser.add_argument('--result_dir', default=os.path.join(os.getcwd(), 'results'),
                        help="Directory Path to store results (default: %(default)s)")
    parser.add_argument('--case', required=True, choices=['atari', 'classic_control'],
                        help="It's used for switching between different domains(default: %(default)s)")
    parser.add_argument('--opr', required=True, choices=['train', 'test'])
    parser.add_argument('--no_cuda', action='store_true', default=False, help='no cuda usage')
    parser.add_argument('--render', action='store_true', default=False, help='Renders the environment')
    parser.add_argument('--force', action='store_true', default=False, help='Overrides past results')
    parser.add_argument('--seed', type=int, default=0, help='seed (default: %(default)s)')
    parser.add_argument('--test_episodes', type=int, default=10, help='Evaluation episode count (default: %(default)s)')
    parser.add_argument('--log_suffix', type=str, default='',
                        help='Log Suffix Attached to the resulting directory (default: %(default)s)')

    # Process arguments
    args = parser.parse_args()
    args.device = 'cuda' if (not args.no_cuda) and torch.cuda.is_available() else 'cpu'

    # create relative paths to store results
    exp_path = make_results_dir(os.path.join(args.result_dir, args.env, 'seed_{}'.format(args.seed)), args)

    # set-up logger and tensorboard
    init_logger(os.path.join(exp_path, args.opr + '.log'))
    logger = logging.getLogger()
    summary_writer = SummaryWriter(exp_path, flush_secs=10)

    # seeding random iterators
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    try:
        # import corresponding configuration , neural networks and envs
        if args.case == 'atari':
            from config.atari import muzero_config
        elif args.case == 'classic_control':
            from config.classic_control import muzero_config
        else:
            raise Exception('Invalid --case option')

        muzero_config.set_game(args.env)
        muzero_config.set_exp_path(exp_path)
        muzero_config.set_device(args.device)

        if args.opr == 'train':
            train(muzero_config, summary_writer)

        elif args.opr == 'test':
            assert os.path.exists(muzero_config.model_path), 'model not found at {}'.format(muzero_config.model_path)
            model = muzero_config.get_uniform_network().to('cpu')
            model.load_state_dict(torch.load(muzero_config.model_path, map_location=torch.device('cpu')))
            test_score = test(muzero_config, model, args.test_episodes, device='cpu', render=args.render)
            logger.info('Test Score: {}'.format(test_score))
        else:
            raise Exception('Please select a valid operation(--opr) to be performed')

        ray.shutdown()
    except Exception as e:
        logger.error(e, exc_info=True)
