import os

import torch

from .mcts import MCTS, Node
from .utils import select_action


def test(config, model, episodes, device, render, save_video=False):
    model.to(device)
    model.eval()

    test_reward = 0
    save_path = os.path.join(config.exp_path, 'recordings')
    env = config.new_game(save_video=save_video, save_path=save_path,
                          video_callable=lambda episode_id: True)
    with torch.no_grad():
        for ep_i in range(episodes):
            done = False
            ep_reward = 0
            obs = env.reset()
            while not done:
                if render:
                    env.render()
                root = Node(0)
                obs = torch.FloatTensor(obs).to(device).unsqueeze(0)
                root.expand(env.to_play(), env.legal_actions(), model.initial_inference(obs))
                MCTS(config).run(root, env.action_history(), model)
                action = select_action(root, temperature=1, deterministic=True)
                obs, reward, done, info = env.step(action.index)
                ep_reward += reward
            test_reward += ep_reward

    return test_reward / episodes
