import logging

import ray
import torch
import torch.optim as optim
from torch.nn import L1Loss

from .mcts import MCTS, Node
from .replay_buffer import ReplayBuffer
from .test import test
from .utils import select_action

train_logger = logging.getLogger('train')
test_logger = logging.getLogger('train_test')


def _log(config, step_count, loss_data, model, replay_buffer, test_score, best_test_score, lr, worker_logs,
         summary_writer):
    weighted_loss, loss, policy_loss, reward_loss, value_loss, target_reward, target_value, pred_reward, pred_value, batch_weights, batch_indices = loss_data
    worker_reward, worker_eps_len = worker_logs
    replay_episodes_collected = ray.get(replay_buffer.episodes_collected.remote())
    replay_buffer_size = ray.get(replay_buffer.size.remote())
    _msg = '#{:<10} Loss: {:<8.3f} [weighted Loss:{:<8.3f} Policy Loss: {:<8.3f} Value Loss: {:<8.3f} ' \
           'Reward Loss: {:<8.3f} ] Replay Episodes Collected: {:<10d} Buffer Size: {:<10d} Lr: {:<8.3f}'
    _msg = _msg.format(step_count, loss, weighted_loss, policy_loss, value_loss, reward_loss,
                       replay_episodes_collected, replay_buffer_size, lr)
    train_logger.info(_msg)

    if test_score is not None:
        test_msg = '#{:<10} Test Score: {:<10} Best Test Score: {:<10}'.format(step_count, test_score, best_test_score)
        test_logger.info(test_msg)

    if summary_writer is not None:
        if config.debug:
            for name, W in model.named_parameters():
                summary_writer.add_histogram('after_grad_clip' + '/' + name + '_grad', W.grad.data.cpu().numpy(),
                                             step_count)
                summary_writer.add_histogram('network_weights' + '/' + name, W.data.cpu().numpy(), step_count)
        summary_writer.add_histogram('train/replay_buffer_priorities', ray.get(replay_buffer.get_priorities.remote()),
                                     step_count)
        summary_writer.add_histogram('train/batch_weight', batch_weights, step_count)
        summary_writer.add_histogram('train/batch_indices', batch_indices, step_count)

        summary_writer.add_scalar('train/loss', loss, step_count)
        summary_writer.add_scalar('train/weighted_loss', weighted_loss, step_count)
        summary_writer.add_scalar('train/policy_loss', policy_loss, step_count)
        summary_writer.add_scalar('train/value_loss', value_loss, step_count)
        summary_writer.add_scalar('train/reward_loss', reward_loss, step_count)
        summary_writer.add_scalar('train/episodes_collected', ray.get(replay_buffer.episodes_collected.remote()),
                                  step_count)
        summary_writer.add_scalar('train/replay_buffer_len', ray.get(replay_buffer.size.remote()), step_count)
        summary_writer.add_scalar('train/lr', lr, step_count)
        summary_writer.add_histogram('train_data_dist/target_reward', target_reward.flatten(), step_count)
        summary_writer.add_histogram('train_data_dist/target_value', target_value.flatten(), step_count)
        summary_writer.add_histogram('train_data_dist/pred_reward', pred_reward.flatten(), step_count)
        summary_writer.add_histogram('train_data_dist/pred_value', pred_value.flatten(), step_count)

        if worker_reward is not None:
            summary_writer.add_scalar('train/worker_reward', worker_reward, step_count)
            summary_writer.add_scalar('train/worker_eps_len', worker_eps_len, step_count)

        if test_score is not None:
            if test_score == best_test_score:
                summary_writer.add_scalar('train/best_test_score', best_test_score, step_count)
            summary_writer.add_scalar('train/test_score', test_score, step_count)


@ray.remote
class SharedStorage(object):
    def __init__(self, model):
        self.step_counter = 0
        self.model = model
        self.reward_log = []
        self.eps_lengths = []

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        return self.model.set_weights(weights)

    def incr_counter(self):
        self.step_counter += 1

    def get_counter(self):
        return self.step_counter

    def set_data_worker_logs(self, eps_len, eps_reward):
        self.eps_lengths.append(eps_len)
        self.reward_log.append(eps_reward)

    def get_data_worker_logs(self):
        if len(self.reward_log) > 0:
            reward = sum(self.reward_log) / len(self.reward_log)
            eps_lengths = sum(self.eps_lengths) / len(self.eps_lengths)

            self.reward_log = []
            self.eps_lengths = []

        else:
            reward = None
            eps_lengths = None

        return reward, eps_lengths


@ray.remote
class DataWorker(object):
    def __init__(self, rank, config, shared_storage, replay_buffer):
        self.rank = rank
        self.config = config
        self.shared_storage = shared_storage
        self.replay_buffer = replay_buffer

    def run(self):
        model = self.config.get_uniform_network()
        with torch.no_grad():
            while ray.get(self.shared_storage.get_counter.remote()) < self.config.training_steps:
                # i = 0
                # while i < 2:
                model.set_weights(ray.get(self.shared_storage.get_weights.remote()))
                env = self.config.new_game(self.config.seed + self.rank)

                obs = env.reset()
                done = False
                priorities = []
                eps_reward, eps_steps = 0, 0
                trained_steps = ray.get(self.shared_storage.get_counter.remote())
                _temperature = self.config.visit_softmax_temperature_fn(num_moves=len(env.history),
                                                                        trained_steps=trained_steps)
                while not done and eps_steps <= self.config.max_moves:
                    root = Node(0)
                    obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                    network_output = model.initial_inference(obs)
                    root.expand(env.to_play(), env.legal_actions(), network_output)
                    root.add_exploration_noise(dirichlet_alpha=self.config.root_dirichlet_alpha,
                                               exploration_fraction=self.config.root_exploration_fraction)
                    MCTS(self.config).run(root, env.action_history(), model)
                    action = select_action(root, temperature=_temperature, deterministic=False)
                    obs, reward, done, info = env.step(action.index)
                    env.store_search_stats(root)

                    eps_reward += reward
                    eps_steps += 1

                    if not self.config.use_max_priority:
                        error = L1Loss(reduction='none')(network_output.value,
                                                         torch.tensor([[root.value()]])).item()
                        priorities.append(error + 1e-5)

                env.close()
                self.replay_buffer.save_game.remote(env,
                                                    priorities=None if self.config.use_max_priority else priorities)
                # Todo: refactor with env attributes to reduce variables
                self.shared_storage.set_data_worker_logs.remote(eps_steps, eps_reward)
                # i += 1


def update_weights(model, target_model, optimizer, replay_buffer, config):
    batch = ray.get(replay_buffer.sample_batch.remote(config.num_unroll_steps, config.td_steps,
                                                      model=target_model if config.use_target_model else None,
                                                      config=config))
    obs_batch, action_batch, target_reward, target_value, target_policy, indices, weights = batch

    obs_batch = obs_batch.to(config.device)
    action_batch = action_batch.to(config.device).unsqueeze(-1)
    target_reward = target_reward.to(config.device)
    target_value = target_value.to(config.device)
    target_policy = target_policy.to(config.device)
    weights = weights.to(config.device)

    value, _, policy_logits, hidden_state = model.initial_inference(obs_batch)
    predicted_values, predicted_rewards = value, None

    value_loss = config.scalar_loss(value.squeeze(-1), target_value[:, 0])
    new_priority = L1Loss(reduction='none')(value.squeeze(-1), target_value[:, 0]).data.cpu().numpy() + 1e-5
    policy_loss = -(torch.log_softmax(policy_logits, dim=1) * target_policy[:, 0]).sum(1)
    reward_loss = torch.zeros(config.batch_size, device=config.device)

    gradient_scale = 1 / config.num_unroll_steps
    for step_i in range(config.num_unroll_steps):
        value, reward, policy_logits, hidden_state = model.recurrent_inference(hidden_state, action_batch[:, step_i])
        # policy_loss += gradient_scale * -(torch.log_softmax(policy_logits, dim=1) * target_policy[:, step_i + 1]).sum(1)
        # value_loss += gradient_scale * config.scalar_loss(value.squeeze(-1), target_value[:, step_i + 1])
        # reward_loss += gradient_scale * config.scalar_loss(reward.squeeze(-1), target_reward[:, step_i])

        policy_loss += -(torch.log_softmax(policy_logits, dim=1) * target_policy[:, step_i + 1]).sum(1)
        value_loss += config.scalar_value_loss(value.squeeze(-1), target_value[:, step_i + 1])
        reward_loss += config.scalar_reward_loss(reward.squeeze(-1), target_reward[:, step_i])

        # collecting for logging
        predicted_values = torch.cat((predicted_values, value))
        if predicted_rewards is not None:
            predicted_rewards = torch.cat((predicted_rewards, reward))
        else:
            predicted_rewards = reward

        hidden_state.register_hook(lambda grad: grad * 0.5)

    # optimize
    loss = (policy_loss + config.value_loss_coeff * value_loss + reward_loss)
    loss.register_hook(lambda grad: grad * gradient_scale)
    weighted_loss = (weights * loss).mean()
    loss = loss.mean()

    optimizer.zero_grad()
    weighted_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
    optimizer.step()

    # update priorities
    replay_buffer.update_priorities.remote(indices, new_priority)

    return weighted_loss.item(), loss.item(), policy_loss.mean().item(), reward_loss.mean().item(), \
           value_loss.mean().item(), target_reward, target_value, predicted_rewards, predicted_values, weights, indices


def adjust_lr(config, optimizer, step_count):
    lr = config.lr_init * config.lr_decay_rate ** (step_count / config.lr_decay_steps)
    lr = max(lr, 0.0005)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def _train(config, shared_storage, replay_buffer, summary_writer):
    model = config.get_uniform_network().to(config.device)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=config.lr_init, momentum=config.momentum,
                          weight_decay=config.weight_decay)
    test_model = config.get_uniform_network().to(config.device)
    target_model = config.get_uniform_network().to('cpu')
    best_test_score = None

    # wait for replay buffer to be non-empty
    while ray.get(replay_buffer.size.remote()) == 0:
        pass

    for step_count in range(config.training_steps):
        shared_storage.incr_counter.remote()
        lr = adjust_lr(config, optimizer, step_count)

        if step_count % config.checkpoint_interval == 0:
            shared_storage.set_weights.remote(model.get_weights())
            target_model.set_weights(model.get_weights())

        loss_data = update_weights(model, target_model, optimizer, replay_buffer, config)

        test_score = None
        if step_count % config.test_interval == 0:
            test_model.set_weights(model.get_weights())
            test_score = test(config, test_model, config.test_episodes, 'cpu', False)
            if best_test_score is None or test_score >= best_test_score:
                best_test_score = test_score
                torch.save(model.state_dict(), config.model_path)

        _log(config, step_count, loss_data, model, replay_buffer, test_score, best_test_score, lr,
             ray.get(shared_storage.get_data_worker_logs.remote()), summary_writer)

        if step_count % 50 == 0:
            replay_buffer.remove_to_fit.remote()

    shared_storage.set_weights.remote(model.get_weights())


def train(config, summary_writer=None):
    storage = SharedStorage.remote(config.get_uniform_network())
    replay_buffer = ReplayBuffer.remote(batch_size=config.batch_size, capacity=config.window_size,
                                        prob_alpha=config.priority_prob_alpha)
    workers = [DataWorker.remote(rank, config, storage, replay_buffer).run.remote()
               for rank in range(0, config.num_actors)]
    _train(config, storage, replay_buffer, summary_writer)
    ray.wait(workers, len(workers))

    return config.get_uniform_network().set_weights(ray.get(storage.get_weights.remote()))
