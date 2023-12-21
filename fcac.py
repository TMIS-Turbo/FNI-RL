import copy
import random
import torch
from torch import nn
from config import BaseConfig, Configurable
from policy import Actor
from torch_util import device, Module, mlp, update_ema, freeze_module


def pythonic_mean(x):
    return sum(x) / len(x)


class CriticEnsemble(Configurable, Module):
    class Config(BaseConfig):
        n_critics = 2
        hidden_layers = 2
        hidden_dim = 256
        learning_rate = 3e-4

    def __init__(self, config, state_dim, action_dim):
        Configurable.__init__(self, config)
        Module.__init__(self)
        dims = [state_dim + action_dim, *([self.hidden_dim] * self.hidden_layers), 1]
        self.qs = torch.nn.ModuleList([
            mlp(dims, squeeze_output=True) for _ in range(self.n_critics)
        ])
        self.optimizer = torch.optim.Adam(self.qs.parameters(), lr=self.learning_rate)

    def all(self, state, action):
        sa = torch.cat([state, action], 1)
        return [q(sa) for q in self.qs]

    def min(self, state, action):
        return torch.min(*self.all(state, action))

    def mean(self, state, action):
        return pythonic_mean(self.all(state, action))

    def random_choice(self, state, action):
        sa = torch.cat([state, action], 1)
        return random.choice(self.qs)(sa)


class FAC(Module):
    class Config(BaseConfig):
        discount = 0.99
        deterministic_backup = False
        critic_update_multiplier = 1
        actor_lr = 3e-4
        critic_cfg = CriticEnsemble.Config()
        adversary_lr = 1e-4
        tau = 0.005
        batch_size = 64
        hidden_dim = 256
        hidden_layers = 2
        update_violation_cost = True

    def __init__(self, config, state_dim, action_dim, optimizer_factory=torch.optim.Adam):
        Configurable.__init__(self, config)
        Module.__init__(self)
        self.violation_cost = 0.0

        self.actor = Actor(state_dim, action_dim, min_log_std=-10.0, max_log_std=5.0)
        self.critic = CriticEnsemble(self.critic_cfg, state_dim, action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        freeze_module(self.critic_target)
        self.adversary = Actor(state_dim, action_dim, min_log_std=-5.0, max_log_std=10.0)

        self.actor_optimizer = optimizer_factory(self.actor.parameters(), lr=self.actor_lr)
        self.adversary_optimizer = optimizer_factory(self.adversary.parameters(), lr=self.adversary_lr)

        self.beta = torch.zeros(1, requires_grad=True, device=device)
        self.log_beta = torch.zeros(1, requires_grad=True, device=device)
        self.beta_optimizer = optimizer_factory([self.log_beta], lr=self.adversary_lr)

        self.fear = torch.zeros(1, requires_grad=True, device=device)
        self.target_fear = 0.5

        self.criterion = nn.MSELoss()
        self.register_buffer('total_updates', torch.zeros([]))

    @property
    def violation_value(self):
        return -self.violation_cost / (1. - self.discount)

    def update_r_bounds(self, r_min, r_max, horizon):
        if self.update_violation_cost:
            self.violation_cost = (r_max - r_min) / self.discount**horizon - r_max + (1-self.discount)*self.beta.detach()/self.discount**horizon

    def critic_loss(self, obs, action, next_obs, reward, done, violation):
        target = self.compute_target(next_obs, reward, done, violation)
        return self.critic_loss_given_target(obs, action, target)

    def compute_target(self, next_obs, reward, done, violation):
        with torch.no_grad():
            _, _, next_action = self.actor(next_obs)
            next_value = self.critic_target.min(next_obs, next_action)
            if not self.deterministic_backup:
                next_value = next_value - self.beta.detach() * self.fear
            q = reward + self.discount * (1. - done.float()) * next_value
            q[violation] = self.violation_value
            return q

    def critic_loss_given_target(self, obs, action, target):
        qs = self.critic.all(obs, action)
        return pythonic_mean([self.criterion(q, target) for q in qs])

    def update_critic(self, *critic_loss_args):
        critic_loss = self.critic_loss(*critic_loss_args)
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()
        update_ema(self.critic_target, self.critic, self.tau)
        return critic_loss.detach()

    def actor_loss(self, obs):
        _, _, action = self.actor(obs)
        actor_Q = self.critic.random_choice(obs, action)
        actor_loss = torch.mean(self.beta * self.fear - actor_Q)

        return [actor_loss]

    def update_actor(self, obs):
        losses = self.actor_loss(obs)
        optimizers = [self.actor_optimizer]

        assert len(losses) == len(optimizers)
        for loss, optimizer in zip(losses, optimizers):
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

    def update_beta(self):
        loss = (self.log_beta * (self.target_fear - self.fear).detach()).mean()

        self.beta_optimizer.zero_grad()
        loss.backward()
        self.beta_optimizer.step()

        self.beta = self.log_beta.exp()

    def update_adv(self):
        loss = - self.fear
        self.adversary_optimizer.zero_grad()
        loss.backward()
        self.adversary_optimizer.step()

    def update(self, replay_buffer, fear):
        assert self.critic_update_multiplier >= 1
        self.fear = fear

        for _ in range(self.critic_update_multiplier):
            samples = replay_buffer.sample(self.batch_size)
            self.update_critic(*samples)
        self.update_actor(samples[0])
        self.update_beta()
        self.update_adv()

        self.total_updates += 1

    def save_model(self, model_name, env_name):
        name = "./models/" + env_name + "/policy_v%d" % model_name
        torch.save(self.actor, "{}.pkl".format(name))