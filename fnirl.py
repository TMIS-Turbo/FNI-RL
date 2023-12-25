import numpy as np
import torch

from config import BaseConfig, Configurable
from world_model import BatchedGaussianEnsemble
from sampling import SampleBuffer
from fcac import FAC
from torch_util import Module, DummyModuleWrapper, device, random_choice


class FNIRL(Configurable, Module):
    class Config(BaseConfig):
        fac_cfg = FAC.Config()
        model_cfg = BatchedGaussianEnsemble.Config()
        model_steps = 500
        horizon = 5
        safe_horizon = 100
        buffer_min = 5
        buffer_max = 10**6
        rollout_batch_size = 64
        solver_updates_per_step = 8
        real_fraction = 0.1

    def __init__(self, config, env, state_dim, action_dim):
        Configurable.__init__(self, config)
        Module.__init__(self)

        self.real_env = env
        self.state_dim, self.action_dim = state_dim, action_dim

        self.solver = FAC(self.fac_cfg, self.state_dim, self.action_dim)
        self.model_ensemble = BatchedGaussianEnsemble(self.model_cfg, self.state_dim, self.action_dim)

        self.replay_buffer = self._create_buffer(self.buffer_max)
        self.virt_buffer = self._create_buffer(self.buffer_max)

        self.ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1))
        self.safe_interaction = 0
        self.safe_horizon = 10
        self.env_model_error = 0.0

    @property
    def actor(self):
        return self.solver.actor

    @property
    def adversary(self):
        return self.solver.adversary

    def _create_buffer(self, capacity):
        buffer = SafetySampleBuffer(self.state_dim, self.action_dim, capacity)
        buffer.to(device)
        return DummyModuleWrapper(buffer)

    def interaction(self, max_episode_steps, score, v, v_epi, speed_range, max_a, n_epi, warm_up=10, scale=10.0):
        episode = self._create_buffer(max_episode_steps)
        state = self.real_env.reset()

        for t in range(max_episode_steps):
            if n_epi > warm_up:
                if t == 0:
                    self.update_world_model(self.model_steps)
                self.rollout_and_update()

            mu, _, pi = self.actor(torch.tensor(state))
            action = max_a*pi.item()

            r, next_s, done, r_, c_, info = self.real_env.step(action)
            for buffer in [episode, self.replay_buffer]:
                buffer.append(states=torch.tensor(state), actions=pi.detach(), next_states=torch.tensor(next_s), rewards=r/scale, dones=done, violations=done)

            state = next_s
            score += r
            v.append(state[24]*speed_range)
            v_epi.append(state[24]*speed_range)
            xa = info[0]
            ya = info[1]

            if done is False:
                self.safe_interaction += 1
            else:
                break

        self.safe_horizon = self.safe_interaction
        self.safe_interaction = 0

        if (n_epi+1) % 100 == 0:
            self.solver.save_model(int(score), self.real_env.spec.id)
            self.save_model(self.env_model_error)
            print("#The models are saved!#", n_epi+1)
            print("======>env_model_error:", self.env_model_error)

        return score, v, v_epi, xa, ya, done

    def interaction_long_dis_task(self, max_episode_steps, score, v, v_epi, speed_range, n_epi, warm_up=10, scale=10.0, max_sh=10):
        episode = self._create_buffer(max_episode_steps)
        state = self.real_env.reset()
        infraction_check_epi = False

        for t in range(max_episode_steps):
            if n_epi > warm_up:
                if t == 0:
                    self.update_world_model(self.model_steps)
                self.rollout_and_update()

            mu, _, pi = self.actor(torch.tensor(state))

            r, next_s, collision, cost, infraction_check, infraction, navigation_check, done, info = self.real_env.step(pi)
            for buffer in [episode, self.replay_buffer]:
                buffer.append(states=torch.tensor(state), actions=pi.detach(), next_states=torch.tensor(next_s), rewards=r/scale, dones=done, violations=(collision or infraction_check))

            state = next_s
            score += r
            v.append(state[24]*speed_range)
            v_epi.append(state[24]*speed_range)

            if infraction_check is True:
                infraction_check_epi = True

            if done is False:
                self.safe_interaction += 1
            else:
                break

        self.safe_horizon = min(self.safe_interaction, max_sh)
        self.safe_interaction = 0

        if (n_epi+1) % 100 == 0:
            self.solver.save_model(int(score), self.real_env.spec.id)
            self.save_model(self.env_model_error)
            print("#The models are saved!#", n_epi+1)
            print("======>env_model_error:", self.env_model_error)

        return score, v, v_epi, done, infraction_check_epi, navigation_check, collision

    def update_world_model(self, model_steps):
        losses = self.model_ensemble.fit(self.replay_buffer, steps=model_steps)
        self.env_model_error = np.mean(losses)

        buffer_rewards = self.replay_buffer.get('rewards')
        r_min = buffer_rewards.min().item()
        r_max = buffer_rewards.max().item()
        self.solver.update_r_bounds(r_min, r_max, self.safe_horizon)

    def rollout(self, policy, adv_policy, initial_states=None, k=0.9, c=0.1):
        if initial_states is None:
            initial_states = random_choice(self.replay_buffer.get('states'), size=self.rollout_batch_size)
        buffer = self._create_buffer(self.rollout_batch_size * self.horizon)
        states = initial_states

        for t in range(self.horizon):
            with torch.no_grad():
                _, _, actions = policy(states)
                _, _, adv_actions = adv_policy(states)
                mix_actions = k * actions + (1-k) * adv_actions
                next_states, costs, uncertainty = self.model_ensemble.sample(states, mix_actions)

            violations = costs > c
            dones = violations
            buffer.extend(states=states, actions=mix_actions, next_states=next_states,
                          rewards=0.0, dones=dones, violations=violations)
            continues = ~(dones | violations)
            if continues.sum() == 0:
                break
            states = next_states[continues]

        self.virt_buffer.extend(**buffer.get(as_dict=True))
        return (k*torch.mean(violations.float()) + (1-k)*torch.clamp(uncertainty, 0.0, 1.0)).requires_grad_(True)

    def update_solver(self, fear):
        solver = self.solver
        n_real = int(self.real_fraction * solver.batch_size)
        real_samples = self.replay_buffer.sample(n_real)
        virt_samples = self.virt_buffer.sample(solver.batch_size - n_real)
        combined_samples = [
            torch.cat([real, virt]) for real, virt in zip(real_samples, virt_samples)
        ]

        solver.update(self.replay_buffer, fear)
        solver.update_actor(combined_samples[0])

    def rollout_and_update(self):
        fear = self.rollout(self.actor, self.adversary)

        for _ in range(self.solver_updates_per_step):
            self.update_solver(fear)

    def process_inputs(self, s, s_mean, s_std, s_epsilon):
        return (torch.tensor(s) - s_mean) / (s_std + s_epsilon)

    def save_model(self, model_name):
        name = "./models/" + self.real_env.spec.id + "/env_model%d" % model_name
        torch.save([self.model_ensemble.state_dict(), self.model_ensemble.state_normalizer.mean, \
                    self.model_ensemble.state_normalizer.std, self.model_ensemble.state_normalizer.epsilon], "{}.pkl".format(name))


class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x


class SafetySampleBuffer(SampleBuffer):
    COMPONENT_NAMES = (*SampleBuffer.COMPONENT_NAMES, 'violations')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._create_buffer('violations', torch.bool, [])
