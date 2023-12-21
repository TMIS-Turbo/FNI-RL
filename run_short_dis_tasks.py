import pandas as pd
import random as rn
import gym
import matplotlib.pyplot as plt
import argparse
import Environment.environment
from fnirl import *
import os

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', default="TrafficEnv1-v0", help='name of the environment to run')
parser.add_argument('--seed', default=0)
parser.add_argument('--train_step', default=2000)
parser.add_argument('--T_horizon', default=30)
parser.add_argument('--print_interval', default=10)
parser.add_argument('--speed_range', default=15.0, help='Maximum speed')
parser.add_argument('--max_a', default=7.6, help='Maximum Acceleration')
parser.add_argument('--state_dim', default=26)
parser.add_argument('--action_dim', default=1)
args = parser.parse_args()

# Set environment
env = gym.make(args.env_name)

# Set a random seed
env.seed(args.seed)
np.random.seed(args.seed)
rn.seed(args.seed)
torch.manual_seed(args.seed)

model_dir_base = os.getcwd() + '/models/' + args.env_name
train_result_dir = os.getcwd() + '/results/' + args.env_name

if not os.path.exists(model_dir_base):
    os.makedirs(model_dir_base)

if not os.path.exists(train_result_dir):
    os.makedirs(train_result_dir)


def train():
    env.start(gui=False)
    alg_cfg = FNIRL.Config()
    alg = FNIRL(alg_cfg, env, state_dim=args.state_dim, action_dim=args.action_dim)

    score = 0.0
    total_reward = []
    episode = []

    v = []
    v_epi = []
    v_epi_mean = []

    cn = 0.0
    cn_epi = []

    sn = 0.0
    sn_epi = []

    for n_epi in range(args.train_step):
        score, v, v_epi, xa, ya, done = alg.interaction(args.T_horizon, score, v, v_epi, args.speed_range, args.max_a, n_epi)

        if done is True:
            cn += 1

        if args.env_name == 'TrafficEnv1-v0' or args.env_name == 'TrafficEnv3-v0':
            if xa < -50.0 and ya > 4.0 and done is False:
                sn += 1
        elif args.env_name == 'TrafficEnv2-v0':
            if xa > 50.0 and ya > -5.0 and done is False:
                sn += 1
        elif args.env_name == 'TrafficEnv4-v0':
            if ya < -50.0 and done is False:
                sn += 1

        if (n_epi+1) % args.print_interval == 0:
            print("# of episode :{}, avg score_v : {:.1f}".format(n_epi+1, score / args.print_interval))

            episode.append(n_epi+1)
            total_reward.append(score / args.print_interval)
            cn_epi.append(cn/args.print_interval)
            sn_epi.append(sn/args.print_interval)
            print("######cn & sn rate:", cn/args.print_interval, sn/args.print_interval)

            v_mean = np.mean(v_epi)
            v_epi_mean.append(v_mean)

            v_epi = []
            score = 0.0
            cn = 0.0
            sn = 0.0

    df = pd.DataFrame([])
    df["n_epi"] = episode
    df["total_reward"] = total_reward
    df["v_epi_mean"] = v_epi_mean
    df["cn_epi"] = cn_epi
    df["sn_epi"] = sn_epi

    df.to_csv(train_result_dir + '/train_data.csv', index=0)

    plt.plot(episode, total_reward)
    plt.xlabel('episode')
    plt.ylabel('total_reward')
    plt.show()

    env.close()


if __name__ == "__main__":
    train()