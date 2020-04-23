"""
Main script to train DQN/ Double DQN/ Dueling DQN.
"""
import argparse
import torch
import gfootball.env as football_env
from tqdm import trange
from agents import DQN_Agent, DoubleDQN_Agent, DuelingNetwork_Agent
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='DQN, Double DQN, Dueling Networks.')
parser.add_argument('--agent', default='DQN', help='keywords: DQN, DDQN, Duel')
args = parser.parse_args()

env = football_env.create_environment(env_name='academy_empty_goal', #academy_3_vs_1_with_keeper
    representation='simple115',
    number_of_left_players_agent_controls=1,
    stacked=False, logdir='/tmp/football',
    write_goal_dumps=False,
    write_full_episode_dumps=False, render=False)

writer = SummaryWriter('log/')

max_episodes = 100000
max_steps = 10000000

print('State shape: ', env.observation_space.shape) # (115, )
print('Number of actions: ', env.action_space.n)    # 19


agent = DQN_Agent(state_size=115, action_size=env.action_space.n) if args.agent == 'DQN' \
        else DoubleDQN_Agent(state_size=115, action_size=env.action_space.n) if args.agent == 'DDQN' \
        else DuelingNetwork_Agent(state_size=115, action_size=env.action_space.n)

total_steps = 0
# env.render()
for i_episode in trange(max_episodes):
    obs = env.reset()

    steps = 0
    episode_reward = 0
    episode_loss = 0
    episode_loss_count = 0

    done = False

    while not done:
        action = agent.act(obs)
        next_obs, reward, done, info = env.step(action)
        loss = agent.step(obs, action, reward, next_obs, done)
        obs = next_obs

        if loss is not None:
            # writer.add_scalar('Loss', loss, total_steps)
            episode_loss += loss
            episode_loss_count += 1
        
        steps += 1
        total_steps += 1
        episode_reward += reward
    
    if episode_reward >= 1.0:
        torch.save(agent.q_eval.state_dict(), 'checkpoint-' + args.agent + '.pth')
    
    writer.add_scalar(args.agent + '/Reward', episode_reward, i_episode)
    writer.add_scalar(args.agent + '/Step', steps, i_episode)
    if episode_loss_count > 0:
        writer.add_scalar(args.agent + '/Episode Loss', episode_loss / episode_loss_count, i_episode)

    if total_steps >= max_steps:
        break

env.close()
