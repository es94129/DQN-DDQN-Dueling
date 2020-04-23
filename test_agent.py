import torch
import gfootball.env as football_env

from model import QNetwork, DuelingNetwork

# PATH = 'checkpoint-DQN.pth'
# PATH = 'checkpoint-DDQN.pth'
PATH = 'checkpoint-Duel.pth'

env = football_env.create_environment(env_name='academy_empty_goal', #academy_3_vs_1_with_keeper
    representation='simple115',
    number_of_left_players_agent_controls=1,
    stacked=False, logdir='/tmp/football',
    write_goal_dumps=False,
    write_full_episode_dumps=False, render=False)

# q_network = QNetwork(state_size=115, action_size=env.action_space.n, seed=42)
q_network = DuelingNetwork(state_size=115, action_size=env.action_space.n, seed=42)

q_network.load_state_dict(torch.load(PATH))

# render the football game
env.render()

obs = env.reset()
done = False
episode_reward = 0
steps = 0

while not done:
    with torch.no_grad():
        state = torch.from_numpy(obs).float().unsqueeze(0)
        q_values = q_network(state)
    
    action = torch.argmax(q_values, dim=1).item()

    next_obs, reward, done, info = env.step(action)
    obs = next_obs
    
    steps += 1
    episode_reward += reward

env.close()

print('steps:', steps)
print('reward:', episode_reward)