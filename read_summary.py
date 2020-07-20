"""
Read TFEvent files and plot smoothed curves.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='valid')
    return y_smooth

def plot_tensorboard_logs(path):
    tf_size_guidance = {
        'scalars': 0
    }

    events = EventAccumulator(path, tf_size_guidance)
    events.Reload()

    # print(events.Tags())

    losses = events.Scalars('Duel/Episode_Loss')
    rewards = events.Scalars('Duel/Reward')
    steps = events.Scalars('Duel/Step')


    df = pd.DataFrame(losses)


    y = smooth(df['value'], 1000)
    x = np.arange(len(y))

    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title('Dueling')
    plt.plot(x, y)
    plt.show()

dqn_log_file = './cloud_log/events.out.tfevents.1587191743.torch-2'
ddqn_log_file = './cloud_log/events.out.tfevents.1587192005.torch-2'
duel_log_file = './cloud_log/events.out.tfevents.1587191911.torch-gpu'

plot_tensorboard_logs(duel_log_file)