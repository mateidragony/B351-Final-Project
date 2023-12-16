import torch
import numpy as np
import matplotlib.pyplot as plt

def epsilon_decay(start_eps, end_eps, decay_rate, steps):
    """Returns the decayed epsilon value following an exponential decay model."""
    return max(end_eps, start_eps * (decay_rate ** steps))

def save_model_weights(model, filename):
    """Save the model weights to a file."""
    torch.save(model.state_dict(), filename)

def load_model_weights(model, filename):
    """Load weights from a saved model file into a model."""
    model.load_state_dict(torch.load(filename))

def plot_scores(scores, rolling_window=100):
    """Plot the scores and the rolling average over a specified window."""
    plt.plot(np.arange(len(scores)), scores)
    rolling_mean = np.convolve(scores, np.ones(rolling_window)/rolling_window, mode='valid')
    plt.plot(np.arange(len(rolling_mean)), rolling_mean)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
