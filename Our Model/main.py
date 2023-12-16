import torch
import numpy as np
from agent import Agent
from environment import UnityEnvWrapper
from collections import deque

""" 
- Included params for epsilon decay schedule for epsilon-greedy-policy
- Tracking scores of each episode + tracking average score over last 100 episodes
- Prints average every 100 episodes for monitor / logging
- Saving the model weights whenever score improves, to resume training or eval agent from said weights at a later time (in the even where we're unable to train entirely in one go)
- If env is solved, we save the model and stop trianing
TODO: need to adapt the `max_t` + `if np.mean(scores_window) >= 200.0` condition to our solving criteria for env
"""

def train(n_episodes=1000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """
    Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    env = UnityEnvWrapper(env_name='our_env_name')
    agent = Agent(state_size=env.state_size, action_size=env.action_size, buffer_size=int(1e5), batch_size=64,
                  gamma=0.99, lr=5e-4, update_every=4, device="cuda" if torch.cuda.is_available() else "cpu")
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if np.any(done):  # assumes `done` is a boolean array
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), f'checkpoint_episode_{i_episode}.pth')
        if np.mean(scores_window)>=200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_solved.pth')
            break
    return scores

if __name__ == '__main__':
    scores = train()
