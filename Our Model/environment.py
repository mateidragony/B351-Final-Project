from mlagents_envs.environment import UnityEnvironment
import numpy as np

## Assuming we work with a single agent - needs update if we plan on doing multi-agent batch training

class UnityEnvWrapper:
    
    # `no_graphics` in constructor to init the Unity environment w/ out rendering env for faster training
    def __init__(self, env_name, no_graphics=True):
        self.env = UnityEnvironment(file_name=env_name, no_graphics=no_graphics)
        self.brain_name = self.env.get_agent_groups()[0]
        self.env_info = self.env.reset()[self.brain_name]

        # Get the number of agents in the environment
        self.num_agents = len(self.env_info.agent_id)

        # Get the action size
        self.action_size = self.env.brains[self.brain_name].vector_action_space_size

        # Get the state size
        self.state_size = self.env_info.obs[0].shape[1]

    def reset(self):
        self.env_info = self.env.reset()[self.brain_name]
        return self.get_states()

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = action.tolist()
        elif isinstance(action, int):
            action = [action]
            
        self.env_info = self.env.step({self.brain_name: action})[self.brain_name]
        next_state = self.get_states()
        reward = self.env_info.reward
        done = self.env_info.done
        return next_state, reward, done

    # Reshaping observation array if necessary to handle multiple agents
    def get_states(self):
        return np.array(self.env_info.obs).reshape(self.num_agents, -1)
    
    # Terminates Unity environment when completed
    def close(self):
        self.env.close()
