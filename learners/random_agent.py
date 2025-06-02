import numpy as np
import torch
class RandomAgent:
    def __init__(self, num_agents):
        self.num_agents = num_agents

    def act(self, available_actions):
        actual_actions = []
        agents = []
        for i in range(self.num_agents):
            avals = available_actions[i]
            for batch in avals:
                rem = batch[batch>0]
                rem = np.array([acts for acts in range(rem.shape[0]) ])
                r_acts = np.random.choice(rem)
                actual_actions.append(r_acts)
            agents.append(np.array(actual_actions))
            actual_actions = []

        return np.array(agents).swapaxes(0,1)









