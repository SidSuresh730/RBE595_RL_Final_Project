import torch
from torch import nn, optim
import numpy as np
import random
from custom_functions import *



data = [[1,2],[3,4]]
np_arr = np.array(data)
x_np = torch.from_numpy(np_arr)


class NeuralNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.flatten = nn.Flatten()
        # stack for processing image data
        self.img_stack = nn.Sequential(
            nn.Linear(N_ROWS*N_COLS,128),
            nn.Linear(128,16),
            nn.Linear(16,8)
            # nn.Linear(8,16),
            # nn.Linear(16,32),
            # nn.Linear(32,18)
        )
        self.pos_stack = 0

    # Forward method of the neural network, defining the forward pass through the layers.
    # DO NOT CALL DIRECTLY, automatically called when passing data into model.
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits

# ----------------------------------------------------------------------------------------------------------------
# Please note that this is purely using torch. Above, I attempted to do the adam as a function but then realized
# that torch have a built in attr we could use: https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
# I am also using: https://pytorch.org/docs/stable/generated/torch.nn.functional.relu.html for the nn.

class QNetwork(nn.Module):
    # This defines a neural network class (QNetwork) using PyTorch's neural network module (nn.Module). 
    # It has two fully connected layers (fc and fc2) with ReLU activation.
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, output_size)

    

def episodic_deep_q_learning(episodes, min_interaction_limit, update_frequency, gamma, learning_rate, input_size, output_size, env):
    
    # This initializes the main Q-network and a target Q-network. 
    # The target Q-network is used for stability in training, and its parameters are 
    # initially set to be the same as the main Q-network.
    q_network = QNetwork(input_size, output_size)
    target_q_network = QNetwork(input_size, output_size)
    target_q_network.load_state_dict(q_network.state_dict())
    target_q_network.eval()

    # Sets up the Adam optimizer for updating the Q-network parameters 
    # and uses the Huber loss as the loss function
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
    criterion = nn.SmoothL1Loss()  # Huber loss

    # initializes an empty list to store the experiences for experience replay, 
    # and a variable to keep track of the total interactions.
    experience_replay = []
    total_interactions = 0

    # Algorithm 1 pseudo code from paper. 
    # Starts the main loop over episodes
    for episode in range(episodes):
        state = env.reset() # resets the environment for each new episode.

        # loop iterates over time steps within each episode, 
        # selects actions using the epsilon-greedy policy, 
        # and obtains the next state and reward from the environment.
        for t in range(min_interaction_limit):
            q_values = q_network(torch.tensor(state, dtype=torch.float32))
            action = epsilon_greedy(q_values, epsilon=0.1) # from above function
            next_state, reward, done, _ = env.step(action.item())

            # Appends the current state, action, next state, and reward to the experience replay buffer.
            experience_replay.append((state, action, next_state, reward))

            # This breaks the inner loop if the reward is negative or the episode is done
            if reward < 0 or done:
                break

            # Updates the current state for the next time step.
            state = next_state

        # 3 Update interaction count
        total_interactions += min_interaction_limit

        # Update the network parameters
        if total_interactions >= update_frequency:
            for _ in range(update_frequency):
                # This selects a random minibatch from the experience replay buffer.
                minibatch = random.sample(experience_replay, k=min(min_interaction_limit, len(experience_replay)))

                # This unpacks the minibatch into separate lists for states, actions, next states, and rewards.
                states, actions, next_states, rewards = zip(*minibatch)

                # Converts the lists into PyTorch tensors.
                states = torch.tensor(states, dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.long).view(-1, 1)
                next_states = torch.tensor(next_states, dtype=torch.float32)
                rewards = torch.tensor(rewards, dtype=torch.float32).view(-1, 1)

                with torch.no_grad():
                    # Calculates the target Q-values using the target Q-network.
                    target_q_values = target_q_network(next_states)
                    target_max_q_values, _ = torch.max(target_q_values, dim=1, keepdim=True)
                    targets = rewards + gamma * target_max_q_values

                # Calculates the Q-values for the selected actions in the current Q-network.
                q_values = q_network(states)
                selected_q_values = torch.gather(q_values, 1, actions)
                
                # Hubber loss - between the predicted Q-values and the target Q-values.
                loss = criterion(selected_q_values, targets)

                # Backpropagation and optimization steps to update the Q-network parameters.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Update target Q-network
            target_q_network.load_state_dict(q_network.state_dict())

            experience_replay = []  # Clear experience replay buffer

    return q_network



# Call airsim below

# Items that still be done:
# 1. NN input and output size? are these 32x32? how to define based on our enviroment?
# 2. episodes? how many?
# 3. epsilon? is this arbitrary? may have overlooked on the paper if is there.
# 4. 
