import torch
from torch import nn, optim
import numpy as np
import random
from custom_functions import *



# data = [[1,2],[3,4]]
# data = [1,2,3,4]
# np_arr = np.array(data)
# x_np = torch.from_numpy(np_arr)
# t1 = torch.cat([x_np,x_np,x_np])
# print(t1)
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Neural network to approximate Q function
class DQNetwork(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.flatten = nn.Flatten()
        # stack for processing image data for images 1 and 2
        self.img_stack_12 = nn.Sequential(
            nn.Linear(IMG_NN_INPUT,IMG_NN_H1),
            nn.Linear(IMG_NN_H1,IMG_NN_H2),
            nn.Linear(IMG_NN_H2,IMG_NN_H3)
        )
        # stack for processing image data for image 3
        self.img_stack_3 = nn.Sequential(
            nn.Linear(IMG_NN_INPUT,IMG_NN_H1),
            nn.Linear(IMG_NN_H1,IMG_NN_H2),
            nn.Linear(IMG_NN_H2,IMG3_NN_H3)
        )
        # stack for processing position data for y and z components of position
        self.pos_stack_yz = nn.Sequential(
            nn.Linear(POS_NN_INPUT,POS_NN_H1),
            nn.Linear(POS_NN_H1,POS_NN_H2),
        )
        # stack for processing position data for x components of position
        self.pos_stack_x = nn.Sequential(
            nn.Linear(POS_NN_INPUT,POS_NN_H1),
            nn.Linear(POS_NN_H1,POSX_NN_H2),
        )
        # stack for processing combined raw image data
        self.combine_img_stack = nn.Linear(2*IMG_NN_H3+IMG3_NN_H3,IMG_NN_H4)
        # stack for processing combined setpoint components
        self.combine_pos_stack = nn.Linear(2*POS_NN_H2+POSX_NN_H2,POS_NN_H3)
        # stack for processing image and setpoint data combined
        self.combine_stack = nn.Sequential(
            nn.Linear(IMG_NN_H4+POS_NN_H3,NN_H5),
            nn.Linear(NN_H5,NN_OUTPUT)
        )
        

    # Forward method of the neural network, defining the forward pass through the layers.
    # DO NOT CALL DIRECTLY, automatically called when passing data into model.
    def forward(self, input):
        img1,img2,img3,x,y,z = input
        img1 = self.flatten(img1)
        img2 = self.flatten(img2)
        img3 = self.flatten(img3)
        x = self.flatten(x)
        y = self.flatten(y)
        z = self.flatten(z)
        img1_out = self.img_stack_12(img1)
        img2_out = self.img_stack_12(img2)
        img3_out = self.img_stack_3(img3)
        img_in = torch.cat([img1_out,img2_out,img3_out])
        x_out = self.pos_stack_x(x)
        y_out = self.pos_stack_yz(y)
        z_out = self.pos_stack_yz(z)
        pos_in = torch.cat([x_out,y_out,z_out])
        img_out = self.combine_img_stack(img_in)
        pos_out = self.combine_pos_stack(pos_in)
        logits = self.combine_stack(torch.cat([img_out,pos_out]))
        return logits

# ----------------------------------------------------------------------------------------------------------------
# Please note that this is purely using torch. Above, I attempted to do the adam as a function but then realized
# that torch have a built in attr we could use: https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
# I am also using: https://pytorch.org/docs/stable/generated/torch.nn.functional.relu.html for the nn.

def episodic_deep_q_learning(episodes, min_interaction_limit, update_frequency, gamma, learning_rate, input_size, output_size, env):
    
    # This initializes the main Q-network and a target Q-network. 
    # The target Q-network is used for stability in training, and its parameters are 
    # initially set to be the same as the main Q-network.
    q_network = DQNetwork(input_size, output_size)
    # target_q_network = DQNetwork(input_size, output_size)
    # target_q_network.load_state_dict(q_network.state_dict())
    # target_q_network.eval()

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
# 2. episodes? how many?
# 3. epsilon? is this arbitrary? may have overlooked on the paper if is there.
# 4. 
