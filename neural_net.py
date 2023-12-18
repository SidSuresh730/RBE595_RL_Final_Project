import torch
from torch import nn, optim
import numpy as np
import random
from custom_functions import *
import datetime
import os


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
        self.flatten = nn.Flatten(start_dim=0)
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
        if os.path.exists(PT_PATH):
            print("Loading NN from file: ",PT_NAME)
            self.load_state_dict(torch.load(PT_PATH,map_location=torch.device(device)))
        
        

    # Forward method of the neural network, defining the forward pass through the layers.
    # DO NOT CALL DIRECTLY, automatically called when passing data into model.
    def forward(self, input):
        img1,img2,img3,x,y,z = input
        img1 = self.flatten(torch.tensor(img1,dtype=torch.float32))
        img2 = self.flatten(torch.tensor(img2,dtype=torch.float32))
        img3 = self.flatten(torch.tensor(img3,dtype=torch.float32))
        x = torch.tensor([x],dtype=torch.float32)#self.flatten(x)
        y = torch.tensor([y],dtype=torch.float32)#self.flatten(y)
        z = torch.tensor([z],dtype=torch.float32)#self.flatten(z)
        img1_out = self.img_stack_12(img1)
        # print("Img1: Good")
        img2_out = self.img_stack_12(img2)
        # print("Img2: Good")
        img3_out = self.img_stack_3(img3)
        # print("Img3: Good")
        img_in = torch.cat([img1_out,img2_out,img3_out])
        # print("Concat: Good")
        x_out = self.pos_stack_x(x)
        # print("posx: Good")
        y_out = self.pos_stack_yz(y)
        # print("posy: Good")
        z_out = self.pos_stack_yz(z)
        # print("posz: Good")
        pos_in = torch.cat([x_out,y_out,z_out])
        # print("Concat 2: Good")
        img_out = self.combine_img_stack(img_in)
        # print("Total img: Good")
        pos_out = self.combine_pos_stack(pos_in)
        # print("Total pos: Good")
        logits = self.combine_stack(torch.cat([img_out,pos_out]))
        # print("Logits: Good")
        return logits

# ----------------------------------------------------------------------------------------------------------------
# Please note that this is purely using torch. Above, I attempted to do the adam as a function but then realized
# that torch have a built in attr we could use: https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
# I am also using: https://pytorch.org/docs/stable/generated/torch.nn.functional.relu.html for the nn.

def episodic_deep_q_learning(episodes, min_interaction_limit, episode_start=0, gamma=GAMMA, learning_rate=LEARNING_RATE,client=None):
    file_name = "rewards_"+str(time.time())+".csv"
    # This initializes the Q-network
    q_network = DQNetwork().to(device=device)

    # Sets up the Adam optimizer for updating the Q-network parameters 
    # and uses the Huber loss as the loss function
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
    criterion = nn.SmoothL1Loss()  # Huber loss

    # initializes an empty list to store the experiences for experience replay, 
    # and a variable to keep track of the total interactions.
    experience_replay = []
    total_interactions = 0
    epsilon = 1
    # Algorithm 1 pseudo code from paper. 
    # Starts the main loop over episodes
    for i in range(episode_start,episodes):
        print("Episode: ",i)
        rewards = []
        if i<episodes//2:
            epsilon=1-.0009*i
        else:
            epsilon = 0.1
        ep = Episode(client=client,n=np.random.choice([1,2,3]))
        # state = ep.state.get_inputs()
        # loop iterates over time steps within each episode, 
        # selects actions using the epsilon-greedy policy, 
        # and obtains the next state and reward from the environment.
        for t in range(min_interaction_limit):
            state = ep.state.get_inputs()
            try:
                with torch.no_grad():
                    q_values = q_network(state)
                q_values = q_values.detach().numpy()
                action = epsilon_greedy(q_values, epsilon) # from above function
                next_state, reward, done = ep.step(a=action)
                
                # Appends the current state, action, next state, and reward to the experience replay buffer.
                experience_replay.append((state, action, next_state.get_inputs(), reward))
                rewards.append(reward)
                # print(state)
                # This breaks the inner loop if the reward is negative or the episode is done
                if reward < 0:
                    break
                if done:
                    print("goal reached!")
                    break
                # Updates the current state for the next time step.
                ep.state = next_state
            except:
                t-=1
                print("Try again")
                continue
            
        write_to_csv(filename=file_name,episode=i,data=rewards)
        # 3 Update interaction count
        total_interactions += t
        # Update the network parameters
        if total_interactions >= min_interaction_limit: #update_frequency: 
            experience_replay = np.array(experience_replay,dtype=object)
            rand_perms = torch.randperm(len(experience_replay))
            for i in range(0,len(experience_replay),MINIBATCH_SIZE):
                # This selects a random minibatch from the experience replay buffer.
                try:
                    indices = rand_perms[i:i+MINIBATCH_SIZE]
                    minibatch = experience_replay[indices]
                   
                    # This unpacks the minibatch into separate lists for states, actions, next states, and rewards.
                    states, actions, next_states, rewards = zip(*minibatch)

                    # Converts the lists into PyTorch tensors.
                    # states = torch.tenso(np.array(states), dtype=torch.float32)
                    actions = torch.tensor(np.array(actions), dtype=torch.long).view(-1, 1)
                    # next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
                    rewards = torch.tensor(np.array(rewards), dtype=torch.float32).view(-1, 1)
                    
                    with torch.no_grad():
                        # Calculates the target Q-values using the target Q-network.
                        target_q_values = torch.stack([q_network(next_state) for next_state in next_states])

                        target_max_q_values, _ = torch.max(target_q_values, dim=1, keepdim=False)
            
                        targets = torch.tensor([rewards[i] + gamma * target_max_q_values[i] for i in range(len(rewards))])
                        targ_resize = (len(targets),1)
                        targets = targets.view(targ_resize)
                    # Calculates the Q-values for the selected actions in the current Q-network.
                    q_values = torch.stack([q_network(state) for state in states])
                    selected_q_values = torch.gather(q_values, 1, actions)
                    
                    # Hubber loss - between the predicted Q-values and the target Q-values.
                    loss = criterion(selected_q_values, targets)
                    print("Backpropogation...")
                    # Backpropagation and optimization steps to update the Q-network parameters.
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_interactions=0
                except:
                    print("Skipping bad batch")
                    continue
            torch.save(q_network.state_dict(),PT_NAME)
            experience_replay = []  # Clear experience replay buffer
        ep.reset()
        # write_to_csv(filename=file_name,episode=i,data=rewards)
    return q_network

def run_policy(client:airsim.MultirotorClient,episode_number:int):
    done = False
    crashed = False
    q_network = DQNetwork().to(device=device)
    ep = Episode(client=client,n=episode_number)
    state = ep.state.get_inputs()
    while not(done or crashed):
        state = ep.state.get_inputs()
        with torch.no_grad():
            q_val = q_network(state)
        q_val = q_val.detach().numpy()
        skip = 0
        action = np.argmax(q_val[skip:])+skip #epsilon_greedy(q_val,0.1)#
        print(action)
        next_state, reward, done = ep.step(a=action)
        # if reward<0:
        #     print("Collision")
        #     return
        if done:
            print("Goal Reached")
            return
        ep.state = next_state
    ep.reset()


# Call airsim below

# Items that still be done:
# 2. episodes? how many?
# 3. epsilon? is this arbitrary? may have overlooked on the paper if is there.
# 4. 
