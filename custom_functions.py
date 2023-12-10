import airsim
from Bezier import Bezier
import numpy as np
from math import tanh

# Learning Constants
D_MAX = 5.0
R_L = 0.0
R_U = 1.0
DEL_D_L = 0.0
DEL_D_U = 1.0
R_DP = -0.5
R_CP = -1.0
N = 1.0
# Motion primitive constants
NODES_0 = np.array([[0.0,0.0,0.0]])
NODES_1 = np.array([[0.0,0.0,-i] for i in np.arange(0.0,N+.01,0.01)])
NODES_2 = np.array([[0.0,-i,-i] for i in np.arange(0.0,N+.01,0.01)])
NODES_3 = np.array([[0.0,-i,0.0] for i in np.arange(0.0,N+.01,0.01)])
NODES_4 = np.array([[0.0,-i,i] for i in np.arange(0.0,N+.01,0.01)])
NODES_5 = np.array([[0.0,0.0,i] for i in np.arange(0.0,N+.01,0.01)])
NODES_6 = np.array([[0.0,i,i] for i in np.arange(0.0,N+.01,0.01)])
NODES_7 = np.array([[0.0,i,0.0] for i in np.arange(0.0,N+.01,0.01)])
NODES_8 = np.array([[0.0,i,-i] for i  in np.arange(0.0,N+.01,0.01)])
NODES_9 = np.array([[i,0.0,0.0] for i in np.arange(0.0,N+.01,0.01)])
NODES_10 = np.array([[0.0,0.0,0.0],[N,0.0,0.0],[0.0,0.0,-N],[N,0.0,-N]])
NODES_11 = np.array([[0.0,0.0,0.0],[N,-N,0.0],[0.0,0.0,-N],[N,-N,-N]])
NODES_12 = np.array([[0.0,0.0,0.0],[N,0.0,0.0],[0.0,-N,0.0],[N,-N,0.0]])
NODES_13 = np.array([[0.0,0.0,0.0],[N,-N,0.0],[0.0,0.0,N],[N,-N,N]])
NODES_14 = np.array([[0.0,0.0,0.0],[N,0.0,0.0],[0.0,0.0,N],[N,0.0,N]])
NODES_15 = np.array([[0.0,0.0,0.0],[N,N,0.0],[0.0,0.0,N],[N,N,N]])
NODES_16 = np.array([[0.0,0.0,0.0],[N,0.0,0.0],[0.0,N,0.0],[N,N,0.0]])
NODES_17 = np.array([[0.0,0.0,0.0],[N,N,0.0],[0.0,0.0,-N],[N,N,-N]])
NODES_LIST = [NODES_0,NODES_1,NODES_2,NODES_3,NODES_4,NODES_5,
                  NODES_6,NODES_7,NODES_8,NODES_9,NODES_10,NODES_11,
                  NODES_12,NODES_13,NODES_14,NODES_15,NODES_16,NODES_17]
N_LINEAR_PATHS = 10
N_POINTS = np.arange(0.0,1.0,0.01)

# Drone Initialization Constants
GOAL1 = "Goal1"
GOAL2 = "Goal2"
GOAL3 = "Goal3"
START1 = "Start1"
START2 = "Start2"
START3 = "Start3"
STARTS = [START1,START2,START3]
GOALS = [GOAL1,GOAL2,GOAL3]



# input: MultirotorClient to get path for, index of motion primitive
# output: Path in world frame for client to execute
def getPath(client: airsim.MultirotorClient,i: int) -> list:
    # state data of drone
    state = client.getMultirotorState()
    orientation = state.kinematics_estimated.orientation
    pos = state.kinematics_estimated.position
    # list to hold the path
    p = list()
    # nodes that define linear path
    path = NODES_LIST[i]
    # nodes that define Bezier curve
    if i>=N_LINEAR_PATHS:
        path = Bezier.Curve(N_POINTS,path)
    # convert each point in path from multirotor body frame to world frame
    for point in path:
        # print(path)
        body_frame_vec = airsim.Vector3r(point[0],point[1],point[2])
        world_frame_vec = pos + orientation*body_frame_vec.to_Quaternionr()*orientation.star()
        p.append(world_frame_vec)   
    return p

# This should not be used
# input: MultirotorClient
# output: list of motion primitive paths
def generateMotionPrimitives(client: airsim.MultirotorClient) -> list:
    nodes_list = [NODES_0,NODES_1,NODES_2,NODES_3,NODES_4,NODES_5,
                  NODES_6,NODES_7,NODES_8,NODES_9,NODES_10,NODES_11,
                  NODES_12,NODES_13,NODES_14,NODES_15,NODES_16,NODES_17]
    # number of linear paths (Nodes 0 - 9) kept as variable in case it changes in future
    
    # list of motion primitives to output
    primitives_list = list()
    # state data of drone
    state = client.getMultirotorState()
    orientation = state.kinematics_estimated.orientation
    pos = state.kinematics_estimated.position
    for i in range(len(nodes_list)):
        # nodes that define linear path
        path = NODES_LIST[i]
        # nodes that define Bezier curve
        if i>=N_LINEAR_PATHS:
            path = Bezier.Curve(N_POINTS,path)
        # convert each point in path from multirotor body frame to world frame
        for point in path:
            # print(path)
            body_frame_vec = airsim.Vector3r(point[0],point[1],point[2])
            world_frame_vec = pos + orientation*body_frame_vec.to_Quaternionr()*orientation.star()
            p.append(world_frame_vec)   
        primitives_list.append(p)
        
    return primitives_list

# execute the motion primitive from the list based on index i with velocity vel
# input: MultirotorClient to move, index of motion primitive, desired velocity
# output: None
def execute_motion_primitive(client: airsim.MultirotorClient, i: int, vel: float) -> None:
    p = getPath(client, i)
    client.moveOnPathAsync(path=p,velocity=vel).join()
    # client.hoverAsync().join()

# reward function for the RL algorithm based on Camci et al.
# input: euclidean distance at beginning of timestep, at end of timestep, boolean asserted if drone has collided
# output: reward
def reward_function(d_t_min: float,d_t: float,collision: bool) -> float:
    if collision:
        return R_CP
    if abs(d_t)>D_MAX:
        return R_DP
    f = 0.5*(tanh((2*D_MAX-d_t)/D_MAX)+1)
    del_d = d_t-d_t_min
    if DEL_D_U < del_d:
        return R_L*f
    if DEL_D_L <= del_d and del_d <= DEL_D_U:
        return (R_L+(R_U-R_L)*(DEL_D_U-del_d)/(DEL_D_U-DEL_D_L))*f
    if del_d < DEL_D_L:
        return R_U*f

# processing for images with pixels interpreted as uint8
# input: ImageResponse from airsim
# output: np.array of img data
def img_format_uint8(response: airsim.ImageResponse) -> np.array:
    # convert string of bytes to array of uint8
    img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
    print("Image height and width: ", response.height,response.width, len(img1d))
    # reshape linear array into 2-d pixel array
    img_rgb = img1d.reshape(response.height,response.width,3)
    return img_rgb
   
# processing for images with pixels interpreted as floats
# input: ImageResponse from airsim
# output: np.array of img data
def img_format_float(response: airsim.ImageResponse) -> np.array:
    # convert list to np.array
    img1d = np.array(response.image_data_float)
    print("Image height and width: ", response.height,response.width, len(img1d))
    # reshape tp 2-d
    img_rgb = img1d.reshape(response.height,response.width)
    return img_rgb

# calculates the relative position of the moving setpoint wrt the drone in drone body frame
# input: MultirotorClient, moving setpoint
# output: vector representing relative position in body frame
def calculate_relative_pos(client: airsim.MultirotorClient, set_pt: airsim.Vector3r) -> airsim.Vector3r:
    state = client.getMultirotorState()
    pos = state.kinematics_estimated.position
    orientation = state.kinematics_estimated.orientation
    diff_vec_world = set_pt-pos
    diff_vec_body = orientation.star()*diff_vec_world*orientation
    
    return diff_vec_body

# calculates global path in world frame 
# input: client, environment number (1,2,3)
# output: vector representing global path
def init_episode(client: airsim.MultirotorClient, i: int) -> airsim.Vector3r:
    start_pose = client.simGetObjectPose(STARTS[i-1])
    goal_pose = client.simGetObjectPose(GOALS[i-1])
    client.simSetVehiclePose(start_pose,ignore_collision=True)
    client.armDisarm(True)
    client.takeoffAsync().join()

    return goal_pose.position-client.getMultirotorState().kinematics_estimated.position

# calculates moving setpoint for timestep in world frame
# input: client, global_path, timestep
# output: moving setpoint in world frame, maxed at the end goal
def get_moving_setpoint(client:airsim.MultirotorClient,global_path:airsim.Vector3r,timestep:int) -> airsim.Vector3r:
    gp_unit = global_path/global_path.get_length()
    sp = gp_unit*timestep
    return min(np.array([global_path,sp]),key=lambda p: p.get_length())

# returns index of action to take based on epsilon-greedy policy
# input: q function, state, epsilon
# output: index of action to be taken
def epsilon_greedy(q,s,epsilon):
    s = (s[0],s[1])
    mag_A = len(NODES_LIST)
    # print("mag_A: ", mag_A)
    a_star = np.argmax(q[s])
    weights = np.zeros(mag_A)+epsilon/mag_A
    weights[a_star]=1-epsilon+epsilon/mag_A 
    return np.random.choice(mag_A,p=weights)

class Episode:
    # initializes episode parameters
    # input: client, environment number (1,2,3)
    # output: None
    def __init__(self,client: airsim.MultirotorClient, n:int) -> None:
        self.n = n
        self.client = client
        start = client.simGetObjectPose(STARTS[n-1])
        self.goal_pose = client.simGetObjectPose(GOALS[n-1])
        self.client.simSetVehiclePose(start,ignore_collision=True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        self.start_pose = self.client.getMultirotorState().kinematics_estimated.position
        self.global_path = self.goal_pose.position-self.start_pose
    # calculates moving setpoint for timestep in world frame
    # input: client, global_path, timestep
    # output: moving setpoint in world frame, maxed at the end goal
    def get_moving_setpoint(self,timestep) -> airsim.Vector3r:
        gp_unit = self.global_path/self.global_path.get_length()
        sp = gp_unit*timestep
        return min(np.array([self.global_path,sp]),key=lambda p: p.get_length())+self.start_pose

'''
# Not sure if we need this directly. But I coded the Adam algorithm from the paper. The objective
# function is something that I am still confused about but looks like torch have a built in attr optim.Adam()
# that we can use as well https://pytorch.org/docs/stable/generated/torch.optim.Adam.html

import torch

# Example constant usage:  Good default settings for the tested machine learning problems are alpha = 0.001,
beta1 = 0.9, beta2 = 0.999 and epsilon = 10^-8. All operations on vectors are element-wise. With beta
t1 and betat2 we denote β1 and β2 to the power t.

F = # Stochastic objective function with parameters θ
ALPHA = 0.001 # step size
BETA1 = 0.9 # exponential decay rates for moment estimates
BETA2 = 0.999 # exponential decay rates for moment estimates
EPSILON = 1e-8 # small constant to prevent division by zero
THETA0 = np.zeros(10)  # initial parameter vector - What is the actual size and values of the paper?

def adam_optimizer(ALPHA, BETA1, BETA2, EPSILON, F, THETA0, convergence_threshold=1e-6):
    # Initialize parameters
    theta_t = torch.tensor(THETA0, requires_grad=True, dtype=torch.float) # : Parameter vector
    m_t = torch.zeros_like(theta_t) # The first moment vector initialized with zeros
    v_t = torch.zeros_like(theta_t) # The second moment vector initialized with zeros.
    t = 0 # Time step initialize at zero

    # The loop will be terminated when the convergence condition is met.
    while True:
        t += 1

        # Compute gradients
        g_t = torch.autograd.grad(F(theta_t), theta_t)[0]

        # Update biased first moment estimate
        m_t = BETA1 * m_t + (1 - BETA1) * g_t

        # Update biased second raw moment estimate
        v_t = BETA2 * v_t + (1 - BETA2) * (g_t ** 2)

        # Compute bias-corrected first moment estimate
        m_hat_t = m_t / (1 - BETA1 ** t)

        # Compute bias-corrected second raw moment estimate
        v_hat_t = v_t / (1 - BETA2 ** t)

        # Update parameters
        theta_t = theta_t - ALPHA * m_hat_t / (torch.sqrt(v_hat_t) + EPSILON)

        # Check for convergence
        if torch.norm(g_t) < convergence_threshold:
            break

    # Return the final parameter vector as a NumPy array after detaching it from the computational graph
    return theta_t.detach().numpy()

# Function run:
# result = adam_optimizer(alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, f=your_stochastic_objective_function, theta0=your_initial_parameters)

# ----------------------------------------------------------------------------------------------------------------
# Please note that this is purely using torch. Above, I attempted to do the adam as a function but then realized
# that torch have a built in attr we could use: https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
# I am also using: https://pytorch.org/docs/stable/generated/torch.nn.functional.relu.html for the nn.

import torch
import torch.nn as nn
import torch.optim as optim
import random

class QNetwork(nn.Module):
    # This defines a neural network class (QNetwork) using PyTorch's neural network module (nn.Module). 
    # It has two fully connected layers (fc and fc2) with ReLU activation.
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, output_size)

    # Forward method of the neural network, defining the forward pass through the layers.
    def forward(self, x):
        x = torch.relu(self.fc(x))
        x = self.fc2(x)
        return x

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

        3 Update interaction count
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

# Example of constants
episodes = 1000 # arbitrary at the moment
min_interaction_limit = 64 # per the min allowed on the paper that we should run
update_frequency = 100  # Update every 100 episodes
gamma = 0.99 # default
learning_rate = 0.001 # default
input_size = 4  # Need to modify based on the paper. Right now these are just dummies
output_size = 2  # Need to modify based on the paper. Right now these are just dummies

# Call airsim below

# Items that still be done:
1. NN input and output size? are these 32x32? how to define based on our enviroment?
2. episodes? how many?
3. epsilon? is this arbitrary? may have overlooked on the paper if is there.
4. 

'''