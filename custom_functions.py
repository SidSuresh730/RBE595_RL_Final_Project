import airsim
from Bezier import Bezier
import numpy as np
from math import tanh
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

# Learning Constants
D_MAX = 5.0
R_L = 0.0
R_U = 1.0
DEL_D_L = 0.0
DEL_D_U = 1.0
R_DP = -0.5
R_CP = -1.0
N = 1.0
MIN_INTERACTION_LIMIT = 64
LEARNING_RATE = 1.0
EPISODES = 2000 
UPDATE_FREQUENCY = 100  # Update every 100 episodes
BATCH_SIZE = 64 # Per paper
GAMMA = 0.5
N_ROWS = 32
N_COLS = 32
NUM_EPISODES = 2000
# image neural network layer sizes
IMG_NN_INPUT = N_ROWS*N_COLS
IMG_NN_H1 = 128
IMG_NN_H2 = 16
IMG_NN_H3 = 8
IMG3_NN_H3 = 16
IMG_NN_H4 = 16
# combined neural network layer sizes
NN_H5 = 32
NN_OUTPUT = 18
# position neural network layer sizes
POS_NN_INPUT = 1
POS_NN_H1 = 8
POS_NN_H2 = 4
POSX_NN_H2 = 8
POS_NN_H3 = 8


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

# File writing constants
CSV_NAME = "rewards.csv"
PT_NAME = "dqn.pt"

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
    response1 = client.simGetImages([airsim.ImageRequest("0", image_type=airsim.ImageType.DisparityNormalized,compress=False, pixels_as_float=True)])
    move=client.moveOnPathAsync(path=p,velocity=vel)
    response2 = client.simGetImages([airsim.ImageRequest("0", image_type=airsim.ImageType.DisparityNormalized,compress=False, pixels_as_float=True)])
    response3= client.simGetImages([airsim.ImageRequest("0", image_type=airsim.ImageType.DisparityNormalized,compress=False, pixels_as_float=True)])

    img1 = img_format_float(response1[0])
    img2 = img_format_float(response2[0])
    img3 = img_format_float(response3[0])
    move.join()
    return img1,img2,img3
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
# input: global_path, timestep
# output: moving setpoint in world frame, maxed at the end goal
def get_moving_setpoint(global_path:airsim.Vector3r,timestep:int) -> airsim.Vector3r:
    gp_unit = global_path/global_path.get_length()
    sp = gp_unit*timestep
    return min(np.array([global_path,sp]),key=lambda p: p.get_length())

# returns index of action to take based on epsilon-greedy policy
# input: q function, state, epsilon
# output: index of action to be taken
def epsilon_greedy(q_s,epsilon):
    # s = (s[0],s[1])
    mag_A = len(NODES_LIST)
    # print("mag_A: ", mag_A)
    a_star = np.argmax(q_s)
    weights = np.zeros(mag_A)+epsilon/mag_A
    weights[a_star]=1-epsilon+epsilon/mag_A 
    return np.random.choice(mag_A,p=weights)

class Episode:
    # initializes episode parameters
    # input: client, environment number (1,2,3)
    # output: None
    def __init__(self,client: airsim.MultirotorClient, n:int) -> None:
        self.n = n
        self.t = 0
        self.client = client
        start = client.simGetObjectPose(STARTS[n-1])
        self.goal_pose = client.simGetObjectPose(GOALS[n-1])
        self.client.simSetVehiclePose(start,ignore_collision=True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        self.start_pos = self.client.getMultirotorState().kinematics_estimated.position
        self.global_path = self.goal_pose.position-self.start_pos
        im1,im2,im3 = execute_motion_primitive(self.client,0,vel=1.0)
        sp = self.get_moving_setpoint(self.t)
        rel_pos = calculate_relative_pos(self.client,sp)
        self.state = State(im1,im2,im3,rel_pos,sp)

    # calculates moving setpoint for timestep in world frame
    # input: client, global_path, timestep
    # output: moving setpoint in world frame, maxed at the end goal
    def get_moving_setpoint(self,timestep) -> airsim.Vector3r:
        gp_unit = self.global_path/self.global_path.get_length()
        sp = gp_unit*(timestep+1)
        return min(np.array([self.global_path,sp]),key=lambda p: p.get_length())+self.start_pose
    
    # steps through episode
    # input: action to take 
    # output: next state, reward
    def step(self,a:int) -> tuple:
        d_t_min = self.state.get_dist()
        sp = self.state.get_sp()
        img1,img2,img3 = execute_motion_primitive(client=self.client,i=a,vel=1.0)
        d_t = calculate_relative_pos(self.client,sp).get_length()
        collided = self.hasCollided()
        done = self.reachedGoal()
        r = reward_function(d_t_min=d_t_min,d_t=d_t,collision=collided)
        self.t+=1
        new_sp = self.get_moving_setpoint(self.t)
        rel_pos = calculate_relative_pos(self.client,new_sp)
        new_state = State(img1=img1,img2=img2,img3=img3,pos=rel_pos,sp=new_sp)
        return (new_state,r,done)

    
    # logic to determine if drone has collided with an object that is not the goal
    # returns True if drone has collided with an obstacle
    def hasCollided(self) -> bool:
        collision_info = self.client.simGetCollisionInfo()
        return collision_info.has_collided and collision_info.object_name != GOALS[self.n-1]
    
    # logic to determine if drone has collided with the goal
    # returns True if drone has reached the goal 
    def reachedGoal(self) -> bool:
        collision_info = self.client.simGetCollisionInfo()
        return collision_info.has_collided and collision_info.object_name == GOALS[self.n-1]

# class to hold state data for use in NN and reward function
class State:
    def __init__(self,img1:np.array,img2:np.array,img3:np.array,pos:airsim.Vector3r,sp:airsim.Vector3r) -> None:
        self.img1 = img1
        self.img2 = img2
        self.img3 = img3
        # relative pos
        self.pos = pos
        # moving setpoint
        self.sp = sp
    
    def get_inputs(self) -> tuple:
        return (self.img1,self.img2,self.img3,self.pos.x_val,self.pos.y_val,self.pos.z_val)
    
    def get_dist(self) -> float:
        return self.pos.get_length()
    
    def get_pos(self) -> airsim.Vector3r:
        return self.pos

    def get_sp(self) -> airsim.Vector3r:
        return self.sp
    

def write_to_csv(filename: str, data:list) -> None:
    f = open(filename,"a")
    f.write(time.ctime(time.time()))
    f.write(",")
    for e in data:
        f.write(e)
        f.write(",")
    f.close()
# ----------------------------------------------------------------------------------------------------------------
# Please note that this is purely using torch. Above, I attempted to do the adam as a function but then realized
# that torch have a built in attr we could use: https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
# I am also using: https://pytorch.org/docs/stable/generated/torch.nn.functional.relu.html for the nn.

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

def DQN():
    # Initialize AirSim client and environment number
    client = airsim.MultirotorClient()
    client.confirmConnection() # # Confirm connection to the AirSim simulator
    client.enableApiControl(True)
    client.armDisarm(True)
    environment_number = 1  # Set the appropriate environment number

    # Initialize your deep Q-network
    q_network = DQNetwork().to(device)
    optimizer = torch.optim.Adam(q_network.parameters(), lr=LEARNING_RATE)

    # Initialize experience replay memory
    experience_replay = []

    for episode in range(EPISODES):
        # Reset environment and initialize state
        episode_obj = Episode(client, environment_number)
        state = State(*[execute_motion_primitive(client, i, vel=1.0) for i in range(N_LINEAR_PATHS)], episode_obj.start_pose)

        for t in range(MIN_INTERACTION_LIMIT):
            # Epsilon-greedy action selection
            epsilon = max(0.1, 1.0 - episode / 1000)  # Decay epsilon over time
            action = epsilon_greedy(q_network(state.get_inputs()), epsilon)

            # Execute selected action and observe new state and reward
            episode_obj.step(action)
            new_images = [execute_motion_primitive(client, i, vel=1.0) for i in range(N_LINEAR_PATHS)]
            new_position = client.getMultirotorState().kinematics_estimated.position
            new_state = State(*new_images, new_position)

            # Calculate reward and check for termination conditions
            reward = reward_function(state.get_dist(), new_state.get_dist(), episode_obj.hasCollided())
            if episode_obj.reachedGoal():
                reward = 1.0

            # Store the experience in replay memory
            experience_replay.append((state, action, reward, new_state))

            # Update state for the next iteration
            state = new_state

            # Perform experience replay if applicable
            if len(experience_replay) >= MIN_INTERACTION_LIMIT and t % UPDATE_FREQUENCY == 0:
                # Sample a random minibatch from experience replay
                minibatch = random.sample(experience_replay, BATCH_SIZE)

                # Compute Q-learning targets (y) for the minibatch
                targets = []
                for sample in minibatch:
                    s, a, r, s_prime = sample
                    target = r + GAMMA * torch.max(q_network(s_prime.get_inputs()))
                    targets.append(target)

                targets = torch.stack(targets)

                # Compute Q-values for the minibatch
                q_values = q_network(torch.stack([s.get_inputs() for s, _, _, _ in minibatch]))

                # Calculate Huber loss
                loss = F.smooth_l1_loss(q_values.gather(1, torch.tensor([[a] for _, a, _, _ in minibatch], device=device)), targets.unsqueeze(1))

                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Clear experience replay memory
                experience_replay = []

            # Break if collision or goal reached
            if episode_obj.hasCollided() or episode_obj.reachedGoal():
                break

# Call the DQN function to start training
DQN()

'''def DQN(episodes, min_interaction_limit, update_frequency, gamma, learning_rate, input_size, output_size, env):
    
    # This initializes the main Q-network and a target Q-network. 
    # The target Q-network is used for stability in training, and its parameters are 
    # initially set to be the same as the main Q-network.
    q_network = DQNetwork(input_size, output_size)
    target_q_network = DQNetwork(input_size, output_size)
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

        # Update interaction count
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


dqn=DQN(episodes, min_interaction_limit, update_frequency, gamma, learning_rate, input_size, output_size, env)
print(dqn)

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
        return x'''

