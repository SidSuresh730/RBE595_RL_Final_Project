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
MIN_INTERACTION_LIMIT = 64
DELTA = 1.0
EPISODES = 2000 
update_frequency = 100  # Update every 100 episodes
GAMMA = 0.99 # default
learning_rate = 0.001 # default
N_ROWS = 32
N_COLS = 32
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
POS1_NN_H2 = 8
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
    
    def run_DQN(self):
        pass
    
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
    