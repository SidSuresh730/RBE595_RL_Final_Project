import airsim
import time
import torch
from Bezier import Bezier
import numpy as np
import matplotlib.pyplot as plt
from math import tanh
from concurrent.futures import Future
# Learning Constants
D_MAX = 5.0 # Changed from 5 (paper)
R_L = 0.0
R_U = 1.0
DEL_D_L = 0 # Changed from 0 (paper)
DEL_D_U = 1.0
R_DP = -0.5
R_CP = -1.0
N = 1.0
MIN_INTERACTION_LIMIT = 64
MINIBATCH_SIZE = 8
LEARNING_RATE = 1.0
EPISODES = 2000 
# UPDATE_FREQUENCY = 100  # Update every 100 episodes
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
COLLISION_BUFFER_SIZE = 20
ZERO_VECTOR = airsim.Vector3r()
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
# OBSTACLE_2_NAME = "Obstacle_2a"
# OBSTACLE_3_NAME = "Obstacle_3a"
# OBSTACLE_4_NAME = "Obstacle_3b"
ASSET = '1M_Cube_Chamfer'
# OBSTACLE_2_POSITION = airsim.Vector3r(x_val=5.6,y_val=-0.2,z_val=8.92)
# OBSTACLE_3_POSITION = airsim.Vector3r(x_val=3.8,y_val=-2.6,z_val=8.92)
# OBSTACLE_4_POSITION = airsim.Vector3r(x_val=7.9,y_val=2.4,z_val=8.92)
# OBSTACLE_ORIENTATION = airsim.Quaternionr(w_val=1.0,x_val=-0.0,y_val=0.0,z_val=0.0)
# OBSTACLE_SCALE = airsim.Vector3r(x_val=0.75,y_val=6.25,z_val=9.0)
# OBSTACLE_2_POSE = airsim.Pose(position_val=OBSTACLE_2_POSITION,orientation_val=OBSTACLE_ORIENTATION)
# OBSTACLE_3_POSE = airsim.Pose(position_val=OBSTACLE_3_POSITION,orientation_val=OBSTACLE_ORIENTATION)
# OBSTACLE_4_POSE = airsim.Pose(position_val=OBSTACLE_4_POSITION,orientation_val=OBSTACLE_ORIENTATION)
# File writing constants
# CSV_NAME = "rewards.csv"
PT_NAME = "dqn.pt"
PT_PATH = "./dqn.pt"

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

# Creates obstacles for episode type n = (1,2,3)
def setup_obstacles(client: airsim.MultirotorClient,n:int) -> None:
    if n==1:
        return
    if n==2:
        worked = client.simSpawnObject(OBSTACLE_2_NAME,asset_name=ASSET,pose=OBSTACLE_2_POSE,scale=OBSTACLE_SCALE)
        while not worked:
            print("Spawning Obstacle_2")
            worked = client.simSpawnObject(OBSTACLE_2_NAME,asset_name=ASSET,pose=OBSTACLE_2_POSE,scale=OBSTACLE_SCALE)
        return
    if n==3:
        worked1 = client.simSpawnObject(OBSTACLE_3_NAME,asset_name=ASSET,pose=OBSTACLE_3_POSE,scale=OBSTACLE_SCALE)
        worked2 = client.simSpawnObject(OBSTACLE_4_NAME,asset_name=ASSET,pose=OBSTACLE_4_POSE,scale=OBSTACLE_SCALE)
        while not worked1:
            print("Spawning Obstacle_3")
            worked1 = client.simSpawnObject(OBSTACLE_3_NAME,asset_name=ASSET,pose=OBSTACLE_3_POSE,scale=OBSTACLE_SCALE)
        while not worked2:
            print("Spawning Obstacle_4")
            worked2 = client.simSpawnObject(OBSTACLE_4_NAME,asset_name=ASSET,pose=OBSTACLE_4_POSE,scale=OBSTACLE_SCALE)
        return
# Destroys obstacles for episode type n = (1,2,3)
def cleanup_obstacles(client: airsim.MultirotorClient,n:int) -> None:
    if n==1:
        return
    if n==2:
        worked = client.simDestroyObject(OBSTACLE_2_NAME)
        while not worked:
            print("Destroying Obstacle_2")
            worked = client.simDestroyObject(OBSTACLE_2_NAME)
        return
    if n==3:
        worked1 = client.simDestroyObject(OBSTACLE_3_NAME)
        worked2 = client.simDestroyObject(OBSTACLE_4_NAME)
        while not worked1:
            print("Destroying Obstacle_3")
            worked1 = client.simDestroyObject(OBSTACLE_3_NAME)
        while not worked2:
            print("Destroying Obstacle_4")
            worked2 = client.simDestroyObject(OBSTACLE_4_NAME)
        return

# execute the motion primitive from the list based on index i with velocity vel
# input: MultirotorClient to move, index of motion primitive, desired velocity
# output: None
def execute_motion_primitive(client: airsim.MultirotorClient, i: int, vel: float) -> None:
    p = getPath(client, i)
    response1 = client.simGetImages([airsim.ImageRequest("0", image_type=airsim.ImageType.DisparityNormalized,compress=False, pixels_as_float=True)])
    move=client.moveOnPathAsync(path=p,velocity=vel)
    response2 = client.simGetImages([airsim.ImageRequest("0", image_type=airsim.ImageType.DisparityNormalized,compress=False, pixels_as_float=True)])
    response3= client.simGetImages([airsim.ImageRequest("0", image_type=airsim.ImageType.DisparityNormalized,compress=False, pixels_as_float=True)])
    
    move.join()
    print("Motion primitive: %d. Joined."%i)
    # print(client.getMultirotorState())
    img1 = img_format_float(response1[0])
    img2 = img_format_float(response2[0])
    img3 = img_format_float(response3[0])
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
    # print("Image height and width: ", response.height,response.width, len(img1d))
    # reshape linear array into 2-d pixel array
    img_rgb = img1d.reshape(response.height,response.width,3)
    return img_rgb
   
# processing for images with pixels interpreted as floats
# input: ImageResponse from airsim
# output: np.array of img data
def img_format_float(response: airsim.ImageResponse) -> np.array:
    # convert list to np.array
    img1d = np.array(response.image_data_float)
    # print("Image height and width: ", response.height,response.width, len(img1d))
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
    diff_vec_body = orientation.star()*diff_vec_world.to_Quaternionr()*orientation
    
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
    a_star = np.random.choice(np.where(q_s==q_s.max())[0])
    weights = np.zeros(mag_A)+epsilon/mag_A
    weights[a_star]=1-epsilon+epsilon/mag_A 
    return np.random.choice(mag_A,p=weights)

class Episode:
    # initializes episode parameters
    # input: client, environment number (1,2,3)
    # output: None
    def __init__(self,client: airsim.MultirotorClient, n:int) -> None:
        print("Episode type: %d" %n)
        self.n = n
        self.t = 0
        self.client = client
        self.start = client.simGetObjectPose(STARTS[n-1])
        # setup_obstacles(client=self.client,n=self.n)
        self.goal_pose = client.simGetObjectPose(GOALS[n-1])
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.simSetVehiclePose(self.start,ignore_collision=True)
        # self.collision_buffer = np.zeros()
        # time.sleep(5)
        # while(self.client.getMultirotorState().landed_state==1):
        self.client.takeoffAsync().join()
        self.start_pos = self.client.getMultirotorState().kinematics_estimated.position
        self.global_path = self.goal_pose.position-self.start_pos
        # im1,im2,im3 = execute_motion_primitive(self.client,0,vel=1.0)
        response1 = client.simGetImages([airsim.ImageRequest("0", image_type=airsim.ImageType.DisparityNormalized,compress=False, pixels_as_float=True)])
        response2 = client.simGetImages([airsim.ImageRequest("0", image_type=airsim.ImageType.DisparityNormalized,compress=False, pixels_as_float=True)])
        response3= client.simGetImages([airsim.ImageRequest("0", image_type=airsim.ImageType.DisparityNormalized,compress=False, pixels_as_float=True)])
        # print(client.getMultirotorState())
        im1 = img_format_float(response1[0])
        im2 = img_format_float(response2[0])
        im3 = img_format_float(response3[0])
        sp = self.get_moving_setpoint(self.t)
        rel_pos = calculate_relative_pos(self.client,sp)
        self.state = State(im1,im2,im3,rel_pos,sp)
        self.last_collision_time = self.client.simGetCollisionInfo().time_stamp

    # For test purposes
    # Checking to see that drone flies the global path properly
    def flyGlobal(self):
        # for i in range(30):
        #     print(i)
        #     sp = self.get_moving_setpoint(i)
            # self.client.moveToPositionAsync(sp.x_val,sp.y_val,sp.z_val,velocity=1).join()
        self.t+=1
        self.client.moveToPositionAsync(self.goal_pose.position.x_val,self.goal_pose.position.y_val,self.goal_pose.position.z_val,velocity=2,timeout_sec=20).join()
        if self.hasCollided():
            print("Collision!")
            return
        if self.reachedGoal():
            print("Reached goal!")
            return
    # calculates moving setpoint for timestep in world frame
    # input: client, global_path, timestep
    # output: moving setpoint in world frame, maxed at the end goal
    def get_moving_setpoint(self,timestep) -> airsim.Vector3r:
        gp_unit = self.global_path/self.global_path.get_length()
        sp = gp_unit*(timestep+1)
        return min(np.array([self.global_path,sp]),key=lambda p: p.get_length())+self.start_pos
    
    # steps through episode
    # input: action to take 
    # output: next state, reward
    def step(self,a:int) -> tuple:
        # self.client.simPause(True)
        d_t_min = self.state.get_dist()
        sp = self.state.get_sp()
        img1,img2,img3,collided,done = self.execute_motion_primitive(client=self.client,i=a,vel=1.0)
        d_t = calculate_relative_pos(self.client,sp).get_length()
        # self.t+=1
        # collided = self.hasCollided()
        done = done or self.reachedGoal()
        r = reward_function(d_t_min=d_t_min,d_t=d_t,collision=(collided or self.hasCollided()))
        self.t+=1
        # if not self.t%3:
        #     self.state.display_imgs()
        new_sp = self.get_moving_setpoint(self.t)
        rel_pos = calculate_relative_pos(self.client,new_sp)
        new_state = State(img1=img1,img2=img2,img3=img3,pos=rel_pos,sp=new_sp)
        return (new_state,r,done)

    
    # logic to determine if drone has collided with an object that is not the goal
    # returns True if drone has collided with an obstacle
    def hasCollided(self) -> bool:
        collision_info = self.client.simGetCollisionInfo()
        # collision_info = self.client.getMultirotorState().collision
        # print(collision_info)
        has_collided = collision_info.time_stamp>self.last_collision_time
        if self.t == 0:
            self.last_collision_time = collision_info.time_stamp
        # has_collided = collision_info.impact_point.x_val != 0.0 or collision_info.impact_point.y_val != 0 or collision_info.impact_point.z_val!=0
        return (has_collided and collision_info.object_name != GOALS[self.n-1] ) and self.t != 0
            # return collision_info.impact_point != self.start.position
                
    # logic to determine if drone has collided with the goal
    # returns True if drone has reached the goal 
    def reachedGoal(self) -> bool:
        collision_info = self.client.simGetCollisionInfo()
        has_collided = collision_info.time_stamp>self.last_collision_time
        return has_collided and collision_info.object_name == GOALS[self.n-1]
    
    # execute the motion primitive from the list based on index i with velocity vel
    # input: MultirotorClient to move, index of motion primitive, desired velocity
    # output: None
    def execute_motion_primitive(self,client: airsim.MultirotorClient, i: int, vel: float) -> None:
        p = getPath(client, i)
        move=client.moveOnPathAsync(path=p,velocity=vel,timeout_sec=5)
        response1 = client.simGetImages([airsim.ImageRequest("0", image_type=airsim.ImageType.DisparityNormalized,compress=False, pixels_as_float=True)])
        response2 = client.simGetImages([airsim.ImageRequest("0", image_type=airsim.ImageType.DisparityNormalized,compress=False, pixels_as_float=True)])
        response3= client.simGetImages([airsim.ImageRequest("0", image_type=airsim.ImageType.DisparityNormalized,compress=False, pixels_as_float=True)])
        collided = False
        done = False
        # for _ in range(COLLISION_BUFFER_SIZE):
        #     collided = collided or self.hasCollided()
        #     done = done or self.reachedGoal()
        #     if collided or done:
        #         print("Collided or Done!")
        #         client.cancelLastTask()
        #         break
        move.join()
        # while(move.join()):
        collided = self.hasCollided()
        done = self.reachedGoal()
        print("Motion primitive: %d. Joined."%i)
        # print(client.getMultirotorState())
        img1 = img_format_float(response1[0])
        img2 = img_format_float(response2[0])
        img3 = img_format_float(response3[0])
        return img1,img2,img3,collided,done
    
    def reset(self) -> None:
        # self.client.landAsync().join()
        self.client.hoverAsync().join()
        while(self.hasCollided()):
            print("Resetting...")
            self.client.reset()
        # self.client.armDisarm(False)
        # self.client.simPause(True)
        # self.client.simPause(False)
        # self.client.enableApiControl(False)
        # cleanup_obstacles(client=self.client,n=self.n)
        
        
        
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
    
    def display_imgs(self):
        plt.figure(1)
        plt.imshow(self.img1,cmap='binary')
        plt.figure(2)
        plt.imshow(self.img2,cmap='binary')
        plt.figure(3)
        plt.imshow(self.img3,cmap='binary')
        plt.show()
    
    def diff(self,other: 'State'):
        self.pos.distance_to(other.pos)
    

def write_to_csv(filename: str, episode: int,data:list) -> None:
    f = open(filename,"a")
    f.write("Episode: %d"%episode)
    f.write(",")
    for e in data:
        f.write(str(e))
        f.write(",")
    f.write("\n")
    f.close()