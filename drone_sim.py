#import setup_path
from matplotlib.colors import Colormap
import airsim

# Python Data Science Libraries
import numpy as np
import matplotlib.pyplot as plt

import os
import tempfile
import pprint
# import cv2
from Bezier import Bezier
# functions and variables associated with the RL Problem
from custom_functions import *
import neural_net
# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)

################################### TRAINING ###################################
# q_nn = neural_net.episodic_deep_q_learning(
#     episodes=NUM_EPISODES,
#     min_interaction_limit=MIN_INTERACTION_LIMIT,
#     episode_start=0,
#     # update_frequency=UPDATE_FREQUENCY,
#     gamma=GAMMA,
#     learning_rate=LEARNING_RATE,
#     client=client
# )
################################################################################

################################### TESTING ####################################
for _ in range(10):
    neural_net.run_policy(client,1)







##################################### DEBUGGING ################################
# ep = Episode(client,1)
# client.reset()
# for _ in range(10):
#     ep = Episode(client,np.random.choice([1,2,3]))
#     ep.flyGlobal()
    # mp = np.random.choice([4,5,6,13,14,15])
    # while(not ep.hasCollided()):
    #     ep.execute_motion_primitive(client,mp,1)
    #     ep.t+=1
    #     # print(client.getMultirotorState())
    #     print(ep.client.simGetCollisionInfo().object_name)
    # print(client.simGetCollisionInfo())
    # ep.reset()
#################################################################################

airsim.wait_key('Press any key to reset to original state')

client.reset()
client.armDisarm(False)

# that's enough fun for now. let's quit cleanly
client.enableApiControl(False)
