#import setup_path
from matplotlib.colors import Colormap
import airsim

# Python Data Science Libraries
import numpy as np
import matplotlib.pyplot as plt

import os
import tempfile
import pprint
import cv2
from Bezier import Bezier
# functions and variables associated with the RL Problem
import custom_functions as custom

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
gp = custom.Episode(client=client,n=3)
# startPose1 = client.simGetObjectPose("PlayerStart1")
# s = pprint.pformat(startPose1)
# print("start pose: %s" % s)
# state = client.getMultirotorState()
# s = pprint.pformat(state)
# print("state: %s" % s)

# imu_data = client.getImuData()
# s = pprint.pformat(imu_data)
# print("imu_data: %s" % s)

# barometer_data = client.getBarometerData()
# s = pprint.pformat(barometer_data)
# print("barometer_data: %s" % s)

# magnetometer_data = client.getMagnetometerData()
# s = pprint.pformat(magnetometer_data)
# print("magnetometer_data: %s" % s)

# gps_data = client.getGpsData()
# s = pprint.pformat(gps_data)
# print("gps_data: %s" % s)

# airsim.wait_key('Press any key to takeoff')
# print("Taking off...")
# client.armDisarm(True)
# client.takeoffAsync().join()

state = client.getMultirotorState()
print("state: %s" % pprint.pformat(state))

airsim.wait_key('Press any key to follow the global path')
#client.moveToPositionAsync(-10, 10, -10, 5).join()
# custom.execute_motion_primitive(client,10,1.0)
# p = custom.getPath(client)
# p = custom.generateMotionPrimitives(client)[17]
for i in range(0,50):
    p = gp.get_moving_setpoint(timestep=i)
    x,y,z = p.x_val,p.y_val,p.z_val
    client.moveToPositionAsync(x=x,y=y,z=z,velocity=5.0).join()
    print(client.simGetCollisionInfo().has_collided)

# state = client.getMultirotorState()
# print("state: %s" % pprint.pformat(state))
print("state: %s" % state.kinematics_estimated)
airsim.wait_key('Press any key to take images')
# get camera images from the car
# responses = client.simGetImages([
#     airsim.ImageRequest("0", airsim.ImageType.DepthVis),  #depth visualization image
#     airsim.ImageRequest("1", airsim.ImageType.DepthPerspective, True), #depth in perspective projection
#     airsim.ImageRequest("1", airsim.ImageType.Scene), #scene vision image in png format
#     airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)])  #scene vision image in uncompressed RGBA array
# print('Retrieved images: %d' % len(responses))
response = client.simGetImages([airsim.ImageRequest("0", image_type=airsim.ImageType.DisparityNormalized,compress=False, pixels_as_float=True)])
# print(response[0].image_data_uint8)
# img = custom.img_format(airsim.string_to_uint8_array(response))
print(response[0])
img = custom.img_format_float(response[0])
print(img)
plt.imshow(img,cmap='binary')
plt.show()
# tmp_dir = os.path.join(tempfile.gettempdir(), "airsim_drone")
# print ("Saving images to %s" % tmp_dir)
# try:
#     os.makedirs(tmp_dir)
# except OSError:
#     if not os.path.isdir(tmp_dir):
#         raise

# for idx, response in enumerate(responses):

#     filename = os.path.join(tmp_dir, str(idx))

#     if response.pixels_as_float:
#         print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
#         airsim.write_pfm(os.path.normpath(filename + '.pfm'), airsim.get_pfm_array(response))
#     elif response.compress: #png format
#         print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
#         airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
#     else: #uncompressed array
#         print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
#         img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) # get numpy array
#         img_rgb = img1d.reshape(response.height, response.width, 3) # reshape array to 4 channel image array H X W X 3
#         cv2.imwrite(os.path.normpath(filename + '.png'), img_rgb) # write to png

airsim.wait_key('Press any key to reset to original state')

client.reset()
client.armDisarm(False)

# that's enough fun for now. let's quit cleanly
client.enableApiControl(False)
