import os, sys, time
import math
import random
from lib.utils import *
from lib.CarlaEnv import CarlaEnv
import cv2
import matplotlib.pyplot as plt

try:
    carla_egg = get_carla_egg()
    sys.path.append(carla_egg)
except (IndexError, TypeError) as e:
    raise e("Please specify path to Carla .egg file in config/config.ini")

"""import carla

env = CarlaEnv()
env.reset()

time.sleep(3.0)
obs, reward, done = env.step([0, 1])
# time.sleep(3.0)
print(reward, done)
# print(obs[0])
# print(obs[1])
# plt.imshow(obs[0])
plt.imshow(obs[1])
plt.show()
# time.sleep(10.0)

env._clean_actors()

obs_shape = (2, 720, 1280, 3)
x = (obs_shape[0], obs_shape[-1], *obs_shape[1:-1])
print(type(x), x)

env = CarlaEnv(max_steps=100)
obs_shape = np.array(env.observation_space_shape[1:])
obs_shape = np.roll(obs_shape, 1)
print(obs_shape)

obs_shape = env.observation_space_shape
print(type(obs_shape), obs_shape)
obs_shape = [obs_shape[0], obs_shape[-1], *obs_shape[1:-1]]
obs_shape = list(obs_shape)
print(obs_shape)

env.close()"""

"""t = torch.zeros(1, 2, 3, 720, 1280)
print(t.size())
x = t[:,0,:,:,:]
y = t[:,1,:,:,:]
print(x.size())
print(y.size())
x = x.unsqueeze(1)
y = y.unsqueeze(1)
z = torch.cat((x,y), 1)
p = torch.cat((x,y), 0)

print(z.size())
print(p.size())"""

x = [x/10 for x in range(1, 101)]
y1 = []
y2 = []
y3 = []
y4 = []
y5 = []
for n in x:
    y1.append(math.log(n+0.7))
    y2.append(math.log1p(n))
    y3.append(math.log10(n))
    y4.append(math.log2(n))
    y5.append(math.log(n, 4))

# plt.plot(x,y1)
# plt.plot(x,y2)
# plt.plot(x,y3)
# plt.plot(x,y4)
# plt.plot(x,y5)
plt.show()
# input('continue\n')


