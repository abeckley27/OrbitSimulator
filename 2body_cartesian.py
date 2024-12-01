# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 18:51:18 2024

@author: Aidan
"""

import time
import numpy as np
from matplotlib import pyplot as plt
import numba

clock0 = time.time()

m1 = 6e24
m2 = 3e24
M = 1.989e30
G = 6.674e-11

t = 0.0
dt = 86400
t_final = int(4e7)

x0 = np.array([1.496e11, 0, 0])
v0 = np.array([0.0, 3e4, 0.0])

@numba.jit()
def distance(x):
    return np.sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2])

@numba.jit()
def acceleration(x):
    r2 = x[0]*x[0] + x[1]*x[1] + x[2]*x[2]
    u = x / np.sqrt(r2)     #Unit vector toward (0, 0, 0)
    return (-1 * G * M / r2) * u
    

r = x0
v = v0
a = acceleration(x0)
trajectory = []
speeds = []

while (t < t_final):
    r = r + v * dt + 0.5 * dt * dt * a
    a1 = acceleration(r)
    v = v + 0.5 * dt * (a + a1) 
    a = a1
    t += dt
    trajectory.append(r)
    speeds.append(v)

x = []
y = []
z = []
sma = 0

for pos in trajectory:
    x.append(pos[0])
    y.append(pos[1])
    z.append(pos[2])
    dist = distance(pos)
    if dist > sma:
        sma = dist


t = np.linspace(0, t_final - dt, len(x))
period = np.sqrt(4 * np.pi * sma**3 / (G*M))

x.append(0)
y.append(0)
z.append(0)


x = np.array(x)
y = np.array(y)
z = np.array(z)



# Creating figure
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
 
# Creating plot
ax.scatter3D(x, y, z, color = "blue")


print(time.time() - clock0)




