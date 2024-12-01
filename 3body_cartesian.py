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

#Earth's initial conditions
x0 = np.array([1.496e11, 0, 0])
v0 = np.array([0.0, 3e4, 0.0])


#Planet 2's initial conditions
x20 = np.array([0, 1e11, 0])
v20 = np.array([4e4, 0, 0])

@numba.jit(nopython=True)
def distance(x):
    return np.sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2])

@numba.jit(nopython=True)
def acceleration(x1, x2):
    #calculate the distances between the objects
    r_1s = x1[0]*x1[0] + x1[1]*x1[1] + x1[2]*x1[2]  #Actually r^2
    r_2s = x2[0]*x2[0] + x2[1]*x2[1] + x2[2]*x2[2]
    r_12 = (x1[0] - x2[0])**2 + (x1[1] - x2[1])**2 + (x1[2] - x2[2])**2
    
    #Then the forces
    F_1s = -1 * G * M * m1 / r_1s
    F_2s = -1 * G * M * m2 / r_2s
    F_12 = G * m1 * m2 / r_12
    
    #Create unit vectors in each direction
    u1 = x1 / np.sqrt(r_1s)     #Unit vectors toward (0, 0, 0)
    u2 = x2 / np.sqrt(r_2s)
    u3 = (x2 - x1) / np.sqrt(r_12)
    
    #calculate the acceleration for each mass
    return ((F_1s * u1 + F_12 * u3) / m1, (F_2s * u2 - F_12 * u3) / m2)
    

r1 = x0
v1 = v0
trajectory1 = []
speeds1 = []
r2 = x20
v2 = v20
a = acceleration(x0, x20)
trajectory2 = []
speeds2 = []

while (t < t_final):
    r1 = r1 + v1 * dt + 0.5 * dt * dt * a[0]
    r2 = r2 + v2 * dt + 0.5 * dt * dt * a[1]
    a_new = acceleration(r1, r2)
    v1 = v1 + 0.5 * dt * (a[0] + a_new[0])
    v2 = v2 + 0.5 * dt * (a[1] + a_new[1])
    a = a_new
    t += dt
    trajectory1.append(r1)
    speeds1.append(v1)
    trajectory2.append(r2)
    speeds2.append(v2)

x = []
y = []
z = []
sma = 0

for pos in trajectory1:
    x.append(pos[0])
    y.append(pos[1])
    z.append(pos[2])
    dist = distance(pos)
    if dist > sma:
        sma = dist
        
x2 = []
y2 = []
z2 = []
sma2 = 0

for pos in trajectory2:
    x2.append(pos[0])
    y2.append(pos[1])
    z2.append(pos[2])
    dist = distance(pos)
    if dist > sma2:
        sma2 = dist


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
ax.scatter3D(x2, y2, z2, color = "green")


print(time.time() - clock0)




