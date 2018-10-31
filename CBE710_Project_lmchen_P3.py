# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 14:40:48 2018

@author: Lawrence
"""

# Preamble
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
from MD_functions import AssembleCubeLattice, AssignInitialVelocities, Plot_3D_configuration, Make_3D_animation

# Function for calculating forces
"""
def CalcForces(Distances, sig):
    # Distances is the distance in xyz-coordinates between each pair of particles
    # sig is the minimum distance between two particles
    Distance_Norm = np.linalg.norm(Distances,axis=2)  
    forces = np.zeros(Distances.shape)
    F = np.zeros([len(forces),len(forces[0][0])])
    for i in range(len(forces)-1):
        for j in range(i+1,len(forces)):
            # BE SURE TO CHECK IF THE NEXT LINE NEEDS THE NEGATIVE IN THE FRONT OR NOT!!!!!!!!!!!!
            forces[i,j,:] = 48/Distance_Norm[i,j]**2 * ((sig/Distance_Norm[i,j])**12 - 0.5*(sig/Distance_Norm[i,j])**6) * Distances[i,j,:]
            forces[j,i,:] = -forces[i,j,:]
        F[i,:] = np.sum(forces[i,:,:],axis=0)
    F[N-1,:] = np.sum(forces[N-1,:,:],axis=0)  
    return F
"""
def CalcForces2(Distances,sig,energy):
    Norms = np.linalg.norm(Distances,axis=2)
    Norms[Norms == 0] = 0.1
    Norms = 1/Norms
    Norms[Norms == 10] = 0
    term = (sig*Norms)**6
    Potential = np.sum(2*energy*(term**2 - term))
    dx = np.sum(-48*energy*Norms**2*(term**2 - 0.5*term)*Distances[:,:,0],axis=0)
    dy = np.sum(-48*energy*Norms**2*(term**2 - 0.5*term)*Distances[:,:,1],axis=0)
    dz = np.sum(-48*energy*Norms**2*(term**2 - 0.5*term)*Distances[:,:,2],axis=0)
    F = np.stack((dx,dy,dz),axis=-1)
    return F, Potential


plt.close("all")
dt = 0.001
Nt = 50000
sample = 50
m = 1.0
e = 1.0
kb = 1.0

sigma = 1
T = 1.2*e/kb
rho = 0.01/sigma**3
eta = 0.1

npx, npy, npz = 5,5,5
N = npx*npy*npz
box_length = (N/rho)**(1/3)
diam = box_length/(max(npx,npy,npz))

# Initialize particle positions and size of box
Positions, bsize = AssembleCubeLattice(diam,rho,npx,npy,npz)

# Initialize velocities based on Boltzmann Distribution at Temperature T
Vels = AssignInitialVelocities(T,kb,m,npx*npy*npz)

Plot_3D_configuration(Positions)
# d1 is the xyz-distances between every pair of particles
d1 = Positions[:,np.newaxis,:]-Positions[np.newaxis,:,:]

# This next line rescales the xyz-distances to consider only nearest-image distances due to PBC's
d1 = d1 - (np.abs(d1) > bsize/2)*(np.sign(d1))*np.array([[bsize,]*N]*N)

# Calculate initial forces between particles based ONLY on nearest-image distances from PBC's
Forces,PE = CalcForces2(d1,sigma,e)

Saved_Pos = np.zeros([int(Nt/sample)+1,N,3])
t = np.arange(int(Nt/sample)+1)
Ktot = np.zeros(int(Nt/sample)+1)
Ptot = np.zeros(int(Nt/sample)+1)

# BEGIN FOR-LOOP HERE AFTER FINISHING EVERYTHING ELSE
for Trials in range(Nt+1):
    if(Trials % sample == 0):
        Saved_Pos[int(Trials/sample),:,:] = Positions        
        Ktot[int(Trials/sample)] = 0.5*m*np.sum(Vels**2)/N
        Ptot[int(Trials/sample)] = PE/N
        
    print(Trials)
    # Calculating new positions
    #print(Positions)
    Positions = Positions + Vels*dt + 0.5*Forces/m*dt**2
    #print(Positions)
    
    # Make sure the new Positions are all still in the box
    Positions = Positions - (Positions > bsize)*np.array([bsize,]*N) + (Positions < 0)*np.array([bsize,]*N)
    #print(Positions)    
    
    VelsHalf = Vels + 0.5*Forces/m*dt
    
    d1 = Positions[:,np.newaxis,:]-Positions[np.newaxis,:,:]
    d1 = d1 - (np.abs(d1) > bsize/2)*(np.sign(d1))*np.array([[bsize,]*N]*N)
    Forces2,PE2 = CalcForces2(d1,sigma,e)
    
    Vels = VelsHalf + 0.5*Forces2/m*dt
    Forces = Forces2
    PE = PE2
    
Plot_3D_configuration(Positions)
Make_3D_animation(Saved_Pos,)

plt.figure()
plt.plot(t,Ktot,'-r')
plt.figure()
plt.plot(t,Ptot,'-b')
plt.figure()
plt.plot(t,Ktot+Ptot,'-')
plt.show()