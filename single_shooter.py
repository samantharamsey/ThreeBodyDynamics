# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 17:39:11 2022

@author: saman
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve


plt.style.use('seaborn-poster')

# function to integrate
F = lambda t, s: \
  np.dot(np.array([[0,1],[0,-9.8/s[1]]]),s)
  
def f(t, s, mu):
  '''
  Differentiates the state and STM
  '''
  
  # initialize an empty array for the derivative of f
  F   = np.zeros([42])
  
  # unpack the state vector
  x, y, z, dx, dy, dz = s[0], s[1], s[2], s[3], s[4], s[5]
  
  # distance to the primary and secondary
  r1  = np.sqrt((x     - mu)**2 + y**2 + z**2)
  r2  = np.sqrt((x + 1 - mu)**2 + y**2 + z**2)
  
  # differential equations of motion
  ddx = x + 2*dy - (1 - mu)*(x - mu)/r1**3 - mu*(x + 1 - mu)/r2**3
  ddy = y - 2*dx - (1 - mu)*y       /r1**3 - mu*y           /r2**3
  ddz =          - (1 - mu)*z       /r1**3 - mu*z           /r2**3
  
  # differential state vector
  F[0], F[1], F[2], F[3], F[4], F[5]  = dx, dy, dz, ddx, ddy, ddz
  
  # define A matrix (page 242 of Szebehely)
  A11 = 1 - (1 - mu)/r1**3 - mu/r2**3 + 3*(1 - mu)*(x - mu)**2/r1**5 + 3*mu*(x + 1 - mu)**2/r2**5
  A12 =                                 3*(1 - mu)*(x - mu)*y /r1**5 + 3*mu*(x + 1 - mu)*y /r2**5
  A13 =                                 3*(1 - mu)*(x - mu)*z /r1**5 + 3*mu*(x + 1 - mu)*z /r2**5
  A21 = A12
  A22 = 1 - (1 - mu)/r1**3 - mu/r2**3 + 3*(1 - mu)*y**2       /r1**5 + 3*mu*y**2           /r2**5
  A23 =                                 3*(1 - mu)*y*z        /r1**5 + 3*mu*y*z            /r2**5
  A31 = A13
  A32 = A23
  A33 =   - (1 - mu)/r1**3 - mu/r2**3 + 3*(1 - mu)*z**2       /r1**5 + 3*mu*z**2           /r2**5
  
  A      = np.array([[   0,     0,     0,     1,     0,     0],
                     [   0,     0,     0,     0,     1,     0],
                     [   0,     0,     0,     0,     0,     1],
                     [ A11,   A12,   A13,     0,     2,     0],
                     [ A21,   A22,   A23,    -2,     0,     0],
                     [ A31,   A32,   A33,     0,     0,     0]])
  
  # state transition matrix
  phi    = np.array([[ s[6],  s[7],  s[8],  s[9], s[10], s[11]],
                     [s[12], s[13], s[14], s[15], s[16], s[17]],
                     [s[18], s[19], s[20], s[21], s[22], s[23]],
                     [s[24], s[25], s[26], s[27], s[28], s[29]],
                     [s[30], s[31], s[32], s[33], s[34], s[35]],
                     [s[36], s[37], s[38], s[39], s[40], s[41]]])
  
  # update the STM
  STM    = A@phi
  
  # add STM elements to the state
  F[6::] = STM.reshape(1, 36)
  
  return F

# initial conditions
t_span = np.linspace(0, 5, 100)
y0 = 0
v0 = 25
t_eval = np.linspace(0, 5, 10)

# constants
tol    = 1e-6
mu     = 0.2

# initial position and velocity
x0     = -1.01
y0     = 0.0
z0     = 0.58
vx0    = 0.0
vy0    = 0.413250
vz0    = 0.0

# initial state
state0 = np.array([x0, y0, z0, vx0, vy0, vz0])

# initialize the STM with identity matrix reshaped to an array
phi0   = np.identity(6)
phi0   = phi0.reshape(1, 36)

# add STM to state
state0 = np.concatenate([state0, phi0[0]])

# initial time
t0     = 0.0

# time step
dt     = 0.00001

# final time
tf     = 1.3464*2
# tf = 3.39

# differential correction process
xdd    = 1
zdd    = 1
itr    = 0
states = []

# time span from t0 to tf in increments of dt
tspan = np.arange(t0, tf, dt)

def objective(v0):
    sol = solve_ivp(lambda t, y: f(t, y, mu), (t0, tf), state0,
                    t_eval = tspan, rtol = tol, atol = tol)
    y = sol.y[0]
    return y[-1] - y


def newton_method(function, x0, step_size, tolerance, mu):
    '''
    Determines the roots of a non-linear single variable function using 
    derivative estimation and Taylor Series Expansion
    Args:
        x0 - initial condition estimate
        tolerance - the tolerance for convergence
        step_size - determines the size of delta x
    '''
    
    tolerance = np.array([10**-12]*42)
    
    f0        = f(x0, mu)
    f1        = f(x0 + step_size, mu)
    residual  = abs(f0 - f1)
    print(residual)
    iteration = 1
    fx        = (f(x0 + step_size, mu) - f0) / step_size
    
    while abs(np.linalg.norm(residual)) > abs(np.linalg.norm(tolerance)):
        x1        = x0 + ((0 - f0)/fx)
        f1        = f(x1, mu)
        residual  = abs(f1)
        f0        = f1
        x0        = x1
        fx        = (f(x0 + step_size, mu) - f0) / step_size
        iteration = iteration + 1
        print(iteration)
        
    return x1

# x = newton_method(f, state0, 0.01, 1*10**-12, mu)

# use fsolve to correct the initial guess
# v0 = fsolve(objective, state0)

# use solve_ivp witht he corrected initial conditions to integrate over time
sol = solve_ivp(lambda t, y: f(t, y, mu), (t0, tf), x,
                t_eval = tspan, rtol = tol, atol = tol)

# plt.plot(sol.y[0], sol.y[1])
# plt.xlabel('time (s)')
# plt.ylabel('altitude (m)')
# plt.title(f'root finding v={v0} m/s')
# plt.show()
