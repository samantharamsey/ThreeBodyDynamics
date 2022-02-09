# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 10:15:27 2021

@author: saman
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def gamma1(g, mu):
    '''
    Equation of motion for Collinear Libration point 1
    '''
    gamma = g**5 - (3-mu)*g**4 + (3-2*mu)*g**3 - mu*g**2 + 2*mu*g - mu
    return gamma


def gamma2(g, mu):
    '''
    Equation of motion for Collinear Libration point 2
    '''
    gamma = g**5 + (3-mu)*g**4 + (3-2*mu)*g**3 - mu*g**2 + 2*mu*g - mu
    return gamma


def gamma3(g, mu):
    '''
    Equation of motion for Collinear Libration point 3
    '''
    gamma = g**5 + (2+mu)*g**4 + (1 + 2*mu)*g**3 - (1-mu)*g**2 - 2*(1-mu)*g - (1-mu)
    return gamma


def newton_method(function, x0, step_size, tolerance, mu):
    '''
    Determines the roots of a non-linear single variable function using 
    derivative estimation and Taylor Series Expansion
    Args:
        x0 - initial condition estimate
        tolerance - the tolerance for convergence
        step_size - determines the size of delta x
    '''
    
    f0        = function(x0, mu)
    residual  = abs(f0)
    iteration = 1
    fx        = (function(x0 + step_size, mu) - f0) / step_size
    
    while residual > tolerance:
        x1        = x0 + ((0 - f0)/fx)
        f1        = function(x1, mu)
        residual  = abs(f1)
        f0        = f1
        x0        = x1
        fx        = (function(x0 + step_size, mu) - f0) / step_size
        iteration = iteration + 1
        
    return x1


def libsolver(mu, tol):
    '''
    Solves for the Libration points for a given system
    '''
    
    # solve for the first collinear point (between the primaries)
    g1   = (mu/(3*(1 - mu)))**(1/3)
    g1   = newton_method(gamma1, g1, 1, tol, mu)
    x1   = 1 - mu - g1
    
    # solve for the second collinear point (behind the second primary)
    g2   = (mu/(3*(1 - mu)))**(1/3)
    g2   = newton_method(gamma2, g2, 1, tol, mu)
    x2   = 1 - mu + g2
    
    # solve for the third collinear point (behind the first primary)
    g3   = (-7/12)*mu + 1
    g3   = newton_method(gamma3, g3, 1, tol, mu)
    x3   = -mu - g3
    
    x4   = (1/2) - mu
    x5   = (1/2) - mu
    
    xvec = [x1, x2, x3, x4, x5]
    yvec = [0, 0, 0, np.sqrt(3)/2, -np.sqrt(3)/2]
    
    return xvec, yvec

def eventfunc(t, y, mu):
    '''
    Event function to stop integration
    '''
    
    return y[1]

event = lambda t, y: eventfunc(t, y, mu)
event.terminal = True
event.direction = 1

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


if __name__ == '__main__':
    
    # constants
    tol    = 1e-6
    mu     = 0.2
    # mu     = 0.012150585
    R1     = 0.03
    R2     = 0.01
    
    # initial position and velocity
    x0     = -1.01
    y0     = 0.0
    z0     = 0.58
    vx0    = 0.0
    vy0    = 0.413250
    vz0    = 0.0
    # x0 = 1.1745
    # y0 = 0
    # z0 = 0
    # vx0 = 0
    # vy0 = -0.1231
    # vz0 = 0
    # [-1.0144607, 0, 0.58761728, 0, 0.41325, 0]
    
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
    
    while xdd > tol or zdd > tol:
        
        # time span from t0 to tf in increments of dt
        tspan = np.arange(t0, tf, dt)
        
        # integrate the state and STM
        sol   = solve_ivp(lambda t, y: f(t, y, mu), (t0, tf), state0,
                          t_eval = tspan, rtol = tol, atol = tol)
        
        # unpack the integration results
        t        = sol.t
        s        = sol.y
        t_events = sol.t_events
        y_events = sol.y_events
        
        # get final state + STM
        yf = []
        for i in range(42):
            yf.append(s[i][-1])
        
        x, y, z, dx, dy, dz = yf[0], yf[1], yf[2], yf[3], yf[4], yf[5]
        
        # calculate correction values
        r1 = np.sqrt((x     - mu)**2 + y**2 + z**2)
        r2 = np.sqrt((x + 1 - mu)**2 + y**2 + z**2)
        
        ddx = x + 2*dy - (1 - mu)*(x - mu)/r1**3 - mu*(x + 1 - mu)/r2**3
        ddz =          - (1 - mu)*z       /r1**3 - mu*z           /r2**3
        
        # define remaining state values
        xdd, ydd, zdd = yf[3], yf[4], yf[5]
        phi21, phi23  = yf[12], yf[14]
        phi41, phi43  = yf[24], yf[26]
        phi61, phi63  = yf[36], yf[38]
        
        # create vector
        df = np.array([[xdd, zdd]]).T
        
        # calculate corrections assuming y velocity component is correct
        mat1 = np.array([[phi41, phi43],
                          [phi61, phi63]])
        mat2 = np.array([[  ddx,   ddz]]).T
        mat3 = np.array([[phi21, phi23]])
        mult = (1/ydd)*mat2@mat3
        F = np.linalg.inv(mat1 - mult)@df
        
        states.append(s)
        
        # correct initial conditions
        state0[0] = state0[0] - F[0][0]
        state0[2] = state0[2] - F[1][0]
        
        plt.plot(s[0], s[1])
        
        if itr > 100:
            break
        
        itr += 1
     
    # x, y, z = zip(*states)
    
    x, y, z = [], [], []
    for i in range(len(states)):
        # x, y, z = zip(*states[i])
        x.append(states[i][0])
        y.append(states[i][1])
        z.append(states[i][2])
        
    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    for i in range(itr):
        if i%4 == 0:
            plt.plot(x[i], y[i])
            # ax.plot(x[i],y[i],z[i])
    plt.show()
    
    
    # # determine the location of the systems libration points
    # libpoints = libsolver(mu, tol)
    # L1 = (libpoints[0][0], libpoints[1][0])
    # L2 = (libpoints[0][1], libpoints[1][1])
    # L3 = (libpoints[0][2], libpoints[1][2])
    # L4 = (libpoints[0][3], libpoints[1][3])
    # L5 = (libpoints[0][4], libpoints[1][4])
    
    # # determine the location of the primary bodies in the system
    # secondary = (mu       , 0, 0)
    # primary   = (-(1 - mu), 0, 0)

    # center = [primary, secondary]
    # radius = [R1, R2]
    
    
    # def axisEqual3D(ax):
    #     extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    #     sz = extents[:,1] - extents[:,0]
    #     centers = np.mean(extents, axis=1)
    #     maxsize = max(abs(sz))
    #     r = maxsize/2
    #     for ctr, dim in zip(centers, 'xyz'):
    #         getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)
        
        
    # def plt_sphere(center, radius):
    #    for c, r in zip(center, radius):
    #       ax = fig.gca(projection='3d')
    
    #       # draw sphere
    #       u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
    #       x1 = r*np.cos(u)*np.sin(v)
    #       y1 = r*np.sin(u)*np.sin(v)
    #       z1 = r*np.cos(v)
        
    #       # for i in range(itr):
    #       #    if i%4 == 0:
    #       #        ax.plot(x[i], y[i], z[i])
            
    #       ax.plot_surface(x1 - c[0], y1 - c[1], z1 - c[2], color=np.random.choice(['g','b']), alpha=0.5*np.random.random()+0.5)
    #       # ax.set_box_aspect((np.ptp(          x), np.ptp(y), np.ptp(z)))
    
          # ax.plot(libpoints[0][0], libpoints[1][0])
          # ax.plot(libpoints[0][1], libpoints[1][1])
          # ax.plot(libpoints[0][2], libpoints[1][2])
          # ax.plot(libpoints[0][3], libpoints[1][3])
          # ax.plot(libpoints[0][4], libpoints[1][4])
        # ax.axes.set_xlim3d(left=-0.5, right=1)
        # ax.axes.set_ylim3d(bottom=-0.75, top=0.75)
        # ax.axes.set_zlim3d(bottom=-0.75, top=0.75)
        # ax.margins(0.0)
        # ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))
        # axisEqual3D(ax)
    # ax.plot(L1)
    # ax.plot(L2)
    # ax.plot(L3)
    # ax.plot(L4)
    # ax.plot(L5)
    
    # fig = plt.figure()
    # plt_sphere(center, radius) 



