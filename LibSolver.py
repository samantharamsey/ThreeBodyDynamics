# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 15:50:37 2021

@author: saman
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


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
    g1   = (mu/(3*(1-mu)))**(1/3)
    g1   = newton_method(gamma1, g1, 1, tol, mu)
    x1   = 1-mu-g1
    
    # solve for the second collinear point (behind the second primary)
    g2   = (mu/(3*(1-mu)))**(1/3)
    g2   = newton_method(gamma2, g2, 1, tol, mu)
    x2   = 1-mu+g2
    
    # solve for the third collinear point (behind the first primary)
    g3   = (-7/12)*mu+1
    g3   = newton_method(gamma3, g3, 1, tol, mu)
    x3   = -mu-g3
    
    x4   = (1/2)-mu
    x5   = (1/2)-mu
    
    xvec = [x1, x2, x3, x4, x5]
    yvec = [0, 0, 0, np.sqrt(3)/2, -np.sqrt(3)/2]
    
    return xvec, yvec


def f(t, s, mu):
    '''
    Differentiates the state and STM
    '''
    
    # unpack the state vector
    x  = s[0]
    y  = s[1]
    z  = s[2]
    xd = s[3]
    yd = s[4]
    zd = s[5]
    
    d  = ((x+mu)**2    + y**2 + z**2)**(1/2)
    r  = ((x-1+mu)**2  + y**2 + z**2)**(1/2)
    
    if d > 100:
        print(aaa)
    
    # equations of motion
    Fxx = 1   - (1-mu)/d**3 + 3*(1-mu)*(x+mu)**2/d**5 - mu/r**3 + 3*mu*(x-1+mu)**2/r**5
    Fxy =                     3*(1-mu)*(x+mu)*y/d**5           + 3*mu*(x-1+mu)*y/r**5
    Fxz =                     3*(1-mu)*(x+mu)*z/d**5           + 3*mu*(x-1+mu)*z/r**5
    Fyx = Fxy
    Fyy = 1   - (1-mu)/d**3 + 3*(1-mu)*y**2/d**5      - mu/r**3 + 3*mu*y**2/r**5
    Fyz =                     3*(1-mu)*y*z/d**5                 + 3*mu*(x-1+mu)*z/r**5 
    Fzx = Fxz
    Fzy = Fyz
    Fzz =     - (1-mu)/d**3 + 3*(1-mu)*z**2/d**5      - mu/r**3 + 3*mu*z**2/r**5
    
    A   = [[   0,     0,     0,     1,     0,     0],
           [   0,     0,     0,     0,     1,     0],
           [   0,     0,     0,     0,     0,     1],
           [ Fxx,   Fxy,   Fxz,     0,     2,     0],
           [ Fyx,   Fyy,   Fyz,    -2,     0,     0],
           [ Fzx,   Fzy,   Fzz,     0,     0,     0]]
    
    # state transition matrix
    phi = [[ s[6],  s[7],  s[8],  s[9], s[10], s[11]],
           [s[12], s[13], s[14], s[15], s[16], s[17]],
           [s[18], s[19], s[20], s[21], s[22], s[23]],
           [s[24], s[25], s[26], s[27], s[28], s[29]],
           [s[30], s[31], s[32], s[33], s[34], s[35]],
           [s[36], s[37], s[38], s[39], s[40], s[41]]]
    
    phid   = np.matmul(A, phi).conj().T
    phidot = np.reshape(phid, (36,1))
    phidot = np.concatenate(phidot)
    
    # initialize an empty aray for the new state and STM matrix
    F = [0]*42
    
    # equations of motion
    F[0] = xd
    F[1] = yd
    F[2] = zd
    F[3] =  2*yd + x - (1-mu)*(x+mu)/d**3 - mu*(x-1+mu)/r**3
    F[4] = -2*xd + y - (1-mu)*y/d**3      - mu*y/r**3
    F[5] =           - (1-mu)*z/d**3      - mu*z/r**3
    
    # add STM to the velocity acceleration state matrix
    F[6:42] = phidot[0:36]
    
    return F


def doubledot(s, mu):
    '''
    differentiates the state but not the STM
    '''
    
    # unpack the state vector
    x  = s[0]
    y  = s[1]
    z  = s[2]
    xd = s[3]
    yd = s[4]
    zd = s[5]
    
    d  = ((x+mu)**2    + y**2 + z**2)**(1/2)
    r  = ((x-1+mu)**2 + y**2 + z**2)**(1/2)
    
    # initialize an empty aray for the new state and STM matrix
    F = [0]*6
    
    # equations of motion
    F[0] = xd
    F[1] = yd
    F[2] = zd
    F[3] =  2*yd + x - (1-mu)*(x+mu)/d**3 - mu*(x-1+mu)/r**3
    F[4] = -2*xd + y - (1-mu)*y/d**3      - mu*y/r**3
    F[5] =           - (1-mu)*z/d**3      - mu*z/r**3
    
    return F


def jacobic(s, mu):
    '''
    calculates the jacobian constant at a given state
    '''
    
    # unpack the state vector
    x  = s[0]
    y  = s[1]
    z  = s[2]
    xd = s[3]
    yd = s[4]
    zd = s[5]
    
    vsq   = xd**2 + yd**2 + zd**2
    d     = np.sqrt((x+mu)**2   + y**2 + z**2)
    r     = np.sqrt((x-1+mu)**2 + y**2 + z**2)
    U     = (1-mu)/d + mu/r
    Ustar = U + 1/2*(x**2 + y**2)
    jc    = 2*Ustar - vsq
    
    return jc


    

if __name__ == '__main__':
    
    tol     = 1*10**-13
    mu      = 0.012150584673413891
    tau     = 3.3981
    norbits = 100
    away    = 0.000001
    Delta_s = 0.05
    step    = 1
    
    x1, y1  = libsolver(mu, tol)
    z1      = [0, 0, 0, 0, 0]
    
    libpt   = 1
    x       = x1[libpt]
    y       = y1[libpt]
    z       = z1[libpt]
    
    d       = ((x+mu)**2   + y**2 + z**2)**(1/2)
    r       = ((x-1+mu)**2 + y**2 + z**2)**(1/2)
    
    Fxx     = 1 - (1-mu)/d**3 + 3*(1-mu)*(x+mu)**2/d**5 - mu/r**3 + 3*mu*(x-1+mu)**2/r**5
    Fyy     = 1 - (1-mu)/d**3 + 3*(1-mu)*y**2/d**5      - mu/r**3 + 3*mu*y**2/r**5
    
    beta1   = 2 - (Fxx+Fyy)/2
    beta2   = (-Fxx/Fyy)**(1/2)
    s       = (beta1 + (beta1**2+beta2**2)**(1/2))**(1/2)
    beta3   = (s**2 + Fxx)/(2*s)
    
    x0      = x1[libpt] + away
    y0      = 0.0
    z0      = 0.0
    xd0     = 0.0
    yd0     = -beta3*away*s
    zd0     = 0.0
    
    # define the set of inital conditions
    # state0  = [x0, y0, z0, xd0, yd0, zd0]
    state0  = [1.1762, 0, 0, 0, -0.1235, 0]
    phi0    = np.reshape(np.identity(6), (1, 36)).tolist()
    ic      = state0 + phi0[0]
    ts      = (0, tau*1)
    
    # integrate F to get state at each time step
    sol     = solve_ivp(lambda t, y: f(t, y, mu), ts, ic, rtol = tol, atol = tol)
    plt.plot(sol.y[0], sol.y[1])
    
    # at every state, output jacobian constant - should stay constant
    jc      = jacobic(sol.y, mu)
    # plt.plot(jc)
    
    ic      = []
    for i in range(42):
        ic.append(sol.y[i][-1])
    tf      = sol.t[-1]
    i       = 0
    error   = [1, 1, 1]
    print(np.array(ic).T)
    # while np.linalg.norm(error) > tol:
    for j in range(1): 
        print(j)
        
        ytarg   = 0
        xdtarg  = 0
        ydtarg  = 0
        zdtarg  = 0
        
        # target perpandicular crossing
        target  = [ytarg, xdtarg, zdtarg]
        
        ts      = (0, tf)
        sol1    = solve_ivp(lambda t, y: f(t, y, mu), ts, ic, rtol = 1e-10, atol = 1e-10)
        t, s    = sol1.t, sol1.y
        # plt.plot(sol.y[0], sol.y[1])
        # get the final state vector
        trajend = [s[0][-1], s[1][-2], s[2][-1], s[3][-1], s[4][-1], s[5][-1]]
        # print('state: ', trajend)
        # print('time: ', t[-1])
        
        # evaluate EOM to get acceleration components
        doubdot = doubledot(trajend, mu)
        # print(doubdot)
        xdd     = doubdot[3]
        zdd     = doubdot[5]
        
        # get STM from last 36 elements of s
        temp    = s[6::]
        phi     = []
        for i in temp:
            phi.append(i[-1])
        phi     = np.reshape(phi, (6, 6))
        
        # correction matrix
        phisub  =  [[phi[1, 2], phi[1, 4], trajend[4]],
                    [phi[3, 2], phi[3, 4],        xdd],
                    [phi[5, 2], phi[5, 4],        zdd]]
        
        endst   = [trajend[1], trajend[3], trajend[5]]
        error   = [target[0] - endst[0], target[1] - endst[1], target[2] - endst[2]]
        
        update  = np.linalg.inv(np.array(phisub))@np.array(error)
        ic[2]   = ic[2] + update[0]
        ic[4]   = ic[4] + update[1]
        tf      =    tf + update[2]
        print('ic = ', np.array(ic).T)
        
    #     # i = i + 1
    
    
    
    
    
    
    
    
    
    
    
    