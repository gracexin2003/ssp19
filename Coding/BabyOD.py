# Baby OD code
# Grace Xin
# Due 7/13/19

import numpy as np
from math import *

def findQuadrant(sine, cosine):
    if cosine > 0 and sine > 0: #1
        return asin(sine)

    if cosine < 0 and sine > 0: #2
        return acos(cosine)

    if cosine < 0 and sine < 0: #3
        return pi - asin(sine)

    if cosine > 0 and sine < 0: #4
        return 2*pi + asin(sine)
    
def getOrbitalElements(r2, r2dot, t2):
    a = ((2 / np.linalg.norm(r2)) - np.dot(r2dot, r2dot))**(-1)

    e = sqrt(1 - (np.linalg.norm((np.cross(r2, r2dot)))**2 / a))
                            
    I = degrees(acos(h[2] / hNorm))

    OM = degrees(atan2((h[0]/(hNorm * sin(I))), (-h[1]/(hNorm * sin(I))))) % 360

    U = degrees(atan2((r2[2] / (r2Norm * sin(radians(I)))), ((r2[0] * cos(radians(OM)) + r2[1] * sin(radians(OM))) / r2Norm)))
    v = degrees(asin((((a * (1 - e**2)) / hNorm) * (np.dot(r2, r2dot) / r2Norm)) / e))
    w = U - v

    E = acos((1 / e) * (1 - (r2Norm / a)))
    M = 360 - degrees(E - e * sin(E))

    E = degrees(E)

    sqrtMu = .01720209895
    n = degrees(sqrtMu / (a**(3/2)))

    T = t2 - radians(M) / n

    P = 2 * pi / radians(n)
    
    return a, e, I, OM, w, M, E, n, T, P

'''
def getOrbitalElements(r2, r2dot, t2):
    sqrtMu = 0.01720209895
    mu = sqrtMu**2
    tau2 = sqrtMu*t2
    tau2dot = t2/sqrtMu
    epsilon = radians(23.4358)
    r2 = [r2[0], r2[1]*cos(epsilon)+r2[2]*sin(epsilon), -r2[1]*sin(epsilon)+r2[2]*cos(epsilon)]
    r2dot = [r2dot[0], r2dot[1]*cos(epsilon)+r2dot[2]*sin(epsilon), -r2dot[1]*sin(epsilon)+r2dot[2]*cos(epsilon)]
    #print(r2)
    #print(r2dot)
    h = np.cross(r2, r2dot)
    hNorm = np.linalg.norm(h)
    # 1: get a
    a = (2/np.linalg.norm(r2) - np.dot(r2dot,r2dot))**-1
    #print(a)
    # 2: get e
    e = sqrt(1-((np.linalg.norm(np.cross(r2, r2dot)))**2)/a)
    #print(e)
    # 3: get I
    I = degrees(acos(h[2]/hNorm))
    #print((degrees(I)+360)%360)
    # 4: get OM
    OM = degrees(atan2(h[0]/(hNorm*sin(I)), (-h[1]/(hNorm*sin(I))))) % 360
    #print((degrees(OM)+360)%360)
    # 5: get om
    sin_nu_om = r2[2]/(np.linalg.norm(r2)*sin(I))
    nu_om = findQuadrant(sin_nu_om, (1/cos(OM))*((r2[0]/np.linalg.norm(r2)+cos(I)*sin_nu_om*sin(OM))))
    cos_nu = (1/e)*((a*(1-e**2))/np.linalg.norm(r2)-1)
    sin_nu = (np.dot(r2, r2dot)/(e*np.linalg.norm(r2)))*sqrt(a*(1-e**2))
    nu = findQuadrant(sin_nu, cos_nu)
    om = nu_om-nu
    #print((degrees(om)+360)%360)
    # 6: get E, M, and n
    E = acos((1/e)*(1-np.linalg.norm(r2)/a))
    #print(360-degrees(E))
    M = E - e*sin(E)
    #print(360-degrees(M))
    n = sqrtMu/(a**1.5)
    #print(n)
    P = ((2*pi*a**1.5)/sqrtMu)/365.25
    T = t2-M/n
    return a, e, I, (degrees(OM)+360)%360, (degrees(om)+360)%360, degrees(M), degrees(E), n, P, T
'''
r2 = [-0.07246599652159999, -1.1366234814834766, 0.18273589584345765]
r2dot = [0.7711625417266533, -0.1689150502856931, 0.2513453276216283]
t2 = 2458304.74796
a, e, I, OM, om, M, E, n, P, T = getOrbitalElements(r2, r2dot, t2)
print("a = %f AU" %a)
print("e = %f" %e)
print("I = %f degrees" %I)
print("OM = %f degrees" %OM)
print("om = %f degrees" %om)
print("M = %f degrees" %M)
print("E = %f degrees" %E)
print("n = %f day" %n)
print("P = %f yrs" %P)
print("T = %f" %T)
