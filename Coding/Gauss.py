# Grace Xin
# Method of Gauss OD Code
# Due 7/19/2019

import numpy as np
from math import *
import importlib

# converts an equatorial vector to an ecliptic vector
def toEcliptic(arr):
    epsilon = radians(23.4358)
    transformation = [[1,            0,             0],
                      [0, cos(epsilon),  sin(epsilon)],
                      [0, -sin(epsilon), cos(epsilon)]]
    return np.matmul(transformation, arr)

#---------------------------------------------------------------------------------------------------------------------------------------------#

def NewtonMethod(deltaT, a, n, r2M, r2dotM):
    deltaEguess = n*deltaT
    ##print(deltaEguess)
    x = deltaEguess
    fx = x - (1-r2M/a)*sin(x) + (r2M*r2dotM)*(1-cos(x))/(n*a**2) - n*deltaT
    fpx = 1 - (1-r2M/a)*cos(x) + (r2M*r2M)*sin(x)/(n*a**2)
    prevX = x
    x -= fx/fpx
    while abs(prevX-x) > 10**-9:
        fx = x - (1-r2M/a)*sin(x) + (r2M*r2dotM)*(1-cos(x))/(n*a**2) - n*deltaT
        fpx = 1 - (1-r2M/a)*cos(x) + (r2M*r2M)*sin(x)/(n*a**2)
        prevX = x
        x -= fx/fpx
    deltaE = x
    return deltaE

#---------------------------------------------------------------------------------------------------------------------------------------------#

def findQuadrant(sine, cosine):
    if cosine > 0 and sine > 0: #1
        return asin(sine)

    if cosine < 0 and sine > 0: #2
        return acos(cosine)

    if cosine < 0 and sine < 0: #3
        return pi - asin(sine)

    if cosine > 0 and sine < 0: #4
        return 2*pi + asin(sine)

# BabyOD
def getOrbitalElements(r2, r2dot, t0, t2):
    sqrtMu = 0.01720209895
    mu = sqrtMu**2
    tau2 = sqrtMu*t2
    tau2dot = t2/sqrtMu
    epsilon = radians(23.4358)
    #r2 = [r2[0], r2[1]*cos(epsilon)+r2[2]*sin(epsilon), -r2[1]*sin(epsilon)+r2[2]*cos(epsilon)]
    #r2dot = [r2dot[0], r2dot[1]*cos(epsilon)+r2dot[2]*sin(epsilon), -r2dot[1]*sin(epsilon)+r2dot[2]*cos(epsilon)]
    #print(r2)
    #print(r2dot)
    h = np.cross(r2, r2dot)
    #print(h)
    # 1: get a
    a = (2/np.linalg.norm(r2) - np.dot(r2dot,r2dot))**-1
    #print(a)
    # 2: get e
    e = (1-((np.linalg.norm(np.cross(r2, r2dot)))**2)/a)**0.5
    #print(e)
    # 3: get I
    I = acos(h[2]/np.linalg.norm(h))
    #print((degrees(I)+360)%360)
    # 4: get OM
    OM = findQuadrant(h[0]/(np.linalg.norm(h)*sin(I)), h[1]/(-np.linalg.norm(h)*sin(I)))
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
    #print(M)
    #print(360-degrees(M))
    n = sqrtMu/(a**1.5)
    #print(n)
    #print(n)
    P = ((2*pi*a**1.5)/sqrtMu)/365.25
    T = t2-M/n
    #print(t0-t2)
    M0 = M + n*abs(t0-t2)
    #print(M0)
    return a, e, I, OM, om, M0, M, E, n, P, T

#---------------------------------------------------------------------------------------------------------------------------------------------#

def RA_DEC(e, a, I, OM, om, M, ind, t, R): # M0
    t0 = JDs[ind]
    #I = radians(I)
    #OM = radians(OM)
    #om = radians(om)
    #M0 = radians(M0)
    
    ##print(e)
    ##print(a)
    # 1: calculate n
    
    ###sqrtMu = 0.01720209895
    ###sqrtMu = 1
    sqrtMu = 0.01720209895
    mu = sqrtMu**2
    
    ##print("mu: " + str(mu))
    
    n = sqrtMu/(a**1.5)
    
    ##print("n: " + str(n))
    
    # 2: calculate M
    #####M = M0 + n*abs(t-t0)
    M = M - n*(JDs[1]-t0)
    ##print(M0)
    ##print("M: " + str(M))

    # 3: calculate E
    prevGuess = M
    currGuess = M
    while True:
        currGuess = M+e*sin(prevGuess)
        if(abs(currGuess-prevGuess) < 10**-12):
            break
        prevGuess = currGuess
        
    E = currGuess

    # 4: calculate physics x, y (z = 0)
    z = 0
    x = a*(cos(E)-e)
    y = a*sin(E)*sqrt(1-e**2)
    r = sqrt(x**2+y**2)
    phys_coords = [x, y, z]
    cosnu = x/r
    sinnu = y/r
    nu = findQuadrant(sinnu, cosnu)
    ##print(nu)
    ##print("r: " + str(r) + "; " + str(phys_coords))
    ##print(x)
    ##print(y)
    ##print(z)
    
    # 5: calculate ecliptic
    R1 = np.array([[cos(om), -sin(om), 0],
                   [sin(om), cos(om),  0],
                   [0,       0,        1]])
    R2 = np.array([[1,      0,      0 ],
                   [0, cos(I), -sin(I)],
                   [0, sin(I), cos(I) ]])
    R3 = np.array([[cos(OM), -sin(OM), 0],
                   [sin(OM), cos(OM),  0],
                   [0,       0,        1]])
    ecliptic = np.matmul(R3, np.matmul(R2, np.matmul(R1, phys_coords)))
    #ecliptic = phys_coords
    ##print("ecliptic: " + str(ecliptic))
    
    # 6: repeat (1) - (5) for Earth to get
    ###Xe  = -2.027873566936922*10**-1
    ###Ye  =  9.963238789875005*10**-1
    ###Ze  = -4.453100906916791*10**-5
    Xe = R[0]
    Ye = R[1]
    Ze = R[2]
    ###VXe = -1.658517789076126*10**-2
    ###VYe = -3.369780157118222*10**-3
    ###VZe =  8.777197100513998*10**-7
    ###Rdot = [VXe, VYe, VZe]
    ##print(R[0]**2+R[1]**2+R[2]**2)
    ##print("R: " + str(R))
    ##print("Rdot: " + str(Rdot))

    # 7: get Earth to asteroid rho
    rho = ecliptic + R
    rho_x = ecliptic[0] + Xe
    rho_y = ecliptic[1] + Ye
    rho_z = ecliptic[2] + Ze
    ##print("rho: " + str(rho))
    ##print(rho_x)
    ##print(rho_y)
    ##print(rho_z)

    # 8: convert ecliptic rho's to equatorial rho's
    EQ_rho_x = rho_x
    epsolon = radians(23.4358)
    ##print(rho_y*cos(epsolon))
    ##print(rho_z*sin(epsolon))
    EQ_rho_y = rho_y*cos(epsolon) - rho_z*sin(epsolon)
    ##print(rho_y*sin(epsolon))
    ##print(rho_z*cos(epsolon))
    ##print(rho_y*sin(epsolon)+rho_z*cos(epsolon))
    EQ_rho_z = rho_y*sin(epsolon) + rho_z*cos(epsolon)
    EQ_rho = [EQ_rho_x, EQ_rho_y, EQ_rho_z]
    ##print("EQ_rho: " + str(EQ_rho))
    EQ_mag = sqrt(EQ_rho_x**2+EQ_rho_y**2+EQ_rho_z**2)
    for i in range(len(EQ_rho)):
        EQ_rho[i] /= EQ_mag
    ##print("EQ_rho (norm): " + str(EQ_rho))

    # 9: calculate RA and DEC
    DEC = asin(EQ_rho[2])
    sinRA = EQ_rho[1]/cos(DEC)
    cosRA = EQ_rho[0]/cos(DEC)
    ##print(cosRA)
    ##print(cos(DEC))
    RA = findQuadrant(sinRA, cosRA)

    return degrees(RA)/15, degrees(DEC)

#---------------------------------------------------------------------------------------------------------------------------------------------#

# import file with 3 RAs, DECs, and times
data = open("GaussInput.txt")
lines = []
for line in data:
    lines.append(line)
# there are 4 inputs, so split them up into 2 sets of 3 inputs
lineN = 0
RAs, DECs, JDs, taus = [], [], [], []
Rs = np.array([[0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0]])
for line in lines:
    ##print("in")
    values = line.split()
    ##print(values[0] + " " + values[1] + " " + values[2] + " " + values[3])
    Y = int(values[0])
    M = int(values[1])
    D = int(values[2])
    ##print(Y, M, D)
    t = values[3]
    timeParts = t.split(":")
    ##print(timeParts)
    UT = float(timeParts[0])+float(timeParts[1])/60+float(timeParts[2])/3600
    ##print(UT)
    J0 = 367*Y-int((7/4)*(Y+int((M+9)/12)))+int(275*M/9)+D+1721013.5
    ##print(J0)
    JD = J0 + UT/24
    JDs.append(JD)
    ra = float(values[4])+float(values[5])/60+float(values[6])/3600
    RAs.append(radians(ra*15))
    if float(values[7]) > 0:
        dec = float(values[7])+float(values[8])/60+float(values[9])/3600
    else:
        dec = float(values[7])-float(values[8])/60-float(values[9])/3600
    DECs.append(radians(dec))
    a = float(values[10])
    b = float(values[11])
    c = float(values[12])
    ##print(lineN)
    Rs[lineN][0] = a
    Rs[lineN][1] = b
    Rs[lineN][2] = c
    lineN += 1
##print(RAs, "\n", DECs, "\n", JDs)
##print(Rs)
origJD = JDs.copy()

# calculae rho_i_hat's

rho1hat = [cos(RAs[0])*cos(DECs[0]),
           sin(RAs[0])*cos(DECs[0]),
           sin(DECs[0])]
rho2hat = [cos(RAs[1])*cos(DECs[1]),
           sin(RAs[1])*cos(DECs[1]),
           sin(DECs[1])]    
rho3hat = [cos(RAs[2])*cos(DECs[2]),
           sin(RAs[2])*cos(DECs[2]),
           sin(DECs[2])]     
##print(rho1hat, "\n", rho2hat, "\n", rho3hat)
rho1hat = toEcliptic(rho1hat)
rho2hat = toEcliptic(rho2hat)
rho3hat = toEcliptic(rho3hat)
##print(rho1hat, "\n", rho2hat, "\n", rho3hat)

# calculate tau's
k = 0.01720209895
tau1, tau3 = k*(JDs[0]-JDs[1]), k*(JDs[2]-JDs[1])
tau = tau3-tau1
##print(tau1, "\n", tau3, "\n", tau)

# calculate D
D11, D12, D13 = np.dot(np.cross(Rs[0], rho2hat), rho3hat), np.dot(np.cross(Rs[1], rho2hat), rho3hat), np.dot(np.cross(Rs[2], rho2hat), rho3hat)
D21, D22, D23 = np.dot(np.cross(rho1hat, Rs[0]), rho3hat), np.dot(np.cross(rho1hat, Rs[1]), rho3hat), np.dot(np.cross(rho1hat, Rs[2]), rho3hat)
D31, D32, D33 = np.dot(rho1hat, np.cross(rho2hat, Rs[0])), np.dot(rho1hat, np.cross(rho2hat, Rs[1])), np.dot(rho1hat, np.cross(rho2hat, Rs[2]))
D0 = np.dot(rho1hat, np.cross(rho2hat, rho3hat))
##print(D11, D12, D13, "\n", D21, D22, D23, "\n", D31, D32, D33, "\n", D0)

# calculate A1, A2, B1, B2
A1 = tau3/tau
B1 = (A1/6)*(tau**2 - tau3**2)
A3 = -tau1/tau
B3 = (A3/6)*(tau**2 - tau1**2)
##print(A1, A3, B1, B3)

# calculate A and B
A = (A1*D21 - D22 + A3*D23)/(-D0)
B = (B1*D21 + B3*D23)/(-D0)
# calculate E and F
E = -2*np.dot(rho2hat, Rs[1])
F = np.linalg.norm(Rs[1])**2
# define mu = G*M(sun)
'''
G = 6.67*10**-11
mSun = 1.989*10**30
mu = G*mSun
'''
mu = 1
# calculate a, b, c
a = -(A**2+A*E+F)
b = -mu*(2*A*B+B*E)
c = -mu**2*B**2
##print(A, B, "\n", E, F, "\n", a, b, c)

# calculate possible r2 values
roots = np.roots([1, 0, a, 0, 0, b, 0, 0, c]) #np.polynomial.polynomial.polyroots([c, 0, 0, b, 0, 0, a, 0, 1])
r2s = []
for r2 in roots:
    if r2 == abs(r2) and "0j" in str(r2):
        r2s.append(np.real(r2))
##print("r2s:",r2s)

# loop thorugh real r2 values
for r2mag in r2s:
    print("root:", r2mag)
    # use truncated taylor series to find f and g, iterating for 1 and 3
    f1 = 1 - (mu*tau1**2)/(2*r2mag**3)
    g1 = tau1 - (mu*tau1**3)/(6*r2mag**3)
    f3 = 1 - (mu*tau3**2)/(2*r2mag**3)
    g3 = tau3 - (mu*tau3**3)/(6*r2mag**3)
    ##print(f1, f3, g1, g3)
    
    # get c1, c2, c3
    c1 = g3/(f1*g3-g1*f3)
    c2 = -1
    c3 = -g1/(f1*g3-g1*f3)
    ##print(c1, c2, c3)
    
    # calculate rho1, rho2, rho3
    rho1 = (c1*D11+c2*D12+c3*D13)/(c1*D0)
    rho2 = (c1*D21+c2*D22+c3*D23)/(c2*D0)
    rho3 = (c1*D31+c2*D32+c3*D33)/(c3*D0)
    ##print(rho1, rho2, rho3)
    
    # get guesses
    r1, r2, r3 = [], [], []
    for index in range(len(rho1hat)):
        r1.append(rho1*rho1hat[index] - Rs[0][index])
        r2.append(rho2*rho2hat[index] - Rs[1][index])
        r3.append(rho3*rho3hat[index] - Rs[2][index])
    r2mag = np.linalg.norm(r2)
    ##print("r2",r2)
    ##print(r1, "\n", r2, "\n", r3)
    
    # calculate d1 and d3
    d1 = -f3/(f1*g3-f3*g1)
    d3 = f1/(f1*g3-f3*g1)
    ##print(d1, d3)
    
    # calculate r2dot
    r2dot = []
    for index in range(len(r1)):
        r2dot.append(d1*r1[index]+d3*r3[index])
    ##print("r2dot",r2dot)

    # correct for light-travel time
    C = 2.99792548*10**8
    JDs[0] = origJD[0] - rho1/C
    JDs[1] = origJD[1] - rho2/C
    JDs[2] = origJD[2] - rho3/C
    ##print(JDs)
    tau1, tau3 = k*(JDs[0]-JDs[1]), k*(JDs[2]-JDs[1])
    tau = tau3-tau1
    ##print(tau1, tau3, tau)

    # get better f's and g's
    f1 = 1 - (mu*tau1**2)/(2*r2mag**3) + (mu*np.dot(r2, r2dot)*tau1**3)/(2*r2mag**5)
    f3 = 1 - (mu*tau3**2)/(2*r2mag**3) + (mu*np.dot(r2, r2dot)*tau3**3)/(2*r2mag**5)
    g1 = tau1 - (mu*tau1**3)/(6*r2mag**3)
    g3 = tau3 - (mu*tau3**3)/(6*r2mag**3)
    ##print(f1, f3, g1, g3)


    # ITERATE!!!

    
    c1 = g3/(f1*g3-g1*f3)
    c2 = -1
    c3 = -g1/(f1*g3-g1*f3)
    ##print(c1, c2, c3)
    
    rho1 = (c1*D11+c2*D12+c3*D13)/(c1*D0)
    rho2 = (c1*D21+c2*D22+c3*D23)/(c2*D0)
    rho3 = (c1*D31+c2*D32+c3*D33)/(c3*D0)
    ##print(rho1, rho2, rho3)

    prevR2mag = r2mag
    r1, r2, r3 = [], [], []
    for index in range(len(rho1hat)):
        r1.append(rho1*rho1hat[index] - Rs[0][index])
        r2.append(rho2*rho2hat[index] - Rs[1][index])
        r3.append(rho3*rho3hat[index] - Rs[2][index])
    r2mag = np.linalg.norm(r2)
    ##print(r2)
    ##print(r1, "\n", r2, "\n", r3)
    
    d1 = -f3/(f1*g3-f3*g1)
    d3 = f1/(f1*g3-f3*g1)
    ##print(d1, d3)
    
    r2dot = []
    for index in range(len(r1)):
        r2dot.append(d1*r1[index]+d3*r3[index])
    ##print(r2dot)
    count = 0
    while abs(prevR2mag-r2mag)>(10**-12):
        # correct for light-travel time
        JDs[0] = origJD[0] - rho1/C
        JDs[1] = origJD[1] - rho2/C
        JDs[2] = origJD[2] - rho3/C
        ####print(JDs)
        tau1, tau3 = k*(JDs[0]-JDs[1]), k*(JDs[2]-JDs[1])
        tau = tau3-tau1
        ####print(tau1, tau3, tau)

        # get better f's and g's
        f1 = 1 - (mu*tau1**2)/(2*r2mag**3) + (mu*np.dot(r2, r2dot)*tau1**3)/(2*r2mag**5)
        f3 = 1 - (mu*tau3**2)/(2*r2mag**3) + (mu*np.dot(r2, r2dot)*tau3**3)/(2*r2mag**5)
        g1 = tau1 - (mu*tau1**3)/(6*r2mag**3)
        g3 = tau3 - (mu*tau3**3)/(6*r2mag**3)
        ####print(f1, f3, g1, g3)
        # iterate
        
        c1 = g3/(f1*g3-g1*f3)
        c2 = -1
        c3 = -g1/(f1*g3-g1*f3)
        ####print(c1, c2, c3)
        
        rho1 = (c1*D11+c2*D12+c3*D13)/(c1*D0)
        rho2 = (c1*D21+c2*D22+c3*D23)/(c2*D0)
        rho3 = (c1*D31+c2*D32+c3*D33)/(c3*D0)
        ####print(rho1, rho2, rho3)

        prevR2mag = r2mag
        r1, r2, r3 = [], [], []
        for index in range(len(rho1hat)):
            r1.append(rho1*rho1hat[index] - Rs[0][index])
            r2.append(rho2*rho2hat[index] - Rs[1][index])
            r3.append(rho3*rho3hat[index] - Rs[2][index])
        r2mag = np.linalg.norm(r2)
        ##print(r2)
        ####print(r1, "\n", r2, "\n", r3)
        
        d1 = -f3/(f1*g3-f3*g1)
        d3 = f1/(f1*g3-f3*g1)
        ####print(d1, d3)
        
        r2dot = []
        for index in range(len(r1)):
            r2dot.append(d1*r1[index]+d3*r3[index])
        ###print(r2dot)
        count += 1
    print("Done in", count, "iterations")
    print(r2)
    print(r2dot)

    # Newton's Method, which makes my data worse :(
    '''
    r2M = np.linalg.norm(r2)
    r2dotM = np.linalg.norm(r2dot)
    ##print(r2M, r2dotM, "hi")
    
    # find a
    a = ((2/(r2M))-np.dot(r2dot, r2dot)/mu)**-1
    ##print(a)

    # find n
    n = sqrt(mu/(a**3))
    ##print(n)
    
    # use Newton's Method to find deltaE's
    deltaT = tau1
    deltaE1 = NewtonMethod(deltaT, a, n, r2M, r2dotM)
    ##print(deltaE1)
    deltaT = tau3
    deltaE3 =  NewtonMethod(deltaT, a, n, r2M, r2dotM)
    ##print(deltaE3)
    
    # find final fi and gi
    f1 = 1 - a*(1-cos(deltaE1))/r2M
    f3 = 1 - a*(1-cos(deltaE3))/r2M
    g1 = tau1 + (sin(deltaE1)-deltaE1)/n
    g3 = tau3 + (sin(deltaE3)-deltaE3)/n
    ##print(f1, f3, g1, g3)

    # update r2 and r2dot values
    c1 = g3/(f1*g3-g1*f3)
    c2 = -1
    c3 = -g1/(f1*g3-g1*f3)
    ##print(c1, c2, c3)
    
    rho1 = (c1*D11+c2*D12+c3*D13)/(c1*D0)
    rho2 = (c1*D21+c2*D22+c3*D23)/(c2*D0)
    rho3 = (c1*D31+c2*D32+c3*D33)/(c3*D0)
    ##print(rho1, rho2, rho3)

    prevR2mag = r2mag
    r1, r2, r3 = [], [], []
    for index in range(len(rho1hat)):
        r1.append(rho1*rho1hat[index] - Rs[0][index])
        r2.append(rho2*rho2hat[index] - Rs[1][index])
        r3.append(rho3*rho3hat[index] - Rs[2][index])
    r2mag = np.linalg.norm(r2)
    ##print(r2)
    ##print(r1, "\n", r2, "\n", r3)
    
    d1 = -f3/(f1*g3-f3*g1)
    d3 = f1/(f1*g3-f3*g1)
    ##print(d1, d3)
    
    r2dot = []
    for index in range(len(r1)):
        r2dot.append(d1*r1[index]+d3*r3[index])
    ##print(r2dot)
    print(r2)
    print(r2dot)
    '''

    #t0 = July 21, 2019 6:00 UT
    t0 = 2456809.5
    a, e, I, OM, om, M0, M, E, n, P, T = getOrbitalElements(r2, r2dot, JDs[1], 2458685.75)
    Ideg = (degrees(I)+360)%360
    OMdeg = (degrees(OM)+360)%360
    omdeg = (degrees(om)+360)%360
    Mdeg = (degrees(M0))%360
    Edeg = (360-degrees(E))%360
    ndeg = degrees(n)
    print("a = %f AU" %a)
    print("e = %f" %e)
    print("I = %f degrees" %Ideg)
    print("OM = %f degrees" %OMdeg)
    print("om = %f degrees" %omdeg)
    print("M = %f degrees" %Mdeg)
    print("E = %f degrees" %Edeg)
    print("n = %f degrees/day" %ndeg)
    print("P = %f years" %P)
    print("T = %f" %T)

    # Ephemeris generation, which doesn't work :(
    
    #RA, DEC = RA_DEC(e, a, I, OM, om, M0, JDs[1], 2458685.75)
    JDs.append(2458674.77332) ########################
    R = [-3.054941548263765E-01, 9.697136604922600E-01, -7.648306366634412E-05]
    RA, DEC = RA_DEC(e, a, I, OM, om, M, 3, 2458685.75, R)
    print(RA, DEC)
    RA_deg = int(RA)
    RA_arcmin = int((RA-RA_deg)*60)
    RA_arcsec = ((RA-RA_deg)*60 - RA_arcmin)*60

    print(" RA =  " + str(RA_deg) + ":" + str(RA_arcmin) + ":" + str(RA_arcsec))
    DEC_deg = int(DEC)
    DEC_arcmin = int((DEC-DEC_deg)*60)
    DEC_arcsec = ((DEC-DEC_deg)*60 - DEC_arcmin)*60
    DEC_str = str(DEC_deg) + ":" + str(abs(DEC_arcmin)) + ":" + str(abs(DEC_arcsec))
    if DEC>0:
        print("DEC = +" + DEC_str)
    else:
        print("DEC = " + DEC_str)
    
    print()
        
