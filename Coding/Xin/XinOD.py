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
    #print(360-degrees(M))
    n = sqrtMu/(a**1.5)
    #print(n)
    P = ((2*pi*a**1.5)/sqrtMu)/365.25
    T = t2-M/n
    M0 = M + n*abs(t0-t2)
    #print(M0)
    return a, e, I, OM, om, M0, E, n, P, T

#---------------------------------------------------------------------------------------------------------------------------------------------#

# import file with 3 RAs, DECs, and times
data = open("XinInput.txt")
lines = []
for line in data:
    lines.append(line)
# there are 4 inputs, so split them up into 2 sets of 3 inputs
for LINEn in range(1, 3):
    print("obs 1,",(4-LINEn),"and 4")
    lineN = 0
    RAs, DECs, JDs, taus = [], [], [], []
    Rs = np.array([[0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0]])
    for line in lines:
        if lineN == LINEn:
            lineN += 1
            continue
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
        if lineN < LINEn:
            Rs[lineN][0] = a
            Rs[lineN][1] = b
            Rs[lineN][2] = c
        else:
            Rs[lineN-1][0] = a
            Rs[lineN-1][1] = b
            Rs[lineN-1][2] = c
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
    
    # because we are in ecliptic, mu = 1
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
            ##print(JDs)
            tau1, tau3 = k*(JDs[0]-JDs[1]), k*(JDs[2]-JDs[1])
            tau = tau3-tau1
            ##print(tau1, tau3, tau)

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
        print("r2:", r2)
        print("r2dot:", r2dot)
        print("Range to asteroid for central obs:", rho2)
        
        # Run Baby OD
        a, e, I, OM, om, M0, E, n, P, T = getOrbitalElements(r2, r2dot, JDs[1], 2458685.75)
        Ideg = (degrees(I)+360)%360
        OMdeg = (degrees(OM)+360)%360
        omdeg = (degrees(om)+360)%360
        Mdeg = (degrees(M0)+360)%360
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
        print()
        
