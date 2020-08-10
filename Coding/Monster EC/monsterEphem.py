# Monster Ephem Gen.
# Grace Xin

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

# Ephemeris Generation
def RA_DEC(e, a, I, OM, om, M, ind, t, R):
    t0 = JDs[ind]
    
    sqrtMu = 0.01720209895
    mu = sqrtMu**2
    
    n = sqrtMu/(a**1.5)
    
    # calculate M
    M = M - n*(JDs[2]-t0)

    # calculate E
    prevGuess = M
    currGuess = M
    while True:
        currGuess = M+e*sin(prevGuess)
        if(abs(currGuess-prevGuess) < 10**-12):
            break
        prevGuess = currGuess 
    E = currGuess

    # calculate physics x, y (z = 0)
    z = 0
    x = a*(cos(E)-e)
    y = a*sin(E)*sqrt(1-e**2)
    r = sqrt(x**2+y**2)
    phys_coords = [x, y, z]
    cosnu = x/r
    sinnu = y/r
    nu = findQuadrant(sinnu, cosnu)
    
    # calculate ecliptic
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
    
    # repeat (1) - (5) for Earth to get
    Xe = R[0]
    Ye = R[1]
    Ze = R[2]

    # get Earth to asteroid rho
    rho = ecliptic + R
    rho_x = ecliptic[0] + Xe
    rho_y = ecliptic[1] + Ye
    rho_z = ecliptic[2] + Ze

    # convert ecliptic rho's to equatorial rho's
    EQ_rho_x = rho_x
    epsolon = radians(23.4358)
    EQ_rho_y = rho_y*cos(epsolon) - rho_z*sin(epsolon)
    EQ_rho_z = rho_y*sin(epsolon) + rho_z*cos(epsolon)
    EQ_rho = [EQ_rho_x, EQ_rho_y, EQ_rho_z]
    EQ_mag = sqrt(EQ_rho_x**2+EQ_rho_y**2+EQ_rho_z**2)
    for i in range(len(EQ_rho)):
        EQ_rho[i] /= EQ_mag

    # calculate RA and DEC
    DEC = asin(EQ_rho[2])
    sinRA = EQ_rho[1]/cos(DEC)
    cosRA = EQ_rho[0]/cos(DEC)
    RA = findQuadrant(sinRA, cosRA)

    return degrees(RA)/15, degrees(DEC)

e, a, I, OM, om, M = [], [], [], [], [], []
Es, As, Is, Os, Ws, Ms = open("egrace.txt"), open("agrace.txt"), open("Igrace.txt"), open("Ograce.txt"), open("wgrace.txt"), open("Mgrace.txt"), 
for line in Es:
    trimmed = line.strip()
    e.append(float(trimmed))
for line in As:
    trimmed = line.strip()
    a.append(float(trimmed))
for line in Is:
    trimmed = line.strip()
    I.append(float(trimmed))
for line in Os:
    trimmed = line.strip()
    OM.append(float(trimmed))
for line in Ws:
    trimmed = line.strip()
    om.append(float(trimmed))
for line in Ms:
    trimmed = line.strip()
    M.append(float(trimmed))
#print(e, a, I, OM, om, M)
RAs, DECs, JDs = [], [], [] #7 each
Rs = np.array([[0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0]]) #7 by 3 array

data = open("Input.txt")
lineN = 0
for line in data:
    values = line.split()
    Y = int(values[0])
    m = int(values[1])
    D = int(values[2])
    t = values[3]
    timeParts = t.split(":")
    UT = float(timeParts[0])+float(timeParts[1])/60+float(timeParts[2])/3600
    J0 = 367*Y-int((7/4)*(Y+int((m+9)/12)))+int(275*m/9)+D+1721013.5
    JD = J0 + UT/24
    JDs.append(JD)
    ra = float(values[4])+float(values[5])/60+float(values[6])/3600
    RAs.append(radians(ra*15))
    if float(values[7]) > 0:
        dec = float(values[7])+float(values[8])/60+float(values[9])/3600
    else:
        dec = float(values[7])-float(values[8])/60-float(values[9])/3600
    DECs.append(radians(dec))
    A = float(values[10])
    B = float(values[11])
    C = float(values[12])
    Rs[lineN][0] = A
    Rs[lineN][1] = B
    Rs[lineN][2] = C
    lineN += 1
#print(RAs, DECs, JDs)
#print(Rs)

cE, cA, cI, cO, cW, cM = 0, 0, 0, 0, 0, 0
minErr = 10**31
for index in range(len(e)):
    currE, currA, currI, currO, currW, currM = e[index], a[index], I[index], OM[index], om[index], M[index]
    t = 2458685.75
    fitRAs, fitDECs = [], []
    for ind in range(0, 7):
        ra, dec, jd = RAs[ind], DECs[ind], JDs[ind]
        R = Rs[ind]
        RA, DEC = RA_DEC(currE, currA, currI, currO, currW, currM, ind, t, R)
        fitRAs.append(RA)
        fitDECs.append(DEC)
    alphaSq, deltaSq = 0, 0
    for ind in range(len(RAs)):
        alphaSq += (RAs[ind] - fitRAs[ind])**2
        deltaSq += (DECs[ind] - fitDECs[ind])**2
    error = sqrt((alphaSq+deltaSq)/(14 - 6))
    if(error < minErr):
        minErr = error
        cE, cA, cI, cO, cW, cM = currE, currA, currI, currO, currW, currM
print("a = %f AU" %cA)
print("e = %f" %cE)
print("I = %f degrees" %cI)
print("OM = %f degrees" %cO)
print("om = %f degrees" %cW)
print("M = %f degrees" %cM)
