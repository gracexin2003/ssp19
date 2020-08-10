# Ephemeris Generation
# Math Pset 5
# Grace Xin
# Due 7/8/19
from math import *
import numpy as np

def findQuadrant(sine, cosine):
    if cosine > 0 and sine > 0: #1
        return asin(sine)

    if cosine < 0 and sine > 0: #2
        return acos(cosine)

    if cosine < 0 and sine < 0: #3
        return pi - asin(sine)

    if cosine > 0 and sine < 0: #4
        return 2*pi + asin(sine)

def RA_DEC(e, a, I, OM, om, M0, t0, t):
    ##print(e)
    ##print(a)
    # 1: calculate n
    sqrtMu = 0.01720209895
    mu = sqrtMu**2
    
    ##print("mu: " + str(mu))
    
    n = sqrt(mu/(a**3))
    
    ##print("n: " + str(n))
    
    # 2: calculate M
    M = M0 + n*(t-t0)
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
    ##print("E: " + str(E))

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
    ##print("ecliptic: " + str(ecliptic))

    # 6: repeat (1) - (5) for Earth to get
    Xe  = -2.027873566936922*10**-1
    Ye  =  9.963238789875005*10**-1
    Ze  = -4.453100906916791*10**-5
    R = [Xe, Ye, Ze]
    VXe = -1.658517789076126*10**-2
    VYe = -3.369780157118222*10**-3
    VZe =  8.777197100513998*10**-7
    Rdot = [VXe, VYe, VZe]
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
    
RA, DEC = (RA_DEC(0.6587595515873473,
       3.092704185336301,
       radians(11.74759129647092),
       radians(82.15763948051409),
       radians(356.34109239),
       radians(0.01246738682149958),
       2458465.5,
       2458668.5))
RA_deg = int(RA)
RA_arcmin = int((RA-RA_deg)*60)
RA_arcsec = ((RA-RA_deg)*60 - RA_arcmin)*60

print(" RA =  " + str(RA_deg) + ":" + str(RA_arcmin) + ":" + str(RA_arcsec))
DEC_deg = int(DEC)
DEC_arcmin = int((DEC-DEC_deg)*60)
DEC_arcsec = ((DEC-DEC_deg)*60 - DEC_arcmin)*60
DEC_str = str(DEC_deg) + ":" + str(DEC_arcmin) + ":" + str(DEC_arcsec)
if DEC>0:
    print("DEC = +" + DEC_str)
else:
    print("DEC = " + DEC_str)
