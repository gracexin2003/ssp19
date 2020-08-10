# LSPR Code
# Grace Xin
# Due Friday July 5 2019
import numpy as np
from math import sqrt, degrees, radians, cos, sin, atan

def LSPR(x, y, sRA, sDEC, x_cent, y_cent, flatten): # arrays of: centroid coordinates, RA+DEC coordinates of reference stars
    #print(sRA)
    #print(sDEC)
    #print()
    if flatten:
        sRA, sDEC = flat(x, y, sRA, sDEC)
    #print(sRA)
    #print(sDEC)
    # calculating plate constants
    sRA_sum = sum(sRA)
    
    N = len(x) # = len(y) = len(sRA) = len(sDEC)
    x_sum = sum(x)
    y_sum = sum(y)
    
    sRA_x_sum = 0
    for i in range(len(x)):
        sRA_x_sum += x[i]*sRA[i]

    x_sq = x.copy()
    for i in range(len(x)):
        x_sq[i] *= x_sq[i]
    x_sq_sum = sum(x_sq)
    x_y_sum = 0
    for i in range(len(x)):
        x_y_sum += x[i]*y[i]

    sRA_y_sum = 0
    for i in range(len(y)):
        sRA_y_sum += y[i]*sRA[i]
    y_sq = y.copy()
    for i in range(len(y)):
        y_sq[i] *= y_sq[i]
    y_sq_sum = sum(y_sq)

    a = np.array([[N, x_sum, y_sum],
                  [x_sum, x_sq_sum, x_y_sum],
                  [y_sum, x_y_sum, y_sq_sum]])
    b = np.array([sRA_sum, sRA_x_sum, sRA_y_sum])
    b1a11a12 = np.linalg.solve(a,b)
    b1 = b1a11a12[0]*15
    a11 = b1a11a12[1]*15
    a12 = b1a11a12[2]*15

    sDEC_sum = sum(sDEC)
    sDEC_x_sum = 0
    for i in range(len(x)):
        sDEC_x_sum += x[i]*sDEC[i]
    sDEC_y_sum = 0
    for i in range(len(y)):
        sDEC_y_sum += y[i]*sDEC[i]

    a = np.array([[N, x_sum, y_sum],
                  [x_sum, x_sq_sum, x_y_sum],
                  [y_sum, x_y_sum, y_sq_sum]])
    b = np.array([sDEC_sum, sDEC_x_sum, sDEC_y_sum])
    b2a21a22 = np.linalg.solve(a,b)
    b2 = b2a21a22[0]*15
    a21 = b2a21a22[1]*15
    a22 = b2a21a22[2]*15

    #calculating astrometry
    RA_ast = (b1+a11*x_cent+a12*y_cent)/15
    DEC_ast = b2+a21*x_cent+a22*y_cent

    #calculating uncertainty
    RAs, DECs = [], []
    for i in range(len(x)):
        RAs.append((b1+a11*x[i]+a12*y[i])/15)
        DECs.append((b2+a21*x[i]+a22*y[i])/15)
    
    RA_diffs = 0
    #print(sRA)
    #print(RAs)
    #print()
    for i in range(len(sRA)):
        RA_diffs += (sRA[i]-RAs[i])**2
    RA_sd = sqrt(RA_diffs/(N-3))*3600*15
    #print(RA_diffs)

    DEC_diffs = 0
    for i in range(len(sDEC)):
        DEC_diffs += (sDEC[i]-DECs[i])**2
    DEC_sd = sqrt(DEC_diffs/(N-3))*3600*15

    return b1, b2, a11, a12, a21, a22, RA_ast, DEC_ast, RA_sd, DEC_sd, sRA, sDEC

def flat(x, y, sRA, sDEC):
    A, D = radians(15*sum(sRA)/len(sRA)), radians(15*sum(sDEC)/len(sDEC))
    #print(A)
    #print(D)
    L = 3.911/(9*(10**(-6)))
    ##flatalpha = b1+a11*x+a12*y-x/L
    ##flatdelta = b2+a21*x+a22*y-y/L
    flatRAs, flatDECs = [], []
    sRArad, sDECrad = sRA.copy(), sDEC.copy()
    for i in range(len(sRA)):
        sRArad[i] = radians(sRA[i]*15)
        sDECrad[i] = radians(sDEC[i]*15)
    #print(sRArad)
    #print(sDECrad)
    for i in range(len(sRArad)):
        H = sin(sDECrad[i])*sin(D) + cos(sDECrad[i])*cos(D)*cos(sRArad[i]-A)
        flatRAs.append((degrees((cos(sDECrad[i])*sin(sRArad[i]-A))/H - x[i]/L))/15)
        flatDECs.append((degrees((sin(sDECrad[i])*cos(D)-cos(sDECrad[i])*sin(D)*cos(sRArad[i]-A))/H - y[i]/L))/15)
    #print(flatRAs)
    #print(flatDECs)
    return flatRAs, flatDECs

def unflat(flatRA, flatDEC, sRA, sDEC):
    A, D = radians(15*sum(sRA)/len(sRA)), radians(15*sum(sDEC)/len(sDEC))
    delta = cos(D) - radians(flatDEC)*sin(D)
    r = sqrt(radians(flatRA)**2+delta**2)
    RA = A+atan(radians(flatRA)/delta)
    DEC = atan((sin(D)+radians(flatDEC)*cos(D))/r)
    return RA, DEC

data = open("July4Seq1Img1Input.txt") #July4Seq1Img1Input.txt
X, Y, SRA, SDEC = [], [], [], []
for line in data:
    values = line.split()
    vi = 0
    for x in values:
        if vi == 0:
            X.append(float(x))
        elif vi == 1:
            Y.append(float(x))
        elif vi == 2:
            RAnums = x.split(":")
            SRA.append(float(RAnums[0]) + float(RAnums[1])/60 + float(RAnums[2])/3600)
        else:
            DECnums = x.split(":")
            SDEC.append((float(DECnums[0]) + float(DECnums[1])/60 + float(DECnums[2])/3600)/15)
        vi += 1
    '''
    X.append(float(values[0]))
    Y.append(float(values[1]))
    RAnums = values[2].split(":")
    SRA.append(float(RAnums[0]) + float(RAnums[1])/60 + float(RAnums[2])/3600)
    DECnums = values[3].split(":")
    SDEC.append(float(DECnums[0]) + float(DECnums[1])/60 + float(DECnums[2])/3600)
    '''
#for i in range(len(X)):
#    print(X[i], Y[i], SRA[i], SDEC[i])
x, y = float(input("x = ")), float(input("y = "))
#1403.6, 1585.9
flatten = False
b1, b2, a11, a12, a21, a22, RA, DEC, RAsd, DECsd, SRA, SDEC = LSPR(X, Y, SRA, SDEC, x, y, flatten)
if(flatten):
    RA, DEC = unflat(RA, DEC, SRA, SDEC)
print("***************")
print("plate constants")
print("***************")
print("b1: " +  str(b1) +  " deg")
print("b2: " +  str(b2) +  " deg")
print("a11: " + str(a11) + " deg/pix")
print("a12: " + str(a12) + " deg/pix")
print("a21: " + str(a21) + " deg/pix")
print("a22: " + str(a22) + " deg/pix")
print("***********")
print("uncertainty")
print("***********")
print(" RA: " + str(RAsd))
print("DEC: " + str(DECsd))
print("*********************************")
print("astrometry for (x,y)=")
print("( " + str(x) + ", " + str(y) + ")")
print("*********************************")
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
