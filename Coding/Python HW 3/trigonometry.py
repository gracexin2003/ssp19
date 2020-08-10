# Purpose: code trig
# Project: trigonometry
# Name: Grace Xin
# Due June 28

from math import acos, asin, cos, sin, degrees, radians

# a function to determine the quadrant of an angle based on its sine and cosine (in radians)
# returns the angle in the correct quadrant (in radians)
def findQuadrant(sine, cosine):
    if cosine > 0 and sine > 0: #1
        return asin(sine)

    if cosine < 0 and sine > 0: #2
        return acos(cosine)

    if cosine < 0 and sine < 0: #3
        return pi
    - asin(sine)

    if cosine > 0 and sine < 0: #4
        return 2*pi + asin(sine)

# a function that given the values (in radians) of two sides and the included angle of a spheical triangle
# returns the values of the remaining side and two angles (in radians)
def SAS(a, B, c):
    # YOUR CODE HERE (part a)
    ## cos(b)=cos(a)*cos(c)+sin(a)*sin(c)*cos(B)
    b = acos(cos(a)*cos(c)+sin(a)*sin(c)*cos(B))
    
    ## sin(a)/sin(A) = sin(b)/sin(B) = sin(c)/sin(C)
    A = asin(sin(B)*sin(a)/sin(b))
    C = asin(sin(B)*sin(c)/sin(b))
    return b, A, C

# YOUR CODE HERE (part b)
a = 106
B = 114
c = 42
b, A, C = SAS(radians(a), radians(B), radians(c))
print(degrees(b))
print(degrees(A))
print(degrees(C))

