# Purpose: to write code that converts global coordinates to local ones
# Project: coordinate systems
# Due: June 21
# Name: Grace Xin

from math import pi, radians, degrees, sin, cos, asin, acos

# DEFINE FUNCTION CONVERTING RA AND DEC TO ALTITUDE AND AZIMUTH HERE
# Your function should have parameters for 
# > RA and dec in decimal degrees
# > longitude and latitude of the observer in decimal degrees
# > year, month, day, and UT in decimal hours
# note: to have a function return two values, just do: return value1, value2
def global_to_local(RA, dec, long, lat, year, month, day, UT):
    ## find siderial time from year, month, day, UT
    J0 = 367*year-int(7*(year+int((month+9)/12))/4)+int(275*month/9)+day+1721013.5
    print(J0)
    J = (J0-2451545)/36525
    print(J)
    meanGMST = 100.46061837+36000.77053608*J+3.87933*(10**-4)*(J**2)-(J**3)/(3.871*10**7)
    print(meanGMST)
    GMST = meanGMST+360.985647366*UT/24
    print(GMST)
    siderial = (GMST+long)%360
    print(siderial)
    Hdeg = RA-siderial
    print(Hdeg)
    ## sin(altitude) = sin(lat)*sin(dec)+cos(lat)*cos(dec)*cos(H)
    altitude = asin(sin(radians(lat))*sin(radians(dec))+cos(radians(lat))
                           *cos(radians(dec))*cos(radians(Hdeg)))
    
    ## sin(dec)=sin(altitude)*sin(lat)+cos(altitude)*cos(lat)*cos(A)
    ## cos(A) = (sin(dec)-sin(altitude)*sin(lat))/(cos(altitude)*cos(lat))
    A = acos((sin(radians(dec))-sin(altitude)*sin(radians(lat)))/(cos(altitude)*cos(radians(lat))))

    ## return altitude and azimuth values
    return degrees(altitude), (360-degrees(A))

print("testing RA/Dec to Alt/Az")
# test cases (Etscorn) --> expected results (approx.)
# RA: 156.65116, dec: 24.90443, longitude: 253.08608, latitude: 34.0727, year: 2019, month: 6, day: 12, UT: 5
# approximate expected result: altitude: 28.17247, azimuth: 282.38397
print(global_to_local(156.65116, 24.90443, 253.08608, 34.0727, 2019, 6, 12, 5))
# RA: 238.40339, dec: -19.19939, longitude: 253.086083, latitude: 34.0727, year: 2019, month: 7, day: 12, UT: 5
# approximate expected result: altitude: 33.58536, azimuth: 202.22719
print(global_to_local(238.40339, -19.19939, 253.086083, 34.0727, 2019, 7, 12, 5))


# DEFINE FUNCTION CONVERTING EQUATORIAL TO RECTANGULAR ECLIPTIC HERE
# note: to have a function return two values, just do: return value1, value2
def equatorial_to_rectangular(RA, dec):
    obliquity = 23.4358
    x = cos(radians(RA))*cos(radians(dec))
    y = sin(radians(RA))*cos(radians(dec))
    z = sin(radians(dec))
    y1 = y*cos(radians(obliquity))+z*sin(radians(obliquity))
    z1 = -y*sin(radians(obliquity))+z*cos(radians(obliquity))
    return x, y1, z1
    

print("testing equatorial to rectangular ecliptic")
# INCLUDE TEST CASES FROM HOMEWORK
## RA = 14h12m38.5s, dec = +23 48'38''
## approx. expected result = (-0,765889, -0.298581, 0.569441)
print(equatorial_to_rectangular(14.21069444444444*15, 23.81055555555555))
## RA = 6h00m00s, dec = +00 00'00''
## approx. expected result = (0, 0.917506, -0.397721)
print(equatorial_to_rectangular(6*15, 0))
