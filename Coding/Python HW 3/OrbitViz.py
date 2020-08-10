from vpython import *
from math import *

a = 1.00000011                   #semi-major axis
e = 0.01671022#0.1750074901308245                  #eccentricity
M = radians(336.0050001501443)          #mean anomaly
Oprime = radians(-11.26064)      #uppercase omega: longitude of the ascending node
iprime = radians(0.00005)     #inclination
wprime = radians(102.94719)     #lowercase omega: argument of perihelion

def solvekep(M): #step 2: get E from inverting the kepler equation, M = E-e*sinE
    Eguess = M #first guess for E is M
    Mguess = Eguess - e*sin(Eguess) #M = (E - e*sinE) to find f
    Mtrue = M #the real M
    while abs(Mguess - Mtrue) > 1e-004: #runs until there is minimal difference between Mguess and Mtrue
        Mguess = Eguess - e*sin(Eguess) #M = (E - e*sinE) to find f
        Eguess = Eguess - (Eguess - e*sin(Eguess) - Mtrue) / (1 - e*cos(Eguess)) # E(x+1) = E(x)-((E(x)-e*sin(E(x))-M)/(1-e*cos(E(x))
    return Eguess

sqrtmu = 0.01720209895
mu = sqrtmu**2
time = 0
dt = .05
period = sqrt(4*pi**2*a**3/mu)
r1ecliptic = vector(0, 0, 0)
Mtrue = 2*pi/period*(time) + M
Etrue = solvekep(Mtrue)
r1ecliptic.x = (cos(wprime)*cos(Oprime) - sin(wprime)*cos(iprime)*sin(Oprime))*(a*cos(Etrue)-a*e) - (cos(wprime)*cos(iprime)*sin(Oprime) + sin(wprime)*cos(Oprime))*(a*sqrt(1-e**2)*sin(Etrue))
r1ecliptic.y = (cos(wprime)*sin(Oprime) + sin(wprime)*cos(iprime)*cos(Oprime))*(a*cos(Etrue)-a*e) + (cos(wprime)*cos(iprime)*cos(Oprime) - sin(wprime)*sin(Oprime))*(a*sqrt(1-e**2)*sin(Etrue))
# extra parentheses ")"...                                                                      ^here
r1ecliptic.z = sin(wprime)*sin(iprime)*(a*cos(Etrue)-a*e) + cos(wprime)*sin(iprime)*(a*sqrt(1-e**2)*sin(Etrue))
earth = sphere(pos=r1ecliptic*150, radius=(15), color=color.blue)
earth.trail = curve(color=color.blue)
sun = sphere(pos=vector(0,0,0), radius=(50), color=color.yellow)


aA = 1.057555                #semi-major axis
eA = 0.345394#0.1750074901308245                  #eccentricity
MA = radians(165.202877)          #mean anomaly
OprimeA = radians(236.258165)      #uppercase omega: longitude of the ascending node
iprimeA = radians(25.220846)     #inclination
wprimeA = radians(255.533346)     #lowercase omega: argument of perihelion

sqrtmu = 0.01720209895
mu = sqrtmu**2
time = 0
dt = .05
periodA = sqrt(4*pi**2*a**3/mu)
r1eclipticA = vector(0, 0, 0)
MtrueA = 2*pi/periodA*(time) + MA
EtrueA = solvekep(MtrueA)
r1eclipticA.x = (cos(wprimeA)*cos(OprimeA) - sin(wprimeA)*cos(iprimeA)*sin(OprimeA))*(aA*cos(EtrueA)-aA*eA) - (cos(wprimeA)*cos(iprimeA)*sin(OprimeA) + sin(wprime)*cos(Oprime))*(a*sqrt(1-e**2)*sin(Etrue))
r1eclipticA.y = (cos(wprimeA)*sin(OprimeA) + sin(wprimeA)*cos(iprimeA)*cos(OprimeA))*(aA*cos(EtrueA)-aA*eA) + (cos(wprimeA)*cos(iprimeA)*cos(OprimeA) - sin(wprime)*sin(Oprime))*(a*sqrt(1-e**2)*sin(Etrue))
# extra parentheses ")"...                                                                      ^here
r1eclipticA.z = sin(wprimeA)*sin(iprimeA)*(aA*cos(EtrueA)-aA*eA) + cos(wprimeA)*sin(iprimeA)*(aA*sqrt(1-eA**2)*sin(EtrueA))
asteroid = sphere(pos=r1eclipticA*150, radius=(5), color=color.white)
asteroid.trail = curve(color=color.white)

while True: 
    rate(100)
    time = time + 1
    Mtrue = 2*pi/period*(time) + M
    Etrue = solvekep(Mtrue)
    r1ecliptic.x = (cos(wprime)*cos(Oprime) - sin(wprime)*cos(iprime)*sin(Oprime))*(a*cos(Etrue)-a*e) - (cos(wprime)*cos(iprime)*sin(Oprime) + sin(wprime)*cos(Oprime))*(a*sqrt(1-e**2)*sin(Etrue))
    r1ecliptic.y = (cos(wprime)*sin(Oprime) + sin(wprime)*cos(iprime)*cos(Oprime))*(a*cos(Etrue)-a*e) + (cos(wprime)*cos(iprime)*cos(Oprime) - sin(wprime)*sin(Oprime))*(a*sqrt(1-e**2)*sin(Etrue))
    # parentheses  ^here                                                                            ^and here
    r1ecliptic.z = sin(wprime)*sin(iprime)*(a*cos(Etrue)-a*e) + cos(wprime)*sin(iprime)*(a*sqrt(1-e**2)*sin(Etrue))
    # tabbed the two lines below into the while loop
    earth.pos = r1ecliptic*150
    earth.trail.append(pos=earth.pos)

    MtrueA = 2*pi/periodA*(time) + MA
    EtrueA = solvekep(MtrueA)
    r1eclipticA.x = (cos(wprimeA)*cos(OprimeA) - sin(wprimeA)*cos(iprimeA)*sin(OprimeA))*(aA*cos(EtrueA)-aA*eA) - (cos(wprimeA)*cos(iprimeA)*sin(OprimeA) + sin(wprime)*cos(Oprime))*(a*sqrt(1-e**2)*sin(Etrue))
    r1eclipticA.y = (cos(wprimeA)*sin(OprimeA) + sin(wprimeA)*cos(iprimeA)*cos(OprimeA))*(aA*cos(EtrueA)-aA*eA) + (cos(wprimeA)*cos(iprimeA)*cos(OprimeA) - sin(wprime)*sin(Oprime))*(a*sqrt(1-e**2)*sin(Etrue))
    # parentheses  ^here                                                                            ^and here
    r1eclipticA.z = sin(wprimeA)*sin(iprimeA)*(aA*cos(EtrueA)-aA*eA) + cos(wprimeA)*sin(iprimeA)*(aA*sqrt(1-eA**2)*sin(EtrueA))    # tabbed the two lines below into the while loop
    asteroid.pos = r1eclipticA*150
    asteroid.trail.append(pos=asteroid.pos)

'''
Changes I made to the OrbitViz.py program:
    Line 15: changed "<" in [while abs(Mguess - Mtrue) < 1e-004] to ">". I got
this from William Beard, and it changes the object's speed as it goes closer to
or further away from the sun. I'm not fully sure why it works, though.
    Line 28: had an extra parentheses ")" in the middle of the calculations that
caused a syntax error
    Line 34: changed while (1==1) to while True because (1) it looked better and
(2) parentheses are not needed
    Line 40: took out extra parentheses to make the equations for r1ecliptic.x and
r1ecliptic.y the same length (not necessary, but it looks better)
    Last two lines: tabbed them up to fit into the while loop and to make the
earth move successfully
'''
