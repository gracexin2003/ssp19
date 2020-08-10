# Purpose: to write a program that does vector operations
# Project: vectors and vector operations
# Due: June 21
# Name: Grace Xin

from math import sqrt

def magnitude(vec):
    if len(vec) == 0:
        return 0
    if len(vec) == 1:
        return vec[0]
    sum = 0
    for x in vec:
        sum += abs(x)
    mag = sqrt(sum)
    return mag
print("testing magnitude")
testvec = []
print(magnitude(testvec))
testvec = [3]
print(magnitude(testvec))
testvec = [1, -1]
print(magnitude(testvec))
testvec = [1, 1, 1, 1]
print(magnitude(testvec))

def dot(v1, v2):
    dot = 0
    for i in range(len(v1)):
        dot += v1[i]*v2[i]
    return dot
print("testing dot")
vec1, vec2 = [], []
print(dot(vec1, vec2))
vec1, vec2 = [2, 5, 6], [3, 7, 8]
print(dot(vec1, vec2))
vec1, vec2 = [1, -1, 0], [-1, -1, 5]
print(dot(vec1, vec2))
vec1, vec2 = [1, 0, 1, 0], [2, 2, 0, 2]
print(dot(vec1, vec2))

def cross(v1, v2):
    return [v1[1]*v2[2]-v1[2]*v2[1],
            -(v1[0]*v2[2]-v1[2]*v2[0]),
            v1[0]*v2[1]-v1[1]*v2[0]]
print("testing cross")
vec1, vec2 = [1, 0, 0], [0, 1, 0]
print(cross(vec1, vec2))
vec1, vec2 = [1, 0, 0], [0, 0, 1]
print(cross(vec1, vec2))
vec1, vec2 = [2, 5, 6], [3, 7, 8]
print(cross(vec1, vec2))
