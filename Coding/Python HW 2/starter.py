# starter code for exercise 0 on programming homework 2

import numpy as np

fruits = np.array([["Apple","Banana","Blueberry","Cherry"],
["Coconut","Grapefruit","Kumquat","Mango"],
["Nectarine","Orange","Tangerine","Pomegranate"],
["Lemon","Raspberry","Strawberry","Tomato"]])

#a: extract the bottom right element in one command.
print(fruits[3, 3])
print("")

#b: extract the inner 2X2 square in one command.
print(fruits[1:3, 1:3])
print("")

#c: extract the first and third rows in one command.
print(fruits[0:4:2])
print("")

#d: extract the inner 2X2 square flipped vertically and horizontally in one command.
print(fruits[2:0:-1, 2:0:-1])
print("")

#e: swap the first and last columns in three commands. Hint: make a copy of an array using the copy() method.
swapped = fruits.copy() #copy(fruits) # np.copy(fruits)
for x in range(len(fruits[0])):
    swapped[x, 0] = fruits[x, 3]
    swapped[x, 3] = fruits[x, 0]
print(swapped)
print("")

#f: replace every element with the string "SLICED!" in one command.
fruits.fill("SLICED!")

print(fruits)
