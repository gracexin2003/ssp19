# Purpose: Write a Hangman game
# Project: Hangman
# Due: 6/21/19
# Name: Grace Xin

import math
import random

lives = 7 # total number of lives 
# list of words -_-
word_list = ["apple", "alligator", "animal", "almond", "apricot", "berry",
             "blueberry", "bird", "beetle", "bell", "banana", "flamingo",
             "cherry", "cashew", "camel", "corn", "crocodile", "duck",
             "camel", "zebra", "pineapple", "mango", "pear", "sheep", "penguin",
             "hippo", "watermelon", "bubblegum", "zuchinni", "pidgeon"]

# choose a random word from the word list
# word_index = int(random.random()*list_length)
word = random.choice(word_list) # word_list[word_index]

word_length = len(word)
print("Length of word: " + str(word_length))
win = False; # whether the player found the word or not

shown_str = "" # the string shown to the player
for x in range(word_length):
    shown_str += "_"

while lives > 0 and not win: # loop through until the player wins or loses
    print(shown_str)
    print("Remaining lives: " + str(lives))
    letter_guess = input("Guess a letter: ")
    print(letter_guess)
    guessed = False
    for i in range(word_length):
        char = word[i]
        if letter_guess == char:
            shown_str = shown_str[:i]+char+shown_str[i+1:]
            guessed = True
    if not guessed:
        lives -= 1
    if word == shown_str:
        win = True
        print("Congratulations! The word was " + word)
if lives == 0:
    print("You lost! The word was: " + word)
