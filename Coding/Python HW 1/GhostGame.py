# Purpose: Write a word game
# Project: Ghost Game
# Due: 6/21/19
# Name: Grace Xin

def ghost():
    player = 1 # starting player is player 1, will switch in between turns

    # read words.txt into a list of valid words
    valid_words = []
    for line in open("words.txt", "r"):
        valid_words.append(line[:-1]) # to delete the /n at the end of the line

    game_over = False # determines when to break from the game loop
    winner = 0 # the winner of the game: will eventually be 1 or 2

    current_str = ""
    while not game_over: # loop until one player loses
        print("Current word: " + current_str.lower())
        next_letter = input("Player " + str(player) + ", enter a letter: ")
        
        # check for invalid inputs
        if next_letter.isalpha():
            current_str += next_letter
        else:
            print("Please enter a letter.")
            continue
        
        # check if the input creates a word longer than 3 letters (losing move)
        losing_move = False
        for word in valid_words:
            if word.lower() == current_str.lower() and len(word) > 3:
                losing_move = True
                winner = 3-player # winner is the other player (3-1=2, 3-2=1)
                print("You created a word longer than 3 letters!")
                game_over = True
                break
        if game_over:
            break

        # check if there are no words that can be made by the input (losing move)
        losing_move = True
        for word in valid_words:
            if word[:len(current_str)].lower() == current_str.lower():
                losing_move = False
        if losing_move:
            print("You can't make any words with that sequence!")
            winner = 3-player
            game_over = True
        else: # if no one lost, switch players
            player = 3-player
    
    print("Game over! The winner is Player " + str(winner) + "!") #print winner

ghost() # initiate the game
