import numpy as np
import random

# voir p, voir q, voir non-p, voir non-q
q_dict = np.array([[0.0, 0.0, 0.0, 0.0], # ignorer
                  [0.0, 0.0, 0.0, 0.0]]) # vérifier

epsilon = 0.1 # 10% du temps, l'agent choisi au hasard

def choice() : 
    situation =  random.randint(0, 3)
    epsilon = random.randint(1, 10)
    if epsilon>1 : # us epsilon-greedy policy  
        if q_dict[0, situation] > q_dict[1, situation] : 
            return situation, 0
        elif q_dict[0, situation] < q_dict[1, situation] : 
            return situation, 1
        else : 
            return situation, random.randint(0,1)
    else : 
        return situation, random.randint(0,1)

def verification(situation, choice1) :
    final_cost = 0
    primary_cost = 0
    secondary_cost  = 0
    if choice1 == 1 : 
            primary_cost = 1
    if situation == 3 or situation == 0 : 
        issue=random.randint(1, 100)
        if issue > 80 : # 20 pourcents de chance que la carte retournée soit fautive (qu'on ait non-Q et P)
            if choice1 == 0 : 
                secondary_cost = 10 # D
            else : 
                secondary_cost = -7 # V

    final_cost = final_cost-primary_cost-secondary_cost
    return final_cost 

def update_q(situation, choice1, final_cost) : 
    q_previous = q_dict[choice1][situation]
    alpha = 0.1 # taux d'apprentissage
    R = final_cost # récompense réelle
    q_dict[choice1][situation] =     q_previous + alpha * (R - q_previous)


for i in range (5000) :
    situation, choice1 = choice()
    final_cost = verification(situation, choice1)
    update_q(situation, choice1, final_cost)
print("q_dict :", q_dict)



