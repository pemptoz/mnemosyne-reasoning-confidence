import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import Counter

# ==========================================
# 1. PARAMÈTRES
# ==========================================
NUM_AGENTS = 100
NUM_GENERATIONS = 40
STEPS_PER_GEN = 100
CULTURAL_TRANSMISSION = True

class AgentBrain(nn.Module):
    def __init__(self):
        super(AgentBrain, self).__init__()
        self.fc1 = nn.Linear(5, 8) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def get_state_vector(situation, is_social):
    state = [0.0, 0.0, 0.0, 0.0, float(is_social)]
    state[situation] = 1.0
    return torch.tensor(state, dtype=torch.float32)

def verification(situation, decision, is_social):
    reward = 0
    if decision == 1: reward -= 1 # Le coût de l'effort cognitif/physique
    
    if is_social:
        # CONTEXTE SOCIAL : La survie économique prime
        if situation == 0 or situation == 3: # P ou non-Q
            if random.random() < 0.2: 
                if decision == 1: reward += 10  # On attrape le tricheur
                else: reward -= 7               # On se fait avoir
    else:
        # CONTEXTE ABSTRAIT : Biais de confirmation / d'appariement
        # L'agent ressent le "besoin" ou le "plaisir" de vérifier les éléments 
        # explicitement nommés dans la règle (P et Q).
        if situation == 0 or situation == 1: # P ou Q
            if random.random() > 0.2: 
                if decision == 1:
                    # La satisfaction psychologique de vérifier ce qui est écrit (+3)
                    # compense largement l'effort de retourner la carte (-1).
                    reward += 2
    return reward

# ==========================================
# 2. INITIALISATION ET ENTRAÎNEMENT
# ==========================================
population = [AgentBrain() for _ in range(NUM_AGENTS)]
optimizers = [optim.Adam(agent.parameters(), lr=0.01) for agent in population]
criterion = nn.MSELoss() 

print("Entraînement mixte (1 population, 5 neurones) en cours...")

for gen in range(NUM_GENERATIONS):
    scores = [0] * NUM_AGENTS
    
    for agent_id, agent in enumerate(population):
        optimizer = optimizers[agent_id]
        
        for step in range(STEPS_PER_GEN):
            situation = random.randint(0, 3)
            contexte_rencontre = random.choice([True, False]) 
            state_tensor = get_state_vector(situation, is_social=contexte_rencontre)
            q_values = agent(state_tensor)
            
            if random.randint(1, 100) > 10: decision = torch.argmax(q_values).item()
            else: decision = random.randint(0, 1)
                
            reward = verification(situation, decision, is_social=contexte_rencontre)
            scores[agent_id] += reward
            
            target_q_values = q_values.clone().detach()
            target_q_values[decision] = float(reward)
            
            optimizer.zero_grad()
            loss = criterion(q_values, target_q_values)
            loss.backward()
            optimizer.step()
            
    if CULTURAL_TRANSMISSION:
        ranked_agents = np.argsort(scores)
        for i in range(10):
            pire_idx = ranked_agents[i]
            meilleur_idx = ranked_agents[-(i+1)]
            population[pire_idx].load_state_dict(population[meilleur_idx].state_dict())

# ==========================================
# 3. ANALYSE STATISTIQUE DE LA POPULATION
# ==========================================
def analyser_population(is_social_test):
    contexte = "SOCIAL" if is_social_test else "ABSTRAIT"
    print(f"\n{'='*50}\n RÉSULTATS POUR LE CONTEXTE : {contexte}\n{'='*50}")
    
    noms_cartes = ["P", "Q", "non-P", "non-Q"]
    strategies_compte = []
    moyennes_q = np.zeros((4, 2)) # 4 cartes, 2 actions (Ignorer, Vérifier)
    
    for agent in population:
        agent.eval()
        decision_agent = []
        for situation in range(4):
            state = get_state_vector(situation, is_social=is_social_test)
            with torch.no_grad():
                q_values = agent(state)
                moyennes_q[situation, 0] += q_values[0].item()
                moyennes_q[situation, 1] += q_values[1].item()
                
                decision_idx = torch.argmax(q_values).item()
                decision_agent.append("VÉRIFIER" if decision_idx == 1 else "Ignorer")
        
        strategies_compte.append(str(decision_agent))
    
    moyennes_q /= NUM_AGENTS # Moyenne sur les 100 agents
    
    print("1. MOYENNES DES PARAMÈTRES (Q-Values) DE LA POPULATION :")
    print("Cartes   | Score moyen Ignorer | Score moyen Vérifier")
    print("-" * 55)
    for i in range(4):
        print(f"Voir {noms_cartes[i]:<5} | {moyennes_q[i, 0]:>19.2f} | {moyennes_q[i, 1]:>20.2f}")
    
    print("\n2. PROPORTIONS DES STRATÉGIES FINALES :")
    compteur = Counter(strategies_compte)
    for strat, count in compteur.most_common():
        pourcentage = (count / NUM_AGENTS) * 100
        print(f"{pourcentage:5.1f}% des agents ont choisi : {strat}")

analyser_population(is_social_test=True)
analyser_population(is_social_test=False)