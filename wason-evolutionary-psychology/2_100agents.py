import numpy as np
import random
import matplotlib.pyplot as plt

# 1. PARAMÈTRES
NUM_AGENTS = 100
NUM_GENERATIONS = 50  
STEPS_PER_GEN = 30    

def choice(agent_id, population_q):
    situation = random.randint(0, 3)
    epsilon = random.randint(1, 100)
    q_dict = population_q[agent_id]
    
    if epsilon > 10:
        if q_dict[0, situation] > q_dict[1, situation]: return situation, 0
        elif q_dict[0, situation] < q_dict[1, situation]: return situation, 1
        else: return situation, random.randint(0, 1)
    else: return situation, random.randint(0, 1)

def verification(situation, decision):
    reward = 0
    if decision == 1: reward -= 1
    if situation == 0 or situation == 3:
        if random.randint(1, 100) > 80:
            if decision == 1: reward += 10
            else: reward -= 7
    return reward 

def update_q(agent_id, situation, decision, reward, population_q):
    q_previous = population_q[agent_id][decision, situation]
    alpha = 0.1
    population_q[agent_id][decision, situation] = q_previous + alpha * (reward - q_previous)

# L'agent applique-t-il exactement P et non-Q ?
def is_wason(q_table):
    check_p = q_table[1, 0] > q_table[0, 0]
    ignore_q = q_table[0, 1] > q_table[1, 1]
    ignore_non_p = q_table[0, 2] > q_table[1, 2]
    check_non_q = q_table[1, 3] > q_table[0, 3]
    return check_p and ignore_q and ignore_non_p and check_non_q

# 2. FONCTION PRINCIPALE DE SIMULATION
def run_simulation(cultural_transmission):
    population_q = [np.zeros((2, 4)) for _ in range(NUM_AGENTS)]
    scores = [0] * NUM_AGENTS
    history_wason_percent = [] # Pour stocker les données du graphique
    
    for gen in range(NUM_GENERATIONS):
        # Phase A : Apprentissage
        for agent_id in range(NUM_AGENTS):
            for step in range(STEPS_PER_GEN):
                situation, decision = choice(agent_id, population_q)
                reward = verification(situation, decision)
                update_q(agent_id, situation, decision, reward, population_q)
                scores[agent_id] += reward 
                
        # Phase B : Culture
        if cultural_transmission:
            ranked_agents = np.argsort(scores) 
            bottom_10 = ranked_agents[:10]
            top_10 = ranked_agents[-10:]
            for i in range(10):
                population_q[bottom_10[i]] = population_q[top_10[i]].copy()
                
        scores = [0] * NUM_AGENTS
        
        # Mesure : Combien d'agents ont la stratégie Wason ?
        wason_count = sum([1 for q in population_q if is_wason(q)])
        history_wason_percent.append(wason_count / NUM_AGENTS * 100)

    moyenne_population = np.mean(population_q, axis=0)

    print("Cerveau moyen de la population à la fin :")
    print("          Voir P | Voir Q | Voir non-P | Voir non-Q")
    print("Ignorer :", np.round(moyenne_population[0], 2))
    print("Vérifier:", np.round(moyenne_population[1], 2))
    print("---------------------------------------------")
    return history_wason_percent

# 3. LANCEMENT ET GRAPHIQUE
print(f"Début de la simulation (Transmission culturelle : {False})")
history_false = run_simulation(cultural_transmission=False)

print(f"Début de la simulation (Transmission culturelle : {True})")
history_true = run_simulation(cultural_transmission=True)

# Tracé du graphique
plt.figure(figsize=(10, 6))
plt.plot(history_false, label="Apprentissage Individuel (Sans culture)", color="red", linestyle="--")
plt.plot(history_true, label="Apprentissage Social (Avec culture)", color="blue", linewidth=2)

plt.title("Évolution de la stratégie Wason déontique dans la population")
plt.xlabel("Générations (Le temps qui passe)")
plt.ylabel("% d'agents appliquant la stratégie Wason")
plt.legend()
plt.grid(True)
plt.ylim(0, 105)
plt.show()