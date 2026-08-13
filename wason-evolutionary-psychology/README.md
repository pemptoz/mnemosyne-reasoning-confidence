# Simulations du test de Wason avec apprentissage par renforcement

 L’objectif est d’observer comment des agents apprennent à choisir quelles cartes vérifier dans différentes situations : `P`, `Q`, `non-P` et `non-Q`.

Les agents peuvent soit :

- **ignorer** une carte ;
- **vérifier** une carte.

La stratégie logique attendue dans le test de Wason consiste à vérifier **P** et **non-Q**.

---

## Contenu du projet

### 1. Agent simple avec Q-learning

Le premier script simule un seul agent utilisant une table de valeurs `Q`.

L’agent apprend progressivement quelle action choisir pour chaque type de carte.  
Il utilise une stratégie **epsilon-greedy** :

- la plupart du temps, il choisit l’action avec la meilleure valeur connue ;
- parfois, il choisit au hasard pour explorer.

À la fin, le programme affiche la table `q_dict`, qui représente les préférences apprises par l’agent.

---

### 2. Population d’agents et transmission culturelle

Le deuxième script étend le modèle à une population de `100` agents.

Chaque agent possède sa propre table de Q-learning.  
La simulation se déroule sur plusieurs générations.

Deux conditions sont comparées :

- **sans transmission culturelle** : chaque agent apprend seul ;
- **avec transmission culturelle** : les moins bons agents copient les meilleurs.

Le programme affiche le cerveau moyen de la population et trace un graphique montrant l’évolution du pourcentage d’agents appliquant la stratégie de Wason.

---

### 3. Réseau de neurones avec contexte social et abstrait

Le troisième script remplace la table Q par un petit réseau de neurones en PyTorch.

Chaque agent reçoit en entrée :

- la carte observée : `P`, `Q`, `non-P` ou `non-Q` ;
- le type de contexte : social ou abstrait.

Dans le contexte social, détecter un tricheur est fortement récompensé.  
Dans le contexte abstrait, la récompense est plus faible.

Le programme analyse ensuite les stratégies apprises par la population dans les deux contextes.

---

### 4. Réseau de neurones avec biais de confirmation

Le quatrième script reprend le modèle neuronal, mais modifie les récompenses dans le contexte abstrait.

Dans ce cas, l’agent est encouragé à vérifier les éléments explicitement présents dans la règle : `P` et `Q`.

Cela permet de simuler un **biais de confirmation** ou un **biais d’appariement**, souvent observé dans le test de Wason abstrait.

---

## Installation

Créer un environnement Python, puis installer les dépendances :

```bash
pip install numpy matplotlib torch
```