# Analyse des modèles de raisonnement avec mReasoner

## 1. Logiciels et projets utilisés

### CCOBRA

[CCOBRA](https://github.com/CognitiveComputationLab/ccobra) est utilisé
pour exécuter les benchmarks et ajuster les paramètres des modèles aux
réponses individuelles.

### pymreasoner

Ce projet utilise et adapte le package
[pymreasoner](https://github.com/nriesterer/pymreasoner), qui fournit
une interface Python vers le modèle cognitif [mReasoner](https://github.com/skhemlani/mReasoner) exécuté avec
Clozure Common Lisp. 

## Clozure Common Lisp et mReasoner

Ce projet utilise **pymreasoner**, une interface Python pour le modèle
cognitif **mReasoner**. mReasoner s’exécute avec
[Clozure Common Lisp](https://ccl.clozure.com/).

Les répertoires suivants ne sont pas versionnés dans Git :

```text
data-analyse/.ccl/
data-analyse/.mreasoner/
```

Pour créer ces dossiers,, commencez par exécuter : 

```python
python3 -m pip install -r requirements.txt
python3 initialization_mreasoner.py
```

## 2. Sources des données

### Jeu de données Ragni2016

Le fichier `data/raw/Ragni2016.csv` est utilisé pour les analyses de consensus et d'entropie. Il contient des réponses humaines à des syllogismes catégoriques.

Nous utilisons le jeu de données « Ragni2016, obtenu via le framework CCOBRA (CognitiveComputationLab, s.d.), issu d’une expérience en ligne menée en 2016 au Cognitive Computation Lab de l’Université de Freiburg sous la direction de Marco Ragni : [CognitiveComputationLab](!https://github.com/CognitiveComputationLab/ccobra/blob/master/benchmarks/syllogistic/data/Ragni2016.csv)


### Expériences E1 et E2

Les données expérimentales E1 et E2 proviennent du projet OSF suivant :

<https://osf.io/aejkf/overview>

Dans ce dépôt, les fichiers bruts utilisés sont :

```text
data/raw/E3_syllogismData_full.csv
data/raw/E4_syllogismData_full.csv
```

Le premier est converti en dataset CCOBRA E1 et le second en deux
datasets E2, correspondant aux phases intuitive et réfléchie.


## 3. Organisation du dépôt

```text
.
├── config/
│   └── benchmarks/             # Configurations CCOBRA
├── data/
│   ├── raw/                    # Données originales
│   ├── processed/              # Données converties au format CCOBRA
│   └── cache/                  # Prédictions et caches mReasoner
├── mreasoner/                  # Interface Python commune vers mReasoner
├── results/
│   ├── analysis/               # Figures produites
│   ├── benchmarks/             # Sorties des benchmarks
│   ├── logs/                   # Paramètres ajustés par participant
│   └── tables/                 # Résultats tabulaires
└── src/
    ├── analysis/
    │   ├── confidence/         # Analyses de confiance E1 et E2
    │   ├── consensus/          # Analyses de consensus
    │   └── entropy/            # Analyses d'entropie
    ├── models/
    │   ├── pymreasoner_conditional/
    │   └── pymreasoner_syllogistic/
    └── scripts/                # Conversion et analyses post-hoc
```

Toutes les commandes présentées ci-dessous doivent être lancées depuis
la racine du dépôt.

## 4. Installation

### Création d'un environnement virtuel

```bash
python3 -m venv venv
source venv/bin/activate
```

### Installation des dépendances Python

Les dépendances principales sont notamment :

- `ccobra`
- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `tqdm`

Elles peuvent être installées avec :

```bash
python3 -m pip install \
    ccobra \
    numpy \
    pandas \
    scipy \
    scikit-learn \
    matplotlib \
    seaborn \
    tqdm
```

Il est recommandé de conserver les versions exactes utilisées dans un
fichier `requirements.txt`.

### Vérification de Clozure Common Lisp

mReasoner nécessite Clozure Common Lisp. L'installation locale est
normalement placée dans :

```text
.ccl/
```

Les sources Lisp de mReasoner sont placées dans :

```text
.mreasoner/
```

## 5. Conversion des données E1 et E2

### E1

Le script :

```text
src/scripts/convert_E1_to_ccobra.py
```

lit :

```text
data/raw/E3_syllogismData_full.csv
```

et produit :

```text
data/processed/dataset_ccobra_E1.csv
```

Commande :

```bash
python3 src/scripts/convert_E1_to_ccobra.py
```

Le fichier produit contient notamment :

- l'identifiant du participant ;
- le numéro de l'essai ;
- les prémisses traduites au format CCOBRA ;
- la réponse `Yes` ou `No` ;
- la correction de la réponse ;
- la confiance ;
- les variables expérimentales disponibles.

### E2

Le script :

```text
src/scripts/convert_E2_to_ccobra.py
```

lit :

```text
data/raw/E4_syllogismData_full.csv
```

et produit :

```text
data/processed/dataset_ccobra_E2_int.csv
data/processed/dataset_ccobra_E2_ref.csv
```

Commande :

```bash
python3 src/scripts/convert_E2_to_ccobra.py
```

Les deux fichiers contiennent les mêmes essais :

- `dataset_ccobra_E2_int.csv` utilise `response_int` comme réponse cible ;
- `dataset_ccobra_E2_ref.csv` utilise `response_ref` comme réponse cible.

Les variables de confiance et de temps de réponse sont conservées même
lorsqu'elles sont manquantes. Un essai n'est supprimé que si une
information essentielle à l'ajustement est absente.

## 7. Benchmarks CCOBRA

Les fichiers de configuration se trouvent dans :

```text
config/benchmarks/
```

Ils permettent d'ajuster les paramètres de mReasoner pour chaque individu.

L'exécution d'un benchmark génère aussi un fichier html disponible dans `results/benchmarks/` qui analyse la correspondance des choix de l'individu aux paramètres choisis par mReasoner.

### Benchmark Ragni2016

```bash
ccobra config/benchmarks/benchmark.json
```

Ce benchmark utilise :

```text
data/raw/Ragni2016.csv
src/models/pymreasoner_syllogistic/ccobra_mreasoner_cache.py
```

Le modèle syllogistique utilise un cache de prédictions pour éviter de
réexécuter inutilement mReasoner pour les mêmes combinaisons de tâches
et de paramètres.

### Benchmark E1

```bash
ccobra config/benchmarks/benchmark_E1.json
```

Ce benchmark utilise :

```text
data/processed/dataset_ccobra_E1.csv
src/models/pymreasoner_conditional/ccobra_mreasoner_E1.py
```

Les prédictions sont mémorisées dans :

```text
data/cache/pymreasoner_conditional/
```

Les paramètres individuels ajustés sont enregistrés dans :

```text
results/logs/full_log_E1.json
```

### Benchmark E2 — phase intuitive

```bash
ccobra config/benchmarks/benchmark_E2_int.json
```

Entrée :

```text
data/processed/dataset_ccobra_E2_int.csv
```

Journal attendu :

```text
results/logs/log_full_E2_int.json
```

### Benchmark E2 — phase réfléchie

```bash
ccobra config/benchmarks/benchmark_E2_ref.json
```

Entrée :

```text
data/processed/dataset_ccobra_E2_ref.csv
```

Journal attendu :

```text
results/logs/log_full_E2_ref.json
```

Les benchmarks intuitif et réfléchi peuvent partager le même cache de
prédictions, car une prédiction mReasoner dépend de la tâche et des
paramètres, et non du nom de la phase expérimentale.

### Variante intuitive avec Système 1 fixé

```bash
ccobra \
    config/benchmarks/benchmark_E2_int_fixed_system1.json
```

Cette variante utilise :

```text
src/models/pymreasoner_conditional/
    ccobra_mreasoner_E2_intuitive_fixed_system1.py
```

Elle force le paramètre configuré pour désactiver la recherche de
modèles alternatifs dans la condition intuitive.

Le journal associé est :

```text
results/logs/log_full_E2_int_fixed_system1.json
```

Cette variante doit utiliser un cache distinct du benchmark intuitif
standard, afin d'éviter toute ambiguïté entre les deux configurations.

## 8. Comptage post-hoc des modèles mentaux

Les scripts de comptage relisent les paramètres ajustés par CCOBRA,
interrogent mReasoner avec `query_trace()` et comptent les occurrences
de l'objet Lisp :

```text
#<Q-MODEL>
```

Les caches post-hoc permettent de réutiliser les simulations déjà
effectuées.

On peut visualiser le résultat du comptage des modèles avec `results/benchmarks/dashboard_mental_count.html`

### E1

Version avec 10 simulations :

```bash
python3 src/scripts/count_mental_models_E1_n10.py
```

Les résultats sont placés dans :

```text
results/tables/mental_models/
```

Ils comprennent :

- un fichier détaillé par participant et par tâche ;
- un résumé par participant ;
- un cache JSON des simulations post-hoc.

### E2 standard

Après avoir exécuté les benchmarks intuitif et réfléchi :

```bash
python3 src/scripts/count_mental_models_E2.py
```

Ce script :

1. vérifie que les datasets intuitif et réfléchi contiennent les mêmes
   essais ;
2. charge les deux journaux de paramètres ;
3. estime séparément le nombre de modèles pour chaque phase ;
4. calcule la différence entre la phase réfléchie et la phase intuitive ;
5. produit des tableaux détaillés, par participant et par type de tâche.

### E2 avec la variante Système 1 fixé

Après le benchmark correspondant :

```bash
python3 src/scripts/count_mental_models_E2_int_fixed_system1.py
```

Ce script utilise les paramètres de la variante intuitive fixée et les
compare aux paramètres réfléchis.


## 9. Analyse de consensus

Les scripts de consensus se trouvent dans :

```text
src/analysis/consensus/
```

Ils utilisent principalement :

```text
data/raw/Ragni2016.csv
results/tables/mental_models/mental_models_count.csv
```


```bash
python3 src/analysis/consensus/plot_correlation.py
```


```bash
python3 src/analysis/consensus/plot_quadrants.py
```

Les figures sont enregistrées dans :

```text
results/analysis/consensus/
```

## 10. Analyse d'entropie

Les scripts se trouvent dans :

```text
src/analysis/entropy/
```

Ils utilisent les réponses du dataset Ragni2016 et les nombres de
modèles mentaux calculés en amont.

Commandes :

```bash
python3 src/analysis/entropy/plot_entropy_correlation.py
python3 src/analysis/entropy/plot_3d_accuracy.py
python3 src/analysis/entropy/plot_kmeans_3d_k2.py
python3 src/analysis/entropy/plot_kmeans_3d_k3.py
python3 src/analysis/entropy/plot_statistical_quadrants.py
```

Les figures sont enregistrées dans :

```text
results/analysis/entropy/
```

## 11. Analyse de confiance

### E1

Après la conversion, le benchmark et le comptage post-hoc :

```bash
python3 src/analysis/confidence/E1/plot_quadrant.py
```

Les sorties sont placées dans :

```text
results/analysis/confidence/E1/
```

### E2 standard

Après les benchmarks intuitif et réfléchi et le comptage post-hoc :

```bash
python3 src/analysis/confidence/E2/plot_quadrant.py
```

Les sorties sont placées dans :

```text
results/analysis/confidence/E2/
```

### E2 avec Système 1 fixé

Pour la variante intuitive fixée :

```bash
python3 src/analysis/confidence/E2/plot_quadrant_2.py
```


### Confiance E1

```bash
python3 src/scripts/convert_E1_to_ccobra.py
ccobra config/benchmarks/benchmark_E1.json
python3 src/scripts/count_mental_models_E1.py
python3 src/analysis/confidence/E1/plot_quadrant.py
```

Pour une estimation avec davantage de simulations :

```bash
python3 src/scripts/count_mental_models_E1_n10.py
```

### Confiance E2 standard

```bash
python3 src/scripts/convert_E2_to_ccobra.py

ccobra config/benchmarks/benchmark_E2_int.json
ccobra config/benchmarks/benchmark_E2_ref.json

python3 src/scripts/count_mental_models_E2.py
python3 src/analysis/confidence/E2/plot_quadrant.py
```

### Confiance E2 avec Système 1 fixé

La phase réfléchie peut être réutilisée si son journal existe déjà.







