# GAM/GAMM

## Sommaire

- [Rappel](#rappel)
- [Exécution rapide](#exécution-rapide)
- [1. Justification du choix du GAM](#1-justification-du-choix-du-gam)
- [2. Python : fonctionnement du GAM avec pyGAM](#2-python--fonctionnement-du-gam-avec-pygam)
- [4. Limites de Python et passage à R](#4-les-limites-de-python-et-le-passage-à-r)
- [5. GAMM ajusté avec R](#5-gamm-ajusté-avec-r)
- [6. Analyse des résultats du GAMM](#6-analyse-des-résultats-du-gamm)
- [7. Effets aléatoires](#7-les-effets-aléatoires)
- [8. Composantes retournées par gam.vcomp](#8-point-sur-gamvcomp)
- [9. Fichiers produits par le GAMM](#9-fichiers-produits-par-le-gamm)
- [10. Vérification de la dimension des bases](#10-vérification-de-la-dimension-des-bases)
- [11. Conclusion du modèle exploratoire](#11-conclusion-du-modèle-exploratoire)
- [Explication détaillée du k-index](#explication-plus-détaillée-du-k-index)


## Rappel

Nous cherchons à expliquer la **confiance**, mesurée entre 0 et 100, à partir de plusieurs variables :

- la condition expérimentale : `Neutral` ou `Standard` ;
- la position de l’essai dans l’expérience ;
- la précision moyenne du participant ;
- l’entropie de l’item ;
- le nombre moyen de modèles mentaux générés par le participant ;
- la variation du nombre de modèles chez un même participant ;
- la validité logique de l’essai.

Les données ont une structure particulière :

- **9 024 observations** ;
- **141 participants** ;
- **128 items** ;
- chaque participant répond à plusieurs items ;
- chaque item est présenté à plusieurs participants.

Nous avions d’abord utilisé un **modèle linéaire mixte** (voir [computational-model](../linear-mixed-model/)). Ce modèle supposait que les relations entre les prédicteurs numériques et la confiance étaient des droites.

Nous avons ensuite utilisé un **GAM**, puis un véritable **GAMM**, afin de vérifier si certaines de ces relations étaient en réalité courbées ou plus complexes.

---

## Exécution rapide

Toutes les commandes doivent être lancées depuis la racine du dépôt.

### 1. Vérifier la préparation des données pour Python

```bash
python3 src/analysis/computational-model/gam/prepare_gam_data_E1_n20.py
```

Ce script vérifie les données, construit les variables standardisées et encode les facteurs nécessaires à pyGAM. Il ne produit pas directement de fichier de résultats.

### 2. Ajuster le GAM exploratoire avec Python

```bash
python3 src/analysis/computational-model/gam/fit_gam_model_E1_n20.py

python3 src/analysis/computational-model/gam/fit_gam_model_parsimonious_E1_n20.py
```

Cette étape produit les courbes exploratoires obtenues avec pyGAM. Les participants et les items y sont représentés par des facteurs pénalisés et non par des effets aléatoires dont la variance est explicitement estimée.

Les résultats se trouvent ici : 
- [gamm_model_E1_n20](../../../../results/analysis/computational-model/gam/gam_model_E1_n20/)
- [gamm_parsimonious_E1_n20](../../../../results/analysis/computational-model/gam/gam_parsimonious_E1_n20/)

### 3. Ajuster le GAMM avec R

```bash
Rscript src/analysis/computational-model/gam/fit_gamm_exploratory_E1_n20.R
```
Cette étape constitue l’analyse principale. Elle estime :

- les fonctions lisses ;
- leurs paramètres de pénalisation ;
- les intercepts aléatoires des participants et des items ;
- les diagnostics du modèle.

Les résultats se trouvent ici : [gamm_exploratory_E1_n20](../../../../results/analysis/computational-model/gam/gamm_exploratory_E1_n20/)

---

# 1. Justification du choix du GAM

## 1.1 Limite du modèle linéaire

Le modèle linéaire impose une relation de la forme :

\[
f(x)=\beta x.
\]

Ce n’est pas forcément réaliste

---

## 1.2 Définition GAM ?

GAM signifie **Generalized Additive Model**, ou **modèle additif généralisé**.

Le GAM remplace certains coefficients linéaires par des fonctions C-infini (smooth functions) :

Il est dit additif parce qu’il additionne les contributions des différents termes.

\[
Y
=
\beta_0
+\beta_1X_1
+s_2(X_2)
+s_3(X_3)
+\varepsilon.
\]

La notation \(s(X)\) désigne une fonction C-infini estimée à partir des données.

Dans notre cas, le modèle pouvait s’écrire :

\[
\begin{aligned}
\text{confiance}_{ij}={}&
\beta_0
+\beta_1\text{condition}_i
+\beta_2\text{validité}_{ij}\\
&+s_1(\text{séquence}_{ij})
+s_2(\text{précision}_i)\\
&+s_3(\text{entropie}_j)
+s_4(\text{modèles moyens}_i)\\
&+s_5(\text{modèles intra}_{ij})
+\varepsilon_{ij}.
\end{aligned}
\]

---

## 1.3 GAMM définition

Un GAM ne contient pas nécessairement d’effets aléatoires.

Un **GAMM**, ou *Generalized Additive Mixed Model*, combine :

- des fonctions lisses ;
- des effets aléatoires.

Le modèle que nous voulons réellement utiliser est donc :

\[
\begin{aligned}
\text{confiance}_{ij}={}&
\beta_0
+\beta_1\text{condition}_i
+\beta_2\text{validité}_{ij}\\
&+s_1(\text{séquence}_{ij})
+s_2(\text{précision}_i)\\
&+s_3(\text{entropie}_j)
+s_4(\text{modèles moyens}_i)\\
&+s_5(\text{modèles intra}_{ij})\\
&+u_i+v_j+\varepsilon_{ij},
\end{aligned}
\]

avec :

\[
u_i\sim\mathcal N(0,\sigma^2_{\text{participant}}),
\]

\[
v_j\sim\mathcal N(0,\sigma^2_{\text{item}}),
\]

\[
\varepsilon_{ij}\sim\mathcal N(0,\sigma^2_\varepsilon).
\]

C’est l’extension naturelle de notre modèle linéaire mixte :

---

# 2. Python : fonctionnement du GAM avec `pyGAM`

## 2.1 Préparation des données

Nous avons commencé par créer :

```text
prepare_gam_data_E1_n20.py
```

Son rôle est de préparer correctement les données et de vérifier que le GAM reçoit une matrice exploitable.

Il réalise les opérations suivantes :

1. charger `dataset_analysis_E1_n20.csv` ;
2. vérifier les colonnes nécessaires ;
3. filtrer les lignes complètes ;
4. contrôler que la confiance était comprise entre 0 et 100 ;
5. centrer la séquence ;
6. standardiser les prédicteurs numériques ;
7. encoder les variables catégorielles ;
8. construire la matrice \(X\) et la variable \(y\).

Ce code ne génère pas de fichiers : il est appelé ensuite par les autre codes.

---

## 2.2 La matrice des prédicteurs

`pyGAM` ne reçoit pas une formule comme en R. Il reçoit une matrice numérique \(X\).

Nous avons défini neuf colonnes :

| Indice | Variable |
|---:|---|
| 0 | `condition_code` |
| 1 | `sequence_c10` |
| 2 | `subject_accuracy_z` |
| 3 | `item_entropy_z` |
| 4 | `subject_mean_models_z` |
| 5 | `models_within_subject_z` |
| 6 | `validity_binary` |
| 7 | `subject_code` |
| 8 | `item_code` |

Une ligne de \(X\) correspond à un essai.

---

## 2.3 Standardisation de certaines variables 

Nous avons standardisé plusieurs prédicteurs :

\[
Z=\frac{X-\overline{X}}{SD(X)}.
\]

Cela concerne :

- `subject_accuracy` ;
- `item_entropy` ;
- `subject_mean_models` ;
- `models_within_subject`.

Après standardisation :

- \(Z=0\) représente la moyenne ;
- \(Z=1\) représente un écart-type au-dessus de la moyenne ;
- \(Z=-1\) représente un écart-type sous la moyenne.

Cela facilite :

- la comparaison des effets ;
- la stabilité numérique ;
- l’interprétation des effets linéaires ;
- la cohérence avec le modèle mixte précédent.

---

## 2.4 Les trois types de termes dans `pyGAM`

### Terme catégoriel : `f()`

Nous avons utilisé `f()` pour :

- la condition ;
- la validité ;
- le participant ;
- l’item.

Le modèle ne traite donc pas la condition comme une quantité continue.

### Terme linéaire : `l()`

Nous avons utilisé `l()` lorsque nous voulions imposer une droite :

### Terme lisse : `s()`

Nous avons utilisé `s()` lorsque nous voulions autoriser une courbe :

La contribution devient :

\[
s(\text{sequence}).
\]

---

## 2.5 Spline

`pyGAM` ne teste pas toutes les fonctions C-infini possibles car cela serait trop coûteux en ressource. A la place, il utilise des **spline** qui sont des fonctions définies par morceaux par des polynomes. 

Mathématiquement :

\[
s(x)
=
\sum_{k=1}^{K}\alpha_k B_k(x).
\]

Dans cette formule :

- \(B_k(x)\) est la \(k\)-ième fonction de base ;
- \(\alpha_k\) est son coefficient ;
- \(K\) dépend du nombre de fonctions de base autorisées.

---

## 2.7 Pénalité des courbes 

Sans contrainte, le modèle pourrait produire une courbe qui suit chaque fluctuation accidentelle des données. Ce serait du surajustement.

L’estimation minimise donc un compromis :

\[
\underbrace{
\sum_{i=1}^{n}
\left(
y_i-\widehat{y}_i
\right)^2
}_{\text{erreur d'ajustement}}
+
\underbrace{
\lambda\boldsymbol{\alpha}^{\mathsf T}
S\boldsymbol{\alpha}
}_{\text{pénalité de complexité}}.
\]

Dans cette expression :

- \(\boldsymbol{\alpha}\) contient les coefficients de la spline ;
- \(S\) est une matrice mesurant sa rugosité ;
- \(\lambda\) contrôle la force de la pénalisation.

Si \(\lambda\) est faible, la courbe peut être très flexible.

Si \(\lambda\) est élevé, la courbe est fortement lissée.

Si \(\lambda\) est très élevé; la partie courbée peut pratiquement disparaître et la relation devenir presque linéaire.

---

## 2.8 Premier modèle `pyGAM` entièrement flexible

Nous avons commencé avec :

```python
LinearGAM(
    f(0)
    + s(1)
    + s(2)
    + s(3)
    + s(4)
    + s(5)
    + f(6)
    + f(7)
    + f(8)
)
```

Les pénalisations étaient fixées manuellement :

```python
SPLINE_LAMBDA = 10
GROUP_FACTOR_LAMBDA = 10
```

Ce modèle était donc volontairement exploratoire.

---

## 2.9 Résultats globaux du premier modèle Python

Le premier `pyGAM` produisait :

```text
Pseudo R-Squared = 0,4391
Effective DoF = 243,73 (effective degrees of freedom)
```

Il représentait donc environ 43,9 % de la variabilité observée sur les données utilisées pour l’ajustement.

Une grande partie de sa complexité venait de :

- l’effet participant : environ 107 degrés de liberté effectifs ;
- l’effet item : environ 100 degrés de liberté effectifs.

Cela indiquait que les différences entre participants et entre items occupaient une part importante du modèle.

### Interprétation des degrés de liberté effectifs

Les **degrés de liberté effectifs**, ou `edf`, indiquent la complexité réellement utilisée par un terme après pénalisation.

Ils ne correspondent pas simplement au nombre de coefficients présents dans le programme. 

En pratique :

- un `edf` proche de 1 correspond à une relation presque linéaire ;
- un `edf` compris entre 1 et 2 indique une légère courbure ;
- un `edf` plus élevé indique une forme plus complexe, avec davantage de changements de pente ;
- un `edf` proche de sa limite maximale suggère que la spline utilise presque toute la flexibilité autorisée.

---

## 2.10 Les courbes exploratoires de Python

Les courbes sont disponibles dans [results](../../../../results/analysis/computational-model/gam/) après exécution du programme `fit_gam_model_E1_n20.py`

### Séquence

La courbe montre :

- une forte diminution au début ;
- une légère remontée vers le milieu ;
- une nouvelle diminution ;
- une remontée dans les derniers essais.

Entre le premier et le dernier essai, la contribution diminuait d’environ trois points.

### Entropie

La relation est :

- monotone ;
- décroissante ;
- régulière ;
- proche d’une droite.

Entre les valeurs extrêmes observées, la contribution diminuait d’environ 9 à 10 points.

### Précision

La courbe ressemble à une forme en U :

- confiance élevée chez certains participants peu précis ;
- confiance plus faible autour de la précision moyenne ;
- confiance fortement élevée chez les participants très précis.

Cette forme peut être psychologiquement intéressante. Cependant, les extrémités ne sont représentées que par un très petit nombre de participants.

### Nombre moyen de modèles

La courbe présente également une forme en U très marquée.

Mais cette variable ne prend qu’un nombre limité de valeurs bien représentées. Entre ces valeurs, la spline interpole souvent dans des régions ne contenant aucun participant.

### Composante intra-individuelle

La courbe présente une forme en cloche :

- augmentation jusqu’à un maximum ;
- puis diminution.

Mais les intervalles sont larges et le support des données était discontinu.

---

## 2.11 GAM parcimonieux 

Nous avons voulu vérifier si les résultats plus stables persistaient en imposant une forme linéaire aux prédicteurs dont les courbes semblaient fragiles.

Le modèle est :

```python
LinearGAM(
    f(condition)
    + s(sequence)
    + l(subject_accuracy)
    + s(item_entropy)
    + l(subject_mean_models)
    + l(models_within_subject)
    + f(validity)
    + f(subject)
    + f(item)
)
```

Cela ne signifiait pas que nous avions démontré que la précision ou le nombre de modèles étaient linéaires. C’était seulement une spécification de comparaison plus simple.

---

## 2.12 Effets paramétriques du GAM parcimonieux

Nous avons obtenu :

| Prédicteur | Effet ajusté |
|---|---:|
| Standard − Neutral | \(+5{,}20\) |
| Valid − Invalid | \(+0{,}89\) |
| Précision, +1 ET | \(+0{,}70\) |
| Nombre moyen de modèles, +1 ET | \(-2{,}24\) |
| Modèles intra-individuels, +1 ET | \(-0{,}31\) |

Ces valeurs étaient presque identiques à celles du modèle linéaire mixte.

**Cela montre que l’introduction de courbes pour la séquence et l’entropie ne bouleversait pas les autres estimations.**

---

# 4. Les limites de Python et le passage à R

## 4.1 Bibliothèque

Dans `pyGAM`, nous avions écrit :

```python
f(subject_code, lam=10)
f(item_code, lam=10)
```

Cela créait un coefficient pour chaque participant et chaque item, avec une pénalisation L2 :

\[
\lambda\sum_i u_i^2.
\]

Le logiciel ne permettait pas de faire un modèle mixte et n’estimait donc pas explicitement :

\[
\sigma^2_{\text{participant}}
\]

et :

\[
\sigma^2_{\text{item}}.
\]


## 4.2 Pénalisation

Le choix arbitraire de la pénalisation participant déterminait en partie quelle quantité de variation était attribuée :

- à la précision ;
- au nombre de modèles ;
- aux différences individuelles résiduelles.

Le même problème existait entre :

# 4.3 item et participation dans les courbes données


---

## 4.4 `mgcv` en R est plus approprié

Nous sommes passés au package R `mgcv`, qui permet d’écrire :

```r
s(subject_id, bs = "re")
s(item_id, bs = "re")
```

Ces termes représentent de véritables effets aléatoires.

Le modèle estime :

\[
u_i\sim\mathcal N(0,\sigma^2_{\text{participant}})
\]

et :

\[
v_j\sim\mathcal N(0,\sigma^2_{\text{item}}).
\]

En parallèle, le logiciel estime séparément le lissage de chaque courbe.

Nous n’imposons donc plus :

```text
lambda = 10
```

à tous les termes.

L’estimation par REML avec `gam()` — choisit la régularisation à partir des données.

---

# 5. GAMM ajusté avec R

## 5.1 Formule utilisée

Le modèle est :

```r
confidence ~
    condition
    + validity
    + s(sequence_c10, k = 10)
    + s(subject_accuracy_z, k = 6)
    + s(item_entropy_z, k = 8)
    + s(subject_mean_models_z, k = 5)
    + s(models_within_subject_z, k = 5)
    + s(subject_id, bs = "re")
    + s(item_id, bs = "re")
```

Il contient :

### Effets catégoriels

- condition ;
- validité

### Effets lisses

- séquence ;
- précision ;
- entropie ;
- nombre moyen de modèles ;
- nombre de modèles intra-individuel.

### Effets aléatoires

- participant ;
- item.

---

### `k`

Dans :

```r
s(subject_accuracy_z, k = 6)
```

le paramètre \(k\) définit la dimension maximale de la base utilisée pour construire la courbe.

---

# 6. Analyse des résultats du GAMM

## 6.1 Ajustement général

Le modèle donne :

```text
R² ajusté = 0,428
Déviance expliquée = 44,2 %
```
## 6.2 Condition expérimentale

\[
\widehat{\beta}_{\text{Standard}}
=
4{,}67.
\]

\[
p=0{,}062.
\]

> Toutes les autres variables étant contrôlées, la confiance est estimée environ 4,67 points plus élevée en condition Standard qu’en condition Neutral.


Cet effet peut également être lié au phénomène de plafond déjà identifié : la condition Standard produit davantage de réponses égales à 100.

---

## 6.3 Validité

\[
\widehat{\beta}_{\text{Valid}}
=
0{,}84.
\]

\[
p=0{,}199.
\]

---

## 6.4 Séquence

Résultat : 

```text
edf = 4,094
F = 7,303
p < 0,001
```

L’`edf` de 4,094 montre que la relation est réellement non linéaire.

Une droite ne suffit donc pas à décrire l’évolution de la confiance au cours de l’expérience.

Le GAMM confirme la forme observée avec `pyGAM` :

- baisse importante au début ;
- évolution plus stable au milieu ;
- fluctuations en fin d’expérience.

> La confiance évolue significativement au cours de l’expérience, mais cette évolution n’est pas une diminution parfaitement constante.

---

## 6.5 Précision du participant

Résultat :

```text
edf = 2,239
F = 2,252
p = 0,0958
```

L’edf supérieur à 2 montre que la forme estimée possède une courbure. Le véritable GAMM ne réduit donc pas complètement la précision à une droite.

Cela suggère que la forme en U observée en Python n’était peut-être pas entièrement artificielle.

\[
p=0{,}0958.
\]

La relation reste incertaine.

> Le modèle suggère une relation non linéaire entre précision et confiance, mais cette relation ne dispose pas encore d’un soutien statistique suffisamment fort pour être considérée comme établie.

---

## 6.6 Entropie de l’item

Résultat :

```text
edf = 1,009
F = 73,916
p < 0,001
```

L’`edf` est presque exactement égal à 1. La relation est donc pratiquement linéaire.

\[
p<0{,}001.
\]

La relation est très probable. 

---

## 6.7 Nombre moyen de modèles du participant

Résultat :

```text
edf = 1,001
F = 1,088
p = 0,296
```

L’`edf` est pratiquement égal à 1. Le véritable GAMM ne confirme donc pas la forte forme en U observée dans le premier `pyGAM`.

La relation estimée est essentiellement linéaire.

\[
p=0{,}296.
\]

Elle n’est pas clairement associée à la confiance.

Cela suggère que la forme en U de Python venait probablement de plusieurs éléments :

- une pénalisation arbitraire des participants ;
- une distribution discontinue de la variable ;
- la difficulté de séparer la spline du facteur participant.


> Le nombre moyen de modèles présente éventuellement une tendance linéaire négative, mais le GAMM ne fournit pas de preuve claire d’une association avec la confiance et ne confirme pas une forme en U.

---

## 6.8 Composante intra-individuelle du nombre de modèles

Résultat :

```text
edf = 1,665
F = 1,373
p = 0,275
```

L’`edf` supérieur à 1 indique une légère courbure. Elle est cependant beaucoup moins marquée que dans le premier modèle Python.

Le terme n’est pas clairement associé à la confiance :

\[
p=0{,}275.
\]

La forme en cloche observée précédemment n’est donc pas suffisamment robuste.

---

# 7. Les effets aléatoires

Les écarts-types estimés sont :

| Composante | Écart-type |
|---|---:|
| Participant | 13,74 |
| Item | 2,25 |
| Résiduel | 16,86 |

Les variances correspondantes sont obtenues en mettant les écarts-types au carré.

## Variance participant

\[
13{,}74^2\approx188{,}76.
\]

## Variance item

\[
2{,}25^2\approx5{,}07.
\]

## Variance résiduelle

\[
16{,}86^2\approx284{,}27.
\]

La somme vaut environ :

\[
188{,}76+5{,}07+284{,}27
=
478{,}10.
\]

Les proportions approximatives sont donc :

| Composante | Proportion |
|---|---:|
| Participant | 39,5 % |
| Item | 1,1 % |
| Résiduelle | 59,5 % |

La confiance dépend fortement de la manière générale dont chaque participant utilise l’échelle.

---

# 8. Point sur `gam.vcomp()`

Le tableau contient aussi des lignes comme :

```text
s(sequence_c10)
s(subject_accuracy_z)
s(item_entropy_z)
```

Leurs écarts-types ne doivent pas être interprétés comme des composantes de variance comparables à :

```text
s(subject_id)
s(item_id)
scale
```

Pour les splines, ces valeurs décrivent la régularisation de la partie lisse.

Par exemple, pour l’entropie :

```text
std.dev = 0,076
```

ne signifie pas que l’entropie a un petit effet.

Au contraire, son association est très forte. Cette petite valeur indique surtout que la partie courbée de la spline est presque totalement supprimée, parce que la relation est pratiquement linéaire.

Pour la décomposition de la variance, il faut donc se concentrer sur :

- `s(subject_id)` ;
- `s(item_id)` ;
- `scale`.

---

# 9. Fichiers produits par le GAMM

Pour rappel, les résultats sont enregistrés dans [gamm_exploratory_E1_n20/](../../../../results/analysis/computational-model/gam/gamm_exploratory_E1_n20/)

| Fichier | Contenu | Utilité |
| --- | --- | --- |
| gamm_exploratory_summary.txt | Coefficients, tests des fonctions lisses et qualité d’ajustement | Résultat principal |
| gamm_exploratory_smooth_effects.png | Représentation des cinq fonctions lisses | Interprétation des formes |
| gamm_exploratory_smooth_effects.csv | Valeurs numériques des courbes et intervalles à 95 % | Reproductibilité et analyse détaillée |
| gamm_exploratory_variance_components.csv | Paramètres de lissage et écarts-types des effets aléatoires | Décomposition participant, item et résiduelle |
| gamm_exploratory_k_check.csv | Vérification de la dimension des bases | Diagnostic de la flexibilité des splines |

---

# 10. Vérification de la dimension des bases

Le fichier `gamm_exploratory_k_check.csv` vérifie si la dimension \(k\) choisie pour chaque fonction lisse est suffisante.

| Colonne | Signification |
|---|---|
| `k'` | Nombre maximal approximatif de degrés de liberté disponibles |
| `edf` | Complexité effectivement utilisée après pénalisation |
| `k-index` | Indice recherchant une structure non représentée dans les résidus |
| `p-value` | Résultat d’un test par permutation associé au `k-index` |

Un problème est suspecté lorsque les trois conditions suivantes sont réunies :

- le `k-index` est nettement inférieur à 1 ;
- la valeur \(p\) est faible ;
- l’`edf` est proche de `k'`.

Les résultats obtenus sont :

| Terme | `k'` | `edf` | `k-index` | Valeur \(p\) |
|---|---:|---:|---:|---:|
| Séquence | 9 | 4,094 | 0,983 | 0,090 |
| Précision | 5 | 2,239 | 1,017 | 0,875 |
| Entropie | 7 | 1,009 | 0,999 | 0,445 |
| Nombre moyen de modèles | 4 | 1,001 | 0,995 | 0,338 |
| Modèles intra-individuels | 4 | 1,665 | 0,985 | 0,178 |

Les `k-index` sont proches de 1 et les `edf` restent nettement inférieurs à leur maximum. Aucune fonction ne présente donc de signe clair indiquant que \(k\) est trop faible.

Les valeurs sont absentes pour les effets aléatoires participant et item, car ce diagnostic n’est pas défini pour les facteurs.

---

# 11. Conclusion du modèle exploratoire

Le GAMM met principalement en évidence :

1. une évolution non linéaire de la confiance au cours de l’expérience ;
2. une association négative robuste et presque linéaire avec l’entropie ;
3. une forte variabilité générale entre participants.

Les autres relations sont plus incertaines :

- la précision présente une possible forme en U ;
- le nombre moyen de modèles suit une tendance négative non clairement détectée ;
- la composante intra-individuelle possède un effet faible ;
- la validité n’est pas clairement associée à la confiance ;
- la condition Standard présente une tendance positive.

Les diagnostics n’indiquent pas que les valeurs de \(k\) sont insuffisantes.

---

# Explication plus détaillée du `k-index`

## 1. Le problème que le `k-index` cherche à détecter

Quand on écrit :

```r
s(sequence_c10, k = 10)
```

on limite la capacité maximale de la spline.

Si \(k\) est trop petit, la fonction ne peut pas devenir suffisamment flexible pour représenter la vraie relation. Même après ajustement, il reste alors une structure dans les résidus.

---

## 2. Principe du calcul

Après l’ajustement, chaque observation possède un résidu :

\[
e_i=y_i-\widehat y_i.
\]

Si le modèle a correctement représenté la relation, deux observations proches sur l’axe du prédicteur ne devraient pas avoir des résidus systématiquement semblables.

`k.check()` compare alors deux estimations de la variance résiduelle. (voir [documentation](https://rdrr.io/cran/mgcv/man/k.check.html))

### Variance résiduelle globale

\[
\widehat{\sigma}^2
=
\frac{\sum_i e_i^2}{df_{\text{résiduel}}}.
\]

### Variance fondée sur les différences entre voisins

Les observations proches pour le prédicteur sont comparées. De manière simplifiée :

\[
\widehat{\sigma}^2_{\text{voisins}}
\approx
\frac{1}{2M}
\sum_{(i,j)\text{ voisins}}
(e_i-e_j)^2.
\]

Le facteur \(1/2\) vient du fait que, si deux résidus indépendants possèdent chacun une variance \(\sigma^2\), alors :

\[
\operatorname{Var}(e_i-e_j)=2\sigma^2.
\]

Le `k-index` est approximativement :

\[
\text{k-index}
=
\frac{
\widehat{\sigma}^2_{\text{voisins}}
}{
\widehat{\sigma}^2
}.
\]

---

## 3. Interprétation

### `k-index` proche de 1

\[
\widehat{\sigma}^2_{\text{voisins}}
\approx
\widehat{\sigma}^2.
\]

Les résidus des observations voisines ne sont pas particulièrement similaires. Il ne reste pas de structure locale évidente.

C’est le résultat attendu.

### `k-index` inférieur à 1

Les résidus proches sont plus similaires que prévu :

\[
(e_i-e_j)^2
\]

est généralement petit.

Cela suggère que le modèle n’a pas capturé toute la structure associée au prédicteur. Une cause possible est un \(k\) trop faible.

### `k-index` supérieur à 1

Les résidus voisins diffèrent légèrement plus que prévu. Cela n’est généralement pas interprété comme un manque de flexibilité.

---

## 4. Rôle de la p-value

La p-value est obtenue par permutation :

1. les résidus sont mélangés aléatoirement ;
2. le `k-index` est recalculé ;
3. l’opération est répétée plusieurs fois ;
4. le résultat observé est comparé à cette distribution aléatoire.

Une petite valeur \(p\) indique que le `k-index` observé est anormalement faible.

Un problème de \(k\) devient convaincant lorsque :

- le `k-index` est nettement inférieur à 1 ;
- la p-value est faible ;
- l’`edf` est proche de `k'`.

---



