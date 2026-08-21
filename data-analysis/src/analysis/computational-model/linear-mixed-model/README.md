# Modèle de l’expérience E1

## Sommaire

[Présentation](#présentation)

[Structure générale du pipeline](#1-structure-générale-du-pipeline)
   - [Scripts archivés](#note--scripts-archivés)

[Dépendances Python](#2-dépendances-python)

[Fichiers d'entrée](#3-fichiers-dentrée)
   - [Données expérimentales](#31-données-expérimentales)
   - [Sorties de MReasoner](#32-sorties-de-mreasoner)

[Construction du dataset analytique](#4-construction-du-dataset-analytique)
   - [Variables principales construites](#variables-principales-construites)
   - [Précision du participant](#précision-du-participant)
   - [Entropie de l'item](#entropie-de-litem)
   - [Nombre moyen de modèles du participant](#nombre-moyen-de-modèles-du-participant)
   - [Composante intra-individuelle](#composante-intra-individuelle)
   - [Indicateur de plafond](#indicateur-de-plafond)
   - [Sorties](#sorties)

[Modèle mixte nul](#5-modèle-mixte-nul)
   - [Modèle](#modèle)
   - [Objectif](#objectif)
   - [Résultats principaux](#résultats-principaux)
   - [Sorties](#sorties-1)

[Modèle mixte de contrôle](#6-modèle-mixte-de-contrôle)
   - [Modèle](#modèle-1)
   - [Résultats principaux](#résultats-principaux-1)
   - [Sorties](#sorties-2)

[Modèle mixte cognitif principal](#7-modèle-mixte-cognitif-principal)
   - [Modèle](#modèle-2)
   - [Estimation](#estimation)
   - [Résultats principaux avec validité](#résultats-principaux-avec-validité)
   - [Sorties](#sorties-3)

[Analyses de sensibilité](#8-analyses-de-sensibilité)
   - [Objectifs](#objectifs-1)
   - [Modèles comparés](#modèles-comparés)
   - [Résultats principaux](#résultats-principaux-2)
   - [Sorties](#sorties-4)

[Analyse de sensibilité au plafond](#9-analyse-de-sensibilité-au-plafond)
    - [Problème étudié](#problème-étudié)
    - [Résultats principaux sous le plafond](#résultats-principaux-sous-le-plafond)
    - [Sorties](#sorties-5)

[Modèle logistique de l'utilisation de 100](#10-modèle-logistique-de-lutilisation-de-100)
    - [Variable dépendante](#variable-dépendante)
    - [Modèle](#modèle-3)
    - [Interprétation](#interprétation)
    - [Résultats principaux](#résultats-principaux-3)
    - [Sorties](#sorties-6)

[Exécution du pipeline](#11-exécution-du-pipeline)

[Organisation finale des résultats](#12-organisation-des-résultats-finale)

[Synthèse des résultats scientifiques](#13-synthèse-des-résultats-scientifiques)
    - [Entropie de l'item](#131-entropie-de-litem)
    - [Condition expérimentale](#132-condition-expérimentale)
    - [Progression dans l'expérience](#133-progression-dans-lexpérience)
    - [MReasoner](#134-mreasoner)
    - [Différences entre participants](#135-différences-entre-participants)

[Conclusion](#14-conclusion)


## Présentation

Ce dossier contient le pipeline utilisé pour analyser la **confiance exprimée dans l’expérience E1**.

Cette analyse fait partie du projet général.  
Pour consulter la présentation globale du projet, voir le
[README principal](../../../README.md).


L’analyse cherche à déterminer si la confiance varie en fonction :

- de la condition expérimentale, `Standard` ou `Neutral` ;
- de la position de l’essai dans l’expérience ;
- de la précision moyenne du participant ;
- de l’entropie des réponses à l’item ;
- du nombre de modèles mentaux estimé par MReasoner ;
- de la validité ou du type de tâche ;
- de l’utilisation fréquente de la borne maximale de confiance, égale à 100.

Les données comprennent :

| Élément | Nombre |
|---|---:|
| Observations | 9024 |
| Participants | 141 |
| Items | 128 |
| Essais par participant | 64 |
| Types de tâche | MP, MT, AC et DA |
| Conditions | Standard et Neutral |

Les modèles statistiques utilisent des intercepts aléatoires croisés pour les participants et les items. Cette structure tient compte du fait que chaque participant répond à plusieurs items et que chaque item est présenté à plusieurs participants.

---

# 1. Structure générale du pipeline

Le pipeline principal contient les scripts suivants :

```text
build_dataset_analysis_E1.py
fit_null_mixed_model_E1.py
fit_control_mixed_model_E1.py
fit_cognitive_mixed_model_E1_n20.py
fit_sensitivity_mixed_model_E1.py
fit_ceiling_sensitivity_E1_n20.py
fit_ceiling_logistic_mixed_model_E1.py
```

Ils sont exécutés dans cet ordre :

```text
Données expérimentales + sorties MReasoner
                    │
                    ▼
       Construction du dataset analytique
                    │
                    ▼
              Modèle mixte nul
                    │
                    ▼
           Modèle mixte de contrôle
                    │
                    ▼
          Modèle mixte cognitif principal
                    │
                    ▼
          Analyses de sensibilité
                    │
                    ▼
      Analyse linéaire sous le plafond
                    │
                    ▼
       Modèle logistique de la valeur 100
```
## Note : Scripts archivés

Le dossier `archive/` contient les scripts utilisés pendant les phases
exploratoires ou les analyses complémentaires, mais qui ne font pas
partie du pipeline principal final.

Ils sont conservés pour assurer la traçabilité du projet et permettre
de retrouver les étapes antérieures de l'analyse.

---

# 2. Dépendances Python

Les scripts ont été développés avec Python 3 et utilisent principalement :

- `numpy` ;
- `pandas` ;
- `scipy` ;
- `statsmodels`.

Installation minimale :

```bash
python3 -m pip install numpy pandas scipy statsmodels
```

Il est recommandé d’utiliser un environnement virtuel :

```bash
python3 -m venv cogsci-env
source cogsci-env/bin/activate
python3 -m pip install numpy pandas scipy statsmodels
```

---

# 3. Fichiers d’entrée

## 3.1 Données expérimentales

```text
dataset_ccobra_E1.csv
```

Ce fichier contient une ligne par essai expérimental, avec notamment :

- l’identifiant du participant ;
- la position de l’essai ;
- l’identifiant de l’item ;
- la condition ;
- le type de tâche ;
- la réponse ;
- l’exactitude ;
- la confiance.

## 3.2 Sorties de MReasoner

```text
mental_models_count_E1_n20.csv
```

Ce fichier contient une ligne par combinaison :

```text
participant × type de tâche
```

Les estimations sont fondées sur **20 simulations de MReasoner**.

Le fichier comprend notamment :

- le nombre moyen de modèles générés ;
- l’écart-type entre simulations ;
- le minimum ;
- le maximum ;
- le nombre de simulations.

La fusion avec les essais expérimentaux est réalisée à partir de :

```text
subject_id + task_type
```

et non à partir de la position de l’essai.

---

# 4. Construction du dataset analytique

## Script

```text
build_dataset_analysis_E1.py
```

## Objectif

Ce script :

1. charge les données expérimentales ;
2. charge les résultats de MReasoner ;
3. vérifie les identifiants et la structure des tâches ;
4. calcule les variables analytiques ;
5. fusionne les deux sources ;
6. vérifie que toutes les observations sont exploitables ;
7. crée le dataset utilisé par les modèles.

## Variables principales construites

### Précision du participant

```text
subject_accuracy
```

Proportion moyenne de réponses correctes du participant :

\[
\text{subject accuracy}_i
=
\frac{\text{nombre de réponses correctes du participant }i}
{\text{nombre total de ses essais}}
\]

### Entropie de l’item

```text
item_entropy
```

Entropie binaire de Shannon calculée à partir de la répartition des réponses `Yes` et `No` :

\[
H(p)
=
-p\log_2(p)
-(1-p)\log_2(1-p)
\]

- \(H=0\) : accord presque total ;
- \(H=1\) : répartition proche de 50/50.

L’entropie mesure donc le **désaccord collectif** suscité par l’item.

### Nombre moyen de modèles du participant

```text
subject_mean_models
```

Moyenne des estimations MReasoner du participant sur les quatre types de tâche.

### Composante intra-individuelle

```text
models_within_subject
```

Différence entre le nombre de modèles associé au type de tâche courant et la moyenne personnelle :

\[
M^{\text{within}}_{it}
=
M_{it}
-
\overline{M}_i
\]

Cette décomposition permet de séparer :

- les différences générales entre participants ;
- les variations entre types de tâche pour un même participant.

### Indicateur de plafond

```text
is_ceiling
```

Codage binaire :

\[
\text{is ceiling}
=
\begin{cases}
1 & \text{si confiance}=100\\
0 & \text{sinon}
\end{cases}
\]

## Sorties

```text
results/tables/computational-model/linear-mixed-model/mixed/dataset_analysis_E1_n20.csv
results/analysis/computational-models/analysis_E1_outputs/data_audit_E1.txt
```

Si une erreur est détectée, le script peut également produire un fichier contenant uniquement les lignes problématiques.

---

# 5. Modèle mixte nul

## Script

```text
fit_null_mixed_model_E1.py
```

## Modèle

\[
\text{confidence}_{ij}
=
\beta_0
+
u_i
+
v_j
+
\varepsilon_{ij}
\]

où :

- \(\beta_0\) est la confiance moyenne générale ;
- \(u_i\) est l’intercept aléatoire du participant ;
- \(v_j\) est l’intercept aléatoire de l’item ;
- \(\varepsilon_{ij}\) est la variation résiduelle.

## Objectif

Le modèle nul sert à déterminer où se situe la variabilité de la confiance avant d’introduire les prédicteurs.

Il est ajusté :

- en **REML** pour estimer les composantes de variance ;
- en **ML** pour servir de référence aux comparaisons ultérieures.

## Résultats principaux

| Composante | Variance approximative | Proportion |
|---|---:|---:|
| Participant | 201 | 40,3 % |
| Item | 11,9 | 2,4 % |
| Résiduelle | 285 | 57,3 % |

La confiance moyenne générale est proche de :

\[
75{,}7
\]

La variance participant est beaucoup plus grande que la variance item. Les participants diffèrent donc fortement dans leur manière générale d’utiliser l’échelle de confiance.

## Sorties

Dans `results/analysis/computational-model/linear-mixed-model/null_mixed_model_E1_n20/` :
```text
null_model_REML_summary.txt
null_model_variance_components.csv
null_model_fit_statistics.csv
```

---

# 6. Modèle mixte de contrôle

## Script

```text
fit_control_mixed_model_E1.py
```

## Modèle

\[
\text{confidence}
\sim
\text{condition}
+
\text{sequence\_c10}
\]

avec les mêmes intercepts aléatoires participant et item.

La condition `Neutral` est la référence.

La séquence est transformée par :

\[
\text{sequence\_c10}
=
\frac{\text{sequence}-\overline{\text{sequence}}}{10}
\]

Le coefficient de séquence représente donc l’évolution moyenne de la confiance pour **dix essais supplémentaires**.

## Résultats principaux

| Prédicteur | Estimation approximative | Interprétation |
|---|---:|---|
| Standard vs Neutral | +5,15 | Confiance plus élevée en Standard |
| Séquence, par 10 essais | -0,43 | Légère diminution de confiance |

Le modèle de contrôle améliore clairement le modèle nul :

\[
LR\approx24{,}47,\qquad p<0{,}001
\]

## Sorties

Dans `results/analysis/computational-model/linear-mixed-model/control_mixed_model_E1_n20/` : 

```text
control_model_REML_summary.txt
control_model_fixed_effects.csv
model_comparison.csv
control_model_metrics.csv
```

---

# 7. Modèle mixte cognitif principal

## Script

```text
fit_cognitive_mixed_model_E1_n20.py
```

## Modèle

À cette étape, le modèle cognitif contient encore la validité :

\[
\begin{aligned}
\text{confidence}\sim{}&
\text{condition}
+\text{sequence\_c10}\\
&+\text{subject accuracy}_z\\
&+\text{item entropy}_z\\
&+\text{subject mean models}_z\\
&+\text{models within subject}_z\\
&+\text{validity binary}
\end{aligned}
\]

Les prédicteurs continus cognitifs sont standardisés :

\[
Z
=
\frac{X-\overline X}{SD(X)}
\]

Leur coefficient représente donc l’association avec la confiance pour une augmentation d’un écart-type.

## Estimation

Trois modèles sont ajustés en ML :

1. modèle nul ;
2. modèle de contrôle ;
3. modèle cognitif.

Le modèle cognitif est ensuite ajusté en REML pour produire les coefficients principaux.

## Résultats principaux avec validité

Les valeurs sont approximativement :

| Prédicteur | Estimation | p-value | Conclusion |
|---|---:|---|---|
| Standard vs Neutral | +5,3 | 0,03 |Effet positif |
| Séquence | -0,44 | 0,00 | Diminution au cours des essais |
| Précision du participant | +0,7 | 0,66 | Effet non clairement détecté |
| Entropie de l’item | -2,4 | 0,00 |Effet négatif très robuste |
| Nombre moyen de modèles | -2,2 | 0,15 | Effet non clairement détecté |
| Modèles intra-individuels | -0,3 | 0,19 |Effet faible et incertain |
| Validité | 0,7 | 0,24 | Effet non clairement détecté |

L’ajout du bloc cognitif améliore clairement le modèle de contrôle. Cette amélioration est principalement portée par l’entropie des items.

## Sorties

Dans `results/analysis/computational-model/linear-mixed-model/cognitive_mixed_model_E1_n20`

```text
cognitive_model_REML_summary.txt
cognitive_model_fixed_effects.csv
model_fit_statistics.csv
likelihood_ratio_tests.csv
cognitive_model_metrics.csv
```

---

# 8. Analyses de sensibilité

## Script

```text
fit_sensitivity_mixed_model_E1.py
```

## Objectifs

Ce script vérifie :

1. si le bloc cognitif améliore le modèle de contrôle ;
2. si la validité améliore le modèle cognitif ;
3. si le type de tâche améliore le modèle cognitif ;
4. quelle est la contribution propre de chaque prédicteur.

## Modèles comparés

```text
Control
Cognitive_without_validity
Cognitive_validity
Cognitive_task_type
```

Les comparaisons sont réalisées en ML.

## Résultats principaux

| Comparaison | LR approximatif | ddl | Valeur \(p\) |
|---|---:|---:|---:|
| Contrôle vs cognitif sans validité | 73,1 | 4 | \(< 0,001\) |
| Cognitif sans validité vs validité | 1,2 | 1 | 0,27 |
| Cognitif sans validité vs type de tâche | 3,8 | 3 | 0,28 |

La validité et le type de tâche n’améliorent donc pas clairement le modèle cognitif.

Le modèle final plus parcimonieux est ainsi retenu **sans validité et sans type de tâche**.

Les tests `drop-one` montrent que le retrait de l’entropie dégrade fortement le modèle, alors que le retrait des autres prédicteurs produit des changements plus faibles.

## Sorties

```text
global_likelihood_ratio_tests.csv
drop_one_tests.csv
model_fit_comparison.csv
task_type_model_fixed_effects.csv
task_type_model_REML_summary.txt
```

---

# 9. Analyse de sensibilité au plafond

## Script

```text
fit_ceiling_sensitivity_E1.py
```

## Problème étudié

Une proportion importante des observations se trouve à la valeur maximale (environ 25%):

\[
\text{confidence}=100
\]

Dans les données :

| Mesure | Valeur |
|---|---:|
| Observations totales | 9024 |
| Réponses égales à 100 | 2336 |
| Taux au plafond | 25,9 % |
| Réponses inférieures à 100 | 6688 |

Le modèle linéaire est donc réajusté après exclusion des réponses égales à 100.

## Résultats principaux sous le plafond

| Prédicteur | Estimation approximative | p-value |Conclusion |
|---|---:|---|---|
| Standard vs Neutral | +2,1 | 0,36 | Non clairement détecté |
| Séquence | +0,061 | 0,63 |Non clairement détectée |
| Précision | 0,69 | 0,63 | Non clairement détectée |
| Entropie | -2,3 | 0,0 | Effet négatif robuste |
| Nombre moyen de modèles | -2,9 | 0,046 |Tendance négative |
| Modèles intra-individuels | -0,29 | 0,26 |Non clairement détecté |

L’effet de l’entropie ne dépend donc pas uniquement des réponses situées à 100.

En revanche, les effets de condition et de séquence semblent être largement liés à l’utilisation de la borne maximale.

## Sorties

Dans `results/analysis/computational-model/linear-mixed-model/ceiling_sensitivity_E1_n20/` :

```text
ceiling_summary.csv
below_ceiling_fixed_effects.csv
below_ceiling_REML_summary.txt
```

---

# 10. Modèle logistique de l’utilisation de 100

## Script

```text
fit_ceiling_logistic_mixed_model_E1.py
```

## Variable dépendante

\[
\text{at ceiling}
=
\begin{cases}
1 & \text{si confidence}=100\\
0 & \text{sinon}
\end{cases}
\]

## Modèle

\[
\begin{aligned}
\operatorname{logit}
\left[
P(\text{at ceiling}=1)
\right]
\sim{}&
\text{condition}
+\text{sequence\_c10}\\
&+\text{subject accuracy}_z\\
&+\text{item entropy}_z\\
&+\text{subject mean models}_z\\
&+\text{models within subject}_z
\end{aligned}
\]

Le modèle contient des intercepts aléatoires pour les participants et les items.

Il est estimé avec l’approximation variationnelle bayésienne de :

```python
BinomialBayesMixedGLM.fit_vb()
```

## Interprétation

Les coefficients sont exprimés en log-odds.

Les odds ratios sont calculés par :

\[
OR=e^\beta
\]

- \(OR>1\) : augmentation des odds d’utiliser 100 ;
- \(OR<1\) : diminution des odds ;
- \(OR=1\) : absence d’association.

## Résultats principaux

| Prédicteur | Odds ratio approximatif | Interprétation |
|---|---:|---|
| Standard vs Neutral | 3,96 | Odds d’utiliser 100 presque quatre fois plus élevées |
| Séquence | 0,85 | Diminution d’environ 15 % des odds par 10 essais |
| Précision participant | 0,83 | Association négative |
| Entropie | 0,74 | Diminution d’environ 26 % des odds |
| Nombre moyen de modèles | 1,17 | Légère association positive |
| Modèles intra-individuels | 0,92 | Légère association négative |

Les taux bruts sont approximativement :

| Condition | Taux de réponses égales à 100 |
|---|---:|
| Neutral | 19,2 % |
| Standard | 32,5 % |

Les probabilités ajustées au milieu de la séquence sont approximativement :

| Condition | Probabilité ajustée |
|---|---:|
| Neutral | 8,1 % |
| Standard | 25,8 % |

Les résultats de ce modèle mettent en exergue les paramètres conditionnels : on observe une "fatigue" du participant face à 64 items à répondre et il apparaît aussi qu'un participant répondant aux items Standard est plus à même d'avoir entièrement confiance dans ses réponses.  

## Sorties
Dans `results/analysis/computational-model/linear-mixed-model/ceiling_logistic_mixed_model_E1_n20/` :

```text
ceiling_by_condition.csv
ceiling_logistic_fixed_effects.csv
adjusted_ceiling_probabilities.csv
ceiling_logistic_model_summary.txt
```

> Les intervalles produits par ce modèle sont des intervalles crédibles approximatifs issus d’une approximation variationnelle. Ils doivent être interprétés avec prudence.

---

# 11. Exécution du pipeline

Les scripts doivent être exécutés dans l’ordre suivant :

```bash
python3 build_dataset_analysis_E1.py
python3 fit_null_mixed_model_E1.py
python3 fit_control_mixed_model_E1.py
python3 fit_cognitive_mixed_model_E1_n20.py
python3 fit_sensitivity_mixed_model_E1.py
python3 fit_ceiling_sensitivity_E1.py
python3 fit_ceiling_logistic_mixed_model_E1.py
```

Chaque script :

- vérifie ses fichiers d’entrée ;
- contrôle les colonnes nécessaires ;
- teste la convergence ;
- essaie plusieurs optimiseurs si nécessaire ;
- écrit ses résultats dans un dossier dédié.

---

# 12. Organisation des résultats finale

```text
results/
├── tables/
│   └── computational-model/
│       ├── dataset_analysis_E1_n20.csv
│       └── analysis_E1_outputs/
│           └── data_audit_E1.txt
│
└── analysis/
    └── computational-model/
        └── linear-mixed-model/
            ├── null_mixed_model_E1_n20/
            ├── control_mixed_model_E1_n20/
            ├── cognitive_mixed_model_E1_n20/
            ├── sensitivity_mixed_model_E1_n20/
            ├── ceiling_sensitivity_E1_n20/
            └── ceiling_logistic_mixed_model_E1_n20/
```
---

# 13. Synthèse des résultats scientifiques

## 13.1 Entropie de l’item

Le résultat le plus robuste est l’association négative entre l’entropie et la confiance :

> Les participants sont moins confiants pour les items qui suscitent une plus grande dispersion des réponses `Yes` et `No`.

L’effet reste présent :

- dans le modèle cognitif ;
- après contrôle de la validité ;
- après contrôle du type de tâche ;
- après exclusion des réponses égales à 100 ;
- dans le modèle logistique de l’utilisation de 100.

## 13.2 Condition expérimentale

La condition Standard est associée à une confiance globale plus élevée.

Cependant, cette différence semble être principalement portée par une utilisation plus fréquente de la valeur maximale 100.

## 13.3 Progression dans l’expérience

La confiance moyenne diminue au cours de l’expérience.

L’analyse du plafond indique que cette diminution correspond surtout à une réduction progressive de l’utilisation de la réponse 100.

## 13.4 MReasoner

Le nombre moyen de modèles mentaux du participant n’est pas clairement associé à la confiance.

La composante intra-individuelle présente une direction généralement négative : pour un même participant, un type de tâche associé à davantage de modèles que sa moyenne personnelle tend à produire une confiance légèrement plus faible.

Cet effet reste toutefois modeste et sensible à la spécification du modèle.

## 13.5 Différences entre participants

Environ 40 % de la variance aléatoire de la confiance est associée aux différences générales entre participants.

Les différences individuelles d’utilisation de l’échelle sont donc beaucoup plus importantes que les différences résiduelles moyennes entre items.

---

# 14. Conclusion

L’analyse met en évidence trois résultats principaux :

1. **L’entropie des réponses est le prédicteur le plus robuste de la confiance.**
2. **La condition Standard augmente principalement l’utilisation de la confiance maximale.**
3. **L’utilisation de la valeur 100 diminue au cours de l’expérience.**

Les résultats liés à MReasoner sont plus faibles :

- l’effet interindividuel n’est pas clairement détecté ;
- l’effet intra-individuel est négatif mais dépend davantage de la spécification statistique.
