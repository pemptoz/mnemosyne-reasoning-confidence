# Étape 1 — Analyse préliminaire et choix de la stratégie statistique

## Sommaire de cette étape

1. [But général de l’analyse préliminaire](#1-but-général-de-lanalyse-préliminaire)
2. [Le phénomène scientifique étudié](#2-le-phénomène-scientifique-étudié)
3. [Structure des données expérimentales](#3-structure-des-données-expérimentales)
4. [Les variables disponibles au départ](#4-les-variables-disponibles-au-départ)
5. [Les variables qu’il fallait encore calculer](#5-les-variables-quil-fallait-encore-calculer)
6. [Transformer une question scientifique en modèle statistique](#6-transformer-une-question-scientifique-en-modèle-statistique)
7. [Pourquoi envisager une régression ?](#7-pourquoi-envisager-une-régression)
8. [La régression linéaire ordinaire](#8-la-régression-linéaire-ordinaire)
9. [Pourquoi une régression ordinaire ne suffisait pas](#9-pourquoi-une-régression-ordinaire-ne-suffisait-pas)
10. [Le modèle linéaire mixte](#10-le-modèle-linéaire-mixte)
11. [Effets fixes et effets aléatoires](#11-effets-fixes-et-effets-aléatoires)
12. [Pourquoi utiliser des effets croisés participant–item ?](#12-pourquoi-utiliser-des-effets-croisés-participantitem)
13. [Décomposer les variables interindividuelles et intra-individuelles](#13-décomposer-les-variables-interindividuelles-et-intra-individuelles)
14. [Les objectifs scientifiques formulés au départ](#14-les-objectifs-scientifiques-formulés-au-départ)
15. [Les modèles envisagés et l’ordre prévu](#15-les-modèles-envisagés-et-lordre-prévu)
16. [Les solutions alternatives](#16-les-solutions-alternatives)
17. [Les hypothèses du modèle linéaire mixte](#17-les-hypothèses-du-modèle-linéaire-mixte)
18. [Les difficultés anticipées](#18-les-difficultés-anticipées)
19. [Ce que l’analyse préliminaire nous a appris](#19-ce-que-lanalyse-préliminaire-nous-a-appris)
20. [Pourquoi cette étape conduit à la création de `dataset_analysis_E1.csv`](#20-pourquoi-cette-étape-conduit-à-la-création-de-dataset_analysis_e1csv)

---

# 1. But général de l’analyse préliminaire

Avant d’écrire un modèle statistique ou un script Python, il fallait répondre à une question fondamentale :

> **Quelle structure statistique correspond réellement à nos données et à nos questions scientifiques ?**

Il aurait été possible de commencer immédiatement par calculer des corrélations ou d’ajuster une régression. Cela aurait toutefois été dangereux, car un modèle statistique n’est pas une simple machine à produire des nombres.

Un modèle doit représenter correctement :

- ce que mesure chaque ligne du fichier ;
- quelles observations proviennent du même participant ;
- quelles observations concernent le même item ;
- quelles variables varient entre participants ;
- quelles variables varient entre essais ;
- quelles variables sont liées entre elles par construction ;
- quelles hypothèses scientifiques nous souhaitons tester.

L’analyse préliminaire avait donc quatre fonctions.

## 1.1 Comprendre les données

Nous devions identifier :

- le nombre de participants ;
- le nombre d’essais par participant ;
- le nombre d’items ;
- la nature des conditions expérimentales ;
- les types de tâches ;
- les variables déjà présentes ;
- les variables absentes qu’il faudrait calculer.

## 1.2 Formuler les questions scientifiques

Nous ne voulions pas seulement prédire la confiance. Nous voulions comprendre pourquoi elle varie.

Nos questions générales étaient notamment :

1. Les participants diffèrent-ils durablement dans leur façon d’utiliser l’échelle de confiance ?
2. Certains items produisent-ils systématiquement plus ou moins de confiance ?
3. La confiance varie-t-elle selon la condition expérimentale ?
4. Diminue-t-elle ou augmente-t-elle au fil de l’expérience ?
5. Les participants les plus précis sont-ils aussi les plus confiants ?
6. Les items qui suscitent davantage de désaccord produisent-ils une confiance plus faible ?
7. Le nombre de modèles mentaux générés par MReasoner est-il associé à la confiance ?
8. La validité logique ou le type de tâche modifient-ils la confiance ?

## 1.3 Choisir une méthode statistique adaptée

Nous avons d’abord envisagé une **régression**, parce que nous cherchions à expliquer une variable numérique, la confiance, à partir de plusieurs prédicteurs.

Nous avons ensuite constaté que les observations n’étaient pas indépendantes :

- un même participant fournissait 64 réponses ;
- un même item était présenté à environ 70 ou 71 participants.

Cette structure nous a conduits vers un **modèle linéaire mixte**, avec des effets aléatoires pour les participants et les items.

## 1.4 Préparer un pipeline reproductible

Un **pipeline** est une succession organisée d’étapes de traitement.

On peut l’imaginer comme une chaîne de fabrication :

```text
Fichiers bruts
      ↓
Nettoyage et vérification
      ↓
Calcul des variables
      ↓
Fusion des fichiers
      ↓
Dataset analytique
      ↓
Modèles statistiques
      ↓
Diagnostics
      ↓
Analyses de sensibilité
      ↓
Rapport final
```

L’objectif était de pouvoir reconstruire les résultats automatiquement, plutôt que de modifier manuellement des tableaux.

---

# 2. Le phénomène scientifique étudié

## 2.1 La confiance comme variable principale

À chaque essai, un participant répondait à un problème de raisonnement puis indiquait son niveau de confiance sur une échelle allant de 0 à 100.

Par exemple :

```text
Réponse au problème : Yes
Confiance : 80
```

La confiance est ici une mesure subjective.

Elle ne nous dit pas directement si la réponse est correcte. Elle indique à quel point le participant pense, ou ressent, que sa réponse est fiable.

Deux participants peuvent donner la même réponse, tout en ayant des niveaux de confiance différents :

```text
Participant A : Yes, confiance = 95
Participant B : Yes, confiance = 55
```

De même, une réponse incorrecte peut être donnée avec une forte confiance :

```text
Réponse incorrecte, confiance = 100
```

C’est précisément cette distinction qui rend la confiance scientifiquement intéressante.

---

## 2.2 Confiance, exactitude et métacognition

La **métacognition** désigne la capacité à évaluer ses propres processus cognitifs.

Dans notre contexte, cela signifie notamment :

> Le participant sait-il reconnaître les moments où sa réponse est probablement correcte ou incorrecte ?

Il faut distinguer trois concepts.

### Exactitude

L’**exactitude**, ou précision, indique si la réponse donnée est objectivement correcte.

```text
is_correct = 1  → réponse correcte
is_correct = 0  → réponse incorrecte
```

### Confiance

La **confiance** est l’évaluation subjective fournie par le participant.

```text
confidence = 90
```

### Calibration

La **calibration** décrit l’accord entre la confiance et l’exactitude.

Par exemple, si un participant donne une confiance moyenne de 80 % et répond correctement dans environ 80 % des cas, il est bien calibré.

S’il donne une confiance moyenne de 90 % mais n’a que 60 % de réponses correctes, il est surconfiant.

---

## 2.3 Deux personnes peuvent avoir la même précision mais une confiance différente

Imaginons deux participants ayant chacun 40 réponses correctes sur 64 :

\[
\text{précision}=\frac{40}{64}=0{,}625
\]

Le premier utilise l’échelle de manière prudente :

```text
Confiance moyenne = 60
```

Le second l’utilise de manière très affirmée :

```text
Confiance moyenne = 90
```

Leur performance est identique, mais leur comportement métacognitif est très différent.

Cela explique pourquoi il était important de tenir compte des différences propres à chaque participant.

---

# 3. Structure des données expérimentales

## 3.1 Une ligne correspond à un essai

Dans le fichier expérimental initial, une ligne représentait une observation de type :

```text
un participant × un essai × un item
```

Par exemple :

| participant | essai | item | réponse | confiance | correct |
|---:|---:|---:|---|---:|---:|
| 63873 | 1 | 125 | No | 99 | 0 |
| 63873 | 2 | 17 | Yes | 100 | 1 |
| 63873 | 3 | 49 | Yes | 100 | 0 |

Le fichier comprenait finalement :

- **9 024 lignes** ;
- **141 participants** ;
- **64 essais par participant** ;
- **128 items distincts**.

Nous pouvons vérifier le nombre total d’observations :

\[
141\times64=9\,024
\]

Le plan était donc complet au niveau des participants : chaque participant avait exactement 64 essais analysés.

---

## 3.2 Un même participant apparaît plusieurs fois

Un participant apparaît sur 64 lignes.

Par exemple :

```text
Participant 63873
├── essai 1
├── essai 2
├── essai 3
├── ...
└── essai 64
```

Ces 64 observations ne sont pas totalement indépendantes.

Pourquoi ?

Parce qu’elles proviennent de la même personne. Elles partagent probablement :

- son style d’utilisation de l’échelle ;
- sa prudence ;
- sa tendance à utiliser 100 ;
- son niveau général de confiance ;
- sa capacité de raisonnement ;
- sa motivation ;
- sa fatigue ;
- d’autres caractéristiques non mesurées.

Une personne très confiante aura tendance à donner plusieurs valeurs élevées, et pas une seule valeur élevée isolée.

---

## 3.3 Un même item apparaît également plusieurs fois

Chaque item a été présenté à environ 70 ou 71 participants.

Exemple conceptuel :

```text
Item 125
├── réponse du participant 63873
├── réponse du participant 64120
├── réponse du participant 64591
├── ...
└── réponse d'environ 70 participants
```

Ces observations partagent les caractéristiques de l’item :

- son contenu ;
- sa formulation ;
- sa difficulté ;
- son caractère intuitif ou contre-intuitif ;
- son niveau de désaccord ;
- sa structure logique.

Ainsi, les réponses données au même item ne sont pas non plus complètement indépendantes.

---

## 3.4 Une structure croisée

La structure est dite **croisée** parce que les participants répondent à plusieurs items et les items sont vus par plusieurs participants.

On peut la représenter ainsi :

```text
                     Items
               I1   I2   I3   I4   ...
Participant P1  ×    ×    ×    ×
Participant P2  ×    ×    ×    ×
Participant P3  ×    ×    ×    ×
...
```

Les regroupements participant et item se croisent.

Ce n’est pas simplement :

```text
un participant dans un item
```

ou :

```text
un item dans un participant
```

Chaque dimension traverse l’autre.

> **Analogie**
>
> Imaginons que plusieurs critiques évaluent plusieurs films.
>
> Une note dépend :
>
> - du critique, car certains critiques sont sévères ;
> - du film, car certains films sont meilleurs ;
> - de la rencontre particulière entre ce critique et ce film.
>
> Notre situation est similaire :
>
> - certains participants utilisent une confiance élevée ;
> - certains items inspirent plus de confiance ;
> - chaque essai contient encore une part particulière non expliquée.

---

# 4. Les variables disponibles au départ

Le fichier principal était `dataset_ccobra_E1.csv`.

Ses colonnes étaient :

```text
id
sequence
domain
response_type
task
choices
response
confidence
is_correct
task_type
condition
validity
believability
conflict
stimulus
qnum
total_qnum
rt
logRT
rt_for
statementEval
```

Nous allons distinguer les variables indispensables des variables secondaires.

---

## 4.1 L’identifiant du participant : `id`

La colonne `id` indiquait quel participant avait produit la réponse.

Elle a ensuite été renommée ou copiée sous le nom plus explicite :

```text
subject_id
```

Le terme anglais `subject` signifie ici « participant expérimental ».

Cette variable est indispensable pour reconnaître les lignes produites par une même personne.

Sans elle, nous serions incapables :

- de calculer la précision moyenne d’un participant ;
- de calculer sa moyenne de modèles mentaux ;
- d’estimer la variabilité entre participants ;
- d’ajouter un effet aléatoire participant.

---

## 4.2 La position de l’essai : `sequence`

`sequence` correspond à la position de l’essai dans la séquence du participant.

Elle varie de 1 à 64.

Exemple :

```text
sequence = 1   → premier essai
sequence = 32  → milieu approximatif
sequence = 64  → dernier essai
```

Cette variable permet d’étudier un possible **effet d’ordre**.

Un effet d’ordre est une modification du comportement au fil de l’expérience.

Il peut refléter :

- la fatigue ;
- l’apprentissage ;
- l’habituation ;
- une meilleure compréhension de la tâche ;
- une baisse d’attention ;
- une modification de l’utilisation de l’échelle.

---

## 4.3 La confiance : `confidence`

`confidence` est notre variable principale.

Elle varie entre 0 et 100.

En statistique, on appelle **variable dépendante** la variable que le modèle cherche à expliquer.

Ici :

\[
Y=\text{confidence}
\]

Le mot « dépendante » ne signifie pas qu’elle est nécessairement causée par toutes les autres variables. Il signifie simplement qu’elle est placée du côté de ce que nous voulons expliquer.

> **Analogie**
>
> Si nous cherchons à expliquer le prix d’une maison à partir de sa surface et de sa localisation :
>
> - prix = variable dépendante ;
> - surface et localisation = variables explicatives.
>
> Dans notre projet :
>
> - confiance = variable dépendante ;
> - condition, séquence, entropie et modèles mentaux = variables explicatives.

---

## 4.4 L’exactitude : `is_correct`

`is_correct` indique si la réponse est correcte.

```text
1 = correcte
0 = incorrecte
```

Il s’agit d’une variable **binaire**, c’est-à-dire une variable qui ne possède que deux valeurs possibles.

Elle permet :

- de calculer la précision moyenne de chaque participant ;
- de calculer la difficulté ou l’exactitude de chaque item ;
- d’étudier la calibration ;
- de comparer la confiance des réponses correctes et incorrectes.

---

## 4.5 La réponse : `response`

`response` contenait des réponses comme :

```text
Yes
No
```

Cette variable était textuelle. Pour certains calculs, il fallait la convertir en nombres :

```text
Yes → 1
No  → 0
```

Cette transformation permet notamment de calculer :

- le taux de réponses Yes ;
- le taux de réponses No ;
- l’entropie de l’item.

---

## 4.6 La condition : `condition`

La colonne `condition` distinguait :

```text
Standard
Neutral
```

### Condition Standard

Les items Standard utilisaient un contenu ordinaire et compréhensible. Les prémisses pouvaient être croyables ou non croyables, et la conclusion pouvait elle aussi entrer en accord ou en conflit avec les connaissances générales.

### Condition Neutral

Les items Neutral reprenaient la même structure, mais un terme était remplacé par un non-mot.

L’objectif était de réduire l’influence des croyances sémantiques sur les prémisses.

### Variable entre participants

La condition était une variable **interindividuelle**, ou **between-subjects**.

Cela signifie qu’un participant appartenait à une seule condition :

```text
Participant A → Standard uniquement
Participant B → Neutral uniquement
```

Nous avions :

- 71 participants Standard ;
- 70 participants Neutral.

La condition ne variait donc pas à l’intérieur d’un participant.

---

## 4.7 Le type de tâche : `task_type`

Les quatre types étaient :

```text
MP
MT
AC
DA
```

Ils correspondent à quatre formes d’inférence conditionnelle.

### Modus ponens — MP

Structure simplifiée :

```text
Si A, alors B.
A.
Donc B.
```

C’est une forme valide.

### Modus tollens — MT

```text
Si A, alors B.
Pas B.
Donc pas A.
```

C’est également une forme valide.

### Affirmation du conséquent — AC

```text
Si A, alors B.
B.
Donc A.
```

Cette inférence est invalide.

Le fait que B soit vrai ne prouve pas que A est la seule cause possible.

### Déni de l’antécédent — DA

```text
Si A, alors B.
Pas A.
Donc pas B.
```

Cette inférence est également invalide.

B pourrait être produit par une autre cause.

---

## 4.8 La validité : `validity`

La colonne `validity` indiquait si l’inférence était logiquement valide.

Elle a ensuite été transformée en :

```text
validity_binary
```

avec :

```text
Valid   → 1
Invalid → 0
```

Mais un problème important a été découvert :

| Type de tâche | Validité |
|---|---|
| MP | valide |
| MT | valide |
| AC | invalide |
| DA | invalide |

La validité est donc entièrement déterminée par le type de tâche.

Autrement dit :

```text
connaître task_type permet de connaître validity
```

Cette relation structurelle aura une conséquence importante :

> Nous ne devrons pas introduire simultanément la validité et l’ensemble des catégories de type de tâche sans précaution.

Sinon, le modèle recevrait deux descriptions fortement redondantes de la même structure.

Ce problème est lié à la **colinéarité**.

La colinéarité apparaît lorsque deux variables explicatives contiennent une information identique ou presque identique.

> **Analogie**
>
> Supposons que l’on tente d’expliquer une température avec :
>
> - la température en degrés Celsius ;
> - la même température convertie en degrés Fahrenheit.
>
> Ces deux variables semblent différentes, mais elles contiennent exactement la même information.
>
> Le modèle ne peut pas déterminer clairement laquelle porte l’effet.

---

## 4.9 La croyabilité : `believability`

`believability` indiquait si le contenu ou la conclusion était croyable ou non croyable.

Valeurs typiques :

```text
Believable
Unbelievable
```

Cette variable pouvait être scientifiquement intéressante, mais elle n’a pas été incluse immédiatement dans le modèle principal.

Pourquoi ?

Parce que nous voulions d’abord construire un modèle simple et contrôlable, avant d’ajouter de nombreux facteurs et interactions.

---

## 4.10 Le conflit : `conflict`

`conflict` indiquait si la logique et les croyances entraient en conflit.

Exemple conceptuel :

- la logique pousse vers une réponse ;
- les connaissances générales poussent vers une autre.

Valeurs typiques :

```text
Conflict
No-conflict
```

Cette variable pourrait être utilisée dans une analyse spécifique des effets de croyance, mais elle ne constituait pas le cœur initial de l’analyse du nombre de modèles mentaux et de la confiance.

---

## 4.11 L’identifiant de l’item : `total_qnum`

`total_qnum` identifiait les items du corpus principal, en comptant les items Standard et Neutral utilisés dans l’analyse.

Il a été renommé ou recopié comme :

```text
item_id
```

Ce point est important : `qnum` et `total_qnum` n’avaient pas la même fonction.

`item_id` devait permettre de reconnaître un item identique présenté à plusieurs participants.

Sans identifiant d’item fiable, il serait impossible de :

- calculer l’entropie par item ;
- calculer l’exactitude par item ;
- ajouter un effet aléatoire item ;
- vérifier combien de participants ont vu chaque item.

---

## 4.12 Le temps de réponse : `rt` et `logRT`

`rt` représentait le temps de réponse.

`logRT` était une transformation logarithmique de ce temps.

Le logarithme est souvent utilisé pour les temps de réponse, car leur distribution est généralement asymétrique :

- beaucoup de réponses relativement rapides ;
- quelques réponses extrêmement longues.

La transformation logarithmique comprime les grandes valeurs.

Exemple :

\[
\log(1\,000)\approx6{,}91
\]

\[
\log(10\,000)\approx9{,}21
\]

La valeur brute a été multipliée par 10, mais la valeur logarithmique n’augmente que modérément.

Les temps de réponse ont été conservés dans le dataset analytique, mais ils n’étaient pas les prédicteurs centraux du premier modèle.

---

# 5. Les variables qu’il fallait encore calculer

Le fichier brut ne contenait pas directement toutes les variables nécessaires.

Nous devions produire un fichier analytique enrichi.

---

## 5.1 La précision moyenne du participant

La variable `subject_accuracy` a été calculée pour chaque participant.

Si le participant \(i\) a réalisé \(n_i\) essais, sa précision est :

\[
\text{subject\_accuracy}_i
=
\frac{1}{n_i}
\sum_{j=1}^{n_i}
\text{is\_correct}_{ij}
\]

Comme `is_correct` vaut 0 ou 1, sa moyenne est directement une proportion.

### Exemple

Supposons cinq essais :

```text
1, 1, 0, 1, 0
```

La précision est :

\[
\frac{1+1+0+1+0}{5}
=
\frac{3}{5}
=
0{,}60
\]

Le participant a donc 60 % de réponses correctes.

### Pourquoi calculer cette variable ?

Nous voulions savoir si les participants globalement plus précis utilisent aussi différemment l’échelle de confiance.

Elle représente une propriété au niveau du participant.

Toutes les lignes d’un même participant reçoivent donc la même valeur :

| participant | essai | subject_accuracy |
|---|---:|---:|
| A | 1 | 0,625 |
| A | 2 | 0,625 |
| A | 3 | 0,625 |

---

## 5.2 La réponse normalisée

Les réponses textuelles ont été converties en catégories standardisées :

```text
Yes
No
```

Puis éventuellement en nombres :

```text
Yes → 1
No  → 0
```

On obtient notamment :

```text
response_normalized
response_binary
```

### Pourquoi ?

Les calculs mathématiques ne peuvent pas directement utiliser des chaînes de caractères comme `"Yes"` et `"No"` pour calculer une moyenne ou une entropie.

Une fois converties :

\[
\text{taux de Yes}
=
\frac{\text{nombre de Yes}}{\text{nombre total de réponses}}
\]

---

## 5.3 Le taux de réponses Yes par item

Pour chaque item \(j\), nous avons calculé :

\[
p_j
=
\frac{\text{nombre de réponses Yes à l’item }j}
{\text{nombre total de réponses à l’item }j}
\]

Cette variable est devenue :

```text
item_yes_rate
```

Le taux de réponses No est :

\[
1-p_j
\]

et correspond à :

```text
item_no_rate
```

### Exemple

Si 70 participants répondent à un item :

```text
49 Yes
21 No
```

alors :

\[
p(\text{Yes})=\frac{49}{70}=0{,}70
\]

\[
p(\text{No})=\frac{21}{70}=0{,}30
\]

---

## 5.4 L’entropie de l’item

### Définition intuitive

L’**entropie** mesure ici à quel point les réponses à un item sont partagées entre Yes et No.

- Si tout le monde répond la même chose, l’entropie est faible.
- Si les réponses sont divisées en deux groupes presque égaux, l’entropie est élevée.

> **Analogie**
>
> Imagine une salle où l’on demande de voter Yes ou No.
>
> - 100 % Yes : tout le monde est d’accord, très peu d’incertitude.
> - 50 % Yes et 50 % No : désaccord maximal.
>
> L’entropie transforme ce degré de désaccord en nombre.

### Formule

Pour un item binaire :

\[
H_j
=
-p_j\log_2(p_j)
-
(1-p_j)\log_2(1-p_j)
\]

où \(p_j\) est la proportion de réponses Yes.

La base 2 produit une entropie comprise entre 0 et 1 pour une réponse binaire.

### Cas 1 : consensus complet

Si tout le monde répond Yes :

\[
p=1
\]

alors :

\[
H=0
\]

Par convention, le terme \(0\log(0)\) est traité comme 0.

### Cas 2 : désaccord maximal

Si la moitié répond Yes et l’autre moitié No :

\[
p=0{,}5
\]

alors :

\[
H
=
-0{,}5\log_2(0{,}5)
-0{,}5\log_2(0{,}5)
=1
\]

### Cas 3 : 90 % contre 10 %

\[
H
=
-0{,}9\log_2(0{,}9)
-0{,}1\log_2(0{,}1)
\approx0{,}469
\]

Il existe un certain désaccord, mais bien moins qu’avec une division 50–50.

### Pourquoi utiliser l’entropie ?

Nous cherchions à savoir si la confiance est plus faible pour les items qui suscitent davantage d’incertitude collective.

L’hypothèse intuitive était :

\[
\text{entropie élevée}
\quad\Rightarrow\quad
\text{confiance plus faible}
\]

### Limite importante

L’entropie est calculée à partir des réponses du même échantillon.

Elle n’est donc pas une mesure indépendante et objective de la difficulté intrinsèque de l’item.

La formulation correcte est :

> L’entropie mesure le désaccord empirique observé entre les participants.

Il faut éviter de dire automatiquement :

> L’entropie mesure la difficulté objective de l’item.

---

## 5.5 L’exactitude moyenne de l’item

Pour chaque item, nous avons calculé :

\[
\text{item\_accuracy}_j
=
\frac{1}{n_j}
\sum_i \text{is\_correct}_{ij}
\]

Cette variable indique la proportion de participants ayant répondu correctement à cet item.

Exemple :

```text
60 réponses correctes sur 70
```

\[
\text{item\_accuracy}
=
\frac{60}{70}
\approx0{,}857
\]

Une valeur élevée signifie que l’item est souvent réussi.

Une valeur faible signifie qu’il est souvent échoué.

Cette variable est proche d’une mesure empirique de difficulté, mais elle dépend elle aussi de l’échantillon de participants.

---

## 5.6 La validité binaire

La variable textuelle :

```text
Valid
Invalid
```

a été transformée en :

```text
1
0
```

La transformation numérique facilite son introduction dans une formule statistique.

Le coefficient peut alors être interprété comme la différence moyenne entre :

```text
validity_binary = 1
```

et :

```text
validity_binary = 0
```

sous réserve des autres variables du modèle.

---

## 5.7 Le nombre de modèles mentaux générés

Les valeurs MReasoner étaient fournies dans un autre fichier :

```text
mental_models_count_E1.csv
```

Ce fichier comportait notamment :

```text
subject_id
task
premise_1
premise_2
number_models_generated
std_models_generated
minimum_models_generated
maximum_models_generated
n_samples
n_parameter_sets_used
epsilon
lambda
omega
sigma
```

### Que représente `number_models_generated` ?

Cette variable représente le nombre moyen de modèles mentaux produits par les simulations MReasoner pour :

```text
un participant × un type de tâche
```

Ce n’est pas une mesure exacte et directe, essai par essai.

Elle est produite par un modèle computationnel.

---

## 5.8 Qu’est-ce qu’un modèle computationnel ?

Un **modèle computationnel** est un ensemble explicite de règles et de paramètres implémentés sous forme d’algorithme.

Il cherche à simuler un mécanisme cognitif.

> **Analogie**
>
> Un modèle computationnel est comparable à une maquette fonctionnelle.
>
> Une maquette d’avion ne reproduit pas exactement un avion réel, mais elle permet de tester des principes :
>
> - forme des ailes ;
> - résistance ;
> - trajectoire.
>
> De même, MReasoner ne reproduit pas parfaitement un cerveau humain. Il formalise certaines hypothèses sur la façon dont des représentations mentales peuvent être construites.

Les paramètres :

```text
epsilon
lambda
omega
sigma
```

contrôlent certains comportements internes du modèle.

Les détails exacts de leur signification relèvent de la théorie MReasoner, mais dans notre pipeline statistique, ils ont surtout servi à identifier la configuration utilisée pour générer les estimations.

---

## 5.9 Le nombre de simulations

Initialement, chaque combinaison participant × tâche reposait sur trois simulations :

```text
n_samples = 3
```

La moyenne :

```text
number_models_generated
```

était donc calculée sur ces trois répétitions.

L’écart-type :

```text
std_models_generated
```

indiquait la variabilité entre les simulations.

### Exemple

Supposons que trois simulations donnent :

```text
2, 2, 3 modèles
```

La moyenne est :

\[
\bar{x}
=
\frac{2+2+3}{3}
=
2{,}333
\]

L’écart-type mesure à quel point les trois résultats s’écartent de leur moyenne.

Plus tard, nous avons répété les simulations avec 10 puis 20 répétitions afin de vérifier leur stabilité.

---

## 5.10 Comprendre la clé de fusion

Le fichier expérimental contenait 64 essais par participant.

Le fichier MReasoner ne contenait que quatre lignes par participant :

```text
une ligne MP
une ligne MT
une ligne AC
une ligne DA
```

La colonne `task` du fichier MReasoner n’était donc pas le numéro de l’essai expérimental.

Elle représentait l’index du type de tâche.

Il aurait été incorrect de fusionner les fichiers avec :

```text
subject_id + sequence
```

La fusion correcte devait utiliser :

```text
subject_id + task_type
```

Le type de tâche MReasoner a été déduit des prémisses :

| Prémisse 1 | Prémisse 2 | Type |
|---|---|---|
| All B are C | All A are B | MP |
| All B are C | No A are C | MT |
| All B are C | All A are C | AC |
| All B are C | No A are B | DA |

Ainsi, les 16 essais MP d’un participant recevaient la même estimation MReasoner associée à ce participant et au type MP.

---

# 6. Transformer une question scientifique en modèle statistique

## 6.1 Qu’est-ce qu’un modèle statistique ?

Un **modèle statistique** est une représentation mathématique simplifiée du mécanisme ayant produit les données.

Il sépare généralement :

1. une partie systématique, que l’on cherche à expliquer ;
2. une partie aléatoire ou non expliquée.

Pour la confiance :

\[
\text{confiance observée}
=
\text{partie expliquée}
+
\text{partie non expliquée}
\]

La partie expliquée pourrait dépendre de :

- la condition ;
- la séquence ;
- l’entropie ;
- la précision ;
- le nombre de modèles mentaux.

La partie non expliquée contient :

- des facteurs non mesurés ;
- des fluctuations momentanées ;
- des erreurs de mesure ;
- des variations propres à l’essai ;
- des mécanismes absents du modèle.

---

## 6.2 Variable dépendante et variables explicatives

La **variable dépendante** est celle que l’on cherche à expliquer :

```text
confidence
```

Les **variables explicatives**, aussi appelées **prédicteurs**, sont les variables utilisées pour expliquer ses variations :

```text
condition
sequence
subject_accuracy
item_entropy
subject_mean_models
models_within_subject
validity_binary
```

Le mot « prédicteur » ne suppose pas obligatoirement une causalité.

Si l’entropie est associée à la confiance, cela ne prouve pas automatiquement que l’entropie cause la confiance.

Cela signifie que les deux variables covarient selon le modèle.

---

## 6.3 Pourquoi ne pas se contenter de moyennes ?

Nous aurions pu calculer :

```text
confiance moyenne Standard
confiance moyenne Neutral
```

Puis soustraire les deux.

Cela répondrait partiellement à la question de condition, mais pas aux autres questions.

Une simple moyenne ne permet pas facilement de tenir compte simultanément :

- de la séquence ;
- de l’entropie ;
- de la précision ;
- des modèles mentaux ;
- des différences entre participants ;
- des différences entre items.

La modélisation permet d’étudier un prédicteur tout en tenant compte des autres.

C’est ce que l’on appelle un **effet ajusté** ou **conditionnel**.

---

## 6.4 Exemple d’effet brut et d’effet ajusté

Supposons que la condition Standard possède une confiance plus élevée.

Une différence brute pourrait être :

\[
78-73=5
\]

Mais imaginons que les essais Standard soient aussi, par hasard, associés à des items moins entropiques.

Une partie de la différence pourrait provenir des items et non de la condition elle-même.

Un modèle comprenant simultanément la condition et l’entropie cherche à estimer :

> La différence Standard–Neutral pour des observations comparables sur les autres variables incluses.

Le modèle ne résout pas tous les problèmes causaux, mais il permet de séparer statistiquement plusieurs sources de variation.

---

# 7. Pourquoi envisager une régression ?

## 7.1 Définition simple

Une **régression** est une méthode permettant de décrire la relation entre une variable à expliquer et une ou plusieurs variables explicatives.

La forme la plus simple est :

\[
Y=\beta_0+\beta_1X+\varepsilon
\]

où :

- \(Y\) est la variable dépendante ;
- \(X\) est un prédicteur ;
- \(\beta_0\) est l’intercept ;
- \(\beta_1\) est le coefficient de \(X\) ;
- \(\varepsilon\) est l’erreur ou résidu.

---

## 7.2 L’intercept

L’**intercept** est la valeur prédite de \(Y\) lorsque \(X=0\).

Exemple :

\[
\text{confiance}
=
70-2\times\text{entropie standardisée}
\]

L’intercept vaut 70.

Lorsque l’entropie standardisée vaut 0, la confiance prédite est 70.

Si les prédicteurs ont été centrés ou standardisés, 0 correspond généralement à une valeur moyenne, ce qui rend l’intercept plus facile à comprendre.

---

## 7.3 Le coefficient

Dans :

\[
Y=\beta_0+\beta_1X
\]

\(\beta_1\) indique la variation attendue de \(Y\) lorsqu’on augmente \(X\) d’une unité.

Exemple :

\[
\text{confiance}
=
75-2{,}5\times\text{entropie standardisée}
\]

Le coefficient de l’entropie est \(-2{,}5\).

Cela signifie qu’une augmentation d’un écart-type de l’entropie est associée à une diminution moyenne de 2,5 points de confiance.

---

## 7.4 Le résidu

Le **résidu** est la différence entre la valeur observée et la valeur prédite :

\[
e_i=y_i-\hat{y}_i
\]

où :

- \(y_i\) est la valeur observée ;
- \(\hat{y}_i\) est la valeur prédite.

### Exemple

```text
Confiance observée = 80
Confiance prédite = 74
```

Alors :

\[
e=80-74=6
\]

Le modèle a sous-estimé la confiance de 6 points.

Si :

```text
Confiance observée = 60
Confiance prédite = 74
```

alors :

\[
e=60-74=-14
\]

Le modèle a surestimé la confiance de 14 points.

> **Analogie**
>
> Le modèle est comme un météorologue.
>
> - température prévue : 20 °C ;
> - température observée : 23 °C ;
> - erreur de prévision : +3 °C.
>
> Le résidu joue le rôle de cette erreur de prévision.

---

# 8. La régression linéaire ordinaire

## 8.1 Forme avec plusieurs prédicteurs

Une régression linéaire multiple pourrait prendre la forme :

\[
\begin{aligned}
\text{confidence}_{ij}
={}&
\beta_0
+\beta_1\text{condition}_{ij}
+\beta_2\text{sequence}_{ij}\\
&+\beta_3\text{accuracy}_{ij}
+\beta_4\text{entropy}_{ij}\\
&+\beta_5\text{models}_{ij}
+\varepsilon_{ij}
\end{aligned}
\]

L’indice \(i\) peut représenter le participant et \(j\) l’essai ou l’item.

Le terme **linéaire** signifie que les prédicteurs sont combinés par des additions de coefficients multipliés par des variables.

Cela ne signifie pas que toutes les variables du monde sont réellement reliées par une ligne droite. Il s’agit d’une approximation.

---

## 8.2 Comment la régression choisit-elle les coefficients ?

Dans une régression linéaire classique, les coefficients sont généralement choisis afin de minimiser la somme des carrés des résidus :

\[
\sum_i e_i^2
=
\sum_i(y_i-\hat{y}_i)^2
\]

Pourquoi élever les résidus au carré ?

1. Les erreurs positives et négatives ne s’annulent pas.
2. Les grandes erreurs sont davantage pénalisées.
3. L’expression possède de bonnes propriétés mathématiques.

### Exemple

Trois résidus :

```text
+2, -2, +5
```

La somme simple vaut :

\[
2-2+5=5
\]

La somme des carrés vaut :

\[
2^2+(-2)^2+5^2
=
4+4+25
=
33
\]

L’erreur de 5 pèse fortement dans le résultat.

---

## 8.3 Avantages d’une régression ordinaire

Une régression linéaire ordinaire est :

- relativement simple ;
- rapide à calculer ;
- facile à expliquer ;
- adaptée à une variable dépendante numérique ;
- capable d’inclure plusieurs prédicteurs ;
- capable de produire des coefficients, erreurs-types et intervalles de confiance.

---

## 8.4 Limites dans notre projet

Le principal problème est l’hypothèse d’indépendance.

Une régression ordinaire suppose en général que, une fois les prédicteurs pris en compte, les erreurs des observations sont indépendantes.

Dans notre fichier, cette hypothèse est peu crédible :

```text
64 observations du même participant
environ 70 observations du même item
```

Une régression ordinaire traiterait les 9 024 lignes comme si elles provenaient de 9 024 personnes et items complètement séparés.

Ce serait faux.

---

# 9. Pourquoi une régression ordinaire ne suffisait pas

## 9.1 Le problème de la pseudo-réplication

La **pseudo-réplication** apparaît lorsque l’on traite des observations répétées comme si elles étaient des unités indépendantes.

Exemple extrême :

Supposons que nous mesurions 100 fois la taille d’une seule personne.

Nous n’avons pas vraiment un échantillon de 100 personnes. Nous avons 100 mesures de la même personne.

Traiter ces 100 mesures comme 100 individus indépendants donnerait une impression trompeuse de quantité d’information.

Dans notre expérience :

- 9 024 lignes ne signifient pas 9 024 participants indépendants ;
- elles proviennent de 141 participants seulement.

---

## 9.2 Conséquence sur les erreurs-types

L’**erreur-type** mesure l’incertitude autour d’une estimation.

Une petite erreur-type indique une estimation précise.

Une grande erreur-type indique une estimation plus incertaine.

Si l’on ignore les regroupements, on risque de sous-estimer les erreurs-types.

Le modèle peut alors annoncer qu’un effet est très précis simplement parce qu’il croit disposer de davantage d’observations indépendantes qu’en réalité.

Cela peut produire des **faux positifs**.

Un faux positif est la conclusion erronée qu’un effet existe alors que les données ne fournissent pas réellement une preuve suffisante.

---

## 9.3 Exemple conceptuel

Imaginons deux participants :

```text
Participant A : confiance toujours proche de 90
Participant B : confiance toujours proche de 50
```

Si chaque participant donne 64 réponses, une régression ordinaire voit 128 nombres.

Mais une grande partie de la variation provient simplement du fait que A est globalement plus confiant que B.

Il faut permettre au modèle de représenter cette différence stable.

---

## 9.4 Le même problème existe pour les items

Supposons deux items :

```text
Item X : presque tout le monde donne une confiance élevée
Item Y : presque tout le monde donne une confiance faible
```

Si nous ignorons l’identité des items, cette différence peut être attribuée à tort à un prédicteur corrélé à l’item.

Le modèle doit donc reconnaître que certains items ont un niveau général de confiance différent.

---

# 10. Le modèle linéaire mixte

## 10.1 Définition simple

Un **modèle linéaire mixte** est une extension de la régression linéaire qui combine :

1. des **effets fixes**, correspondant aux relations que nous voulons estimer explicitement ;
2. des **effets aléatoires**, correspondant aux regroupements et aux variations propres aux participants ou aux items.

Le mot « mixte » vient du mélange de ces deux types d’effets.

---

## 10.2 Forme générale

Notre modèle peut être représenté comme :

\[
\begin{aligned}
\text{confidence}_{ij}
={}&
\beta_0
+\beta_1X_{1ij}
+\beta_2X_{2ij}
+\cdots\\
&+u_i
+v_j
+\varepsilon_{ij}
\end{aligned}
\]

où :

- \(\beta_0\) est l’intercept général ;
- \(\beta_1,\beta_2,\ldots\) sont les effets fixes ;
- \(u_i\) est l’effet aléatoire du participant \(i\) ;
- \(v_j\) est l’effet aléatoire de l’item \(j\) ;
- \(\varepsilon_{ij}\) est le résidu de l’observation.

---

## 10.3 Décomposition intuitive

Le modèle dit que la confiance d’un essai dépend de plusieurs couches :

```text
Confiance observée
=
moyenne générale
+ caractéristiques mesurées
+ tendance propre au participant
+ tendance propre à l’item
+ variation restante
```

### Exemple numérique

Supposons :

```text
Moyenne générale                   = 75
Effet de la condition Standard     = +5
Effet de l’entropie                = -3
Effet propre au participant        = +8
Effet propre à l’item              = -2
Résidu particulier de l’essai      = +1
```

La confiance observée serait approximativement :

\[
75+5-3+8-2+1=84
\]

---

## 10.4 Pourquoi cette structure convient-elle ?

Elle représente explicitement les deux sources de répétition :

```text
plusieurs essais par participant
plusieurs participants par item
```

Elle permet donc :

- d’éviter de traiter les lignes comme indépendantes ;
- d’estimer la variabilité entre participants ;
- d’estimer la variabilité entre items ;
- d’estimer les effets des prédicteurs en tenant compte de ces regroupements.

---

# 11. Effets fixes et effets aléatoires

## 11.1 Effet fixe

Un **effet fixe** est un coefficient associé à une variable dont nous voulons estimer la relation moyenne avec la variable dépendante.

Exemples :

```text
condition
sequence
item_entropy
subject_accuracy
number_models_generated
```

Si le coefficient de l’entropie vaut \(-2{,}5\), nous interprétons une relation moyenne dans la population étudiée :

> Une augmentation d’un écart-type de l’entropie est associée à une diminution moyenne de 2,5 points de confiance.

---

## 11.2 Pourquoi le mot « fixe » ?

Le terme ne signifie pas que la variable ne change jamais.

`sequence` change à chaque essai, mais son coefficient est appelé effet fixe.

« Fixe » signifie ici que nous estimons un coefficient global explicite :

\[
\beta_{\text{sequence}}
\]

Nous voulons le rapporter et l’interpréter directement.

---

## 11.3 Effet aléatoire

Un **effet aléatoire** représente la variation de groupes appartenant à un ensemble plus large.

Dans notre cas :

```text
participants
items
```

Nous ne cherchons pas principalement à publier une liste de 141 coefficients individuels.

Nous cherchons plutôt à estimer :

- à quel point les participants diffèrent entre eux ;
- à quel point les items diffèrent entre eux.

Le modèle suppose que les effets participants suivent approximativement une distribution normale :

\[
u_i\sim\mathcal{N}(0,\sigma^2_{\text{participant}})
\]

et que les effets items suivent :

\[
v_j\sim\mathcal{N}(0,\sigma^2_{\text{item}})
\]

Cela signifie :

- leur moyenne est définie comme 0 autour de l’intercept général ;
- certains sont positifs ;
- certains sont négatifs ;
- leur dispersion est mesurée par une variance.

---

## 11.4 Exemple d’effets participants

Supposons une moyenne générale de 75.

Quelques effets aléatoires participants pourraient être :

| Participant | Effet | Niveau général prédit |
|---|---:|---:|
| A | +12 | 87 |
| B | −8 | 67 |
| C | +2 | 77 |
| D | −15 | 60 |

Le participant A utilise globalement l’échelle plus haut que la moyenne.

Le participant D l’utilise globalement plus bas.

---

## 11.5 Exemple d’effets items

| Item | Effet | Conséquence |
|---|---:|---|
| I1 | +4 | inspire davantage de confiance |
| I2 | −3 | inspire moins de confiance |
| I3 | 0 | proche de la moyenne |

Ces effets représentent les différences d’items qui restent après prise en compte des prédicteurs mesurés.

---

## 11.6 Intercept aléatoire

Dans notre projet, nous avons commencé avec des **intercepts aléatoires**.

Un intercept aléatoire autorise chaque participant et chaque item à avoir son propre niveau de base.

La pente des prédicteurs reste initialement commune.

Graphiquement :

```text
Participant A : -------- pente commune, niveau élevé
Participant B : -------- pente commune, niveau moyen
Participant C : -------- pente commune, niveau faible
```

Les lignes sont décalées verticalement, mais elles ont la même pente.

---

## 11.7 Pente aléatoire

Une **pente aléatoire** permettrait également à l’effet d’un prédicteur de varier entre participants.

Par exemple, l’effet de l’entropie pourrait être :

```text
Participant A : très négatif
Participant B : légèrement négatif
Participant C : presque nul
```

Mathématiquement :

\[
\text{confidence}_{ij}
=
\beta_0
+\beta_1\text{entropy}_{ij}
+u_{0i}
+u_{1i}\text{entropy}_{ij}
+\varepsilon_{ij}
\]

où \(u_{1i}\) est une variation individuelle de la pente.

Nous n’avons pas commencé par cette structure, car elle est plus complexe et plus difficile à estimer.

Avec plusieurs prédicteurs, quatre types de tâches et 141 participants, une structure aléatoire très riche pourrait produire :

- des problèmes de convergence ;
- des variances estimées à zéro ;
- une instabilité numérique ;
- des résultats difficiles à interpréter.

Nous avons donc commencé avec une structure plus robuste :

```text
intercept aléatoire participant
intercept aléatoire item
```

---

# 12. Pourquoi utiliser des effets croisés participant–item ?

## 12.1 La logique des deux regroupements

Chaque observation appartient simultanément :

- à un participant ;
- à un item.

On peut écrire :

\[
Y_{ij}
\]

où :

- \(i\) identifie le participant ;
- \(j\) identifie l’item.

Le modèle doit reconnaître les deux identités.

---

## 12.2 Pourquoi ne mettre qu’un effet participant serait insuffisant ?

Un modèle avec seulement :

```text
(1 | participant)
```

corrigerait la répétition des mesures par participant.

Mais il continuerait à traiter les observations d’un même item comme indépendantes.

Il pourrait alors surestimer la quantité d’information associée aux propriétés des items.

C’est particulièrement problématique pour `item_entropy`, car cette variable est constante pour toutes les observations d’un même item.

---

## 12.3 Pourquoi ne mettre qu’un effet item serait insuffisant ?

Un modèle avec seulement :

```text
(1 | item)
```

corrigerait les répétitions par item.

Mais il ignorerait que chaque participant fournit 64 réponses.

Il serait incapable de représenter les styles individuels d’utilisation de la confiance.

---

## 12.4 Pourquoi ne pas agréger toutes les données ?

Nous aurions pu réduire les données.

### Agrégation par participant

Une ligne par participant :

```text
confiance moyenne
précision moyenne
modèles moyens
condition
```

Nous aurions alors 141 lignes.

Mais nous aurions perdu :

- la variation entre items ;
- la séquence ;
- l’entropie par item ;
- la variation intra-individuelle du nombre de modèles ;
- la possibilité d’analyser les essais.

### Agrégation par item

Une ligne par item aurait conservé l’entropie, mais supprimé les différences individuelles.

### Avantage du modèle mixte

Le modèle mixte conserve les 9 024 essais tout en tenant compte de leur structure.

Il utilise donc l’information fine sans prétendre que toutes les lignes sont indépendantes.

---

# 13. Décomposer les variables interindividuelles et intra-individuelles

Cette partie est particulièrement importante pour `number_models_generated`.

---

## 13.1 Deux questions différentes cachées dans une même variable

Supposons que le nombre de modèles mentaux soit plus élevé dans certaines observations.

Cela peut signifier deux choses très différentes.

### Question interindividuelle

Les participants qui génèrent généralement davantage de modèles sont-ils moins confiants que les autres ?

Comparaison :

```text
Participant A : moyenne de 2 modèles
Participant B : moyenne de 4 modèles
```

### Question intra-individuelle

Pour un même participant, les types de tâches qui produisent plus de modèles que sa moyenne personnelle sont-ils associés à une confiance plus faible ?

Comparaison :

```text
Participant A :
MP = 2 modèles
MT = 2 modèles
AC = 3 modèles
DA = 4 modèles
```

Ces deux questions ne sont pas équivalentes.

---

## 13.2 Pourquoi une seule variable peut les mélanger

Si nous utilisons directement :

```text
number_models_generated
```

le coefficient mélange potentiellement :

- les différences entre participants ;
- les différences entre types de tâches à l’intérieur des participants.

Cela peut produire une interprétation ambiguë.

---

## 13.3 Moyenne personnelle

Pour chaque participant \(i\), nous calculons :

\[
\bar{M}_i
=
\frac{1}{K_i}
\sum_k M_{ik}
\]

où \(M_{ik}\) est le nombre de modèles du participant \(i\) pour le type de tâche \(k\).

Cette variable est :

```text
subject_mean_models
```

Elle représente la composante interindividuelle.

---

## 13.4 Écart à la moyenne personnelle

Nous calculons ensuite :

\[
M_{ik}^{\text{within}}
=
M_{ik}-\bar{M}_i
\]

Cette variable devient :

```text
models_within_subject
```

Elle indique si un type de tâche produit plus ou moins de modèles que la moyenne propre au participant.

---

## 13.5 Exemple complet

Supposons :

| Tâche | Nombre de modèles |
|---|---:|
| MP | 2 |
| MT | 2 |
| AC | 3 |
| DA | 5 |

La moyenne personnelle est :

\[
\bar{M}
=
\frac{2+2+3+5}{4}
=
3
\]

Les composantes intra-individuelles sont :

| Tâche | Modèles | Moyenne | Écart personnel |
|---|---:|---:|---:|
| MP | 2 | 3 | −1 |
| MT | 2 | 3 | −1 |
| AC | 3 | 3 | 0 |
| DA | 5 | 3 | +2 |

La somme des écarts vaut :

\[
-1-1+0+2=0
\]

C’est normal : les écarts à une moyenne s’équilibrent autour de zéro.

---

## 13.6 Interprétation des deux coefficients

Dans le modèle :

\[
\text{confidence}
=
\beta_B\text{subject\_mean\_models}
+
\beta_W\text{models\_within\_subject}
+\cdots
\]

### \(\beta_B\) : effet entre participants

Il répond à :

> Les participants ayant une moyenne de modèles plus élevée ont-ils une confiance différente ?

### \(\beta_W\) : effet à l’intérieur des participants

Il répond à :

> Pour un participant donné, les types de tâches qui génèrent plus de modèles que son niveau habituel produisent-ils une confiance différente ?

Cette décomposition est souvent appelée **centrage par la moyenne du groupe** ou décomposition **between–within**.

---

## 13.7 Analogie scolaire

Imaginons des élèves appartenant à plusieurs classes.

On observe que les élèves de classes ayant beaucoup d’heures de cours réussissent mieux.

Deux mécanismes sont possibles :

1. les classes qui ont généralement plus d’heures réussissent mieux ;
2. pour une même classe, les semaines avec plus d’heures produisent de meilleurs résultats.

Comparer les classes et comparer les semaines dans une classe sont deux questions différentes.

De même :

- `subject_mean_models` compare les participants ;
- `models_within_subject` compare les types de tâches à l’intérieur d’un participant.

---

# 14. Les objectifs scientifiques formulés au départ

L’analyse préliminaire a conduit à plusieurs objectifs organisés.

---

## 14.1 Objectif 1 — Décrire la variabilité de confiance

Avant d’expliquer la confiance, nous devions savoir d’où venait sa variation.

Questions :

- Quelle proportion de la variance est liée aux participants ?
- Quelle proportion est liée aux items ?
- Quelle proportion reste au niveau des essais ?

Cette question justifiait le futur **modèle nul**.

---

## 14.2 Objectif 2 — Contrôler le plan expérimental

Nous voulions estimer les effets de :

- la condition Standard versus Neutral ;
- la position de l’essai.

Ces variables ne constituaient pas nécessairement nos hypothèses cognitives principales, mais elles pouvaient influencer la confiance.

Il fallait donc les inclure avant les prédicteurs cognitifs.

---

## 14.3 Objectif 3 — Tester l’incertitude liée aux items

Nous voulions tester l’hypothèse :

\[
\text{entropie plus élevée}
\Rightarrow
\text{confiance plus faible}
\]

L’idée est qu’un item suscitant des réponses partagées pourrait aussi produire davantage d’incertitude subjective.

---

## 14.4 Objectif 4 — Tester les différences de précision

Nous voulions savoir si les participants plus précis sont aussi plus confiants.

Deux possibilités étaient envisageables :

### Hypothèse positive

Les participants performants reconnaissent leur compétence et sont plus confiants.

### Hypothèse nulle ou négative

La confiance peut refléter un style de réponse indépendant de la performance.

Un participant peut être très confiant sans être très précis.

---

## 14.5 Objectif 5 — Tester le rôle des modèles mentaux

L’hypothèse théorique générale était que la représentation de plusieurs possibilités pourrait réduire la certitude.

Intuition :

```text
une seule représentation plausible
→ décision plus directe
→ confiance potentiellement élevée

plusieurs représentations possibles
→ davantage d’alternatives
→ décision moins évidente
→ confiance potentiellement faible
```

Mais nous devions distinguer :

- le nombre moyen de modèles entre participants ;
- la variation des modèles à l’intérieur d’un participant.

---

## 14.6 Objectif 6 — Étudier validité et type de tâche

Nous voulions savoir si la confiance variait entre :

- inférences valides et invalides ;
- MP, MT, AC et DA.

Mais comme la validité était déterminée par le type de tâche, ces analyses devaient être séparées ou traitées comme des analyses de sensibilité.

---

## 14.7 Objectif 7 — Vérifier la calibration

Finalement, nous voulions savoir si la confiance correspondait réellement à l’exactitude.

Deux aspects devaient être distingués :

### Calibration moyenne

La confiance moyenne correspond-elle à la proportion correcte ?

### Discrimination essai par essai

La confiance est-elle plus élevée pour les réponses correctes que pour les réponses incorrectes ?

Ces analyses ont été réalisées plus tard, après le modèle principal.

---

# 15. Les modèles envisagés et l’ordre prévu

Nous avons choisi une démarche progressive.

---

## 15.1 Pourquoi construire plusieurs modèles ?

Un seul grand modèle aurait été difficile à comprendre.

Nous avons préféré construire des couches :

```text
Modèle nul
    ↓
Modèle de contrôle
    ↓
Modèle cognitif
    ↓
Analyses de sensibilité
    ↓
Diagnostics et calibration
```

Cette progression permet d’observer ce que chaque groupe de variables apporte.

---

## 15.2 Modèle nul

Forme conceptuelle :

\[
\text{confidence}
=
\beta_0
+
u_{\text{participant}}
+
v_{\text{item}}
+
\varepsilon
\]

Aucun prédicteur explicatif n’est inclus.

Objectifs :

- estimer la moyenne générale ;
- décomposer la variance ;
- établir une référence.

---

## 15.3 Modèle de contrôle

Forme conceptuelle :

\[
\text{confidence}
=
\beta_0
+
\beta_1\text{condition}
+
\beta_2\text{sequence}
+
u_{\text{participant}}
+
v_{\text{item}}
+
\varepsilon
\]

Objectifs :

- contrôler la condition ;
- contrôler l’évolution temporelle ;
- vérifier si ces facteurs améliorent le modèle nul.

---

## 15.4 Modèle cognitif

Forme conceptuelle :

\[
\begin{aligned}
\text{confidence}
={}&
\beta_0
+\beta_1\text{condition}
+\beta_2\text{sequence}\\
&+\beta_3\text{subject accuracy}
+\beta_4\text{item entropy}\\
&+\beta_5\text{mean models}
+\beta_6\text{within models}\\
&+\beta_7\text{validity}
+u_{\text{participant}}
+v_{\text{item}}
+\varepsilon
\end{aligned}
\]

Objectifs :

- tester l’apport global des prédicteurs cognitifs ;
- estimer leur contribution conditionnelle ;
- distinguer effets interindividuels et intra-individuels.

---

## 15.5 Analyses de sensibilité

Une **analyse de sensibilité** vérifie si une conclusion reste similaire lorsqu’on modifie raisonnablement la méthode.

> **Analogie**
>
> Si une table tient debout uniquement lorsque le sol est parfaitement plat, elle est fragile.
>
> Si elle reste stable lorsqu’on la déplace légèrement, elle est robuste.
>
> Une analyse de sensibilité teste cette robustesse pour les résultats statistiques.

Nous avions prévu notamment :

- remplacer la validité par le type de tâche ;
- retirer les prédicteurs un par un ;
- examiner l’effet plafond ;
- augmenter le nombre de simulations MReasoner ;
- retirer successivement les participants.

---

# 16. Les solutions alternatives

## 16.1 Comparer uniquement des moyennes

Nous aurions pu utiliser des tests t ou des analyses de variance.

### Test t

Un **test t** compare généralement les moyennes de deux groupes.

Exemple :

```text
confiance Standard
contre
confiance Neutral
```

### Pourquoi ne pas nous limiter à cela ?

Parce que nous avions :

- plusieurs prédicteurs ;
- des observations répétées ;
- des items répétés ;
- des variables continues ;
- des questions interindividuelles et intra-individuelles.

Un simple test t ne peut pas représenter toute cette structure.

---

## 16.2 ANOVA classique

Une **ANOVA**, ou analyse de variance, compare des moyennes entre plusieurs conditions ou catégories.

Elle aurait pu être utilisée pour comparer MP, MT, AC et DA.

Mais une ANOVA classique est moins flexible pour :

- l’entropie continue ;
- la séquence continue ;
- les données déséquilibrées ;
- les effets participant et item simultanés ;
- la décomposition des modèles mentaux.

---

## 16.3 Agréger par participant

Avantage :

- simplicité ;
- observations approximativement indépendantes.

Limite :

- perte de l’information par item ;
- perte de la séquence ;
- perte de la variation intra-individuelle ;
- impossibilité d’estimer un effet item.

---

## 16.4 Agréger par item

Avantage :

- analyse simple des propriétés des items.

Limite :

- disparition des différences individuelles ;
- perte de la condition entre participants ;
- incapacité à tester les modèles mentaux individuels.

---

## 16.5 Régression avec effets fixes pour tous les participants et items

Nous aurions pu créer une catégorie pour chacun des 141 participants et chacun des 128 items.

Forme conceptuelle :

```text
confidence ~ prédicteurs + C(subject_id) + C(item_id)
```

Cela aurait créé plusieurs centaines de coefficients.

### Avantage

Cela contrôle les différences de niveau propres à chaque participant et item.

### Limites

- très grand nombre de paramètres ;
- difficulté à estimer les effets de variables constantes par participant ;
- impossibilité de quantifier directement une variance participant généralisable ;
- modèle plus lourd et moins élégant ;
- interprétation moins adaptée à notre objectif.

Le modèle mixte représente plutôt les participants et les items comme des échantillons provenant de populations plus larges.

---

## 16.6 Modèle bayésien complet

Un **modèle bayésien** combine :

- une information a priori ;
- les données observées ;
- une distribution postérieure des paramètres.

Il aurait été possible d’ajuster tous nos modèles dans un environnement comme Stan, PyMC ou `brms`.

### Avantages

- grande flexibilité ;
- excellente gestion des structures hiérarchiques ;
- intervalles probabilistes directs ;
- modèles bornés ou non gaussiens plus naturels.

### Pourquoi ne pas commencer ainsi ?

- apprentissage plus complexe ;
- calcul plus lent ;
- nécessité de choisir et justifier des distributions a priori ;
- diagnostics supplémentaires ;
- objectif initial de construire un pipeline compréhensible en Python avec `statsmodels`.

Nous avons toutefois utilisé plus tard un modèle bayésien variationnel pour la réponse binaire `confidence == 100`, car `statsmodels` ne propose pas un GLMM logistique fréquentiste croisé aussi direct que son modèle linéaire mixte.

---

## 16.7 Modèle ordinal

La confiance est enregistrée entre 0 et 100, mais peut être considérée comme une échelle ordonnée.

Un **modèle ordinal** traite les valeurs comme des catégories ordonnées plutôt que comme des distances parfaitement continues.

Par exemple, il reconnaît que :

```text
80 > 70 > 60
```

sans supposer nécessairement que la différence 80–70 a exactement le même sens psychologique que 30–20.

### Pourquoi ne pas l’utiliser comme modèle principal ?

- 101 niveaux possibles ;
- complexité accrue ;
- modèle mixte ordinal croisé moins accessible dans notre environnement Python ;
- interprétation plus difficile ;
- la régression linéaire reste souvent utilisée pour des échelles numériques comportant de nombreux niveaux.

L’effet plafond a cependant été pris au sérieux dans des analyses spécifiques.

---

## 16.8 Modèle bêta

Une **régression bêta** est utilisée pour des proportions strictement comprises entre 0 et 1.

Nous aurions pu transformer :

\[
\text{confidence}/100
\]

Mais une distribution bêta ne permet pas directement les valeurs exactes 0 et 1.

Or nos données contenaient :

- des confiances égales à 0 ;
- beaucoup de confiances égales à 100.

Il aurait fallu utiliser un modèle bêta gonflé en zéro et en un, beaucoup plus complexe.

---

## 16.9 Modèle censuré ou Tobit

Un modèle censuré peut être utile lorsqu’une variable est limitée par une borne et que certaines observations s’accumulent à cette borne.

Mais une confiance de 100 n’est pas nécessairement une valeur latente supérieure à 100 qui aurait été artificiellement coupée.

Le participant peut réellement choisir la catégorie maximale 100.

Il était donc plus prudent de traiter le plafond séparément :

1. modèle linéaire principal ;
2. modèle linéaire sous le plafond ;
3. modèle logistique de la probabilité d’utiliser exactement 100.

---

# 17. Les hypothèses du modèle linéaire mixte

Une **hypothèse statistique** est une condition sous laquelle la méthode possède les propriétés attendues.

Une hypothèse n’est pas toujours parfaitement vraie. Il faut examiner si les écarts sont acceptables et s’ils modifient les conclusions.

---

## 17.1 Relation approximativement linéaire

Le modèle suppose que l’effet moyen d’un prédicteur continu peut être raisonnablement représenté par une pente constante.

Exemple :

\[
+1\text{ écart-type d’entropie}
\Rightarrow
-2{,}5\text{ points de confiance}
\]

Cela suppose que la relation n’a pas une forme fortement courbe.

Si la vraie relation est en U, une seule pente linéaire serait insuffisante.

---

## 17.2 Résidus centrés autour de zéro

Les erreurs devraient être en moyenne proches de zéro.

Sinon, le modèle surestime ou sous-estime systématiquement la confiance.

---

## 17.3 Variance résiduelle relativement stable

L’**homoscédasticité** signifie que la dispersion des résidus est relativement stable le long des valeurs prédites.

L’**hétéroscédasticité** signifie que cette dispersion change.

Exemple :

```text
faible confiance prédite → erreurs petites
forte confiance prédite  → erreurs très grandes
```

Cela pourrait affecter les erreurs-types et les intervalles de confiance.

---

## 17.4 Normalité approximative des résidus

Le modèle linéaire suppose généralement que les résidus conditionnels suivent approximativement une distribution normale :

\[
\varepsilon\sim\mathcal{N}(0,\sigma^2)
\]

Une distribution normale possède une forme en cloche, symétrique autour de sa moyenne.

Cette hypothèse concerne les erreurs conditionnelles, pas nécessairement la distribution brute de `confidence`.

### Pourquoi cette hypothèse existe-t-elle ?

Elle permet de calculer la vraisemblance, les erreurs-types et les intervalles de confiance du modèle.

### Difficulté anticipée

La confiance était bornée entre 0 et 100.

Une variable bornée ne peut pas suivre parfaitement une distribution normale.

De plus, beaucoup de valeurs étaient exactement égales à 100.

Nous savions donc dès le départ que des diagnostics et analyses de sensibilité seraient nécessaires.

---

## 17.5 Normalité des effets aléatoires

Le modèle suppose que les effets participants et items suivent approximativement des distributions normales autour de zéro.

Cela ne signifie pas que la confiance brute de chaque participant doit être normale.

Cela concerne la distribution des décalages de niveau estimés entre les groupes.

---

## 17.6 Indépendance conditionnelle

Les observations ne sont pas supposées indépendantes de manière brute.

Le modèle mixte reconnaît justement leurs dépendances via les effets participant et item.

L’hypothèse devient :

> Après prise en compte des effets fixes et des effets aléatoires, les résidus restants sont suffisamment indépendants.

C’est une hypothèse plus réaliste que celle de la régression ordinaire.

---

## 17.7 Absence de colinéarité parfaite

Le modèle ne doit pas recevoir deux prédicteurs contenant exactement la même information.

C’est pourquoi validité et type de tâche ne devaient pas être introduits ensemble sans précaution.

---

# 18. Les difficultés anticipées

## 18.1 L’effet plafond

Un **effet plafond** apparaît lorsqu’une grande quantité de réponses se trouve à la valeur maximale de l’échelle.

Exemple :

```text
100, 100, 100, 100, 95, 100, 87...
```

Conséquences possibles :

- asymétrie de la distribution ;
- résidus non normaux ;
- impossibilité d’observer des augmentations au-delà de 100 ;
- coefficients linéaires influencés par la propension à utiliser la borne.

Nous avons donc prévu des analyses spécifiques.

---

## 18.2 Le nombre limité de simulations MReasoner

Avec trois simulations seulement, une moyenne peut être instable.

Exemple :

```text
Simulations : 2, 2, 7
Moyenne : 3,67
```

Une seule valeur élevée influence fortement la moyenne.

Il fallait donc :

- mesurer l’écart-type des simulations ;
- examiner les minima et maxima ;
- comparer plus tard 3, 10 et 20 simulations.

---

## 18.3 L’entropie construite à partir des mêmes réponses

L’entropie est calculée à partir des réponses des participants.

Elle est ensuite utilisée pour expliquer leur confiance.

Cette dépendance ne rend pas automatiquement l’analyse invalide, mais impose une interprétation prudente.

L’entropie doit être décrite comme une caractéristique empirique de l’item dans cet échantillon.

---

## 18.4 La précision comme prédicteur participant

`subject_accuracy` est calculée à partir des mêmes 64 essais que ceux utilisés dans le modèle.

Cela signifie qu’elle n’est pas une mesure externe et indépendante de compétence.

Elle résume la performance du participant dans cette expérience.

Il faut donc éviter une interprétation causale forte du type :

> La compétence mesurée indépendamment cause la confiance.

La formulation correcte est :

> La précision moyenne observée du participant est-elle associée à son niveau de confiance ?

---

## 18.5 La condition entre participants

Comme chaque participant appartient à une seule condition, l’effet de condition est estimé à partir d’une comparaison entre groupes de participants.

Il est donc moins précisément identifié qu’un facteur variant à l’intérieur de chaque personne.

Une partie des différences de condition peut coïncider avec des différences individuelles aléatoires, même si l’affectation expérimentale vise à équilibrer les groupes.

---

## 18.6 La validité confondue avec le type de tâche

Le terme **confondu** signifie ici que deux variables ne peuvent pas être entièrement séparées dans le plan observé.

Nous n’avions pas :

```text
MP invalide
MT invalide
AC valide
DA valide
```

Nous avions uniquement :

```text
MP et MT → valides
AC et DA → invalides
```

Nous ne pouvions donc pas estimer une validité complètement indépendante de la forme logique.

---

# 19. Ce que l’analyse préliminaire nous a appris

À la fin de cette première étape, nous avions établi plusieurs conclusions méthodologiques.

---

## 19.1 L’unité d’analyse devait rester l’essai

Nous voulions conserver :

- la séquence ;
- l’identité de l’item ;
- la condition ;
- la réponse ;
- la confiance ;
- l’exactitude ;
- le type de tâche ;
- les prédicteurs cognitifs.

Nous avons donc gardé une ligne par essai au lieu d’agréger immédiatement les données.

---

## 19.2 Une régression était pertinente

La confiance est une variable numérique.

Nous voulions estimer simultanément les relations avec plusieurs variables.

La famille de méthodes de régression était donc appropriée.

---

## 19.3 Une régression ordinaire n’était pas suffisante

Les 9 024 observations étaient regroupées :

```text
64 observations par participant
environ 70 observations par item
```

Ignorer ces regroupements aurait produit une structure statistique incorrecte.

---

## 19.4 Un modèle linéaire mixte croisé était le meilleur point de départ

Le modèle devait inclure :

```text
intercept aléatoire participant
intercept aléatoire item
```

Cela permettait de distinguer :

- la variance entre participants ;
- la variance entre items ;
- la variance résiduelle.

---

## 19.5 Les prédicteurs devaient être organisés par niveaux

### Niveau participant

```text
condition
subject_accuracy
subject_mean_models
```

### Niveau item

```text
item_entropy
item_accuracy
```

### Niveau essai

```text
sequence
confidence
is_correct
response
```

### Niveau participant × type de tâche

```text
number_models_generated
models_within_subject
```

Cette organisation est essentielle pour interpréter correctement les coefficients.

---

## 19.6 Le nombre de modèles devait être décomposé

Nous ne pouvions pas interpréter correctement `number_models_generated` sans séparer :

```text
différences entre participants
et
variations à l’intérieur des participants
```

Cela a motivé :

```text
subject_mean_models
models_within_subject
```

---

## 19.7 La construction d’un dataset analytique était indispensable

Les informations nécessaires étaient dispersées entre :

```text
dataset_ccobra_E1.csv
mental_models_count_E1.csv
```

Certaines variables devaient être calculées :

```text
subject_accuracy
item_entropy
item_accuracy
subject_mean_models
models_within_subject
validity_binary
response_binary
```

Il fallait donc créer un fichier unique et vérifié.

---

# 20. Pourquoi cette étape conduit à la création de `dataset_analysis_E1.csv`

L’analyse préliminaire a révélé que nous ne pouvions pas ajuster proprement les modèles directement sur les fichiers bruts.

Il manquait un niveau intermédiaire.

```text
Fichier expérimental brut
                    \
                     → dataset analytique → modèles
                    /
Fichier MReasoner brut
```

Le futur fichier `dataset_analysis_E1.csv` devait remplir plusieurs fonctions.

## 20.1 Rassembler toutes les informations

Chaque ligne devait contenir simultanément :

- l’identité du participant ;
- l’identité de l’item ;
- la position de l’essai ;
- la confiance ;
- l’exactitude ;
- la réponse ;
- la condition ;
- le type de tâche ;
- l’entropie ;
- la précision du participant ;
- le nombre de modèles ;
- la décomposition inter/intra-individuelle.

---

## 20.2 Vérifier les fusions

Le fichier devait prouver que :

- chaque essai avait trouvé sa ligne MReasoner ;
- les prémisses correspondaient ;
- aucun participant expérimental n’était perdu ;
- aucun nombre de modèles nécessaire n’était manquant.

---

## 20.3 Produire une base stable pour tous les scripts

Sans dataset analytique central, chaque script aurait dû refaire :

- le nettoyage ;
- les conversions ;
- la fusion ;
- les calculs d’entropie ;
- les calculs de précision ;
- les vérifications.

Cela aurait créé un risque d’incohérence.

Avec un fichier central :

```text
dataset_analysis_E1.csv
```

tous les modèles utilisent la même définition des variables.

---

## 20.4 Séparer préparation et analyse

Cette séparation est une bonne pratique scientifique :

```text
Étape de préparation
→ produire des données analytiques fiables

Étape de modélisation
→ utiliser ces données sans les reconstruire différemment
```

> **Analogie**
>
> Avant de cuisiner plusieurs recettes, on lave, découpe et range les ingrédients.
>
> On ne redécoupe pas différemment les mêmes légumes pour chaque recette sans garder de trace.
>
> `dataset_analysis_E1.csv` joue le rôle d’un ensemble d’ingrédients préparés, contrôlés et documentés.

---

# Bilan de l’étape 1

L’analyse préliminaire a permis de passer d’une question générale — comprendre la confiance dans une tâche de raisonnement — à une stratégie statistique structurée.

Nous avons établi que :

1. la confiance serait la variable dépendante principale ;
2. chaque ligne représenterait un essai ;
3. les observations étaient regroupées par participant et par item ;
4. une régression ordinaire ignorerait cette dépendance ;
5. un modèle linéaire mixte croisé était plus adapté ;
6. les participants et les items auraient des intercepts aléatoires ;
7. la condition et la séquence seraient d’abord introduites comme contrôles ;
8. l’entropie, la précision et les modèles mentaux seraient les prédicteurs cognitifs principaux ;
9. le nombre de modèles mentaux devait être décomposé en composantes interindividuelle et intra-individuelle ;
10. la validité et le type de tâche ne devaient pas être introduits naïvement ensemble ;
11. l’effet plafond et la stabilité des simulations MReasoner nécessiteraient des analyses de sensibilité ;
12. toutes les variables devaient être rassemblées dans un dataset analytique central.

La prochaine étape sera donc consacrée exclusivement à la construction de :

```text
dataset_analysis_E1.csv
```

# Étape 2 — Construction de `dataset_analysis_E1.csv`

## Sommaire

1. [Rôle du dataset analytique](#1-rôle-du-dataset-analytique)
2. [Différence entre données brutes et données analytiques](#2-différence-entre-données-brutes-et-données-analytiques)
3. [Les deux fichiers utilisés](#3-les-deux-fichiers-utilisés)
4. [Structure générale du script de construction](#4-structure-générale-du-script-de-construction)
5. [Chargement du fichier expérimental](#5-chargement-du-fichier-expérimental)
6. [Nettoyage et normalisation](#6-nettoyage-et-normalisation)
7. [Construction de l’identifiant d’item](#7-construction-de-lidentifiant-ditem)
8. [Vérification du type de tâche](#8-vérification-du-type-de-tâche)
9. [Calcul de la précision des participants](#9-calcul-de-la-précision-des-participants)
10. [Calcul des statistiques des items](#10-calcul-des-statistiques-des-items)
11. [Calcul détaillé de l’entropie](#11-calcul-détaillé-de-lentropie)
12. [Chargement des résultats MReasoner](#12-chargement-des-résultats-mreasoner)
13. [Déduction du type de tâche MReasoner](#13-déduction-du-type-de-tâche-mreasoner)
14. [Fusion des deux fichiers](#14-fusion-des-deux-fichiers)
15. [Décomposition interindividuelle et intra-individuelle](#15-décomposition-interindividuelle-et-intra-individuelle)
16. [Construction des variables de contrôle](#16-construction-des-variables-de-contrôle)
17. [Création des résumés participant et item](#17-création-des-résumés-participant-et-item)
18. [Contrôle des valeurs manquantes](#18-contrôle-des-valeurs-manquantes)
19. [La variable `analysis_complete`](#19-la-variable-analysis_complete)
20. [Description de toutes les colonnes finales](#20-description-de-toutes-les-colonnes-finales)
21. [Les fichiers secondaires produits](#21-les-fichiers-secondaires-produits)
22. [Interprétation du rapport d’audit](#22-interprétation-du-rapport-daudit)
23. [Résultat final de la construction](#23-résultat-final-de-la-construction)
24. [Passage de 3 à 20 simulations](#24-passage-de-3-à-20-simulations)
25. [Limites et précautions méthodologiques](#25-limites-et-précautions-méthodologiques)
26. [Ce que cette étape a changé](#26-ce-que-cette-étape-a-changé)
27. [Pourquoi elle conduit au modèle nul](#27-pourquoi-elle-conduit-au-modèle-nul)

---

# 1. Rôle du dataset analytique

Le fichier :

```text
dataset_analysis_E1.csv
```

est le fichier central de tout le projet statistique.

Il a été créé à partir de deux sources différentes :

```text
dataset_ccobra_E1.csv
mental_models_count_E1.csv
```

Le premier fichier contenait les réponses expérimentales. Le second contenait les résultats produits par MReasoner.

Le dataset analytique rassemble ces informations dans un tableau unique où chaque ligne représente toujours :

```text
un participant × un essai expérimental
```

mais où cette ligne contient désormais toutes les variables utiles aux analyses.

---

## 1.1 Pourquoi ne pas utiliser directement les fichiers bruts ?

Le fichier expérimental ne contenait pas directement :

- l’entropie des items ;
- la précision moyenne des participants ;
- le taux de réponses Yes par item ;
- le nombre moyen de modèles mentaux du participant ;
- la variation intra-individuelle du nombre de modèles ;
- les diagnostics de fusion avec MReasoner.

Inversement, le fichier MReasoner ne contenait pas :

- les réponses expérimentales ;
- la confiance ;
- l’exactitude ;
- la séquence ;
- la condition ;
- l’identité exacte des items.

Il fallait donc réunir les deux sources.

---

## 1.2 Une ligne enrichie

Avant la construction, une ligne expérimentale ressemblait conceptuellement à ceci :

| participant | essai | item | confiance | réponse | correct |
|---:|---:|---:|---:|---|---:|
| 63873 | 1 | 125 | 99 | No | 0 |

Après la construction, la même ligne contient également :

| précision participant | entropie item | modèles MReasoner | moyenne personnelle | écart personnel |
|---:|---:|---:|---:|---:|
| 0,546875 | 0,988378 | 2,0000 | 2,583325 | −0,583325 |

Le fichier analytique ne remplace donc pas l’observation brute. Il l’enrichit.

---

## 1.3 Analogie

On peut comparer cette étape à la création du dossier complet d’un patient.

Au départ, les informations sont dispersées :

```text
dossier administratif
résultats biologiques
imagerie
questionnaires
historique
```

Avant l’analyse, il faut les réunir sous un identifiant commun, vérifier qu’elles concernent bien la même personne et signaler les données manquantes.

Dans notre projet :

```text
réponses expérimentales
+
résultats MReasoner
+
variables calculées
=
dataset analytique
```

---

# 2. Différence entre données brutes et données analytiques

## 2.1 Données brutes

Les **données brutes** sont les informations enregistrées directement par l’expérience ou produites directement par une simulation.

Exemples :

```text
confidence = 99
response = No
is_correct = 0
number_models_generated = 2.0
```

Elles sont proches de la source originale.

---

## 2.2 Données dérivées

Une **variable dérivée** est une variable calculée à partir d’autres variables.

Exemples :

```text
subject_accuracy
item_entropy
subject_mean_models
models_within_subject
```

`subject_accuracy` n’est pas directement saisie par le participant. Elle est calculée à partir de ses 64 valeurs de `is_correct`.

`item_entropy` est calculée à partir des réponses Yes et No obtenues pour un item.

---

## 2.3 Données analytiques

Un **dataset analytique** est un tableau préparé spécifiquement pour les analyses statistiques.

Il doit être :

- propre ;
- cohérent ;
- documenté ;
- reproductible ;
- complet pour les modèles prévus ;
- accompagné de diagnostics.

Un dataset analytique n’est donc pas seulement une copie du fichier brut. Il correspond à une traduction du plan scientifique en variables statistiques utilisables.

---

# 3. Les deux fichiers utilisés

# 3.1 `dataset_ccobra_E1.csv`

Ce fichier contenait les données expérimentales.

Nous avions observé :

```text
Nombre de lignes : 9024
Nombre de participants : 141
Nombre d’items : 128
Essais par participant : 64
```

Ses colonnes initiales étaient :

```text
id
sequence
domain
response_type
task
choices
response
confidence
is_correct
task_type
condition
validity
believability
conflict
stimulus
qnum
total_qnum
rt
logRT
rt_for
statementEval
```

---

## 3.2 `mental_models_count_E1.csv`

Ce fichier contenait les estimations MReasoner.

Il comportait initialement :

```text
604 lignes
151 participants
4 lignes par participant
```

Pourquoi 604 lignes ?

\[
151\times4=604
\]

Chaque participant avait une ligne pour chacun des quatre types de tâches :

```text
MP
MT
AC
DA
```

Ses colonnes étaient :

```text
subject_id
task
premise_1
premise_2
number_models_generated
std_models_generated
minimum_models_generated
maximum_models_generated
n_samples
n_parameter_sets_used
epsilon
lambda
omega
sigma
```

---

## 3.3 Pourquoi 151 participants dans MReasoner mais 141 dans l’expérience ?

Le fichier MReasoner comportait 151 identifiants, alors que les données expérimentales analysées en comportaient 141.

Cela ne constituait pas automatiquement une erreur.

Une fusion peut ignorer les participants présents uniquement dans le fichier MReasoner, tant que :

- les 141 participants expérimentaux sont tous appariés ;
- aucun essai expérimental ne reste sans valeur MReasoner ;
- aucun participant expérimental n’est accidentellement supprimé.

Le problème important n’est donc pas :

> Les deux fichiers doivent-ils avoir exactement le même nombre d’identifiants ?

La vraie question est :

> Tous les participants nécessaires à l’analyse expérimentale sont-ils présents dans le fichier MReasoner ?

Le diagnostic final a montré que la réponse était oui.

---

# 4. Structure générale du script de construction

Le script était :

```text
build_dataset_analysis_E1.py
```

Son travail peut être résumé en dix grandes opérations :

```text
1. Définir les chemins
2. Charger les réponses expérimentales
3. Nettoyer et normaliser les variables
4. Vérifier les identifiants et les items
5. Calculer les statistiques des participants
6. Calculer les statistiques des items
7. Charger et vérifier les résultats MReasoner
8. Fusionner les deux sources
9. Construire les variables inter/intra-individuelles
10. Exporter le dataset et les rapports d’audit
```

Schéma général :

```text
dataset_ccobra_E1.csv
        │
        ├── nettoyage
        ├── réponses binaires
        ├── précision participant
        └── entropie item
                    \
                     \
                      ── fusion ──> dataset_analysis_E1.csv
                     /
                    /
mental_models_count_E1.csv
        │
        ├── type de tâche déduit
        ├── vérification des prémisses
        └── statistiques MReasoner
```

---

## 4.1 Les bibliothèques principales

Même si le code exact contenait éventuellement d’autres imports, la construction reposait principalement sur des bibliothèques de ce type :

```python
from pathlib import Path
import numpy as np
import pandas as pd
```

### `pathlib.Path`

`Path` sert à manipuler les chemins de fichiers.

Exemple :

```python
BASE_DIR = Path(__file__).resolve().parent
```

Cette expression récupère le dossier contenant le script.

Elle permet ensuite d’écrire :

```python
DATA_FILE = BASE_DIR.parent / "dataset_ccobra_E1.csv"
```

au lieu d’inscrire un chemin absolu propre à une seule machine.

#### Pourquoi est-ce important ?

Un chemin absolu comme :

```text
/home/paul/Etudes/.../dataset_ccobra_E1.csv
```

fonctionne uniquement si le fichier se trouve exactement à cet emplacement.

Un chemin relatif au script rend le projet plus transportable.

---

### `pandas`

`pandas` est la bibliothèque principale de manipulation de tableaux en Python.

Sa structure centrale est le `DataFrame`.

Un `DataFrame` ressemble à une feuille Excel :

```text
lignes × colonnes
```

Exemple :

```python
data = pd.read_csv(DATA_FILE)
```

Cette ligne :

1. ouvre un fichier CSV ;
2. lit son contenu ;
3. transforme le tableau en objet Python manipulable.

`pandas` permet ensuite de :

- filtrer les lignes ;
- convertir les colonnes ;
- regrouper les observations ;
- calculer des moyennes ;
- fusionner des fichiers ;
- exporter des CSV.

---

### `numpy`

`numpy` est une bibliothèque de calcul numérique.

Elle est notamment utilisée pour :

- représenter les valeurs manquantes avec `np.nan` ;
- effectuer des calculs mathématiques ;
- calculer des logarithmes ;
- vérifier si une valeur est finie ;
- effectuer des opérations vectorisées.

Pour l’entropie, par exemple, on peut utiliser :

```python
np.log2(p)
```

qui calcule le logarithme en base 2.

---

# 5. Chargement du fichier expérimental

Une ligne de chargement typique est :

```python
experimental = pd.read_csv(EXPERIMENTAL_FILE)
```

Après le chargement, le script affichait :

```text
Nombre de lignes brutes : 9024
Colonnes : [...]
```

---

## 5.1 Pourquoi afficher ces informations ?

Cela constitue un contrôle immédiat.

Si le script indique soudainement :

```text
Nombre de lignes brutes : 8500
```

alors que nous attendions 9024, il faut arrêter l’analyse et rechercher la cause.

Les causes possibles seraient :

- mauvais fichier ;
- lignes supprimées auparavant ;
- erreur d’export ;
- changement involontaire du corpus ;
- problème de lecture.

---

## 5.2 Vérifier les colonnes

Le script vérifie que les colonnes indispensables existent.

Exemple conceptuel :

```python
required_columns = [
    "id",
    "sequence",
    "confidence",
    "is_correct",
    "task_type",
    "condition",
    "total_qnum",
]
```

Puis :

```python
missing_columns = [
    column
    for column in required_columns
    if column not in experimental.columns
]
```

### Explication

La boucle demande, pour chaque nom attendu :

> Ce nom est-il absent des colonnes du tableau ?

Si oui, il est ajouté à `missing_columns`.

Le script peut ensuite arrêter l’exécution :

```python
if missing_columns:
    raise ValueError(...)
```

### Pourquoi arrêter ?

Continuer sans une colonne essentielle pourrait créer des résultats silencieusement faux.

Par exemple, sans `total_qnum`, il serait impossible de construire correctement l’effet aléatoire item.

Il vaut mieux obtenir une erreur explicite que produire un fichier incomplet apparemment valide.

---

# 6. Nettoyage et normalisation

Le **nettoyage des données** désigne l’ensemble des opérations servant à rendre les informations cohérentes et exploitables.

Il ne s’agit pas de modifier les résultats pour qu’ils deviennent intéressants.

Il s’agit de corriger ou signaler des problèmes de format et d’identification.

---

## 6.1 Renommer `id` en `subject_id`

Le fichier expérimental utilisait :

```text
id
```

Le fichier MReasoner utilisait :

```text
subject_id
```

Pour faciliter la fusion, nous avons adopté un nom commun :

```text
subject_id
```

Une opération typique serait :

```python
experimental = experimental.rename(
    columns={"id": "subject_id"}
)
```

### Pourquoi renommer ?

Une fusion demande de savoir quelles colonnes jouent le même rôle.

Si un fichier utilise `id` et l’autre `subject_id`, le sens est le même mais le nom diffère.

Le renommage rend ce sens explicite.

---

## 6.2 Convertir les variables numériques

Une colonne lue depuis un CSV peut parfois être interprétée comme du texte.

Exemple :

```text
"99"
"100"
"75"
```

visuellement, il s’agit de nombres, mais Python pourrait les considérer comme des chaînes de caractères.

Une conversion typique est :

```python
data["confidence"] = pd.to_numeric(
    data["confidence"],
    errors="coerce",
)
```

### `errors="coerce"`

Cette option signifie :

> Si une valeur ne peut pas être convertie en nombre, remplace-la par une valeur manquante.

Exemple :

```text
"75"     → 75
"100"    → 100
"error"  → NaN
```

`NaN` signifie « Not a Number » et sert souvent à représenter une valeur manquante.

### Pourquoi ne pas ignorer les erreurs ?

Une valeur comme `"error"` ne doit pas participer à une moyenne.

La convertir en valeur manquante permet ensuite de la repérer et de décider explicitement quoi faire.

---

## 6.3 Vérifier les limites de la confiance

La confiance devait appartenir à :

\[
[0,100]
\]

Le script a compté les valeurs hors de cet intervalle.

Résultat :

```text
Confiances hors de [0, 100] : 0
```

### Pourquoi cette vérification ?

Une valeur comme :

```text
confidence = 150
```

serait impossible selon l’échelle expérimentale.

Elle pourrait signaler :

- une erreur d’enregistrement ;
- une mauvaise unité ;
- un problème de colonne ;
- une faute de saisie.

---

## 6.4 Normaliser les chaînes de caractères

Une catégorie peut être écrite de plusieurs façons :

```text
Yes
yes
YES
 Yes
Yes 
```

Pour un humain, ces valeurs semblent identiques. Pour Python, elles peuvent être différentes.

Une normalisation typique serait :

```python
data["response_normalized"] = (
    data["response"]
    .astype(str)
    .str.strip()
    .str.lower()
)
```

Cette suite d’opérations signifie :

1. convertir en texte ;
2. supprimer les espaces au début et à la fin ;
3. convertir en minuscules.

Résultat :

```text
" Yes " → "yes"
"NO"    → "no"
```

Le script a ensuite probablement reconverti l’affichage final vers :

```text
Yes
No
```

ou conservé les catégories normalisées dans une colonne séparée.

---

# 7. Construction de l’identifiant d’item

Le fichier expérimental contenait :

```text
qnum
total_qnum
```

Nous avons utilisé :

```text
total_qnum
```

comme identifiant principal de l’item, puis créé :

```text
item_id
```

Conceptuellement :

```python
data["item_id"] = data["total_qnum"]
```

---

## 7.1 Pourquoi `total_qnum` ?

D’après la description du plan :

> `total_qnum` correspond au numéro de l’item parmi les items de la tâche principale, en comptant les items Standard et Neutral.

Il identifie donc les 128 items analysés.

Le résultat final a confirmé :

```text
Items distincts : 128
```

---

## 7.2 Pourquoi ne pas utiliser `sequence` comme item ?

`sequence` indique la position de présentation pour un participant.

Deux participants peuvent voir des items différents en position 1.

Inversement, le même item peut apparaître à des positions différentes selon les participants.

Ainsi :

```text
sequence ≠ item_id
```

Cette distinction est essentielle.

---

## 7.3 Vérification des doublons participant–séquence

Le script a contrôlé les doublons avec une clé du type :

```text
subject_id + sequence
```

Résultat :

```text
Lignes impliquées dans un doublon subject_id + sequence : 0
```

### Qu’est-ce qu’un doublon ici ?

Un doublon signifierait qu’un même participant possède deux lignes pour la même position d’essai.

Exemple problématique :

| participant | sequence |
|---|---:|
| A | 12 |
| A | 12 |

Cela pourrait signifier :

- que la ligne a été dupliquée ;
- qu’un essai est enregistré deux fois ;
- que la clé ne représente pas correctement un essai.

L’absence de doublons confirme qu’une paire participant–séquence identifie bien une observation unique.

---

# 8. Vérification du type de tâche

Le fichier expérimental fournissait déjà :

```text
task_type
```

avec :

```text
MP
MT
AC
DA
```

Mais le script a aussi déduit le type de tâche à partir des prémisses formelles.

---

## 8.1 Pourquoi déduire une information déjà présente ?

C’est un contrôle de cohérence.

Si le fichier indique :

```text
task_type = MP
```

mais que les prémisses correspondent à AC, il existe une incohérence.

Le script comparait donc probablement :

```text
task_type
task_type_inferred
```

Résultat :

```text
Incohérences entre task_type et les prémisses : 0
```

Cette vérification montre que l’étiquette expérimentale et la structure logique sont cohérentes.

---

## 8.2 Construction de `task_formal`

Les prémisses formelles étaient regroupées dans une chaîne comme :

```text
All B are C/No A are C
```

Cette variable est devenue :

```text
task_formal
```

Puis une version normalisée a été créée :

```text
task_formal_normalized
```

Exemple :

```text
All B are C/No A are C
```

devient :

```text
all b are c/no a are c
```

### Pourquoi créer une version normalisée ?

Pour comparer deux chaînes sans être perturbé par :

- les majuscules ;
- les espaces ;
- la ponctuation ;
- d’éventuelles variations mineures de format.

---

## 8.3 Déduction des prémisses expérimentales

Les deux parties de `task_formal` ont été séparées en :

```text
experiment_premise_1
experiment_premise_2
```

Exemple :

```text
task_formal = All B are C/No A are C
```

donne :

```text
experiment_premise_1 = All B are C
experiment_premise_2 = No A are C
```

Ces colonnes ont ensuite servi à vérifier que la ligne MReasoner fusionnée correspondait bien à la structure logique de l’essai.

---

# 9. Calcul de la précision des participants

La variable :

```text
subject_accuracy
```

a été calculée par participant.

---

## 9.1 Regroupement avec `groupby`

En `pandas`, `groupby` permet de regrouper des lignes qui partagent une même valeur.

Exemple conceptuel :

```python
subject_accuracy = (
    data
    .groupby("subject_id")["is_correct"]
    .mean()
)
```

Cette instruction signifie :

1. regrouper les lignes par participant ;
2. sélectionner `is_correct` ;
3. calculer sa moyenne dans chaque groupe.

---

## 9.2 Pourquoi la moyenne de 0 et 1 donne-t-elle une proportion ?

Supposons :

```text
1, 0, 1, 1
```

La moyenne est :

\[
\frac{1+0+1+1}{4}
=
0{,}75
\]

Il y a donc 75 % de réponses correctes.

---

## 9.3 Résultats observés

Nous avons obtenu :

```text
Participants : 141
Précision moyenne entre participants : 0.623116
Précision médiane entre participants : 0.5625
```

### Moyenne

La moyenne de 0,623 signifie que, globalement, la proportion moyenne de réponses correctes par participant était d’environ :

\[
62{,}31\%
\]

### Médiane

La médiane était :

\[
0{,}5625
\]

Avec 64 essais :

\[
0{,}5625\times64=36
\]

Le participant médian avait donc 36 réponses correctes sur 64.

---

## 9.4 Pourquoi la moyenne est-elle supérieure à la médiane ?

La moyenne était environ 62,3 %, alors que la médiane était 56,25 %.

Cela suggère que certains participants très performants tiraient la moyenne vers le haut.

La distribution n’était donc probablement pas parfaitement symétrique.

---

## 9.5 Autres résumés participant

Le dataset final contenait aussi :

```text
subject_n_trials
subject_n_accuracy_trials
subject_correct_count
subject_mean_confidence
subject_median_confidence
subject_std_confidence
subject_n_confidence_values
subject_zero_confidence_rate
subject_hundred_confidence_rate
subject_condition
```

Ces variables ont été calculées par participant, puis réinjectées sur chaque ligne de ce participant.

### Exemple

Pour le participant 63873 :

```text
subject_n_trials = 64
subject_correct_count = 35
subject_accuracy = 35 / 64 = 0,546875
subject_mean_confidence = 99,328125
```

Toutes ses lignes contiennent ces mêmes statistiques participant.

---

# 10. Calcul des statistiques des items

Les données ont ensuite été regroupées par :

```text
item_id
```

afin de calculer des caractéristiques communes aux réponses portant sur le même item.

---

## 10.1 Nombre de lignes par item

```text
item_n_rows
```

Cette variable indique combien de lignes sont associées à l’item.

Dans notre plan :

```text
minimum = 70
médiane = 70,5
maximum = 71
```

Certains items ont donc été vus par 70 participants, d’autres par 71.

---

## 10.2 Nombre de réponses utilisables

```text
item_n_responses
```

Cette variable indique combien de réponses Yes/No valides étaient disponibles pour calculer les taux.

Elle peut différer de `item_n_rows` s’il existe des réponses manquantes ou non reconnues.

Dans notre dataset final, les items avaient des réponses complètes.

---

## 10.3 Nombre de participants distincts

```text
item_n_subjects
```

Cette variable compte les participants uniques ayant répondu à l’item.

Elle sert à repérer un éventuel problème de doublon.

Si :

```text
item_n_rows = 71
item_n_subjects = 70
```

cela suggérerait qu’un participant possède deux lignes pour cet item.

Dans les observations affichées :

```text
item_n_rows = 71
item_n_subjects = 71
```

La structure était cohérente.

---

## 10.4 Comptage des réponses

```text
item_yes_count
item_no_count
```

Exemple pour l’item 125 :

```text
item_yes_count = 31
item_no_count = 40
```

Le total est :

\[
31+40=71
\]

---

## 10.5 Taux de réponse

```text
item_yes_rate
item_no_rate
```

Pour l’item 125 :

\[
\text{item\_yes\_rate}
=
\frac{31}{71}
\approx0{,}436620
\]

\[
\text{item\_no\_rate}
=
\frac{40}{71}
\approx0{,}563380
\]

Les deux taux s’additionnent :

\[
0{,}436620+0{,}563380=1
\]

---

## 10.6 Exactitude de l’item

```text
item_accuracy
```

Pour l’item 125, si la bonne réponse est Yes et 31 personnes répondent Yes :

\[
\text{item\_accuracy}
=
\frac{31}{71}
=
0{,}436620
\]

Pour un item dont la bonne réponse est No, l’exactitude correspond au taux de No.

Ainsi, `item_accuracy` n’est pas toujours identique à `item_yes_rate`.

---

# 11. Calcul détaillé de l’entropie

La colonne :

```text
item_entropy
```

a été calculée pour chacun des 128 items.

---

## 11.1 Formule utilisée

Pour une réponse binaire :

\[
H(p)
=
-p\log_2(p)
-(1-p)\log_2(1-p)
\]

avec :

\[
p=\text{item\_yes\_rate}
\]

---

## 11.2 Pourquoi utiliser le logarithme en base 2 ?

La base 2 est naturelle lorsqu’il existe deux réponses possibles.

Avec cette base, l’entropie maximale d’une variable binaire vaut exactement 1.

Ainsi :

\[
0\leq H\leq1
\]

Cette échelle est facile à interpréter.

---

## 11.3 Exemple avec l’item 125

Nous avions :

\[
p=\frac{31}{71}\approx0{,}436620
\]

et :

\[
1-p\approx0{,}563380
\]

L’entropie est :

\[
H
=
-0{,}436620\log_2(0{,}436620)
-0{,}563380\log_2(0{,}563380)
\]

ce qui donne environ :

\[
H\approx0{,}988378
\]

Cette valeur est proche de 1.

L’item a donc suscité un désaccord presque maximal.

---

## 11.4 Exemple avec un item consensuel

Un autre item avait :

```text
70 Yes
1 No
```

Son taux de Yes était :

\[
p=\frac{70}{71}\approx0{,}985915
\]

Son entropie était environ :

\[
H\approx0{,}106792
\]

Cette valeur est faible : presque tout le monde a donné la même réponse.

---

## 11.5 Cas limites

Si :

\[
p=0
\]

la formule semble contenir :

\[
0\times\log_2(0)
\]

Or \(\log(0)\) n’est pas défini.

Mathématiquement, on utilise la limite :

\[
\lim_{p\to0^+}p\log(p)=0
\]

Le code doit donc traiter explicitement les cas \(p=0\) et \(p=1\).

Une fonction typique serait :

```python
def binary_entropy(p):
    if p in (0, 1):
        return 0.0

    return (
        -p * np.log2(p)
        -(1 - p) * np.log2(1 - p)
    )
```

Sans ce traitement, le script produirait des erreurs numériques ou des valeurs manquantes pour les items parfaitement consensuels.

---

## 11.6 Résultats globaux

Nous avons obtenu :

```text
Nombre d’items avec une entropie calculable : 128
Entropie minimale : 0.0
Entropie médiane : 0.794102
Entropie maximale : 0.999857
```

### Entropie minimale de 0

Au moins un item a produit un consensus complet.

### Médiane de 0,794

La moitié des items avait une entropie inférieure à environ 0,794 et l’autre moitié supérieure.

Une médiane relativement élevée indique que de nombreux items suscitaient une dispersion importante des réponses.

### Maximum de 0,999857

Cette valeur est presque égale à 1.

Certains items ont donc produit une répartition presque parfaitement équilibrée entre Yes et No.

---

## 11.7 Répétition de l’entropie sur les lignes

L’entropie est calculée une seule fois par item, mais elle est ajoutée à toutes les lignes de cet item.

Exemple :

| participant | item | item_entropy |
|---|---:|---:|
| A | 125 | 0,988378 |
| B | 125 | 0,988378 |
| C | 125 | 0,988378 |

Cela est nécessaire pour utiliser l’entropie comme prédicteur dans un modèle où chaque ligne est un essai.

Mais cela ne signifie pas que nous possédons 71 mesures indépendantes de l’entropie pour cet item.

Nous ne possédons qu’une caractéristique d’item répétée sur 71 lignes.

L’effet aléatoire item sert notamment à respecter cette structure.

---

# 12. Chargement des résultats MReasoner

Le second fichier était chargé séparément :

```python
models = pd.read_csv(MODEL_FILE)
```

Le script affichait :

```text
Nombre de lignes brutes : 604
Participants dans le fichier de modèles : 151
Nombre médian de lignes par participant : 4
Nombre maximal de lignes par participant : 4
```

---

## 12.1 Pourquoi vérifier le nombre de lignes par participant ?

Nous attendions quatre types de tâches.

Chaque participant devait donc avoir exactement quatre lignes MReasoner.

Si un participant n’en avait que trois, une valeur manquerait pour un type de tâche.

Si un participant en avait cinq, il pourrait exister :

- un doublon ;
- une simulation supplémentaire mal agrégée ;
- une tâche non attendue.

Le résultat indiquait :

```text
Participants ayant moins de quatre types de tâches : 0
```

La structure était donc complète.

---

## 12.2 Les statistiques de simulation

### `number_models_generated`

Moyenne du nombre de modèles générés sur les simulations.

### `std_models_generated`

Écart-type du nombre de modèles entre les simulations.

### `minimum_models_generated`

Valeur minimale observée.

### `maximum_models_generated`

Valeur maximale observée.

### `n_samples`

Nombre de simulations utilisées.

Initialement :

```text
n_samples = 3
```

### `n_parameter_sets_used`

Nombre de configurations de paramètres utilisées pour produire la moyenne.

Dans les premiers fichiers :

```text
n_parameter_sets_used = 1
```

### `epsilon`, `lambda`, `omega`, `sigma`

Paramètres de MReasoner utilisés pour la simulation.

Ils sont conservés pour assurer la traçabilité.

---

# 13. Déduction du type de tâche MReasoner

Le fichier MReasoner possédait une colonne :

```text
task
```

avec des valeurs :

```text
1
2
3
4
```

Mais ce numéro n’était pas le numéro de l’essai expérimental.

Il désignait les quatre formes de tâche.

Pour éviter une fusion erronée, le type a été déduit à partir des prémisses.

---

## 13.1 Table de correspondance

| `premise_1` | `premise_2` | Type |
|---|---|---|
| All B are C | All A are B | MP |
| All B are C | No A are C | MT |
| All B are C | All A are C | AC |
| All B are C | No A are B | DA |

Le script a créé une colonne telle que :

```text
model_task_type
```

ou a utilisé une version équivalente pour la fusion.

---

## 13.2 Fonction de déduction

Une fonction conceptuelle peut être écrite ainsi :

```python
def infer_task_type(premise_1, premise_2):
    pair = (
        normalize(premise_1),
        normalize(premise_2),
    )

    mapping = {
        ("all b are c", "all a are b"): "MP",
        ("all b are c", "no a are c"): "MT",
        ("all b are c", "all a are c"): "AC",
        ("all b are c", "no a are b"): "DA",
    }

    return mapping.get(pair)
```

### `mapping`

Un dictionnaire Python associe une clé à une valeur.

Ici :

```text
paire de prémisses → type de tâche
```

### `.get(pair)`

Cette opération renvoie le type correspondant.

Si la paire n’existe pas dans le dictionnaire, elle renvoie généralement `None` ou une valeur manquante.

---

## 13.3 Contrôle des types indéductibles

Le script a affiché :

```text
Lignes dont le type de tâche ne peut pas être déduit : 0
```

Toutes les lignes MReasoner correspondaient donc à l’une des quatre structures attendues.

---

## 13.4 Contrôle des doublons

La clé prévue pour le fichier MReasoner était :

```text
subject_id + model_task_type
```

Le script a affiché :

```text
Lignes impliquées dans un doublon subject_id + model_task_type : 0
```

Il existait donc au maximum une ligne MReasoner par participant et par type de tâche.

Cette propriété est indispensable pour éviter qu’une fusion multiplie accidentellement les lignes expérimentales.

---

# 14. Fusion des deux fichiers

La **fusion**, appelée `merge` dans `pandas`, relie les lignes de deux tableaux à partir de colonnes communes.

---

## 14.1 Clé de fusion

La fusion correcte était :

```text
subject_id + task_type
```

Pourquoi ?

Le nombre de modèles était défini au niveau :

```text
participant × type de tâche
```

Il fallait donc rechercher, pour chaque essai expérimental :

- le même participant ;
- le même type de tâche.

---

## 14.2 Pourquoi `subject_id + sequence` aurait été faux ?

Dans le fichier expérimental :

```text
sequence = numéro de l’essai, de 1 à 64
```

Dans le fichier MReasoner :

```text
task = index parmi les 4 types de tâche
```

Les deux variables ne représentent pas la même chose.

Une fusion sur ces colonnes aurait associé :

```text
essai 1 → tâche MReasoner 1
essai 2 → tâche MReasoner 2
essai 3 → tâche MReasoner 3
essai 4 → tâche MReasoner 4
```

puis aurait échoué pour les essais 5 à 64.

Ce serait une erreur conceptuelle majeure.

---

## 14.3 Fusion de type gauche

Une fusion typique est :

```python
merged = experimental.merge(
    models,
    how="left",
    left_on=["subject_id", "task_type"],
    right_on=["subject_id", "model_task_type"],
    indicator=True,
)
```

### `how="left"`

Une fusion gauche signifie :

> Conserver toutes les lignes du tableau expérimental, même si aucune correspondance MReasoner n’est trouvée.

Pourquoi choisir cette option ?

Le fichier expérimental définit les observations à analyser.

Nous ne voulons pas supprimer silencieusement un essai simplement parce que sa valeur MReasoner est absente.

Nous préférons conserver l’essai et obtenir une valeur manquante identifiable.

---

## 14.4 L’indicateur de fusion

L’option :

```python
indicator=True
```

crée une colonne spéciale indiquant l’origine de chaque ligne.

Elle prend généralement les valeurs :

```text
both
left_only
right_only
```

### `both`

La ligne expérimentale a trouvé une correspondance dans le fichier MReasoner.

### `left_only`

La ligne expérimentale n’a pas trouvé de correspondance.

### `right_only`

Une ligne MReasoner n’existe que dans le tableau de droite. Dans une fusion gauche, ces lignes ne sont normalement pas conservées dans la sortie principale.

---

## 14.5 Résultat de la fusion

Le diagnostic indiquait :

```text
both : 9024
left_only : 0
right_only : 0
```

Cela signifie que les 9 024 essais expérimentaux ont tous trouvé une correspondance MReasoner.

La colonne finale a été conservée sous un nom explicite :

```text
model_merge_status
```

avec :

```text
both
```

sur toutes les lignes.

---

## 14.6 Attention à l’interprétation

Les 9 024 appariements ne correspondent pas à 9 024 estimations MReasoner distinctes.

La même estimation participant × tâche est répétée sur plusieurs essais.

Pour un participant donné, tous les essais MP reçoivent la même valeur MP.

Schéma :

```text
Participant 63873
├── essais MP → valeur MReasoner MP = 2,3333
├── essais MT → valeur MReasoner MT = 2,0000
├── essais AC → valeur MReasoner AC = 2,3333
└── essais DA → valeur MReasoner DA = 3,6667
```

---

# 15. Décomposition interindividuelle et intra-individuelle

Après la fusion, chaque ligne possédait :

```text
number_models_generated
```

Mais nous avons vu à l’étape 1 que cette variable mélange deux niveaux d’information.

Nous avons donc créé :

```text
subject_mean_models
models_within_subject
```

---

## 15.1 Calcul de `subject_mean_models`

Pour chaque participant :

\[
\bar{M}_i
=
\text{moyenne de number\_models\_generated}
\]

Dans le fichier d’essais, chaque type de tâche apparaît le même nombre de fois. La moyenne sur les 64 essais correspond donc à la moyenne des quatre types de tâches si le plan est équilibré.

Une opération typique est :

```python
data["subject_mean_models"] = (
    data
    .groupby("subject_id")[
        "number_models_generated"
    ]
    .transform("mean")
)
```

---

## 15.2 Différence entre `agg` et `transform`

En `pandas`, `groupby().agg()` crée généralement un tableau résumé avec une ligne par groupe.

`groupby().transform()` calcule une statistique par groupe, puis la répète à la longueur du tableau original.

### Avec `agg`

```text
participant A → moyenne 2,5
participant B → moyenne 3,2
```

Deux lignes seulement.

### Avec `transform`

| participant | essai | moyenne |
|---|---:|---:|
| A | 1 | 2,5 |
| A | 2 | 2,5 |
| B | 1 | 3,2 |
| B | 2 | 3,2 |

`transform` est donc utile pour ajouter une statistique participant à chaque essai.

---

## 15.3 Calcul de `models_within_subject`

La formule est :

\[
\text{models\_within\_subject}_{ij}
=
\text{number\_models\_generated}_{ij}
-
\text{subject\_mean\_models}_i
\]

En Python :

```python
data["models_within_subject"] = (
    data["number_models_generated"]
    - data["subject_mean_models"]
)
```

---

## 15.4 Exemple du participant 63873

Ses valeurs étaient :

| Type | Nombre de modèles |
|---|---:|
| MT | 2,0000 |
| MP | 2,3333 |
| AC | 2,3333 |
| DA | 3,6667 |

La moyenne est :

\[
\frac{2+2{,}3333+2{,}3333+3{,}6667}{4}
=
2{,}583325
\]

Pour MT :

\[
2-2{,}583325
=
-0{,}583325
\]

Pour MP :

\[
2{,}3333-2{,}583325
=
-0{,}250025
\]

Pour DA :

\[
3{,}6667-2{,}583325
=
1{,}083375
\]

Ces valeurs apparaissent effectivement dans le fichier final.

---

## 15.5 Propriété du centrage personnel

Pour chaque participant, la moyenne de `models_within_subject` est théoriquement égale à zéro :

\[
\frac{1}{n_i}
\sum_j
\text{models\_within\_subject}_{ij}
=0
\]

Cette propriété a été confirmée par les corrélations :

```text
corrélation entre subject_mean_models
et models_within_subject ≈ 0
```

Cela montre que les composantes entre participants et à l’intérieur des participants ont bien été séparées.

---

## 15.6 Résultats de la décomposition initiale

```text
Moyenne globale du nombre de modèles : 2.722801
Moyenne des moyennes individuelles : 2.722801
Écart-type des moyennes individuelles : 0.471033
Participants sans variation entre types de tâches : 0
```

### Moyenne globale égale à la moyenne des moyennes

Le plan étant équilibré, chaque participant contribue avec le même nombre d’essais. La moyenne globale correspond donc à la moyenne des moyennes individuelles.

### Aucun participant sans variation

Tous les participants possédaient au moins deux valeurs différentes de nombre de modèles entre les quatre types de tâches.

Il était donc possible d’estimer une variation intra-individuelle.

---

# 16. Construction des variables de contrôle

## 16.1 `validity_binary`

La variable textuelle a été convertie :

```text
Valid   → 1
Invalid → 0
```

Une conversion conceptuelle pourrait être :

```python
validity_mapping = {
    "Valid": 1,
    "Invalid": 0,
}

data["validity_binary"] = (
    data["validity"].map(validity_mapping)
)
```

### Pourquoi utiliser `.map()` ?

`.map()` remplace chaque catégorie par une valeur définie dans un dictionnaire.

Si une catégorie inattendue apparaît, elle devient généralement manquante, ce qui permet de détecter le problème.

---

## 16.2 Relation avec `task_type`

Le tableau de contrôle était :

| `task_type` | `validity_binary = 0` | `validity_binary = 1` |
|---|---:|---:|
| AC | 2256 | 0 |
| DA | 2256 | 0 |
| MP | 0 | 2256 |
| MT | 0 | 2256 |

Le script a conclu :

```text
La validité est-elle constante à l'intérieur de chaque task_type ? True
```

Cela confirme la dépendance structurelle :

```text
MP ou MT → valide
AC ou DA → invalide
```

---

## 16.3 `sequence`

À ce stade, `sequence` était conservée sous sa forme brute :

```text
1 à 64
```

Le centrage en :

```text
sequence_c10
```

n’a pas nécessairement été effectué dans le dataset analytique lui-même. Il a été réalisé dans les scripts de modélisation.

Cette séparation est utile :

- le dataset conserve la valeur brute ;
- chaque modèle peut documenter précisément sa transformation.

---

# 17. Création des résumés participant et item

Le script ne s’est pas contenté de produire le grand dataset.

Il a aussi généré des fichiers plus petits et faciles à consulter.

---

## 17.1 Résumé des participants

Fichier :

```text
analysis_E1_outputs/subject_summary_E1.csv
```

Ce tableau contient une ligne par participant.

Il résume notamment :

- le nombre d’essais ;
- la précision ;
- le nombre de réponses correctes ;
- la confiance moyenne ;
- la confiance médiane ;
- l’écart-type de confiance ;
- la proportion de confiances égales à 0 ;
- la proportion de confiances égales à 100 ;
- la condition ;
- la moyenne des modèles ;
- la médiane des modèles ;
- la variation des modèles.

### Pourquoi ce fichier ?

Le grand dataset contient 64 lignes par participant.

Pour examiner les différences individuelles, un tableau d’une ligne par participant est plus pratique.

Il permet par exemple de repérer :

- un participant utilisant toujours 100 ;
- un participant ayant une précision exceptionnellement élevée ;
- un participant sans variation de confiance ;
- un participant avec des valeurs MReasoner atypiques.

---

## 17.2 Résumé des items

Fichier :

```text
analysis_E1_outputs/item_entropy_summary_E1.csv
```

Il contient une ligne par item.

Il résume notamment :

- le nombre de réponses ;
- le nombre de participants ;
- les comptes Yes et No ;
- les taux Yes et No ;
- l’entropie ;
- l’exactitude de l’item ;
- les propriétés expérimentales de l’item.

### Pourquoi ce fichier ?

Il facilite l’étude des propriétés des 128 items sans répéter chaque valeur sur 70 ou 71 lignes.

---

## 17.3 Contrôle de cohérence des items

Fichier :

```text
analysis_E1_outputs/item_consistency_E1.csv
```

Ce fichier vérifie qu’un même `item_id` possède des propriétés constantes.

Pour un item donné, nous attendons normalement que les colonnes suivantes soient stables :

- type de tâche ;
- condition ;
- validité ;
- croyabilité ;
- conflit ;
- stimulus ;
- structure formelle.

Si le même `item_id` apparaît parfois comme MP et parfois comme DA, l’identification des items est incorrecte.

Le rapport indiquait :

```text
Items présentant au moins une incohérence : 0
```

Les 128 identifiants représentaient donc des items cohérents.

---

# 18. Contrôle des valeurs manquantes

Une **valeur manquante** est une information absente.

Dans `pandas`, elle est souvent représentée par :

```text
NaN
```

Le script a compté les valeurs manquantes pour les variables essentielles.

Résultat :

```text
subject_id: 0
sequence: 0
item_id: 0
task_type: 0
condition: 0
confidence: 0
is_correct: 0
subject_accuracy: 0
response_normalized: 0
validity_binary: 0
item_entropy: 0
item_accuracy: 0
number_models_generated: 0
std_models_generated: 0
subject_mean_models: 0
models_within_subject: 0
```

---

## 18.1 Pourquoi compter les valeurs manquantes variable par variable ?

Une ligne peut être utilisable pour une analyse mais pas pour une autre.

Exemple :

```text
confiance présente
exactitude présente
nombre de modèles absent
```

Cette ligne pourrait servir à décrire la confiance, mais pas au modèle cognitif utilisant MReasoner.

Un diagnostic par colonne indique précisément où se trouve le problème.

---

## 18.2 Pourquoi ne pas remplacer automatiquement les valeurs manquantes ?

Remplacer une valeur manquante par une moyenne est appelé **imputation**.

Cela peut être pertinent dans certains projets, mais l’imputation doit être justifiée.

Dans notre cas, aucune imputation n’était nécessaire, car les variables principales étaient complètes.

Une imputation automatique aurait ajouté des valeurs artificielles sans nécessité.

---

# 19. La variable `analysis_complete`

Le dataset final contient :

```text
analysis_complete
analysis_missing_reasons
```

---

## 19.1 `analysis_complete`

Cette variable indique si la ligne possède toutes les informations nécessaires au modèle principal.

Valeurs :

```text
True
False
```

Dans notre résultat :

```text
Lignes complètes pour le modèle principal : 9024
Lignes incomplètes : 0
Pourcentage complet : 100 %
```

Toutes les lignes pouvaient donc être utilisées.

---

## 19.2 `analysis_missing_reasons`

Cette colonne explique pourquoi une ligne serait incomplète.

Exemples possibles :

```text
missing_confidence
missing_item_entropy
missing_model_count
premise_mismatch
```

Dans notre fichier, elle était vide parce que toutes les lignes étaient complètes.

---

## 19.3 Pourquoi conserver ces variables si tout est complet ?

Parce qu’elles rendent le pipeline plus robuste.

Si une future version contient des valeurs manquantes, les scripts pourront filtrer :

```python
data = data.loc[data["analysis_complete"]]
```

sans devoir reconstruire les critères de complétude.

Elles constituent également une trace d’audit.

---

# 20. Description de toutes les colonnes finales

Le fichier final contenait 74 colonnes. Elles peuvent être organisées par fonction.

---

## 20.1 Identifiants et position expérimentale

| Colonne | Signification | Niveau |
|---|---|---|
| `subject_id` | Identifiant du participant | Participant |
| `sequence` | Position de l’essai, de 1 à 64 | Essai |
| `item_id` | Identifiant de l’item, dérivé de `total_qnum` | Item |
| `qnum` | Numéro de question dans sa série d’origine | Item/plan |

### Comment les lire ?

Une ligne comme :

```text
subject_id = 63873
sequence = 1
item_id = 125
qnum = 8
```

signifie :

> Le participant 63873 a rencontré l’item analytique 125 lors de son premier essai ; cet item correspondait au numéro 8 dans sa série d’origine.

---

## 20.2 Variable dépendante et précision du participant

| Colonne | Signification |
|---|---|
| `confidence` | Confiance déclarée de 0 à 100 |
| `subject_accuracy` | Proportion de réponses correctes du participant |
| `item_entropy` | Entropie binaire des réponses à l’item |
| `number_models_generated` | Nombre moyen de modèles MReasoner pour participant × tâche |
| `validity_binary` | 1 si valide, 0 si invalide |
| `subject_mean_models` | Moyenne personnelle du nombre de modèles |
| `models_within_subject` | Écart du nombre de modèles à la moyenne personnelle |

Ces colonnes ont été placées au début du fichier parce qu’elles étaient centrales pour le modèle prévu.

---

## 20.3 Réponse et exactitude de l’essai

| Colonne | Signification |
|---|---|
| `response` | Réponse originale |
| `response_normalized` | Réponse normalisée en Yes/No |
| `response_binary` | Yes = 1, No = 0 |
| `is_correct` | Exactitude de la réponse, 1 ou 0 |

### Différence entre réponse et exactitude

`response_binary = 1` signifie que la personne a répondu Yes.

Cela ne signifie pas nécessairement que la réponse est correcte.

Exemple :

```text
response_binary = 1
is_correct = 0
```

La personne a répondu Yes, mais la bonne réponse était No.

---

## 20.4 Statistiques de l’item

| Colonne | Signification |
|---|---|
| `item_n_rows` | Nombre de lignes associées à l’item |
| `item_n_responses` | Nombre de réponses Yes/No utilisables |
| `item_n_subjects` | Nombre de participants distincts |
| `item_yes_count` | Nombre de réponses Yes |
| `item_no_count` | Nombre de réponses No |
| `item_yes_rate` | Proportion de réponses Yes |
| `item_no_rate` | Proportion de réponses No |
| `item_accuracy` | Proportion de réponses correctes |

Ces valeurs sont constantes sur toutes les lignes du même item.

---

## 20.5 Description logique et expérimentale

| Colonne | Signification |
|---|---|
| `task_type` | MP, MT, AC ou DA |
| `task_formal` | Prémisses formelles réunies |
| `condition` | Standard ou Neutral |
| `validity` | Valid ou Invalid |
| `believability` | Believable ou Unbelievable |
| `conflict` | Conflict ou No-conflict |
| `stimulus` | Texte du stimulus affiché |

Exemple :

```text
task_type = MT
task_formal = All B are C/No A are C
condition = Standard
validity = Valid
believability = Unbelievable
conflict = Conflict
```

---

## 20.6 Statistiques MReasoner

| Colonne | Signification |
|---|---|
| `std_models_generated` | Écart-type entre simulations |
| `minimum_models_generated` | Nombre minimal observé |
| `maximum_models_generated` | Nombre maximal observé |
| `n_samples` | Nombre de simulations |
| `n_parameter_sets_used` | Nombre de configurations de paramètres |
| `epsilon` | Paramètre MReasoner |
| `lambda` | Paramètre MReasoner |
| `omega` | Paramètre MReasoner |
| `sigma` | Paramètre MReasoner |

Ces colonnes permettent de distinguer :

- l’estimation moyenne ;
- sa stabilité ;
- la configuration computationnelle utilisée.

---

## 20.7 Résumé du participant

| Colonne | Signification |
|---|---|
| `subject_n_trials` | Nombre total d’essais |
| `subject_n_accuracy_trials` | Nombre d’essais utilisables pour la précision |
| `subject_correct_count` | Nombre de réponses correctes |
| `subject_mean_confidence` | Confiance moyenne |
| `subject_median_confidence` | Confiance médiane |
| `subject_std_confidence` | Écart-type de confiance |
| `subject_n_confidence_values` | Nombre de valeurs de confiance disponibles |
| `subject_zero_confidence_rate` | Proportion de réponses à 0 |
| `subject_hundred_confidence_rate` | Proportion de réponses à 100 |
| `subject_condition` | Condition unique du participant |

---

## 20.8 Résumé MReasoner du participant

| Colonne | Signification |
|---|---|
| `subject_median_models` | Médiane personnelle du nombre de modèles |
| `subject_std_models_across_trials` | Écart-type du nombre de modèles sur les essais |
| `subject_min_models` | Minimum personnel |
| `subject_max_models` | Maximum personnel |
| `subject_n_model_trials` | Nombre d’essais avec valeur de modèle |
| `subject_n_distinct_model_values` | Nombre de valeurs distinctes |

### Attention à `subject_n_model_trials`

Cette valeur pouvait être 64, car la statistique MReasoner était répétée sur les essais.

Elle ne signifie pas que 64 simulations MReasoner distinctes ont été produites pour le participant.

---

## 20.9 Temps de réponse

| Colonne | Signification |
|---|---|
| `rt` | Temps de réponse brut |
| `logRT` | Logarithme du temps de réponse |
| `rt_for` | Autre mesure temporelle ou temps associé à une phase spécifique |

Ces variables ont été conservées pour d’éventuelles analyses futures.

---

## 20.10 Diagnostic de fusion

| Colonne | Signification |
|---|---|
| `model_task_index` | Index de tâche dans le fichier MReasoner |
| `model_merge_status` | Résultat de la fusion |
| `model_premises_match` | Correspondance des prémisses |
| `analysis_complete` | Ligne complète pour l’analyse |
| `analysis_missing_reasons` | Raison d’incomplétude |

Dans le fichier final :

```text
model_merge_status = both
model_premises_match = True
analysis_complete = True
```

pour toutes les lignes.

---

## 20.11 Métadonnées expérimentales conservées

| Colonne | Signification |
|---|---|
| `domain` | Domaine de la tâche |
| `response_type` | Type de réponse attendu |
| `choices` | Choix disponibles |
| `statementEval` | Évaluation ou index lié au stimulus |

Ces variables n’étaient pas nécessairement utilisées dans le modèle principal, mais ont été conservées pour la traçabilité.

---

## 20.12 Colonnes de contrôle formel

| Colonne | Signification |
|---|---|
| `experiment_premise_1` | Première prémisse expérimentale |
| `experiment_premise_2` | Deuxième prémisse expérimentale |
| `task_formal_normalized` | Forme normalisée de la tâche expérimentale |
| `task_type_inferred` | Type déduit des prémisses expérimentales |
| `model_task_formal_normalized` | Forme normalisée provenant de MReasoner |
| `model_premise_1` | Première prémisse MReasoner |
| `model_premise_2` | Deuxième prémisse MReasoner |

Ces colonnes permettent d’auditer précisément la fusion.

---

# 21. Les fichiers secondaires produits

Le script a généré plusieurs fichiers dans :

```text
analysis_E1_outputs/
```

---

## 21.1 `data_audit_E1.txt`

### Rôle

Ce fichier conserve le journal de construction.

Il contient notamment :

- les fichiers lus ;
- les nombres de lignes ;
- les contrôles de doublons ;
- les valeurs manquantes ;
- les statistiques descriptives ;
- les diagnostics de fusion ;
- les avertissements.

### Pourquoi le conserver ?

Les messages du terminal disparaissent une fois la session fermée.

Le fichier d’audit fournit une trace permanente.

Il permet de répondre plus tard à des questions comme :

- Combien de lignes ont été supprimées ?
- Combien d’essais n’ont pas été fusionnés ?
- Quels types de tâches étaient présents ?
- Quel était le taux de données complètes ?

---

## 21.2 `item_entropy_summary_E1.csv`

Une ligne par item avec :

- taux de Yes ;
- taux de No ;
- entropie ;
- exactitude ;
- effectifs.

Il sert à étudier les 128 items.

---

## 21.3 `subject_summary_E1.csv`

Une ligne par participant avec :

- précision ;
- confiance moyenne ;
- taux de plafond ;
- condition ;
- statistiques MReasoner.

Il sert à étudier les différences individuelles.

---

## 21.4 `model_merge_diagnostic_E1.csv`

Ce fichier documente la fusion entre les essais et MReasoner.

Il contient les clés et les colonnes de contrôle :

```text
subject_id
sequence
item_id
task_type
task_formal
model_task_index
model_premise_1
model_premise_2
number_model_generated
model_merge_status
model_premises_match
```

### Utilité

Si certaines lignes avaient `left_only`, ce fichier aurait permis de retrouver précisément :

- le participant concerné ;
- le type de tâche ;
- les prémisses ;
- la cause potentielle de l’échec.

---

## 21.5 `predictor_correlations_E1.csv`

Ce fichier contient les corrélations descriptives entre variables numériques.

Une **corrélation** mesure la force et la direction d’une relation linéaire entre deux variables.

Elle varie généralement entre :

\[
-1\quad\text{et}\quad+1
\]

- proche de \(+1\) : les deux variables augmentent ensemble ;
- proche de \(-1\) : l’une augmente lorsque l’autre diminue ;
- proche de 0 : faible relation linéaire.

### Attention

Une corrélation n’est pas un effet ajusté.

Elle examine deux variables sans tenir compte des autres prédicteurs ni des regroupements participant–item.

---

## 21.6 `item_consistency_E1.csv`

Ce fichier vérifie que chaque `item_id` correspond à une définition expérimentale stable.

---

## 21.7 `model_count_structure_E1.csv`

Ce fichier résume la structure du fichier MReasoner.

Il permet de vérifier :

- le nombre de tâches par participant ;
- les doublons ;
- les types de tâches présents ;
- le nombre de simulations ;
- la stabilité des résultats.

---

# 22. Interprétation du rapport d’audit

Le rapport final de construction a fourni plusieurs résultats importants.

---

## 22.1 Aucun doublon essentiel

```text
Lignes impliquées dans un doublon subject_id + sequence : 0
```

Chaque essai était identifié de façon unique.

---

## 22.2 Aucune incohérence de tâche

```text
Incohérences entre task_type et les prémisses : 0
```

Les étiquettes logiques étaient cohérentes.

---

## 22.3 Tous les items étaient cohérents

```text
Items présentant au moins une incohérence : 0
```

Le même identifiant d’item ne désignait pas plusieurs stimuli contradictoires.

---

## 22.4 Fusion parfaite pour les besoins expérimentaux

```text
both : 9024
left_only : 0
```

Chaque essai a reçu ses statistiques MReasoner.

---

## 22.5 Correspondance des prémisses

```text
Essais appariés dont les prémisses ne correspondent pas : 0
```

Même lorsqu’une clé participant–type de tâche correspond, il est possible qu’une mauvaise ligne ait été appariée.

La vérification des prémisses constitue donc une sécurité supplémentaire.

---

## 22.6 Aucun nombre de modèles manquant

```text
Essais sans nombre de modèles après fusion : 0
```

Le modèle cognitif pouvait utiliser les 9 024 lignes.

---

## 22.7 Plan parfaitement équilibré par participant

```text
Essais par participant :
minimum = 64
médiane = 64
maximum = 64
```

Tous les participants contribuaient avec le même nombre d’essais.

Cela simplifie l’interprétation des moyennes individuelles.

---

## 22.8 Répartition presque équilibrée des conditions

```text
Standard : 71 participants
Neutral : 70 participants
```

Les groupes avaient des tailles presque identiques.

Cette propriété améliore généralement la précision de la comparaison.

---

## 22.9 Répartition équilibrée des types de tâches

```text
MT : 2256 essais
MP : 2256 essais
AC : 2256 essais
DA : 2256 essais
```

Chaque type représente exactement un quart des observations :

\[
\frac{2256}{9024}=0{,}25
\]

---

## 22.10 Stabilité initiale des simulations

Avec trois simulations :

```text
564 combinaisons participant × tâche dans l’échantillon expérimental
Écart-type moyen : 0,509848
Écart-type médian : 0
Écart-type nul : 54,433 %
Amplitude max-min ≥ 2 : 16,844 %
```

### Pourquoi 564 combinaisons ?

\[
141\times4=564
\]

### Médiane nulle

Plus de la moitié des combinaisons ont produit le même nombre de modèles dans les trois simulations.

### Mais 16,844 % avec une amplitude d’au moins 2

Pour certaines combinaisons, le résultat variait beaucoup.

Cela a motivé plus tard les simulations à 10 et 20 répétitions.

---

# 23. Résultat final de la construction

Le script a créé :

```text
dataset_analysis_E1.csv
```

avec :

```text
9024 lignes
141 participants
128 items
74 colonnes
100 % de lignes complètes
```

---

## 23.1 Pourquoi 9 024 lignes ont-elles été conservées ?

Aucune ligne n’a été supprimée parce que :

- les identifiants essentiels étaient présents ;
- la confiance était valide ;
- les réponses étaient utilisables ;
- la fusion MReasoner était complète ;
- les variables calculées étaient disponibles.

---

## 23.2 Pourquoi le fichier contient-il beaucoup de valeurs répétées ?

Parce que les variables appartiennent à différents niveaux.

### Variable d’essai

```text
confidence
sequence
is_correct
```

Elle peut changer à chaque ligne.

### Variable participant

```text
subject_accuracy
subject_mean_models
condition
```

Elle est répétée sur les 64 essais d’un participant.

### Variable item

```text
item_entropy
item_accuracy
```

Elle est répétée sur toutes les réponses à l’item.

### Variable participant × tâche

```text
number_models_generated
models_within_subject
```

Elle est répétée sur les essais du même type pour un participant.

Cette répétition est normale. Elle reflète la structure hiérarchique et croisée des données.

---

# 24. Passage de 3 à 20 simulations

Plus tard, nous avons produit :

```text
mental_models_count_E1_n10.csv
mental_models_count_E1_n20.csv
```

puis reconstruit :

```text
dataset_analysis_E1_n20.csv
```

---

## 24.1 Pourquoi reconstruire le dataset ?

Le nombre de modèles intervient dans trois colonnes liées :

```text
number_models_generated
subject_mean_models
models_within_subject
```

Si `number_models_generated` change avec davantage de simulations, les deux variables dérivées doivent être recalculées.

Il ne suffit pas de remplacer manuellement une colonne.

---

## 24.2 Ce qui ne change pas

Le passage à 20 simulations ne modifie pas :

- les réponses expérimentales ;
- la confiance ;
- l’exactitude ;
- la condition ;
- la séquence ;
- l’entropie ;
- l’identité des participants ;
- l’identité des items.

Il modifie seulement les variables provenant directement ou indirectement de MReasoner.

---

## 24.3 Pourquoi conserver les deux versions ?

Nous avons conservé :

```text
dataset_analysis_E1_n3.csv
dataset_analysis_E1_n20.csv
```

Cela permet de comparer les résultats.

Cette pratique est préférable à l’écrasement silencieux de l’ancien fichier.

Elle assure la reproductibilité de l’analyse de sensibilité.

---

## 24.4 Résultat de la comparaison

Les estimations à 10 et 20 simulations étaient très proches :

\[
r_{\text{Pearson}}=0{,}981
\]

(voir [coefficient de Pearson](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient))
La différence absolue moyenne entre 10 et 20 était :

\[
0{,}187
\]

Les estimations à 3 et 20 différaient davantage :

\[
\text{différence absolue moyenne}=0{,}384
\]

Cela a justifié l’utilisation de 20 simulations pour la version finale.

---

# 25. Limites et précautions méthodologiques

## 25.1 `subject_accuracy` utilise les mêmes essais

La précision individuelle est calculée à partir des mêmes observations que celles analysées.

Elle n’est pas une mesure externe de compétence.

L’interprétation doit rester associative.

---

## 25.2 `item_entropy` utilise les réponses du même échantillon

L’entropie décrit le désaccord observé parmi les participants étudiés.

Elle ne doit pas être présentée comme une propriété physique ou universelle de l’item.

---

## 25.3 Les statistiques agrégées sont répétées

La présence de `item_entropy` sur 70 ou 71 lignes ne fournit pas 70 ou 71 mesures indépendantes d’entropie.

De même, la présence de `subject_accuracy` sur 64 lignes ne fournit pas 64 mesures indépendantes de précision.

Le modèle mixte doit tenir compte de ces niveaux de regroupement.

---

## 25.4 Le nombre de modèles n’est pas spécifique à l’item exact

L’estimation MReasoner était définie au niveau :

```text
participant × type de tâche
```

et non :

```text
participant × item exact
```

Il faut donc éviter d’écrire :

> MReasoner a prédit le nombre de modèles générés pour chaque item exact.

La formulation correcte est :

> MReasoner a fourni une estimation pour chaque combinaison participant × type de tâche, répétée sur les essais correspondants.

---

## 25.5 Validité et type de tâche sont structurellement liés

Cette dépendance a été documentée dans le dataset afin de guider les modèles ultérieurs.

---

## 25.6 Un dataset complet n’est pas forcément un dataset parfait

L’absence de valeurs manquantes signifie que toutes les variables requises sont disponibles.

Cela ne garantit pas :

- que toutes les variables mesurent parfaitement les concepts théoriques ;
- que les hypothèses statistiques sont respectées ;
- que les relations sont causales ;
- que le modèle est correctement spécifié.

La complétude est une condition utile, mais non suffisante.

---

# 26. Ce que cette étape a changé

Avant la construction, nous avions deux fichiers séparés et des questions générales.

Après la construction, nous possédions un tableau où chaque question scientifique correspondait à une variable clairement définie.

| Question | Variable |
|---|---|
| Quel participant ? | `subject_id` |
| Quel item ? | `item_id` |
| À quel moment ? | `sequence` |
| Quel niveau de confiance ? | `confidence` |
| Réponse correcte ? | `is_correct` |
| Performance générale ? | `subject_accuracy` |
| Désaccord autour de l’item ? | `item_entropy` |
| Nombre de modèles ? | `number_models_generated` |
| Différence entre participants ? | `subject_mean_models` |
| Variation interne ? | `models_within_subject` |
| Condition expérimentale ? | `condition` |
| Forme logique ? | `task_type` |
| Validité ? | `validity_binary` |

Le dataset analytique matérialise donc la stratégie scientifique.

---

# 27. Pourquoi elle conduit au modèle nul

Une fois le dataset construit, nous étions prêts à modéliser la confiance.

Mais avant d’introduire condition, entropie ou MReasoner, il fallait répondre à une question plus fondamentale :

> Quelle part de la variation de confiance se situe entre participants, entre items et entre essais ?

Cette question ne nécessite encore aucun prédicteur cognitif.

Elle nécessite un modèle contenant seulement :

- une moyenne générale
- un effet aléatoire participant
- un effet aléatoire item
- un résidu


Ce modèle est appelé **modèle mixte nul**

Il servira de point de départ à toutes les comparaisons ultérieures.

---

# Bilan de l’étape 2

La construction de `dataset_analysis_E1.csv` a permis de :

1. charger et contrôler les 9 024 essais expérimentaux ;
2. identifier correctement les 141 participants et les 128 items ;
3. normaliser les réponses et les variables catégorielles ;
4. calculer la précision de chaque participant ;
5. calculer les taux de réponses et l’entropie de chaque item ;
6. vérifier la cohérence des types de tâches et des items ;
7. identifier les quatre types de tâches dans le fichier MReasoner ;
8. fusionner les données sur la clé correcte `subject_id + task_type` ;
9. vérifier les prémisses après la fusion ;
10. obtenir une valeur MReasoner pour chacun des 9 024 essais ;
11. décomposer le nombre de modèles en composantes interindividuelle et intra-individuelle ;
12. documenter la validité, le type de tâche et leur dépendance structurelle ;
13. produire des résumés par participant et par item ;
14. produire des diagnostics de fusion et de complétude ;
15. obtenir un dataset entièrement exploitable pour les modèles statistiques ;
16. préparer une version finale utilisant 20 simulations MReasoner.

Le fichier analytique constitue désormais la source commune de tous les scripts de modélisation.

# Étape 3 — Le modèle mixte nul avec `fit_null_crossed_mixed_model_E1.py`

## Sommaire

1. [Rôle de cette étape](#1-rôle-de-cette-étape)
2. [Qu’est-ce qu’un modèle nul ?](#2-quest-ce-quun-modèle-nul)
3. [Pourquoi commencer par un modèle nul ?](#3-pourquoi-commencer-par-un-modèle-nul)
4. [La notion de variance](#4-la-notion-de-variance)
5. [Décomposition de la variance dans notre modèle](#5-décomposition-de-la-variance-dans-notre-modèle)
6. [La structure participant–item croisée](#6-la-structure-participantitem-croisée)
7. [Formulation mathématique du modèle nul](#7-formulation-mathématique-du-modèle-nul)
8. [Les hypothèses du modèle](#8-les-hypothèses-du-modèle)
9. [Maximum de vraisemblance et REML](#9-maximum-de-vraisemblance-et-reml)
10. [Organisation générale du script](#10-organisation-générale-du-script)
11. [Importation des bibliothèques](#11-importation-des-bibliothèques)
12. [Configuration des chemins](#12-configuration-des-chemins)
13. [Chargement des données](#13-chargement-des-données)
14. [Sélection des observations utilisables](#14-sélection-des-observations-utilisables)
15. [Préparation des identifiants](#15-préparation-des-identifiants)
16. [Construction du modèle croisé dans `statsmodels`](#16-construction-du-modèle-croisé-dans-statsmodels)
17. [La formule fixe `confidence ~ 1`](#17-la-formule-fixe-confidence--1)
18. [Ajustement avec plusieurs optimiseurs](#18-ajustement-avec-plusieurs-optimiseurs)
19. [Ajustement REML](#19-ajustement-reml)
20. [Ajustement ML](#20-ajustement-ml)
21. [Lecture du tableau de résultats](#21-lecture-du-tableau-de-résultats)
22. [Interprétation de l’intercept](#22-interprétation-de-lintercept)
23. [Interprétation de la variance participant](#23-interprétation-de-la-variance-participant)
24. [Interprétation de la variance item](#24-interprétation-de-la-variance-item)
25. [Interprétation de la variance résiduelle](#25-interprétation-de-la-variance-résiduelle)
26. [Proportions de variance et ICC](#26-proportions-de-variance-et-icc)
27. [Les prédictions du modèle nul](#27-les-prédictions-du-modèle-nul)
28. [RMSE et MAE](#28-rmse-et-mae)
29. [Les effets aléatoires estimés](#29-les-effets-aléatoires-estimés)
30. [Les diagnostics graphiques](#30-les-diagnostics-graphiques)
31. [Les fichiers CSV produits](#31-les-fichiers-csv-produits)
32. [Les fichiers texte et JSON](#32-les-fichiers-texte-et-json)
33. [Les graphiques produits](#33-les-graphiques-produits)
34. [Résultats numériques complets](#34-résultats-numériques-complets)
35. [Ce que le modèle nul permet de conclure](#35-ce-que-le-modèle-nul-permet-de-conclure)
36. [Ce que le modèle nul ne permet pas de conclure](#36-ce-que-le-modèle-nul-ne-permet-pas-de-conclure)
37. [Limites de cette première modélisation](#37-limites-de-cette-première-modélisation)
38. [Pourquoi cette étape conduit au modèle de contrôle](#38-pourquoi-cette-étape-conduit-au-modèle-de-contrôle)
39. [Bilan pédagogique](#39-bilan-pédagogique)

---

# 1. Rôle de cette étape

Une fois `dataset_analysis_E1.csv` construit et vérifié, nous pouvions commencer la modélisation statistique.

Nous n’avons toutefois pas immédiatement ajouté :

- la condition expérimentale ;
- la séquence ;
- l’entropie ;
- la précision du participant ;
- le nombre de modèles mentaux ;
- la validité.

Nous avons commencé par le modèle le plus simple compatible avec la structure des données :

```text
une moyenne générale de confiance
+
des différences entre participants
+
des différences entre items
+
une variation résiduelle entre essais
```

Ce modèle est appelé **modèle mixte nul**.

Le script correspondant est :

```text
fit_null_crossed_mixed_model_E1.py
```

Le mot `crossed` indique que le modèle contient des effets aléatoires croisés pour :

```text
les participants
et
les items
```

---

## 1.1 La question centrale

Le modèle nul répond à la question :

> Avant même d’étudier les prédicteurs cognitifs, comment la variabilité de la confiance se répartit-elle entre les participants, les items et les essais ?

Il permet donc de distinguer trois sources générales :

```text
Différences stables entre participants
Différences stables entre items
Variations restantes entre essais
```

---

## 1.2 Exemple intuitif

Supposons que deux participants répondent aux mêmes problèmes.

```text
Participant A : confiance généralement autour de 90
Participant B : confiance généralement autour de 60
```

Cette différence correspond à de la variabilité entre participants.

Supposons ensuite que certains items produisent une confiance élevée chez presque tout le monde :

```text
Item X : confiance généralement élevée
Item Y : confiance généralement faible
```

Cette différence correspond à de la variabilité entre items.

Enfin, même pour un participant et un item donnés, une partie du comportement reste imprévisible :

- hésitation momentanée ;
- distraction ;
- fatigue ;
- erreur de mesure ;
- facteur non inclus dans le modèle.

Cette dernière partie correspond à la variabilité résiduelle.

---

# 2. Qu’est-ce qu’un modèle nul ?

## 2.1 Définition simple

Un **modèle nul** est un modèle ne contenant aucun prédicteur explicatif substantiel.

Dans une régression ordinaire, sa forme minimale est :

\[
Y_i=\beta_0+\varepsilon_i
\]

Ici :

- \(Y_i\) est la confiance observée ;
- \(\beta_0\) est la moyenne générale estimée ;
- \(\varepsilon_i\) est l’écart entre l’observation et cette moyenne.

Notre modèle est plus riche, car les données sont regroupées par participant et par item :

\[
Y_{ij}
=
\beta_0
+
u_i
+
v_j
+
\varepsilon_{ij}
\]

Il est donc « nul » au niveau des prédicteurs fixes, mais il n’est pas vide.

Il contient :

- un intercept général ;
- un intercept aléatoire participant ;
- un intercept aléatoire item ;
- une erreur résiduelle.

---

## 2.2 Pourquoi le mot « nul » ?

Le terme signifie ici :

> Aucun effet explicatif fixe n’est encore ajouté, à l’exception de l’intercept.

Cela ne signifie pas :

- que le modèle ne fait rien ;
- que tous les effets sont nuls ;
- que la confiance vaut zéro ;
- que le modèle teste l’absence de toute variation.

Au contraire, le modèle nul estime précisément la quantité de variation existant entre participants et entre items.

---

## 2.3 Analogie avec une classe

Imaginons que nous étudions les notes scolaires.

Avant d’expliquer les notes avec :

- le nombre d’heures de travail ;
- le sommeil ;
- la méthode pédagogique ;

nous pouvons construire un modèle indiquant seulement :

```text
note moyenne générale
+
différence propre à chaque élève
+
différence propre à chaque exercice
+
variation restante
```

Ce modèle constitue une photographie de départ.

Notre modèle nul joue exactement ce rôle pour la confiance.

---

# 3. Pourquoi commencer par un modèle nul ?

Le modèle nul remplit plusieurs fonctions importantes.

---

## 3.1 Établir une référence

Les modèles suivants seront comparés à une situation de base.

Nous pourrons demander :

> Le modèle avec la condition et la séquence explique-t-il mieux les données que le modèle nul ?

Puis :

> Le modèle cognitif explique-t-il mieux les données que le modèle de contrôle ?

La progression devient :

```text
Modèle nul
    ↓ ajout de condition et séquence
Modèle de contrôle
    ↓ ajout des prédicteurs cognitifs
Modèle cognitif
```

Sans modèle de référence, il serait difficile de quantifier ce que les nouvelles variables apportent.

---

## 3.2 Estimer la moyenne générale

Le modèle nul estime la confiance moyenne en tenant compte de la structure participant–item.

Il fournit donc une valeur centrale de référence.

Dans nos résultats :

\[
\hat{\beta}_0\approx75{,}736
\]

La confiance générale estimée était donc proche de 75,7 sur 100.

---

## 3.3 Décomposer la variance

Le modèle nul sépare la variation totale en :

\[
\text{variance participant}
+
\text{variance item}
+
\text{variance résiduelle}
\]

Cette décomposition répond à une question essentielle :

> La confiance dépend-elle surtout de la personne, surtout de l’item ou surtout de fluctuations propres aux essais ?

---

## 3.4 Justifier le modèle mixte

Si la variance participant et la variance item étaient toutes deux pratiquement nulles, un modèle mixte serait peut-être inutilement complexe.

Si elles sont importantes, cela confirme que les regroupements doivent être pris en compte.

Dans nos données :

- la variance participant était importante ;
- la variance item était plus faible mais non nulle.

Le modèle nul a donc confirmé l’intérêt de la structure mixte.

---

## 3.5 Calculer les ICC

Le modèle nul permet de calculer des **coefficients de corrélation intraclasse**, ou ICC.

L’ICC mesure la ressemblance attendue entre des observations appartenant au même groupe.

Nous avons calculé :

- un ICC participant ;
- un ICC item.

Ces mesures seront expliquées en détail plus loin.

---

# 4. La notion de variance

Pour comprendre le modèle nul, il faut comprendre la **variance**.

---

## 4.1 Définition intuitive

La variance mesure à quel point des valeurs sont dispersées autour de leur moyenne.

Deux séries peuvent avoir la même moyenne mais des dispersions très différentes.

### Série A

```text
74, 75, 75, 76
```

### Série B

```text
40, 60, 90, 105
```

Les deux pourraient avoir une moyenne similaire, mais la seconde série est beaucoup plus dispersée.

---

## 4.2 Formule

Pour des valeurs \(x_1,\dots,x_n\), la variance empirique est :

\[
s^2
=
\frac{1}{n-1}
\sum_{i=1}^{n}
(x_i-\bar{x})^2
\]

où :

- \(\bar{x}\) est la moyenne ;
- \(x_i-\bar{x}\) est l’écart à la moyenne ;
- les écarts sont élevés au carré ;
- les carrés sont additionnés puis moyennés.

---

## 4.3 Pourquoi élever les écarts au carré ?

Cela remplit plusieurs fonctions :

1. les écarts positifs et négatifs ne s’annulent pas ;
2. les grandes différences reçoivent davantage de poids ;
3. la variance possède de bonnes propriétés mathématiques.

---

## 4.4 Exemple simple

Considérons :

```text
60, 70, 80
```

La moyenne est :

\[
\bar{x}=70
\]

Les écarts sont :

```text
−10, 0, +10
```

Les carrés sont :

```text
100, 0, 100
```

La variance empirique est :

\[
s^2
=
\frac{100+0+100}{3-1}
=
100
\]

---

## 4.5 La variance est exprimée en unités au carré

La confiance est mesurée en points.

La variance est donc mesurée en :

```text
points de confiance au carré
```

Cette unité est difficile à interpréter directement.

C’est pourquoi on calcule souvent l’**écart-type** :

\[
s=\sqrt{s^2}
\]

Dans l’exemple :

\[
s=\sqrt{100}=10
\]

L’écart-type revient dans l’unité originale :

```text
10 points de confiance
```

---

## 4.6 Analogie géographique

La moyenne est comparable au centre d’une ville.

La variance mesure à quel point les habitants vivent loin de ce centre.

- faible variance : presque tout le monde habite près du centre ;
- grande variance : les habitants sont largement dispersés.

L’écart-type donne une distance typique dans l’unité originale.

---

# 5. Décomposition de la variance dans notre modèle

Le modèle nul suppose :

\[
Y_{ij}
=
\beta_0+u_i+v_j+\varepsilon_{ij}
\]

Chaque partie aléatoire possède sa propre variance.

---

## 5.1 Variance participant

\[
u_i\sim\mathcal{N}(0,\sigma^2_{\text{participant}})
\]

La quantité :

\[
\sigma^2_{\text{participant}}
\]

mesure à quel point les niveaux généraux de confiance des participants diffèrent entre eux.

Une grande valeur signifie que certains participants utilisent l’échelle beaucoup plus haut ou plus bas que d’autres.

---

## 5.2 Variance item

\[
v_j\sim\mathcal{N}(0,\sigma^2_{\text{item}})
\]

La quantité :

\[
\sigma^2_{\text{item}}
\]

mesure à quel point les niveaux généraux de confiance diffèrent entre les items.

Une grande valeur signifierait que certains items inspirent systématiquement beaucoup plus de confiance que d’autres.

---

## 5.3 Variance résiduelle

\[
\varepsilon_{ij}
\sim
\mathcal{N}(0,\sigma^2_{\text{résiduelle}})
\]

Cette variance correspond à ce qui reste au niveau des observations après prise en compte :

- de la moyenne générale ;
- du participant ;
- de l’item.

Elle contient notamment :

- la variation momentanée ;
- les facteurs non mesurés ;
- les erreurs de mesure ;
- les relations non incluses dans le modèle.

---

## 5.4 Variance totale

Dans ce modèle simple, la variance totale peut être approximativement décomposée comme :

\[
\sigma^2_{\text{totale}}
=
\sigma^2_{\text{participant}}
+
\sigma^2_{\text{item}}
+
\sigma^2_{\text{résiduelle}}
\]

Dans nos résultats REML :

\[
\sigma^2_{\text{participant}}
=
200{,}930
\]

\[
\sigma^2_{\text{item}}
=
11{,}875
\]

\[
\sigma^2_{\text{résiduelle}}
=
285{,}412
\]

Donc :

\[
\sigma^2_{\text{totale}}
=
200{,}930+11{,}875+285{,}412
\]

\[
\sigma^2_{\text{totale}}
\approx498{,}217
\]

---

# 6. La structure participant–item croisée

## 6.1 Deux regroupements simultanés

Chaque observation appartient simultanément :

- à un participant ;
- à un item.

Exemple :

```text
Observation 1
├── participant 63873
└── item 125
```

Une autre observation peut partager seulement le participant :

```text
Observation 2
├── participant 63873
└── item 17
```

Une autre peut partager seulement l’item :

```text
Observation 3
├── participant 70001
└── item 125
```

Le modèle doit donc reconnaître les deux types de ressemblance.

---

## 6.2 Pourquoi les effets sont-ils croisés ?

Ils sont croisés parce que les participants et les items ne sont pas emboîtés les uns dans les autres.

Une structure **emboîtée** serait par exemple :

```text
élèves dans des classes
```

Chaque élève appartient normalement à une seule classe.

Dans notre cas, un participant répond à plusieurs items et un item reçoit les réponses de plusieurs participants.

Schéma :

```text
                 Item 1   Item 2   Item 3
Participant A      ×        ×        ×
Participant B      ×        ×        ×
Participant C      ×        ×        ×
```

Les deux dimensions se croisent.

---

## 6.3 Pourquoi ne pas utiliser `groups=subject_id` uniquement ?

Dans `statsmodels`, une utilisation simple de `MixedLM` consiste souvent à écrire :

```python
groups=data["subject_id"]
```

Cela crée un effet aléatoire participant.

Mais nous avions aussi besoin d’un effet aléatoire item.

Le script a donc utilisé une technique fondée sur les **composantes de variance**, appelée `vc_formula`.

---

# 7. Formulation mathématique du modèle nul

Pour l’observation produite par le participant \(i\) sur l’item \(j\) :

\[
Y_{ij}
=
\beta_0+u_i+v_j+\varepsilon_{ij}
\]

avec :

\[
u_i\sim\mathcal{N}(0,\sigma^2_u)
\]

\[
v_j\sim\mathcal{N}(0,\sigma^2_v)
\]

\[
\varepsilon_{ij}
\sim
\mathcal{N}(0,\sigma^2_\varepsilon)
\]

---

## 7.1 Signification de chaque symbole

| Symbole | Signification |
|---|---|
| \(Y_{ij}\) | Confiance observée |
| \(\beta_0\) | Moyenne générale |
| \(u_i\) | Décalage propre au participant |
| \(v_j\) | Décalage propre à l’item |
| \(\varepsilon_{ij}\) | Variation résiduelle de l’essai |

---

## 7.2 Exemple numérique

Supposons :

\[
\beta_0=75{,}7
\]

Pour un participant très confiant :

\[
u_i=+12
\]

Pour un item légèrement peu rassurant :

\[
v_j=-3
\]

Pour un essai particulier :

\[
\varepsilon_{ij}=+2
\]

Alors :

\[
Y_{ij}
=
75{,}7+12-3+2
=
86{,}7
\]

La confiance observée serait proche de 87.

---

## 7.3 Pourquoi les effets aléatoires sont-ils centrés sur zéro ?

L’intercept \(\beta_0\) représente la moyenne générale.

Les effets participants et items représentent des écarts autour de cette moyenne.

Par convention :

\[
E(u_i)=0
\]

et :

\[
E(v_j)=0
\]

Certains effets sont positifs, d’autres négatifs.

La moyenne de tous les décalages est approximativement nulle.

---

# 8. Les hypothèses du modèle

Le modèle nul repose sur plusieurs hypothèses.

---

## 8.1 Additivité

Le modèle suppose que les composantes s’additionnent :

```text
moyenne générale
+ effet participant
+ effet item
+ résidu
```

Il ne contient encore aucune interaction complexe.

---

## 8.2 Normalité des effets aléatoires

Les effets participants sont supposés suivre approximativement une distribution normale.

Même principe pour les items.

Cela signifie que :

- beaucoup d’effets sont proches de zéro ;
- quelques effets sont plus fortement positifs ou négatifs ;
- les valeurs extrêmes sont moins fréquentes.

---

## 8.3 Normalité des résidus

Les résidus sont supposés approximativement normaux.

Cette hypothèse sera imparfaite dans notre projet à cause :

- de la borne 0–100 ;
- de l’accumulation à 100 ;
- de l’asymétrie de la confiance.

Le modèle nul sert néanmoins de point de départ, puis les diagnostics permettent d’évaluer cette limite.

---

## 8.4 Variance résiduelle constante

Le modèle suppose une variance résiduelle commune.

Il ne prévoit pas, par exemple :

```text
une variance résiduelle pour Standard
une autre pour Neutral
```

ni :

```text
une variance différente selon le niveau de confiance prédit
```

---

## 8.5 Indépendance conditionnelle

Une fois les effets participant et item pris en compte, les résidus restants sont supposés suffisamment indépendants.

Le modèle reconnaît donc une dépendance entre les observations partageant un participant ou un item, mais suppose que cette dépendance est correctement représentée par les effets aléatoires inclus.

---

# 9. Maximum de vraisemblance et REML

Le script a ajusté le modèle deux fois :

```text
REML
ML
```

Pour comprendre cette décision, il faut introduire la **vraisemblance**.

---

## 9.1 Qu’est-ce que la vraisemblance ?

La vraisemblance mesure à quel point un ensemble de paramètres rend les données observées plausibles selon le modèle.

Le raisonnement est :

> Si la moyenne, les variances et les effets du modèle avaient ces valeurs, à quel point les données que nous avons réellement observées seraient-elles plausibles ?

Le modèle cherche les paramètres qui maximisent cette plausibilité.

---

## 9.2 Exemple avec une pièce de monnaie

Supposons que nous lancions une pièce dix fois et obtenions neuf fois face.

Deux hypothèses :

```text
Hypothèse A : probabilité de face = 0,5
Hypothèse B : probabilité de face = 0,9
```

Observer neuf faces est possible avec les deux hypothèses, mais beaucoup plus vraisemblable avec la seconde.

La vraisemblance compare la compatibilité entre :

- les paramètres proposés ;
- les données observées.

---

## 9.3 Maximum de vraisemblance

Le **maximum de vraisemblance**, abrégé ML pour *maximum likelihood*, cherche les paramètres qui rendent les données aussi vraisemblables que possible.

Dans notre modèle, ML estime simultanément :

- l’intercept ;
- la variance participant ;
- la variance item ;
- la variance résiduelle.

---

## 9.4 Log-vraisemblance

Les probabilités de nombreuses observations multipliées ensemble deviennent extrêmement petites.

On utilise donc le logarithme de la vraisemblance :

\[
\log L
\]

Cette transformation convertit les multiplications en additions et facilite les calculs.

Les log-vraisemblances sont souvent négatives.

Cela n’indique pas un mauvais résultat en soi.

Dans notre modèle ML :

\[
\log L=-38670{,}894
\]

La valeur est surtout utile lorsqu’on compare des modèles ajustés sur les mêmes données avec la même méthode.

Une log-vraisemblance plus élevée, donc moins négative, indique généralement un meilleur ajustement brut.

---

## 9.5 Limite de ML pour les variances

Le maximum de vraisemblance classique peut sous-estimer les composantes de variance, notamment avec de petits échantillons de groupes.

Pourquoi ?

Parce que les mêmes données servent à :

- estimer les effets fixes ;
- estimer la variance restante.

L’incertitude liée aux effets fixes n’est pas entièrement compensée dans l’estimation naïve des variances.

---

## 9.6 Maximum de vraisemblance restreint

Le **maximum de vraisemblance restreint**, abrégé REML, ajuste l’estimation des composantes de variance pour tenir compte du fait que les effets fixes ont eux-mêmes été estimés.

REML signifie :

```text
Restricted Maximum Likelihood
```

ou :

```text
Residual Maximum Likelihood
```

L’idée intuitive est de baser plus directement l’estimation des variances sur les combinaisons des données qui ne dépendent pas des effets fixes.

---

## 9.7 Analogie avec une correction de degrés de liberté

Lorsqu’on calcule une variance empirique, on divise par :

\[
n-1
\]

et non simplement par \(n\), parce que la moyenne a été estimée à partir des données.

REML suit une intuition comparable, mais dans un cadre beaucoup plus complexe.

Il tient compte du coût lié à l’estimation des effets fixes.

---

## 9.8 Quand utiliser ML ?

ML est particulièrement utile pour comparer deux modèles ayant des effets fixes différents.

Exemple :

```text
Modèle nul :
confidence ~ 1

Modèle de contrôle :
confidence ~ condition + sequence
```

Ces modèles ne possèdent pas la même partie fixe.

Pour comparer leurs log-vraisemblances, [AIC](https://fr.wikipedia.org/wiki/Crit%C3%A8re_d'information_d'Akaike), [BIC](https://fr.wikipedia.org/wiki/Crit%C3%A8re_d'information_d'Akaike) ou réaliser un test du rapport de vraisemblance, nous utilisons ML.

---

## 9.9 Quand utiliser REML ?

REML est généralement préféré pour présenter :

- les composantes de variance finales ;
- les écarts-types aléatoires ;
- les estimations d’un modèle dont la structure fixe a déjà été choisie.

Dans notre étape nulle, les deux versions ont été conservées :

- REML comme résultat principal de variance ;
- ML comme base de comparaison avec les modèles suivants.

---

## 9.10 Pourquoi ne pas comparer directement des modèles REML ayant des effets fixes différents ?

La vraisemblance REML dépend de la matrice des effets fixes.

Si la partie fixe change, les critères REML ne reposent pas exactement sur le même objet probabiliste.

Comparer directement leurs log-vraisemblances serait donc inapproprié.

C’est pour cela que notre pipeline a suivi la règle :

```text
ML pour comparer les effets fixes
REML pour présenter le modèle final
```

---

# 10. Organisation générale du script

Le script `fit_null_crossed_mixed_model_E1.py` suivait une architecture de ce type :

```text
1. Importer les bibliothèques
2. Définir les chemins
3. Créer le dossier de sortie
4. Charger dataset_analysis_E1.csv
5. Vérifier les colonnes nécessaires
6. Filtrer les lignes complètes
7. Préparer subject_id et item_id
8. Construire le modèle nul
9. Ajuster le modèle en REML
10. Ajuster le modèle en ML
11. Extraire les coefficients fixes
12. Extraire les composantes de variance
13. Calculer les proportions et les ICC
14. Extraire les effets participants et items
15. Construire les prédictions et résidus
16. Générer les graphiques
17. Exporter les CSV, textes et JSON
```

Les détails exacts de certaines fonctions peuvent varier, mais cette architecture correspond au comportement documenté par les sorties.

---

# 11. Importation des bibliothèques

Le script utilisait des bibliothèques comme :

```python
from pathlib import Path
import json
import sys
import warnings

import numpy as np
import pandas as pd
import scipy
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels
import statsmodels.formula.api as smf
```

---

## 11.1 `json`

`json` permet d’enregistrer des résultats structurés dans un fichier lisible par une machine et par un humain.

Un fichier JSON ressemble à :

```json
{
  "n_observations": 9024,
  "n_subjects": 141,
  "converged": true,
  "participant_variance": 200.93
}
```

Il est utile pour :

- archiver des paramètres ;
- réutiliser les résultats dans un autre script ;
- conserver les métadonnées ;
- automatiser la génération de rapports.

---

## 11.2 `sys`

`sys` permet notamment de récupérer la version de Python :

```python
sys.version
```

Pourquoi enregistrer la version ?

Les bibliothèques et les algorithmes peuvent évoluer. Un résultat produit avec Python 3.12 et `statsmodels` 0.14 peut ne pas se comporter exactement comme une future version.

La version logicielle participe donc à la reproductibilité.

---

## 11.3 `warnings`

Le module `warnings` intercepte les avertissements produits par les bibliothèques.

Un avertissement n’est pas toujours une erreur fatale.

Exemples :

```text
optimisation non convergée
matrice singulière
paramètre proche d’une frontière
```

Le script peut capturer ces messages, les afficher et éventuellement essayer un autre optimiseur.

---

## 11.4 `scipy.stats`

`scipy.stats` fournit des distributions et des outils statistiques.

Il peut servir à :

- calculer des quantiles ;
- construire des intervalles ;
- produire un QQ-plot ;
- calculer certaines probabilités.

---

## 11.5 `matplotlib.pyplot`

`matplotlib` est une bibliothèque générale de graphiques.

L’alias :

```python
plt
```

permet d’écrire :

```python
plt.figure(...)
plt.scatter(...)
plt.savefig(...)
plt.close()
```

---

## 11.6 `seaborn`

`seaborn` construit des graphiques statistiques au-dessus de `matplotlib`.

Il facilite notamment :

- les histogrammes ;
- les courbes de densité ;
- les nuages de points ;
- les styles graphiques.

---

## 11.7 `statsmodels.formula.api`

`statsmodels` est la bibliothèque ayant ajusté le modèle mixte.

L’interface par formule permet d’écrire :

```python
smf.mixedlm(
    formula="confidence ~ 1",
    ...
)
```

La syntaxe ressemble à celle du langage R.

---

# 12. Configuration des chemins

Le script définissait le dossier de travail :

```python
BASE_DIR = Path(__file__).resolve().parent
```

Puis le fichier d’entrée :

```python
DATA_FILE = BASE_DIR / "dataset_analysis_E1.csv"
```

Et le dossier de sortie :

```python
OUTPUT_DIR = BASE_DIR / "null_mixed_model_E1"
```

Le dossier était créé avec une instruction comme :

```python
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)
```

---

## 12.1 `parents=True`

Cette option autorise Python à créer les dossiers parents manquants.

---

## 12.2 `exist_ok=True`

Cette option évite une erreur si le dossier existe déjà.

Sans elle, relancer le script pourrait échouer simplement parce que le dossier de sortie a déjà été créé.

---

## 12.3 Pourquoi un dossier séparé ?

Le modèle nul a produit de nombreux fichiers :

- résumés ;
- prédictions ;
- effets aléatoires ;
- graphiques ;
- JSON.

Les ranger dans :

```text
null_mixed_model_E1/
```

évite de mélanger les résultats avec ceux des modèles de contrôle ou cognitifs.

---

# 13. Chargement des données

Le script a chargé :

```text
dataset_analysis_E1.csv
```

avec :

```python
data = pd.read_csv(DATA_FILE)
```

Puis il a affiché :

```text
Nombre de lignes brutes : 9024
Nombre de colonnes : 74
```

---

## 13.1 Pourquoi relire le fichier analytique ?

Chaque script statistique doit partir d’une source explicite.

Il ne doit pas dépendre d’un objet Python laissé en mémoire par un ancien script.

Cela garantit qu’on peut exécuter :

```bash
python3 fit_null_crossed_mixed_model_E1.py
```

dans une nouvelle session et reproduire le résultat.

---

## 13.2 Vérification de l’existence du fichier

Le script utilisait probablement :

```python
if not DATA_FILE.exists():
    raise FileNotFoundError(...)
```

Cette vérification produit un message clair si le dataset n’a pas encore été construit ou si le chemin est incorrect.

---

# 14. Sélection des observations utilisables

Le dataset contenait :

```text
analysis_complete
```

Le script a filtré les lignes pour ne conserver que les observations complètes.

Résultat :

```text
Lignes retirées car analysis_complete=False : 0
```

---

## 14.1 Pourquoi filtrer malgré l’absence de lignes incomplètes ?

Cette étape rend le script robuste à de futures versions du dataset.

Si dix lignes devenaient incomplètes, le script les exclurait explicitement et l’indiquerait dans le rapport.

---

## 14.2 Conversion de la variable booléenne

Une variable booléenne contient :

```text
True
False
```

Mais après lecture d’un CSV, elle peut parfois être interprétée comme du texte :

```text
"True"
"False"
```

Une conversion prudente ressemble à :

```python
complete_mask = (
    data["analysis_complete"]
    .astype(str)
    .str.strip()
    .str.lower()
    .isin(["true", "1", "yes"])
)
```

### `.isin(...)`

Cette fonction vérifie si chaque valeur appartient à une liste autorisée.

Les valeurs reconnues comme vraies sont ici :

```text
true
1
yes
```

---

## 14.3 Le problème initial du booléen manquant

Lors d’une première version du pipeline, une erreur était apparue :

```text
TypeError: Invalid value 'nan' for dtype 'bool'
```

Cela signifiait qu’une colonne déclarée comme booléenne stricte recevait une valeur manquante `NaN`.

Une colonne booléenne ordinaire n’accepte que :

```text
True
False
```

mais pas toujours :

```text
NaN
```

La correction a consisté à gérer explicitement les valeurs manquantes ou à utiliser une conversion plus prudente.

Cette erreur s’est produite pendant la construction du dataset, mais elle explique pourquoi les scripts suivants normalisent attentivement les variables booléennes.

---

## 14.4 Suppression des données essentielles manquantes

Pour le modèle nul, les colonnes indispensables étaient seulement :

```text
confidence
subject_id
item_id
```

Le script a probablement utilisé :

```python
required_columns = {
    "subject_id",
    "item_id",
    "confidence",
}
```

Résultat :

```text
Lignes supprimées pour donnée essentielle manquante : 0
```

---

# 15. Préparation des identifiants

Les identifiants participant et item ont été convertis en chaînes de caractères :

```python
data["subject_id"] = (
    data["subject_id"].astype(str)
)

data["item_id"] = (
    data["item_id"].astype(str)
)
```

---

## 15.1 Pourquoi convertir des identifiants numériques en texte ?

Un identifiant comme :

```text
63873
```

ressemble à un nombre, mais nous ne voulons pas effectuer des opérations arithmétiques dessus.

Le participant 63874 n’est pas « un participant de plus » que 63873 au sens quantitatif.

Ces nombres sont des étiquettes.

La conversion en texte indique clairement au modèle :

```text
63873 est une catégorie
```

et non une mesure numérique continue.

---

## 15.2 Que se passerait-il sans conversion ?

Selon la formule et la bibliothèque, Python pourrait :

- traiter correctement la variable comme catégorie grâce à `C(...)` ;
- ou risquer de l’interpréter comme une variable numérique.

La conversion explicite réduit l’ambiguïté.

---

# 16. Construction du modèle croisé dans `statsmodels`

Le cœur technique du script ressemblait à ceci :

```python
model_data = data.copy()
model_data["_global_group"] = 1

model = smf.mixedlm(
    formula="confidence ~ 1",
    data=model_data,
    groups=model_data["_global_group"],
    re_formula="0",
    vc_formula={
        "item": "0 + C(item_id)",
        "subject": "0 + C(subject_id)",
    },
)
```

Ce bloc demande une explication détaillée.

---

## 16.1 Pourquoi copier les données ?

```python
model_data = data.copy()
```

Cette ligne crée une copie indépendante.

Elle permet d’ajouter une colonne technique sans modifier l’objet original.

Sans `.copy()`, certaines opérations peuvent modifier directement le tableau utilisé ailleurs ou provoquer des avertissements de `pandas`.

---

## 16.2 Le groupe global artificiel

```python
model_data["_global_group"] = 1
```

Cette ligne donne la même valeur à toutes les observations.

Le tableau contient donc :

| ligne | `_global_group` |
|---:|---:|
| 1 | 1 |
| 2 | 1 |
| 3 | 1 |
| ... | 1 |

Le résumé du modèle affiche par conséquent :

```text
No. Groups: 1
Min. group size: 9024
Max. group size: 9024
Mean group size: 9024
```

---

## 16.3 Pourquoi le résumé indique-t-il un seul groupe ?

Cela peut sembler contradictoire :

```text
141 participants
128 items
mais No. Groups = 1
```

Ce n’est pas une erreur.

Dans cette implémentation, le groupe principal de `MixedLM` est un groupe artificiel unique.

Les vrais regroupements participant et item sont représentés par :

```python
vc_formula
```

Le nombre `1` dans le résumé correspond donc au groupe technique, pas au nombre de participants.

---

## 16.4 `groups=model_data["_global_group"]`

`statsmodels.MixedLM` exige un argument `groups`.

Nous lui fournissons le groupe global artificiel.

Cette stratégie permet ensuite de définir les participants et les items comme deux familles de composantes de variance croisées.

---

## 16.5 `re_formula="0"`

Cette option indique qu’on ne souhaite pas ajouter un autre effet aléatoire associé au groupe global.

Le groupe global est seulement une construction technique.

Si nous utilisions un intercept aléatoire pour ce groupe unique, il n’aurait pas de sens comme variation entre groupes, puisqu’il n’existe qu’un seul groupe.

---

## 16.6 `vc_formula`

Le dictionnaire :

```python
vc_formula={
    "item": "0 + C(item_id)",
    "subject": "0 + C(subject_id)",
}
```

définit deux composantes de variance.

### Composante item

```python
"item": "0 + C(item_id)"
```

Elle crée une variable indicatrice pour chaque item et suppose que leurs coefficients appartiennent à une même distribution de variance.

### Composante participant

```python
"subject": "0 + C(subject_id)"
```

Elle fait la même chose pour les participants.

---

## 16.7 Que signifie `C(item_id)` ?

`C(...)` signifie que la variable doit être traitée comme catégorielle.

Pour trois items, une représentation conceptuelle pourrait être :

| Observation | Item 1 | Item 2 | Item 3 |
|---|---:|---:|---:|
| item 1 | 1 | 0 | 0 |
| item 2 | 0 | 1 | 0 |
| item 3 | 0 | 0 | 1 |

Ces colonnes servent à associer un décalage à chaque item.

Le modèle ne rapporte toutefois pas 128 variances séparées.

Il estime une variance commune de la distribution des effets items.

---

## 16.8 Que signifie le `0 +` ?

Dans une formule :

```text
0 + C(item_id)
```

le `0` retire l’intercept général de cette sous-formule.

Nous voulons une colonne propre à chaque niveau de catégorie, sans ajouter un intercept redondant dans la composante de variance.

Sans ce `0 +`, la construction de la matrice pourrait ne pas correspondre à la structure désirée.

---

## 16.9 Pourquoi cette écriture est-elle adaptée aux effets croisés ?

Tous les participants et tous les items sont placés sous un groupe global, puis leurs identités sont représentées séparément.

Cela permet d’écrire :

```text
un effet pour chaque participant
+
un effet pour chaque item
```

sans prétendre que les items sont emboîtés dans les participants ou inversement.

---

# 17. La formule fixe `confidence ~ 1`

La formule du modèle nul était :

```python
"confidence ~ 1"
```

---

## 17.1 Partie gauche

```text
confidence
```

La partie gauche indique la variable dépendante.

Le modèle cherche à expliquer la confiance.

---

## 17.2 Le symbole `~`

Le symbole se lit approximativement :

```text
est modélisé en fonction de
```

Ainsi :

```text
confidence ~ 1
```

se lit :

> La confiance est modélisée uniquement avec un intercept.

---

## 17.3 Le chiffre `1`

Dans une formule statistique, `1` représente l’intercept.

Le modèle contient donc une moyenne générale.

Il ne signifie pas que la confiance est fixée à 1.

---

## 17.4 Que se passerait-il avec `confidence ~ 0` ?

Le modèle serait forcé à ne pas avoir d’intercept fixe.

La moyenne de référence serait implicitement contrainte à zéro.

Cela serait absurde ici, car la confiance moyenne se situe autour de 75.

---

## 17.5 Pourquoi l’intercept est-il indispensable ?

Sans intercept, les effets aléatoires seraient centrés autour de zéro et devraient reconstituer artificiellement tout le niveau moyen de confiance.

L’intercept fournit le centre général, tandis que les effets aléatoires représentent des écarts autour de ce centre.

---

# 18. Ajustement avec plusieurs optimiseurs

```python
OPTIMIZATION_METHODS = [
    "lbfgs",
    "bfgs",
    "cg",
    "powell",
]
```

---

## 18.1 Qu’est-ce qu’un optimiseur ?

Un **optimiseur** est un algorithme numérique qui cherche les paramètres maximisant la vraisemblance.

On peut imaginer une montagne :

- la hauteur représente la vraisemblance ;
- la position représente les paramètres ;
- l’optimiseur cherche le sommet.

---

## 18.2 Analogie du randonneur

Supposons qu’un randonneur se trouve dans le brouillard.

Il veut atteindre le sommet.

Il peut utiliser différentes stratégies :

- suivre la pente la plus forte ;
- mémoriser les directions précédentes ;
- faire des pas plus prudents ;
- explorer plusieurs directions.

Les optimiseurs `lbfgs`, `bfgs`, `cg` et `powell` utilisent des stratégies mathématiques différentes pour rechercher le meilleur point.

---

## 18.3 Pourquoi essayer plusieurs optimiseurs ?

Un modèle mixte peut être difficile à ajuster.

Un optimiseur peut :

- converger rapidement ;
- s’arrêter trop tôt ;
- rencontrer une zone presque plate ;
- produire une erreur numérique.

Le script essaie plusieurs méthodes afin d’augmenter la robustesse.

---

## 18.4 Qu’est-ce que la convergence ?

La **convergence** signifie que l’algorithme a atteint un point où les paramètres ne changent presque plus et où il considère avoir trouvé un optimum.

Dans nos résultats :

```text
Converged: Yes
```

et :

```text
Ajustement convergé avec : lbfgs
```

Cela constitue un bon signe numérique.

---

## 18.5 Ce que la convergence ne garantit pas

La convergence ne prouve pas que :

- le modèle est scientifiquement correct ;
- les hypothèses sont respectées ;
- les variables sont bien mesurées ;
- l’optimum est nécessairement le meilleur optimum global ;
- les résultats sont causaux.

Elle signifie seulement que l’algorithme numérique a terminé correctement selon ses critères.

---

# 19. Ajustement REML

Le modèle REML a produit :

```text
Log-Likelihood: -38669.7581
Converged: Yes
Scale: 285.4123
```

Le script a utilisé :

```python
result_reml = model.fit(
    reml=True,
    method="lbfgs",
    maxiter=2000,
    full_output=True,
    disp=False,
)
```

---

## 19.1 `reml=True`

Cette option demande l’estimation par maximum de vraisemblance restreint.

---

## 19.2 `method="lbfgs"`

Cette option sélectionne l’optimiseur L-BFGS.

L-BFGS est une version à mémoire limitée de l’algorithme BFGS.

Il estime la courbure de la fonction à optimiser sans conserver une matrice trop volumineuse.

---

## 19.3 `maxiter=2000`

Cette option autorise jusqu’à 3 000 itérations.

Une **itération** est une étape de mise à jour des paramètres.

Un nombre élevé évite que l’optimiseur s’arrête uniquement parce que la limite est trop faible.

---

## 19.4 `full_output=True`

Cette option demande des informations détaillées sur le processus d’optimisation.

---

## 19.5 `disp=False`

Cette option évite que l’optimiseur imprime tous ses messages internes.

Le script produit ses propres messages plus lisibles.

---

# 20. Ajustement ML

Le même modèle a ensuite été ajusté avec :

```python
reml=False
```

Résultats :

```text
Log-Likelihood: -38670.8944
Converged: Yes
Scale: 285.4127
```

---

## 20.1 Pourquoi les résultats ML et REML diffèrent-ils légèrement ?

Les deux méthodes n’utilisent pas exactement le même critère pour estimer les variances.

Nous avons obtenu :

| Composante | REML | ML |
|---|---:|---:|
| Participant | 200,930 | 199,565 |
| Item | 11,875 | 11,865 |
| Résiduelle | 285,412 | 285,413 |

Les différences sont faibles.

Cela montre que les estimations étaient stables vis-à-vis du choix ML/REML.

---

## 20.2 Pourquoi conserver le modèle ML ?

Le modèle ML sera utilisé comme référence lors de la comparaison avec le modèle de contrôle.

La comparaison devra opposer :

```text
modèle nul ML
à
modèle de contrôle ML
```

et non un modèle nul REML à un modèle de contrôle ML.

---

# 21. Lecture du tableau de résultats

Le résumé REML était :

```text
          Mixed Linear Model Regression Results
=========================================================
Model:            MixedLM Dependent Variable: confidence
No. Observations: 9024    Method:             REML
No. Groups:       1       Scale:              285.4123
Min. group size:  9024    Log-Likelihood:     -38669.7581
Max. group size:  9024    Converged:          Yes
Mean group size:  9024.0
---------------------------------------------------------
               Coef.  Std.Err.   z    P>|z| [0.025 0.975]
---------------------------------------------------------
Intercept      75.736    1.245 60.844 0.000 73.297 78.176
item Var       11.875    0.120
subject Var   200.930    1.467
=========================================================
```

---

## 21.1 `Model: MixedLM`

Cela indique qu’un modèle linéaire mixte a été ajusté.

---

## 21.2 `Dependent Variable: confidence`

La variable expliquée est `confidence`.

---

## 21.3 `No. Observations: 9024`

Les 9 024 essais ont été utilisés.

---

## 21.4 `Method: REML`

Les résultats affichés proviennent de l’estimation REML.

---

## 21.5 `No. Groups: 1`

Cela correspond au groupe global artificiel.

Ce nombre ne signifie pas que le modèle ignore les 141 participants.

Les participants et les items sont représentés dans les composantes de variance.

---

## 21.6 `Scale: 285.4123`

Dans `MixedLM`, `Scale` correspond à la variance résiduelle estimée.

Donc :

\[
\sigma^2_{\text{résiduelle}}
=
285{,}4123
\]

---

## 21.7 `Log-Likelihood`

La log-vraisemblance REML est :

\[
-38669{,}7581
\]

Elle ne s’interprète pas seule comme un score absolu de qualité.

Elle sert surtout dans des comparaisons compatibles.

---

## 21.8 `Converged: Yes`

L’optimisation a convergé.

---

# 22. Interprétation de l’intercept

Le tableau donne :

\[
\hat{\beta}_0
=
75{,}736
\]

---

## 22.1 Signification

Dans le modèle nul, l’intercept représente la confiance générale attendue pour :

- un participant d’effet aléatoire égal à zéro ;
- un item d’effet aléatoire égal à zéro.

Autrement dit, il représente le centre général de la population modélisée.

---

## 22.2 Erreur-type

\[
SE=1{,}245
\]

L’erreur-type mesure l’incertitude de l’estimation de l’intercept.

Elle ne mesure pas la dispersion des confiances individuelles.

Il faut distinguer :

```text
Écart-type des observations
≠
Erreur-type de la moyenne estimée
```

L’écart-type des confiances brutes était environ 22,3 points.

L’erreur-type de l’intercept était seulement 1,245 point, car la moyenne générale est estimée à partir de nombreuses observations et de nombreux groupes.

---

## 22.3 Statistique z

\[
z
=
\frac{\hat{\beta}_0}{SE}
\]

Ici :

\[
z
=
\frac{75{,}736}{1{,}245}
\approx60{,}844
\]

Cette statistique compare l’estimation à zéro en unités d’erreur-type.

---

## 22.4 Valeur p

Le tableau donne :

```text
P>|z| = 0.000
```

Cela signifie en réalité une valeur extrêmement petite, arrondie à trois décimales.

Le test demande :

> L’intercept pourrait-il être égal à zéro ?

Cette question est peu intéressante scientifiquement, car nous savions déjà que la confiance moyenne n’était pas proche de zéro.

Le résultat statistiquement significatif de l’intercept ne constitue donc pas une découverte importante.

---

## 22.5 Intervalle de confiance

L’intervalle à 95 % était :

\[
[73{,}297\,;\,78{,}176]
\]

Une interprétation fréquentiste prudente est :

> Si nous répétions l’expérience et la procédure d’estimation un grand nombre de fois, environ 95 % des intervalles construits de cette manière contiendraient la vraie valeur du paramètre.

Dans une lecture pratique, les données sont compatibles avec une confiance moyenne générale située approximativement entre 73,3 et 78,2.

---

## 22.6 Pourquoi l’intercept diffère-t-il légèrement de la moyenne brute ?

La moyenne brute était :

\[
75{,}737478
\]

L’intercept REML était :

\[
75{,}736385
\]

La différence est minuscule.

Dans un plan presque parfaitement équilibré, l’intercept du modèle nul est très proche de la moyenne brute.

Dans un plan fortement déséquilibré, l’intercept ajusté pourrait différer davantage.

---

# 23. Interprétation de la variance participant

Le modèle REML donne :

\[
\sigma^2_{\text{participant}}
=
200{,}930
\]

L’écart-type correspondant est :

\[
\sigma_{\text{participant}}
=
\sqrt{200{,}930}
\approx14{,}175
\]

---

## 23.1 Signification de l’écart-type participant

Les niveaux de confiance propres aux participants se dispersent typiquement d’environ 14,2 points autour de la moyenne générale, sur l’échelle des effets aléatoires.

Cela ne signifie pas que chaque participant est exactement à ±14,2 points.

La distribution supposée est continue.

---

## 23.2 Ordre de grandeur

Si les effets participants suivent approximativement une loi normale, environ 68 % des participants devraient avoir un effet compris entre :

\[
-14{,}175
\quad\text{et}\quad
+14{,}175
\]

Autour de l’intercept :

\[
75{,}736-14{,}175
\approx61{,}56
\]

\[
75{,}736+14{,}175
\approx89{,}91
\]

Cela donne une intuition de l’ampleur des différences individuelles.

Environ 95 % des effets seraient théoriquement dans un intervalle d’environ :

\[
\pm1{,}96\times14{,}175
\approx\pm27{,}78
\]

soit des niveaux de base approximatifs entre :

\[
47{,}96
\quad\text{et}\quad
103{,}52
\]

La borne supérieure théorique dépasse 100, ce qui rappelle que le modèle linéaire ne respecte pas parfaitement les limites de l’échelle.

---

## 23.3 Conclusion substantielle

La variabilité entre participants était importante.

Cela signifie que la confiance dépend fortement de la personne qui utilise l’échelle.

Certains participants sont généralement beaucoup plus confiants que d’autres.

---

## 23.4 Ce que cette variance ne dit pas

Elle ne nous dit pas pourquoi les participants diffèrent.

Les causes possibles pourraient être :

- style d’utilisation de l’échelle ;
- compétence ;
- personnalité ;
- compréhension de la consigne ;
- motivation ;
- condition expérimentale ;
- nombre de modèles mentaux ;
- variables non mesurées.

Le modèle nul constate la variation sans encore l’expliquer.

---

# 24. Interprétation de la variance item

Le modèle donne :

\[
\sigma^2_{\text{item}}
=
11{,}875
\]

L’écart-type correspondant est :

\[
\sigma_{\text{item}}
=
\sqrt{11{,}875}
\approx3{,}446
\]

---

## 24.1 Signification

Les items diffèrent dans le niveau moyen de confiance qu’ils suscitent.

L’ampleur typique de cette différence est d’environ 3,45 points autour de la moyenne générale, après prise en compte de la structure du modèle nul.

---

## 24.2 Comparaison avec les participants

```text
Écart-type participant : 14,17
Écart-type item : 3,45
```

La variation entre participants est donc beaucoup plus importante que la variation moyenne entre items.

En variance :

\[
\frac{200{,}930}{11{,}875}
\approx16{,}9
\]

La variance participant est environ 17 fois plus grande que la variance item.

---

## 24.3 L’effet item est-il inutile parce qu’il est faible ?

Non.

Une variance de 2,38 % du total est modeste, mais non nulle.

De plus, les prédicteurs d’item comme l’entropie sont répétés sur de nombreuses observations.

Ignorer l’effet item pourrait conduire à surestimer la quantité d’information indépendante disponible pour ces prédicteurs.

L’effet aléatoire item reste donc méthodologiquement important.

---

# 25. Interprétation de la variance résiduelle

La variance résiduelle était :

\[
\sigma^2_{\text{résiduelle}}
=
285{,}412
\]

Son écart-type était :

\[
\sigma_{\text{résiduelle}}
=
\sqrt{285{,}412}
\approx16{,}894
\]

---

## 25.1 Signification

Après avoir tenu compte :

- du niveau général ;
- du participant ;
- de l’item ;

les observations individuelles restent dispersées avec un écart-type d’environ 16,9 points.

Cette variation est importante.

---

## 25.2 Que contient-elle ?

La variance résiduelle peut contenir :

- l’effet de la condition, pas encore inclus ;
- l’effet de la séquence ;
- l’entropie ;
- les modèles mentaux ;
- la validité ;
- la correction ou l’erreur de réponse ;
- la fatigue momentanée ;
- le bruit de mesure ;
- les non-linéarités ;
- d’autres facteurs inconnus.

Au stade du modèle nul, tous les prédicteurs non inclus se retrouvent indirectement dans la variation résiduelle ou dans les composantes participant/item.

---

## 25.3 Pourquoi la variance résiduelle est-elle plus grande que la variance participant ?

La confiance varie fortement d’un essai à l’autre, même à l’intérieur d’une personne.

Un participant peut donner :

```text
100 sur un essai
55 sur un autre
80 sur un troisième
```

Le style général participant explique une partie de la confiance, mais pas l’ensemble des fluctuations.

---

# 26. Proportions de variance et ICC

Le fichier :

```text
null_model_variance_components.csv
```

contenait :

| Composante | Variance | Écart-type | Proportion |
|---|---:|---:|---:|
| Participant | 200,930 | 14,175 | 0,4033 |
| Item | 11,875 | 3,446 | 0,0238 |
| Résiduelle | 285,412 | 16,894 | 0,5729 |
| Total | 498,217 | 22,321 | 1,0000 |

---

## 26.1 Proportion participant

\[
\frac{200{,}930}{498{,}217}
=
0{,}4033
\]

soit :

\[
40{,}33\%
\]

Environ 40 % de la variation totale du modèle nul est associée aux différences entre participants.

---

## 26.2 Proportion item

\[
\frac{11{,}875}{498{,}217}
=
0{,}0238
\]

soit :

\[
2{,}38\%
\]

Environ 2,4 % de la variation est associée aux différences entre items.

---

## 26.3 Proportion résiduelle

\[
\frac{285{,}412}{498{,}217}
=
0{,}5729
\]

soit :

\[
57{,}29\%
\]

Plus de la moitié de la variation reste au niveau des essais.

---

## 26.4 Variance structurée

La part attribuée aux regroupements participant ou item est :

\[
40{,}33\%+2{,}38\%
=
42{,}71\%
\]

Cette quantité montre qu’une part considérable de la variance est structurée par les regroupements.

Une régression ordinaire qui ignorerait ces regroupements serait donc inadaptée.

---

## 26.5 Qu’est-ce qu’un ICC ?

ICC signifie :

```text
Intraclass Correlation Coefficient
```

en français :

```text
coefficient de corrélation intraclasse
```

Il mesure la ressemblance entre des observations appartenant au même groupe.

---

## 26.6 ICC participant

La formule utilisée était :

\[
ICC_{\text{participant}}
=
\frac{\sigma^2_{\text{participant}}}
{\sigma^2_{\text{participant}}
+\sigma^2_{\text{item}}
+\sigma^2_{\text{résiduelle}}}
\]

Donc :

\[
ICC_{\text{participant}}
=
\frac{200{,}930}{498{,}217}
\approx0{,}4033
\]

---

## 26.7 Interprétation de l’ICC participant

Deux observations prises chez le même participant ont une ressemblance importante liée au style général de cette personne.

Un ICC de 0,403 ne signifie pas que leurs confiances sont toujours identiques.

Il signifie qu’environ 40 % de la variance totale est attribuable à des différences stables entre participants.

---

## 26.8 ICC item

\[
ICC_{\text{item}}
=
\frac{11{,}875}{498{,}217}
\approx0{,}0238
\]

Deux observations portant sur le même item partagent une ressemblance plus faible liée à cet item.

---

## 26.9 Analogie familiale

L’ICC peut être comparé à une ressemblance familiale.

Si les membres d’une même famille se ressemblent fortement sur une mesure, l’appartenance familiale explique une part importante de la variation.

Dans notre projet :

- le participant joue le rôle du groupe principal ;
- les réponses d’une même personne se ressemblent fortement ;
- les réponses au même item se ressemblent aussi, mais beaucoup moins.

---

## 26.10 Attention au terme « corrélation »

L’ICC participant ne correspond pas à une corrélation calculée entre deux colonnes ordinaires.

Il est dérivé des composantes de variance du modèle.

Il représente une corrélation théorique attendue entre deux observations partageant le même participant, sous les hypothèses du modèle.

---

# 27. Les prédictions du modèle nul

Le script a produit :

```text
null_model_predictions.csv
```

Une prédiction du modèle nul peut être construite de plusieurs manières.

---

## 27.1 Prédiction marginale

La prédiction marginale utilise seulement l’intercept fixe :

\[
\hat{Y}^{\text{marginal}}_{ij}
=
\hat{\beta}_0
\]

Toutes les observations reçoivent alors environ :

\[
75{,}736
\]

Cette prédiction représente la tendance générale de population.

---

## 27.2 Prédiction conditionnelle

La prédiction conditionnelle ajoute les effets aléatoires estimés :

\[
\hat{Y}^{\text{conditionnelle}}_{ij}
=
\hat{\beta}_0
+
\hat{u}_i
+
\hat{v}_j
\]

Elle tient donc compte :

- du participant ;
- de l’item.

---

## 27.3 Exemple

Supposons :

```text
intercept = 75,736
effet participant = +10
effet item = −2
```

La prédiction conditionnelle est :

\[
75{,}736+10-2
=
83{,}736
\]

---

## 27.4 Pourquoi les prédictions conditionnelles sont-elles meilleures en échantillon ?

Elles utilisent l’identité du participant et de l’item observés.

Elles connaissent donc déjà leur tendance générale estimée.

Elles ne constituent pas nécessairement une prédiction pour un nouveau participant inconnu.

Pour un nouveau participant sans historique, son effet aléatoire n’est pas disponible et serait généralement pris comme zéro.

---

# 28. RMSE et MAE

Le script a calculé :

```text
RMSE conditionnel descriptif : 16,779
MAE conditionnelle descriptive : 11,441
```

---

## 28.1 RMSE

RMSE signifie :

```text
Root Mean Squared Error
```

ou :

```text
racine de l’erreur quadratique moyenne
```

La formule est :

\[
RMSE
=
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
}
\]

---

## 28.2 Intuition du RMSE

Le RMSE mesure la taille typique des erreurs de prédiction, en pénalisant fortement les grandes erreurs.

Un RMSE de 16,78 signifie que les prédictions conditionnelles s’écartent typiquement des observations d’un ordre de grandeur d’environ 17 points.

---

## 28.3 MAE

MAE signifie :

```text
Mean Absolute Error
```

ou :

```text
erreur absolue moyenne
```

La formule est :

\[
MAE
=
\frac{1}{n}
\sum_{i=1}^{n}
|y_i-\hat{y}_i|
\]

---

## 28.4 Intuition du MAE

Le MAE de 11,44 signifie que l’écart absolu moyen entre la confiance observée et la confiance prédite était d’environ 11,4 points.

---

## 28.5 Pourquoi le RMSE est-il supérieur au MAE ?

Le RMSE élève les erreurs au carré.

Il est donc davantage influencé par les grandes erreurs.

Si certaines observations sont très éloignées des prédictions, le RMSE augmente plus fortement que le MAE.

---

## 28.6 Pourquoi qualifier ces scores de descriptifs ?

Ils ont été calculés sur les mêmes données que celles utilisées pour ajuster le modèle.

Il ne s’agit pas d’une évaluation hors échantillon.

Le modèle a déjà vu les participants et les items.

Ces scores décrivent l’ajustement interne, mais ne mesurent pas directement la capacité à généraliser à de nouvelles données.

---

# 29. Les effets aléatoires estimés

Le script a produit :

```text
null_model_subject_effects.csv
null_model_item_effects.csv
```

---

## 29.1 Que contient le fichier participant ?

Une ligne par participant avec son effet aléatoire estimé. Celui-ci est calculé grâce à l'espérance conditionnelle :

\[
\boxed{
E(u_g \mid \bar Y_g)
=
\frac{\tau^2}{\tau^2 + \sigma^2 / n_g}
(\bar Y_g - \mu)
}
\]

Exemple conceptuel :

| participant | effet |
|---|---:|
| A | +14 |
| B | −9 |
| C | +2 |

Un effet positif signifie que le participant utilise en moyenne une confiance plus élevée que le niveau général.

Un effet négatif signifie une confiance générale plus faible.

---

## 29.2 Que contient le fichier item ?

Une ligne par item avec son effet aléatoire estimé.

Exemple :

| item | effet |
|---|---:|
| 1 | +3,5 |
| 2 | −2,1 |
| 3 | +0,4 |

---

## 29.3 Les effets sont-ils simplement des moyennes brutes ?

Non.

Il s’agit d’estimations régularisées par le modèle.

Elles sont souvent appelées :

```text
BLUP
```

pour :

```text
Best Linear Unbiased Predictors
```

ou prédictions empiriques bayésiennes dans certains contextes.

---

## 29.4 Le rétrécissement

Le **rétrécissement**, ou *shrinkage*, signifie que les effets aléatoires sont rapprochés de zéro lorsque l’information disponible est limitée ou bruitée.

> **Analogie**
>
> Si un joueur marque trois buts dans un seul match, nous ne concluons pas immédiatement qu’il marquera trois buts à chaque match.
>
> Nous rapprochons notre estimation de la moyenne générale, car l’échantillon est petit.
>
> Le modèle mixte applique une logique comparable aux effets de groupes.

Dans notre plan, les participants ont tous 64 essais et les items environ 70 réponses, donc l’information par groupe est relativement substantielle.

---

## 29.5 Pourquoi ne pas interpréter chaque effet comme un test individuel ?

Les effets aléatoires servent principalement à représenter la distribution des groupes et à améliorer l’estimation globale.

Le projet ne cherchait pas à tester séparément :

```text
le participant 63873 est-il significativement plus confiant ?
```

La lecture principale porte sur :

- la variance participant ;
- la variance item ;
- la structure globale.

---

# 30. Les diagnostics graphiques

Le script a produit plusieurs graphiques.

---

## 30.1 Résidus contre valeurs ajustées

Fichier :

```text
null_model_residuals_vs_fitted.png
```

### Axes

```text
axe horizontal : valeur prédite
axe vertical   : résidu
```

### Ce que l’on aimerait voir

Idéalement :

- un nuage centré autour de zéro ;
- aucune courbe systématique ;
- une dispersion relativement constante.

### Problèmes possibles

#### Courbure

Une courbure suggère qu’une relation non linéaire manque au modèle.

#### Forme en entonnoir

Une dispersion qui augmente avec les prédictions suggère de l’hétéroscédasticité.

#### Bande liée au plafond

Comme la confiance ne peut pas dépasser 100, les résidus positifs sont limités lorsque la prédiction est élevée.

Cela peut produire une structure asymétrique.

---

## 30.2 Distribution des résidus

Fichier :

```text
null_model_residual_distribution.png
```

Tu avais observé :

> La distribution est presque symétrique autour de zéro.

C’est un point positif, mais une symétrie visuelle approximative ne garantit pas une normalité parfaite.

Les analyses finales ont montré une asymétrie négative et une kurtosis élevée, surtout à cause du plafond.

---

## 30.3 QQ-plot

Fichier :

```text
null_model_qqplot.png
```

QQ signifie :

```text
Quantile–Quantile
```

Le graphique compare :

- les quantiles observés des résidus ;
- les quantiles attendus sous une loi normale.

---

## 30.4 Qu’est-ce qu’un quantile ?

Un quantile est un seuil divisant une distribution.

Exemples :

- médiane : quantile 50 % ;
- premier quartile : quantile 25 % ;
- troisième quartile : quantile 75 %.

Le QQ-plot compare les positions relatives des valeurs, des plus faibles aux plus fortes.

---

## 30.5 Lecture du QQ-plot

Si les résidus sont approximativement normaux, les points suivent une ligne droite.

Tu avais observé :

> Les points s’écartent aux extrémités.

Cela signifie que les queues de la distribution ne correspondent pas parfaitement à une loi normale.

Une **queue** est une partie extrême de la distribution.

Ces écarts peuvent refléter :

- des observations très éloignées ;
- une asymétrie ;
- une forte concentration au plafond ;
- des queues plus épaisses que la normale.

---

## 30.6 Pourquoi les extrémités sont-elles importantes ?

Les intervalles de confiance et les tests du modèle linéaire reposent en partie sur l’hypothèse normale.

Des écarts modérés sont souvent tolérables avec un grand échantillon, mais des écarts forts motivent des analyses de sensibilité.

C’est précisément ce que nous avons réalisé plus tard :

- exclusion des valeurs 100 ;
- modèle logistique de la borne supérieure.

---

# 31. Les fichiers CSV produits

# 31.1 `null_model_variance_components.csv`

Contenu :

```text
component
variance
standard_deviation
proportion_total_variance
```

### Utilité

Il présente la décomposition centrale du modèle nul.

### Lecture

Une ligne :

```text
Participant,200.930,14.175,0.4033
```

signifie :

- variance participant : 200,930 ;
- écart-type participant : 14,175 ;
- part de la variance totale : 40,33 %.

---

## 31.2 `null_model_fit_statistics.csv`

Ce fichier contient :

- méthode ML et REML ;
- log-vraisemblance ;
- AIC ;
- BIC ;
- nombre de paramètres ;
- nombre d’observations ;
- convergence ;
- variance résiduelle.

### Pourquoi le générer ?

Les statistiques d’ajustement servent à comparer les modèles.

Le modèle nul ML doit être conservé pour la future comparaison avec le modèle de contrôle ML.

---

## 31.3 `null_model_fixed_effects.csv`

Contenu :

```text
parameter
estimate
standard_error
ci_95_lower
ci_95_upper
```

Dans le modèle nul, il n’existe qu’un seul effet fixe :

```text
Intercept
```

Ce fichier est donc très court.

---

## 31.4 `null_model_predictions.csv`

Ce fichier contient les observations et leurs prédictions.

Il peut notamment inclure :

```text
subject_id
item_id
confidence
predicted_confidence
residual
```

### Utilité

Il permet :

- d’examiner les erreurs individuelles ;
- de construire les graphiques ;
- de repérer les observations atypiques ;
- de recalculer RMSE et MAE.

---

## 31.5 `null_model_subject_effects.csv`

Une ligne par participant.

### Utilité

- identifier les participants généralement très confiants ;
- examiner la distribution des effets ;
- construire le graphique des effets participants.

---

## 31.6 `null_model_item_effects.csv`

Une ligne par item.

### Utilité

- identifier les items inspirant plus ou moins de confiance ;
- examiner leur distribution ;
- vérifier l’ampleur relativement faible de la variation item.

---

# 32. Les fichiers texte et JSON

## 32.1 `null_model_REML_summary.txt`

Ce fichier contient le résumé complet du modèle REML.

Il constitue le résultat principal du modèle nul.

---

## 32.2 `null_model_ML_summary.txt`

Ce fichier contient le résumé ML.

Il servira à la comparaison avec le modèle de contrôle.

---

## 32.3 Pourquoi enregistrer les résumés texte ?

Le tableau affiché dans le terminal disparaît une fois la session fermée.

Le fichier texte permet :

- de conserver le résultat exact ;
- de vérifier plus tard la convergence ;
- de documenter la méthode ;
- de citer les coefficients et variances ;
- de comparer les versions.

---

## 32.4 `null_model_results.json`

Ce fichier stocke les principaux résultats sous forme structurée.

Il peut contient: :

```json
{
  "input_file": "/home/paul/Etudes/Annee_4/TSP/INRIA/cogsci-individualization/paul-benchmark/computational_model/dataset_analysis_E1.csv",
  "n_observations": 9024,
  "n_subjects": 141,
  "n_items": 128,
  "estimation_primary": "REML",
  "converged": true,
  "intercept": 75.73638539153357,
  "subject_variance": 200.93019639453854,
  "item_variance": 11.874523130868583,
  "residual_variance": 285.41232923486825,
  "total_variance": 498.21704876027536,
  "subject_icc": 0.40329851596712246,
  "item_icc": 0.023834036110197805,
  "residual_proportion": 0.5728674479226797,
  "total_cluster_icc": 0.4271325520773203,
  "conditional_rmse_descriptive": 16.77927012905288,
  "conditional_mae_descriptive": 11.44102053422769,
  "reml_log_likelihood": -38669.75812326601,
  "ml_log_likelihood": -38670.89438865008,
  "ml_aic": 77349.78877730016,
  "ml_bic": 77378.2193491951
}
```

### Différence avec le résumé texte

Le résumé texte est destiné principalement à la lecture humaine.

Le JSON est pratique pour un autre script.

---

# 33. Les graphiques produits

## 33.1 `null_model_variance_decomposition.png`

Ce graphique représente les parts :

```text
Participant : 40,3 %
Item        : 2,4 %
Résiduelle  : 57,3 %
```

Il peut prendre la forme d’un diagramme en barres ou d’un diagramme circulaire.

### Message principal

La variation participant est importante, tandis que la variation item est beaucoup plus petite.

---

## 33.2 `null_model_subject_effects.png`

Ce graphique représente les effets participants triés.

---

### Pourquoi une courbe triée ressemble-t-elle à une droite ?

Si les effets sont triés du plus petit au plus grand, le graphique est forcément monotone.

Il ne représente pas une relation entre deux variables cognitives.

L’axe horizontal correspond généralement au rang du participant après tri.

L’axe vertical correspond à son effet estimé.

Une forme approximativement rectiligne au centre est fréquente pour une distribution proche de la normale.

---

### Que signifie le passage autour de zéro ?

Les effets aléatoires sont centrés autour de zéro.

Un effet de zéro correspond à un participant proche du niveau général :

\[
75{,}736+0
=
75{,}736
\]

Un effet de +20 correspond à un niveau général proche de :

\[
95{,}736
\]

Un effet de −20 correspond à :

\[
55{,}736
\]

---

## 33.3 `null_model_item_effects.png`

Tu avais observé une forme proche d’un début de fonction cubique allant environ de −10 à +10.

Là encore, cette forme provient principalement du tri des effets.

Elle ne signifie pas que les items suivent réellement une fonction cubique.

La distribution des effets peut produire une forme en S lorsqu’ils sont triés :

- extrémité négative ;
- zone centrale dense ;
- extrémité positive.

---

## 33.4 Pourquoi les effets items semblent-ils plus resserrés ?

Parce que leur écart-type est seulement :

\[
3{,}446
\]

contre :

\[
14{,}175
\]

pour les participants.

Le graphique confirme visuellement la différence d’échelle.

---

## 33.5 `null_model_residual_distribution.png`

Ce graphique montre la distribution des erreurs après prise en compte des participants et items.

Il sert à vérifier :

- le centrage autour de zéro ;
- la symétrie ;
- les queues ;
- les éventuelles valeurs extrêmes.

---

## 33.6 `null_model_qqplot.png`

Il examine plus finement la normalité.

Les écarts aux extrémités ont annoncé les diagnostics finaux :

- asymétrie négative ;
- kurtosis élevée ;
- influence du plafond.

---

# 34. Résultats numériques complets

## 34.1 Résultats REML

| Paramètre | Estimation |
|---|---:|
| Intercept | 75,736 |
| Erreur-type de l’intercept | 1,245 |
| IC à 95 % | [73,297 ; 78,176] |
| Variance participant | 200,930 |
| Écart-type participant | 14,175 |
| Variance item | 11,875 |
| Écart-type item | 3,446 |
| Variance résiduelle | 285,412 |
| Écart-type résiduel | 16,894 |
| Log-vraisemblance REML | −38 669,758 |
| Convergence | Oui |

---

## 34.2 Résultats ML

| Paramètre | Estimation |
|---|---:|
| Intercept | 75,736 |
| Erreur-type de l’intercept | 1,241 |
| IC à 95 % | [73,304 ; 78,168] |
| Variance participant | 199,565 |
| Variance item | 11,865 |
| Variance résiduelle | 285,413 |
| Log-vraisemblance ML | −38 670,894 |
| Convergence | Oui |

---

## 34.3 Décomposition REML

| Source | Variance | Pourcentage |
|---|---:|---:|
| Participant | 200,930 | 40,33 % |
| Item | 11,875 | 2,38 % |
| Résiduelle | 285,412 | 57,29 % |
| Total | 498,217 | 100 % |

---

## 34.4 ICC

\[
ICC_{\text{participant}}
=
0{,}4033
\]

\[
ICC_{\text{item}}
=
0{,}0238
\]

---

## 34.5 Ajustement descriptif conditionnel

\[
RMSE=16{,}779
\]

\[
MAE=11{,}441
\]

---

# 35. Ce que le modèle nul permet de conclure

## 35.1 La confiance moyenne est élevée

La confiance moyenne générale est proche de :

\[
75{,}7/100
\]

Cela décrit une tendance globale vers des niveaux de confiance relativement élevés.

---

## 35.2 Les différences entre participants sont importantes

Environ 40,3 % de la variance totale est associée aux participants.

Cela indique que la façon d’utiliser l’échelle est fortement individuelle.

---

## 35.3 Les différences entre items existent mais sont plus faibles

Environ 2,4 % de la variance totale est associée aux items.

Les items ne sont pas interchangeables, mais leur variation moyenne est bien moins importante que celle des participants.

---

## 35.4 Une grande part reste au niveau des essais

Environ 57,3 % de la variance reste résiduelle.

La confiance ne peut donc pas être résumée uniquement comme un trait stable de participant ou une propriété stable de l’item.

Elle varie aussi fortement d’un essai à l’autre.

---

## 35.5 Le modèle mixte est justifié

L’ICC participant de 0,403 est beaucoup trop important pour ignorer les mesures répétées.

Traiter les 9 024 essais comme indépendants aurait été problématique.

---

## 35.6 Les items doivent également rester dans le modèle

Même si leur part de variance est faible, les items constituent une unité d’échantillonnage répétée.

L’effet aléatoire item protège notamment l’estimation des prédicteurs d’item comme l’entropie.

---

# 36. Ce que le modèle nul ne permet pas de conclure

Le modèle nul ne dit pas :

- pourquoi les participants diffèrent ;
- pourquoi les items diffèrent ;
- si Standard augmente la confiance ;
- si la confiance diminue avec la séquence ;
- si l’entropie réduit la confiance ;
- si le nombre de modèles mentaux a un effet ;
- si la validité influence la confiance ;
- si la confiance correspond à l’exactitude.

Il fournit une structure de départ, pas une explication cognitive complète.

---

## 36.1 Il ne teste pas l’effet de la condition

La condition n’est pas incluse dans :

```text
confidence ~ 1
```

Une différence Standard–Neutral pourrait contribuer à la variance participant, puisque la condition varie entre participants.

---

## 36.2 Il ne teste pas l’effet de la séquence

La séquence n’est pas incluse.

Une baisse progressive de confiance pourrait contribuer à la variance résiduelle.

---

## 36.3 Il ne teste pas l’entropie

Les différences liées à l’entropie pourraient apparaître dans :

- la variance item ;
- la variance résiduelle.

Mais le modèle nul ne les isole pas.

---

# 37. Limites de cette première modélisation

## 37.1 Variable bornée

La confiance est limitée à :

\[
0\leq\text{confidence}\leq100
\]

Le modèle linéaire, lui, peut théoriquement produire des valeurs inférieures à 0 ou supérieures à 100.

---

## 37.2 Effet plafond

Environ 25,9 % des réponses étaient exactement égales à 100.

Cette concentration viole partiellement l’hypothèse de normalité.

---

## 37.3 Structure aléatoire limitée aux intercepts

Le modèle suppose que les participants diffèrent par leur niveau général, mais pas encore par les pentes des prédicteurs.

Par exemple, il ne permet pas à l’effet de la séquence de varier selon le participant.

---

## 37.4 Absence de prédicteurs

La variance résiduelle contient de nombreux mécanismes potentiellement explicables.

Le modèle nul est volontairement incomplet.

---

## 37.5 RMSE et MAE en échantillon

Les métriques de prédiction sont descriptives.

Elles ne mesurent pas la généralisation à de nouveaux participants ou de nouveaux items.

---

## 37.6 Hypothèse normale des effets

Les effets participants et items sont supposés normaux.

Les graphiques triés donnaient une première impression, mais une validation complète nécessitait les diagnostics ultérieurs.

---

# 38. Pourquoi cette étape conduit au modèle de contrôle

Le modèle nul a montré qu’il existe une forte structure participant–item.

Nous devions maintenant commencer à expliquer une partie de cette variation.

La prochaine question était :

> Avant d’ajouter les prédicteurs cognitifs, la confiance varie-t-elle selon la condition expérimentale et au fil des essais ?

Ces variables ont été traitées comme des contrôles.

Le prochain modèle sera donc :

\[
\text{confidence}
=
\beta_0
+
\beta_1\text{condition}
+
\beta_2\text{sequence}
+
u_{\text{participant}}
+
v_{\text{item}}
+
\varepsilon
\]

Il sera ajusté dans :

```text
fit_control_mixed_model_E1.py
```

Nous comparerons ensuite ce modèle au modèle nul avec :

- la log-vraisemblance ML ;
- le test du rapport de vraisemblance ;
- l’AIC ;
- le BIC ;
- les changements de composantes de variance ;
- les coefficients de condition et de séquence.

---

# 39. Bilan pédagogique

Le script `fit_null_crossed_mixed_model_E1.py` a établi la base statistique de toute l’analyse.

Il a permis de :

1. charger les 9 024 essais complets ;
2. conserver les 141 participants et 128 items ;
3. représenter la structure croisée participant–item ;
4. ajuster un modèle contenant uniquement une moyenne générale et des intercepts aléatoires ;
5. utiliser un groupe global artificiel pour implémenter les effets croisés dans `statsmodels` ;
6. ajuster le modèle en REML et en ML ;
7. vérifier la convergence avec l’optimiseur L-BFGS ;
8. estimer une confiance générale d’environ 75,7 ;
9. estimer une variance participant d’environ 200,9 ;
10. estimer une variance item d’environ 11,9 ;
11. estimer une variance résiduelle d’environ 285,4 ;
12. attribuer 40,3 % de la variance aux participants ;
13. attribuer 2,4 % de la variance aux items ;
14. laisser 57,3 % de la variance au niveau résiduel ;
15. calculer un ICC participant de 0,403 ;
16. calculer un ICC item de 0,024 ;
17. montrer que les réponses d’une même personne sont fortement dépendantes ;
18. confirmer que l’effet aléatoire participant est indispensable ;
19. confirmer qu’un effet aléatoire item reste méthodologiquement pertinent ;
20. produire les résumés, effets aléatoires, prédictions et diagnostics graphiques nécessaires.

La conclusion centrale est :

> La confiance est fortement structurée par des différences stables entre participants, tandis que les différences moyennes entre items sont plus modestes. Une grande part de la variation reste néanmoins propre aux essais et doit être étudiée avec des prédicteurs supplémentaires.


# Étape 4 — Le modèle de contrôle avec `fit_control_mixed_model_E1.py`

## Sommaire

1. [Rôle de cette étape](#1-rôle-de-cette-étape)
2. [Pourquoi ne pas ajouter immédiatement les variables cognitives ?](#2-pourquoi-ne-pas-ajouter-immédiatement-les-variables-cognitives)
3. [Qu’est-ce qu’une variable de contrôle ?](#3-quest-ce-quune-variable-de-contrôle)
4. [Les deux variables de contrôle retenues](#4-les-deux-variables-de-contrôle-retenues)
5. [Pourquoi ajouter la condition expérimentale ?](#5-pourquoi-ajouter-la-condition-expérimentale)
6. [Pourquoi ajouter la position de l’essai ?](#6-pourquoi-ajouter-la-position-de-lessai)
7. [Pourquoi centrer la séquence ?](#7-pourquoi-centrer-la-séquence)
8. [Pourquoi diviser la séquence centrée par dix ?](#8-pourquoi-diviser-la-séquence-centrée-par-dix)
9. [Formulation mathématique du modèle de contrôle](#9-formulation-mathématique-du-modèle-de-contrôle)
10. [Différence entre modèle nul et modèle de contrôle](#10-différence-entre-modèle-nul-et-modèle-de-contrôle)
11. [Organisation générale du script](#11-organisation-générale-du-script)
12. [Les bibliothèques utilisées](#12-les-bibliothèques-utilisées)
13. [Configuration des chemins et des formules](#13-configuration-des-chemins-et-des-formules)
14. [Chargement et préparation des données](#14-chargement-et-préparation-des-données)
15. [Traitement de la condition](#15-traitement-de-la-condition)
16. [Construction de `sequence_c10`](#16-construction-de-sequence_c10)
17. [Construction de la structure aléatoire croisée](#17-construction-de-la-structure-aléatoire-croisée)
18. [Ajustement du modèle nul en ML](#18-ajustement-du-modèle-nul-en-ml)
19. [Ajustement du modèle de contrôle en ML](#19-ajustement-du-modèle-de-contrôle-en-ml)
20. [Ajustement du modèle de contrôle en REML](#20-ajustement-du-modèle-de-contrôle-en-reml)
21. [Pourquoi ajuster trois modèles dans le même script ?](#21-pourquoi-ajuster-trois-modèles-dans-le-même-script)
22. [Le test du rapport de vraisemblance](#22-le-test-du-rapport-de-vraisemblance)
23. [Notion de modèles emboîtés](#23-notion-de-modèles-emboîtés)
24. [Calcul du test du rapport de vraisemblance](#24-calcul-du-test-du-rapport-de-vraisemblance)
25. [Les degrés de liberté du test](#25-les-degrés-de-liberté-du-test)
26. [Interprétation de la valeur p](#26-interprétation-de-la-valeur-p)
27. [L’AIC](#27-laic)
28. [Le BIC](#28-le-bic)
29. [Pourquoi utiliser plusieurs critères de comparaison ?](#29-pourquoi-utiliser-plusieurs-critères-de-comparaison)
30. [Extraction des effets fixes](#30-extraction-des-effets-fixes)
31. [Interprétation de l’intercept](#31-interprétation-de-lintercept)
32. [Interprétation de l’effet de la condition](#32-interprétation-de-leffet-de-la-condition)
33. [Interprétation de l’effet de la séquence](#33-interprétation-de-leffet-de-la-séquence)
34. [Prédictions ajustées selon la condition et la séquence](#34-prédictions-ajustées-selon-la-condition-et-la-séquence)
35. [Composantes de variance du modèle de contrôle](#35-composantes-de-variance-du-modèle-de-contrôle)
36. [Comparer les variances du modèle nul et du modèle de contrôle](#36-comparer-les-variances-du-modèle-nul-et-du-modèle-de-contrôle)
37. [Le coefficient de détermination \(R^2\)](#37-le-coefficient-de-détermination-r2)
38. [Le \(R^2\) marginal](#38-le-r2-marginal)
39. [Le \(R^2\) conditionnel](#39-le-r2-conditionnel)
40. [Les fichiers générés](#40-les-fichiers-générés)
41. [Résultats complets du modèle](#41-résultats-complets-du-modèle)
42. [Ce que les résultats permettent de conclure](#42-ce-que-les-résultats-permettent-de-conclure)
43. [Ce que les résultats ne permettent pas de conclure](#43-ce-que-les-résultats-ne-permettent-pas-de-conclure)
44. [Limites du modèle de contrôle](#44-limites-du-modèle-de-contrôle)
45. [Lien avec le modèle nul](#45-lien-avec-le-modèle-nul)
46. [Pourquoi cette étape conduit au modèle cognitif](#46-pourquoi-cette-étape-conduit-au-modèle-cognitif)
47. [Bilan pédagogique](#47-bilan-pédagogique)

---

# 1. Rôle de cette étape

Le modèle nul avait montré que la confiance variait fortement :

- entre les participants ;
- dans une moindre mesure entre les items ;
- et surtout entre les essais eux-mêmes.

Cependant, le modèle nul ne contenait aucun prédicteur explicatif. Il se limitait à :

\[
\text{confiance}
=
\text{moyenne générale}
+
\text{effet participant}
+
\text{effet item}
+
\text{résidu}
\]

Nous devions maintenant commencer à expliquer une partie de cette variation.

Nous avons choisi de commencer avec deux caractéristiques fondamentales du plan expérimental :

1. la condition expérimentale ;
2. la position de l’essai dans l’expérience.

Le script correspondant est :

```text
fit_control_mixed_model_E1.py
```

Le modèle obtenu est appelé **modèle de contrôle**.

---

## 1.1 Question scientifique

Le modèle répond à deux premières questions :

> La confiance diffère-t-elle entre la condition Standard et la condition Neutral ?

et :

> La confiance évolue-t-elle progressivement au cours des 64 essais ?

Ces deux questions sont examinées avant les variables cognitives comme :

- l’entropie ;
- la précision individuelle ;
- le nombre de modèles mentaux ;
- la validité.

---

# 2. Pourquoi ne pas ajouter immédiatement les variables cognitives ?

Nous aurions pu construire immédiatement un grand modèle contenant toutes les variables :

```text
condition
sequence
subject_accuracy
item_entropy
subject_mean_models
models_within_subject
validity_binary
```

Cela aurait toutefois rendu l’analyse plus difficile à comprendre.

---

## 2.1 Distinguer plusieurs niveaux d’explication

Nous voulions séparer :

```text
effets du plan expérimental
```

et :

```text
effets des prédicteurs cognitifs
```

La condition et la séquence existaient indépendamment de nos hypothèses sur MReasoner.

Elles pouvaient influencer la confiance même si :

- l’entropie n’avait aucun effet ;
- la précision n’avait aucun effet ;
- le nombre de modèles mentaux n’avait aucun effet.

---

## 2.2 Mesurer l’apport supplémentaire des variables cognitives

La progression permet d’effectuer deux comparaisons distinctes :

```text
Modèle nul
contre
Modèle de contrôle
```

puis :

```text
Modèle de contrôle
contre
Modèle cognitif
```

La première comparaison demande :

> La condition et la séquence apportent-elles de l’information ?

La seconde demandera :

> Les prédicteurs cognitifs apportent-ils encore de l’information après contrôle de la condition et de la séquence ?

---

## 2.3 Analogie avec une enquête médicale

Supposons que l’on cherche à expliquer la tension artérielle avec un nouveau marqueur biologique.

Avant d’attribuer un effet au marqueur, on peut contrôler :

- l’âge ;
- le sexe ;
- l’heure de la mesure.

Si le marqueur reste associé à la tension après ces contrôles, son information est plus spécifique.

Dans notre projet :

```text
condition + séquence
```

jouent un rôle comparable de variables contextuelles fondamentales.

---

# 3. Qu’est-ce qu’une variable de contrôle ?

## 3.1 Définition

Une **variable de contrôle** est une variable ajoutée au modèle pour tenir compte d’une source de variation connue, même si elle ne constitue pas nécessairement l’hypothèse scientifique principale.

Elle sert à estimer les autres relations « toutes choses égales par ailleurs », dans les limites du modèle.

---

## 3.2 Exemple simple

Supposons que l’on observe une relation entre :

```text
consommation de café
et
fatigue
```

Les personnes buvant davantage de café sont aussi plus fatiguées.

On pourrait conclure à tort :

> Le café provoque la fatigue.

Mais il existe une autre explication :

```text
les personnes dormant peu
→ sont plus fatiguées
→ boivent plus de café
```

Le nombre d’heures de sommeil est une variable de contrôle importante.

---

## 3.3 Dans notre projet

Si la confiance diminue au fil de l’expérience et que certains types de tâches apparaissent plus souvent à la fin, un effet attribué au type de tâche pourrait en réalité refléter la fatigue.

Contrôler `sequence` permet de réduire ce risque.

De même, si les conditions Standard et Neutral produisent des styles de confiance différents, il faut en tenir compte avant d’interpréter les prédicteurs cognitifs.

---

## 3.4 Contrôler n’est pas éliminer physiquement

Ajouter une variable de contrôle ne retire pas des observations et ne transforme pas les participants.

Cela signifie que le modèle estime simultanément plusieurs coefficients.

Par exemple :

\[
\text{confiance}
=
\beta_0
+
\beta_1\text{condition}
+
\beta_2\text{séquence}
+\cdots
\]

Le coefficient de condition est alors estimé en tenant compte de la séquence, et inversement.

---

# 4. Les deux variables de contrôle retenues

Les prédicteurs du modèle étaient :

```text
condition
sequence_c10
```

---

## 4.1 `condition`

Variable catégorielle à deux modalités :

```text
Neutral
Standard
```

Elle varie entre participants.

Chaque participant appartient à une seule condition.

---

## 4.2 `sequence_c10`

Variable numérique construite à partir de :

```text
sequence
```

Elle représente la position de l’essai :

- centrée sur la position moyenne ;
- exprimée par tranches de dix essais.

---

# 5. Pourquoi ajouter la condition expérimentale ?

## 5.1 Standard et Neutral ne sont pas équivalentes

Dans la condition Standard, les énoncés avaient un contenu ordinaire et interprétable.

Dans la condition Neutral, un terme était remplacé par un non-mot afin de neutraliser une partie du contenu sémantique ou des croyances associées aux prémisses.

Ces conditions pouvaient modifier :

- la compréhension ;
- le sentiment de familiarité ;
- la disponibilité des connaissances ;
- la facilité subjective ;
- la confiance.

---

## 5.2 Une variable entre participants

La condition ne change pas au cours des 64 essais d’une même personne.

Nous avions :

```text
71 participants Standard
70 participants Neutral
```

Le coefficient de condition est donc estimé principalement à partir des différences entre ces deux groupes de participants.

---

## 5.3 Pourquoi l’effet aléatoire participant reste-t-il nécessaire ?

On pourrait penser :

> Puisque nous ajoutons la condition, l’effet participant devient inutile.

Ce serait incorrect.

La condition explique seulement une différence moyenne entre deux groupes.

À l’intérieur de chaque condition, les participants continuent à différer fortement.

Exemple :

```text
Participants Standard :
moyennes de confiance = 55, 70, 85, 100...

Participants Neutral :
moyennes de confiance = 45, 65, 75, 90...
```

Le modèle a besoin :

- d’un effet fixe de condition pour la différence moyenne Standard–Neutral ;
- d’un effet aléatoire participant pour les différences individuelles restantes.

---

## 5.4 Forme conceptuelle

Pour le participant \(i\) :

\[
\text{niveau de base}_i
=
\beta_0
+
\beta_{\text{Standard}}\text{Standard}_i
+
u_i
\]

où :

- \(\beta_0\) est le niveau moyen Neutral ;
- \(\beta_{\text{Standard}}\) est la différence moyenne Standard–Neutral ;
- \(u_i\) est l’écart individuel restant.

---

# 6. Pourquoi ajouter la position de l’essai ?

## 6.1 L’expérience dure 64 essais

Chaque participant réalise une succession de 64 problèmes.

Son comportement peut évoluer au cours du temps.

---

## 6.2 Mécanismes possibles

Une baisse de confiance pourrait refléter :

- la fatigue ;
- une prise de conscience progressive de la difficulté ;
- un recalibrage de l’échelle ;
- une diminution de l’attention ;
- un changement de stratégie.

Une hausse pourrait au contraire refléter :

- l’apprentissage ;
- une meilleure compréhension de la consigne ;
- l’habituation à l’échelle ;
- une stratégie plus fluide.

---

## 6.3 Effet d’ordre

Un **effet d’ordre** désigne une modification d’une réponse liée à la position d’une observation dans la séquence.

Exemple :

```text
Essais 1 à 10 : confiance moyenne élevée
Essais 55 à 64 : confiance moyenne plus faible
```

---

## 6.4 Pourquoi ne pas ignorer ce phénomène ?

Si la confiance évolue au fil du temps et que d’autres variables sont inégalement réparties dans la séquence, leurs coefficients pourraient être contaminés par l’ordre.

Même avec une randomisation des items, contrôler la séquence permet :

- d’augmenter la précision ;
- de documenter un phénomène temporel ;
- de réduire une source de variation résiduelle.

---

# 7. Pourquoi centrer la séquence ?

La variable originale allait de :

\[
1\quad\text{à}\quad64
\]

Sa moyenne est :

\[
\bar{s}
=
\frac{1+64}{2}
=
32{,}5
\]

Le centrage consiste à soustraire cette moyenne :

\[
\text{sequence\_centered}
=
\text{sequence}-32{,}5
\]

---

## 7.1 Exemples

| `sequence` | `sequence - 32.5` |
|---:|---:|
| 1 | −31,5 |
| 10 | −22,5 |
| 32,5 | 0 |
| 50 | 17,5 |
| 64 | 31,5 |

---

## 7.2 Pourquoi centrer ?

Le centrage change l’interprétation de l’intercept.

### Sans centrage

Dans :

\[
Y=\beta_0+\beta_1\text{sequence}
\]

l’intercept représente la confiance prédite lorsque :

\[
\text{sequence}=0
\]

Mais il n’existe aucun essai 0.

L’intercept correspondrait donc à une extrapolation située avant le début de l’expérience.

### Avec centrage

Dans :

\[
Y=\beta_0+\beta_1(\text{sequence}-32{,}5)
\]

l’intercept correspond à :

\[
\text{sequence}=32{,}5
\]

c’est-à-dire au milieu de l’expérience.

Cette référence est beaucoup plus facile à interpréter.

---

## 7.3 Le centrage change-t-il l’effet de la séquence ?

Non.

Supposons :

\[
Y=\alpha+\beta s
\]

et :

\[
s_c=s-\bar s
\]

Comme :

\[
s=s_c+\bar s
\]

nous obtenons :

\[
Y
=
\alpha+\beta(s_c+\bar s)
\]

\[
Y
=
(\alpha+\beta\bar s)+\beta s_c
\]

La pente \(\beta\) ne change pas.

Seul l’intercept change.

---

## 7.4 Le centrage modifie-t-il les prédictions ?

Non, si le modèle est correctement réécrit.

Il prédit les mêmes valeurs.

Le centrage change principalement :

- la référence de l’intercept ;
- parfois la stabilité numérique ;
- l’interprétation des interactions si elles sont présentes.

---

# 8. Pourquoi diviser la séquence centrée par dix ?

Le script a défini :

\[
\text{sequence\_c10}
=
\frac{\text{sequence}-32{,}5}{10}
\]

---

## 8.1 Exemples

| Essai | Calcul | `sequence_c10` |
|---:|---|---:|
| 1 | \((1-32,5)/10\) | −3,15 |
| 10 | \((10-32,5)/10\) | −2,25 |
| 32,5 | 0 | 0 |
| 50 | \((50-32,5)/10\) | 1,75 |
| 64 | \((64-32,5)/10\) | 3,15 |

---

## 8.2 Pourquoi diviser par dix ?

Sans division, le coefficient représenterait l’effet d’un seul essai.

Nous avons obtenu approximativement :

\[
-0{,}0434
\]

point de confiance par essai.

Ce nombre est petit et moins intuitif.

En divisant la variable par dix, le coefficient représente l’effet de dix essais :

\[
-0{,}434
\]

point de confiance pour dix essais supplémentaires.

---

## 8.3 Cela change-t-il les prédictions ?

Non.

Si :

\[
s_{10}=\frac{s_c}{10}
\]

alors le coefficient associé devient dix fois plus grand en valeur absolue afin de produire le même résultat.

Exemple :

\[
-0{,}0434\times10
=
-0{,}434
\]

Le choix de l’unité modifie seulement l’échelle du coefficient.

---

## 8.4 Analogie avec les distances

Une même distance peut être exprimée en :

```text
1 000 mètres
```

ou :

```text
1 kilomètre
```

Le nombre change, mais la distance physique reste la même.

Ici, nous exprimons l’évolution temporelle par dix essais plutôt que par un seul essai.

---

# 9. Formulation mathématique du modèle de contrôle

Pour le participant \(i\), l’item \(j\) et l’essai \(k\) :

\[
Y_{ijk}
=
\beta_0
+
\beta_1\text{Standard}_i
+
\beta_2\text{SequenceC10}_{ik}
+
u_i
+
v_j
+
\varepsilon_{ijk}
\]

---

## 9.1 Définition des termes

| Terme | Signification |
|---|---|
| \(Y_{ijk}\) | Confiance observée |
| \(\beta_0\) | Confiance moyenne Neutral au milieu de l’expérience |
| \(\beta_1\) | Différence Standard–Neutral |
| \(\beta_2\) | Variation pour dix essais supplémentaires |
| \(u_i\) | Effet aléatoire participant |
| \(v_j\) | Effet aléatoire item |
| \(\varepsilon_{ijk}\) | Résidu de l’essai |

---

## 9.2 Codage de la condition

La condition est codée conceptuellement :

\[
\text{Standard}_i=
\begin{cases}
0 & \text{si Neutral}\\
1 & \text{si Standard}
\end{cases}
\]

---

## 9.3 Prédiction Neutral au milieu

Pour Neutral :

\[
\text{Standard}=0
\]

Au milieu :

\[
\text{SequenceC10}=0
\]

Pour un participant et un item moyens :

\[
u_i=0,\quad v_j=0
\]

La prédiction est :

\[
Y=\beta_0
\]

---

## 9.4 Prédiction Standard au milieu

Pour Standard :

\[
\text{Standard}=1
\]

Donc :

\[
Y=\beta_0+\beta_1
\]

---

# 10. Différence entre modèle nul et modèle de contrôle

## 10.1 Modèle nul

\[
Y_{ij}
=
\beta_0+u_i+v_j+\varepsilon_{ij}
\]

Formule Python :

```python
confidence ~ 1
```

---

## 10.2 Modèle de contrôle

\[
Y_{ijk}
=
\beta_0
+
\beta_1\text{condition}
+
\beta_2\text{sequence\_c10}
+
u_i+v_j+\varepsilon_{ijk}
\]

Formule Python :

```python
confidence ~
    C(condition, Treatment(reference='Neutral'))
    + sequence_c10
```

---

## 10.3 Paramètres supplémentaires

Le modèle de contrôle ajoute deux coefficients fixes :

1. différence Standard–Neutral ;
2. pente de la séquence.

Les deux modèles conservent la même structure aléatoire :

```text
intercept participant
intercept item
```

---

# 11. Organisation générale du script

Le script suivait une architecture proche de :

```text
1. Importer les bibliothèques
2. Définir les chemins
3. Définir les formules
4. Charger le dataset analytique
5. Vérifier les colonnes
6. Filtrer les lignes complètes
7. Centrer et redimensionner sequence
8. Construire le modèle nul
9. Ajuster le modèle nul en ML
10. Construire le modèle de contrôle
11. Ajuster le contrôle en ML
12. Ajuster le contrôle en REML
13. Comparer nul et contrôle
14. Extraire les effets fixes
15. Comparer les variances
16. Calculer les R²
17. Produire les prédictions
18. Exporter les CSV, graphiques, textes et JSON
```

---

# 12. Les bibliothèques utilisées

Le script utilisait principalement :

```python
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
```

et probablement :

```python
import matplotlib.pyplot as plt
import seaborn as sns
```

pour les graphiques.

---

## 12.1 `scipy.stats`

Dans ce script, `scipy.stats` sert notamment au test du rapport de vraisemblance :

```python
stats.chi2.sf(...)
```

`chi2` représente la loi du khi-deux.

`sf` signifie *survival function*, c’est-à-dire la probabilité de dépasser une valeur donnée.

Cette fonction produit la valeur p du test.

---

# 13. Configuration des chemins et des formules

Les chemins étaient définis comme :

```python
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "dataset_analysis_E1.csv"
OUTPUT_DIR = BASE_DIR / "control_mixed_model_E1"
```

Les formules étaient :

```python
NULL_FORMULA = "confidence ~ 1"
```

et :

```python
CONTROL_FORMULA = (
    "confidence ~ "
    "C(condition, Treatment(reference='Neutral')) "
    "+ sequence_c10"
)
```

---

## 13.1 Pourquoi conserver les formules dans des variables ?

Cela évite de réécrire la formule à plusieurs endroits.

Avantages :

- moins de risques de faute ;
- affichage facile dans les rapports ;
- sauvegarde dans les CSV ou JSON ;
- modification centralisée.

---

# 14. Chargement et préparation des données

Le script chargeait :

```python
data = pd.read_csv(DATA_FILE)
```

Puis il vérifiait la présence des colonnes essentielles :

```text
confidence
subject_id
item_id
condition
sequence
```

---

## 14.1 Filtrage des lignes complètes

Comme précédemment :

```text
Lignes retirées car analysis_complete=False : 0
Lignes supprimées pour donnée essentielle manquante : 0
```

Les 9 024 observations étaient utilisées.

---

## 14.2 Conversion numérique

Les colonnes :

```text
confidence
sequence
```

étaient converties en valeurs numériques.

---

## 14.3 Conversion des identifiants

Les identifiants devenaient des chaînes :

```python
data["subject_id"] = data["subject_id"].astype(str)
data["item_id"] = data["item_id"].astype(str)
```

---

# 15. Traitement de la condition

Le script normalisait :

```python
observed_conditions = set(
    dataframe[
        "condition"
    ]
    .dropna()
    .astype(str)
    .unique()
)
```

Puis il vérifiait les modalités :

```text
Neutral
Standard
```

---

## 15.1 Syntaxe `C(...)`

Dans la formule :

```python
C(condition, Treatment(reference='Neutral'))
```

`C` demande à `statsmodels` de traiter `condition` comme une variable catégorielle.

---

## 15.2 Codage de traitement

Le **codage de traitement** compare chaque catégorie à une catégorie de référence.

Ici :

```text
référence = Neutral
```

Le modèle crée un coefficient :

```text
C(condition, Treatment(reference='Neutral'))[T.Standard]
```

Ce coefficient représente :

\[
\text{Standard}-\text{Neutral}
\]

---

## 15.3 Pourquoi choisir Neutral comme référence ?

Neutral constitue une référence théorique naturelle parce qu’elle vise à réduire l’influence sémantique des prémisses.

L’intercept peut alors être lu comme :

> Confiance attendue en condition Neutral, au milieu de l’expérience.

Nous aurions pu choisir Standard comme référence.

Dans ce cas :

- les prédictions resteraient identiques ;
- l’intercept deviendrait le niveau Standard ;
- le coefficient serait Neutral–Standard et changerait de signe.

Le choix de référence change la formulation, pas les différences réelles prédites.

---

# 16. Construction de `sequence_c10`

Le code était de la forme :

```python
sequence_mean = data["sequence"].mean()

data["sequence_c10"] = (
    data["sequence"] - sequence_mean
) / 10.0
```

---

## 16.1 Calcul de la moyenne

Comme chaque participant avait les séquences 1 à 64 :

\[
\text{sequence\_mean}=32{,}5
\]

---

## 16.2 Calcul ligne par ligne

Pour l’essai 1 :

\[
\frac{1-32{,}5}{10}
=
-3{,}15
\]

Pour l’essai 64 :

\[
\frac{64-32{,}5}{10}
=
3{,}15
\]

---

## 16.3 Que se passerait-il si nous supprimions cette transformation ?

Le modèle pourrait encore fonctionner avec `sequence`.

Mais :

- l’intercept correspondrait à l’essai 0 ;
- la pente serait exprimée par essai ;
- son interprétation serait moins intuitive ;
- les scénarios de prédiction seraient moins lisibles.

---

# 17. Construction de la structure aléatoire croisée

```python
model = smf.mixedlm(
    formula=formula,
    data=dataframe,
    groups=dataframe[
        "_global_group"
    ],
    re_formula="0",
    vc_formula=
        VARIANCE_COMPONENT_FORMULAS,
    use_sparse=USE_SPARSE_MATRICES,
)
```

---

## 17.1 Ce qui reste identique

La structure aléatoire n’est pas modifiée entre les deux modèles.

Cela est important pour leur comparaison.

Les seules différences sont les effets fixes.

---

# 18. Ajustement du modèle nul en ML

Le script réajustait le modèle nul en ML :

```python
null_ml = model.fit(
    reml=False,
    ...
)
```

Résultat :

\[
\log L_{\text{nul}}
=
-38670{,}8944
\]

---

## 18.1 Pourquoi ne pas simplement lire le résultat d’un ancien fichier ?

Réajuster les modèles dans le même script garantit :

- qu’ils utilisent exactement les mêmes lignes ;
- qu’ils utilisent les mêmes conversions ;
- qu’ils utilisent la même version des données ;
- que la comparaison est reproductible.

---

# 19. Ajustement du modèle de contrôle en ML

Le modèle de contrôle ML avait :

\[
\log L_{\text{contrôle}}
=
-38658{,}6599
\]

Il a convergé avec `lbfgs`.

---

## 19.1 Pourquoi ML ?

Parce que nous allons comparer les effets fixes de deux modèles différents :

```text
nul : intercept seulement
contrôle : intercept + condition + séquence
```

---

# 20. Ajustement du modèle de contrôle en REML

Le modèle de contrôle était également ajusté avec :

```python
reml=True
```

La version REML était utilisée pour présenter les coefficients et les composantes de variance finales de cette étape.

Les résultats transmis étaient :

| Paramètre | Estimation | SE | p |
|---|---:|---:|---:|
| Intercept | 73,146 | 1,748 | < .001 |
| Standard | 5,146 | 2,464 | .0367 |
| `sequence_c10` | −0,434 | 0,0967 | < .001 |

---

# 21. Pourquoi ajuster trois modèles dans le même script ?

Le script ajustait :

1. nul ML ;
2. contrôle ML ;
3. contrôle REML.

Chaque ajustement avait un rôle distinct.

| Modèle | Rôle |
|---|---|
| Nul ML | Référence de comparaison |
| Contrôle ML | Comparaison des effets fixes |
| Contrôle REML | Présentation des estimations finales |

Cela évite de comparer des résultats obtenus avec des méthodes incompatibles.

---

# 22. Le test du rapport de vraisemblance

Le **test du rapport de vraisemblance**, abrégé LRT pour *likelihood-ratio test*, compare deux modèles emboîtés.

Il demande :

> L’amélioration d’ajustement obtenue en ajoutant les nouveaux coefficients est-elle plus grande que ce que l’on attendrait par simple fluctuation ?

---

# 23. Notion de modèles emboîtés

Deux modèles sont **emboîtés** si le modèle réduit peut être obtenu en fixant certains paramètres du modèle complet à zéro.

---

## 23.1 Dans notre cas

Modèle nul :

\[
Y=\beta_0+u_i+v_j+\varepsilon
\]

Modèle de contrôle :

\[
Y
=
\beta_0
+
\beta_1\text{Standard}
+
\beta_2\text{SequenceC10}
+
u_i+v_j+\varepsilon
\]

Si :

\[
\beta_1=0
\]

et :

\[
\beta_2=0
\]

le modèle de contrôle devient le modèle nul.

Les modèles sont donc emboîtés.

---

# 24. Calcul du test du rapport de vraisemblance

La statistique est :

\[
LR
=
2
\left(
\log L_{\text{complet}}
-
\log L_{\text{réduit}}
\right)
\]

Ici :

\[
LR
=
2
\left[
-38658{,}6599
-
(-38670{,}8944)
\right]
\]

Commençons par la différence :

\[
-38658{,}6599+38670{,}8944
=
12{,}2345
\]

Puis :

\[
LR
=
2\times12{,}2345
=
24{,}469
\]

Nous obtenons donc :

\[
\chi^2(2)=24{,}47
\]

---

## 24.1 Pourquoi multiplier par deux ?

Sous certaines conditions théoriques, deux fois la différence de log-vraisemblance suit approximativement une loi du khi-deux lorsque le modèle réduit est correct.

Cette propriété permet de transformer l’amélioration d’ajustement en test statistique.

---

# 25. Les degrés de liberté du test

Le modèle nul avait trois paramètres estimés dans le résumé de comparaison :

```text
1 effet fixe
2 composantes de variance indiquées comme paramètres
```

Le modèle de contrôle en avait cinq :

```text
3 effets fixes
2 composantes de variance indiquées comme paramètres
```

La différence est :

\[
5-3=2
\]

Ces deux paramètres supplémentaires sont :

```text
condition Standard
sequence_c10
```

Le test possède donc deux degrés de liberté :

\[
df=2
\]

---

# 26. Interprétation de la valeur p

Le résultat était :

\[
p=4{,}86\times10^{-6}
\]

soit :

\[
0{,}00000486
\]

Cette valeur est très inférieure à 0,05.

---

## 26.1 Que signifie cette valeur p ?

Sous l’hypothèse que :

\[
\beta_{\text{Standard}}=0
\]

et :

\[
\beta_{\text{sequence}}=0
\]

la probabilité d’obtenir une amélioration de vraisemblance au moins aussi grande que celle observée serait très faible, selon les hypothèses du test.

---

## 26.2 Ce que la valeur p ne signifie pas

Elle ne signifie pas :

> Il y a 99,9995 % de probabilité que le modèle de contrôle soit vrai.

Elle ne mesure pas directement :

- la probabilité du modèle ;
- l’importance pratique des effets ;
- la causalité ;
- la qualité absolue des prédictions.

---

## 26.3 Conclusion correcte

> L’ajout conjoint de la condition et de la séquence améliore significativement l’ajustement par rapport au modèle nul.

Cela ne dit pas encore lequel des deux prédicteurs est responsable de l’amélioration. Il faut examiner leurs coefficients individuels.

Dans nos résultats, les deux étaient associés à la confiance.

---

# 27. L’AIC

AIC signifie :

```text
Akaike Information Criterion
```

ou :

```text
critère d’information d’Akaike
```

La formule générale est :

\[
AIC
=
-2\log L+2k
\]

où :

- \(\log L\) est la log-vraisemblance ;
- \(k\) est le nombre de paramètres estimés.

---

## 27.1 Intuition

L’AIC combine :

```text
qualité d’ajustement
+
pénalité de complexité
```

Un modèle avec davantage de paramètres s’ajuste presque toujours au moins aussi bien aux données d’apprentissage.

La pénalité empêche de préférer automatiquement le modèle le plus complexe.

---

## 27.2 Analogie avec un tailleur

Un costume fabriqué avec des centaines de réglages peut épouser parfaitement une personne précise, mais être inutilisable pour d’autres personnes.

L’AIC recherche un compromis entre :

- ajustement précis ;
- simplicité suffisante.

---

## 27.3 Résultats

| Modèle | AIC |
|---|---:|
| Nul | 77 349,79 |
| Contrôle | 77 329,32 |

La différence est :

\[
77329{,}32-77349{,}79
=
-20{,}47
\]

Le modèle de contrôle possède un AIC inférieur d’environ 20,5 points.

---

## 27.4 Règle d’interprétation

Pour des modèles ajustés sur les mêmes données :

```text
AIC plus faible = meilleur compromis ajustement–complexité
```

Une différence supérieure à 10 est souvent considérée comme une préférence nette, même si ces seuils restent des règles pratiques.

L’AIC favorise donc clairement le modèle de contrôle.

---

# 28. Le BIC

BIC signifie :

```text
Bayesian Information Criterion
```

ou :

```text
critère d’information bayésien
```

Sa formule générale est :

\[
BIC
=
-2\log L+k\log(n)
\]

où :

- \(k\) est le nombre de paramètres ;
- \(n\) est le nombre d’observations.

---

## 28.1 Différence avec l’AIC

L’AIC pénalise chaque paramètre par :

\[
2
\]

Le BIC pénalise chaque paramètre par :

\[
\log(n)
\]

Avec :

\[
n=9024
\]

nous avons :

\[
\log(9024)\approx9{,}11
\]

Le BIC pénalise donc davantage les paramètres supplémentaires.

---

## 28.2 Résultats

| Modèle | BIC |
|---|---:|
| Nul | 77 378,22 |
| Contrôle | 77 371,97 |

La différence est :

\[
77371{,}97-77378{,}22
=
-6{,}25
\]

Le BIC préfère lui aussi le modèle de contrôle, mais moins fortement que l’AIC.

---

## 28.3 Pourquoi cette différence ?

Le modèle de contrôle gagne en ajustement, mais ajoute deux paramètres.

L’AIC considère que le gain compense largement la complexité.

Le BIC, plus sévère, considère que le gain compense encore la complexité, mais de manière plus modérée.

---

# 29. Pourquoi utiliser plusieurs critères de comparaison ?

Aucun critère unique ne résume parfaitement toutes les considérations.

---

## 29.1 Test du rapport de vraisemblance

Il fournit un test formel pour des modèles emboîtés.

---

## 29.2 AIC

Il vise un compromis orienté vers la qualité prédictive relative.

---

## 29.3 BIC

Il pénalise plus fortement la complexité et favorise souvent des modèles plus parcimonieux.

---

## 29.4 Convergence des résultats

Dans notre cas, les trois indicateurs vont dans le même sens :

```text
LRT : amélioration significative
AIC : contrôle préféré
BIC : contrôle préféré
```

Cette convergence renforce la conclusion.

---

# 30. Extraction des effets fixes

Le script extrayait :

```text
estimate
standard_error
z_value
p_value
ci_95_lower
ci_95_upper
```

pour les trois coefficients :

```text
Intercept
Standard
sequence_c10
```

---

## 30.1 Statistique z

Pour chaque coefficient :

\[
z
=
\frac{\text{estimation}}{\text{erreur-type}}
\]

Exemple pour Standard :

\[
z
=
\frac{5{,}1456}{2{,}4637}
\approx2{,}089
\]

---

## 30.2 Test individuel

Le test demande :

\[
H_0:\beta=0
\]

contre :

\[
H_1:\beta\neq0
\]

Il vérifie donc si le coefficient est suffisamment éloigné de zéro relativement à son incertitude.

---

# 31. Interprétation de l’intercept

Le résultat REML était :

\[
\beta_0
=
73{,}146
\]

avec :

\[
SE=1{,}748
\]

et :

\[
IC_{95\%}
=
[69{,}721\,;\,76{,}572]
\]

---

## 31.1 Signification exacte

L’intercept représente la confiance attendue lorsque :

```text
condition = Neutral
sequence_c10 = 0
effet participant = 0
effet item = 0
```

Or :

\[
\text{sequence\_c10}=0
\]

correspond au milieu de l’expérience.

L’intercept est donc :

> La confiance prédite en condition Neutral au milieu de l’expérience, pour un participant et un item moyens selon le modèle.

---

## 31.2 Pourquoi l’intercept du modèle nul était-il plus élevé ?

Modèle nul :

\[
75{,}736
\]

Modèle de contrôle :

\[
73{,}146
\]

Le modèle nul mélangeait les deux conditions.

Le nouvel intercept concerne spécifiquement Neutral.

Comme Standard possède une confiance plus élevée, la moyenne globale du modèle nul était supérieure à la moyenne Neutral.

---

## 31.3 Vérification avec les moyennes ajustées

L’effet Standard était :

\[
5{,}146
\]

La confiance Standard prédite au milieu est donc :

\[
73{,}146+5{,}146
=
78{,}292
\]

Ces valeurs correspondent aux moyennes observées par condition :

```text
Neutral : 73,146
Standard : 78,292
```

Comme la séquence est équilibrée dans chaque condition, les moyennes ajustées au centre coïncident ici presque exactement avec les moyennes brutes.

---

# 32. Interprétation de l’effet de la condition

Le coefficient était :

\[
\beta_{\text{Standard}}
=
5{,}146
\]

avec :

\[
SE=2{,}464
\]

\[
z=2{,}089
\]

\[
p=0{,}0367
\]

et :

\[
IC_{95\%}
=
[0{,}317\,;\,9{,}974]
\]

---

## 32.1 Signification

À position égale dans l’expérience, la condition Standard est associée à une confiance moyenne environ 5,15 points plus élevée que la condition Neutral.

---

## 32.2 Pourquoi dire « associée » ?

Même si la condition est expérimentale, notre coefficient décrit d’abord une différence estimée.

Une interprétation causale dépend également :

- de l’affectation correcte des participants ;
- du respect du protocole ;
- de l’absence de biais systématique ;
- de la définition précise des conditions.

Le terme « associée » reste prudent.

---

## 32.3 Intervalle de confiance

L’intervalle va d’environ :

\[
0{,}32
\]

à :

\[
9{,}97
\]

La valeur zéro n’est pas incluse, ce qui correspond à :

\[
p<0{,}05
\]

Mais l’intervalle est assez large.

Les données sont compatibles avec :

- un petit effet proche de 0,3 point ;
- un effet proche de 10 points.

L’estimation centrale est 5,15, mais sa précision reste limitée.

---

## 32.4 Pourquoi l’erreur-type est-elle relativement grande ?

La condition varie entre les participants.

Nous avons essentiellement :

```text
71 unités participantes Standard
70 unités participantes Neutral
```

et non 9 024 unités indépendantes pour estimer cet effet.

Les 64 essais d’un participant fournissent de l’information sur son niveau moyen, mais ils ne transforment pas ce participant en 64 personnes indépendantes.

---

## 32.5 Signification pratique

Une différence de 5,15 points sur une échelle de 0 à 100 n’est pas énorme, mais elle n’est pas négligeable.

Elle représente environ :

\[
\frac{5{,}15}{22{,}30}
\approx0{,}23
\]

écart-type brut de confiance.

Il s’agit donc d’un effet de petite ampleur par rapport à la dispersion totale des observations.

---

# 33. Interprétation de l’effet de la séquence

Le coefficient était :

\[
\beta_{\text{sequence}}
=
-0{,}434
\]

avec :

\[
SE=0{,}0967
\]

\[
z=-4{,}487
\]

\[
p=7{,}23\times10^{-6}
\]

et :

\[
IC_{95\%}
=
[-0{,}624\,;\,-0{,}244]
\]

---

## 33.1 Signification

Chaque augmentation de dix essais est associée à une diminution moyenne de confiance d’environ :

\[
0{,}434
\]

point.

---

## 33.2 Effet par essai

Comme le coefficient concerne dix essais :

\[
\frac{-0{,}434}{10}
=
-0{,}0434
\]

La confiance diminue en moyenne d’environ 0,043 point par essai.

---

## 33.3 Du premier au dernier essai

La différence de séquence est :

\[
64-1=63
\]

soit :

\[
6{,}3
\]

tranches de dix essais.

L’évolution prédite est :

\[
-0{,}434\times6{,}3
\approx-2{,}734
\]

La confiance prédite diminue donc d’environ 2,7 points entre l’essai 1 et l’essai 64.

---

## 33.4 Pourquoi un petit effet peut-il avoir une petite valeur p ?

La taille de l’effet et sa précision sont deux choses différentes.

L’effet est faible en amplitude :

```text
−0,434 point par dix essais
```

Mais son erreur-type est également faible :

```text
0,0967
```

Le rapport est :

\[
\frac{-0{,}434}{0{,}0967}
\approx-4{,}49
\]

Le modèle détecte donc précisément une petite tendance.

---

## 33.5 Interprétation scientifique

Cette baisse peut correspondre à :

- une fatigue ;
- un recalibrage ;
- une moindre utilisation du maximum 100 ;
- une prise de conscience de la difficulté ;
- une autre évolution temporelle.

Le coefficient seul ne permet pas de distinguer ces mécanismes.

Les analyses ultérieures du plafond montreront que cette baisse est largement liée à une diminution de l’utilisation de la réponse 100.

---

# 34. Prédictions ajustées selon la condition et la séquence

Le modèle permet de calculer la confiance attendue pour différents scénarios.

La formule fixe est :

\[
\widehat Y
=
73{,}146
+
5{,}146\times\text{Standard}
-
0{,}434\times\text{sequence\_c10}
\]

---

## 34.1 Neutral au milieu

\[
\text{Standard}=0
\]

\[
\text{sequence\_c10}=0
\]

Donc :

\[
\widehat Y=73{,}146
\]

---

## 34.2 Standard au milieu

\[
\widehat Y
=
73{,}146+5{,}146
=
78{,}292
\]

---

## 34.3 Neutral au premier essai

Au premier essai :

\[
\text{sequence\_c10}
=
-3{,}15
\]

Donc :

\[
\widehat Y
=
73{,}146
-
0{,}434\times(-3{,}15)
\]

\[
\widehat Y
=
73{,}146+1{,}367
\]

\[
\widehat Y
\approx74{,}513
\]

---

## 34.4 Neutral au dernier essai

\[
\text{sequence\_c10}
=
3{,}15
\]

\[
\widehat Y
=
73{,}146
-
0{,}434\times3{,}15
\]

\[
\widehat Y
\approx71{,}779
\]

---

## 34.5 Standard au premier essai

\[
\widehat Y
=
73{,}146
+
5{,}146
+
1{,}367
\]

\[
\widehat Y
\approx79{,}659
\]

---

## 34.6 Standard au dernier essai

\[
\widehat Y
=
73{,}146
+
5{,}146
-
1{,}367
\]

\[
\widehat Y
\approx76{,}925
\]

---

## 34.7 Tableau récapitulatif

| Condition | Essai 1 | Milieu | Essai 64 |
|---|---:|---:|---:|
| Neutral | 74,51 | 73,15 | 71,78 |
| Standard | 79,66 | 78,29 | 76,93 |

Ces valeurs utilisent seulement les effets fixes. Elles correspondent à un participant et un item d’effets aléatoires nuls.

---

# 35. Composantes de variance du modèle de contrôle

Les résultats ML transmis étaient :

| Composante | Variance |
|---|---:|
| Participant | 196,394 environ dans un export, ou 193,734 dans le réajustement final du script cognitif |
| Item | 11,894 environ |
| Résiduelle | 284,781 environ |

Les petites différences entre exports provenaient de versions ou de réajustements très proches du script. L’interprétation générale ne change pas.

La première comparaison fournie indiquait :

| Composante | Nul | Contrôle |
|---|---:|---:|
| Participant | 199,565 | 196,394 |
| Item | 11,865 | 11,894 |
| Résiduelle | 285,413 | 284,781 |

---

## 35.1 Pourquoi les variances changent-elles lorsqu’on ajoute des effets fixes ?

Dans le modèle nul, toute la variation systématique de condition et de séquence doit être absorbée par :

- la variance participant ;
- la variance item ;
- la variance résiduelle.

Dans le modèle de contrôle, une partie de cette variation est explicitement représentée par les effets fixes.

Les variances restantes peuvent donc diminuer.

---

# 36. Comparer les variances du modèle nul et du modèle de contrôle

## 36.1 Variance participant

Elle passe approximativement de :

\[
199{,}565
\]

à :

\[
196{,}394
\]

La réduction est :

\[
199{,}565-196{,}394
=
3{,}171
\]

La proportion expliquée relativement au modèle nul est :

\[
\frac{3{,}171}{199{,}565}
\approx0{,}0159
\]

soit :

\[
1{,}59\%
\]

---

## 36.2 Pourquoi la condition réduit-elle surtout la variance participant ?

La condition est constante pour chaque participant.

Une différence moyenne entre Standard et Neutral apparaissait donc partiellement comme une différence entre participants dans le modèle nul.

Une fois la condition ajoutée, une petite partie de la variance participant est expliquée.

---

## 36.3 Variance item

Elle change très peu :

\[
11{,}865\rightarrow11{,}894
\]

La légère augmentation ne signifie pas que le modèle crée réellement de la difficulté item.

Les composantes sont estimées conjointement. De petites variations positives ou négatives peuvent apparaître lorsque la structure fixe change.

Le changement d’environ 0,25 % est négligeable.

---

## 36.4 Variance résiduelle

Elle diminue légèrement :

\[
285{,}413\rightarrow284{,}781
\]

soit une réduction d’environ :

\[
0{,}22\%
\]

La séquence explique donc une petite partie de la variation au niveau des essais.

---

## 36.5 Conclusion

La condition et la séquence améliorent clairement la vraisemblance, mais elles n’expliquent qu’une petite portion de la variance totale.

Cela est possible parce qu’un effet peut être :

- statistiquement détectable ;
- mais faible en ampleur.

---

# 37. Le coefficient de détermination \(R^2\)

Dans une régression ordinaire, le \(R^2\) mesure la proportion de variance expliquée par les prédicteurs.

Dans un modèle mixte, il existe plusieurs formes de \(R^2\), car le modèle contient :

- des effets fixes ;
- des effets aléatoires.

Nous avons calculé :

- le \(R^2\) marginal ;
- le \(R^2\) conditionnel.

---

## 37.1 Variance des effets fixes

Pour chaque observation, le modèle calcule une prédiction fixe :

\[
\hat Y^{\text{fixe}}_i
=
\mathbf x_i^\top\hat{\boldsymbol\beta}
\]

La variance de ces prédictions était :

\[
\sigma^2_{\text{fixe}}
=
7{,}262
\]

Cette valeur mesure à quel point les prédictions changent à cause de la condition et de la séquence.

---

## 37.2 Variance totale utilisée pour les R²

La variance totale est définie ici comme :

\[
\sigma^2_{\text{totale}}
=
\sigma^2_{\text{fixe}}
+
\sigma^2_{\text{participant}}
+
\sigma^2_{\text{item}}
+
\sigma^2_{\text{résiduelle}}
\]

---

# 38. Le \(R^2\) marginal

Le \(R^2\) marginal mesure la part de variance attribuée aux effets fixes :

\[
R^2_m
=
\frac{
\sigma^2_{\text{fixe}}
}{
\sigma^2_{\text{fixe}}
+
\sigma^2_{\text{participant}}
+
\sigma^2_{\text{item}}
+
\sigma^2_{\text{résiduelle}}
}
\]

Nous avons obtenu environ :

\[
R^2_m=0{,}0145
\]

soit :

\[
1{,}45\%
\]

---

## 38.1 Interprétation

La condition et la séquence expliquent ensemble environ 1,45 % de la variance totale selon cette décomposition.

---

## 38.2 Est-ce peu ?

Oui, c’est une proportion modeste.

Mais cela ne rend pas les effets inutiles.

La confiance est très variable et fortement structurée par les différences individuelles. Deux contrôles simples ne pouvaient pas en expliquer une grande partie.

---

# 39. Le \(R^2\) conditionnel

Le \(R^2\) conditionnel inclut :

- les effets fixes ;
- les effets aléatoires participant ;
- les effets aléatoires item.

La formule est :

\[
R^2_c
=
\frac{
\sigma^2_{\text{fixe}}
+
\sigma^2_{\text{participant}}
+
\sigma^2_{\text{item}}
}{
\sigma^2_{\text{totale}}
}
\]

Résultat :

\[
R^2_c\approx0{,}4308
\]

soit :

\[
43{,}08\%
\]

---

## 39.1 Interprétation

Le modèle complet, en incluant les différences stables entre participants et items, représente environ 43 % de la variance.

La majeure partie de cette valeur vient des effets participants, pas de condition et séquence.

---

## 39.2 Pourquoi ne pas dire que le modèle explique causalement 43 % ?

Les effets aléatoires participants décrivent une structure de regroupement. Ils ne fournissent pas une explication psychologique de la cause de ces différences.

Le \(R^2\) conditionnel mesure une capacité de représentation statistique, pas une explication causale complète.

---

# 40. Les fichiers générés

Le dossier de sortie était :

```text
control_mixed_model_E1/
```

Les noms exacts pouvaient inclure les éléments suivants.

---

## 40.1 Résumés de modèles

```text
control_model_ML_summary.txt
control_model_REML_summary.txt
```

Ils contiennent les tableaux complets de `statsmodels`.

---

## 40.2 Effets fixes

```text
control_model_fixed_effects.csv
```

Contenu :

```text
parameter
estimate
standard_error
z_value
p_value
ci_95_lower
ci_95_upper
```

---

## 40.3 Comparaison des modèles

```text
model_comparison.csv
```

Contenu observé :

```text
model
formula
log_likelihood
aic
bic
number_of_estimated_parameters
likelihood_ratio_vs_null
degrees_of_freedom_difference
likelihood_ratio_p_value
```

---

## 40.4 Comparaison des variances

```text
variance_comparison.csv
```

Contenu :

```text
component
null_variance
control_variance
absolute_change
percentage_change
proportion_explained_relative_to_null
```

---

## 40.5 R²

Le script enregistrait les valeurs :

```text
fixed_effect_variance
marginal_r2
conditional_r2
```

dans un fichier CSV, texte ou JSON selon la version.

---

## 40.6 Prédictions

Un fichier de prédictions pouvait contenir :

```text
subject_id
item_id
condition
sequence
confidence
predicted_fixed
residual
```

Les prédictions fixes permettent notamment de représenter :

- les différences entre conditions ;
- la baisse selon la séquence.

---

## 40.7 Graphiques

Le script a produit des graphiques comme :

- confiance prédite selon la séquence et la condition ;
- résidus contre valeurs ajustées ;
- distribution des résidus ;
- décomposition de variance ;
- comparaison nul–contrôle.

Leur rôle est descriptif et diagnostique.

---

# 41. Résultats complets du modèle

## 41.1 Effets fixes REML

| Paramètre | Estimation | Erreur-type | z | p | IC 95 % |
|---|---:|---:|---:|---:|---:|
| Intercept | 73,146 | 1,748 | 41,849 | < .001 | [69,721 ; 76,572] |
| Standard vs Neutral | 5,146 | 2,464 | 2,089 | .0367 | [0,317 ; 9,974] |
| Séquence, +10 essais | −0,434 | 0,0967 | −4,487 | < .001 | [−0,624 ; −0,244] |

---

## 41.2 Comparaison ML

| Modèle | Log-vraisemblance | AIC | BIC | Paramètres |
|---|---:|---:|---:|---:|
| Nul | −38 670,894 | 77 349,789 | 77 378,219 | 3 |
| Contrôle | −38 658,660 | 77 329,320 | 77 371,966 | 5 |

---

## 41.3 Test global

\[
\chi^2(2)=24{,}47
\]

\[
p=4{,}86\times10^{-6}
\]

---

## 41.4 R²

\[
R^2_{\text{marginal}}
\approx0{,}0145
\]

\[
R^2_{\text{conditionnel}}
\approx0{,}4308
\]

---

## 41.5 Variance

| Source | Nul ML | Contrôle | Changement |
|---|---:|---:|---:|
| Participant | 199,565 | environ 196,394 | −1,59 % |
| Item | 11,865 | environ 11,894 | +0,25 % |
| Résiduelle | 285,413 | environ 284,781 | −0,22 % |

---

# 42. Ce que les résultats permettent de conclure

## 42.1 Le modèle de contrôle est préférable au modèle nul

Les trois critères convergent :

- test du rapport de vraisemblance significatif ;
- AIC inférieur ;
- BIC inférieur.

La condition et la séquence contiennent donc une information utile sur la confiance.

---

## 42.2 La confiance est plus élevée en Standard

La différence estimée est d’environ :

\[
5{,}15
\]

points.

---

## 42.3 La confiance diminue légèrement au fil des essais

La baisse est approximativement :

\[
0{,}44
\]

point par dix essais, soit environ :

\[
2{,}7
\]

points du premier au dernier essai.

---

## 42.4 Ces effets expliquent peu de variance totale

Le \(R^2\) marginal est d’environ 1,45 %.

Leur présence est statistiquement claire, mais leur ampleur globale reste modeste par rapport aux différences entre participants et aux fluctuations résiduelles.

---

## 42.5 Les différences individuelles restent majeures

La variance participant diminue peu.

La condition n’explique donc qu’une petite partie des fortes différences de confiance entre participants.

---

# 43. Ce que les résultats ne permettent pas de conclure

## 43.1 Ils n’expliquent pas pourquoi Standard augmente la confiance

Le coefficient ne distingue pas entre :

- familiarité sémantique ;
- facilité subjective ;
- croyabilité ;
- usage du plafond ;
- autre mécanisme.

---

## 43.2 Ils n’expliquent pas pourquoi la confiance diminue

Le coefficient de séquence ne permet pas de choisir entre :

- fatigue ;
- apprentissage ;
- recalibrage ;
- baisse de l’utilisation de 100 ;
- effet de composition des essais.

---

## 43.3 Ils ne testent pas encore les variables cognitives

Aucune conclusion ne peut encore être tirée sur :

- l’entropie ;
- la précision ;
- les modèles mentaux ;
- la validité.

---

## 43.4 Ils ne prouvent pas une grande importance pratique

Une faible valeur p peut accompagner un effet de petite taille.

Il faut toujours examiner ensemble :

- l’estimation ;
- l’intervalle de confiance ;
- le \(R^2\) ;
- le contexte scientifique.

---

# 44. Limites du modèle de contrôle

## 44.1 Relation linéaire avec la séquence

Le modèle suppose une baisse constante.

Il ne permet pas, par exemple :

```text
forte baisse au début
plateau au milieu
nouvelle baisse à la fin
```

Une courbe non linéaire nécessiterait :

- un terme quadratique ;
- une spline ;
- des blocs de séquence ;
- une autre représentation temporelle.

---

## 44.2 Même pente pour tous les participants

Le modèle suppose que l’effet de séquence est identique pour tous.

En réalité :

- certains participants peuvent se fatiguer ;
- d’autres peuvent apprendre ;
- d’autres rester stables.

Une pente aléatoire de séquence permettrait de représenter cela, mais elle rendrait le modèle plus complexe.

---

## 44.3 Pas d’interaction condition × séquence

Le modèle suppose que la baisse au fil du temps est la même en Standard et Neutral.

Il n’inclut pas :

```text
condition × sequence_c10
```

Une interaction demanderait :

> La trajectoire temporelle diffère-t-elle entre les deux conditions ?

Cette hypothèse n’était pas nécessaire au modèle de contrôle initial.

---

## 44.4 Effet plafond

Le modèle linéaire ne traite pas spécifiquement les 25,9 % de réponses égales à 100.

Les analyses ultérieures montreront que les effets de condition et de séquence sont largement liés à l’utilisation de cette borne.

---

## 44.5 Condition entre participants

L’effet de condition repose sur seulement 141 unités participantes, même si le fichier possède 9 024 essais.

Cela explique son intervalle relativement large.

---

# 45. Lien avec le modèle nul

Le modèle nul avait montré :

```text
forte variance participant
faible variance item
forte variance résiduelle
```

Le modèle de contrôle apporte maintenant deux premières explications :

```text
une différence entre conditions
une évolution au cours des essais
```

Mais il laisse presque intacte la structure générale de variance.

---

## 45.1 Ce que nous avons appris en plus

Avant :

> La confiance varie fortement.

Après :

> Une petite partie de cette variation est liée à une confiance plus élevée en Standard et à une légère diminution au fil du temps.

---

## 45.2 Ce qui reste inexpliqué

La majeure partie des différences :

- entre participants ;
- entre items ;
- entre essais ;

reste présente.

Nous devons donc ajouter les prédicteurs cognitifs.

---

# 46. Pourquoi cette étape conduit au modèle cognitif

La prochaine étape ajoutera :

```text
subject_accuracy_z
item_entropy_z
subject_mean_models_z
models_within_subject_z
validity_binary
```

tout en conservant :

```text
condition
sequence_c10
```

Le futur modèle demandera :

> Les prédicteurs cognitifs améliorent-ils le modèle au-delà de la condition et de la séquence ?

La comparaison essentielle sera :

```text
Modèle de contrôle ML
contre
Modèle cognitif ML
```

Nous examinerons ensuite :

- l’amélioration globale de la vraisemblance ;
- les coefficients individuels ;
- les changements de variance ;
- le \(R^2\) marginal ;
- la robustesse des effets.

---

# 47. Bilan pédagogique

Le script `fit_control_mixed_model_E1.py` a permis de :

1. conserver la structure croisée participant–item du modèle nul ;
2. ajouter la condition expérimentale comme prédicteur entre participants ;
3. ajouter la position de l’essai comme prédicteur temporel ;
4. centrer la séquence sur 32,5 ;
5. exprimer sa pente par dix essais ;
6. choisir Neutral comme condition de référence ;
7. interpréter l’intercept comme la confiance Neutral au milieu de l’expérience ;
8. ajuster le modèle nul et le modèle de contrôle en ML ;
9. comparer correctement leurs effets fixes ;
10. ajuster le modèle de contrôle en REML pour présenter ses coefficients ;
11. calculer un test du rapport de vraisemblance ;
12. calculer l’AIC et le BIC ;
13. montrer que le contrôle améliore le modèle nul ;
14. estimer une différence Standard–Neutral d’environ 5,15 points ;
15. estimer une baisse d’environ 0,44 point par dix essais ;
16. montrer que les effets fixes de contrôle expliquent environ 1,45 % de la variance totale ;
17. montrer que le modèle complet représente environ 43 % de la variance en incluant les effets aléatoires ;
18. constater que les fortes différences individuelles restent largement inexpliquées ;
19. établir une base équitable pour tester ensuite les prédicteurs cognitifs.

La conclusion centrale est :

> La confiance est plus élevée dans la condition Standard et diminue légèrement au cours de l’expérience. Ces deux effets sont statistiquement détectables et améliorent l’ajustement du modèle, mais ils n’expliquent qu’une petite partie de la variation totale de confiance.

# Étape 5 — Le modèle cognitif principal avec `fit_cognitive_mixed_model_E1.py`

## Sommaire

1. [Rôle de cette étape](#1-rôle-de-cette-étape)
2. [Passage du modèle de contrôle au modèle cognitif](#2-passage-du-modèle-de-contrôle-au-modèle-cognitif)
3. [Questions scientifiques](#3-questions-scientifiques)
4. [Les prédicteurs ajoutés](#4-les-prédicteurs-ajoutés)
5. [Pourquoi standardiser les prédicteurs ?](#5-pourquoi-standardiser-les-prédicteurs)
6. [Calcul mathématique d’un score standardisé](#6-calcul-mathématique-dun-score-standardisé)
7. [Précision moyenne du participant](#7-précision-moyenne-du-participant)
8. [Entropie de l’item](#8-entropie-de-litem)
9. [Nombre moyen de modèles mentaux](#9-nombre-moyen-de-modèles-mentaux)
10. [Composante intra-individuelle du nombre de modèles](#10-composante-intra-individuelle-du-nombre-de-modèles)
11. [Validité logique](#11-validité-logique)
12. [Formulation mathématique complète](#12-formulation-mathématique-complète)
13. [Interprétation conditionnelle des coefficients](#13-interprétation-conditionnelle-des-coefficients)
14. [Organisation du script](#14-organisation-du-script)
15. [Configuration et formules](#15-configuration-et-formules)
16. [Chargement et vérification des données](#16-chargement-et-vérification-des-données)
17. [Standardisation dans le code](#17-standardisation-dans-le-code)
18. [Vérification des corrélations](#18-vérification-des-corrélations)
19. [Construction des modèles](#19-construction-des-modèles)
20. [Les quatre ajustements effectués](#20-les-quatre-ajustements-effectués)
21. [Comparaison du modèle de contrôle et du modèle cognitif](#21-comparaison-du-modèle-de-contrôle-et-du-modèle-cognitif)
22. [AIC et BIC](#22-aic-et-bic)
23. [Lecture des effets fixes](#23-lecture-des-effets-fixes)
24. [Interprétation de l’intercept](#24-interprétation-de-lintercept)
25. [Condition Standard](#25-condition-standard)
26. [Séquence](#26-séquence)
27. [Précision du participant](#27-précision-du-participant)
28. [Entropie de l’item](#28-entropie-de-litem)
29. [Nombre moyen de modèles mentaux](#29-nombre-moyen-de-modèles-mentaux)
30. [Variation intra-individuelle du nombre de modèles](#30-variation-intra-individuelle-du-nombre-de-modèles)
31. [Validité logique](#31-validité-logique)
32. [Pourquoi une amélioration globale peut coexister avec plusieurs coefficients non significatifs](#32-pourquoi-une-amélioration-globale-peut-coexister-avec-plusieurs-coefficients-non-significatifs)
33. [Composantes de variance](#33-composantes-de-variance)
34. [R² marginal et conditionnel](#34-r2-marginal-et-conditionnel)
35. [Prédictions et erreurs](#35-prédictions-et-erreurs)
36. [Les fichiers produits](#36-les-fichiers-produits)
37. [Résultats initiaux avec trois simulations](#37-résultats-initiaux-avec-trois-simulations)
38. [Résultats finaux avec vingt simulations](#38-résultats-finaux-avec-vingt-simulations)
39. [Pourquoi les résultats n3 et n20 diffèrent légèrement](#39-pourquoi-les-résultats-n3-et-n20-diffèrent-légèrement)
40. [Ce que cette étape permet de conclure](#40-ce-que-cette-étape-permet-de-conclure)
41. [Ce qu’elle ne permet pas encore de conclure](#41-ce-quelle-ne-permet-pas-encore-de-conclure)
42. [Limites méthodologiques](#42-limites-méthodologiques)
43. [Lien avec les étapes précédentes](#43-lien-avec-les-étapes-précédentes)
44. [Pourquoi poursuivre avec les analyses de sensibilité](#44-pourquoi-poursuivre-avec-les-analyses-de-sensibilité)
45. [Bilan pédagogique](#45-bilan-pédagogique)

---

# 1. Rôle de cette étape

Le modèle nul avait montré que la confiance variait fortement entre participants, entre items et entre essais.

Le modèle de contrôle avait ensuite montré que la confiance :

- était plus élevée dans la condition Standard ;
- diminuait légèrement au cours de l’expérience.

Nous voulions maintenant examiner nos hypothèses cognitives principales.

Le script utilisé était :

```text
fit_cognitive_mixed_model_E1.py
```

Il ajoutait au modèle de contrôle cinq prédicteurs :

```text
subject_accuracy_z
item_entropy_z
subject_mean_models_z
models_within_subject_z
validity_binary
```

L’objectif n’était plus seulement de décrire le plan expérimental. Il s’agissait de tester si certaines propriétés des participants, des items et des simulations MReasoner étaient associées à la confiance.

---

# 2. Passage du modèle de contrôle au modèle cognitif

## 2.1 Modèle de contrôle

Le modèle précédent était :

\[
\begin{aligned}
\text{confidence}
={}&
\beta_0
+\beta_1\text{Standard}\\
&+\beta_2\text{sequence\_c10}\\
&+u_{\text{participant}}
+v_{\text{item}}
+\varepsilon
\end{aligned}
\]

---

## 2.2 Modèle cognitif initial

Le modèle cognitif est devenu :

\[
\begin{aligned}
\text{confidence}
={}&
\beta_0
+\beta_1\text{Standard}
+\beta_2\text{sequence\_c10}\\
&+\beta_3\text{subject\_accuracy\_z}\\
&+\beta_4\text{item\_entropy\_z}\\
&+\beta_5\text{subject\_mean\_models\_z}\\
&+\beta_6\text{models\_within\_subject\_z}\\
&+\beta_7\text{validity\_binary}\\
&+u_{\text{participant}}
+v_{\text{item}}
+\varepsilon
\end{aligned}
\]

Le modèle conservait donc :

- la condition ;
- la séquence ;
- les intercepts aléatoires participant et item.

Il ajoutait une couche d’explication cognitive.

---

## 2.3 Pourquoi conserver les contrôles ?

Nous voulions interpréter les prédicteurs cognitifs après prise en compte :

- de la différence Standard–Neutral ;
- de l’évolution au fil des essais.

Par exemple, si les items à forte entropie se trouvaient légèrement plus souvent en fin d’expérience, un modèle sans séquence pourrait attribuer à l’entropie une baisse provenant partiellement du temps.

Conserver les contrôles réduit ce risque.

---

# 3. Questions scientifiques

Le modèle cognitif répondait à cinq questions principales.

## 3.1 Précision individuelle

> Les participants globalement plus précis donnent-ils des niveaux de confiance différents ?

## 3.2 Entropie de l’item

> Les items suscitant davantage de désaccord entre participants produisent-ils une confiance plus faible ?

## 3.3 Nombre moyen de modèles mentaux

> Les participants générant généralement davantage de modèles mentaux sont-ils moins confiants ?

## 3.4 Variation intra-individuelle des modèles

> Pour une même personne, les types de tâches générant davantage de modèles que son niveau habituel réduisent-ils la confiance ?

## 3.5 Validité

> Les inférences valides produisent-elles une confiance différente des inférences invalides ?

---

# 4. Les prédicteurs ajoutés

| Prédicteur | Niveau de variation | Question |
|---|---|---|
| `subject_accuracy_z` | Participant | Différences de performance entre personnes |
| `item_entropy_z` | Item | Désaccord empirique entre réponses |
| `subject_mean_models_z` | Participant | Différences générales de modèles mentaux |
| `models_within_subject_z` | Participant × type de tâche | Variation interne des modèles mentaux |
| `validity_binary` | Type de tâche/essai | Valide contre invalide |

Cette distinction de niveau est fondamentale.

Une colonne répétée sur plusieurs lignes n’acquiert pas pour autant un nouveau niveau de variation.

Par exemple, `subject_accuracy_z` apparaît 64 fois pour un participant, mais ne contient qu’une seule information participant.

---

# 5. Pourquoi standardiser les prédicteurs ?

Les variables n’avaient pas la même unité.

Exemples :

```text
subject_accuracy : proportion comprise approximativement entre 0,4 et 1
item_entropy     : valeur entre 0 et 1
subject_mean_models : nombre de modèles autour de 2 à 5
models_within_subject : écart positif ou négatif
```

Comparer directement leurs coefficients serait difficile.

Nous les avons transformées en scores standardisés, notés par le suffixe :

```text
_z
```

---

## 5.1 Définition simple

Standardiser une variable consiste à :

1. soustraire sa moyenne ;
2. diviser par son écart-type.

Après standardisation :

- la moyenne est approximativement 0 ;
- l’écart-type est approximativement 1.

---

## 5.2 Pourquoi cela aide-t-il ?

Le coefficient représente alors l’effet d’une augmentation d’un écart-type.

Exemple :

\[
\beta_{\text{entropy}}=-2{,}44
\]

se lit :

> Une augmentation d’un écart-type de l’entropie est associée à une diminution moyenne de 2,44 points de confiance.

Les prédicteurs deviennent plus comparables sur le plan de leur échelle.

---

## 5.3 Ce que la standardisation ne fait pas

Elle ne :

- rend pas automatiquement la variable normale ;
- supprime pas les valeurs extrêmes ;
- garantit pas la linéarité ;
- change pas le signe de la relation ;
- crée pas une relation statistique ;
- rend pas les niveaux de données indépendants.

Elle modifie l’origine et l’unité de mesure.

---

# 6. Calcul mathématique d’un score standardisé

Pour une variable \(X\), le score standardisé est :

\[
Z_i
=
\frac{X_i-\bar X}{s_X}
\]

où :

- \(X_i\) est la valeur originale ;
- \(\bar X\) est la moyenne ;
- \(s_X\) est l’écart-type.

---

## 6.1 Exemple

Supposons que le nombre moyen de modèles ait :

\[
\bar X=2{,}72
\]

et :

\[
s_X=0{,}47
\]

Un participant ayant :

\[
X_i=3{,}19
\]

obtient :

\[
Z_i
=
\frac{3{,}19-2{,}72}{0{,}47}
=
1
\]

Il se situe un écart-type au-dessus de la moyenne.

Un participant ayant :

\[
X_i=2{,}25
\]

obtient :

\[
Z_i=-1
\]

---

## 6.2 Interprétation de zéro

Une valeur standardisée égale à zéro correspond à la moyenne observée de la variable.

Cela rend l’intercept du modèle plus interprétable :

> Confiance attendue lorsque les prédicteurs cognitifs continus sont à leur moyenne.

---

# 7. Précision moyenne du participant

La variable originale était :

```text
subject_accuracy
```

Elle correspondait à :

\[
\text{subject\_accuracy}_i
=
\frac{\text{nombre de réponses correctes}}
{64}
\]

Elle a été transformée en :

```text
subject_accuracy_z
```

---

## 7.1 Niveau de variation

Cette variable est constante à l’intérieur d’un participant.

Exemple :

| Participant | Essai | `subject_accuracy` |
|---|---:|---:|
| A | 1 | 0,625 |
| A | 2 | 0,625 |
| A | 64 | 0,625 |

Le coefficient compare donc des participants.

---

## 7.2 Hypothèses possibles

### Relation positive

Les participants plus performants reconnaissent leur compétence et sont plus confiants.

### Absence de relation

La confiance générale peut dépendre d’un style de réponse indépendant de la performance.

### Relation négative

Les participants plus précis pourraient être plus prudents, tandis que certains participants moins précis se montrent surconfiants.

Le modèle devait départager ces possibilités à partir des données.

---

## 7.3 Limite

La précision est calculée à partir des mêmes essais que ceux du modèle.

Elle n’est pas une mesure indépendante de compétence.

Il faut donc parler d’association entre précision moyenne observée et confiance, non d’effet causal de la compétence.

---

# 8. Entropie de l’item

La variable originale était :

```text
item_entropy
```

Elle a été transformée en :

```text
item_entropy_z
```

---

## 8.1 Rappel de la formule

Pour un item avec une proportion \(p\) de réponses Yes :

\[
H(p)
=
-p\log_2(p)
-(1-p)\log_2(1-p)
\]

---

## 8.2 Interprétation

### Entropie proche de 0

Les participants donnent presque tous la même réponse.

```text
Consensus élevé
```

### Entropie proche de 1

Les réponses sont presque également réparties entre Yes et No.

```text
Désaccord élevé
```

---

## 8.3 Hypothèse

Nous nous attendions à :

\[
\beta_{\text{entropie}}<0
\]

Une plus grande dispersion des réponses pourrait correspondre à des items pour lesquels la réponse est subjectivement moins évidente.

---

## 8.4 Niveau de variation

L’entropie est constante pour toutes les lignes du même item.

Le coefficient repose donc principalement sur les différences entre les 128 items, même si le fichier contient 9 024 lignes.

L’effet aléatoire item est indispensable pour éviter de traiter les répétitions de l’entropie comme des informations indépendantes.

---

## 8.5 Limite conceptuelle

L’entropie a été calculée à partir des réponses du même échantillon.

Elle représente :

```text
le désaccord empirique observé
```

et non nécessairement :

```text
une difficulté objective intrinsèque
```

L’interprétation doit rester associative.

---

# 9. Nombre moyen de modèles mentaux

La variable :

```text
subject_mean_models
```

est la moyenne personnelle du nombre de modèles MReasoner.

Elle a été standardisée en :

```text
subject_mean_models_z
```

---

## 9.1 Niveau de variation

Elle varie entre participants, mais pas entre les essais d’un même participant.

Elle répond à une question interindividuelle :

> Les personnes dont MReasoner génère généralement davantage de modèles ont-elles une confiance moyenne différente ?

---

## 9.2 Hypothèse théorique

Une hypothèse plausible était :

\[
\beta_{\text{mean models}}<0
\]

Si une personne représente davantage de possibilités alternatives, elle pourrait être moins certaine de sa réponse.

Mais l’hypothèse inverse pouvait aussi être envisagée :

- davantage de modèles pourrait refléter une représentation plus riche ;
- cette richesse pourrait parfois produire une décision plus assurée.

Le signe devait donc être déterminé empiriquement.

---

## 9.3 Ce que le coefficient ne mesure pas

Il ne mesure pas l’effet de générer un modèle supplémentaire sur un essai exact.

La valeur MReasoner était définie au niveau :

```text
participant × type de tâche
```

puis agrégée dans la moyenne personnelle.

---

# 10. Composante intra-individuelle du nombre de modèles

La variable :

```text
models_within_subject
```

était calculée comme :

\[
M_{ik}-\bar M_i
\]

où :

- \(M_{ik}\) est le nombre de modèles du participant \(i\) pour le type de tâche \(k\) ;
- \(\bar M_i\) est sa moyenne personnelle.

Elle a été standardisée en :

```text
models_within_subject_z
```

---

## 10.1 Question exacte

Le coefficient répond à :

> Pour un même participant, lorsque le type de tâche génère davantage de modèles que son niveau personnel habituel, sa confiance est-elle différente ?

Cette interprétation est intra-individuelle.

---

## 10.2 Exemple

Supposons qu’un participant ait :

| Tâche | Nombre de modèles | Moyenne personnelle | Écart |
|---|---:|---:|---:|
| MP | 2 | 3 | −1 |
| MT | 2 | 3 | −1 |
| AC | 3 | 3 | 0 |
| DA | 5 | 3 | +2 |

Si le coefficient est négatif, les essais DA de cette personne devraient être associés à une confiance plus faible que ses essais MP ou MT, toutes choses égales par ailleurs.

---

## 10.3 Pourquoi standardiser après le centrage personnel ?

Le centrage personnel sépare l’effet intra-individuel.

La standardisation change ensuite l’unité pour que le coefficient corresponde à un écart-type global de cette composante.

Ces deux transformations ont des fonctions différentes :

```text
centrage personnel
→ séparer le niveau intra-individuel

standardisation
→ rendre l’unité interprétable et comparable
```

---

# 11. Validité logique

La variable :

```text
validity_binary
```

était codée :

```text
0 = Invalid
1 = Valid
```

---

## 11.1 Interprétation du coefficient

Le coefficient représente :

\[
\text{confiance valide}
-
\text{confiance invalide}
\]

après prise en compte des autres variables.

Un coefficient positif signifie une confiance plus élevée pour les inférences valides.

---

## 11.2 Problème structurel

La validité était entièrement liée au type de tâche :

```text
MP et MT → valides
AC et DA → invalides
```

L’effet de validité ne peut donc pas être interprété comme une manipulation totalement indépendante de la forme d’inférence.

Il compare en pratique deux groupes de formes logiques :

```text
MP/MT contre AC/DA
```

---

## 11.3 Pourquoi l’inclure malgré cette limite ?

La validité fournissait une première représentation simple de la structure logique :

```text
valide contre invalide
```

Nous avons ensuite prévu une analyse de sensibilité remplaçant la validité par les quatre types de tâches.

Cette stratégie permettait de vérifier si la conclusion dépendait de ce résumé binaire.

---

# 12. Formulation mathématique complète

Pour l’essai \(k\) du participant \(i\) portant sur l’item \(j\) :

\[
\begin{aligned}
Y_{ijk}
={}&
\beta_0
+\beta_1 S_i
+\beta_2 Q_{ik}\\
&+\beta_3 A_i
+\beta_4 H_j\\
&+\beta_5 \bar M_i
+\beta_6 M^{W}_{ik}\\
&+\beta_7 V_{ijk}\\
&+u_i+v_j+\varepsilon_{ijk}
\end{aligned}
\]

où :

| Symbole | Variable |
|---|---|
| \(S_i\) | Condition Standard |
| \(Q_{ik}\) | `sequence_c10` |
| \(A_i\) | Précision standardisée |
| \(H_j\) | Entropie standardisée |
| \(\bar M_i\) | Moyenne standardisée des modèles |
| \(M^W_{ik}\) | Modèles intra-individuels standardisés |
| \(V_{ijk}\) | Validité binaire |
| \(u_i\) | Effet aléatoire participant |
| \(v_j\) | Effet aléatoire item |
| \(\varepsilon_{ijk}\) | Résidu |

---

# 13. Interprétation conditionnelle des coefficients

Dans une régression multiple, chaque coefficient est conditionnel aux autres variables du modèle.

Cela signifie que le coefficient de l’entropie ne représente pas simplement la corrélation brute entre entropie et confiance.

Il représente :

> La variation de confiance associée à l’entropie lorsque la condition, la séquence, la précision, les deux composantes MReasoner et la validité sont maintenues constantes dans le modèle.

---

## 13.1 « Maintenir constant » est une opération statistique

Le programme ne trouve pas nécessairement deux lignes réelles identiques sur toutes les variables sauf l’entropie.

Il utilise la structure globale des données pour estimer la contribution propre de chaque prédicteur.

---

## 13.2 Analogie avec le prix des maisons

Supposons que nous expliquions le prix avec :

- la surface ;
- le quartier ;
- l’âge ;
- le nombre de chambres.

Le coefficient de la surface estime la différence liée à la surface en tenant compte des autres caractéristiques.

Dans notre modèle, l’entropie est interprétée de manière analogue.

---

# 14. Organisation du script

Le script suivait approximativement cette structure :

```text
1. Définir les chemins et formules
2. Charger dataset_analysis_E1.csv
3. Vérifier les colonnes
4. Filtrer les lignes complètes
5. Convertir les variables
6. Centrer la séquence
7. Standardiser les prédicteurs
8. Vérifier leurs corrélations
9. Ajuster le modèle nul en ML
10. Ajuster le contrôle en ML
11. Ajuster le cognitif en ML
12. Ajuster le cognitif en REML
13. Comparer les modèles
14. Extraire les coefficients
15. Extraire les variances
16. Calculer les R²
17. Construire les prédictions
18. Exporter les résultats
```

---

# 15. Configuration et formules

Le script définissait :

```python
NULL_FORMULA = "confidence ~ 1"
```

```python
CONTROL_FORMULA = (
    "confidence ~ "
    "C(condition, Treatment(reference='Neutral')) "
    "+ sequence_c10"
)
```

et :

```python
COGNITIVE_FORMULA = (
    "confidence ~ "
    "C(condition, Treatment(reference='Neutral')) "
    "+ sequence_c10 "
    "+ subject_accuracy_z "
    "+ item_entropy_z "
    "+ subject_mean_models_z "
    "+ models_within_subject_z "
    "+ validity_binary"
)
```

---

## 15.1 Pourquoi garder les trois formules ?

Elles permettent de construire une chaîne de modèles emboîtés :

```text
Nul ⊂ Contrôle ⊂ Cognitif
```

Le symbole \(\subset\) signifie ici que chaque modèle réduit est contenu dans le suivant.

---

## 15.2 Structure aléatoire

Elle restait identique :

```python
VC_FORMULA = {
    "item": "0 + C(item_id)",
    "subject": "0 + C(subject_id)",
}
```

Cette stabilité est indispensable pour attribuer les différences d’ajustement aux effets fixes ajoutés.

---

# 16. Chargement et vérification des données

Le script vérifiait la présence de colonnes comme :

```text
confidence
subject_id
item_id
condition
sequence
subject_accuracy
item_entropy
subject_mean_models
models_within_subject
validity_binary
```

---

## 16.1 Pourquoi arrêter en cas de colonne absente ?

Sans `item_entropy`, par exemple, une formule pourrait échouer ou être modifiée involontairement.

Une erreur explicite protège la reproductibilité.

---

## 16.2 Conversion numérique

Les variables continues étaient converties avec :

```python
pd.to_numeric(..., errors="coerce")
```

Les valeurs impossibles devenaient manquantes et étaient détectables.

---

## 16.3 Conditions attendues

Le script vérifiait :

```text
Neutral
Standard
```

et que `validity_binary` ne contenait que :

```text
0
1
```

---

# 17. Standardisation dans le code

La fonction était :

```python
def standardize(series):
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    mean = numeric.mean()
    std = numeric.std(ddof=1)

    if not np.isfinite(std) or std <= 0:
        raise ValueError(...)

    return (
        (numeric - mean) / std,
        mean,
        std,
    )
```

---

## 17.1 Conversion numérique

Elle garantit que les opérations arithmétiques sont possibles.

---

## 17.2 Calcul de la moyenne

```python
mean = numeric.mean()
```

calcule :

\[
\bar X
\]

---

## 17.3 Écart-type avec `ddof=1`

```python
std = numeric.std(ddof=1)
```

`ddof=1` demande l’écart-type empirique corrigé, avec un dénominateur \(n-1\).

---

## 17.4 Vérification de l’écart-type

Si :

\[
s_X=0
\]

toutes les valeurs sont identiques.

La standardisation demanderait une division par zéro :

\[
\frac{X_i-\bar X}{0}
\]

Le script arrête donc l’exécution.

---

## 17.5 Création des colonnes

Pour chaque variable :

```python
z_column = f"{column}_z"
```

puis :

```python
data[z_column], mean, std = standardize(
    data[column]
)
```

Le script enregistrait aussi les paramètres de standardisation dans :

```text
predictor_standardization.csv
```

Cela permet de reconstruire l’échelle originale.

---

# 18. Vérification des corrélations

Le script calculait une matrice de corrélation entre :

```text
sequence_c10
subject_accuracy_z
item_entropy_z
subject_mean_models_z
models_within_subject_z
validity_binary
```

---

## 18.1 Pourquoi ?

Des prédicteurs fortement corrélés peuvent produire de la **multicolinéarité**.

La multicolinéarité signifie que plusieurs variables apportent une information très redondante.

Conséquences possibles :

- erreurs-types élevées ;
- coefficients instables ;
- signes sensibles à la spécification ;
- difficulté à isoler les contributions propres.

---

## 18.2 Seuil de surveillance

Le script signalait les corrélations absolues supérieures ou égales à :

\[
0{,}80
\]

Ce seuil n’est pas une loi universelle. Il sert d’alerte pratique.

---

## 18.3 Corrélations importantes observées

Dans les descriptifs initiaux :

\[
r(
\text{subject\_accuracy},
\text{subject\_mean\_models}
)
\approx0{,}568
\]

et :

\[
r(
\text{models\_within\_subject},
\text{validity}
)
\approx-0{,}586
\]

Ces corrélations sont notables, mais inférieures au seuil de 0,80.

Elles peuvent néanmoins augmenter l’incertitude des coefficients concernés.

---

# 19. Construction des modèles

La même fonction que précédemment construisait le modèle :

```python
def build_model(data, formula):
    model_data = data.copy()
    model_data["_global_group"] = 1

    return smf.mixedlm(
        formula=formula,
        data=model_data,
        groups=model_data["_global_group"],
        re_formula="0",
        vc_formula=VC_FORMULA,
    )
```

La formule changeait, mais la structure participant–item restait identique.

---

# 20. Les quatre ajustements effectués

Le script ajustait :

1. modèle nul ML ;
2. modèle de contrôle ML ;
3. modèle cognitif ML ;
4. modèle cognitif REML.

---

## 20.1 Nul ML

Référence générale.

---

## 20.2 Contrôle ML

Référence immédiate pour le modèle cognitif.

---

## 20.3 Cognitif ML

Utilisé pour comparer les effets fixes avec le modèle de contrôle.

---

## 20.4 Cognitif REML

Utilisé pour présenter :

- les coefficients finaux ;
- les composantes de variance ;
- les intervalles ;
- le \(R^2\).

---

# 21. Comparaison du modèle de contrôle et du modèle cognitif

Les log-vraisemblances ML initiales étaient :

\[
\log L_{\text{contrôle}}
=
-38658{,}6599
\]

\[
\log L_{\text{cognitif}}
=
-38621{,}5096
\]

La statistique du rapport de vraisemblance est :

\[
LR
=
2[
-38621{,}5096
-
(-38658{,}6599)
]
\]

\[
LR
=
2\times37{,}15035
\]

\[
LR
=
74{,}3007
\]

Le modèle cognitif ajoute cinq paramètres :

```text
subject_accuracy_z
item_entropy_z
subject_mean_models_z
models_within_subject_z
validity_binary
```

Donc :

\[
df=5
\]

Résultat :

\[
\chi^2(5)=74{,}30
\]

\[
p=1{,}30\times10^{-14}
\]

---

## 21.1 Interprétation

Les cinq prédicteurs cognitifs, considérés conjointement, améliorent fortement l’ajustement par rapport à la condition et à la séquence seules.

---

## 21.2 Ce que le test ne dit pas

Il ne dit pas que les cinq prédicteurs sont individuellement utiles.

Il dit seulement qu’au moins une partie de l’ensemble ajouté améliore suffisamment le modèle.

Les tests de sensibilité montreront que l’entropie porte l’essentiel de cette amélioration.

---

# 22. AIC et BIC

Les résultats ML initiaux étaient :

| Modèle | AIC | BIC |
|---|---:|---:|
| Nul | 77 349,79 | 77 378,22 |
| Contrôle | 77 329,32 | 77 371,97 |
| Cognitif | 77 265,02 | 77 343,20 |

---

## 22.1 AIC

L’AIC diminue fortement :

\[
77329{,}32-77265{,}02
=
64{,}30
\]

Le modèle cognitif possède un bien meilleur compromis entre ajustement et complexité.

---

## 22.2 BIC

Le BIC diminue aussi :

\[
77371{,}97-77343{,}20
=
28{,}77
\]

Même avec une pénalité plus forte, le modèle cognitif est préféré.

---

# 23. Lecture des effets fixes

Les résultats initiaux REML avec trois simulations étaient :

| Paramètre | Estimation | SE | p |
|---|---:|---:|---:|
| Intercept | 72,805 | 1,788 | < .001 |
| Standard | 5,150 | 2,535 | .042 |
| Séquence | −0,437 | 0,0966 | < .001 |
| Précision | 0,310 | 1,503 | .837 |
| Entropie | −2,437 | 0,277 | < .001 |
| Modèles moyens | −1,801 | 1,508 | .232 |
| Modèles intra | −0,366 | 0,246 | .137 |
| Validité | 0,678 | 0,622 | .276 |

---

# 24. Interprétation de l’intercept

L’intercept était :

\[
72{,}805
\]

Il correspond à :

```text
condition = Neutral
sequence_c10 = 0
subject_accuracy_z = 0
item_entropy_z = 0
subject_mean_models_z = 0
models_within_subject_z = 0
validity_binary = 0
effets aléatoires = 0
```

---

## 24.1 Traduction

Il s’agit de la confiance prédite :

- en condition Neutral ;
- au milieu de l’expérience ;
- pour un participant de précision moyenne ;
- sur un item d’entropie moyenne ;
- pour un participant au nombre moyen de modèles moyen ;
- lorsque l’écart intra-individuel des modèles est moyen, donc zéro ;
- pour une inférence invalide ;
- pour un participant et un item d’effets aléatoires nuls.

---

## 24.2 Pourquoi l’intercept se rapporte-t-il aux essais invalides ?

Parce que :

```text
validity_binary = 0
```

est la catégorie de référence.

Si nous avions centré la validité autour de sa moyenne, l’intercept aurait représenté une validité moyenne abstraite. Nous avons préféré un codage binaire simple.

---

# 25. Condition Standard

Le coefficient initial était :

\[
\beta=5{,}150
\]

\[
SE=2{,}535
\]

\[
p=0{,}042
\]

L’effet reste proche de celui du modèle de contrôle.

---

## 25.1 Signification

Après prise en compte des variables cognitives, Standard reste associé à environ 5,15 points de confiance supplémentaires par rapport à Neutral.

---

## 25.2 Stabilité du coefficient

Modèle de contrôle :

\[
5{,}146
\]

Modèle cognitif :

\[
5{,}150
\]

La valeur change à peine.

Cela suggère que les prédicteurs cognitifs ajoutés n’expliquent pas la différence de condition.

---

# 26. Séquence

Le coefficient initial était :

\[
-0{,}437
\]

par dix essais.

Il est presque identique à celui du modèle de contrôle :

\[
-0{,}434
\]

La baisse temporelle ne semble donc pas être expliquée par les prédicteurs cognitifs ajoutés.

---

# 27. Précision du participant

Résultat initial :

\[
\beta=0{,}310
\]

\[
SE=1{,}503
\]

\[
p=0{,}837
\]

\[
IC_{95\%}
=
[-2{,}636\,;\,3{,}256]
\]

---

## 27.1 Interprétation

Une augmentation d’un écart-type de la précision moyenne est associée à seulement 0,31 point de confiance supplémentaire selon l’estimation centrale.

Mais l’intervalle est large et contient nettement zéro.

Les données sont compatibles avec :

- un effet négatif modéré ;
- aucun effet ;
- un effet positif modéré.

---

## 27.2 Conclusion correcte

Nous n’avons pas détecté d’association conditionnelle claire entre la précision générale et le niveau moyen de confiance.

Il ne faut pas conclure :

> La précision n’a absolument aucun effet.

L’absence de preuve statistique n’est pas une preuve parfaite d’absence.

---

## 27.3 Pourquoi l’incertitude est-elle grande ?

`subject_accuracy_z` varie seulement entre 141 participants.

Elle est aussi corrélée avec `subject_mean_models_z`.

Son information effective est donc beaucoup plus proche de 141 unités que de 9 024 observations indépendantes.

---

# 28. Entropie de l’item

Résultat initial :

\[
\beta=-2{,}437
\]

\[
SE=0{,}277
\]

\[
z=-8{,}789
\]

\[
p\approx1{,}51\times10^{-18}
\]

\[
IC_{95\%}
=
[-2{,}981\,;\,-1{,}894]
\]

---

## 28.1 Interprétation

Une augmentation d’un écart-type de l’entropie est associée à une diminution moyenne d’environ 2,44 points de confiance.

---

## 28.2 Exemple

Supposons deux items identiques sur les autres variables du modèle :

```text
Item A : entropie = moyenne
Item B : entropie = moyenne + 1 écart-type
```

Le modèle prédit environ :

\[
2{,}44
\]

points de confiance en moins pour B.

---

## 28.3 Pourquoi cet effet est-il précis ?

L’erreur-type est petite relativement au coefficient :

\[
\frac{2{,}437}{0{,}277}
\approx8{,}79
\]

L’effet est :

- clairement négatif ;
- précisément estimé ;
- associé à une forte amélioration d’ajustement.

---

## 28.4 Interprétation prudente

L’entropie et la confiance sont associées.

Nous ne pouvons pas conclure directement :

> Le désaccord des autres participants cause la baisse de confiance de l’individu.

L’entropie peut résumer des propriétés non mesurées de l’item qui influencent à la fois :

- la dispersion des réponses ;
- la confiance.

---

# 29. Nombre moyen de modèles mentaux

Résultat initial :

\[
\beta=-1{,}801
\]

\[
SE=1{,}508
\]

\[
p=0{,}232
\]

\[
IC_{95\%}
=
[-4{,}756\,;\,1{,}155]
\]

---

## 29.1 Interprétation centrale

Les participants situés un écart-type au-dessus de la moyenne de modèles étaient estimés environ 1,80 point moins confiants.

La direction est cohérente avec l’hypothèse d’une réduction de la certitude.

---

## 29.2 Mais l’effet est incertain

L’intervalle contient zéro.

Nous ne pouvons pas distinguer suffisamment :

- un effet négatif ;
- aucun effet ;
- un petit effet positif.

---

## 29.3 Pourquoi l’erreur-type est-elle grande ?

La variable varie entre participants et est corrélée à la précision individuelle.

Les deux variables tentent donc d’expliquer certaines différences interindividuelles communes.

Cela rend leur contribution propre plus difficile à estimer.

---

# 30. Variation intra-individuelle du nombre de modèles

Résultat initial :

\[
\beta=-0{,}366
\]

\[
SE=0{,}246
\]

\[
p=0{,}137
\]

\[
IC_{95\%}
=
[-0{,}849\,;\,0{,}117]
\]

---

## 30.1 Direction de l’effet

Le coefficient négatif suggère :

> Lorsqu’un type de tâche génère davantage de modèles que la moyenne personnelle, la confiance tend à diminuer.

Cette direction est conforme à l’hypothèse théorique.

---

## 30.2 Pourquoi ne pas conclure immédiatement ?

L’intervalle contient zéro et la valeur p est supérieure à 0,05.

L’information n’était pas assez précise dans cette spécification initiale.

---

## 30.3 Résultat ultérieur

Avec 20 simulations et dans le modèle final parcimonieux sans validité, le coefficient est devenu :

\[
\beta=-0{,}485
\]

\[
SE=0{,}231
\]

\[
p=0{,}036
\]

Cette évolution sera expliquée dans la section consacrée aux résultats n20 et dans l’étape de sensibilité.

---

# 31. Validité logique

Résultat initial :

\[
\beta=0{,}678
\]

\[
SE=0{,}622
\]

\[
p=0{,}276
\]

\[
IC_{95\%}
=
[-0{,}541\,;\,1{,}896]
\]

---

## 31.1 Interprétation

Les essais valides étaient estimés environ 0,68 point plus confiants que les essais invalides, toutes choses égales par ailleurs.

L’incertitude est toutefois suffisante pour inclure un effet négatif, nul ou positif.

---

## 31.2 Conclusion

La validité n’ajoute pas de relation claire avec la confiance dans ce modèle.

---

## 31.3 Précaution

Comme validité et type de tâche sont structurellement liés, ce résultat ne représente pas un effet pur et indépendant de validité.

---

# 32. Pourquoi une amélioration globale peut coexister avec plusieurs coefficients non significatifs

Nous avons obtenu :

\[
\chi^2(5)=74{,}30,\quad p<.001
\]

mais quatre prédicteurs cognitifs sur cinq n’étaient pas individuellement significatifs.

Ce n’est pas contradictoire.

---

## 32.1 Le test global pose une question collective

Il demande :

> Les cinq coefficients sont-ils tous simultanément égaux à zéro ?

L’hypothèse nulle est :

\[
\beta_3=\beta_4=\beta_5=\beta_6=\beta_7=0
\]

L’effet très fort de l’entropie suffit à rendre cette hypothèse collective fausse.

---

## 32.2 Les tests individuels posent des questions séparées

Le test de `subject_accuracy_z` demande :

\[
\beta_3=0?
\]

Celui de l’entropie demande :

\[
\beta_4=0?
\]

Ainsi, un ensemble peut améliorer le modèle principalement grâce à une seule variable.

---

## 32.3 Analogie avec une équipe

Une équipe de cinq joueurs peut battre nettement une autre équipe grâce principalement à la performance exceptionnelle d’un joueur.

La victoire de l’équipe ne signifie pas que chacun des cinq joueurs a individuellement réalisé une performance exceptionnelle.

---

# 33. Composantes de variance

Dans le modèle cognitif REML initial :

| Composante | Variance | Écart-type |
|---|---:|---:|
| Participant | 196,650 | 14,023 |
| Item | 5,221 | 2,285 |
| Résiduelle | 284,750 | 16,875 |

---

## 33.1 Variance participant

Elle reste proche du modèle de contrôle.

Les variables :

```text
subject_accuracy_z
subject_mean_models_z
```

n’expliquent donc qu’une petite partie des fortes différences individuelles de confiance.

---

## 33.2 Variance item

Elle passe approximativement de :

\[
11{,}87
\]

dans le contrôle à :

\[
5{,}05
\]

dans le cognitif ML.

La réduction relative est :

\[
\frac{11{,}87-5{,}05}{11{,}87}
\approx0{,}574
\]

soit environ :

\[
57{,}4\%
\]

---

## 33.3 Pourquoi cette baisse est-elle importante ?

L’entropie est une variable d’item.

Elle explique une grande partie des différences systématiques entre items qui restaient dans le modèle de contrôle.

Cela renforce l’idée que l’amélioration cognitive est principalement portée par l’entropie.

---

## 33.4 Variance résiduelle

Elle change très peu :

\[
284{,}75
\]

contre environ :

\[
284{,}78
\]

dans le modèle de contrôle.

Les prédicteurs ajoutés expliquent surtout une variation structurée entre items, pas une grande quantité de bruit essai par essai.

---

# 34. R² marginal et conditionnel

Pour le modèle cognitif REML initial :

\[
\sigma^2_{\text{fixe}}
=
16{,}619
\]

\[
R^2_m
=
0{,}0330
\]

\[
R^2_c
=
0{,}4342
\]

---

## 34.1 R² marginal

\[
3{,}30\%
\]

de la variance totale est attribuée aux effets fixes selon cette décomposition.

Dans le modèle de contrôle :

\[
R^2_m\approx1{,}45\%
\]

L’ajout des variables cognitives augmente donc le \(R^2\) marginal d’environ :

\[
3{,}30-1{,}45
=
1{,}85
\]

point de pourcentage.

---

## 34.2 R² conditionnel

\[
43{,}42\%
\]

de la variance est représentée par les effets fixes et les effets aléatoires réunis.

Cette valeur reste proche de celle du modèle de contrôle, car l’importante variance participant dominait déjà le modèle.

---

## 34.3 Pourquoi l’amélioration de vraisemblance est-elle forte avec un petit gain de R² ?

Les critères ne mesurent pas exactement la même chose.

La vraisemblance tient compte de la structure complète des observations et de leur covariance.

Un prédicteur peut améliorer précisément la représentation de différences structurées entre items sans expliquer une énorme proportion de variance totale dominée par les différences participantes et le bruit résiduel.

---

# 35. Prédictions et erreurs

Le script construisait des prédictions fixes :

\[
\widehat{\mathbf y}_{\text{fixe}}
=
\mathbf X\widehat{\boldsymbol\beta}
\]

avec :

```python
result.model.exog @ result.fe_params
```

---

## 35.1 Pourquoi les appeler « fixes » ?

Elles comprennent :

- la condition ;
- la séquence ;
- les variables cognitives.

Elles n’ajoutent pas les effets propres aux participants et items.

---

## 35.2 Résidu fixe

Le script calculait :

\[
e_i^{\text{fixe}}
=
y_i-\widehat y_i^{\text{fixe}}
\]

Ces résidus décrivent l’écart à la tendance de population.

Ils sont plus larges que des résidus conditionnels qui incluraient les effets participants et items.

---

## 35.3 RMSE et MAE

Le script calculait :

\[
RMSE
=
\sqrt{
\frac{1}{n}
\sum_i(e_i^{\text{fixe}})^2
}
\]

et :

\[
MAE
=
\frac{1}{n}
\sum_i|e_i^{\text{fixe}}|
\]

Ces métriques étaient descriptives et calculées sur les données d’ajustement.

---

# 36. Les fichiers produits

Le dossier était :

```text
cognitive_mixed_model_E1/
```

---

## 36.1 `predictor_standardization.csv`

Contenu :

```text
variable
standardized_variable
mean
standard_deviation
```

### Utilité

Il indique comment revenir à l’échelle originale.

Si :

\[
Z=1
\]

la valeur originale est :

\[
X=\bar X+s_X
\]

---

## 36.2 `cognitive_predictor_correlations.csv`

Matrice de corrélation entre les prédicteurs.

### Utilité

Repérer les redondances et préparer l’interprétation de la multicolinéarité.

---

## 36.3 `high_predictor_correlations.csv`

Liste des paires dépassant le seuil choisi, ici \(|r|\geq0{,}80\).

S’il est vide, aucune alerte forte n’a été détectée selon ce seuil.

---

## 36.4 `cognitive_model_ML_summary.txt`

Résumé ML du modèle cognitif.

Il sert principalement aux comparaisons de modèles.

---

## 36.5 `cognitive_model_REML_summary.txt`

Résumé REML contenant :

- les coefficients finaux ;
- leurs erreurs-types ;
- les variances ;
- la convergence.

---

## 36.6 `cognitive_model_fixed_effects_ML.csv`

Coefficients ML.

Ils sont utiles pour documenter les modèles comparés.

---

## 36.7 `cognitive_model_fixed_effects_REML.csv`

Coefficients REML utilisés pour la présentation principale.

Colonnes :

```text
parameter
estimate
standard_error
z_value
p_value
ci_95_lower
ci_95_upper
interpretation
```

---

## 36.8 `model_comparison.csv`

Contient :

```text
model
formula
converged
log_likelihood
aic
bic
number_of_estimated_parameters
n_observations
residual_variance
```

Il permet de voir la progression :

```text
nul → contrôle → cognitif
```

---

## 36.9 `likelihood_ratio_tests.csv`

Contient :

```text
Null vs Control
Control vs Cognitive
Null vs Cognitive
```

avec :

- statistique LR ;
- degrés de liberté ;
- valeur p.

---

## 36.10 `variance_components.csv`

Compare les variances des modèles :

```text
Null_ML
Control_ML
Cognitive_ML
Cognitive_REML
```

---

## 36.11 `model_r2.csv`

Contient :

```text
fixed_effect_variance
participant_variance
item_variance
residual_variance
total_variance
marginal_r2
conditional_r2
```

---

## 36.12 `cognitive_model_predictions.csv`

Contient les prédictions fixes et les résidus correspondants.

---

## 36.13 `cognitive_model_results.json`

Archive :

- les formules ;
- le nombre d’observations ;
- les optimiseurs ;
- la convergence ;
- les tests de vraisemblance ;
- les métriques de prédiction.

---

# 37. Résultats initiaux avec trois simulations

Le modèle initial comprenait `validity_binary`.

| Prédicteur | β | SE | p | Conclusion |
|---|---:|---:|---:|---|
| Standard | 5,150 | 2,535 | .042 | Positif |
| Séquence | −0,437 | 0,097 | < .001 | Négatif |
| Précision | 0,310 | 1,503 | .837 | Non détecté |
| Entropie | −2,437 | 0,277 | < .001 | Fort effet négatif |
| Modèles moyens | −1,801 | 1,508 | .232 | Non détecté |
| Modèles intra | −0,366 | 0,246 | .137 | Non détecté |
| Validité | 0,678 | 0,622 | .276 | Non détecté |

Note : on considère un prédicteur "non détecté" lorsque p est supérieur au seuil conventionnel de 0,5. Cela est équivalent environ à chercher : 

\[
\frac{|\widehat\beta|}{SE}
>
1{,}96
\]

---

## 37.1 Conclusion initiale

Le modèle cognitif améliore fortement l’ajustement, mais cette amélioration paraît principalement liée à l’entropie.

Les variables MReasoner ont des coefficients négatifs, conformes à l’hypothèse, mais leur incertitude est trop grande pour conclure à ce stade.

---

# 38. Résultats finaux avec vingt simulations

Nous avons ensuite reconstruit :

```text
dataset_analysis_E1_n20.csv
```

et ajusté le modèle cognitif avec les estimations MReasoner fondées sur 20 simulations.

Dans le modèle avec validité, les résultats étaient :

| Prédicteur | n3 | n20 |
|---|---:|---:|
| Standard | 5,150 | 5,261 |
| Séquence | −0,437 | −0,437 |
| Précision | 0,310 | 0,696 |
| Entropie | −2,437 | −2,430 |
| Modèles moyens | −1,801 | −2,241 |
| Modèles intra | −0,366 | −0,342 |
| Validité | 0,678 | 0,726 |

Aucun coefficient n’a changé de signe ou de statut statistique.


Les trois simulations initiales donnaient une estimation bruitée du nombre moyen de modèles.

Avec vingt simulations :

- la moyenne se stabilise davantage ;
- certaines valeurs participant × tâche changent ;
- `subject_mean_models` est recalculée ;
- `models_within_subject` est recalculée ;
- les versions standardisées changent ;
- les corrélations avec les autres prédicteurs changent légèrement.

Les coefficients peuvent donc évoluer.

---

## 38.1 Ce qui est resté stable

L’entropie a changé de seulement :

\[
|-2{,}430-(-2{,}437)|
=
0{,}007
\]

dans le modèle avec validité.

La condition et la séquence sont également presque inchangées.

---

## 38.2 Ce qui a davantage changé

La composante interindividuelle est passée de :

\[
-1{,}801
\]

à :

\[
-2{,}241
\]

mais elle reste non significative.

La conclusion générale demeure donc stable.

---

# 39. Ce que cette étape permet de conclure

## 39.1 Les prédicteurs cognitifs améliorent le modèle

Le modèle cognitif s’ajuste mieux que le modèle de contrôle :

\[
\chi^2(5)=74{,}30,\quad p<.001
\]

---

## 39.2 L’entropie est le résultat principal

Les items suscitant davantage de désaccord sont associés à une confiance plus faible.

L’effet est :

- fort relativement à son erreur-type ;
- stable avec n3 et n20 ;
- accompagné d’une réduction importante de la variance item.

---

## 39.3 La précision générale n’explique pas la confiance moyenne

Les participants plus précis ne sont pas clairement plus confiants après prise en compte des autres variables.

Cela annonce les résultats ultérieurs de faible calibration métacognitive.

---

## 39.4 L’effet interindividuel et intraindividuel MReasoner n’est pas clairement détecté

Les participants générant généralement davantage de modèles ne présentent pas une différence fiable de confiance moyenne.

---

## 39.5 La validité n’apporte pas d’effet clair

Une fois les autres prédicteurs inclus, les essais valides ne présentent pas une confiance nettement différente des essais invalides.

---

# 40. Ce qu’elle ne permet pas encore de conclure

## 40.1 Quel prédicteur porte exactement l’amélioration globale ?

Le test global montre que les cinq variables améliorent le modèle ensemble.

Il faut retirer chaque prédicteur à tour de rôle pour déterminer lequel est indispensable.

Ce sera le rôle des tests `drop-one`.

---

## 40.2 Validité ou type de tâche ?

Le modèle utilise une opposition binaire valide/invalide.

Il faut vérifier si les quatre formes MP, MT, AC et DA décrivent mieux les données.

---

## 40.3 Robustesse à l’effet plafond

La variable dépendante possède beaucoup de valeurs à 100.

Il faut vérifier si l’effet d’entropie et les autres résultats subsistent sous une autre façon de traiter le plafond.

---

## 40.4 Causalité

Le modèle reste observationnel pour plusieurs prédicteurs construits :

- entropie ;
- précision ;
- nombre de modèles.

Il estime des associations conditionnelles, pas des mécanismes causaux prouvés.

---

# 41. Limites méthodologiques

## 41.1 Prédicteurs construits à partir des mêmes données

`subject_accuracy` et `item_entropy` utilisent les réponses analysées.

Ils peuvent capturer des structures propres à cet échantillon.

---

## 41.2 Intercepts aléatoires seulement

Le modèle n’autorise pas l’effet de l’entropie ou de la séquence à varier entre participants.

---

## 41.3 Linéarité

Chaque prédicteur continu est représenté par une pente constante.

Une relation courbe ne serait pas détectée correctement.

---

## 41.4 Colinéarités modérées

La précision et le nombre moyen de modèles sont corrélés.

La validité et la composante intra-individuelle sont aussi liées.

Cela peut élargir les erreurs-types et rendre certains coefficients sensibles au retrait d’une variable.

---

## 41.5 Nombre de modèles défini par type de tâche

La variable MReasoner n’est pas propre à chaque item exact.

Elle peut donc être difficile à séparer des caractéristiques des quatre formes logiques.

---

# 42. Lien avec les étapes précédentes

## 42.1 Après le modèle nul

Nous savions que les participants expliquaient une grande part de la variance.

## 42.2 Après le contrôle

Nous savions que Standard était associé à une confiance supérieure et que la confiance diminuait légèrement.

## 42.3 Après le modèle cognitif

Nous savons désormais que :

- le désaccord empirique entre réponses est fortement associé à la confiance ;
- la précision générale ne l’est pas clairement ;
- les variables MReasoner montrent des directions négatives, mais des effets plus faibles ;
- la validité n’apporte pas de relation claire ;
- une grande partie de la variance item est expliquée par les prédicteurs cognitifs, principalement l’entropie.

---

# 43. Pourquoi poursuivre avec les analyses de sensibilité

Plusieurs décisions du modèle pourraient influencer les résultats :

```text
utiliser validity_binary
plutôt que task_type

inclure tous les prédicteurs
même non significatifs

utiliser trois simulations MReasoner

traiter confidence comme une variable linéaire
malgré le plafond
```

L’étape suivante devra donc tester la robustesse de nos conclusions.

Le script sera :

```text
fit_sensitivity_mixed_model_E1.py
```

Il répondra notamment à trois questions :

1. L’entropie porte-t-elle réellement l’amélioration globale ?
2. La validité apporte-t-elle quelque chose après les autres prédicteurs ?
3. Le type détaillé de tâche est-il plus informatif que la validité ?

---

# 45. Bilan pédagogique

Le script `fit_cognitive_mixed_model_E1.py` a permis de :

1. conserver le modèle de contrôle comme base ;
2. ajouter cinq prédicteurs cognitifs ;
3. distinguer les niveaux participant, item et participant × tâche ;
4. standardiser les prédicteurs continus ;
5. interpréter leurs coefficients par augmentation d’un écart-type ;
6. décomposer le nombre de modèles en effets interindividuel et intra-individuel ;
7. vérifier les corrélations entre prédicteurs ;
8. ajuster les modèles nul, contrôle et cognitif en ML ;
9. comparer correctement leurs effets fixes ;
10. ajuster le modèle cognitif final en REML ;
11. montrer une forte amélioration globale par rapport au contrôle ;
12. constater que l’entropie est fortement et négativement associée à la confiance ;
13. constater que la précision moyenne n’est pas clairement associée à la confiance ;
14. ne pas détecter d’effet interindividuel fiable du nombre de modèles ;
15. observer une direction intra-individuelle négative ;
16. ne pas détecter d’effet clair de validité ;
17. expliquer une grande partie de la variance entre items ;
18. augmenter le \(R^2\) marginal d’environ 1,85 point de pourcentage ;
19. vérifier ensuite la stabilité des coefficients avec vingt simulations ;
20. identifier un faible effet intra-individuel dans le modèle final parcimonieux n20.

La conclusion centrale est :

> L’entropie empirique des items constitue le prédicteur cognitif principal et le plus robuste de la confiance. Les variables MReasoner présentent des relations plus faibles : aucun effet interindividuel clair n’est observé, tandis qu’un faible effet intra-individuel négatif apparaît dans la spécification finale fondée sur vingt simulations.

# Étape 6 — Analyses de sensibilité avec `fit_sensitivity_mixed_model_E1.py`

## Sommaire

1. [Rôle de cette étape](#1-rôle-de-cette-étape)
2. [Qu’est-ce qu’une analyse de sensibilité ?](#2-quest-ce-quune-analyse-de-sensibilité)
3. [Pourquoi cette étape vient après le modèle cognitif ?](#3-pourquoi-cette-étape-vient-après-le-modèle-cognitif)
4. [Les questions scientifiques examinées](#4-les-questions-scientifiques-examinées)
5. [Les quatre modèles principaux](#5-les-quatre-modèles-principaux)
6. [Pourquoi comparer validité et type de tâche ?](#6-pourquoi-comparer-validité-et-type-de-tâche)
7. [Pourquoi ne jamais inclure simultanément validité et type de tâche ?](#7-pourquoi-ne-jamais-inclure-simultanément-validité-et-type-de-tâche)
8. [Qu’est-ce qu’un modèle parcimonieux ?](#8-quest-ce-quun-modèle-parcimonieux)
9. [Qu’est-ce qu’un test `drop-one` ?](#9-quest-ce-quun-test-drop-one)
10. [Pourquoi ne pas se contenter des valeurs p du modèle complet ?](#10-pourquoi-ne-pas-se-contenter-des-valeurs-p-du-modèle-complet)
11. [Organisation générale du script](#11-organisation-générale-du-script)
12. [Configuration des termes et des formules](#12-configuration-des-termes-et-des-formules)
13. [Préparation des données](#13-préparation-des-données)
14. [Fonction de construction des formules](#14-fonction-de-construction-des-formules)
15. [Fonction de construction du modèle mixte](#15-fonction-de-construction-du-modèle-mixte)
16. [Fonction d’ajustement et convergence](#16-fonction-dajustement-et-convergence)
17. [Ajustements principaux en ML](#17-ajustements-principaux-en-ml)
18. [Ajustements finaux en REML](#18-ajustements-finaux-en-reml)
19. [Tests globaux](#19-tests-globaux)
20. [Contrôle contre modèle cognitif sans validité](#20-contrôle-contre-modèle-cognitif-sans-validité)
21. [Ajout de la validité](#21-ajout-de-la-validité)
22. [Ajout du type de tâche](#22-ajout-du-type-de-tâche)
23. [Comparaison entre validité et type de tâche](#23-comparaison-entre-validité-et-type-de-tâche)
24. [Tests `drop-one` détaillés](#24-tests-drop-one-détaillés)
25. [Retrait de la précision individuelle](#25-retrait-de-la-précision-individuelle)
26. [Retrait de l’entropie](#26-retrait-de-lentropie)
27. [Retrait du nombre moyen de modèles](#27-retrait-du-nombre-moyen-de-modèles)
28. [Retrait de la composante intra-individuelle](#28-retrait-de-la-composante-intra-individuelle)
29. [Retrait de la validité](#29-retrait-de-la-validité)
30. [Pourquoi une différence négative d’AIC favorise parfois le modèle réduit ?](#30-pourquoi-une-différence-négative-daic-favorise-parfois-le-modèle-réduit)
31. [Le modèle avec `task_type`](#31-le-modèle-avec-task_type)
32. [Interprétation des coefficients de type de tâche](#32-interprétation-des-coefficients-de-type-de-tâche)
33. [Robustesse de l’effet d’entropie](#33-robustesse-de-leffet-dentropie)
34. [Composantes de variance](#34-composantes-de-variance)
35. [Les fichiers générés](#35-les-fichiers-générés)
36. [Ce que cette étape établit](#36-ce-que-cette-étape-établit)
37. [Ce qu’elle ne permet pas encore de conclure](#37-ce-quelle-ne-permet-pas-encore-de-conclure)
38. [Limites de la sélection de modèle](#38-limites-de-la-sélection-de-modèle)
39. [Lien avec les étapes précédentes](#39-lien-avec-les-étapes-précédentes)
40. [Pourquoi passer ensuite à l’analyse du plafond](#40-pourquoi-passer-ensuite-à-lanalyse-du-plafond)
41. [Bilan pédagogique](#41-bilan-pédagogique)

---

# 1. Rôle de cette étape

À l’étape 5, nous avons ajusté un modèle cognitif contenant :

```text
condition
sequence_c10
subject_accuracy_z
item_entropy_z
subject_mean_models_z
models_within_subject_z
validity_binary
```

Ce modèle améliorait fortement le modèle de contrôle :

\[
\chi^2(5)=74{,}30,\qquad p<.001
\]

Mais cette amélioration globale ne permettait pas encore de répondre à plusieurs questions :

1. Les cinq prédicteurs cognitifs contribuent-ils tous à l’amélioration ?
2. L’amélioration vient-elle surtout d’un seul prédicteur ?
3. La validité apporte-t-elle réellement une information supplémentaire ?
4. Le type détaillé de tâche est-il plus informatif que la simple opposition valide–invalide ?
5. L’effet d’entropie reste-t-il présent lorsque la structure logique est codée différemment ?

Le script :

```text
fit_sensitivity_mixed_model_E1.py
```

a été construit pour répondre à ces questions.

---

# 2. Qu’est-ce qu’une analyse de sensibilité ?

## 2.1 Définition

Une **analyse de sensibilité** consiste à modifier raisonnablement certains choix de modélisation afin de vérifier si les conclusions restent stables.

Elle répond à la question :

> Notre résultat dépend-il fortement d’une décision particulière du chercheur ?

---

## 2.2 Exemple simple

Supposons que l’on trouve une relation entre sommeil et performance uniquement lorsque :

- les personnes dormant moins de 5 heures sont supprimées ;
- une variable particulière est incluse ;
- un autre type de codage est exclu.

Le résultat serait sensible aux choix d’analyse.

À l’inverse, s’il reste similaire sous plusieurs décisions raisonnables, il est plus robuste.

---

## 2.3 Analogie avec un objet physique

On peut comparer une analyse à une table.

```text
Une table robuste :
elle reste stable si on la pousse légèrement.

Une table fragile :
elle tombe dès qu’on modifie un peu sa position.
```

L’analyse de sensibilité « pousse » raisonnablement le modèle pour voir si les conclusions résistent.

---

## 2.4 Dans notre projet

Nous avons modifié :

- la présence ou l’absence de la validité ;
- le codage binaire ou détaillé de la structure logique ;
- la présence individuelle de chaque prédicteur cognitif.

Nous n’avons pas modifié arbitrairement les données pour obtenir un résultat souhaité. Chaque variante répondait à une question méthodologique précise.

---

# 3. Pourquoi cette étape vient après le modèle cognitif ?

Une analyse de sensibilité a besoin d’un modèle de référence.

Avant l’étape 5, nous ne savions pas encore quel modèle cognitif complet examiner.

L’ordre logique est donc :

```text
1. Construire un modèle principal théoriquement motivé
2. Observer ses résultats
3. Tester sa dépendance aux choix de spécification
```

Si nous avions commencé par des dizaines de variantes sans modèle principal, l’analyse aurait été difficile à organiser et aurait risqué de devenir une recherche opportuniste de résultats significatifs.

---

## 3.1 Modèle principal et modèles de sensibilité

Le modèle principal initial répondait à une question théorique générale.

Les modèles de sensibilité répondent ensuite à des questions comme :

```text
Que se passe-t-il si l’on retire la validité ?
Que se passe-t-il si l’on utilise les quatre types de tâches ?
Que se passe-t-il si l’on retire l’entropie ?
```

Ils ne remplacent pas rétroactivement le raisonnement initial. Ils évaluent sa robustesse.

---

# 4. Les questions scientifiques examinées

## 4.1 Apport global des quatre prédicteurs continus

> La précision, l’entropie et les deux composantes MReasoner améliorent-elles ensemble le modèle de contrôle, même sans validité ?

---

## 4.2 Apport propre de la validité

> Après les autres prédicteurs, la distinction valide–invalide améliore-t-elle encore le modèle ?

---

## 4.3 Apport propre du type de tâche

> Les différences entre MP, MT, AC et DA améliorent-elles le modèle après les autres prédicteurs ?

---

## 4.4 Contribution de chaque prédicteur

> Quel prédicteur dégrade réellement le modèle lorsqu’on le retire ?

---

## 4.5 Robustesse de l’entropie

> Son coefficient reste-t-il négatif lorsque l’on remplace la validité par le type détaillé de tâche ?

---

# 5. Les quatre modèles principaux

Le script a ajusté quatre modèles ML principaux.

---

## 5.1 Modèle de contrôle

```text
confidence ~
    condition
    + sequence_c10
```

Il contient seulement les contrôles.

---

## 5.2 Modèle cognitif sans validité

```text
confidence ~
    condition
    + sequence_c10
    + subject_accuracy_z
    + item_entropy_z
    + subject_mean_models_z
    + models_within_subject_z
```

Il teste les quatre prédicteurs continus sans ajouter la structure logique binaire.

---

## 5.3 Modèle cognitif avec validité

```text
confidence ~
    condition
    + sequence_c10
    + subject_accuracy_z
    + item_entropy_z
    + subject_mean_models_z
    + models_within_subject_z
    + validity_binary
```

Il correspond au modèle principal de l’étape 5.

---

## 5.4 Modèle cognitif avec type de tâche

```text
confidence ~
    condition
    + sequence_c10
    + subject_accuracy_z
    + item_entropy_z
    + subject_mean_models_z
    + models_within_subject_z
    + C(task_type, Treatment(reference='AC'))
```

Il remplace la validité par les quatre catégories logiques.

---

## 5.5 Structure aléatoire commune

Tous ces modèles conservent :

```text
intercept aléatoire participant
intercept aléatoire item
```

Cette constance permet de comparer les parties fixes sans modifier simultanément la structure de regroupement.

---

# 6. Pourquoi comparer validité et type de tâche ?

La variable de validité résume les tâches en deux catégories :

```text
Valides   : MP, MT
Invalides : AC, DA
```

Mais cette simplification pourrait cacher des différences plus fines.

---

## 6.1 Exemple

Il est possible que :

```text
MP ait une confiance élevée
MT ait une confiance faible
```

Dans ce cas, leur moyenne commune « valide » pourrait masquer une différence.

De même :

```text
AC et DA
```

peuvent produire des niveaux de confiance différents.

---

## 6.2 Modèle avec validité

Il estime un seul contraste :

\[
\text{moyenne MP/MT}
-
\text{moyenne AC/DA}
\]

---

## 6.3 Modèle avec type de tâche

Il estime trois contrastes par rapport à une référence.

Avec AC comme référence :

```text
DA − AC
MP − AC
MT − AC
```

Ce codage est plus flexible.

---

## 6.4 Coût de cette flexibilité

Le modèle avec validité ajoute un paramètre.

Le modèle avec quatre tâches ajoute trois paramètres.

Il peut donc s’ajuster légèrement mieux simplement parce qu’il est plus flexible.

C’est pourquoi nous devons examiner :

- le test du rapport de vraisemblance par rapport au modèle sans structure logique ;
- l’AIC ;
- le BIC.

---

# 7. Pourquoi ne jamais inclure simultanément validité et type de tâche ?

## 7.1 Relation déterministe

Nous avons :

| Tâche | Validité |
|---|---|
| AC | 0 |
| DA | 0 |
| MP | 1 |
| MT | 1 |

La validité peut être reconstruite exactement à partir de `task_type`.

---

## 7.2 Dépendance linéaire

Si l’on crée les indicateurs de tâches, la validité peut être écrite comme :

\[
\text{validity}
=
I(\text{MP})+I(\text{MT})
\]

où \(I\) est une variable indicatrice.

La colonne de validité est donc une combinaison exacte des colonnes MP et MT.

---

## 7.3 Conséquence

Le modèle ne peut pas séparer un effet indépendant de validité d’un ensemble complet d’effets de tâche.

Cela produit une **colinéarité parfaite**.

La matrice de conception perd son rang complet.

---

## 7.4 Qu’est-ce que le rang d’une matrice ?

Le **rang** correspond au nombre de colonnes apportant une information indépendante.

Supposons trois colonnes :

```text
A
B
C = A + B
```

La troisième n’ajoute aucune information nouvelle.

La matrice possède trois colonnes, mais seulement deux directions indépendantes.

---

## 7.5 Conséquences numériques possibles

Inclure simultanément les deux variables pourrait produire :

- une matrice singulière ;
- des coefficients non identifiables ;
- des erreurs-types énormes ;
- une colonne automatiquement supprimée ;
- des résultats dépendant du codage.

---

## 7.6 Stratégie choisie

Nous avons ajusté séparément :

```text
modèle avec validité
```

et :

```text
modèle avec type de tâche
```

C’est la stratégie correcte.

---

# 8. Qu’est-ce qu’un modèle parcimonieux ?

## 8.1 Définition

Un modèle **parcimonieux** explique les données avec aussi peu de paramètres que nécessaire.

Il ne s’agit pas toujours du modèle ayant le moins de variables possible.

Il s’agit du meilleur compromis entre :

- fidélité aux données ;
- simplicité ;
- interprétabilité ;
- stabilité.

---

## 8.2 Analogie avec une carte

Une carte contenant chaque pierre et chaque arbre serait extrêmement détaillée, mais difficile à lire.

Une carte ne montrant que deux villes serait trop simple.

Une bonne carte contient les informations nécessaires à son usage sans surcharge inutile.

Un modèle parcimonieux suit la même logique.

---

## 8.3 Pourquoi rechercher la parcimonie ?

Ajouter une variable inutile peut :

- augmenter l’incertitude des autres coefficients ;
- compliquer l’interprétation ;
- produire du surajustement ;
- diminuer la généralisation ;
- donner l’illusion d’une explication plus riche.

---

## 8.4 Attention

Une variable théoriquement importante peut être conservée même si sa valeur p est supérieure à 0,05.

La parcimonie ne signifie pas :

```text
supprimer automatiquement chaque variable non significative
```

La décision dépend :

- de la théorie ;
- du plan expérimental ;
- de l’objectif confirmatoire ou exploratoire ;
- de la comparaison globale des modèles.

---

# 9. Qu’est-ce qu’un test `drop-one` ?

`drop-one` signifie :

```text
retirer une variable à la fois
```

Le principe est :

1. partir du modèle complet ;
2. retirer un prédicteur ;
3. réajuster le modèle réduit ;
4. comparer le modèle réduit au modèle complet.

---

## 9.1 Exemple

Modèle complet :

```text
contrôles
+ précision
+ entropie
+ modèles moyens
+ modèles intra
+ validité
```

Modèle sans entropie :

```text
contrôles
+ précision
+ modèles moyens
+ modèles intra
+ validité
```

Si retirer l’entropie dégrade fortement l’ajustement, elle apporte une information importante.

---

## 9.2 Pourquoi réajuster le modèle ?

Il ne suffit pas de mettre mentalement le coefficient à zéro dans le tableau final.

Lorsque l’on retire une variable, les autres coefficients et variances sont réestimés.

Ils peuvent absorber une partie de l’information laissée disponible.

Le modèle réduit doit donc être réellement ajusté.

---

# 10. Pourquoi ne pas se contenter des valeurs p du modèle complet ?

Le test individuel d’un coefficient et le test de retrait répondent à des questions proches, mais ne sont pas strictement identiques dans tous les contextes.

---

## 10.1 Test de Wald

La valeur p affichée dans le résumé utilise un test de Wald :

\[
z=
\frac{\widehat\beta}{SE(\widehat\beta)}
\]

Il repose sur une approximation locale autour de l’estimation.

---

## 10.2 Test du rapport de vraisemblance

Le test `drop-one` compare les vraisemblances maximales de deux modèles complètement réajustés.

Il mesure la dégradation globale provoquée par la contrainte :

\[
\beta=0
\]

---

## 10.3 Analogie

Le test de Wald examine la pente locale du terrain autour du sommet.

Le test du rapport de vraisemblance compare directement la hauteur des meilleurs sommets accessibles avec et sans la variable.

Les deux approches donnent souvent des résultats proches, mais elles ne sont pas identiques.

---

## 10.4 Avantage supplémentaire

Le `drop-one` fournit aussi :

- une variation d’AIC ;
- une comparaison globale ;
- une méthode applicable à un facteur possédant plusieurs coefficients, comme `task_type`.

---

# 11. Organisation générale du script

Le script suivait cette structure :

```text
1. Charger les données
2. Vérifier les colonnes
3. Construire sequence_c10
4. Standardiser les prédicteurs
5. Définir les termes des modèles
6. Ajuster le contrôle ML
7. Ajuster le cognitif sans validité ML
8. Ajuster le cognitif avec validité ML
9. Ajuster le cognitif avec task_type ML
10. Ajuster les modèles avec validité et tâche en REML
11. Effectuer les tests globaux
12. Retirer chaque prédicteur à tour de rôle
13. Comparer AIC, BIC et variances
14. Extraire les coefficients
15. Sauvegarder les résumés
```

---

# 12. Configuration des termes et des formules

Le script définissait des listes de termes.

```python
CONTROL_TERMS = [
    "C(condition, Treatment(reference='Neutral'))",
    "sequence_c10",
]
```

```python
COGNITIVE_TERMS = [
    "subject_accuracy_z",
    "item_entropy_z",
    "subject_mean_models_z",
    "models_within_subject_z",
]
```

---

## 12.1 Pourquoi utiliser des listes ?

Les tests `drop-one` doivent retirer automatiquement un terme.

Une liste permet d’écrire :

```python
reduced_terms = [
    term
    for term in all_terms
    if term != removed_term
]
```

Il serait plus difficile et risqué de réécrire manuellement cinq formules.

---

## 12.2 Formules construites

### Contrôle

```python
CONTROL_FORMULA = make_formula(
    CONTROL_TERMS
)
```

### Cognitif sans validité

```python
BASE_COGNITIVE_FORMULA = make_formula(
    CONTROL_TERMS + COGNITIVE_TERMS
)
```

### Avec validité

```python
VALIDITY_FORMULA = make_formula(
    CONTROL_TERMS
    + COGNITIVE_TERMS
    + ["validity_binary"]
)
```

### Avec type de tâche

```python
TASK_TYPE_FORMULA = make_formula(
    CONTROL_TERMS
    + COGNITIVE_TERMS
    + [
        "C(task_type, "
        "Treatment(reference='AC'))"
    ]
)
```

---

# 13. Préparation des données

Le script rechargeait :

```text
dataset_analysis_E1.csv
```

et reconstruisait :

```text
sequence_c10
subject_accuracy_z
item_entropy_z
subject_mean_models_z
models_within_subject_z
```

---

## 13.1 Pourquoi refaire les transformations ?

Chaque script doit être autonome.

Il ne doit pas dépendre d’un objet laissé en mémoire par le script précédent.

En outre, il enregistre ses propres paramètres de standardisation.

---

## 13.2 Vérification de `task_type`

Le script exigeait exactement :

```text
AC
DA
MP
MT
```

Une catégorie inattendue aurait interrompu l’analyse.

---

# 14. Fonction de construction des formules

La fonction était :

```python
def make_formula(terms):
    return "confidence ~ " + " + ".join(terms)
```

---

## 14.1 `.join(terms)`

Supposons :

```python
terms = [
    "sequence_c10",
    "item_entropy_z",
]
```

Alors :

```python
" + ".join(terms)
```

produit :

```text
sequence_c10 + item_entropy_z
```

La fonction ajoute :

```text
confidence ~
```

et renvoie :

```text
confidence ~ sequence_c10 + item_entropy_z
```

---

## 14.2 Que se passerait-il sans cette fonction ?

Il faudrait écrire manuellement toutes les formules.

Cela augmenterait le risque :

- d’oublier un terme ;
- de comparer des modèles incohérents ;
- d’introduire une faute d’orthographe ;
- de retirer accidentellement plusieurs variables.

---

# 15. Fonction de construction du modèle mixte

Le code conservait :

```python
VC_FORMULA = {
    "item": "0 + C(item_id)",
    "subject": "0 + C(subject_id)",
}
```

et créait un groupe global artificiel.

La logique est identique aux étapes 3 à 5 :

\[
\mathbf y
=
\mathbf X\boldsymbol\beta
+
\mathbf Z\mathbf b
+
\boldsymbol\varepsilon
\]

Seule la matrice fixe \(\mathbf X\) change selon la formule.

---

# 16. Fonction d’ajustement et convergence

Le script essayait :

```python
OPTIMIZERS = [
    "lbfgs",
    "bfgs",
    "cg",
    "powell",
]
```

Pour chaque modèle, il :

1. construisait l’objet `MixedLM` ;
2. appelait `fit()` ;
3. vérifiait `result.converged` ;
4. affichait les avertissements ;
5. passait à l’optimiseur suivant en cas d’échec.

---

## 16.1 Résultat

Tous les modèles importants ont convergé avec :

```text
lbfgs
```

Aucun changement d’optimiseur n’a été nécessaire.

---

## 16.2 Pourquoi la convergence de chaque modèle est-elle importante ?

Un test de vraisemblance suppose que les deux modèles ont atteint leurs meilleurs ajustements.

Si un modèle réduit ou complet ne converge pas, leur différence de log-vraisemblance peut être trompeuse.

---

# 17. Ajustements principaux en ML

Les quatre modèles principaux ont été ajustés en ML, car leur structure d’effets fixes différait.

| Modèle | Paramètres fixes ajoutés |
|---|---|
| Contrôle | Condition, séquence |
| Cognitif sans validité | + quatre prédicteurs cognitifs |
| Cognitif avec validité | + validité |
| Cognitif avec tâche | + trois contrastes de tâche |

---

# 18. Ajustements finaux en REML

Le script a aussi ajusté en REML :

```text
modèle avec validité
modèle avec type de tâche
```

Ces modèles fournissaient :

- les coefficients finaux de chaque variante ;
- les composantes de variance ;
- les résumés complets.

---

## 18.1 Pourquoi l’AIC et le BIC REML apparaissent-ils vides ?

Dans le tableau :

```text
Cognitive_validity, REML, AIC vide, BIC vide
Cognitive_task_type, REML, AIC vide, BIC vide
```

`statsmodels` ne fournit pas nécessairement ces critères pour les ajustements REML dans ce contexte.

Ce n’est pas une erreur importante.

Les comparaisons de structures fixes doivent de toute façon utiliser les versions ML.

---

# 19. Tests globaux

Le fichier :

```text
global_likelihood_ratio_tests.csv
```

contenait :

| Comparaison | LR | df | p |
|---|---:|---:|---:|
| Contrôle vs cognitif sans validité | 73,098 | 4 | < .001 |
| Cognitif sans validité vs validité | 1,203 | 1 | .273 |
| Cognitif sans validité vs type de tâche | 3,841 | 3 | .279 |

Ces trois lignes répondent à trois questions distinctes.

---

# 20. Contrôle contre modèle cognitif sans validité

Résultat :

\[
\chi^2(4)=73{,}10
\]

\[
p=5{,}03\times10^{-15}
\]

---

## 20.1 Paramètres ajoutés

Le modèle ajoute :

```text
subject_accuracy_z
item_entropy_z
subject_mean_models_z
models_within_subject_z
```

Donc :

\[
df=4
\]

---

## 20.2 Interprétation

Les quatre prédicteurs considérés conjointement améliorent fortement le modèle de contrôle, même sans validité.

L’amélioration cognitive observée à l’étape 5 ne dépend donc pas de l’inclusion de `validity_binary`.

---

## 20.3 Importance logique

À l’étape 5, nous avions :

\[
\chi^2(5)=74{,}30
\]

pour cinq prédicteurs.

Ici, les quatre prédicteurs sans validité donnent déjà :

\[
\chi^2(4)=73{,}10
\]

La différence est seulement :

\[
74{,}30-73{,}10
\approx1{,}20
\]

Cela annonce que la validité apporte très peu d’amélioration supplémentaire.

---

# 21. Ajout de la validité

Résultat :

\[
\chi^2(1)=1{,}203
\]

\[
p=0{,}273
\]

---

## 21.1 Hypothèse nulle

Le test demande :

\[
H_0:
\beta_{\text{validité}}=0
\]

après prise en compte des autres variables.

---

## 21.2 Interprétation

L’ajout de la validité ne dégrade évidemment pas la log-vraisemblance brute : un modèle plus flexible peut toujours s’ajuster au moins aussi bien.

Mais l’amélioration est trop faible relativement au paramètre ajouté pour être distinguée d’une fluctuation attendue.

---

## 21.3 AIC

Modèle sans validité :

\[
AIC=77264{,}22
\]

Avec validité :

\[
AIC=77265{,}02
\]

L’AIC augmente de :

\[
0{,}797
\]

Un AIC plus élevé est moins favorable.

Le petit gain de vraisemblance ne compense donc pas la pénalité du paramètre supplémentaire.

---

## 21.4 BIC

Sans validité :

\[
BIC=77335{,}30
\]

Avec validité :

\[
BIC=77343{,}20
\]

La différence est :

\[
7{,}90
\]

Le BIC préfère plus nettement le modèle sans validité.

---

## 21.5 Conclusion

La validité n’apporte pas une information supplémentaire suffisante pour justifier sa complexité dans le modèle final parcimonieux.

---

## 21.6 Nuance théorique

Si la validité constituait une hypothèse confirmatoire définie avant l’analyse, il serait possible de la conserver et de rapporter son effet non détecté.

La décision de retrait ne doit pas être présentée comme une preuve que la validité est scientifiquement inutile.

Elle signifie :

> Dans ces données et après les autres prédicteurs, elle n’améliore pas suffisamment l’ajustement.

---

# 22. Ajout du type de tâche

Résultat :

\[
\chi^2(3)=3{,}841
\]

\[
p=0{,}279
\]

---

## 22.1 Pourquoi trois degrés de liberté ?

`task_type` possède quatre catégories :

```text
AC
DA
MP
MT
```

Avec AC comme référence, il faut trois coefficients :

```text
DA − AC
MP − AC
MT − AC
```

Donc :

\[
df=4-1=3
\]

---

## 22.2 Interprétation

Après prise en compte :

- de la condition ;
- de la séquence ;
- de la précision ;
- de l’entropie ;
- des deux composantes MReasoner ;

les différences supplémentaires entre les quatre formes de tâche n’améliorent pas clairement le modèle.

---

## 22.3 AIC et BIC

Avec type de tâche :

\[
AIC=77266{,}38
\]

\[
BIC=77358{,}78
\]

Les deux valeurs sont supérieures à celles du modèle cognitif sans validité :

\[
AIC=77264{,}22
\]

\[
BIC=77335{,}30
\]

Le modèle plus simple est préféré.

---

# 23. Comparaison entre validité et type de tâche

Les modèles :

```text
avec validité
```

et :

```text
avec task_type
```

ne sont pas directement emboîtés l’un dans l’autre.

---

## 23.1 Pourquoi ?

Le modèle de validité impose une structure particulière :

```text
MP et MT ont le même effet de validité
AC et DA ont le même effet de référence
```

Le modèle de tâche estime des différences séparées.

Le modèle de validité peut être vu comme une contrainte particulière dans l’espace des effets de tâche, mais la comparaison directe telle qu’elle est codée n’est pas le simple ajout ou retrait d’un ensemble évident de coefficients entre ces deux formules.

Nous avons donc surtout utilisé :

- leur comparaison commune avec le modèle sans structure logique ;
- l’AIC ;
- le BIC ;
- les coefficients.

---

## 23.2 Résultats

| Modèle | AIC | BIC |
|---|---:|---:|
| Sans validité ni tâche | **77264,22** | **77335,30** |
| Avec validité | 77265,02 | 77343,20 |
| Avec tâche | 77266,38 | 77358,78 |

Le modèle sans ces termes présente le meilleur compromis.

---

# 24. Tests `drop-one` détaillés

Le fichier :

```text
drop_one_tests.csv
```

contenait :

| Prédicteur retiré | LR | p | ΔAIC réduit − complet |
|---|---:|---:|---:|
| Précision | 0,044 | .834 | −1,956 |
| Entropie | 61,140 | < .001 | +59,140 |
| Modèles moyens | 1,460 | .227 | −0,540 |
| Modèles intra | 2,214 | .137 | +0,214 |
| Validité | 1,203 | .273 | −0,797 |

Le modèle « complet » de cette table était le modèle cognitif avec validité.

---

## 24.1 Logique de calcul

Pour chaque prédicteur :

```text
1. Retirer le terme
2. Réajuster en ML
3. Calculer :
   LR = 2(logL_complet − logL_réduit)
4. Calculer la valeur p
5. Comparer les AIC
```

---

# 25. Retrait de la précision individuelle

Résultat :

\[
LR=0{,}044
\]

\[
p=0{,}834
\]

\[
\Delta AIC=-1{,}956
\]

---

## 25.1 Interprétation du LR

Le modèle sans précision possède presque la même vraisemblance que le modèle complet.

La précision n’apporte pratiquement aucune amélioration conditionnelle.

---

## 25.2 Interprétation de l’AIC

La différence est définie comme :

\[
AIC_{\text{réduit}}
-
AIC_{\text{complet}}
\]

Ici :

\[
-1{,}956
\]

Le modèle réduit a donc un AIC inférieur d’environ 1,96 point.

L’AIC préfère légèrement le modèle sans précision.

---

## 25.3 Pourquoi ne pas forcément la supprimer immédiatement ?

La précision est importante pour la question métacognitive.

La conserver peut être utile comme contrôle théorique, même si son apport prédictif est faible.

Le modèle final de notre rapport l’a conservée pour montrer explicitement que l’effet MReasoner et l’entropie sont estimés après prise en compte de la performance générale.

---

# 26. Retrait de l’entropie

Résultat :

\[
LR=61{,}140
\]

\[
p=5{,}31\times10^{-15}
\]

\[
\Delta AIC=59{,}140
\]

---

## 26.1 Interprétation

Retirer l’entropie dégrade massivement l’ajustement.

Le modèle sans entropie possède un AIC supérieur de plus de 59 points.

C’est une différence très importante.

---

## 26.2 Pourquoi le LR et le ΔAIC diffèrent-ils de 2 ?

Pour un retrait d’un seul paramètre :

\[
\Delta AIC
=
AIC_{\text{réduit}}
-
AIC_{\text{complet}}
\]

Avec :

\[
AIC=-2\log L+2k
\]

Le modèle réduit possède un paramètre de moins.

Nous obtenons :

\[
\Delta AIC
=
2(\log L_{\text{complet}}-\log L_{\text{réduit}})
-2
\]

Donc :

\[
\Delta AIC=LR-2
\]

Ici :

\[
61{,}140-2
=
59{,}140
\]

Cela correspond exactement au résultat.

---

## 26.3 Conclusion

L’entropie porte l’essentiel de l’amélioration du modèle cognitif.

---

# 27. Retrait du nombre moyen de modèles

Résultat :

\[
LR=1{,}460
\]

\[
p=0{,}227
\]

\[
\Delta AIC=-0{,}540
\]

---

## 27.1 Interprétation

Retirer `subject_mean_models_z` ne dégrade pas significativement le modèle.

L’AIC préfère légèrement le modèle réduit.

---

## 27.2 Pourquoi le conserver dans le rapport final ?

Cette variable représente l’effet interindividuel théorique de MReasoner.

La retirer empêcherait de distinguer correctement :

- l’effet entre participants ;
- l’effet à l’intérieur des participants.

Dans une décomposition between–within, il est souvent préférable de conserver les deux composantes lorsqu’on souhaite interpréter l’effet intra-individuel.

---

## 27.3 Risque si on la retire

Si `subject_mean_models_z` est supprimée mais `models_within_subject_z` conservée, l’interprétation intra-individuelle reste mathématiquement possible parce que cette dernière est centrée par participant.

Cependant, le modèle ne documenterait plus explicitement la relation interindividuelle et pourrait être moins complet théoriquement.

---

# 28. Retrait de la composante intra-individuelle

Résultat :

\[
LR=2{,}214
\]

\[
p=0{,}137
\]

\[
\Delta AIC=0{,}214
\]

---

## 28.1 Interprétation

Retirer cette composante dégrade légèrement la vraisemblance, mais pas suffisamment pour atteindre le seuil conventionnel.

---

## 28.2 AIC presque identique

Le modèle réduit possède un AIC supérieur de seulement :

\[
0{,}214
\]

Cette différence est négligeable.

Les deux modèles ont donc un soutien AIC pratiquement équivalent.

---

## 28.3 Direction théorique

Le coefficient complet était négatif :

\[
-0{,}366
\]

L’information va dans le sens de l’hypothèse, mais reste insuffisante dans cette spécification n3 avec validité.

---

# 29. Retrait de la validité

Résultat :

\[
LR=1{,}203
\]

\[
p=0{,}273
\]

\[
\Delta AIC=-0{,}797
\]

---

## 29.1 Interprétation

Le modèle sans validité est plus parcimonieux et légèrement préféré par l’AIC.

Le BIC le préfère encore davantage.

---

# 30. Pourquoi une différence négative d’AIC favorise parfois le modèle réduit ?

La colonne était :

```text
delta_aic_reduced_minus_full
```

Elle calcule :

\[
AIC_{\text{réduit}}
-
AIC_{\text{complet}}
\]

---

## 30.1 Si la valeur est positive

\[
AIC_{\text{réduit}}
>
AIC_{\text{complet}}
\]

Le modèle complet est préférable.

Exemple de l’entropie :

\[
+59{,}14
\]

---

## 30.2 Si la valeur est négative

\[
AIC_{\text{réduit}}
<
AIC_{\text{complet}}
\]

Le modèle réduit est préférable.

Exemple de la précision :

\[
-1{,}956
\]

---

## 30.3 Si la valeur est proche de zéro

Les modèles sont pratiquement équivalents selon l’AIC.

Exemple de la composante intra-individuelle :

\[
+0{,}214
\]

---

# 31. Le modèle avec `task_type`

Le modèle REML alternatif utilisait AC comme référence.

Résultats :

| Paramètre | β | SE | p | IC 95 % |
|---|---:|---:|---:|---:|
| DA − AC | −0,600 | 0,845 | .478 | [−2,256 ; 1,057] |
| MP − AC | 1,102 | 0,803 | .170 | [−0,472 ; 2,676] |
| MT − AC | −0,088 | 0,785 | .911 | [−1,625 ; 1,450] |

---

## 31.1 Pourquoi AC est-elle la référence ?

Le choix était explicite :

```python
Treatment(reference='AC')
```

AC sert seulement de point de comparaison.

Cela ne signifie pas qu’AC est théoriquement la catégorie la plus importante.

---

## 31.2 Intercept

L’intercept du modèle task type était :

\[
73{,}017
\]

Il correspond à :

- Neutral ;
- AC ;
- milieu de l’expérience ;
- prédicteurs standardisés à zéro ;
- effets aléatoires nuls.

---

# 32. Interprétation des coefficients de type de tâche

## 32.1 DA contre AC

\[
\beta=-0{,}600
\]

Le modèle estime environ 0,6 point de confiance en moins pour DA que pour AC.

Mais :

\[
p=0{,}478
\]

et l’intervalle contient zéro.

Aucune différence claire n’est détectée.

---

## 32.2 MP contre AC

\[
\beta=1{,}102
\]

MP est estimé environ 1,1 point au-dessus d’AC.

Mais :

\[
p=0{,}170
\]

L’intervalle :

\[
[-0{,}472\,;\,2{,}676]
\]

contient zéro.

---

## 32.3 MT contre AC

\[
\beta=-0{,}088
\]

L’estimation est presque nulle :

\[
p=0{,}911
\]

---

## 32.4 Attention aux comparaisons non directement affichées

Le tableau donne :

```text
DA − AC
MP − AC
MT − AC
```

Il ne teste pas directement :

```text
MP − MT
DA − MP
DA − MT
```

Ces différences pourraient être calculées par des contrastes supplémentaires.

Mais le test global de `task_type` n’était pas significatif, ce qui ne motivait pas fortement de nombreuses comparaisons post hoc.

---

# 33. Robustesse de l’effet d’entropie

Dans le modèle avec validité :

\[
\beta_{\text{entropie}}
=
-2{,}437
\]

Dans le modèle avec type de tâche :

\[
\beta_{\text{entropie}}
=
-2{,}246
\]

Les deux effets sont clairement négatifs :

\[
p<.001
\]

---

## 33.1 Interprétation

L’effet d’entropie ne disparaît pas lorsque l’on remplace la validité par une représentation détaillée des tâches.

Il n’est donc pas simplement une conséquence de l’opposition :

```text
valide contre invalide
```

ou d’une différence moyenne entre les quatre types.

---

## 33.2 Changement d’amplitude

Le coefficient devient légèrement moins négatif :

\[
-2{,}437\rightarrow-2{,}246
\]

Une partie de l’association entre entropie et confiance est partagée avec le type de tâche, mais l’essentiel de l’effet demeure.

---

## 33.3 Conclusion prudente

> À type de tâche contrôlé, les items d’entropie plus élevée restent associés à une confiance plus faible.

Il s’agit d’un résultat de robustesse important.

---

# 34. Composantes de variance

Les modèles ML avaient :

| Modèle | Participant | Item | Résiduelle |
|---|---:|---:|---:|
| Contrôle | 193,734 | 11,874 | 284,750 |
| Cognitif sans validité | 190,969 | 5,124 | 284,701 |
| Cognitif avec validité | 190,896 | 5,053 | 284,696 |
| Cognitif avec tâche | 190,976 | 4,863 | 284,695 |

---

## 34.1 Variance participant

Elle diminue légèrement entre le contrôle et les modèles cognitifs :

\[
193{,}734\rightarrow190{,}9
\]

Les prédicteurs participants expliquent donc une faible part des différences individuelles.

---

## 34.2 Variance item

Elle diminue fortement :

\[
11{,}874\rightarrow environ5
\]

Cela confirme que les variables cognitives, surtout l’entropie, expliquent une grande partie des différences entre items.

---

## 34.3 Validité et tâche changent peu les variances

Entre :

```text
sans validité : item = 5,124
avec validité : item = 5,053
avec tâche     : item = 4,863
```

les réductions supplémentaires sont faibles.

La majorité de l’explication item était déjà apportée par l’entropie et les autres prédicteurs du modèle de base.

---

# 35. Les fichiers générés

Le dossier était :

```text
sensitivity_mixed_model_E1/
```

---

## 35.1 `standardization.csv`

Contient les moyennes et écarts-types utilisés pour standardiser les prédicteurs.

---

## 35.2 `global_likelihood_ratio_tests.csv`

Contient les trois comparaisons globales :

```text
Control vs Cognitive_without_validity
Cognitive_without_validity vs Validity
Cognitive_without_validity vs Task_type
```

---

## 35.3 `drop_one_tests.csv`

Pour chaque prédicteur, il contient :

- la formule réduite ;
- le LR ;
- les degrés de liberté ;
- la valeur p ;
- l’AIC réduit ;
- l’AIC complet ;
- la différence d’AIC ;
- l’optimiseur utilisé.

---

## 35.4 `model_fit_comparison.csv`

Contient :

- le nom du modèle ;
- la méthode ML ou REML ;
- la formule ;
- la convergence ;
- la log-vraisemblance ;
- l’AIC ;
- le BIC ;
- le nombre de paramètres ;
- les trois variances.

---

## 35.5 `validity_model_fixed_effects_REML.csv`

Contient les coefficients du modèle avec validité.

Il reproduit la spécification principale de l’étape 5.

---

## 35.6 `task_type_model_fixed_effects_REML.csv`

Contient les coefficients du modèle alternatif avec les quatre formes de tâche.

---

## 35.7 Résumés texte

```text
validity_model_REML_summary.txt
task_type_model_REML_summary.txt
```

Ils conservent les tableaux complets de `statsmodels`.

---

# 36. Ce que cette étape établit

## 36.1 Les prédicteurs cognitifs améliorent le contrôle sans la validité

\[
\chi^2(4)=73{,}10,\quad p<.001
\]

La validité n’est donc pas responsable de l’amélioration cognitive globale.

---

## 36.2 L’amélioration est essentiellement portée par l’entropie

Son retrait produit :

\[
\chi^2(1)=61{,}14,\quad p<.001
\]

et :

\[
\Delta AIC=59{,}14
\]

Aucun autre retrait ne produit une dégradation comparable.

---

## 36.3 La validité n’améliore pas le modèle

\[
\chi^2(1)=1{,}20,\quad p=.273
\]

Les critères d’information favorisent le modèle sans validité.

---

## 36.4 Le type de tâche n’améliore pas le modèle

\[
\chi^2(3)=3{,}84,\quad p=.279
\]

Aucun contraste par rapport à AC n’est clairement détecté.

---

## 36.5 L’effet d’entropie est robuste à la structure logique

Il reste clairement négatif avec :

- la validité ;
- le type détaillé de tâche.

---

## 36.6 Modèle final parcimonieux retenu

Pour les analyses finales, nous avons retenu :

```text
confidence ~
    condition
    + sequence_c10
    + subject_accuracy_z
    + item_entropy_z
    + subject_mean_models_z
    + models_within_subject_z
    + effets aléatoires participant et item
```

`validity_binary` n’est plus inclus dans cette spécification parcimonieuse.

---

# 37. Ce qu’elle ne permet pas encore de conclure

## 37.1 Le modèle linéaire est-il robuste au plafond ?

Non vérifié à ce stade.

Avec 25,9 % de valeurs égales à 100, il faut examiner si les coefficients sont influencés par la borne.

---

## 37.2 Les variables MReasoner sont-elles stables avec davantage de simulations ?

Pas encore.

Les résultats présentés ici reposent toujours sur trois simulations.

---

## 37.3 La composante intra-individuelle est-elle réellement absente ?

Dans cette spécification :

\[
p=.137
\]

Elle n’est pas clairement détectée.

Mais l’AIC est presque identique avec ou sans elle, et son coefficient est négatif.

Une meilleure estimation MReasoner ou une spécification différente peut modifier sa précision.

---

## 37.4 L’entropie cause-t-elle la confiance ?

Non.

Sa robustesse statistique ne transforme pas l’association en preuve causale.

---

# 38. Limites de la sélection de modèle

## 38.1 Risque de sélection guidée par les données

Si l’on teste de nombreuses variantes puis ne rapporte que celle donnant les plus petites valeurs p, on augmente le risque de résultats opportunistes.

Nous avons limité ce risque en :

- définissant des comparaisons motivées ;
- conservant les contrôles théoriques ;
- rapportant les variantes ;
- distinguant modèle initial et modèle final ;
- réalisant des analyses de robustesse.

---

## 38.2 Le plus petit AIC n’est pas toujours la vérité

L’AIC classe des modèles candidats.

Il ne garantit pas que le meilleur modèle de la liste représente parfaitement le mécanisme réel.

Tous les modèles candidats peuvent être imparfaits.

---

## 38.3 Différences d’AIC faibles

Une différence comme :

\[
0{,}214
\]

est négligeable.

Il ne faut pas déclarer un vainqueur absolu sur cette base.

---

## 38.4 Valeur p non significative

Un test non significatif pour la validité ou la tâche ne prouve pas l’égalité exacte de toutes les formes logiques.

Il indique que l’amélioration supplémentaire n’est pas suffisamment soutenue dans ce modèle.

---

## 38.5 Hiérarchie théorique

Même lorsqu’un effet principal n’est pas significatif, certaines variables peuvent être conservées pour respecter une structure théorique ou permettre l’interprétation d’autres termes.

---

# 39. Lien avec les étapes précédentes

## Après le modèle nul

Nous savions que la confiance était fortement structurée par les participants.

## Après le modèle de contrôle

Nous savions que Standard augmentait la confiance et que la confiance diminuait avec la séquence.

## Après le modèle cognitif

Nous savions que les prédicteurs cognitifs amélioraient le modèle et que l’entropie possédait un fort coefficient négatif.

## Après l’analyse de sensibilité

Nous savons maintenant que :

- l’amélioration ne dépend pas de la validité ;
- le type de tâche n’apporte pas d’amélioration supplémentaire ;
- l’entropie porte l’essentiel du gain ;
- son effet subsiste sous plusieurs codages logiques ;
- le modèle sans validité constitue une spécification finale plus parcimonieuse.

---

# 40. Pourquoi passer ensuite à l’analyse du plafond

Le modèle final parcimonieux reste un modèle linéaire.

Or la confiance est limitée entre 0 et 100 et présente :

\[
2336
\]

valeurs exactement égales à 100, soit :

\[
25{,}89\%
\]

des observations.

Cette concentration peut produire :

- une distribution asymétrique ;
- des résidus non normaux ;
- une relation différente entre prédicteurs et plafond ;
- des effets qui reflètent surtout l’utilisation de 100.

Nous devons donc demander :

> L’effet d’entropie et les autres coefficients subsistent-ils si l’on retire les réponses à 100 ?

Ce sera le rôle de :

```text
fit_ceiling_sensitivity_E1.py
```

Puis nous modéliserons directement :

\[
P(\text{confidence}=100)
\]

dans l’étape suivante.

---

# 41. Bilan pédagogique

Le script `fit_sensitivity_mixed_model_E1.py` a permis de :

1. tester le modèle cognitif sans validité ;
2. montrer que les quatre prédicteurs continus améliorent fortement le contrôle ;
3. isoler l’apport propre de la validité ;
4. montrer que la validité n’améliore pas clairement le modèle ;
5. remplacer la validité par les quatre types de tâches ;
6. montrer que le type détaillé n’améliore pas clairement le modèle ;
7. éviter la colinéarité parfaite entre validité et type de tâche ;
8. effectuer des tests `drop-one` ;
9. réajuster chaque modèle réduit plutôt que de lire seulement les tests de Wald ;
10. montrer que retirer l’entropie dégrade très fortement l’ajustement ;
11. montrer que retirer la précision ne le dégrade pratiquement pas ;
12. montrer que le nombre moyen de modèles n’apporte pas d’amélioration claire ;
13. observer une information faible et incertaine pour la composante intra-individuelle ;
14. comparer les modèles avec AIC, BIC et tests de vraisemblance ;
15. vérifier la robustesse de l’effet d’entropie au codage de la structure logique ;
16. constater que l’entropie explique une grande partie de la variance entre items ;
17. sélectionner un modèle final parcimonieux sans validité ni type de tâche ;
18. préparer l’analyse de sensibilité à l’effet plafond.

La conclusion centrale est :

> L’amélioration du modèle cognitif est presque entièrement portée par l’entropie de l’item. Ni la validité ni le type détaillé de tâche n’apportent une amélioration supplémentaire claire après les autres prédicteurs. L’association négative entre entropie et confiance reste stable sous les différentes spécifications examinées.


# Étape 7 — Analyse de sensibilité à l’effet plafond avec `fit_ceiling_sensitivity_E1.py`

## Sommaire

1. [Rôle de cette étape](#1-rôle-de-cette-étape)
2. [Qu’est-ce qu’un effet plafond ?](#2-quest-ce-quun-effet-plafond)
3. [Pourquoi le plafond pose-t-il un problème statistique ?](#3-pourquoi-le-plafond-pose-t-il-un-problème-statistique)
4. [Pourquoi cette étape arrive-t-elle maintenant ?](#4-pourquoi-cette-étape-arrive-t-elle-maintenant)
5. [Objectif scientifique précis](#5-objectif-scientifique-précis)
6. [Stratégie choisie](#6-stratégie-choisie)
7. [Pourquoi cette analyse ne remplace-t-elle pas le modèle principal ?](#7-pourquoi-cette-analyse-ne-remplace-t-elle-pas-le-modèle-principal)
8. [Les méthodes alternatives](#8-les-méthodes-alternatives)
9. [Formulation du modèle sous le plafond](#9-formulation-du-modèle-sous-le-plafond)
10. [Organisation générale du script](#10-organisation-générale-du-script)
11. [Configuration des fichiers](#11-configuration-des-fichiers)
12. [La formule utilisée](#12-la-formule-utilisée)
13. [Différence avec le modèle cognitif initial](#13-différence-avec-le-modèle-cognitif-initial)
14. [Chargement et vérification des données](#14-chargement-et-vérification-des-données)
15. [Construction de la variable `at_ceiling`](#15-construction-de-la-variable-at_ceiling)
16. [Création du résumé du plafond](#16-création-du-résumé-du-plafond)
17. [Sélection des observations inférieures à 100](#17-sélection-des-observations-inférieures-à-100)
18. [Conséquences de cette sélection](#18-conséquences-de-cette-sélection)
19. [Recentrage de la séquence](#19-recentrage-de-la-séquence)
20. [Standardisation des prédicteurs](#20-standardisation-des-prédicteurs)
21. [Construction du modèle mixte](#21-construction-du-modèle-mixte)
22. [Ajustements ML et REML](#22-ajustements-ml-et-reml)
23. [Extraction des effets fixes](#23-extraction-des-effets-fixes)
24. [Résultats obtenus](#24-résultats-obtenus)
25. [Interprétation de l’intercept](#25-interprétation-de-lintercept)
26. [Effet de la condition](#26-effet-de-la-condition)
27. [Effet de la séquence](#27-effet-de-la-séquence)
28. [Effet de la précision](#28-effet-de-la-précision)
29. [Effet de l’entropie](#29-effet-de-lentropie)
30. [Effet interindividuel des modèles mentaux](#30-effet-interindividuel-des-modèles-mentaux)
31. [Effet intra-individuel des modèles mentaux](#31-effet-intra-individuel-des-modèles-mentaux)
32. [Composantes de variance](#32-composantes-de-variance)
33. [Comparaison avec le modèle principal](#33-comparaison-avec-le-modèle-principal)
34. [Pourquoi condition et séquence disparaissent-elles ?](#34-pourquoi-condition-et-séquence-disparaissent-elles)
35. [Pourquoi l’entropie reste-t-elle importante ?](#35-pourquoi-lentropie-reste-t-elle-importante)
36. [Le cas du nombre moyen de modèles](#36-le-cas-du-nombre-moyen-de-modèles)
37. [Les fichiers générés](#37-les-fichiers-générés)
38. [Ce que cette analyse permet de conclure](#38-ce-que-cette-analyse-permet-de-conclure)
39. [Ce qu’elle ne permet pas de conclure](#39-ce-quelle-ne-permet-pas-de-conclure)
40. [Limites méthodologiques](#40-limites-méthodologiques)
41. [Lien avec les étapes précédentes](#41-lien-avec-les-étapes-précédentes)
42. [Pourquoi passer ensuite au modèle logistique du plafond ?](#42-pourquoi-passer-ensuite-au-modèle-logistique-du-plafond)
43. [Bilan pédagogique](#43-bilan-pédagogique)

---

# 1. Rôle de cette étape

À la fin de l’étape 6, nous avions retenu un modèle cognitif parcimonieux comprenant :

```text
condition
sequence_c10
subject_accuracy_z
item_entropy_z
subject_mean_models_z
models_within_subject_z
```

avec des intercepts aléatoires croisés pour :

```text
participant
item
```

Cependant, un problème important restait à examiner :

```text
2 336 réponses avaient une confiance égale à 100
```

sur :

```text
9 024 observations
```

Cela représente :

\[
\frac{2336}{9024}
\approx0{,}2589
\]

soit :

\[
25{,}89\%
\]

Plus d’une réponse sur quatre se trouvait donc exactement à la borne maximale de l’échelle.

Le script :

```text
fit_ceiling_sensitivity_E1.py
```

a été créé pour vérifier si les conclusions du modèle linéaire principal dépendaient fortement de cette accumulation à 100.

---

# 2. Qu’est-ce qu’un effet plafond ?

## 2.1 Définition

Un **effet plafond** apparaît lorsqu’un nombre important d’observations atteint la valeur maximale possible d’une mesure.

Dans notre expérience :

\[
0\leq\text{confidence}\leq100
\]

La valeur 100 est le plafond.

---

## 2.2 Exemple simple

Supposons que cinq participants souhaitent exprimer les niveaux réels de certitude suivants :

```text
95
100
105
115
130
```

Mais l’échelle s’arrête à 100.

Nous pourrions observer :

```text
95
100
100
100
100
```

Les quatre dernières personnes obtiennent la même valeur observée, même si leur certitude latente pourrait être différente.

---

## 2.3 Valeur latente

Une **variable latente** est une quantité théorique qui n’est pas observée directement.

Par exemple, la certitude psychologique réelle n’est pas nécessairement limitée naturellement à notre codage de 0 à 100.

L’échelle impose simplement que toute réponse maximale soit enregistrée comme :

```text
100
```

Cependant, il faut rester prudent : les participants peuvent aussi utiliser 100 comme une catégorie qualitative signifiant « totalement certain », sans qu’il existe une valeur latente supérieure.

---

## 2.4 Analogie avec une balance

Supposons qu’une balance ne puisse pas afficher plus de 150 kg.

Trois objets de :

```text
150 kg
170 kg
200 kg
```

seraient tous affichés :

```text
150 kg
```

La balance ne permet plus de distinguer les valeurs élevées.

Notre échelle de confiance connaît un problème analogue à sa borne supérieure.

---

# 3. Pourquoi le plafond pose-t-il un problème statistique ?

## 3.1 Asymétrie

Une variable normale peut théoriquement prendre toute valeur réelle :

\[
-\infty<Y<+\infty
\]

Notre confiance est limitée :

\[
0\leq Y\leq100
\]

Lorsque beaucoup de valeurs s’accumulent à 100, la distribution devient asymétrique.

Elle possède souvent :

- une forte concentration à droite ;
- une queue plus longue vers les faibles valeurs ;
- une asymétrie négative.

---

## 3.2 Résidus non normaux

Le modèle linéaire suppose approximativement :

\[
\varepsilon
\sim
\mathcal N(0,\sigma^2)
\]

Mais, près du plafond, les résidus positifs sont limités.

Si la confiance prédite vaut :

\[
95
\]

le résidu positif maximal est :

\[
100-95=5
\]

alors qu’un résidu négatif peut être beaucoup plus grand :

\[
0-95=-95
\]

La distribution des erreurs ne peut donc pas être symétrique autour de zéro dans cette zone.

---

## 3.3 Compression des différences

Deux observations égales à 100 sont traitées comme identiques, même si leur certitude psychologique sous-jacente pourrait différer.

Cela réduit la variation visible parmi les réponses élevées.

---

## 3.4 Effet artificiel sur les coefficients

Supposons que la condition Standard pousse davantage de participants vers 100.

Le modèle linéaire détectera une confiance moyenne plus élevée en Standard.

Mais cet effet pourrait refléter principalement :

```text
une utilisation plus fréquente de la catégorie maximale
```

plutôt qu’une augmentation uniforme sur toute l’échelle.

---

## 3.5 Hétéroscédasticité

L’**hétéroscédasticité** désigne une variance des résidus qui change selon le niveau prédit.

Près de 100, les erreurs positives sont comprimées.

À des niveaux prédits plus faibles, les erreurs peuvent varier dans les deux directions.

La dispersion résiduelle n’est donc pas forcément constante.

---

# 4. Pourquoi cette étape arrive-t-elle maintenant ?

Nous devions d’abord identifier le modèle principal et ses résultats.

Les étapes précédentes ont établi que :

1. la condition Standard augmente la confiance ;
2. la confiance diminue au fil des essais ;
3. l’entropie diminue la confiance ;
4. les variables MReasoner ont des effets plus incertains ;
5. la validité et le type de tâche n’améliorent pas clairement le modèle.

Nous pouvions maintenant demander :

> Ces résultats sont-ils toujours visibles lorsque nous nous concentrons uniquement sur les réponses non maximales ?

Cette question n’aurait pas eu autant de sens avant d’avoir identifié les coefficients à vérifier.

---

# 5. Objectif scientifique précis

L’analyse poursuit deux objectifs.

## 5.1 Vérifier la robustesse des coefficients

Nous voulons comparer :

```text
modèle sur toutes les réponses
```

et :

```text
modèle uniquement sur confidence < 100
```

Si un coefficient conserve :

- le même signe ;
- une amplitude proche ;
- une incertitude comparable ;

il est relativement robuste au plafond.

---

## 5.2 Identifier les effets portés par la valeur 100

Si un effet présent dans l’échantillon complet disparaît sous le plafond, cela suggère qu’il peut être principalement lié à :

```text
la propension à utiliser 100
```

Mais cette conclusion reste provisoire tant que nous n’avons pas modélisé directement cette propension.

---

# 6. Stratégie choisie

Le script réalise les opérations suivantes :

```text
1. Charger les 9 024 observations
2. Créer at_ceiling = 1 si confidence == 100
3. Compter les observations au plafond
4. Conserver uniquement confidence < 100
5. Réajuster le modèle linéaire parcimonieux
6. Comparer les coefficients au modèle principal
```

---

## 6.1 Dataset retenu

Le sous-échantillon contient :

\[
9024-2336=6688
\]

observations.

La proportion conservée est :

\[
\frac{6688}{9024}
\approx0{,}7411
\]

soit :

\[
74{,}11\%
\]

---

# 7. Pourquoi cette analyse ne remplace-t-elle pas le modèle principal ?

Retirer les réponses égales à 100 n’est pas une opération neutre.

Nous sélectionnons les observations en fonction de la variable dépendante :

```text
conserver la ligne si confidence < 100
```

---

## 7.1 Sélection sur le résultat

La variable que nous cherchons à expliquer détermine si l’observation est conservée.

Cela est appelé ici une **sélection sur la variable dépendante**.

---

## 7.2 Exemple

Supposons que Standard augmente fortement la probabilité de répondre 100.

En supprimant les valeurs 100, nous retirons davantage d’observations Standard que Neutral.

Le sous-échantillon Standard restant n’est plus représentatif de toutes les réponses Standard.

---

## 7.3 Analogie médicale

Supposons que nous étudiions la pression artérielle, mais que nous supprimions toutes les valeurs supérieures à 160.

Nous ne pourrions plus utiliser ce sous-échantillon pour décrire toute la population.

Nous pourrions seulement étudier :

> La pression artérielle parmi les personnes dont la mesure est inférieure à 160.

Notre modèle sous le plafond répond de la même manière à une question conditionnelle :

> Quels sont les effets parmi les réponses dont la confiance est inférieure à 100 ?

---

## 7.4 Conséquence

Cette analyse sert à vérifier la robustesse.

Elle ne doit pas remplacer le modèle principal fondé sur l’ensemble des données.

---

# 8. Les méthodes alternatives

Plusieurs méthodes auraient pu être envisagées.

---

## 8.1 Régression Tobit

Un modèle Tobit traite une variable comme une mesure continue censurée.

Il suppose qu’une valeur latente peut dépasser la borne, mais que l’observation est enregistrée à la limite.

### Limite dans notre cas

Nous ne savons pas si une réponse de 100 représente :

- une confiance latente supérieure à 100 censurée ;
- ou un choix réel de la catégorie « totalement certain ».

L’hypothèse Tobit n’est donc pas automatiquement appropriée.

---

## 8.2 Régression bêta

Après division par 100, la confiance serait comprise entre 0 et 1.

Mais la distribution bêta classique exige :

\[
0<Y<1
\]

Elle n’accepte pas les valeurs exactes :

```text
0
1
```

Or nous avons des réponses égales à 0 et beaucoup de réponses égales à 1 après transformation.

---

## 8.3 Modèle bêta gonflé aux bornes

Un modèle plus complexe pourrait distinguer :

- la probabilité de répondre exactement 0 ;
- la probabilité de répondre exactement 1 ;
- la distribution continue entre les deux.

Cette méthode serait théoriquement intéressante, mais plus difficile à implémenter avec des effets aléatoires croisés dans notre environnement Python.

---

## 8.4 Modèle ordinal

La confiance pourrait être considérée comme une variable ordonnée à 101 catégories.

Mais un modèle ordinal mixte croisé à 101 niveaux serait :

- complexe ;
- coûteux ;
- difficile à interpréter ;
- moins directement disponible dans `statsmodels`.

---

## 8.5 Approche en deux parties

Nous avons finalement utilisé une stratégie en deux parties :

```text
Partie 1 :
niveau de confiance parmi les réponses < 100

Partie 2 :
probabilité de répondre exactement 100
```

Cette approche distingue :

- l’intensité de confiance sous le plafond ;
- l’utilisation de la catégorie maximale.

Le script de l’étape 7 traite la première partie.

Le script de l’étape 8 traitera la seconde.

---

# 9. Formulation du modèle sous le plafond

Le modèle était :

\[
\begin{aligned}
Y_{ijk}
={}&
\beta_0
+\beta_1S_i
+\beta_2Q_{ik}\\
&+\beta_3A_i
+\beta_4H_j\\
&+\beta_5\bar M_i
+\beta_6M^W_{ik}\\
&+u_i+v_j+\varepsilon_{ijk}
\end{aligned}
\]

mais seulement pour :

\[
Y_{ijk}<100
\]

---

## 9.1 Formulation conditionnelle

Plus précisément, nous modélisons :

\[
E(
Y_{ijk}
\mid
Y_{ijk}<100,
X
)
\]

et non :

\[
E(Y_{ijk}\mid X)
\]

sur l’ensemble de la population.

Cette distinction doit être conservée dans l’interprétation.

---

# 10. Organisation générale du script

Le script suivait cette logique :

```text
1. Importer les bibliothèques
2. Définir le fichier et le dossier de sortie
3. Définir la formule parcimonieuse
4. Charger les données
5. Vérifier les colonnes
6. Filtrer analysis_complete
7. Convertir les colonnes numériques
8. Centrer la séquence
9. Standardiser les prédicteurs
10. Créer at_ceiling
11. Produire le résumé du plafond
12. Sélectionner confidence < 100
13. Ajuster le modèle en ML
14. Ajuster le modèle en REML
15. Extraire les coefficients
16. Sauvegarder les résumés
```

---

# 11. Configuration des fichiers

Le script définissait :

```python
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "dataset_analysis_E1.csv"
OUTPUT_DIR = BASE_DIR / "ceiling_sensitivity_E1"
```

Puis :

```python
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)
```

---

## 11.1 Pourquoi un dossier séparé ?

Les résultats du modèle sous le plafond ne doivent pas écraser les résultats du modèle principal.

Le dossier séparé matérialise le statut de l’analyse :

```text
analyse de sensibilité
```

et non :

```text
nouveau modèle principal
```

---

# 12. La formule utilisée

Le script définissait :

```python
FORMULA = (
    "confidence ~ "
    "C(condition, Treatment(reference='Neutral')) "
    "+ sequence_c10 "
    "+ subject_accuracy_z "
    "+ item_entropy_z "
    "+ subject_mean_models_z "
    "+ models_within_subject_z"
)
```

---

## 12.1 Pourquoi cette formule ne contient-elle pas `validity_binary` ?

À l’étape 6, le modèle sans validité a été retenu comme spécification parcimonieuse.

Le script de plafond utilise donc cette spécification.

---

## 12.2 Pourquoi conserver les variables non détectées ?

`subject_accuracy_z` et `subject_mean_models_z` restaient présents malgré leurs effets non détectés.

Raisons :

- ce sont des contrôles théoriques importants ;
- la décomposition between–within de MReasoner doit rester complète ;
- nous voulons comparer une formule cohérente avec le modèle final ;
- supprimer des variables uniquement selon leur valeur p pourrait créer une sélection opportuniste.

---

# 13. Différence avec le modèle cognitif initial

Le modèle initial de l’étape 5 contenait :

```text
validity_binary
```

Le modèle sous le plafond ne la contient pas.

La comparaison directe de chaque coefficient n’est donc pas parfaitement identique.

La comparaison la plus cohérente doit idéalement être faite avec le modèle parcimonieux complet sans validité.

Malgré cela, les différences principales étaient suffisamment nettes pour identifier :

- la robustesse de l’entropie ;
- la disparition de condition et séquence sous le plafond.

---

# 14. Chargement et vérification des données

Le script vérifiait les colonnes :

```text
confidence
condition
sequence
subject_id
item_id
subject_accuracy
item_entropy
subject_mean_models
models_within_subject
```

Puis il convertissait les variables numériques.

---

## 14.1 Pourquoi ne pas exiger `validity_binary` ?

Elle n’est pas utilisée dans la formule.

Un script doit exiger uniquement les variables nécessaires à son analyse.

---

## 14.2 Filtrage de complétude

Comme précédemment :

```text
analysis_complete = True
```

et aucune ligne n’a été perdue avant la sélection du plafond.

---

# 15. Construction de la variable `at_ceiling`

Le code était :

```python
data["at_ceiling"] = (
    data["confidence"] == 100
).astype(int)
```

---

## 15.1 Comparaison logique

```python
data["confidence"] == 100
```

produit :

```text
True
```

si la confiance vaut 100, et :

```text
False
```

sinon.

---

## 15.2 Conversion en entier

```python
.astype(int)
```

convertit :

```text
True  → 1
False → 0
```

Nous obtenons une variable binaire :

\[
\text{at\_ceiling}
=
\begin{cases}
1 & \text{si confidence}=100\\
0 & \text{sinon}
\end{cases}
\]

---

## 15.3 Pourquoi créer cette variable dans le script linéaire ?

Elle sert ici à compter le plafond.

Elle préparera également la logique de l’étape suivante, où elle deviendra la variable dépendante d’un modèle logistique.

---

# 16. Création du résumé du plafond

Le code construisait un tableau comme :

```python
ceiling_summary = pd.DataFrame([{
    "n_total": len(data),
    "n_at_ceiling": int(
        data["at_ceiling"].sum()
    ),
    "ceiling_rate": (
        data["at_ceiling"].mean()
    ),
    "n_below_ceiling": int(
        (data["confidence"] < 100).sum()
    ),
}])
```

---

## 16.1 `n_total`

\[
9024
\]

Nombre total d’observations.

---

## 16.2 `n_at_ceiling`

Comme `at_ceiling` vaut 0 ou 1, sa somme compte les 1 :

\[
\sum_i\text{at\_ceiling}_i
=
2336
\]

---

## 16.3 `ceiling_rate`

La moyenne d’une variable binaire est une proportion :

\[
\frac{2336}{9024}
=
0{,}258865
\]

---

## 16.4 `n_below_ceiling`

\[
6688
\]

Nombre de confiances strictement inférieures à 100.

---

## 16.5 Fichier produit

```text
ceiling_summary.csv
```

Contenu :

| `n_total` | `n_at_ceiling` | `ceiling_rate` | `n_below_ceiling` |
|---:|---:|---:|---:|
| 9024 | 2336 | 0,258865 | 6688 |

---

# 17. Sélection des observations inférieures à 100

Le code central était :

```python
below_ceiling = data.loc[
    data["confidence"] < 100
].copy()
```

---

## 17.1 `.loc[...]`

`.loc` sélectionne les lignes satisfaisant une condition.

Ici :

```text
confidence < 100
```

---

## 17.2 `.copy()`

La copie crée un nouveau tableau indépendant.

Cela évite de modifier accidentellement `data` ou de provoquer des avertissements lorsque des colonnes sont ajoutées.

---

## 17.3 Quelles observations sont supprimées ?

Uniquement les lignes où :

```text
confidence = 100
```

Les réponses à :

```text
0 à 99
```

sont conservées.

---

## 17.4 Les participants sont-ils supprimés ?

Un participant n’est complètement supprimé que si ses 64 réponses valent toutes 100.

D’après les résultats, le modèle conservait toujours les participants et items nécessaires, même si le nombre exact devait être vérifié dans la sortie du script.

Le participant 63873, par exemple, avait un taux de réponses à 100 très élevé, mais possédait aussi quelques valeurs à 99 et restait donc partiellement présent.

---

# 18. Conséquences de cette sélection

## 18.1 Nombre d’essais différent par participant

Dans le dataset complet :

```text
64 essais par participant
```

Après exclusion des 100 :

- certains participants conservent presque 64 essais ;
- d’autres en conservent beaucoup moins.

Le plan devient déséquilibré.

---

## 18.2 Condition représentée différemment

Les taux bruts de plafond étaient ensuite estimés à :

```text
Neutral  : 19,15 %
Standard : 32,53 %
```

La suppression retire donc proportionnellement davantage d’observations Standard.

Le sous-échantillon n’est pas un simple échantillon aléatoire du dataset original.

---

## 18.3 Effet sur les variables participant

Les variables comme :

```text
subject_accuracy
subject_mean_models
```

avaient été calculées sur les 64 essais complets dans le dataset analytique.

Elles ne sont pas recalculées après suppression.

C’est généralement approprié si nous voulons conserver la caractéristique générale du participant définie sur toute l’expérience.

Mais il faut le savoir : `subject_accuracy_z` ne représente pas uniquement sa précision parmi les essais sous le plafond.

---

# 19. Recentrage de la séquence

Le script calculait d’abord :

```python
data["sequence_c10"] = (
    data["sequence"]
    - data["sequence"].mean()
) / 10
```

avant de créer `below_ceiling`.

---

## 19.1 Centre utilisé

Le centre restait donc :

\[
32{,}5
\]

la moyenne de la séquence dans le dataset complet.

---

## 19.2 Pourquoi est-ce utile ?

L’intercept du modèle sous le plafond reste interprété au milieu de l’expérience complète.

Si nous recentrions après exclusion, le centre pourrait légèrement se déplacer selon les positions où les réponses 100 ont été supprimées.

Cela compliquerait la comparaison avec le modèle principal.

---

# 20. Standardisation des prédicteurs

Dans la version du script fournie, la standardisation était également calculée avant la sélection :

```python
data[variable + "_z"] = standardize(
    data[variable]
)
```

Puis `below_ceiling` héritait de ces valeurs.

---

## 20.1 Conséquence

Un écart-type correspond à l’échelle du dataset complet, pas seulement aux 6 688 observations restantes.

Cela améliore la comparabilité des coefficients avec le modèle principal.

---

## 20.2 Pourquoi est-ce préférable ici ?

Si nous standardisions séparément dans le sous-échantillon, une unité de `item_entropy_z` pourrait représenter une quantité légèrement différente.

Une différence de coefficient pourrait alors provenir :

- d’un changement de relation ;
- ou simplement d’un changement d’unité.

Conserver la standardisation complète limite cette ambiguïté.

---

# 21. Construction du modèle mixte

La fonction utilisait :

```python
model_data["_global_group"] = 1
```

puis :

```python
model = smf.mixedlm(
    FORMULA,
    model_data,
    groups=model_data["_global_group"],
    re_formula="0",
    vc_formula={
        "item": "0 + C(item_id)",
        "subject": "0 + C(subject_id)",
    },
)
```

La structure aléatoire restait donc identique.

---

## 21.1 Pourquoi conserver les effets croisés ?

Même après exclusion du plafond :

- plusieurs observations proviennent du même participant ;
- plusieurs observations concernent le même item.

La dépendance n’a pas disparu.

---

## 21.2 Plan désormais déséquilibré

Le modèle mixte peut gérer des nombres d’observations différents selon les participants et les items.

C’est l’un de ses avantages par rapport à certaines analyses agrégées plus rigides.

---

# 22. Ajustements ML et REML

Le script ajustait le modèle sous le plafond deux fois :

```python
result_ml
result_reml
```

---

## 22.1 Version ML

Elle peut servir à comparer des modèles ayant des effets fixes différents.

Dans ce script simple, aucune comparaison formelle avec un autre modèle ML n’était directement effectuée.

Le résultat était néanmoins sauvegardé pour cohérence et utilisation future.

---

## 22.2 Version REML

Elle fournissait les coefficients et variances présentés comme résultat principal de la sensibilité.

---

## 22.3 Optimiseurs

Le script essayait :

```text
lbfgs
bfgs
cg
powell
```

jusqu’à convergence.

Le modèle a convergé.

---

# 23. Extraction des effets fixes

La fonction :

```python
fixed_effects(result)
```

récupérait :

- le nom du coefficient ;
- son estimation ;
- son erreur-type ;
- sa statistique z ;
- sa valeur p ;
- son intervalle à 95 %.

Les calculs étaient les mêmes que dans les étapes précédentes :

\[
z=\frac{\widehat\beta}{SE}
\]

\[
IC_{95\%}
=
\widehat\beta
\pm1{,}96SE
\]

\[
p
=
2P(Z\geq|z|)
\]

---

# 24. Résultats obtenus

Le modèle REML sous le plafond a donné :

| Paramètre | Estimation | SE | z | p | IC 95 % |
|---|---:|---:|---:|---:|---:|
| Intercept | 67,327 | 1,586 | 42,440 | < .001 | [64,218 ; 70,436] |
| Standard | 1,977 | 2,300 | 0,860 | .390 | [−2,531 ; 6,485] |
| Séquence | 0,062 | 0,111 | 0,554 | .580 | [−0,156 ; 0,280] |
| Précision | 0,245 | 1,377 | 0,178 | .859 | [−2,454 ; 2,944] |
| Entropie | −2,307 | 0,306 | −7,530 | < .001 | [−2,907 ; −1,706] |
| Modèles moyens | −2,375 | 1,360 | −1,746 | .081 | [−5,041 ; 0,292] |
| Modèles intra | −0,284 | 0,241 | −1,175 | .240 | [−0,757 ; 0,189] |

---

# 25. Interprétation de l’intercept

\[
\widehat\beta_0=67{,}327
\]

Il représente la confiance prédite parmi les observations inférieures à 100 lorsque :

```text
condition = Neutral
sequence_c10 = 0
prédicteurs standardisés = 0
effets aléatoires = 0
```

---

## 25.1 Pourquoi est-il plus faible que dans le modèle complet ?

Le modèle complet avait un intercept autour de 73 dans sa version parcimonieuse.

En retirant les réponses maximales, la moyenne des valeurs restantes diminue naturellement.

L’intercept ne décrit plus toutes les réponses Neutral. Il décrit uniquement :

```text
les réponses Neutral dont confidence < 100
```

---

# 26. Effet de la condition

Résultat :

\[
\beta_{\text{Standard}}=1{,}977
\]

\[
SE=2{,}300
\]

\[
p=0{,}390
\]

\[
IC_{95\%}
=
[-2{,}531\,;\,6{,}485]
\]

---

## 26.1 Interprétation

Parmi les réponses inférieures à 100, Standard est associé à environ 1,98 point de confiance supplémentaire selon l’estimation centrale.

Mais l’incertitude est grande et l’intervalle contient zéro.

Aucun effet clair de condition n’est détecté dans ce sous-échantillon.

---

## 26.2 Comparaison avec le modèle complet

Dans le modèle complet :

\[
\beta_{\text{Standard}}
\approx5{,}15
\]

Sous le plafond :

\[
\beta_{\text{Standard}}
\approx1{,}98
\]

L’estimation diminue fortement.

Cela suggère qu’une grande partie de la différence Standard–Neutral provient de l’utilisation de 100.

---

## 26.3 Ce que nous ne pouvons pas encore affirmer

À ce stade, nous ne pouvons pas encore prouver directement :

> Standard augmente la probabilité de répondre 100.

Cette hypothèse est suggérée par la disparition de l’effet, mais elle doit être testée avec une variable dépendante binaire :

```text
at_ceiling
```

Ce sera l’étape 8.

---

# 27. Effet de la séquence

Résultat :

\[
\beta_{\text{sequence}}=0{,}062
\]

\[
SE=0{,}111
\]

\[
p=0{,}580
\]

\[
IC_{95\%}
=
[-0{,}156\,;\,0{,}280]
\]

---

## 27.1 Interprétation

Parmi les réponses inférieures à 100, il n’existe pas de baisse linéaire claire de confiance au fil des essais.

L’estimation centrale est même légèrement positive, mais pratiquement nulle et très incertaine.

---

## 27.2 Comparaison avec le modèle complet

Modèle complet :

\[
\beta\approx-0{,}437
\]

Sous le plafond :

\[
\beta\approx+0{,}062
\]

La baisse observée dans le modèle principal disparaît complètement.

---

## 27.3 Hypothèse suggérée

La séquence pourrait surtout influencer :

```text
la fréquence d’utilisation de 100
```

plutôt que le niveau de confiance parmi les réponses non maximales.

Cette hypothèse sera testée directement dans l’étape 8.

---

# 28. Effet de la précision

Résultat :

\[
\beta_{\text{précision}}=0{,}245
\]

\[
p=0{,}859
\]

L’effet reste proche de zéro et non détecté.

---

## 28.1 Comparaison avec le modèle principal

Modèle complet initial :

\[
0{,}310
\]

Sous le plafond :

\[
0{,}245
\]

La conclusion est stable :

> La précision moyenne du participant n’est pas clairement associée à son niveau de confiance moyen dans le modèle linéaire.

---

# 29. Effet de l’entropie

Résultat :

\[
\beta_{\text{entropie}}=-2{,}307
\]

\[
SE=0{,}306
\]

\[
z=-7{,}530
\]

\[
p\approx5{,}09\times10^{-14}
\]

\[
IC_{95\%}
=
[-2{,}907\,;\,-1{,}706]
\]

---

## 29.1 Interprétation

Parmi les réponses inférieures à 100, une augmentation d’un écart-type de l’entropie est associée à une diminution moyenne d’environ 2,31 points de confiance.

---

## 29.2 Comparaison avec le modèle complet

Modèle complet initial avec validité :

\[
-2{,}437
\]

Sous le plafond :

\[
-2{,}307
\]

La différence est faible :

\[
|-2{,}307-(-2{,}437)|
=
0{,}130
\]

Le signe et l’ordre de grandeur sont conservés.

---

## 29.3 Conclusion

L’effet négatif de l’entropie ne dépend pas uniquement des réponses à 100.

Il existe également parmi les niveaux de confiance non maximaux.

C’est une preuve importante de robustesse.

---

# 30. Effet interindividuel des modèles mentaux

Résultat :

\[
\beta_{\text{mean models}}=-2{,}375
\]

\[
SE=1{,}360
\]

\[
p=0{,}081
\]

\[
IC_{95\%}
=
[-5{,}041\,;\,0{,}292]
\]

---

## 30.1 Interprétation centrale

Parmi les réponses inférieures à 100, les participants ayant un nombre moyen de modèles plus élevé sont estimés moins confiants.

L’effet central est plus négatif que dans le modèle complet initial.

---

## 30.2 Pourquoi ne pas le déclarer significatif ?

La valeur p reste supérieure à 0,05 :

\[
0{,}081>0{,}05
\]

L’intervalle contient zéro.

L’effet n’est donc pas clairement détecté selon le seuil conventionnel.

---

## 30.3 Pourquoi ne pas parler de « tendance significative » ?

L’expression est contradictoire.

Un résultat est soit déclaré détecté selon un critère défini, soit non détecté.

Nous pouvons écrire :

> L’estimation était négative et plus importante en amplitude, mais restait imprécise.

---

## 30.4 Sélection possible

La suppression des réponses à 100 peut modifier la composition du sous-échantillon selon les caractéristiques participantes.

Le changement du coefficient ne doit donc pas être interprété comme une estimation nécessairement meilleure.

---

# 31. Effet intra-individuel des modèles mentaux

Résultat :

\[
\beta_{\text{within models}}=-0{,}284
\]

\[
SE=0{,}241
\]

\[
p=0{,}240
\]

\[
IC_{95\%}
=
[-0{,}757\,;\,0{,}189]
\]

---

## 31.1 Interprétation

La direction reste négative, mais l’effet est faible et non détecté dans cette analyse sous le plafond.

---

## 31.2 Comparaison initiale

Modèle complet avec validité :

\[
-0{,}366
\]

Sous le plafond :

\[
-0{,}284
\]

La direction est stable, mais la précision ne permet pas de conclure.

---

# 32. Composantes de variance

Le résumé REML donnait :

```text
item Var = 6,300
subject Var = 153,075
Scale = 266,095
```

Donc :

| Composante | Variance | Écart-type approximatif |
|---|---:|---:|
| Participant | 153,075 | 12,373 |
| Item | 6,300 | 2,510 |
| Résiduelle | 266,095 | 16,312 |

---

## 32.1 Variance participant plus faible

Dans le modèle complet, la variance participant était proche de 196.

Sous le plafond :

\[
153{,}075
\]

Une partie importante des différences individuelles concernait donc la propension à utiliser 100.

---

## 32.2 Prudence

Les variances ne sont pas directement comparables comme si les deux modèles utilisaient le même échantillon.

Le sous-échantillon exclut 25,9 % des observations et modifie le nombre de lignes par participant.

La baisse de variance est descriptive, non une preuve causale.

---

## 32.3 Variance résiduelle

Elle passe d’environ 285 à 266.

La dispersion des valeurs 0–99 est légèrement plus faible que celle de l’ensemble incluant une masse à 100.

---

# 33. Comparaison avec le modèle principal

| Prédicteur | Modèle complet initial | Sous le plafond | Lecture |
|---|---:|---:|---|
| Standard | 5,150, \(p=.042\) | 1,977, \(p=.390\) | Effet fortement réduit |
| Séquence | −0,437, \(p<.001\) | 0,062, \(p=.580\) | Effet disparu |
| Précision | 0,310, \(p=.837\) | 0,245, \(p=.859\) | Stable, non détecté |
| Entropie | −2,437, \(p<.001\) | −2,307, \(p<.001\) | Très robuste |
| Modèles moyens | −1,801, \(p=.232\) | −2,375, \(p=.081\) | Plus négatif, encore incertain |
| Modèles intra | −0,366, \(p=.137\) | −0,284, \(p=.240\) | Direction stable, non détecté |

---

## 33.1 Attention à la formule

Le modèle complet présenté dans ce tableau contenait la validité, alors que le modèle sous le plafond ne la contenait pas.

Une comparaison strictement parallèle aurait utilisé le modèle complet parcimonieux sans validité.

Cette limite doit être mentionnée.

Les différences de condition et de séquence sont néanmoins suffisamment fortes pour motiver l’analyse binaire suivante.

---

# 34. Pourquoi condition et séquence disparaissent-elles ?

Plusieurs explications sont possibles.

---

## 34.1 Explication principale suggérée

La condition et la séquence influencent surtout :

```text
la probabilité d’utiliser 100
```

Lorsque toutes les valeurs 100 sont supprimées, leur principal canal d’association disparaît.

---

## 34.2 Exemple de condition

Imaginons :

| Condition | Valeurs sous 100 | Proportion de 100 |
|---|---|---:|
| Neutral | moyenne 68 | 19 % |
| Standard | moyenne 70 | 33 % |

La différence de moyenne globale peut être forte à cause du nombre de 100, même si les valeurs sous 100 sont assez proches.

---

## 34.3 Exemple de séquence

Supposons que les participants utilisent souvent 100 au début, puis moins souvent à la fin.

Le modèle complet détecte une baisse de confiance moyenne.

Mais parmi les valeurs strictement inférieures à 100, le niveau moyen peut rester stable.

---

## 34.4 Autres explications

La disparition peut aussi refléter :

- une perte de puissance liée aux 2 336 observations supprimées ;
- une modification de la composition des groupes ;
- une relation non linéaire ;
- une sélection dépendant des prédicteurs.

Il faut donc tester directement `at_ceiling`.

---

# 35. Pourquoi l’entropie reste-t-elle importante ?

L’entropie semble agir sur deux dimensions potentielles :

1. elle réduit la probabilité de donner 100 ;
2. elle réduit le niveau de confiance même lorsque la réponse reste inférieure à 100.

L’étape 7 établit clairement la seconde partie.

L’étape 8 testera la première.

---

## 35.1 Interprétation intégrée provisoire

Les items à forte entropie ne produisent pas seulement moins de réponses extrêmement confiantes.

Ils produisent aussi des valeurs de confiance plus faibles sur le reste de l’échelle.

Cela rend l’effet plus général et plus robuste.

---

# 36. Le cas du nombre moyen de modèles

Le coefficient devient :

\[
-2{,}375
\]

avec :

\[
p=.081
\]

Ce résultat pourrait suggérer que, parmi les réponses non maximales, les participants générant davantage de modèles sont moins confiants.

Mais cette interprétation est fragile pour trois raisons.

---

## 36.1 Non-détection statistique

L’intervalle contient zéro.

---

## 36.2 Sélection sur la confiance

Nous avons retiré les réponses à 100.

Si le nombre moyen de modèles est aussi lié à la propension à utiliser 100, la composition du sous-échantillon dépend de ce prédicteur.

---

## 36.3 Estimations initiales n3

Cette analyse reposait encore sur les estimations initiales à trois simulations.

La stabilité MReasoner sera examinée plus tard.

---

# 37. Les fichiers générés

Le dossier était :

```text
ceiling_sensitivity_E1/
```

---

## 37.1 `ceiling_summary.csv`

Contient :

- nombre total ;
- nombre au plafond ;
- taux de plafond ;
- nombre inférieur au plafond.

---

## 37.2 `below_ceiling_fixed_effects_ML.csv`

Contient les coefficients ML du modèle sous le plafond.

Ils sont utiles pour des comparaisons de modèles ML ultérieures.

---

## 37.3 `below_ceiling_fixed_effects_REML.csv`

Contient les coefficients principaux présentés dans cette étape.

Colonnes :

```text
parameter
estimate
standard_error
z_value
p_value
ci_95_lower
ci_95_upper
```

---

## 37.4 `below_ceiling_model_ML.txt`

Résumé complet ML.

---

## 37.5 `below_ceiling_model_REML.txt`

Résumé complet REML, contenant aussi :

- variance participant ;
- variance item ;
- variance résiduelle ;
- convergence ;
- log-vraisemblance.

---

# 38. Ce que cette analyse permet de conclure

## 38.1 L’effet d’entropie est robuste

Il reste fortement négatif parmi les réponses inférieures à 100 :

\[
\beta=-2{,}307,\quad p<.001
\]

---

## 38.2 Les effets de condition et de séquence sont liés au plafond

Ils disparaissent dans le sous-échantillon.

Cela suggère qu’ils concernent principalement l’utilisation de la valeur 100.

---

## 38.3 La précision reste non associée au niveau de confiance

Le résultat est stable.

---

## 38.4 Les effets MReasoner restent incertains

La direction reste généralement négative, mais les intervalles contiennent zéro.

---

# 39. Ce qu’elle ne permet pas de conclure

## 39.1 Elle ne teste pas directement la probabilité de répondre 100

La suppression de ces réponses ne modélise pas leur apparition.

---

## 39.2 Elle ne prouve pas que condition et séquence n’affectent que le plafond

Elle fournit une indication forte, mais une analyse directe est nécessaire.

---

## 39.3 Elle ne remplace pas le modèle principal

Le sous-échantillon est sélectionné selon la variable dépendante.

---

## 39.4 Elle ne résout pas toutes les violations du modèle linéaire

Même entre 0 et 99, la confiance reste :

- bornée ;
- discrète ;
- potentiellement asymétrique ;
- possiblement hétéroscédastique.

---

# 40. Limites méthodologiques

## 40.1 Perte de données

\[
2336
\]

observations sont retirées.

La précision statistique diminue.

---

## 40.2 Déséquilibre créé

Le nombre d’essais conservés varie selon :

- le participant ;
- la condition ;
- potentiellement le type de tâche.

---

## 40.3 Sélection dépendante des prédicteurs

Si Standard, la séquence ou MReasoner prédisent le plafond, les observations conservées ne sont pas comparables de manière neutre.

---

## 40.4 Interprétation conditionnelle

Les coefficients valent uniquement pour :

\[
\text{confidence}<100
\]

Ils ne décrivent pas tous les essais.

---

## 40.5 Comparaison des variances

Les variances des modèles complet et sous le plafond utilisent des échantillons différents.

Leur différence doit rester descriptive.

---

# 41. Lien avec les étapes précédentes

## Modèle nul

La confiance était fortement structurée par les participants.

## Modèle de contrôle

Standard augmentait la confiance et la séquence la diminuait.

## Modèle cognitif

L’entropie était le principal prédicteur cognitif.

## Analyse de sensibilité logique

L’entropie restait robuste à la validité et au type de tâche.

## Analyse sous le plafond

L’entropie reste encore robuste, tandis que condition et séquence disparaissent.

Nous avons donc isolé deux types de résultats :

```text
Entropie :
effet général sur le niveau de confiance

Condition et séquence :
effets potentiellement concentrés
sur l’utilisation de la borne 100
```

---

# 42. Pourquoi passer ensuite au modèle logistique du plafond ?

Nous possédons maintenant une hypothèse précise :

> La condition Standard augmente la probabilité d’utiliser 100, tandis que cette probabilité diminue au cours des essais.

Pour la tester, nous transformons la variable dépendante en :

\[
\text{at\_ceiling}
=
\begin{cases}
1 & \text{si confidence}=100\\
0 & \text{sinon}
\end{cases}
\]

Une variable binaire ne doit pas être analysée avec une régression linéaire ordinaire.

Nous utiliserons un **modèle logistique mixte**.

Il estimera directement :

\[
P(\text{confidence}=100)
\]

en fonction de :

- la condition ;
- la séquence ;
- la précision ;
- l’entropie ;
- les modèles mentaux ;
- les effets participants ;
- les effets items.

Le script sera :

```text
fit_ceiling_logistic_mixed_model_E1.py
```

---

# 43. Bilan pédagogique

Le script `fit_ceiling_sensitivity_E1.py` a permis de :

1. identifier explicitement les réponses égales à 100 ;
2. quantifier l’effet plafond à 25,89 % ;
3. conserver les 6 688 observations inférieures à 100 ;
4. comprendre qu’une sélection sur la variable dépendante n’est pas neutre ;
5. réajuster le modèle parcimonieux sur ce sous-échantillon ;
6. conserver la structure croisée participant–item ;
7. conserver le centrage et la standardisation du dataset complet ;
8. montrer que l’effet Standard diminue fortement et n’est plus détecté ;
9. montrer que la baisse temporelle disparaît ;
10. confirmer l’absence d’association claire de la précision ;
11. confirmer la forte association négative de l’entropie ;
12. observer une estimation interindividuelle MReasoner plus négative mais encore incertaine ;
13. observer un effet intra-individuel toujours négatif mais non détecté ;
14. constater une variance participant plus faible parmi les réponses non maximales ;
15. établir que l’entropie agit au-delà de la seule utilisation de 100 ;
16. formuler l’hypothèse que condition et séquence influencent surtout la propension à atteindre le plafond ;
17. préparer le modèle logistique direct de cette propension.

La conclusion centrale est :

> Après exclusion des réponses maximales, l’association négative entre entropie et confiance demeure forte, alors que les effets de condition et de séquence disparaissent. Ces résultats suggèrent que l’entropie affecte le niveau général de confiance, tandis que condition et séquence influencent principalement l’utilisation de la borne supérieure.


# Étape 8 — Modèle logistique mixte du plafond avec `fit_ceiling_logistic_mixed_model_E1.py`

## Sommaire

1. [Rôle de cette étape](#1-rôle-de-cette-étape)
2. [Lien avec l’analyse précédente](#2-lien-avec-lanalyse-précédente)
3. [Nouvelle variable dépendante : `at_ceiling`](#3-nouvelle-variable-dépendante--at_ceiling)
4. [Pourquoi ne pas utiliser un modèle linéaire ?](#4-pourquoi-ne-pas-utiliser-un-modèle-linéaire)
5. [Qu’est-ce qu’une probabilité ?](#5-quest-ce-quune-probabilité)
6. [Qu’est-ce qu’une cote ou *odds* ?](#6-quest-ce-quune-cote-ou-odds)
7. [Le logit et les log-odds](#7-le-logit-et-les-log-odds)
8. [La régression logistique](#8-la-régression-logistique)
9. [Passer des log-odds aux probabilités](#9-passer-des-log-odds-aux-probabilités)
10. [Pourquoi un modèle logistique mixte ?](#10-pourquoi-un-modèle-logistique-mixte)
11. [Formulation mathématique du modèle](#11-formulation-mathématique-du-modèle)
12. [Différence avec le modèle linéaire](#12-différence-avec-le-modèle-linéaire)
13. [Pourquoi une estimation bayésienne variationnelle ?](#13-pourquoi-une-estimation-bayésienne-variationnelle)
14. [Notions élémentaires d’analyse bayésienne](#14-notions-élémentaires-danalyse-bayésienne)
15. [Les distributions a priori](#15-les-distributions-a-priori)
16. [La distribution postérieure](#16-la-distribution-postérieure)
17. [Qu’est-ce que l’approximation variationnelle ?](#17-quest-ce-que-lapproximation-variationnelle)
18. [Organisation générale du script](#18-organisation-générale-du-script)
19. [Bibliothèques utilisées](#19-bibliothèques-utilisées)
20. [Configuration du modèle](#20-configuration-du-modèle)
21. [Chargement et préparation des données](#21-chargement-et-préparation-des-données)
22. [Statistiques descriptives du plafond](#22-statistiques-descriptives-du-plafond)
23. [Construction du modèle avec `BinomialBayesMixedGLM`](#23-construction-du-modèle-avec-binomialbayesmixedglm)
24. [Les effets aléatoires croisés](#24-les-effets-aléatoires-croisés)
25. [Rôle des paramètres `fe_p` et `vcp_p`](#25-rôle-des-paramètres-fe_p-et-vcp_p)
26. [Premier ajustement non convergé](#26-premier-ajustement-non-convergé)
27. [Pourquoi `scale_fe=True` a aidé](#27-pourquoi-scale_fetrue-a-aidé)
28. [Procédure à plusieurs tentatives](#28-procédure-à-plusieurs-tentatives)
29. [Comment la convergence a été évaluée](#29-comment-la-convergence-a-été-évaluée)
30. [Résultat de l’optimisation finale](#30-résultat-de-loptimisation-finale)
31. [Extraction des effets fixes](#31-extraction-des-effets-fixes)
32. [Log-odds, odds ratios et intervalles crédibles](#32-log-odds-odds-ratios-et-intervalles-crédibles)
33. [Interprétation de l’intercept](#33-interprétation-de-lintercept)
34. [Interprétation de la condition Standard](#34-interprétation-de-la-condition-standard)
35. [Interprétation de la séquence](#35-interprétation-de-la-séquence)
36. [Interprétation de la précision](#36-interprétation-de-la-précision)
37. [Interprétation de l’entropie](#37-interprétation-de-lentropie)
38. [Interprétation du nombre moyen de modèles](#38-interprétation-du-nombre-moyen-de-modèles)
39. [Interprétation de la composante intra-individuelle](#39-interprétation-de-la-composante-intra-individuelle)
40. [Les effets aléatoires du modèle logistique](#40-les-effets-aléatoires-du-modèle-logistique)
41. [Pourquoi les variances logistiques sont difficiles à interpréter](#41-pourquoi-les-variances-logistiques-sont-difficiles-à-interpréter)
42. [Probabilités ajustées par condition et séquence](#42-probabilités-ajustées-par-condition-et-séquence)
43. [Pourquoi les probabilités ajustées diffèrent des taux bruts](#43-pourquoi-les-probabilités-ajustées-diffèrent-des-taux-bruts)
44. [Prédictions, score de Brier et log-loss](#44-prédictions-score-de-brier-et-log-loss)
45. [Calibration des prédictions du plafond](#45-calibration-des-prédictions-du-plafond)
46. [Les fichiers générés](#46-les-fichiers-générés)
47. [Erreur d’export rencontrée et correction](#47-erreur-dexport-rencontrée-et-correction)
48. [Ce que cette étape permet de conclure](#48-ce-que-cette-étape-permet-de-conclure)
49. [Ce qu’elle ne permet pas de conclure](#49-ce-quelle-ne-permet-pas-de-conclure)
50. [Limites méthodologiques](#50-limites-méthodologiques)
51. [Lien avec les étapes précédentes](#51-lien-avec-les-étapes-précédentes)
52. [Pourquoi cette étape conduit aux analyses finales](#52-pourquoi-cette-étape-conduit-aux-analyses-finales)
53. [Bilan pédagogique](#53-bilan-pédagogique)

---

# 1. Rôle de cette étape

À l’étape 7, nous avons retiré les 2 336 réponses égales à 100, puis réestimé le modèle linéaire sur les 6 688 réponses restantes.

Nous avons constaté que :

- l’effet de l’entropie restait fortement négatif ;
- l’effet de la condition Standard disparaissait ;
- l’effet négatif de la séquence disparaissait.

Cela suggérait que la condition et la séquence influençaient principalement la propension à utiliser la valeur maximale de l’échelle.

Mais supprimer les réponses à 100 ne constitue pas un test direct de cette hypothèse.

Il fallait maintenant modéliser explicitement :

\[
P(\text{confidence}=100)
\]

c’est-à-dire la probabilité qu’un essai aboutisse à une réponse de confiance maximale.

Le script correspondant est :

```text
fit_ceiling_logistic_mixed_model_E1.py
```

---

# 2. Lien avec l’analyse précédente

L’étape 7 et l’étape 8 forment une analyse en deux parties.

```text
Étape 7
Quelle est la confiance parmi les réponses inférieures à 100 ?
                         +
Étape 8
Quelle est la probabilité de répondre exactement 100 ?
```

Ces deux analyses distinguent deux mécanismes possibles :

1. un prédicteur peut modifier le niveau de confiance sur l’échelle 0–99 ;
2. il peut modifier la décision d’utiliser la catégorie extrême 100.

---

## 2.1 Exemple

Supposons deux conditions :

| Condition | Confiance moyenne sous 100 | Taux de réponses à 100 |
|---|---:|---:|
| Neutral | 68 | 19 % |
| Standard | 70 | 33 % |

La différence globale de confiance pourrait provenir essentiellement du taux de réponses maximales.

Le modèle de l’étape 7 examine la première colonne.

Le modèle de l’étape 8 examine la seconde.

---

# 3. Nouvelle variable dépendante : `at_ceiling`

Le script crée :

```python
data["at_ceiling"] = (
    data["confidence"] == 100
).astype(int)
```

La variable obtenue est :

\[
\text{at\_ceiling}_{ijk}
=
\begin{cases}
1 & \text{si confidence}=100\\
0 & \text{si confidence}<100
\end{cases}
\]

---

## 3.1 Exemples

| Confiance originale | `at_ceiling` |
|---:|---:|
| 0 | 0 |
| 50 | 0 |
| 99 | 0 |
| 100 | 1 |

---

## 3.2 Perte d’information volontaire

En transformant la confiance en variable binaire, nous ne distinguons plus :

```text
0, 50 et 99
```

Ces trois valeurs deviennent toutes :

```text
at_ceiling = 0
```

Cette perte est volontaire, car la question scientifique est désormais très précise :

> Le participant utilise-t-il ou non la catégorie maximale ?

Le modèle logistique du plafond ne remplace pas le modèle linéaire de la confiance. Il répond à une autre question.

---

# 4. Pourquoi ne pas utiliser un modèle linéaire ?

Nous pourrions imaginer :

```text
at_ceiling ~ condition + sequence + ...
```

dans une régression linéaire.

Cette approche est appelée **modèle de probabilité linéaire**, mais elle pose plusieurs problèmes.

---

## 4.1 Prédictions hors de l’intervalle \([0,1]\)

Une probabilité doit satisfaire :

\[
0\leq p\leq1
\]

Une équation linéaire :

\[
p=\beta_0+\beta_1X
\]

peut produire :

```text
p = −0,20
```

ou :

```text
p = 1,35
```

Ces valeurs n’ont aucun sens comme probabilités.

---

## 4.2 Variance non constante

Pour une variable binaire \(Y\) de probabilité \(p\) :

\[
\operatorname{Var}(Y)=p(1-p)
\]

La variance dépend donc de \(p\).

Exemples :

### Si \(p=0{,}5\)

\[
p(1-p)=0{,}25
\]

### Si \(p=0{,}05\)

\[
p(1-p)=0{,}0475
\]

L’hypothèse de variance résiduelle constante du modèle linéaire n’est pas respectée.

---

## 4.3 Distribution non normale

Une observation binaire ne peut prendre que :

```text
0 ou 1
```

Elle ne suit pas une distribution normale continue.

Elle suit une distribution de **Bernoulli**.

---

# 5. Qu’est-ce qu’une probabilité ?

Une probabilité \(p\) mesure la fréquence attendue d’un événement.

Elle est comprise entre :

\[
0\quad\text{et}\quad1
\]

ou, en pourcentage, entre :

```text
0 % et 100 %
```

---

## 5.1 Exemple

Si :

\[
p=0{,}25
\]

cela signifie que, dans des conditions comparables, l’événement devrait se produire environ une fois sur quatre.

Dans notre projet :

\[
p_{ijk}
=
P(
\text{at\_ceiling}_{ijk}=1
)
\]

est la probabilité que l’essai aboutisse à une confiance de 100.

---

# 6. Qu’est-ce qu’une cote ou *odds* ?

La régression logistique ne modélise pas directement la probabilité. Elle utilise d’abord les **odds**, appelées aussi cotes.

La formule est :

\[
\text{odds}
=
\frac{p}{1-p}
\]

Les odds comparent :

```text
probabilité que l’événement arrive
```

à :

```text
probabilité qu’il n’arrive pas
```

---

## 6.1 Exemple avec \(p=0{,}5\)

\[
\text{odds}
=
\frac{0{,}5}{0{,}5}
=
1
\]

On dit que les chances sont de 1 contre 1.

L’événement et le non-événement sont également probables.

---

## 6.2 Exemple avec \(p=0{,}8\)

\[
\text{odds}
=
\frac{0{,}8}{0{,}2}
=
4
\]

L’événement est quatre fois plus probable que son absence.

---

## 6.3 Exemple avec \(p=0{,}2\)

\[
\text{odds}
=
\frac{0{,}2}{0{,}8}
=
0{,}25
\]

Les odds sont de 0,25, soit une chance d’événement pour quatre chances de non-événement.

---

## 6.4 Probabilité et odds ne sont pas identiques

| Probabilité | Odds |
|---:|---:|
| 0,10 | 0,111 |
| 0,20 | 0,250 |
| 0,50 | 1,000 |
| 0,80 | 4,000 |
| 0,90 | 9,000 |

Une augmentation de 1 dans les odds n’équivaut pas à une augmentation fixe de probabilité.

---

# 7. Le logit et les log-odds

La régression logistique applique le logarithme aux odds :

\[
\operatorname{logit}(p)
=
\log
\left(
\frac{p}{1-p}
\right)
\]

Cette quantité est appelée :

- logit ;
- log-odds ;
- logarithme des cotes.

---

## 7.1 Pourquoi prendre le logarithme ?

Les odds sont positives :

\[
0<\text{odds}<+\infty
\]

Le logarithme les transforme sur toute la droite réelle :

\[
-\infty
<
\log(\text{odds})
<
+\infty
\]

Nous pouvons alors écrire une relation linéaire :

\[
\operatorname{logit}(p)
=
\beta_0+\beta_1X_1+\cdots
\]

Le côté droit peut prendre n’importe quelle valeur réelle, tandis que la transformation inverse garantit une probabilité entre 0 et 1.

---

## 7.2 Exemples

### Si \(p=0{,}5\)

\[
\text{odds}=1
\]

\[
\log(1)=0
\]

### Si \(p>0{,}5\)

Les log-odds sont positives.

### Si \(p<0{,}5\)

Les log-odds sont négatives.

---

## 7.3 Tableau

| Probabilité | Odds | Log-odds |
|---:|---:|---:|
| 0,05 | 0,0526 | −2,944 |
| 0,10 | 0,1111 | −2,197 |
| 0,50 | 1 | 0 |
| 0,90 | 9 | 2,197 |
| 0,95 | 19 | 2,944 |

---

# 8. La régression logistique

Une régression logistique écrit :

\[
\log
\left(
\frac{p_i}{1-p_i}
\right)
=
\beta_0+\beta_1X_{1i}+\cdots+\beta_kX_{ki}
\]

Dans notre projet, \(p_i\) est la probabilité d’utiliser 100.

---

## 8.1 Interprétation d’un coefficient

Un coefficient \(\beta_k\) représente la variation des log-odds associée à une augmentation d’une unité de \(X_k\).

Comme les log-odds sont difficiles à interpréter, nous calculons :

\[
OR=e^{\beta_k}
\]

où \(OR\) est l’**odds ratio**, ou rapport de cotes.

---

## 8.2 Si \(\beta=0\)

\[
OR=e^0=1
\]

Les odds ne changent pas.

---

## 8.3 Si \(\beta>0\)

\[
OR>1
\]

Les odds augmentent.

---

## 8.4 Si \(\beta<0\)

\[
OR<1
\]

Les odds diminuent.

---

# 9. Passer des log-odds aux probabilités

Pour convertir une valeur logit \(\eta\) en probabilité :

\[
p
=
\frac{1}{1+e^{-\eta}}
\]

Cette fonction est appelée :

- fonction logistique ;
- fonction sigmoïde ;
- `expit` dans SciPy.

Dans le code :

```python
from scipy.special import expit
```

puis :

```python
probability = expit(linear_predictor)
```

---

## 9.1 Exemple

Si :

\[
\eta=-2
\]

alors :

\[
p
=
\frac{1}{1+e^2}
\approx0{,}119
\]

La probabilité est environ 11,9 %.

---

## 9.2 Forme de la fonction logistique

```text
Probabilité
1.0 |                         ______
    |                      __/
0.5 |--------------------/
    |                 __/
0.0 |________________/
    +--------------------------------> prédicteur linéaire
              0
```

La courbe reste toujours entre 0 et 1.

---

# 10. Pourquoi un modèle logistique mixte ?

Les 9 024 observations restent regroupées :

- 64 essais par participant ;
- environ 70 ou 71 réponses par item.

La propension à utiliser 100 peut être très stable chez certains participants.

Exemple :

```text
Participant A : presque toujours 100
Participant B : jamais 100
```

Certains items peuvent aussi susciter davantage de réponses maximales.

Nous avons donc besoin de :

```text
intercept aléatoire participant
intercept aléatoire item
```

dans un modèle logistique.

---

# 11. Formulation mathématique du modèle

Pour le participant \(i\), l’item \(j\) et l’essai \(k\), définissons :

\[
Y_{ijk}
=
\text{at\_ceiling}_{ijk}
\]

avec :

\[
Y_{ijk}\sim\operatorname{Bernoulli}(p_{ijk})
\]

---

## 11.1 Distribution de Bernoulli

Une variable de Bernoulli vaut :

\[
1
\]

avec probabilité \(p\), et :

\[
0
\]

avec probabilité \(1-p\).

Sa fonction de probabilité peut être écrite :

\[
P(Y=y)
=
p^y(1-p)^{1-y}
\]

Pour \(y=1\) :

\[
P(Y=1)=p
\]

Pour \(y=0\) :

\[
P(Y=0)=1-p
\]

---

## 11.2 Partie linéaire

Le modèle est :

\[
\begin{aligned}
\operatorname{logit}(p_{ijk})
={}&
\beta_0
+\beta_1\text{Standard}_i\\
&+\beta_2\text{SequenceC10}_{ik}\\
&+\beta_3\text{AccuracyZ}_i\\
&+\beta_4\text{EntropyZ}_j\\
&+\beta_5\text{MeanModelsZ}_i\\
&+\beta_6\text{WithinModelsZ}_{ik}\\
&+u_i+v_j
\end{aligned}
\]

---

## 11.3 Effets aléatoires

\[
u_i
\sim
\mathcal N(0,\sigma^2_{\text{participant}})
\]

\[
v_j
\sim
\mathcal N(0,\sigma^2_{\text{item}})
\]

Ils agissent sur l’échelle des log-odds.

---

# 12. Différence avec le modèle linéaire

## Modèle linéaire

\[
Y
=
\eta+\varepsilon
\]

avec une erreur normale explicite :

\[
\varepsilon\sim\mathcal N(0,\sigma^2)
\]

---

## Modèle logistique

\[
Y\sim\operatorname{Bernoulli}(p)
\]

et :

\[
\operatorname{logit}(p)=\eta
\]

Il n’existe pas de variance résiduelle libre analogue à celle du modèle linéaire.

La variance conditionnelle de \(Y\) est déterminée par :

\[
p(1-p)
\]

---

## 12.1 Conséquence

Dans le résumé du modèle logistique, nous n’obtenons pas une ligne :

```text
Residual variance
```

équivalente à celle du modèle linéaire.

---

# 13. Pourquoi une estimation bayésienne variationnelle ?

Pour le modèle linéaire, `statsmodels.MixedLM` permet une estimation ML/REML des effets croisés.

Pour un modèle binaire avec effets aléatoires croisés, l’intégration des effets aléatoires est plus difficile.

La vraisemblance marginale contient une intégrale de la forme :

\[
L(\boldsymbol\beta,\boldsymbol\theta)
=
\int
P(\mathbf y\mid\boldsymbol\beta,\mathbf b)
P(\mathbf b\mid\boldsymbol\theta)
\,d\mathbf b
\]

Dans notre modèle, \(\mathbf b\) contient environ :

```text
141 effets participants
+
128 effets items
```

soit 269 effets aléatoires.

L’intégrale est de grande dimension et ne possède pas de solution analytique simple.

Le script a donc utilisé :

```python
BinomialBayesMixedGLM.fit_vb()
```

où `VB` signifie **variational Bayes**, ou Bayes variationnel.

---

# 14. Notions élémentaires d’analyse bayésienne

Une analyse bayésienne combine :

1. une distribution a priori ;
2. la vraisemblance des données ;
3. une distribution postérieure.

La relation fondamentale est :

\[
P(\theta\mid y)
\propto
P(y\mid\theta)P(\theta)
\]

où :

- \(P(\theta)\) est l’a priori ;
- \(P(y\mid\theta)\) est la vraisemblance ;
- \(P(\theta\mid y)\) est la distribution postérieure.

---

## 14.1 Analogie avec une enquête

Avant d’observer les données, nous avons une croyance générale sur les valeurs plausibles d’un paramètre.

Les observations apportent de nouvelles informations.

La distribution postérieure combine :

```text
ce que l’on considérait plausible avant
+
ce que les données montrent
```

---

# 15. Les distributions a priori

Le script utilisait :

```python
FE_P = 2.0
VCP_P = 0.5
```

puis :

```python
model = BinomialBayesMixedGLM.from_formula(
    formula=FORMULA,
    vc_formulas=VC_FORMULAS,
    data=data,
    fe_p=FE_P,
    vcp_p=VCP_P,
)
```

---

## 15.1 A priori sur les effets fixes

`fe_p=2.0` indique un écart-type a priori de 2 pour les coefficients fixes sur l’échelle logit.

Conceptuellement :

\[
\beta_k\sim\mathcal N(0,2^2)
\]

Cet a priori :

- centre les coefficients autour de zéro ;
- autorise néanmoins des effets substantiels ;
- évite des valeurs extrêmes sans forte justification.

---

## 15.2 A priori sur les composantes aléatoires

`vcp_p=0.5` contrôle l’a priori sur le logarithme des écarts-types aléatoires.

Le modèle travaille avec :

\[
\log(\sigma_{\text{aléatoire}})
\]

plutôt qu’avec la variance directement.

Cela garantit :

\[
\sigma>0
\]

après exponentiation.

---

## 15.3 Pourquoi utiliser des a priori modérément régularisants ?

Les modèles logistiques mixtes peuvent rencontrer :

- des effets très grands ;
- des participants répondant presque toujours 100 ou jamais 100 ;
- une séparation quasi parfaite ;
- des problèmes numériques.

Un a priori régularisant stabilise l’estimation.

Il ne remplace pas les données, mais évite qu’une information limitée produise des coefficients irréalistes.

---

# 16. La distribution postérieure

Après combinaison des données et des a priori, le modèle produit une distribution postérieure pour chaque coefficient.

Pour un effet fixe, le script extrait :

```python
result.fe_mean
result.fe_sd
```

---

## 16.1 `fe_mean`

Moyenne postérieure approximative du coefficient.

Elle joue un rôle analogue à l’estimation centrale.

---

## 16.2 `fe_sd`

Écart-type postérieur approximatif.

Il mesure la dispersion de la distribution postérieure autour de sa moyenne.

---

## 16.3 Différence avec l’erreur-type fréquentiste

Dans les modèles précédents :

```text
standard_error
```

était une erreur-type fréquentiste.

Ici :

```text
posterior_sd
```

est un écart-type de distribution postérieure approximative.

Les deux quantités peuvent jouer des rôles visuellement proches, mais elles proviennent de cadres théoriques différents.

---

# 17. Qu’est-ce que l’approximation variationnelle ?

La distribution postérieure exacte du modèle est trop complexe pour être calculée directement.

L’approximation variationnelle choisit une famille de distributions plus simples, notée :

\[
q(\theta)
\]

et cherche celle qui ressemble le plus possible à la véritable distribution postérieure :

\[
p(\theta\mid y)
\]

---

## 17.1 Problème d’approximation

```text
Distribution postérieure exacte
complexe, difficile à calculer
                 ↓
Choisir une distribution approchée q
                 ↓
Optimiser ses paramètres
pour qu’elle soit aussi proche que possible
```

---

## 17.2 Avantage

L’approximation variationnelle est généralement :

- beaucoup plus rapide qu’un échantillonnage Monte-Carlo complet ;
- adaptée à un modèle avec de nombreux effets aléatoires ;
- relativement simple à utiliser dans `statsmodels`.

---

## 17.3 Limite

Elle peut sous-estimer l’incertitude, notamment si la distribution postérieure exacte est :

- asymétrique ;
- fortement corrélée ;
- multimodale ;
- éloignée de la famille approximative choisie.

Il faut donc présenter ses intervalles comme :

```text
intervalles crédibles postérieurs approximatifs
```

et non comme une mesure parfaite de l’incertitude.

---

# 18. Organisation générale du script

Le script suivait cette architecture :

```text
1. Définir les chemins
2. Définir la formule fixe
3. Définir les effets aléatoires
4. Définir les a priori
5. Charger les 9 024 observations
6. Créer at_ceiling
7. Centrer et standardiser les prédicteurs
8. Produire les descriptifs du plafond
9. Construire BinomialBayesMixedGLM
10. Essayer plusieurs configurations d’optimisation
11. Sélectionner une solution convergée
12. Extraire les effets fixes
13. Calculer les odds ratios
14. Extraire les écarts-types aléatoires
15. Calculer les prédictions fixes
16. Produire les scénarios condition × séquence
17. Évaluer descriptivement les prédictions
18. Exporter les résultats
```

---

# 19. Bibliothèques utilisées

Le script utilisait notamment :

```python
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from scipy.special import expit
from scipy.stats import norm

from statsmodels.genmod.bayes_mixed_glm import (
    BinomialBayesMixedGLM
)
```

---

## 19.1 `expit`

`expit` applique :

\[
\operatorname{expit}(\eta)
=
\frac{1}{1+e^{-\eta}}
\]

Elle convertit les log-odds en probabilité.

---

## 19.2 `norm`

`norm` représente la loi normale.

Elle a servi à calculer les probabilités postérieures approximatives que le coefficient soit positif ou négatif.

---

## 19.3 `BinomialBayesMixedGLM`

Cette classe ajuste un modèle linéaire généralisé mixte binomial avec une estimation bayésienne.

- `Binomial` correspond à la réponse binaire ;
- `Bayes` correspond aux a priori et distributions postérieures ;
- `MixedGLM` correspond aux effets fixes et aléatoires.

---

# 20. Configuration du modèle

La formule était :

```python
FORMULA = (
    "at_ceiling ~ "
    "C(condition, Treatment(reference='Neutral')) "
    "+ sequence_c10 "
    "+ subject_accuracy_z "
    "+ item_entropy_z "
    "+ subject_mean_models_z "
    "+ models_within_subject_z"
)
```

Les effets aléatoires étaient :

```python
VC_FORMULAS = {
    "participant": "0 + C(subject_id)",
    "item": "0 + C(item_id)",
}
```

---

## 20.1 Pourquoi ne pas inclure la validité ?

L’analyse de sensibilité de l’étape 6 avait retenu le modèle parcimonieux sans validité.

Le modèle du plafond conservait cette structure.

---

## 20.2 Pourquoi les prédicteurs sont-ils identiques à ceux du modèle final ?

Nous voulions déterminer si chacun d’eux influençait spécifiquement l’utilisation de 100.

Conserver la même partie fixe facilite la comparaison entre :

- niveau de confiance ;
- probabilité d’atteindre le plafond.

---

# 21. Chargement et préparation des données

Les 9 024 lignes complètes étaient conservées.

Le script :

1. convertissait les colonnes numériques ;
2. normalisait les identifiants ;
3. vérifiait les conditions ;
4. vérifiait que la confiance était dans \([0,100]\) ;
5. créait `at_ceiling` ;
6. centrait la séquence ;
7. standardisait les prédicteurs.

---

## 21.1 Vérification des deux modalités

Le script contrôlait :

```python
if data["at_ceiling"].nunique() != 2:
    raise ValueError(...)
```

Pourquoi ?

Un modèle logistique nécessite que les deux résultats soient présents :

```text
0
1
```

Si personne ne répondait 100, il serait impossible d’estimer les facteurs associés au plafond.

---

# 22. Statistiques descriptives du plafond

Le taux global était :

\[
25{,}89\%
\]

Par condition :

| Condition | Observations | Réponses à 100 | Taux |
|---|---:|---:|---:|
| Neutral | 4 480 | 858 | 19,15 % |
| Standard | 4 544 | 1 478 | 32,53 % |

---

## 22.1 Différence brute

\[
32{,}53\%-19{,}15\%
=
13{,}38
\]

points de pourcentage.

Cette différence brute ne tient pas encore compte :

- de la séquence ;
- de l’entropie ;
- de la précision ;
- de MReasoner ;
- des effets aléatoires.

Le modèle fournit une comparaison ajustée.

---

## 22.2 Moyennes de confiance

| Condition | Confiance moyenne |
|---|---:|
| Neutral | 73,146 |
| Standard | 78,292 |

Les taux de plafond plus élevés en Standard contribuent à la différence de confiance moyenne observée à l’étape 4.

---

# 23. Construction du modèle avec `BinomialBayesMixedGLM`

Le code central était :

```python
model = BinomialBayesMixedGLM.from_formula(
    formula=FORMULA,
    vc_formulas=VC_FORMULAS,
    data=data,
    fe_p=FE_P,
    vcp_p=VCP_P,
)
```

---

## 23.1 `formula`

Définit la partie fixe :

\[
\mathbf X\boldsymbol\beta
\]

---

## 23.2 `vc_formulas`

Définit les familles d’intercepts aléatoires :

- participant ;
- item.

---

## 23.3 `data`

Contient les 9 024 observations.

---

## 23.4 `fe_p`

Définit l’ampleur a priori des effets fixes.

---

## 23.5 `vcp_p`

Définit l’ampleur a priori des paramètres de dispersion aléatoire.

---

# 24. Les effets aléatoires croisés

Pour chaque participant :

\[
u_i\sim\mathcal N(0,\sigma^2_P)
\]

Pour chaque item :

\[
v_j\sim\mathcal N(0,\sigma^2_I)
\]

Ces effets modifient les log-odds.

---

## 24.1 Exemple

Supposons une tendance fixe :

\[
\eta=-2
\]

Donc :

\[
p=\operatorname{expit}(-2)\approx0{,}119
\]

Pour un participant ayant un effet :

\[
u_i=+2
\]

la nouvelle valeur est :

\[
\eta=-2+2=0
\]

Donc :

\[
p=0{,}5
\]

Un effet aléatoire de +2 logits peut donc modifier fortement la probabilité.

---

## 24.2 Non-linéarité

Le même ajout de +1 logit ne produit pas toujours la même augmentation absolue de probabilité.

### De \(-4\) à \(-3\)

\[
p: 1{,}8\%\rightarrow4{,}7\%
\]

augmentation d’environ 2,9 points.

### De \(0\) à \(1\)

\[
p: 50\%\rightarrow73{,}1\%
\]

augmentation d’environ 23,1 points.

Les coefficients logistiques s’interprètent donc plus naturellement avec les odds ratios ou avec des probabilités de scénarios.

---

# 25. Rôle des paramètres `fe_p` et `vcp_p`

## 25.1 `fe_p=2.0`

Un coefficient fixe de ±2 logits est déjà substantiel.

\[
e^2\approx7{,}39
\]

Un coefficient de +2 multiplie les odds par environ 7,4.

L’a priori autorise donc des effets importants tout en pénalisant doucement les valeurs excessives.

---

## 25.2 `vcp_p=0.5`

Le paramètre de dispersion aléatoire est représenté sur l’échelle :

\[
\log(\sigma)
\]

Un a priori de dispersion 0,5 évite des écarts-types aléatoires gigantesques sans preuve suffisante.

---

## 25.3 Dépendance aux a priori

Les résultats bayésiens dépendent théoriquement des a priori.

Avec 9 024 observations, les données apportent beaucoup d’information, mais les effets participants peuvent encore être influencés par la régularisation, notamment pour les participants utilisant presque toujours la même catégorie.

Une analyse bayésienne exhaustive testerait plusieurs a priori raisonnables.

Notre modèle est une analyse complémentaire, pas le seul fondement des conclusions.

---

# 26. Premier ajustement non convergé

La première version utilisait :

```python
result = model.fit_vb(
    fit_method="BFGS",
    minim_opts={
        "maxiter": 5000,
        "gtol": 1e-6,
    },
    scale_fe=False,
)
```

Elle a produit :

```text
success: False
message: Desired error not necessarily achieved due to precision loss.
warning: VB fitting did not converge
```

---

## 26.1 Qu’est-ce que la perte de précision ?

L’optimiseur calcule des différences et des gradients avec des nombres décimaux finis.

Lorsque :

- les échelles des paramètres sont très différentes ;
- la surface d’optimisation est très plate dans certaines directions ;
- les gradients deviennent minuscules ;
- les matrices sont mal conditionnées ;

il peut ne plus distinguer correctement une amélioration réelle du bruit numérique.

---

## 26.2 Pourquoi ne pas utiliser immédiatement les résultats ?

Les coefficients semblaient plausibles, mais l’optimiseur déclarait un échec.

Les écarts-types postérieurs particulièrement petits ne devaient donc pas être considérés comme définitifs.

Une convergence correcte était nécessaire avant l’interprétation.

---

# 27. Pourquoi `scale_fe=True` a aidé

La tentative convergée a utilisé :

```python
scale_fe=True
```

Cette option centre et met temporairement à l’échelle les colonnes des effets fixes pendant l’optimisation.

---

## 27.1 Problème d’échelle

Les prédicteurs n’avaient pas tous exactement la même structure :

- `Standard` vaut 0 ou 1 ;
- `sequence_c10` varie environ entre −3,15 et +3,15 ;
- les variables standardisées ont un écart-type de 1 ;
- l’intercept vaut toujours 1.

Une mauvaise échelle peut rendre la surface d’optimisation étirée.

---

## 27.2 Analogie géographique

Imagine une vallée très longue et très étroite.

```text
Direction A : changement de milliers d’unités
Direction B : changement de millièmes d’unité
```

Un optimiseur peut zigzaguer et perdre en précision.

Mettre les dimensions sur des échelles comparables rend la vallée plus régulière et facilite la recherche du minimum.

---

## 27.3 Retour à l’échelle originale

`scale_fe=True` ne signifie pas que les résultats finaux doivent être interprétés sur une échelle artificielle.

Le logiciel reconvertit les effets fixes sur l’échelle originale avant de les retourner.

C’est pourquoi le résumé convergé contient des coefficients interprétables avec nos variables.

---

# 28. Procédure à plusieurs tentatives

Le script corrigé définissait plusieurs configurations :

```python
attempts = [
    {
        "name": "BFGS_scale_fe_true_gtol_1e-5",
        ...
    },
    {
        "name": "BFGS_scale_fe_true_gtol_1e-4",
        ...
    },
    {
        "name": "L-BFGS-B_scale_fe_true",
        ...
    },
]
```

---

## 28.1 Pourquoi plusieurs tentatives ?

Une seule configuration d’optimisation peut échouer pour des raisons numériques.

Tester plusieurs configurations raisonnables permet de déterminer si une solution stable peut être obtenue.

---

## 28.2 Sélection de la solution

Le script :

1. enregistrait le succès ;
2. enregistrait la valeur de l’objectif ;
3. enregistrait le gradient maximal ;
4. conservait la meilleure solution disponible ;
5. s’arrêtait lorsqu’une tentative réussissait officiellement.

---

## 28.3 Pourquoi ne pas changer arbitrairement jusqu’à obtenir le résultat souhaité ?

Les réglages concernaient la convergence numérique, pas le choix des coefficients.

Les formules, données et a priori restaient identiques.

Il aurait été problématique de modifier les prédicteurs ou les a priori uniquement pour obtenir un signe particulier.

---

# 29. Comment la convergence a été évaluée

Le tableau d’optimisation final indiquait :

```text
success = True
objective = 3032.9114565
iterations = 305
max_absolute_gradient ≈ 0.00001
message = Optimization terminated successfully.
```

---

## 29.1 `success=True`

L’optimiseur considère avoir satisfait ses critères d’arrêt.

---

## 29.2 Gradient absolu maximal

Le gradient indique dans quelle direction l’objectif peut encore s’améliorer.

Une valeur maximale d’environ :

\[
10^{-5}
\]

est très petite.

Cela signifie que la surface est presque plate autour de la solution trouvée.

---

## 29.3 Valeur de l’objectif

Le Bayes variationnel optimise une fonction liée à l’écart entre l’approximation et la distribution postérieure, souvent formulée à travers une borne variationnelle.

La valeur :

```text
3032.911
```

n’a pas une interprétation scientifique directe simple.

Elle sert principalement à comparer les tentatives d’optimisation portant sur le même modèle.

---

# 30. Résultat de l’optimisation finale

Le modèle convergé a produit :

| Paramètre | Moyenne postérieure | SD postérieure |
|---|---:|---:|
| Intercept | −2,4318 | 0,0348 |
| Standard | 1,3751 | 0,0697 |
| Séquence | −0,1621 | 0,0189 |
| Précision | −0,1842 | 0,0383 |
| Entropie | −0,3010 | 0,0339 |
| Modèles moyens | 0,1531 | 0,0350 |
| Modèles intra | −0,0837 | 0,0346 |

Composantes aléatoires :

| Composante | Log-SD | SD sur l’échelle logit |
|---|---:|---:|
| Participant | 1,0946 | 2,988 |
| Item | −1,1274 | 0,324 |

---

# 31. Extraction des effets fixes

Le script utilisait :

```python
names = list(result.model.fep_names)
means = np.asarray(result.fe_mean)
standard_deviations = np.asarray(result.fe_sd)
```

Puis :

```python
lower = means - 1.96 * standard_deviations
upper = means + 1.96 * standard_deviations
```

Enfin :

```python
odds_ratios = np.exp(means)
odds_ratio_lower = np.exp(lower)
odds_ratio_upper = np.exp(upper)
```

---

## 31.1 Intervalle crédible approximatif

L’intervalle :

\[
\mu_{\text{post}}
\pm1{,}96\,SD_{\text{post}}
\]

suppose une distribution postérieure approximativement normale dans la solution variationnelle.

---

## 31.2 Interprétation bayésienne

Dans le cadre de l’approximation utilisée, un intervalle crédible à 95 % représente une région contenant environ 95 % de la masse postérieure approximative du paramètre.

Cette interprétation diffère de celle d’un intervalle de confiance fréquentiste.

Mais il faut toujours rappeler qu’il s’agit ici d’une approximation variationnelle dépendant des a priori et de la famille d’approximation.

---

# 32. Log-odds, odds ratios et intervalles crédibles

Résultats finaux :

| Prédicteur | β log-odds | OR | IC crédible 95 % de l’OR |
|---|---:|---:|---:|
| Standard | 1,375 | 3,956 | [3,450 ; 4,535] |
| Séquence | −0,162 | 0,850 | [0,819 ; 0,882] |
| Précision | −0,184 | 0,832 | [0,772 ; 0,897] |
| Entropie | −0,301 | 0,740 | [0,693 ; 0,791] |
| Modèles moyens | 0,153 | 1,165 | [1,088 ; 1,248] |
| Modèles intra | −0,084 | 0,920 | [0,859 ; 0,984] |

---

## 32.1 Transformer un coefficient en OR

Pour Standard :

\[
OR=e^{1{,}375}
\approx3{,}956
\]

Pour l’entropie :

\[
OR=e^{-0{,}301}
\approx0{,}740
\]

---

## 32.2 Pourcentage de variation des odds

### Si \(OR>1\)

\[
(OR-1)\times100
\]

### Si \(OR<1\)

\[
(1-OR)\times100
\]

Exemple pour l’entropie :

\[
(1-0{,}740)\times100
=
26{,}0\%
\]

---

# 33. Interprétation de l’intercept

L’intercept est :

\[
\beta_0=-2{,}4318
\]

Son odds ratio est :

\[
e^{-2{,}4318}
\approx0{,}0879
\]

Attention : ce nombre est une cote, pas une probabilité.

---

## 33.1 Conversion en probabilité

\[
p
=
\frac{1}{1+e^{2{,}4318}}
\approx0{,}0808
\]

Soit environ :

\[
8{,}08\%
\]

---

## 33.2 Scénario de référence

Cette probabilité correspond à :

- Neutral ;
- milieu de l’expérience ;
- précision moyenne ;
- entropie moyenne ;
- moyenne de modèles moyenne ;
- composante intra-individuelle nulle ;
- effet participant nul ;
- effet item nul.

Il s’agit donc d’une probabilité conditionnelle pour un profil de référence, à partir des effets fixes.

---

# 34. Interprétation de la condition Standard

Coefficient :

\[
\beta=1{,}375
\]

Odds ratio :

\[
OR=3{,}956
\]

Intervalle :

\[
[3{,}450\,;\,4{,}535]
\]

---

## 34.1 Signification

Toutes choses égales par ailleurs, les odds d’utiliser 100 sont environ 3,96 fois plus élevées en Standard qu’en Neutral.

---

## 34.2 Ce que cela ne signifie pas

Cela ne signifie pas :

> La probabilité augmente de 396 points de pourcentage.

Un odds ratio n’est pas un rapport direct de probabilités.

---

## 34.3 Exemple au milieu de l’expérience

Neutral :

\[
p_N\approx0{,}0808
\]

Les odds Neutral sont :

\[
\frac{0{,}0808}{1-0{,}0808}
\approx0{,}0879
\]

En Standard :

\[
\text{odds}_S
=
3{,}956\times0{,}0879
\approx0{,}3477
\]

La probabilité Standard est :

\[
p_S
=
\frac{0{,}3477}{1+0{,}3477}
\approx0{,}2579
\]

soit :

\[
25{,}79\%
\]

---

## 34.4 Conclusion

La différence de condition observée dans le modèle linéaire est largement liée à une propension beaucoup plus forte à utiliser la réponse maximale en Standard.

---

# 35. Interprétation de la séquence

Coefficient :

\[
\beta=-0{,}1621
\]

OR :

\[
0{,}8503
\]

---

## 35.1 Unité

`sequence_c10` augmente d’une unité lorsqu’on avance de dix essais.

Chaque tranche de dix essais multiplie donc les odds de répondre 100 par :

\[
0{,}8503
\]

---

## 35.2 Pourcentage de diminution

\[
1-0{,}8503
=
0{,}1497
\]

soit environ :

\[
15{,}0\%
\]

de diminution des odds par dix essais.

---

## 35.3 Sur 63 essais

Du premier au dernier essai :

\[
6{,}3
\]

tranches de dix.

Le rapport d’odds total est :

\[
0{,}8503^{6{,}3}
\approx0{,}36
\]

Les odds de répondre 100 à la fin sont donc approximativement 36 % des odds du début pour un profil fixe comparable.

La diminution de probabilité dépend du niveau initial, car la relation logistique est non linéaire.

---

# 36. Interprétation de la précision

Coefficient :

\[
\beta=-0{,}184
\]

OR :

\[
0{,}832
\]

Intervalle :

\[
[0{,}772\,;\,0{,}897]
\]

---

## 36.1 Signification

Une augmentation d’un écart-type de la précision moyenne est associée à une diminution d’environ :

\[
1-0{,}832
=
0{,}168
\]

soit 16,8 % des odds d’utiliser 100.

---

## 36.2 Résultat apparemment surprenant

Dans le modèle linéaire, la précision n’était pas clairement associée à la confiance moyenne.

Ici, les participants plus précis semblent utiliser moins souvent l’extrême 100.

Ces résultats ne sont pas contradictoires.

Ils peuvent signifier :

```text
même niveau moyen global approximatif,
mais style d’utilisation de l’échelle différent
```

Les participants plus précis pourraient répartir davantage leur confiance entre 70 et 99 plutôt que d’utiliser 100.

---

## 36.3 Prudence

La précision :

- varie entre participants ;
- est calculée à partir des mêmes essais ;
- coexiste avec une très forte variance participant ;
- possède un intervalle variationnel potentiellement trop étroit.

Ce résultat doit rester complémentaire.

---

# 37. Interprétation de l’entropie

Coefficient :

\[
\beta=-0{,}301
\]

OR :

\[
0{,}740
\]

Intervalle :

\[
[0{,}693\,;\,0{,}791]
\]

---

## 37.1 Signification

Une augmentation d’un écart-type de l’entropie multiplie les odds d’utiliser 100 par 0,74.

Cela correspond à une diminution d’environ :

\[
26\%
\]

des odds.

---

## 37.2 Lien avec l’étape 7

À l’étape 7, l’entropie diminuait le niveau de confiance parmi les valeurs inférieures à 100 :

\[
\beta=-2{,}307
\]

À l’étape 8, elle diminue aussi la probabilité d’utiliser exactement 100 :

\[
OR=0{,}740
\]

---

## 37.3 Conclusion intégrée

L’effet d’entropie ne se limite pas à un seul aspect de l’échelle.

Une entropie élevée est associée :

1. à moins de réponses maximales ;
2. à un niveau inférieur parmi les réponses non maximales.

C’est l’un des principaux résultats de robustesse du projet.

---

# 38. Interprétation du nombre moyen de modèles

Coefficient :

\[
\beta=0{,}153
\]

OR :

\[
1{,}165
\]

Intervalle :

\[
[1{,}088\,;\,1{,}248]
\]

---

## 38.1 Signification

Une augmentation d’un écart-type du nombre moyen de modèles est associée à une augmentation d’environ :

\[
16{,}5\%
\]

des odds d’utiliser 100.

---

## 38.2 Apparente contradiction

Dans les modèles linéaires :

- le coefficient interindividuel était négatif ;
- il n’était pas clairement détecté.

Sous le plafond, l’estimation était encore plus négative.

Mais dans le modèle binaire, il est positif.

---

## 38.3 Interprétation possible

Certains participants ayant davantage de modèles moyens pourraient présenter une distribution plus polarisée :

- davantage de réponses à 100 ;
- mais des réponses sous le plafond relativement moins élevées.

Cela produirait :

```text
plus d’extrêmes à 100
+
niveau inférieur parmi les non-100
```

---

## 38.4 Prudence importante

Cette explication est exploratoire.

Le coefficient :

- varie entre participants ;
- dépend des estimations MReasoner ;
- provenait à ce stade des valeurs initiales n3 ;
- est estimé par approximation variationnelle ;
- peut être influencé par l’importante hétérogénéité participant.

Il ne doit pas devenir une conclusion cognitive principale sans réplication.

---

# 39. Interprétation de la composante intra-individuelle

Coefficient :

\[
\beta=-0{,}0837
\]

OR :

\[
0{,}920
\]

Intervalle :

\[
[0{,}859\,;\,0{,}984]
\]

---

## 39.1 Signification

Lorsqu’un type de tâche génère un nombre de modèles supérieur d’un écart-type au niveau personnel moyen, les odds d’utiliser 100 diminuent d’environ :

\[
8\%
\]

---

## 39.2 Cohérence théorique

La direction est cohérente avec l’idée suivante :

```text
davantage de possibilités représentées
→ moins de certitude extrême
```

---

## 39.3 Prudence

L’effet est faible et n’était pas clairement détecté dans les modèles linéaires initiaux.

Il doit donc être présenté comme un résultat secondaire ou exploratoire.

---

# 40. Les effets aléatoires du modèle logistique

Le résumé donnait :

```text
participant:
log-SD = 1,0946
SD = 2,988

item:
log-SD = −1,1274
SD = 0,324
```

---

## 40.1 Pourquoi les paramètres sont-ils des log-SD ?

Le modèle représente :

\[
\log(\sigma)
\]

afin que l’écart-type reste positif après exponentiation :

\[
\sigma=e^{\log(\sigma)}
\]

Pour les participants :

\[
e^{1{,}0946}
\approx2{,}988
\]

Pour les items :

\[
e^{-1{,}1274}
\approx0{,}324
\]

---

## 40.2 Signification du SD participant

Les participants diffèrent fortement dans leur propension générale à utiliser 100.

Un écart-type de :

\[
2{,}988
\]

sur l’échelle logit est très important.

Un déplacement de +2,988 multiplie les odds par :

\[
e^{2{,}988}
\approx19{,}85
\]

Un participant situé un écart-type au-dessus de la moyenne aléatoire peut donc avoir des odds d’utilisation du plafond presque vingt fois plus fortes qu’un participant d’effet nul, toutes choses égales par ailleurs.

---

## 40.3 Signification du SD item

\[
0{,}324
\]

correspond à un multiplicateur d’odds :

\[
e^{0{,}324}
\approx1{,}38
\]

Les différences entre items sont bien plus modestes que les différences entre participants.

---

# 41. Pourquoi les variances logistiques sont difficiles à interpréter

Dans un modèle linéaire, la variance résiduelle est explicitement estimée.

Dans un modèle logistique, il n’existe pas de résidu normal libre comparable.

Pour calculer un ICC latent approximatif, on utilise parfois une variance logistique théorique :

\[
\frac{\pi^2}{3}
\approx3{,}29
\]

---

## 41.1 ICC latent participant approximatif

Variance participant :

\[
2{,}988^2
\approx8{,}928
\]

Variance item :

\[
0{,}324^2
\approx0{,}105
\]

ICC participant :

\[
\frac{8{,}928}
{8{,}928+0{,}105+3{,}290}
\approx0{,}724
\]

---

## 41.2 ICC latent item approximatif

\[
\frac{0{,}105}
{8{,}928+0{,}105+3{,}290}
\approx0{,}009
\]

---

## 41.3 Interprétation prudente

Sur l’échelle latente logistique, une très grande part de la variation structurée du plafond est associée aux participants, tandis que la part item est très faible.

Ces valeurs ne sont pas directement comparables aux ICC du modèle linéaire sans préciser qu’elles reposent sur une représentation latente.

---

# 42. Probabilités ajustées par condition et séquence

Le script produisait :

```text
adjusted_ceiling_probabilities.csv
```

Résultats :

| Condition | Position | Probabilité fixe |
|---|---|---:|
| Neutral | Essai 1 | 12,77 % |
| Neutral | Milieu | 8,08 % |
| Neutral | Essai 64 | 5,01 % |
| Standard | Essai 1 | 36,68 % |
| Standard | Milieu | 25,79 % |
| Standard | Essai 64 | 17,26 % |

---

## 42.1 Calcul pour Neutral au début

Au premier essai :

\[
\text{sequence\_c10}=-3{,}15
\]

La valeur logit est :

\[
\eta
=
-2{,}4318
+
(-0{,}1621)(-3{,}15)
\]

\[
\eta
\approx-1{,}9211
\]

La probabilité est :

\[
p
=
\frac{1}{1+e^{1{,}9211}}
\approx0{,}1277
\]

---

## 42.2 Standard au début

On ajoute le coefficient Standard :

\[
\eta
=
-1{,}9211+1{,}3751
\]

\[
\eta
\approx-0{,}5460
\]

Puis :

\[
p=\operatorname{expit}(-0{,}5460)
\approx0{,}3668
\]

---

## 42.3 Pourquoi la différence de probabilité change-t-elle au fil du temps ?

L’effet Standard est constant sur l’échelle logit :

\[
+1{,}3751
\]

Mais la transformation logistique est non linéaire.

La différence absolue de probabilité dépend donc de la probabilité de départ.

---

# 43. Pourquoi les probabilités ajustées diffèrent des taux bruts

Taux bruts :

```text
Neutral  : 19,15 %
Standard : 32,53 %
```

Probabilités fixes au milieu :

```text
Neutral  : 8,08 %
Standard : 25,79 %
```

Cette différence n’est pas une erreur.

---

## 43.1 Profil moyen

Les probabilités ajustées fixent :

```text
subject_accuracy_z = 0
item_entropy_z = 0
subject_mean_models_z = 0
models_within_subject_z = 0
```

Les taux bruts mélangent tous les profils réellement observés.

---

## 43.2 Effets aléatoires nuls

Les probabilités calculées utilisent seulement les effets fixes.

Elles supposent :

\[
u_i=0,\quad v_j=0
\]

Mais les participants ont une très grande variance aléatoire.

---

## 43.3 Non-linéarité

Dans un modèle logistique :

\[
E[\operatorname{expit}(\eta+U)]
\neq
\operatorname{expit}(\eta+E[U])
\]

Comme :

\[
E[U]=0
\]

cela signifie généralement :

\[
E[\operatorname{expit}(\eta+U)]
\neq
\operatorname{expit}(\eta)
\]

La moyenne des probabilités intégrant les effets aléatoires n’est pas égale à la probabilité calculée avec un effet aléatoire fixé à zéro.

---

# 44. Prédictions, score de Brier et log-loss

Le script calculait des probabilités fixes pour chaque observation :

```python
linear_predictor = (
    result.model.exog
    @ result.fe_mean
)

probabilities = expit(
    linear_predictor
)
```

---

## 44.1 Prédiction fixe

Pour chaque ligne :

\[
\hat p_i
=
\operatorname{expit}
(\mathbf x_i^\top\hat{\boldsymbol\beta})
\]

Elle n’inclut pas les effets aléatoires particuliers.

---

## 44.2 Score de Brier

\[
\text{Brier}
=
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat p_i)^2
\]

Il mesure l’erreur quadratique des probabilités.

- 0 est parfait ;
- une valeur plus élevée indique de moins bonnes probabilités ;
- son interprétation dépend du taux de base de l’événement.

---

## 44.3 Log-loss

\[
\text{LogLoss}
=
-\frac{1}{n}
\sum_i
[
y_i\log(\hat p_i)
+
(1-y_i)\log(1-\hat p_i)
]
\]

Elle pénalise fortement les prédictions très confiantes mais incorrectes.

Exemple :

```text
prédire p = 0,99
alors que y = 0
```

est beaucoup plus pénalisé que :

```text
prédire p = 0,60
alors que y = 0
```

---

## 44.4 Exactitude au seuil 0,5

Le script classait :

```text
1 si p ≥ 0,5
0 sinon
```

puis calculait la proportion correcte.

Cette métrique peut être trompeuse lorsque la classe 1 est minoritaire.

Comme seulement 25,9 % des observations sont au plafond, prédire toujours 0 donnerait déjà environ 74,1 % d’exactitude.

Il faut donc privilégier les probabilités, le score de Brier et la calibration.

---

# 45. Calibration des prédictions du plafond

Le script divisait les probabilités prédites en groupes, puis comparait :

```text
probabilité moyenne prédite
```

à :

```text
taux réellement observé de réponses à 100
```

---

## 45.1 Bonne calibration

Si un groupe reçoit une probabilité moyenne de 20 %, environ 20 % de ses observations devraient réellement être au plafond.

---

## 45.2 Tableau de calibration

Le fichier :

```text
ceiling_logistic_calibration.csv
```

contient :

- nombre d’observations du groupe ;
- probabilité prédite moyenne ;
- taux observé de plafond.

---

## 45.3 Limite

La calibration était calculée à partir des effets fixes seulement et sur les données d’ajustement.

Elle ne constitue pas une validation externe.

---

# 46. Les fichiers générés

Le dossier était :

```text
ceiling_logistic_mixed_model_E1/
```

---

## 46.1 `ceiling_global_summary.csv`

Contient :

- nombre total ;
- nombre à 100 ;
- taux global ;
- nombre sous le plafond.

---

## 46.2 `ceiling_by_condition.csv`

Contient :

- effectif par condition ;
- nombre de réponses à 100 ;
- taux de plafond ;
- confiance moyenne.

---

## 46.3 `ceiling_by_sequence_block.csv`

Regroupe les séquences en blocs de huit essais :

```text
01–08
09–16
...
57–64
```

Il permet d’observer descriptivement l’évolution du plafond.

---

## 46.4 `predictor_standardization.csv`

Enregistre les moyennes et écarts-types utilisés pour les variables standardisées.

---

## 46.5 `ceiling_logistic_optimization_attempts.csv`

Contient :

- nom de la tentative ;
- succès ;
- objectif ;
- nombre d’itérations ;
- gradient maximal ;
- message ;
- avertissements.

Il documente la convergence.

---

## 46.6 `ceiling_logistic_fixed_effects.csv`

Contient les résultats finaux :

```text
parameter
posterior_mean_log_odds
posterior_sd
credible_95_lower_log_odds
credible_95_upper_log_odds
odds_ratio
credible_95_lower_odds_ratio
credible_95_upper_odds_ratio
posterior_probability_positive
posterior_probability_negative
```

---

## 46.7 `ceiling_logistic_random_effect_standard_deviations.csv`

Contient :

- log-SD postérieur ;
- SD du log-SD ;
- écart-type aléatoire sur l’échelle logit ;
- intervalle crédible ;
- variance approximative.

---

## 46.8 `adjusted_ceiling_probabilities.csv`

Présente les probabilités ajustées pour :

- Neutral et Standard ;
- début, milieu et fin de l’expérience ;
- prédicteurs cognitifs à leur moyenne ;
- effets aléatoires fixés à zéro.

---

## 46.9 `ceiling_logistic_predictions.csv`

Une ligne par essai avec :

- valeur observée `at_ceiling` ;
- logit fixe prédit ;
- probabilité fixe prédite ;
- erreur de prédiction.

---

## 46.10 `ceiling_logistic_calibration.csv`

Compare les probabilités prédites aux taux observés par groupe de probabilité.

---

## 46.11 `ceiling_logistic_model_summary.txt`

Contient :

- le tableau complet du modèle ;
- la tentative retenue ;
- les informations d’optimisation ;
- les avertissements éventuels.

---

## 46.12 `ceiling_logistic_results.json`

Archive :

- la formule ;
- les effets aléatoires ;
- le nombre d’observations ;
- les a priori ;
- la convergence ;
- les métriques prédictives.

---

# 47. Erreur d’export rencontrée et correction

Après la première modification du script, le résumé du modèle convergé indiquait :

```text
Intercept = −2,4318
Standard = 1,3751
```

mais l’ancien CSV contenait encore :

```text
Intercept = −3,0147
Standard = 1,1866
```

---

## 47.1 Cause conceptuelle

Le script avait conservé ou exporté l’objet provenant de l’ancienne tentative non convergée, ou l’export était placé avant la sélection correcte de la solution.

Le résumé et le CSV ne correspondaient donc pas au même objet `result`.

---

## 47.2 Correction

L’export a été déplacé après :

```text
sélection de la tentative convergée
```

avec :

```python
fixed_effects = (
    posterior_fixed_effects_table(result)
)

fixed_effects.to_csv(
    OUTPUT_DIR
    / "ceiling_logistic_fixed_effects.csv",
    index=False,
)
```

---

## 47.3 Leçon de reproductibilité

Il ne suffit pas qu’un script produise un fichier.

Il faut vérifier la cohérence entre :

- le terminal ;
- le résumé texte ;
- le CSV ;
- le JSON ;
- les probabilités calculées.

Cette vérification a permis de détecter l’export obsolète.

---

# 48. Ce que cette étape permet de conclure

## 48.1 La condition Standard augmente fortement l’utilisation de 100

\[
OR=3{,}956
\]

Les odds sont environ quatre fois plus élevées qu’en Neutral.

---

## 48.2 L’utilisation de 100 diminue au cours de l’expérience

\[
OR=0{,}850
\]

par dix essais, soit environ 15 % de diminution des odds.

---

## 48.3 L’entropie diminue la certitude extrême

\[
OR=0{,}740
\]

par écart-type, soit environ 26 % de diminution des odds.

---

## 48.4 Les différences participantes dominent

\[
SD_{\text{participant}}=2{,}988
\]

contre :

\[
SD_{\text{item}}=0{,}324
\]

La propension à utiliser 100 dépend fortement du style individuel.

---

## 48.5 Explication des résultats de l’étape 7

Les effets de condition et de séquence disparaissaient sous le plafond parce qu’ils concernent largement la probabilité d’utiliser 100.

L’entropie restait présente sous le plafond et prédit aussi le plafond. Son association est donc plus générale.

---

# 49. Ce qu’elle ne permet pas de conclure

## 49.1 L’odds ratio n’est pas un effet causal automatique

Le modèle estime une association ajustée.

Une interprétation causale de la condition dépend du plan expérimental. Pour les variables construites comme l’entropie et MReasoner, la prudence causale reste indispensable.

---

## 49.2 Les résultats MReasoner ne sont pas définitivement confirmés

Les coefficients logistiques de MReasoner :

- sont faibles à modérés ;
- reposaient initialement sur trois simulations ;
- sont sensibles aux différences entre participants ;
- proviennent d’une approximation variationnelle.

Ils sont exploratoires.

---

## 49.3 Les intervalles crédibles peuvent être trop étroits

Le Bayes variationnel tend parfois à sous-estimer les dépendances et l’incertitude de la postérieure.

Le fait qu’un intervalle exclue 1 ne doit pas être interprété exactement comme une preuve définitive.

---

## 49.4 Le modèle ne décrit pas le niveau 0–99

Il ne distingue que :

```text
100
contre
moins de 100
```

L’étape 7 reste nécessaire pour le niveau sous le plafond.

---

# 50. Limites méthodologiques

## 50.1 Choix de la frontière

Nous avons défini :

```text
at_ceiling = confidence == 100
```

Un autre choix, comme :

```text
confidence ≥ 95
```

répondrait à une autre question sur la confiance très élevée.

Notre choix est justifié par la borne explicite de l’échelle.

---

## 50.2 Approximation variationnelle

Elle est rapide et pratique, mais ne remplace pas une validation par :

- MCMC ;
- `brms` ou Stan ;
- un GLMM fréquentiste convergé ;
- une analyse de sensibilité aux a priori.

---

## 50.3 Très forte hétérogénéité participante

Cette hétérogénéité peut rendre les effets interindividuels difficiles à distinguer de styles généraux d’utilisation de l’échelle.

---

## 50.4 Prédictions en échantillon

Les scores et probabilités sont produits sur les mêmes données que l’ajustement.

Ils ne mesurent pas la généralisation à de nouveaux participants.

---

## 50.5 Absence d’interactions

Le modèle suppose notamment que :

- l’effet de séquence est identique dans les deux conditions ;
- l’effet de l’entropie est identique selon la condition ;
- l’effet MReasoner est identique pour les quatre tâches.

Ces hypothèses pourraient être étudiées dans des analyses exploratoires, mais augmenteraient la complexité.

---

# 51. Lien avec les étapes précédentes

## Étape 3 — Modèle nul

Les différences participantes étaient très importantes pour le niveau général de confiance.

## Étape 4 — Contrôle

Standard augmentait la confiance et la séquence la diminuait.

## Étape 5 — Cognitif

L’entropie était le principal prédicteur cognitif.

## Étape 6 — Sensibilité logique

L’effet d’entropie ne dépendait ni de la validité ni du type de tâche.

## Étape 7 — Sous le plafond

L’entropie restait négative, tandis que condition et séquence disparaissaient.

## Étape 8 — Modèle du plafond

Nous montrons directement que :

- Standard favorise l’utilisation de 100 ;
- cette utilisation diminue avec la séquence ;
- l’entropie la réduit.

---

# 52. Pourquoi cette étape conduit aux analyses finales

Nous avons maintenant :

- un modèle linéaire principal ;
- une sélection de spécification ;
- une analyse sous le plafond ;
- un modèle direct du plafond.

Il reste toutefois à vérifier :

1. les diagnostics détaillés des résidus ;
2. la stabilité des coefficients lorsqu’un participant est retiré ;
3. la relation entre confiance et exactitude ;
4. la stabilité de MReasoner avec 10 et 20 simulations ;
5. la génération des figures et du rapport final.

Ces tâches seront réalisées à l’étape 9 avec :

```text
run_final_diagnostics_calibration_E1.py
compare_mreasoner_simulations_E1.py
build_final_report_E1.py
```

---

# 53. Bilan pédagogique

Le script `fit_ceiling_logistic_mixed_model_E1.py` a permis de :

1. transformer la confiance en événement binaire `confidence == 100` ;
2. utiliser une distribution de Bernoulli adaptée à cette variable ;
3. modéliser les log-odds avec une fonction logistique ;
4. garantir des probabilités prédites comprises entre 0 et 1 ;
5. conserver les intercepts aléatoires croisés participant et item ;
6. utiliser `BinomialBayesMixedGLM` pour le modèle binaire mixte ;
7. introduire des a priori régularisants sur les effets fixes et les dispersions aléatoires ;
8. employer une approximation bayésienne variationnelle ;
9. détecter et refuser une première optimisation non convergée ;
10. améliorer la stabilité numérique avec `scale_fe=True` ;
11. documenter plusieurs tentatives d’optimisation ;
12. obtenir une convergence officielle avec un gradient maximal très faible ;
13. transformer les coefficients en odds ratios ;
14. calculer des intervalles crédibles approximatifs ;
15. montrer que Standard multiplie les odds du plafond par environ 3,96 ;
16. montrer que dix essais supplémentaires réduisent ces odds d’environ 15 % ;
17. montrer qu’un écart-type d’entropie réduit ces odds d’environ 26 % ;
18. mettre en évidence une très forte hétérogénéité entre participants ;
19. calculer des probabilités ajustées au début, au milieu et à la fin ;
20. expliquer pourquoi les probabilités fixes diffèrent des taux bruts ;
21. produire des prédictions, métriques et tableaux de calibration ;
22. détecter et corriger un ancien export CSV incohérent ;
23. compléter l’analyse sous le plafond par un modèle direct de l’utilisation de 100.

La conclusion centrale est :

> La condition Standard augmente fortement la propension à utiliser la valeur maximale de confiance, tandis que cette propension diminue au cours de l’expérience. L’entropie réduit à la fois le niveau de confiance parmi les réponses non maximales et la probabilité d’utiliser exactement 100. Elle constitue donc le prédicteur cognitif le plus robuste de l’ensemble des analyses réalisées jusqu’ici.

# Étape 9 — Diagnostics finaux, calibration métacognitive, robustesse de MReasoner et construction du rapport

## Sommaire

1. [Position de cette étape dans le projet](#1-position-de-cette-étape-dans-le-projet)
2. [Vue d’ensemble des trois scripts](#2-vue-densemble-des-trois-scripts)
3. [`run_final_diagnostics_calibration_E1.py`](#3-run_final_diagnostics_calibration_e1py)
   1. [Pourquoi réaliser des diagnostics ?](#31-pourquoi-réaliser-des-diagnostics-)
   2. [Reconstruction du modèle linéaire final](#32-reconstruction-du-modèle-linéaire-final)
   3. [Valeurs ajustées et résidus](#33-valeurs-ajustées-et-résidus)
   4. [Distribution des résidus](#34-distribution-des-résidus)
   5. [Résidus standardisés et observations atypiques](#35-résidus-standardisés-et-observations-atypiques)
   6. [Influence et jackknife par participant](#36-influence-et-jackknife-par-participant)
   7. [Stabilité des coefficients](#37-stabilité-des-coefficients)
   8. [Calibration métacognitive](#38-calibration-métacognitive)
   9. [Score de Brier](#39-score-de-brier)
   10. [Discrimination métacognitive et AUC de type 2](#310-discrimination-métacognitive-et-auc-de-type-2)
   11. [Modèle logistique de l’exactitude](#311-modèle-logistique-de-lexactitude)
   12. [Graphique ajusté de l’entropie](#312-graphique-ajusté-de-lentropie)
   13. [Fichiers produits](#313-fichiers-produits)
4. [`compare_mreasoner_simulations_E1.py`](#4-compare_mreasoner_simulations_e1py)
   1. [Pourquoi comparer 3, 10 et 20 simulations ?](#41-pourquoi-comparer-3-10-et-20-simulations-)
   2. [Ce que signifie le nombre de simulations](#42-ce-que-signifie-le-nombre-de-simulations)
   3. [Statistiques calculées](#43-statistiques-calculées)
   4. [Résultats par type de tâche](#44-résultats-par-type-de-tâche)
   5. [Comparaisons 3–10–20](#45-comparaisons-31020)
   6. [Pourquoi reconstruire le dataset avec \(N=20\) ?](#46-pourquoi-reconstruire-le-dataset-avec-n20-)
   7. [Stabilité du modèle statistique entre \(N=3\) et \(N=20\)](#47-stabilité-du-modèle-statistique-entre-n3-et-n20)
   8. [Fichiers produits](#48-fichiers-produits)
5. [`build_final_report_E1.py`](#5-build_final_report_e1py)
   1. [Pourquoi automatiser le rapport ?](#51-pourquoi-automatiser-le-rapport-)
   2. [Ce que fait réellement ce script](#52-ce-que-fait-réellement-ce-script)
   3. [Organisation du code](#53-organisation-du-code)
   4. [Limites d’un rapport automatique](#54-limites-dun-rapport-automatique)
6. [Enchaînement complet de l’étape 9](#6-enchaînement-complet-de-létape-9)
7. [Ce que cette étape nous a appris](#7-ce-que-cette-étape-nous-a-appris)
8. [Ce qu’il reste à faire](#8-ce-quil-reste-à-faire)

---

# 1. Position de cette étape dans le projet

Jusqu’ici, nous avons successivement :

1. construit un fichier analytique propre ;
2. ajusté un modèle mixte nul ;
3. ajouté des variables de contrôle ;
4. ajouté les prédicteurs cognitifs ;
5. comparé différentes spécifications ;
6. étudié le problème des valeurs de confiance égales à 100 ;
7. modélisé séparément la probabilité d’utiliser cette borne supérieure.

À ce stade, nous disposons de résultats statistiques. Cependant, obtenir un tableau de coefficients n’est pas suffisant.

Nous devons encore répondre à plusieurs questions fondamentales :

- Le modèle linéaire décrit-il raisonnablement les données ?
- Existe-t-il des observations très mal prédites ?
- Les conclusions dépendent-elles d’un seul participant ?
- Les coefficients restent-ils stables quand on retire un participant ?
- La confiance exprimée par les participants reflète-t-elle leur exactitude ?
- Les estimations de MReasoner sont-elles suffisamment stables ?
- Les résultats changent-ils quand on utilise 20 simulations au lieu de 3 ?
- Comment rassembler tous les résultats dans un document cohérent ?

C’est précisément le rôle de l’étape 9.

> **Idée générale**
>
> Les premières étapes servent à construire et ajuster les modèles.  
> L’étape 9 sert à vérifier leur fiabilité, à approfondir leur interprétation et à organiser les résultats en vue du rapport scientifique.

---

# 2. Vue d’ensemble des trois scripts

L’étape 9 repose principalement sur trois scripts.

| Script | Rôle principal |
|---|---|
| `run_final_diagnostics_calibration_E1.py` | Vérifier le modèle final, étudier les résidus, l’influence des participants et la calibration métacognitive |
| `compare_mreasoner_simulations_E1.py` | Comparer les estimations produites avec 3, 10 et 20 simulations de MReasoner |
| `build_final_report_E1.py` | Lire les fichiers de résultats et générer automatiquement un rapport Markdown |

Ces trois scripts remplissent des fonctions différentes.

```text
Modèle statistique final
          │
          ├── Vérification statistique
          │      └── diagnostics, résidus, influence
          │
          ├── Interprétation métacognitive
          │      └── confiance comparée à l’exactitude
          │
          ├── Vérification computationnelle
          │      └── MReasoner avec 3, 10 et 20 simulations
          │
          └── Synthèse
                 └── rapport final Markdown
```

---

# 3. `run_final_diagnostics_calibration_E1.py`

# 3.1 Pourquoi réaliser des diagnostics ?

## 3.1.1 Définition d’un diagnostic statistique

Un **diagnostic statistique** est une vérification effectuée après l’ajustement d’un modèle.

Son objectif est de déterminer si :

- le modèle se comporte comme prévu ;
- certaines hypothèses sont fortement violées ;
- certaines observations sont atypiques ;
- les résultats sont excessivement influencés par quelques individus ;
- les coefficients sont stables.

Une analogie simple est celle d’un contrôle technique automobile.

Ajuster le modèle revient à construire la voiture. Le fait qu’elle démarre signifie que l’algorithme a convergé. Mais cela ne garantit pas encore que :

- les freins fonctionnent ;
- les pneus sont en bon état ;
- la direction est stable ;
- la voiture ne dépend pas d’une seule pièce fragile.

Les diagnostics constituent donc le contrôle technique du modèle.

## 3.1.2 Convergence et validité ne sont pas la même chose

Il est important de distinguer deux affirmations :

1. **Le modèle a convergé.**
2. **Le modèle décrit correctement les données.**

La convergence signifie seulement que l’algorithme d’optimisation a trouvé une solution numérique stable selon son critère.

Cela ne prouve pas que :

- la distribution des résidus est normale ;
- la relation entre les variables est bien spécifiée ;
- les coefficients sont robustes ;
- le modèle est scientifiquement pertinent.

> Un modèle peut converger parfaitement tout en étant mal adapté aux données.

---

# 3.2 Reconstruction du modèle linéaire final

Le script recharge d’abord le dataset analytique, de préférence sa version fondée sur 20 simulations :

```python
dataset_analysis_E1_n20.csv
```

Le modèle final est ensuite ajusté à nouveau.

Sa forme générale est :

\[
Y_{ij}
=
\beta_0
+
\beta_1 \text{Condition}_i
+
\beta_2 \text{Séquence}_{ij}
+
\beta_3 \text{Précision}_i
+
\beta_4 \text{Entropie}_j
+
\beta_5 \text{ModèlesMoyens}_i
+
\beta_6 \text{ModèlesIntra}_{ij}
+
u_i
+
v_j
+
\varepsilon_{ij}
\]

où :

- \(Y_{ij}\) est la confiance donnée par le participant \(i\) à l’item \(j\) ;
- \(\beta_0\) est l’interception générale ;
- les \(\beta_1,\ldots,\beta_6\) sont les effets fixes ;
- \(u_i\) est l’effet aléatoire du participant ;
- \(v_j\) est l’effet aléatoire de l’item ;
- \(\varepsilon_{ij}\) est le résidu de l’essai.

Le modèle final utilisé dans le rapport ne contient généralement pas `validity_binary`.

La formule Python correspondante est approximativement :

```python
confidence ~ (
    C(condition, Treatment(reference='Neutral'))
    + sequence_c10
    + subject_accuracy_z
    + item_entropy_z
    + subject_mean_models_z
    + models_within_subject_z
)
```

Les effets aléatoires croisés sont :

```python
0 + C(subject_id)
0 + C(item_id)
```

## Pourquoi ajuster une nouvelle fois le modèle ?

Le script de diagnostic doit disposer de l’objet modèle complet afin de récupérer :

- les coefficients ;
- les valeurs ajustées ;
- les résidus ;
- les composantes de variance ;
- les effets aléatoires ;
- la log-vraisemblance ;
- les informations nécessaires aux graphiques.

Lire uniquement le CSV des coefficients ne serait pas suffisant. Ce CSV contient un résumé, mais pas tout le fonctionnement interne du modèle.

---

# 3.3 Valeurs ajustées et résidus

## 3.3.1 Valeur ajustée

La **valeur ajustée**, également appelée **valeur prédite**, est la confiance que le modèle attribue à une observation à partir des paramètres estimés.

Pour un essai donné :

\[
\widehat{Y}_{ij}
=
\widehat{\beta}_0
+
\widehat{\beta}_1 X_{1,ij}
+\cdots+
\widehat{\beta}_6 X_{6,ij}
+
\widehat{u}_i
+
\widehat{v}_j
\]

Le symbole \(\widehat{}\), appelé « chapeau », signifie que la quantité a été estimée à partir des données.

Dans un modèle mixte, il existe deux types importants de prédiction.

### Prédiction marginale

Elle utilise seulement les effets fixes :

\[
\widehat{Y}_{ij}^{\text{marginal}}
=
X_{ij}\widehat{\beta}
\]

Elle décrit la prédiction pour un participant moyen et un item moyen.

### Prédiction conditionnelle

Elle utilise les effets fixes et les effets aléatoires estimés :

\[
\widehat{Y}_{ij}^{\text{conditionnelle}}
=
X_{ij}\widehat{\beta}
+
\widehat{u}_i
+
\widehat{v}_j
\]

Elle tient donc compte du fait qu’un participant peut être généralement plus ou moins confiant et qu’un item peut susciter une confiance moyenne plus ou moins élevée.

Pour diagnostiquer les erreurs du modèle sur les observations déjà utilisées pour l’ajustement, les résidus conditionnels sont souvent les plus naturels.

## 3.3.2 Résidu

Le **résidu** est la différence entre la valeur réellement observée et la valeur prédite :

\[
e_{ij}
=
Y_{ij}
-
\widehat{Y}_{ij}
\]

Exemple :

- confiance observée : 90 ;
- confiance prédite : 75 ;
- résidu :

\[
90-75=15
\]

Le modèle a sous-estimé la confiance de 15 points.

Autre exemple :

- confiance observée : 40 ;
- confiance prédite : 70 ;
- résidu :

\[
40-70=-30
\]

Le modèle a surestimé la confiance de 30 points.

### Interprétation du signe

| Résidu | Signification |
|---:|---|
| Positif | La confiance réelle est supérieure à la prédiction |
| Nul | La prédiction correspond exactement à l’observation |
| Négatif | La confiance réelle est inférieure à la prédiction |

## 3.3.3 Aucun tirage aléatoire n’est effectué

Comme pour le modèle nul, les valeurs ajustées utilisées dans les graphiques ne sont pas obtenues en simulant un nouveau tirage dans une loi normale.

Le script ne fait pas ceci :

```text
tirer aléatoirement une confiance prédite
puis la comparer à la vraie confiance
```

Il fait ceci :

```text
calculer la moyenne conditionnelle prédite
puis soustraire cette moyenne à la confiance observée
```

La loi normale intervient dans la procédure d’estimation et dans les hypothèses du modèle :

\[
u_i \sim \mathcal{N}(0,\sigma^2_u)
\]

\[
v_j \sim \mathcal{N}(0,\sigma^2_v)
\]

\[
\varepsilon_{ij} \sim \mathcal{N}(0,\sigma^2_\varepsilon)
\]

Mais une fois les paramètres estimés, la valeur ajustée est une quantité déterministe.

---

# 3.4 Distribution des résidus

Le fichier obtenu indiquait approximativement :

| Variable | Valeur |
|---|---:|
| Nombre de résidus | 9024 |
| Moyenne | presque 0 |
| Écart-type | 16,677 |
| Asymétrie | -1,431 |
| Excès de kurtosis | 4,403 |
| Statistique de Shapiro–Wilk | 0,899 |
| Valeur \(p\) de Shapiro–Wilk | \(2,56\times10^{-49}\) |

## 3.4.1 Pourquoi la moyenne des résidus est-elle presque nulle ?

La moyenne était :

\[
3{,}37\times10^{-13}
\]

Cette valeur est pratiquement égale à zéro.

Elle n’est pas exactement zéro à cause des approximations numériques de l’ordinateur.

Une moyenne résiduelle proche de zéro indique que le modèle ne surestime pas ou ne sous-estime pas systématiquement la confiance sur l’ensemble des données.

Mais cela ne garantit pas que les résidus sont bien distribués.

Par exemple, les erreurs suivantes ont une moyenne nulle :

\[
-50,\ -50,\ +50,\ +50
\]

Pourtant, elles sont très grandes.

## 3.4.2 Asymétrie

L’**asymétrie**, ou *skewness*, mesure si une distribution est plus étirée d’un côté que de l’autre.

- asymétrie proche de 0 : distribution approximativement symétrique ;
- asymétrie positive : longue queue vers les valeurs positives ;
- asymétrie négative : longue queue vers les valeurs négatives.

Ici :

\[
\text{asymétrie} \approx -1{,}43
\]

La distribution possède donc une queue importante du côté négatif.

Cela signifie qu’il existe des essais pour lesquels :

\[
Y_{ij} \ll \widehat{Y}_{ij}
\]

Autrement dit, le modèle prédit parfois une confiance relativement élevée alors que le participant donne une confiance beaucoup plus basse.

### Pourquoi cette asymétrie est-elle plausible ?

La confiance est bornée entre 0 et 100.

De plus, beaucoup de participants utilisent fréquemment des valeurs élevées, notamment 100.

Un modèle linéaire utilisant une distribution normale ne tient pas naturellement compte de ces bornes. Il peut donc produire une distribution résiduelle asymétrique.

## 3.4.3 Kurtosis

La **kurtosis** décrit notamment l’importance des queues d’une distribution, c’est-à-dire la fréquence d’observations éloignées du centre.

Pour une loi normale, l’excès de kurtosis est égal à zéro.

Ici :

\[
\text{excès de kurtosis} \approx 4{,}40
\]

Cette valeur positive indique des queues plus lourdes que celles d’une loi normale.

En termes simples, les erreurs très grandes sont plus fréquentes que ce qu’une distribution normale parfaite laisserait attendre.

## 3.4.4 Test de Shapiro–Wilk

Le **test de Shapiro–Wilk** évalue l’hypothèse suivante :

\[
H_0 : \text{les données sont compatibles avec une distribution normale}
\]

contre :

\[
H_1 : \text{les données ne sont pas compatibles avec une distribution normale}
\]

La valeur \(p\) obtenue est extrêmement petite :

\[
p \approx 2{,}56\times10^{-49}
\]

On rejette donc l’hypothèse d’une normalité parfaite.

### Pourquoi n’avons-nous testé que 5000 résidus ?

Les tests de normalité peuvent être très sensibles avec un grand échantillon.

Le script a probablement sélectionné un maximum de 5000 valeurs, notamment parce que certaines implémentations de Shapiro–Wilk deviennent moins appropriées ou produisent des avertissements pour des tailles très élevées.

Cette sélection ne signifie pas que seuls 5000 essais ont été utilisés dans le modèle. Le modèle utilise bien 9024 essais. La limitation concerne uniquement ce test diagnostic.

## 3.4.5 Une valeur \(p\) minuscule signifie-t-elle que le modèle est inutilisable ?

Non.

Avec plusieurs milliers d’observations, un test de normalité détecte des écarts très faibles.

De plus, ici, les écarts ne sont pas totalement surprenants :

- la confiance est bornée à 0 ;
- la confiance est bornée à 100 ;
- 25,9 % des réponses sont exactement égales à 100 ;
- la distribution est donc structurellement difficile à rendre parfaitement normale.

La décision ne doit pas reposer uniquement sur Shapiro–Wilk.

Nous devons considérer conjointement :

- l’histogramme ;
- le graphique quantile–quantile ;
- le graphique résidus–valeurs ajustées ;
- la proportion de grandes erreurs ;
- les analyses de sensibilité ;
- le modèle logistique de la borne supérieure ;
- la stabilité des coefficients.

> **Conclusion**
>
> L’hypothèse de normalité parfaite est clairement violée, mais les analyses complémentaires permettent de déterminer si les conclusions scientifiques principales restent robustes malgré cette imperfection.

---

# 3.5 Résidus standardisés et observations atypiques

## 3.5.1 Résidu standardisé

Un résidu brut est exprimé dans l’unité de la variable dépendante, ici en points de confiance.

Pour savoir si un résidu est grand relativement à la variabilité habituelle, on peut le standardiser :

\[
r_{ij}
=
\frac{e_{ij}}{\widehat{\sigma}_\varepsilon}
\]

où :

- \(e_{ij}\) est le résidu brut ;
- \(\widehat{\sigma}_\varepsilon\) est l’écart-type résiduel estimé.

Dans notre modèle :

\[
\widehat{\sigma}_\varepsilon \approx 16{,}87
\]

Si un essai possède un résidu de 30 points :

\[
r=\frac{30}{16{,}87}\approx1{,}78
\]

L’erreur correspond à environ 1,78 écart-type résiduel.

## 3.5.2 Seuils de 2 et 3

Des repères couramment utilisés sont :

\[
|r|>2
\]

et :

\[
|r|>3
\]

Ces seuils ne sont pas des lois absolues. Ils servent à identifier les observations qui méritent une inspection.

Résultats :

| Critère | Nombre | Proportion |
|---|---:|---:|
| \(|r|>2\) | 523 | 5,80 % |
| \(|r|>3\) | 178 | 1,97 % |

Sous une loi normale parfaite, on attend approximativement :

- 4,55 % au-delà de \(\pm2\) ;
- 0,27 % au-delà de \(\pm3\).

Notre proportion au-delà de 3 est donc nettement supérieure à celle attendue sous une normalité parfaite.

Cela correspond à l’excès de kurtosis observé : la distribution contient davantage d’erreurs extrêmes.

## 3.5.3 Observation atypique et observation influente

Ces deux notions sont différentes.

### Observation atypique

Une **observation atypique** est une observation mal prédite par le modèle.

Elle possède généralement un grand résidu.

### Observation influente

Une **observation influente** est une observation dont la présence modifie fortement les coefficients du modèle.

Une observation peut être atypique sans être influente.

Exemple :

- un participant produit une réponse de confiance très inhabituelle sur un seul essai ;
- cet essai a un grand résidu ;
- mais parmi 9024 essais, il ne modifie presque pas les coefficients.

Inversement, un participant possédant un profil particulier sur 64 essais peut avoir une influence notable, même si aucun de ses essais n’a un résidu extrêmement grand.

C’est pourquoi le script ne se limite pas aux résidus. Il effectue également une analyse par suppression de participant.

---

# 3.6 Influence et jackknife par participant

## 3.6.1 Définition du jackknife

Le **jackknife** est une méthode de stabilité consistant à retirer successivement une unité de l’échantillon et à refaire l’analyse.

Dans notre cas, l’unité retirée est le participant.

Avec 141 participants :

1. on ajuste le modèle complet avec les 141 participants ;
2. on retire le participant 1 et on réajuste le modèle ;
3. on remet le participant 1 ;
4. on retire le participant 2 et on réajuste le modèle ;
5. on répète l’opération pour les 141 participants.

Nous obtenons donc jusqu’à 141 modèles supplémentaires.

```text
Modèle complet : participants 1 à 141

Modèle jackknife 1 : participants 2 à 141
Modèle jackknife 2 : participants 1, 3, 4, ..., 141
Modèle jackknife 3 : participants 1, 2, 4, ..., 141
...
Modèle jackknife 141 : participants 1 à 140
```

## 3.6.2 Pourquoi retirer un participant entier ?

Les observations d’un même participant ne sont pas indépendantes.

Chaque participant fournit 64 essais. Retirer un seul essai ne permettrait pas d’étudier correctement l’influence globale de cette personne.

Nous retirons donc l’ensemble de ses 64 essais.

## 3.6.3 Intuition

Supposons que l’effet estimé de l’entropie soit :

\[
\widehat{\beta}_{\text{entropie}}=-2{,}49
\]

Si, après retrait d’un participant particulier, l’effet devient :

\[
+0{,}50
\]

cela serait inquiétant. La conclusion dépendrait fortement de cette personne.

Si les 141 modèles produisent plutôt des valeurs comprises entre :

\[
-2{,}53
\quad\text{et}\quad
-2{,}44
\]

alors l’effet est extrêmement stable.

---

# 3.7 Stabilité des coefficients

Le fichier de synthèse jackknife contenait les informations suivantes.

| Paramètre | Estimation complète | Minimum jackknife | Maximum jackknife | Changement de signe |
|---|---:|---:|---:|---|
| Interception | 73,151 | 72,815 | 73,747 | Non |
| Standard | 5,136 | 4,504 | 5,555 | Non |
| Séquence | -0,437 | -0,546 | -0,382 | Non |
| Précision du participant | 0,310 | -0,248 | 0,752 | Oui |
| Entropie de l’item | -2,494 | -2,529 | -2,440 | Non |
| Modèles moyens | -1,801 | -2,110 | -1,136 | Non |
| Modèles intra-individuels | -0,491 | -0,561 | -0,386 | Non |

Ces valeurs provenaient de la version du modèle employée lors de cette analyse diagnostique. Lorsque le modèle est réajusté avec le fichier \(N=20\), les valeurs précises évoluent légèrement, mais le principe d’interprétation reste identique.

## 3.7.1 `full_estimate`

C’est le coefficient estimé avec tous les participants.

## 3.7.2 `jackknife_mean`

C’est la moyenne des 141 coefficients obtenus en retirant successivement chaque participant.

Si cette moyenne est très proche de l’estimation complète, cela suggère que l’estimation globale n’est pas fortement biaisée par une personne particulière.

## 3.7.3 `jackknife_sd`

C’est l’écart-type des coefficients jackknife.

Il mesure leur dispersion.

Une petite valeur indique que les coefficients changent peu lorsque l’on retire un participant.

## 3.7.4 `jackknife_min` et `jackknife_max`

Ces colonnes indiquent les valeurs extrêmes obtenues parmi les modèles réajustés.

Exemple pour l’entropie :

\[
-2{,}529
\leq
\widehat{\beta}_{\text{entropie}}
\leq
-2{,}440
\]

L’effet reste toujours négatif et très proche de l’estimation complète.

## 3.7.5 `maximum_absolute_change`

Cette valeur est calculée comme :

\[
\max_i
\left|
\widehat{\beta}_{(-i)}
-
\widehat{\beta}_{\text{complet}}
\right|
\]

où \(\widehat{\beta}_{(-i)}\) est le coefficient obtenu sans le participant \(i\).

Pour l’entropie :

\[
\text{changement absolu maximal}\approx0{,}054
\]

Cela est très faible relativement à un coefficient d’environ \(-2{,}49\).

## 3.7.6 `sign_change_detected`

Cette colonne indique si le coefficient a changé de signe dans au moins un modèle jackknife.

Par exemple, l’effet de la précision moyenne du participant est faiblement positif dans le modèle complet :

\[
\widehat{\beta}=0{,}310
\]

Mais selon le participant retiré, il varie de :

\[
-0{,}248
\quad\text{à}\quad
0{,}752
\]

Il peut donc devenir négatif ou positif.

Cette instabilité est cohérente avec le fait que cet effet n’était pas clairement détecté dans le modèle principal.

En revanche, l’effet de l’entropie reste toujours négatif. Il est donc très stable.

## 3.7.7 Ce que le jackknife montre scientifiquement

Les conclusions ne dépendent pas de la même manière des différents prédicteurs.

### Très robuste

- entropie de l’item ;
- effet de séquence ;
- différence Standard–Neutral.

### Signe stable, mais effet plus incertain

- nombre moyen de modèles ;
- composante intra-individuelle des modèles.

### Instable autour de zéro

- précision moyenne du participant.

> **Attention**
>
> Un signe stable ne signifie pas automatiquement qu’un effet est statistiquement clairement différent de zéro.
>
> Par exemple, un coefficient peut rester négatif dans toutes les analyses jackknife, tout en conservant un intervalle de confiance qui contient zéro.

---

# 3.8 Calibration métacognitive

## 3.8.1 Qu’est-ce que la métacognition ?

La **métacognition** est la capacité à évaluer ses propres processus cognitifs.

Dans notre projet, elle correspond notamment à la capacité d’un participant à savoir quand sa réponse est probablement correcte ou incorrecte.

Une personne peut :

- répondre correctement ;
- répondre incorrectement ;
- être très confiante ;
- être peu confiante.

La question métacognitive n’est donc pas seulement :

> « Le participant a-t-il raison ? »

mais aussi :

> « Sa confiance permet-elle de distinguer les situations où il a raison de celles où il a tort ? »

## 3.8.2 Qu’est-ce que la calibration ?

La **calibration** compare le niveau de confiance annoncé à la fréquence réelle de réussite.

Supposons qu’une personne dise être confiante à 80 % sur 100 essais.

Une calibration parfaite impliquerait environ :

\[
80\text{ réponses correctes sur }100
\]

Si elle n’en obtient que 60 :

\[
80\%-60\%=20\text{ points}
\]

Elle est surconfiante de 20 points de pourcentage.

## 3.8.3 Transformation de la confiance

La confiance varie de 0 à 100.

Pour la comparer à une probabilité, le script la divise par 100 :

\[
p_i=\frac{\text{confiance}_i}{100}
\]

Ainsi :

| Confiance | Probabilité utilisée |
|---:|---:|
| 0 | 0,00 |
| 50 | 0,50 |
| 75 | 0,75 |
| 100 | 1,00 |

Cette transformation suppose que l’échelle de confiance peut être interprétée directement comme une probabilité subjective.

C’est une hypothèse forte.

Un participant peut utiliser « 80 » pour signifier « assez certain », sans penser exactement à une probabilité de 0,80. Néanmoins, cette transformation permet une première analyse quantitative claire.

## 3.8.4 Exactitude observée

L’exactitude est codée :

\[
y_i=
\begin{cases}
1 & \text{si la réponse est correcte}\\
0 & \text{si la réponse est incorrecte}
\end{cases}
\]

La moyenne de cette variable correspond à la proportion de réponses correctes :

\[
\overline{y}
=
\frac{1}{N}\sum_{i=1}^{N}y_i
\]

Dans nos données :

\[
\overline{y}\approx0{,}623
\]

Donc environ 62,3 % des réponses sont correctes.

## 3.8.5 Confiance moyenne

La confiance moyenne transformée en probabilité est :

\[
\overline{p}\approx0{,}757
\]

Donc la confiance moyenne était de 75,7 %.

## 3.8.6 Biais de calibration

Le biais global est :

\[
\text{Biais}
=
\overline{p}
-
\overline{y}
\]

Dans nos données :

\[
0{,}7574-0{,}6231
=
0{,}1343
\]

Cela correspond à environ :

\[
13{,}4\text{ points de pourcentage}
\]

Les participants étaient donc, en moyenne, surconfiants.

### Interprétation

Ils donnaient en moyenne une confiance de 75,7 %, alors que leur réussite réelle était de 62,3 %.

> **Attention**
>
> Cette conclusion concerne la moyenne globale. Elle ne signifie pas que chaque participant est surconfiant ni que chaque niveau de confiance est mal calibré de la même manière.

## 3.8.7 Calibration par classes de confiance

Pour construire une courbe de calibration, on regroupe généralement les observations dans des intervalles.

Par exemple :

| Classe de confiance | Confiance moyenne | Exactitude observée |
|---|---:|---:|
| 0–10 | 0,07 | 0,45 |
| 10–20 | 0,16 | 0,48 |
| 20–30 | 0,25 | 0,51 |
| … | … | … |
| 90–100 | 0,96 | 0,67 |

Pour chaque classe, le script calcule :

1. le nombre d’observations ;
2. la confiance moyenne ;
3. la proportion de réponses correctes ;
4. la différence entre les deux.

La calibration parfaite correspond à la ligne :

\[
y=x
\]

```text
Exactitude observée
1.0 |                         /
    |                      /
0.8 |                   /
    |                /
0.6 |             /
    |          /
0.4 |       /
    |    /
0.2 | /
    +----------------------------
      0.0                  1.0
          Confiance annoncée
```

Si les points sont sous la diagonale, la confiance est supérieure à l’exactitude : il y a surconfiance.

S’ils sont au-dessus, il y a sous-confiance.

---

# 3.9 Score de Brier

## 3.9.1 Définition

Le **score de Brier** mesure l’écart quadratique entre une probabilité prédite et un résultat binaire.

Pour chaque essai :

\[
(p_i-y_i)^2
\]

Le score global est :

\[
\text{Brier}
=
\frac{1}{N}
\sum_{i=1}^{N}
(p_i-y_i)^2
\]

Dans notre cas :

- \(p_i\) est la confiance divisée par 100 ;
- \(y_i=1\) si la réponse est correcte ;
- \(y_i=0\) sinon.

## 3.9.2 Exemples

### Réponse correcte avec forte confiance

- confiance : 90 % ;
- réponse correcte : \(y=1\).

\[
(0{,}90-1)^2
=
0{,}01
\]

Le score est faible : c’est une bonne évaluation métacognitive.

### Réponse incorrecte avec forte confiance

- confiance : 90 % ;
- réponse incorrecte : \(y=0\).

\[
(0{,}90-0)^2
=
0{,}81
\]

Le score est élevé : le participant était très confiant mais avait tort.

### Réponse correcte avec confiance moyenne

- confiance : 50 % ;
- réponse correcte : \(y=1\).

\[
(0{,}50-1)^2
=
0{,}25
\]

## 3.9.3 Interprétation

Le score de Brier varie théoriquement entre 0 et 1 :

- 0 : prédictions probabilistes parfaites ;
- plus la valeur augmente, plus l’écart entre confiance et exactitude est grand.

Notre résultat est :

\[
\text{Brier}\approx0{,}302
\]

Cette valeur indique une correspondance imparfaite entre la confiance exprimée et l’exactitude.

Cependant, elle ne possède pas un seuil universel séparant automatiquement un « bon » et un « mauvais » score. Elle doit être interprétée relativement :

- au taux de réussite ;
- à une stratégie de référence ;
- à la distribution des réponses ;
- aux autres mesures de calibration.

## 3.9.4 Limite du score de Brier

Le score de Brier combine plusieurs phénomènes :

- calibration ;
- discrimination ;
- incertitude intrinsèque des résultats.

Une personne peut avoir une confiance moyenne assez bien calibrée mais une mauvaise discrimination essai par essai.

Exemple :

- elle répond toujours « 62 % » ;
- sa précision globale est de 62 %.

Sa moyenne est parfaitement calibrée. Mais elle ne distingue jamais les essais faciles des essais difficiles.

C’est pourquoi nous calculons également l’AUC métacognitive et la différence de confiance entre réponses correctes et incorrectes.

---

# 3.10 Discrimination métacognitive et AUC de type 2

## 3.10.1 Calibration et discrimination sont différentes

### Calibration

La calibration demande :

> Les niveaux de confiance correspondent-ils aux fréquences réelles de réussite ?

### Discrimination

La discrimination demande :

> Le participant est-il plus confiant lorsqu’il a raison que lorsqu’il a tort ?

Une personne peut être surconfiante mais bien discriminer ses réponses.

Exemple :

| Type de réponse | Confiance moyenne |
|---|---:|
| Correcte | 90 |
| Incorrecte | 75 |

Elle est peut-être globalement surconfiante, mais elle est néanmoins plus confiante quand elle a raison.

À l’inverse :

| Type de réponse | Confiance moyenne |
|---|---:|
| Correcte | 75 |
| Incorrecte | 75 |

Elle peut être globalement bien calibrée, mais ne possède aucune discrimination métacognitive.

## 3.10.2 Différence de confiance

Le script calcule :

\[
\text{Discrimination}
=
\overline{C}_{\text{correct}}
-
\overline{C}_{\text{incorrect}}
\]

où :

- \(\overline{C}_{\text{correct}}\) est la confiance moyenne sur les réponses correctes ;
- \(\overline{C}_{\text{incorrect}}\) est la confiance moyenne sur les réponses incorrectes.

Résultat moyen entre participants :

\[
1{,}62\text{ point}
\]

La médiane était :

\[
0{,}64\text{ point}
\]

En moyenne, les participants étaient donc seulement légèrement plus confiants lorsqu’ils avaient raison.

La dispersion entre participants était importante :

\[
SD\approx7{,}30
\]

Certains participants avaient même une différence négative : ils étaient en moyenne plus confiants lorsqu’ils avaient tort.

## 3.10.3 AUC métacognitive de type 2

L’**AUC**, ou *Area Under the Curve*, est l’aire sous une courbe ROC.

Dans le cadre métacognitif, l’AUC de type 2 peut être comprise comme :

> La probabilité qu’une réponse correcte choisie au hasard ait reçu une confiance supérieure à une réponse incorrecte choisie au hasard.

### Interprétation

| AUC | Interprétation |
|---:|---|
| 0,50 | Niveau du hasard |
| > 0,50 | Les réponses correctes tendent à recevoir plus de confiance |
| 1,00 | Discrimination parfaite |
| < 0,50 | Les réponses incorrectes tendent à recevoir plus de confiance |

La moyenne observée était :

\[
\text{AUC}\approx0{,}522
\]

La médiane était :

\[
\text{AUC médiane}\approx0{,}511
\]

Ces valeurs sont très proches de 0,50.

Cela suggère une faible capacité métacognitive à distinguer, par la confiance, les réponses correctes des réponses incorrectes.

## 3.10.4 Pourquoi seulement 140 participants ?

Le fichier indiquait :

\[
n=140
\]

pour l’AUC, alors qu’il existe 141 participants.

Pour calculer une AUC, il faut qu’un participant possède :

- au moins une réponse correcte ;
- au moins une réponse incorrecte.

Si un participant donne uniquement des réponses correctes, il est impossible de comparer ses confiances correctes à ses confiances incorrectes.

Son AUC est donc manquante.

Cela ne signifie pas qu’il a été supprimé du reste des analyses.

---

# 3.11 Modèle logistique de l’exactitude

## 3.11.1 Pourquoi ajouter un modèle statistique ?

Les mesures descriptives indiquent que l’AUC moyenne est proche de 0,50. Mais nous voulons aussi savoir si, essai par essai, une confiance plus élevée est associée à une probabilité plus élevée de répondre correctement.

La variable dépendante devient :

\[
\texttt{is\_correct}
\]

avec :

\[
\texttt{is\_correct}
=
\begin{cases}
1 & \text{réponse correcte}\\
0 & \text{réponse incorrecte}
\end{cases}
\]

Comme cette variable est binaire, nous utilisons un modèle logistique mixte.

La formule est approximativement :

```python
is_correct ~ (
    C(condition, Treatment(reference="Neutral"))
    + confidence_z
    + sequence_c10
)
```

avec des intercepts aléatoires pour :

- les participants ;
- les items.

## 3.11.2 Pourquoi standardiser la confiance ?

La variable `confidence_z` est une confiance standardisée :

\[
\text{confidence\_z}
=
\frac{\text{confidence}-\overline{\text{confidence}}}
{s_{\text{confidence}}}
\]

Le coefficient représente donc le changement des log-odds de réponse correcte pour une augmentation d’un écart-type de confiance.

Cela facilite l’interprétation et l’optimisation numérique.

## 3.11.3 Résultat pour la confiance

Le résultat principal était :

\[
\beta_{\text{confiance}}
=
-0{,}00785
\]

avec un odds ratio :

\[
OR
=
e^{-0{,}00785}
\approx0{,}992
\]

et un intervalle crédible à 95 % :

\[
[0{,}942;\ 1{,}046]
\]

## 3.11.4 Interprétation

Un odds ratio égal à 1 signifie aucune association.

Ici :

\[
OR\approx0{,}992
\]

est extrêmement proche de 1.

De plus, l’intervalle crédible contient 1.

Les données ne fournissent donc pas d’indication claire qu’une augmentation d’un écart-type de confiance soit associée à une meilleure exactitude.

Cela rejoint :

- l’AUC moyenne proche de 0,50 ;
- la faible différence de confiance entre réponses correctes et incorrectes.

## 3.11.5 Ce résultat signifie-t-il que la confiance est totalement aléatoire ?

Pas nécessairement.

Le résultat signifie qu’après avoir pris en compte :

- la condition ;
- la position de l’essai ;
- les différences moyennes entre participants ;
- les différences moyennes entre items ;

la confiance ne fournit pas ici un signal clair sur l’exactitude essai par essai.

Il est possible que :

- certains participants aient une bonne métacognition et d’autres non ;
- la relation soit non linéaire ;
- l’échelle 0–100 ne soit pas utilisée comme une probabilité ;
- les nombreuses valeurs égales à 100 réduisent la finesse de discrimination ;
- la tâche encourage une forte confiance indépendamment de l’exactitude.

## 3.11.6 Nature bayésienne approchée du résultat

Comme pour le modèle de la borne supérieure, `BinomialBayesMixedGLM` utilise ici une estimation bayésienne variationnelle.

Les colonnes sont donc :

- moyenne postérieure ;
- écart-type postérieur ;
- intervalle crédible ;
- odds ratio.

Ce ne sont pas exactement les mêmes objets qu’un coefficient ML, une erreur standard fréquentiste et une valeur \(p\).

Le modèle a convergé :

| Élément | Valeur |
|---|---:|
| Succès | `True` |
| Itérations | 171 |
| Gradient absolu maximal | \(8,20\times10^{-6}\) |

Un gradient très proche de zéro indique que l’algorithme est arrivé près d’un optimum de sa fonction objectif.

---

# 3.12 Graphique ajusté de l’entropie

Le script produit également :

```text
adjusted_entropy_confidence.png
```

L’objectif est d’illustrer l’effet principal le plus robuste du modèle : la relation négative entre l’entropie de l’item et la confiance.

## 3.12.1 Pourquoi ne pas simplement tracer les données brutes ?

Une relation brute entre entropie et confiance peut être influencée par :

- la condition ;
- la position de l’essai ;
- les différences entre participants ;
- le nombre de modèles mentaux ;
- les différences entre items.

Le graphique ajusté utilise les coefficients du modèle pour représenter la confiance prédite lorsque les autres variables sont fixées à des valeurs de référence.

Par exemple :

- condition fixée à Neutral ou représentée séparément ;
- séquence fixée à sa moyenne ;
- précision du participant fixée à sa moyenne ;
- nombre de modèles fixé à sa moyenne ;
- effets aléatoires fixés à zéro.

La prédiction représente alors un participant moyen et un item moyen.

## 3.12.2 Construction mathématique

Supposons que toutes les autres variables standardisées soient fixées à zéro.

La prédiction devient approximativement :

\[
\widehat{Y}
=
\widehat{\beta}_0
+
\widehat{\beta}_{\text{entropie}}
\times
\text{entropie}_z
\]

Avec un coefficient d’environ :

\[
\widehat{\beta}_{\text{entropie}}
=
-2{,}49
\]

une augmentation d’un écart-type d’entropie est associée à une diminution prédite d’environ 2,49 points de confiance.

## 3.12.3 Interprétation scientifique

Une entropie élevée signifie que les participants sont plus divisés entre les réponses « Yes » et « No ».

Une entropie faible signifie qu’ils répondent majoritairement de la même manière.

Le résultat montre que :

> Plus un item provoque un désaccord collectif, plus la confiance moyenne individuelle tend à être faible.

Cela suggère que la dispersion des réponses entre participants reflète une difficulté ou une ambiguïté psychologique ressentie au niveau individuel.

## 3.12.4 Prudence causale

L’entropie est calculée à partir des réponses du groupe.

Elle n’est pas manipulée expérimentalement.

Nous pouvons donc dire :

> L’entropie est associée négativement à la confiance.

Il est plus prudent de ne pas dire :

> L’entropie cause directement une diminution de la confiance.

L’entropie peut représenter d’autres propriétés de l’item :

- difficulté ;
- ambiguïté ;
- diversité des stratégies ;
- conflit entre logique et croyance ;
- formulation linguistique.

---

# 3.13 Fichiers produits

Les noms exacts peuvent dépendre de la version du script, mais les catégories principales sont les suivantes.

## 3.13.1 Résumé de normalité des résidus

Exemple :

```text
final_model_residual_normality.csv
```

Contenu :

- nombre de résidus ;
- moyenne ;
- écart-type ;
- asymétrie ;
- kurtosis ;
- statistique de Shapiro–Wilk ;
- valeur \(p\).

Utilité :

- documenter la forme de la distribution résiduelle ;
- identifier les écarts à la normalité.

## 3.13.2 Résumé des résidus standardisés

Exemple :

```text
final_model_standardized_residual_summary.csv
```

Contenu :

- nombre d’observations ;
- nombre et proportion de résidus dépassant 2 ;
- nombre et proportion dépassant 3.

Utilité :

- quantifier les observations très mal prédites.

## 3.13.3 Résidus par observation

Exemple :

```text
final_model_observation_diagnostics.csv
```

Chaque ligne peut contenir :

- identifiant du participant ;
- identifiant de l’item ;
- séquence ;
- confiance observée ;
- confiance prédite ;
- résidu ;
- résidu standardisé.

Ce fichier permet d’identifier précisément les essais ayant les plus grandes erreurs.

## 3.13.4 Résumé jackknife

Exemple :

```text
final_model_subject_jackknife_summary.csv
```

Contenu :

- estimation complète ;
- moyenne jackknife ;
- écart-type jackknife ;
- minimum ;
- maximum ;
- changement absolu maximal ;
- changement de signe ;
- nombre de modèles réussis.

Utilité :

- vérifier si un participant contrôle excessivement un coefficient.

## 3.13.5 Calibration globale

Exemple :

```text
overall_metacognitive_calibration.csv
```

Contenu :

- confiance moyenne ;
- exactitude observée ;
- biais de calibration ;
- erreur absolue de calibration ;
- score de Brier.

## 3.13.6 Calibration par participant

Exemple :

```text
subject_metacognitive_calibration.csv
```

Une ligne correspond à un participant.

Colonnes possibles :

- précision ;
- confiance moyenne ;
- biais de calibration ;
- score de Brier ;
- AUC de type 2 ;
- discrimination de confiance.

Ce fichier permet d’étudier les différences individuelles.

## 3.13.7 Modèle logistique d’exactitude

Exemple :

```text
metacognitive_accuracy_logistic_fixed_effects.csv
```

Contenu :

- moyenne postérieure en log-odds ;
- écart-type postérieur ;
- intervalle crédible ;
- odds ratio ;
- intervalle crédible de l’odds ratio.

## 3.13.8 Graphiques

### `final_model_residuals_vs_fitted.png`

- axe horizontal : confiance prédite ;
- axe vertical : résidu ;
- un point représente un essai.

On recherche :

- un nuage approximativement centré autour de zéro ;
- l’absence de courbure systématique ;
- une dispersion qui ne change pas excessivement.

### `final_model_residual_qqplot.png`

Ce graphique compare les quantiles observés des résidus aux quantiles attendus sous une loi normale.

- alignement sur la diagonale : normalité approximative ;
- écart aux extrémités : queues trop lourdes ou trop légères ;
- courbure asymétrique : asymétrie.

### `metacognitive_calibration_curve.png`

- axe horizontal : confiance moyenne ;
- axe vertical : exactitude observée ;
- diagonale : calibration parfaite.

### `subject_mean_calibration.png`

Chaque point représente généralement un participant :

- axe horizontal : confiance moyenne du participant ;
- axe vertical : précision du participant.

### `subject_type2_auc_distribution.png`

Montre la distribution des AUC métacognitives entre participants.

Une ligne à 0,50 représente le hasard.

### `adjusted_entropy_confidence.png`

Montre la confiance prédite en fonction de l’entropie, les autres prédicteurs étant contrôlés.

---

# 4. `compare_mreasoner_simulations_E1.py`

# 4.1 Pourquoi comparer 3, 10 et 20 simulations ?

MReasoner est un modèle computationnel qui peut produire des résultats variables d’une simulation à l’autre.

Nous avions initialement utilisé seulement trois simulations pour estimer le nombre de modèles mentaux générés.

Pour une combinaison donnée :

```text
participant × type de tâche
```

nous disposions de trois valeurs simulées, puis de leur moyenne.

Exemple hypothétique :

```text
Simulation 1 : 2 modèles
Simulation 2 : 7 modèles
Simulation 3 : 2 modèles
```

La moyenne serait :

\[
\frac{2+7+2}{3}
=
3{,}67
\]

Mais cette moyenne peut être instable. Avec seulement trois simulations, une réalisation atypique a beaucoup de poids.

Nous avons donc relancé les simulations avec :

- 10 répétitions ;
- 20 répétitions.

L’objectif était de répondre à la question suivante :

> Les estimations du nombre moyen de modèles mentaux et les conclusions statistiques dépendent-elles fortement du petit nombre initial de simulations ?

---

# 4.2 Ce que signifie le nombre de simulations

## 4.2.1 Simulation et participant ne sont pas la même chose

Une simulation n’est pas un participant supplémentaire.

Pour une même combinaison participant–tâche, MReasoner est exécuté plusieurs fois afin d’observer sa variabilité interne.

Avec \(N=20\), nous demandons vingt réalisations computationnelles au lieu de trois.

## 4.2.2 Moyenne simulée

Si les résultats sont :

\[
M_1,M_2,\ldots,M_N
\]

la moyenne est :

\[
\overline{M}
=
\frac{1}{N}
\sum_{s=1}^{N}M_s
\]

Avec davantage de simulations, cette moyenne tend généralement à mieux approcher la moyenne théorique du processus générateur.

## 4.2.3 Erreur standard de la moyenne simulée

Supposons que l’écart-type des simulations soit \(s\).

L’incertitude de la moyenne est approximativement :

\[
SE(\overline{M})
=
\frac{s}{\sqrt{N}}
\]

Exemple avec :

\[
s=2
\]

### Trois simulations

\[
SE
=
\frac{2}{\sqrt{3}}
\approx1{,}15
\]

### Vingt simulations

\[
SE
=
\frac{2}{\sqrt{20}}
\approx0{,}45
\]

La moyenne fondée sur 20 simulations est donc normalement plus précise.

## 4.2.4 Point subtil : l’écart-type observé peut augmenter

Il ne faut pas s’attendre obligatoirement à ce que `std_models_generated` diminue lorsque le nombre de simulations augmente.

Avec davantage de simulations, on peut observer des résultats rares qui n’apparaissaient pas dans les trois premiers tirages. L’écart-type brut des simulations peut donc augmenter.

Ce qui diminue avec \(N\), toutes choses égales par ailleurs, est l’incertitude sur la moyenne :

\[
\frac{s}{\sqrt{N}}
\]

et non nécessairement \(s\) lui-même.

C’est une distinction très importante.

---

# 4.3 Statistiques calculées

Le script charge les fichiers de simulation correspondant à :

- 3 simulations ;
- 10 simulations ;
- 20 simulations.

Il harmonise ensuite les colonnes et détermine le type de tâche à partir des prémisses.

Chaque combinaison est identifiée par :

```text
subject_id + task_type
```

Il existe :

\[
151\times4=604
\]

combinaisons dans les fichiers MReasoner.

Le script calcule ensuite plusieurs indicateurs.

## 4.3.1 `mean_models`

Moyenne du nombre de modèles générés.

## 4.3.2 `sd_between_combinations`

Écart-type des moyennes entre les différentes combinaisons participant–tâche.

Il mesure l’hétérogénéité entre les combinaisons.

## 4.3.3 `mean_simulation_sd`

Moyenne des écarts-types internes aux simulations.

Pour chaque combinaison participant–tâche :

1. on calcule l’écart-type des \(N\) simulations ;
2. on fait la moyenne de ces écarts-types.

Cette quantité mesure la variabilité interne moyenne de MReasoner.

## 4.3.4 `median_simulation_sd`

Médiane des écarts-types internes.

La médiane est moins sensible que la moyenne aux très grandes valeurs.

## 4.3.5 `mean_range`

Pour une combinaison donnée, l’amplitude est :

\[
\max(M_s)-\min(M_s)
\]

`mean_range` est la moyenne de ces amplitudes.

## 4.3.6 `zero_sd_rate`

Proportion de combinaisons dont toutes les simulations donnent le même résultat.

Si :

```text
2, 2, 2, 2, 2
```

alors :

\[
SD=0
\]

Si :

```text
2, 2, 3, 2, 2
```

alors l’écart-type est supérieur à zéro.

---

# 4.4 Résultats par type de tâche

## 4.4.1 Tâche MT

Pour MT :

\[
\text{mean\_models}=2
\]

avec :

\[
SD=0
\]

pour 3, 10 et 20 simulations.

MReasoner produit donc ici toujours le même nombre de modèles.

La tâche MT est parfaitement stable du point de vue de cette mesure.

## 4.4.2 Tâche AC

Moyennes :

| Simulations | Moyenne des modèles |
|---:|---:|
| 3 | 2,662 |
| 10 | 2,592 |
| 20 | 2,530 |

La moyenne diminue légèrement quand le nombre de simulations augmente.

La proportion d’écarts-types nuls passe de :

| Simulations | Proportion de SD nuls |
|---:|---:|
| 3 | 84,1 % |
| 10 | 64,9 % |
| 20 | 0 % |

Cela ne signifie pas nécessairement que le modèle est devenu moins stable.

Avec 20 essais computationnels, nous avons davantage de chances d’observer au moins une variation rare. Il devient donc beaucoup plus difficile d’obtenir exactement 20 fois la même sortie.

## 4.4.3 Tâche MP

Moyennes :

| Simulations | Moyenne des modèles |
|---:|---:|
| 3 | 2,322 |
| 10 | 2,362 |
| 20 | 2,403 |

L’évolution est modérée.

La plupart des valeurs restent proches de 2 ou 3 modèles.

## 4.4.4 Tâche DA

Moyennes :

| Simulations | Moyenne des modèles |
|---:|---:|
| 3 | 4,018 |
| 10 | 4,941 |
| 20 | 4,814 |

DA est clairement la tâche la plus variable.

Pour \(N=20\) :

- écart-type interne moyen : environ 1,818 ;
- amplitude moyenne : environ 5,848 ;
- aucune combinaison ne possède un écart-type nul.

Cela signifie que MReasoner peut produire des nombres de modèles très différents pour cette structure de tâche.

## 4.4.5 Conclusion par type de tâche

| Tâche | Stabilité |
|---|---|
| MT | Parfaitement stable |
| AC | Relativement stable, mais des variations rares apparaissent avec davantage de simulations |
| MP | Variabilité modérée |
| DA | Variabilité importante |

La qualité d’une estimation de MReasoner dépend donc fortement du type de tâche.

---

# 4.5 Comparaisons 3–10–20

Le script place les estimations côte à côte pour les 604 combinaisons participant–tâche.

Pour chaque comparaison, il calcule :

- corrélation de Pearson ;
- corrélation de Spearman ;
- différence moyenne ;
- différence absolue moyenne ;
- erreur quadratique moyenne ;
- différence absolue maximale ;
- proportion de différences supérieures à 0,25 ;
- proportion de différences supérieures à 0,50.

## 4.5.1 Corrélation de Pearson

La **corrélation de Pearson** mesure la force d’une relation linéaire entre deux séries de valeurs.

Elle varie entre \(-1\) et \(1\).

| Valeur | Interprétation |
|---:|---|
| 1 | Relation linéaire positive parfaite |
| 0 | Pas de relation linéaire |
| -1 | Relation linéaire négative parfaite |

Une corrélation élevée signifie que les combinaisons ayant une grande estimation avec \(N=3\) tendent aussi à avoir une grande estimation avec \(N=20\).

Mais une corrélation élevée ne garantit pas que les valeurs sont identiques.

Exemple :

\[
Y=X+10
\]

donne une corrélation parfaite de 1, même si toutes les valeurs diffèrent de 10.

## 4.5.2 Corrélation de Spearman

La **corrélation de Spearman** compare principalement le classement des observations.

Elle demande :

> Les combinaisons classées parmi les plus élevées dans une simulation le restent-elles dans l’autre ?

Elle est moins dépendante d’une relation strictement linéaire.

## 4.5.3 Différence moyenne

Pour une comparaison \(A\) contre \(B\) :

\[
\text{différence moyenne}
=
\frac{1}{K}
\sum_{k=1}^{K}
(A_k-B_k)
\]

Une différence proche de zéro indique peu de biais systématique.

Mais les différences positives et négatives peuvent s’annuler.

## 4.5.4 Différence absolue moyenne

\[
MAE
=
\frac{1}{K}
\sum_{k=1}^{K}
|A_k-B_k|
\]

Elle indique de combien les deux estimations diffèrent en moyenne, sans annulation des signes.

## 4.5.5 Différence quadratique moyenne

\[
RMSD
=
\sqrt{
\frac{1}{K}
\sum_{k=1}^{K}
(A_k-B_k)^2
}
\]

Elle pénalise davantage les grandes différences que la différence absolue moyenne.

## 4.5.6 Résultats

| Comparaison | Pearson | Spearman | Différence absolue moyenne | RMSD |
|---|---:|---:|---:|---:|
| 3 contre 10 | 0,935 | 0,859 | 0,315 | 0,577 |
| 3 contre 20 | 0,937 | 0,786 | 0,384 | 0,581 |
| 10 contre 20 | 0,981 | 0,931 | 0,187 | 0,282 |

## 4.5.7 Interprétation

### Comparaison 10 contre 20

C’est la comparaison la plus stable :

- Pearson : 0,981 ;
- Spearman : 0,931 ;
- différence absolue moyenne : 0,187 ;
- RMSD : 0,282.

Les estimations à 10 et à 20 simulations sont donc très proches.

### Comparaison 3 contre 20

La corrélation de Pearson reste élevée :

\[
r\approx0{,}937
\]

Cela signifie que la structure générale est préservée.

Cependant :

- la corrélation de classement est moins élevée ;
- la différence absolue moyenne est plus forte ;
- 46,7 % des combinaisons diffèrent de plus de 0,25 ;
- 27,8 % diffèrent de plus de 0,50.

Trois simulations permettent donc d’obtenir une approximation générale raisonnable, mais insuffisamment précise pour certaines combinaisons.

## 4.5.8 Pourquoi une corrélation élevée n’est-elle pas suffisante ?

Prenons :

```text
N=3  : 2, 3, 4, 5
N=20 : 3, 4, 5, 6
```

La corrélation est parfaite :

\[
r=1
\]

Mais chaque estimation diffère de 1.

Or nos prédicteurs statistiques reposent sur les valeurs numériques, pas seulement sur leur classement.

Nous devons donc regarder à la fois :

- les corrélations ;
- les différences absolues ;
- les conséquences sur les coefficients du modèle mixte.

---

# 4.6 Pourquoi reconstruire le dataset avec \(N=20\) ?

Le fichier initial :

```text
dataset_analysis_E1.csv
```

contenait des colonnes issues du fichier MReasoner à trois simulations :

- `number_models_generated` ;
- `std_models_generated` ;
- `minimum_models_generated` ;
- `maximum_models_generated` ;
- `subject_mean_models` ;
- `models_within_subject`.

Si nous produisons un nouveau fichier MReasoner avec 20 simulations, les valeurs de ces colonnes peuvent changer.

Il ne suffit donc pas de conserver l’ancien dataset et de dire que MReasoner a été relancé.

Il faut refaire la fusion.

Le nouveau pipeline devient :

```text
mental_models_count_E1_n20.csv
                │
                ▼
script de construction du dataset
                │
                ▼
dataset_analysis_E1_n20.csv
                │
                ▼
réajustement des modèles statistiques
```

## 4.6.1 Pourquoi `subject_mean_models` change-t-il ?

Pour le participant \(i\), la moyenne interindividuelle est calculée à partir des nouvelles moyennes de MReasoner :

\[
\overline{M}_i
=
\frac{1}{4}
\sum_{t=1}^{4}M_{it}
\]

Si les valeurs \(M_{it}\) changent, la moyenne du participant change également.

## 4.6.2 Pourquoi `models_within_subject` change-t-il ?

La composante intra-individuelle est :

\[
M_{it}^{\text{within}}
=
M_{it}
-
\overline{M}_i
\]

Elle dépend donc à la fois :

- de la nouvelle estimation pour la tâche ;
- de la nouvelle moyenne personnelle.

Il est indispensable de la recalculer.

## 4.6.3 Devions-nous remplacer le fichier original ?

Il est préférable de conserver deux fichiers séparés :

```text
dataset_analysis_E1.csv
dataset_analysis_E1_n20.csv
```

Cela permet :

- de reproduire l’analyse initiale ;
- de comparer \(N=3\) à \(N=20\) ;
- d’éviter d’écraser silencieusement les résultats ;
- de conserver l’historique scientifique du projet.

Le fichier \(N=20\) devient ensuite la base de l’analyse finale.

---

# 4.7 Stabilité du modèle statistique entre \(N=3\) et \(N=20\)

Nous avons comparé les coefficients du modèle cognitif obtenu avec :

- les estimations à 3 simulations ;
- les estimations à 20 simulations.

## 4.7.1 Résumé

| Paramètre | Estimation \(N=3\) | Estimation \(N=20\) | Changement de signe | Changement de significativité |
|---|---:|---:|---|---|
| Standard | 5,150 | 5,261 | Non | Non |
| Interception | 72,805 | 72,725 | Non | Non |
| Entropie | -2,437 | -2,430 | Non | Non |
| Modèles intra | -0,366 | -0,342 | Non | Non |
| Séquence | -0,437 | -0,437 | Non | Non |
| Précision | 0,310 | 0,696 | Non | Non |
| Modèles moyens | -1,801 | -2,241 | Non | Non |
| Validité | 0,678 | 0,726 | Non | Non |

## 4.7.2 Résultats très stables

### Entropie

\[
-2{,}437
\rightarrow
-2{,}430
\]

Le changement est minuscule.

### Séquence

\[
-0{,}4369
\rightarrow
-0{,}4374
\]

Le changement est presque nul.

### Condition

\[
5{,}150
\rightarrow
5{,}261
\]

L’effet reste positif et statistiquement détecté.

## 4.7.3 Prédicteurs MReasoner

### Nombre moyen de modèles

\[
-1{,}801
\rightarrow
-2{,}241
\]

Le changement relatif est plus visible, mais :

- le signe reste négatif ;
- l’effet reste non clairement différent de zéro ;
- la conclusion générale ne change pas.

### Composante intra-individuelle

Dans cette comparaison particulière, avec `validity_binary` dans le modèle :

\[
-0{,}366
\rightarrow
-0{,}342
\]

L’effet reste négatif mais non statistiquement détecté.

Dans la spécification finale sans validité, l’estimation \(N=20\) est devenue approximativement :

\[
-0{,}485
\]

avec :

\[
p\approx0{,}036
\]

Cette différence doit être correctement comprise :

- elle n’est pas seulement due au passage de 3 à 20 simulations ;
- elle dépend également de la spécification du modèle, notamment du retrait de `validity_binary`.

La conclusion prudente est donc :

> L’effet intra-individuel de MReasoner est négatif, mais il est plus sensible à la spécification du modèle que l’effet de l’entropie.

## 4.7.4 Conclusion de robustesse computationnelle

Le passage à 20 simulations :

- améliore la précision des estimations de MReasoner ;
- modifie certaines valeurs individuelles ;
- ne change pas les signes généraux des coefficients ;
- ne change pas les principales conclusions ;
- confirme l’extrême robustesse de l’effet d’entropie ;
- montre que les effets MReasoner sont plus faibles et plus sensibles.

---

# 4.8 Fichiers produits

## 4.8.1 `mreasoner_stability_summary.csv`

Une ligne correspond à une combinaison :

```text
nombre de simulations × type de tâche
```

Il contient :

- nombre de combinaisons ;
- moyenne du nombre de modèles ;
- variabilité entre combinaisons ;
- variabilité interne ;
- amplitude ;
- proportion de résultats constants.

## 4.8.2 Tableau des comparaisons

Un fichier tel que :

```text
mreasoner_simulation_comparisons.csv
```

contient :

- comparaison 3–10 ;
- comparaison 3–20 ;
- comparaison 10–20 ;
- corrélations ;
- différences moyennes ;
- différences absolues ;
- RMSD ;
- proportions dépassant certains seuils.

## 4.8.3 Table large participant–tâche

Un fichier peut contenir :

| subject_id | task_type | models_n3 | models_n10 | models_n20 |
|---:|---|---:|---:|---:|

Il permet d’inspecter directement les combinaisons qui changent le plus.

## 4.8.4 Graphiques de comparaison

Les graphiques peuvent représenter :

- \(N=3\) contre \(N=20\) ;
- \(N=10\) contre \(N=20\) ;
- la diagonale \(y=x\) ;
- la distribution des différences ;
- les différences par type de tâche.

Un point éloigné de la diagonale correspond à une combinaison participant–tâche pour laquelle le nombre estimé de modèles change fortement.

---

# 5. `build_final_report_E1.py`

# 5.1 Pourquoi automatiser le rapport ?

À ce stade, les résultats sont répartis dans de nombreux fichiers :

```text
CSV des coefficients
CSV des composantes de variance
CSV de calibration
CSV des diagnostics
CSV de robustesse
graphiques
résumés textuels
fichiers JSON
```

Copier manuellement chaque valeur dans un rapport présente plusieurs risques :

- erreur de transcription ;
- oubli d’une mise à jour ;
- mélange entre résultats \(N=3\) et \(N=20\) ;
- différence entre une version ancienne et une version récente ;
- incohérence entre les tableaux et le texte.

Le script `build_final_report_E1.py` automatise cette collecte.

> Il agit comme un secrétaire qui ouvre les différents dossiers, extrait les nombres demandés et les place dans une structure de rapport.

---

# 5.2 Ce que fait réellement ce script

Ce script n’ajuste généralement pas de nouveau modèle.

Il ne recalcule pas les coefficients à partir des 9024 observations.

Il :

1. ouvre les fichiers de résultats ;
2. vérifie leur existence ;
3. récupère certaines lignes et colonnes ;
4. formate les nombres ;
5. construit une liste de paragraphes Markdown ;
6. enregistre cette liste dans un fichier `.md`.

La distinction est importante :

```text
Scripts de modélisation
    → produisent les résultats statistiques

build_final_report_E1.py
    → lit et présente ces résultats
```

Si un coefficient est incorrect dans un fichier source, le script de rapport reproduira cette erreur. Il n’évalue pas lui-même la validité scientifique des valeurs.

---

# 5.3 Organisation du code

## 5.3.1 Définition des chemins

Le script commence généralement par définir :

```python
BASE_DIR = Path(__file__).resolve().parent
```

Cette ligne récupère le dossier contenant le script.

Puis il construit les chemins vers :

- les résultats du modèle final ;
- les diagnostics ;
- les résultats de calibration ;
- les résultats de robustesse ;
- le dossier des figures ;
- le rapport de sortie.

L’utilisation de `pathlib.Path` facilite la construction de chemins indépendamment du système.

Exemple :

```python
report_path = BASE_DIR / "final_report_E1.md"
```

L’opérateur `/` ne signifie pas ici une division. Avec un objet `Path`, il sert à joindre des éléments de chemin.

## 5.3.2 Fonction de lecture protégée

Une fonction peut ressembler à :

```python
def read_csv_if_exists(path):
    if not path.exists():
        return None
    return pd.read_csv(path)
```

Pourquoi ?

Parce qu’un fichier peut être absent.

Sans cette vérification, le script s’arrêterait immédiatement avec une erreur telle que :

```text
FileNotFoundError
```

Avec la fonction protégée, le script peut :

- afficher un avertissement ;
- ignorer une section ;
- poursuivre la génération du rapport.

## 5.3.3 Fonction de formatage

Une fonction telle que :

```python
def format_number(value, digits=3):
    if pd.isna(value):
        return "NA"
    return f"{value:.{digits}f}"
```

sert à transformer un nombre informatique en chaîne lisible.

Par exemple :

```python
5.247192846
```

devient :

```text
5.247
```

Pourquoi est-ce utile ?

Parce qu’un rapport scientifique ne doit généralement pas afficher quinze décimales inutiles.

## 5.3.4 Recherche d’un coefficient

Le CSV des effets fixes contient une colonne `parameter`.

Le script peut utiliser une fonction de ce type :

```python
def find_parameter(table, parameter_name):
    row = table.loc[table["parameter"] == parameter_name]
    if row.empty:
        return None
    return row.iloc[0]
```

Elle cherche la ligne correspondant à un prédicteur particulier.

Exemple :

```python
find_parameter(
    fixed_effects,
    "item_entropy_z"
)
```

renvoie la ligne contenant :

- l’estimation ;
- l’erreur standard ;
- la valeur \(p\) ;
- l’intervalle de confiance.

Sans cette fonction, il faudrait connaître manuellement le numéro de ligne du coefficient. Cela serait fragile, car l’ordre des paramètres peut changer.

## 5.3.5 Construction du Markdown

Le script crée généralement une liste :

```python
lines = []
```

Puis ajoute les sections :

```python
lines.append("# Analyse de l’expérience E1")
lines.append("")
lines.append("## Méthodes")
```

Chaque élément correspond à une ligne du futur fichier Markdown.

Pour ajouter un résultat :

```python
lines.append(
    f"- **Entropie de l’item** : "
    f"β = {estimate:.3f}, "
    f"SE = {standard_error:.3f}, "
    f"IC 95 % [{lower:.3f}, {upper:.3f}], "
    f"p = {p_value:.3g}."
)
```

Le préfixe `f` crée une **f-string**, c’est-à-dire une chaîne dans laquelle on peut insérer directement des variables Python.

Exemple :

```python
estimate = -2.487
f"β = {estimate:.3f}"
```

produit :

```text
β = -2.487
```

## 5.3.6 Enregistrement

À la fin :

```python
report_path.write_text(
    "\n".join(lines),
    encoding="utf-8"
)
```

`"\n".join(lines)` assemble les lignes en les séparant par des retours à la ligne.

`encoding="utf-8"` garantit que les caractères tels que :

- é ;
- à ;
- β ;
- × ;

sont enregistrés correctement.

---

# 5.4 Limites d’un rapport automatique

## 5.4.1 Le script ne comprend pas les résultats

Il peut insérer :

```text
p = 0.036
```

mais il ne comprend pas automatiquement :

- si cette valeur est robuste ;
- si le modèle est bien spécifié ;
- si le coefficient est scientifiquement important ;
- si le résultat dépend du retrait de la validité ;
- si l’association est causale.

## 5.4.2 Risque de mélange entre versions

Si le script pointe accidentellement vers :

```text
dataset_analysis_E1.csv
```

au lieu de :

```text
dataset_analysis_E1_n20.csv
```

il peut intégrer des résultats fondés sur trois simulations.

De même, il faut vérifier que les fichiers de diagnostics correspondent bien au même modèle que celui présenté dans la section principale.

## 5.4.3 Un rapport automatique est une première version

La bonne procédure est :

```text
1. Génération automatique
2. Vérification des valeurs
3. Vérification de la cohérence entre sections
4. Réécriture pédagogique
5. Ajout des limites
6. Contrôle scientifique final
```

Le rapport produit par le script doit être considéré comme une base structurée, pas comme un texte scientifique définitivement validé.

## 5.4.4 Exemple d’ambiguïté à éviter

Le rapport final dit :

> Les résultats concernant le nombre de modèles mentaux étaient plus faibles et devront être réévalués avec davantage de simulations.

Mais nous avons précisément réalisé les analyses à 10 et 20 simulations.

La formulation doit donc être actualisée. Une formulation plus exacte serait :

> Les estimations fondées sur 20 simulations ont confirmé le signe général des associations observées avec trois simulations. Cependant, les effets liés au nombre de modèles mentaux étaient plus faibles et plus sensibles à la spécification statistique que l’effet de l’entropie. Des simulations encore plus nombreuses ou une estimation directe de l’incertitude computationnelle pourraient améliorer leur précision.

---

# 6. Enchaînement complet de l’étape 9

Voici la logique opérationnelle complète.

## 6.1 Première phase : simulations supplémentaires

```text
MReasoner
   │
   ├── 3 simulations
   ├── 10 simulations
   └── 20 simulations
```

## 6.2 Deuxième phase : comparaison computationnelle

```text
compare_mreasoner_simulations_E1.py
   │
   ├── stabilité par tâche
   ├── corrélations entre N
   ├── différences absolues
   └── identification des tâches instables
```

## 6.3 Troisième phase : reconstruction du dataset

```text
mental_models_count_E1_n20.csv
   │
   ▼
construction du dataset analytique
   │
   ▼
dataset_analysis_E1_n20.csv
```

## 6.4 Quatrième phase : réajustement du modèle final

```text
dataset_analysis_E1_n20.csv
   │
   ▼
modèle linéaire mixte final
```

## 6.5 Cinquième phase : diagnostics et calibration

```text
run_final_diagnostics_calibration_E1.py
   │
   ├── résidus
   ├── normalité
   ├── observations atypiques
   ├── jackknife
   ├── calibration
   ├── score de Brier
   ├── AUC métacognitive
   └── modèle logistique de l’exactitude
```

## 6.6 Sixième phase : rapport

```text
Tous les CSV et graphiques
          │
          ▼
build_final_report_E1.py
          │
          ▼
final_report_E1.md
```

---

# 7. Ce que cette étape nous a appris

## 7.1 Sur le modèle linéaire

Le modèle linéaire n’est pas parfaitement conforme à l’hypothèse de résidus normaux :

- asymétrie négative ;
- queues lourdes ;
- test de Shapiro–Wilk très significatif ;
- davantage de résidus extrêmes qu’attendu sous une loi normale parfaite.

Cela s’explique en partie par :

- la borne inférieure de 0 ;
- la borne supérieure de 100 ;
- la forte accumulation de réponses à 100.

Cependant, les analyses de sensibilité montrent que plusieurs conclusions importantes restent stables.

## 7.2 Sur l’influence des participants

Le jackknife montre que :

- l’effet d’entropie est extrêmement stable ;
- l’effet de séquence reste négatif ;
- la différence Standard–Neutral reste positive ;
- aucun participant unique ne paraît expliquer ces résultats ;
- l’effet de précision moyenne du participant est beaucoup moins stable.

## 7.3 Sur la calibration métacognitive

Les participants sont globalement surconfiants :

\[
75{,}7\%-62{,}3\%
=
13{,}4\text{ points}
\]

Mais la calibration moyenne n’est pas toute l’histoire.

La discrimination métacognitive est également faible :

- AUC moyenne proche de 0,52 ;
- faible différence de confiance entre réponses correctes et incorrectes ;
- odds ratio de confiance proche de 1 dans le modèle logistique d’exactitude.

Autrement dit :

> Les participants utilisent des niveaux de confiance élevés, mais ces niveaux ne permettent pas clairement de savoir, essai par essai, quand leur réponse est correcte.

## 7.4 Sur MReasoner

Trois simulations étaient suffisantes pour retrouver grossièrement la structure générale, mais pas pour estimer précisément toutes les combinaisons participant–tâche.

Les résultats à 10 et 20 simulations sont beaucoup plus proches entre eux.

La tâche DA est la plus instable computationnellement, tandis que MT est parfaitement stable.

## 7.5 Sur les résultats statistiques

Le passage de 3 à 20 simulations ne modifie pas les conclusions principales :

- l’entropie reste fortement négative ;
- la séquence reste négative ;
- la condition Standard reste associée à davantage de confiance ;
- l’effet moyen de MReasoner reste incertain ;
- l’effet intra-individuel de MReasoner est plus sensible à la spécification du modèle.

## 7.6 Hiérarchie de robustesse des conclusions

On peut résumer la solidité actuelle des résultats ainsi :

### Conclusion très robuste

> Les items dont les réponses sont plus dispersées entre participants suscitent une confiance plus faible.

### Conclusion relativement robuste

> La confiance est en moyenne plus élevée dans la condition Standard que dans la condition Neutral.

### Conclusion relativement robuste mais à interpréter dans le contexte de la borne supérieure

> La confiance diminue au cours de la séquence, surtout parce que l’utilisation de la valeur 100 devient moins fréquente.

### Conclusion plus fragile

> Un nombre de modèles mentaux supérieur à la moyenne personnelle pourrait être associé à une confiance légèrement plus faible.

Cette dernière conclusion dépend davantage :

- du nombre de simulations ;
- de la spécification du modèle ;
- de la présence ou de l’absence de la validité ;
- de la décomposition interindividuelle/intra-individuelle.

---

# 8. Ce qu’il reste à faire

L’étape 9 termine la partie de vérification et de production des sorties.

Il reste principalement l’étape 10 :

> **Interpréter l’ensemble des résultats finaux comme une histoire scientifique cohérente.**

Cette dernière étape devra notamment distinguer :

1. **la signification statistique** ;
2. **la taille des effets** ;
3. **la robustesse des effets** ;
4. **la signification cognitive** ;
5. **les résultats descriptifs et les résultats ajustés** ;
6. **les associations et les relations causales** ;
7. **les résultats principaux et les analyses de sensibilité** ;
8. **les limites du modèle linéaire** ;
9. **les limites de MReasoner** ;
10. **les conclusions que nous pouvons réellement défendre**.


# Étape 10 — Interprétation complète des résultats finaux de l’expérience E1

## Sommaire

1. [Objectif de cette dernière étape](#1-objectif-de-cette-dernière-étape)
2. [Rappel de la question scientifique](#2-rappel-de-la-question-scientifique)
3. [Rappel du modèle final](#3-rappel-du-modèle-final)
4. [Comment lire un résultat statistique](#4-comment-lire-un-résultat-statistique)
5. [Interprétation de l’interception](#5-interprétation-de-linterception)
6. [Effet de la condition Standard](#6-effet-de-la-condition-standard)
7. [Effet de la position dans la séquence](#7-effet-de-la-position-dans-la-séquence)
8. [Effet de la précision moyenne du participant](#8-effet-de-la-précision-moyenne-du-participant)
9. [Effet de l’entropie de l’item](#9-effet-de-lentropie-de-litem)
10. [Effet interindividuel du nombre moyen de modèles mentaux](#10-effet-interindividuel-du-nombre-moyen-de-modèles-mentaux)
11. [Effet intra-individuel du nombre de modèles mentaux](#11-effet-intra-individuel-du-nombre-de-modèles-mentaux)
12. [Composantes de variance et différences individuelles](#12-composantes-de-variance-et-différences-individuelles)
13. [Qualité explicative du modèle](#13-qualité-explicative-du-modèle)
14. [Interprétation des analyses de sensibilité](#14-interprétation-des-analyses-de-sensibilité)
15. [Interprétation du problème de plafond](#15-interprétation-du-problème-de-plafond)
16. [Interprétation de la calibration métacognitive](#16-interprétation-de-la-calibration-métacognitive)
17. [Interprétation de la robustesse de MReasoner](#17-interprétation-de-la-robustesse-de-mreasoner)
18. [Ce que disent les diagnostics](#18-ce-que-disent-les-diagnostics)
19. [Ce que nous pouvons conclure scientifiquement](#19-ce-que-nous-pouvons-conclure-scientifiquement)
20. [Ce que nous ne pouvons pas conclure](#20-ce-que-nous-ne-pouvons-pas-conclure)
21. [Forces et limites du projet](#21-forces-et-limites-du-projet)
22. [Proposition de rédaction finale des résultats](#22-proposition-de-rédaction-finale-des-résultats)
23. [Proposition de discussion scientifique](#23-proposition-de-discussion-scientifique)
24. [Résumé pédagogique de l’ensemble du projet](#24-résumé-pédagogique-de-lensemble-du-projet)
25. [Conclusion générale](#25-conclusion-générale)

---

# 1. Objectif de cette dernière étape

Pendant les étapes précédentes, nous avons construit progressivement une analyse statistique complète :

1. préparation des données ;
2. calcul de nouvelles variables ;
3. construction d’un modèle nul ;
4. ajout des variables de contrôle ;
5. ajout des prédicteurs cognitifs ;
6. analyses de sensibilité ;
7. étude de la borne supérieure de confiance ;
8. modèle logistique de l’utilisation de la valeur 100 ;
9. diagnostics, calibration métacognitive et robustesse computationnelle.

L’objectif de l’étape 10 n’est plus de calculer de nouveaux résultats. Il est de donner un sens scientifique à l’ensemble de ce que nous avons obtenu.

Cela nécessite de distinguer plusieurs niveaux.

## Niveau 1 — Le résultat numérique

Exemple :

\[
\beta_{\text{entropie}}=-2{,}487
\]

## Niveau 2 — Le résultat statistique

L’intervalle de confiance n’inclut pas zéro et la valeur \(p\) est très petite.

## Niveau 3 — L’interprétation substantielle

Les items qui suscitent davantage de désaccord entre participants sont associés à une confiance plus faible.

## Niveau 4 — La robustesse

Cet effet résiste :

- au contrôle de la condition ;
- au contrôle de la séquence ;
- au contrôle des différences entre participants ;
- au contrôle des différences entre items ;
- au remplacement de la validité par le type de tâche ;
- à l’exclusion des réponses égales à 100 ;
- au retrait successif de chaque participant ;
- au passage de 3 à 20 simulations de MReasoner.

## Niveau 5 — La prudence causale

Nous ne pouvons pas automatiquement conclure que l’entropie cause la diminution de confiance, car l’entropie est une mesure empirique calculée à partir des réponses et non une variable manipulée expérimentalement.

> **But de cette étape**
>
> Transformer les nombres produits par les modèles en conclusions compréhensibles, exactes, prudentes et scientifiquement défendables.

---

# 2. Rappel de la question scientifique

La variable principale étudiée était la **confiance** exprimée par les participants après chaque raisonnement.

Cette confiance variait de :

\[
0
\quad\text{à}\quad
100
\]

Nous voulions comprendre pourquoi certains essais recevaient une confiance plus forte que d’autres.

Plus précisément, nous cherchions à déterminer si la confiance variait en fonction :

- de la condition expérimentale ;
- de la progression dans l’expérience ;
- de la performance générale du participant ;
- du désaccord suscité par l’item ;
- du nombre de modèles mentaux estimé par MReasoner ;
- des différences générales entre participants ;
- des différences générales entre items.

Les deux conditions étaient :

- **Standard** : contenu ordinaire, potentiellement croyable ou incroyable ;
- **Neutral** : contenu rendu neutre par le remplacement d’un terme par un non-mot.

Les quatre formes de raisonnement étaient :

- MP ;
- MT ;
- AC ;
- DA.

Les estimations de MReasoner étaient calculées pour chaque combinaison :

```text
participant × type de tâche
```

et non pour chaque item particulier.

---

# 3. Rappel du modèle final

Le modèle linéaire mixte final peut être écrit ainsi :

\[
\begin{aligned}
\text{Confiance}_{ij}
={}&
\beta_0\\
&+\beta_1\text{Standard}_i\\
&+\beta_2\text{Séquence}_{ij}\\
&+\beta_3\text{PrécisionParticipant}_i\\
&+\beta_4\text{EntropieItem}_j\\
&+\beta_5\text{ModèlesMoyensParticipant}_i\\
&+\beta_6\text{ModèlesIntra}_{ij}\\
&+u_i+v_j+\varepsilon_{ij}
\end{aligned}
\]

où :

- \(i\) désigne un participant ;
- \(j\) désigne un item ;
- \(\beta_0\) est l’interception ;
- \(\beta_1,\ldots,\beta_6\) sont les effets fixes ;
- \(u_i\) est l’effet aléatoire du participant ;
- \(v_j\) est l’effet aléatoire de l’item ;
- \(\varepsilon_{ij}\) est l’erreur résiduelle.

Les hypothèses aléatoires sont approximativement :

\[
u_i\sim\mathcal{N}(0,\sigma^2_{\text{participant}})
\]

\[
v_j\sim\mathcal{N}(0,\sigma^2_{\text{item}})
\]

\[
\varepsilon_{ij}\sim\mathcal{N}(0,\sigma^2_{\text{résiduelle}})
\]

## 3.1 Le modèle final est-il celui avec ou sans validité ?

Le modèle final présenté dans le rapport est le modèle sans `validity_binary`.

La validité a été étudiée dans les analyses intermédiaires et de sensibilité, mais n’a pas été conservée dans la spécification finale principale.

Cette décision est justifiée par plusieurs éléments :

1. la validité est parfaitement déterminée par le type de tâche :
   - MP et MT sont valides ;
   - AC et DA sont invalides ;
2. l’ajout de la validité n’améliorait pas clairement le modèle ;
3. son coefficient n’était pas clairement différent de zéro ;
4. l’ajout du type de tâche complet n’améliorait pas non plus clairement le modèle ;
5. l’inclusion simultanée de validité et de type de tâche aurait créé une redondance structurelle.

Le modèle final se concentre donc sur les prédicteurs principaux sans conserver un terme de validité qui n’apportait pas d’information supplémentaire claire.

---

# 4. Comment lire un résultat statistique

Avant d’interpréter chaque prédicteur, il faut rappeler la signification des principales colonnes.

Prenons l’exemple final de l’entropie :

| Quantité | Valeur |
|---|---:|
| Coefficient \(\beta\) | -2,487 |
| Erreur standard | 0,273 |
| IC à 95 % | \([-3,022;-1,953]\) |
| Valeur \(p\) | \(7,58\times10^{-20}\) |

---

## 4.1 Le coefficient

Le **coefficient**, ou estimation de \(\beta\), indique la direction et la taille moyenne de l’association.

Pour l’entropie :

\[
\beta=-2{,}487
\]

Le signe négatif signifie :

> Quand l’entropie augmente, la confiance prédite diminue, toutes les autres variables du modèle étant maintenues constantes.

Comme l’entropie a été standardisée, une augmentation d’une unité correspond à une augmentation d’un écart-type de l’entropie.

Le coefficient signifie donc :

> Une augmentation d’un écart-type de l’entropie de l’item est associée à une diminution moyenne d’environ 2,49 points de confiance.

---

## 4.2 L’erreur standard

L’**erreur standard**, notée \(SE\), mesure l’incertitude de l’estimation du coefficient.

Elle ne mesure pas la dispersion brute des observations. Elle mesure la précision avec laquelle le coefficient a été estimé.

Une analogie consiste à imaginer une balance :

- le coefficient est le poids estimé ;
- l’erreur standard indique à quel point la balance est précise.

Une petite erreur standard relativement au coefficient indique une estimation précise.

Pour l’entropie :

\[
\frac{|-2{,}487|}{0{,}273}\approx9{,}11
\]

Le coefficient est environ neuf fois plus grand que son erreur standard. L’estimation est donc très éloignée de zéro relativement à son incertitude.

---

## 4.3 L’intervalle de confiance à 95 %

L’**intervalle de confiance** donne une plage de valeurs plausibles pour le coefficient, dans le cadre de la méthode statistique utilisée.

Il est approximativement calculé par :

\[
\widehat{\beta}
\pm
1{,}96\times SE
\]

Pour l’entropie :

\[
-2{,}487
\pm
1{,}96\times0{,}273
\]

\[
-2{,}487
\pm
0{,}535
\]

ce qui donne :

\[
[-3{,}022;-1{,}953]
\]

Toutes les valeurs de cet intervalle sont négatives.

Les données sont donc compatibles avec une diminution comprise approximativement entre :

- 1,95 point ;
- 3,02 points ;

pour une augmentation d’un écart-type de l’entropie.

### Ce que l’intervalle ne signifie pas exactement

Dans une interprétation fréquentiste stricte, on ne dit pas :

> Il existe une probabilité de 95 % que ce coefficient précis soit dans cet intervalle.

Le paramètre est considéré comme fixe. C’est la procédure de construction de l’intervalle qui, si elle était répétée un grand nombre de fois, couvrirait le vrai paramètre dans environ 95 % des cas.

Pour une première interprétation pratique, nous pouvons néanmoins retenir :

> L’intervalle indique les tailles d’effet raisonnablement compatibles avec les données et le modèle.

---

## 4.4 La valeur \(p\)

La **valeur \(p\)** répond à la question suivante :

> Si le vrai coefficient était nul, quelle serait la probabilité d’obtenir une estimation au moins aussi éloignée de zéro que celle observée, compte tenu du modèle ?

Une petite valeur \(p\) indique que le résultat observé serait difficile à expliquer par les fluctuations d’échantillonnage si le coefficient réel était exactement nul.

Pour l’entropie :

\[
p=7{,}58\times10^{-20}
\]

Cette valeur est extrêmement petite.

Cela constitue une preuve statistique très forte contre l’hypothèse :

\[
\beta_{\text{entropie}}=0
\]

### Ce que la valeur \(p\) ne mesure pas

La valeur \(p\) ne donne pas :

- la probabilité que l’hypothèse nulle soit vraie ;
- la probabilité que le résultat se reproduise ;
- la taille de l’effet ;
- l’importance scientifique ;
- la probabilité d’une relation causale.

Un coefficient minuscule peut avoir une très petite valeur \(p\) si l’échantillon est grand.

C’est pourquoi nous interprétons ensemble :

- le coefficient ;
- son erreur standard ;
- son intervalle de confiance ;
- sa valeur \(p\) ;
- sa robustesse ;
- sa signification scientifique.

---

## 4.5 « Statistiquement détecté » et « important » ne sont pas synonymes

Un prédicteur est dit **statistiquement détecté** lorsque les données permettent de distinguer son effet de zéro avec le critère choisi.

Dans notre analyse, le seuil conventionnel est :

\[
p<0{,}05
\]

Mais un effet peut être :

- statistiquement détecté et très petit ;
- non détecté mais potentiellement substantiel et mal estimé ;
- détecté dans un modèle et non détecté dans un autre ;
- robuste ou fragile selon les analyses de sensibilité.

Nous éviterons donc de résumer les résultats uniquement avec les mots « significatif » ou « non significatif ».

---

# 5. Interprétation de l’interception

L’interception finale était approximativement :

\[
\beta_0\approx72{,}7
\]

ou, selon la sortie exacte utilisée dans la version du rapport :

\[
\beta_0\approx73
\]

## 5.1 Que représente cette valeur ?

Dans le modèle final, l’interception représente la confiance prédite lorsque :

- la condition de référence est `Neutral` ;
- la séquence centrée vaut zéro, donc l’essai se situe à la position moyenne ;
- `subject_accuracy_z = 0` ;
- `item_entropy_z = 0` ;
- `subject_mean_models_z = 0` ;
- `models_within_subject_z = 0` ;
- les effets aléatoires du participant et de l’item sont fixés à zéro.

Autrement dit, elle correspond approximativement à :

> La confiance prédite pour un essai Neutral situé au milieu de l’expérience, présenté à un participant moyen, portant sur un item d’entropie moyenne, avec des valeurs moyennes pour les prédicteurs MReasoner.

## 5.2 Pourquoi l’interception n’est-elle pas exactement la confiance moyenne globale ?

La confiance moyenne brute est :

\[
75{,}74
\]

L’interception finale est plus proche de 73.

Cela s’explique parce que la moyenne brute mélange :

- les essais Neutral ;
- les essais Standard ;
- toutes les positions de séquence ;
- tous les niveaux d’entropie ;
- tous les profils de participants ;
- tous les items.

L’interception correspond à un scénario de référence précis, pas à la moyenne de toutes les observations.

En particulier, la condition Standard a un coefficient positif. La moyenne générale, qui inclut les essais Standard, est donc supérieure à la référence Neutral.

## 5.3 L’interception est-elle scientifiquement centrale ?

Pas vraiment.

Elle est indispensable pour calculer les prédictions, mais la question scientifique porte surtout sur les différences associées aux prédicteurs.

L’interception sert de point de départ à partir duquel les autres effets s’ajoutent ou se soustraient.

---

# 6. Effet de la condition Standard

Résultat final :

\[
\beta_{\text{Standard}}=5{,}247
\]

\[
SE=2{,}522
\]

\[
IC_{95\%}=[0{,}303;10{,}191]
\]

\[
p=0{,}038
\]

---

## 6.1 Interprétation directe

La condition Neutral est la catégorie de référence.

Le coefficient Standard représente donc :

\[
\text{Confiance Standard}
-
\text{Confiance Neutral}
\]

Toutes les autres variables étant maintenues constantes, les essais de la condition Standard sont associés à une confiance moyenne supérieure d’environ :

\[
5{,}25\text{ points}
\]

par rapport aux essais Neutral.

---

## 6.2 Exemple de prédiction

Supposons qu’un essai Neutral ait une confiance prédite de :

\[
72{,}7
\]

Le même profil de prédicteurs en condition Standard aurait une confiance prédite de :

\[
72{,}7+5{,}25=77{,}95
\]

Cette comparaison est théorique, car les participants appartiennent à une seule condition. Elle représente néanmoins la différence moyenne ajustée entre les deux groupes.

---

## 6.3 Interprétation de l’intervalle de confiance

L’intervalle est :

\[
[0{,}303;10{,}191]
\]

Cela signifie que les données sont compatibles avec :

- un effet positif très faible, autour de 0,3 point ;
- ou un effet plus important, autour de 10,2 points.

L’intervalle est entièrement supérieur à zéro, mais sa largeur indique une incertitude non négligeable sur la taille exacte de l’effet.

Nous pouvons donc dire :

> La direction positive de l’effet est soutenue par les données, mais sa taille est estimée avec une précision modérée.

---

## 6.4 Pourquoi l’erreur standard est-elle relativement grande ?

La condition varie entre participants.

Un participant appartient soit au groupe Standard, soit au groupe Neutral.

Il n’est donc pas possible de comparer la même personne dans les deux conditions.

La différence de condition doit être estimée à partir de la comparaison entre :

- 71 participants Standard ;
- 70 participants Neutral.

Or nous avons observé une variabilité interindividuelle importante de confiance.

La variance entre participants est d’environ :

\[
195{,}8
\]

avec un écart-type d’environ :

\[
13{,}99
\]

Cette forte variabilité rend l’estimation de la différence entre groupes moins précise.

### Analogie

Comparer Standard à Neutral revient à comparer deux classes différentes.

Si les élèves d’une même classe ont des niveaux très variés, il devient plus difficile de déterminer précisément si l’une des classes possède une moyenne supérieure à l’autre.

---

## 6.5 Robustesse de l’effet de condition

L’effet Standard reste positif dans plusieurs analyses :

- modèle de contrôle ;
- modèle cognitif ;
- modèle avec validité ;
- modèle avec type de tâche ;
- modèle fondé sur 20 simulations ;
- analyse jackknife ;
- modèle logistique de la valeur 100.

Cependant, dans l’analyse excluant toutes les réponses égales à 100, le coefficient devient :

\[
\beta\approx1{,}98
\]

avec :

\[
p\approx0{,}390
\]

L’effet n’est alors plus clairement détecté.

---

## 6.6 Que signifie la différence entre l’analyse complète et l’analyse sans les 100 ?

Dans les données brutes :

| Condition | Taux de confiance égale à 100 |
|---|---:|
| Neutral | 19,15 % |
| Standard | 32,53 % |

La condition Standard produit beaucoup plus de réponses exactement égales à 100.

Le modèle logistique de plafond indique un odds ratio d’environ :

\[
OR=3{,}96
\]

Cela signifie que, toutes les autres variables du modèle étant maintenues constantes, les odds d’utiliser la réponse 100 sont presque quatre fois plus élevées en condition Standard qu’en condition Neutral.

Lorsque nous retirons les réponses égales à 100, nous retirons précisément une composante importante de la différence entre les conditions.

Le résultat suggère donc que :

> La différence globale de confiance entre Standard et Neutral provient en grande partie d’une utilisation plus fréquente de la borne supérieure 100 en condition Standard.

Cela ne rend pas l’effet invalide. L’utilisation de 100 fait partie du comportement observé.

Mais cela précise la nature de l’effet :

- il ne s’agit pas nécessairement d’un décalage uniforme de toute l’échelle ;
- il s’agit en partie d’une probabilité plus forte d’exprimer une certitude maximale.

---

## 6.7 Interprétation psychologique possible

La condition Standard contient des concepts ordinaires et sémantiquement interprétables.

La condition Neutral contient un non-mot qui réduit ou empêche l’utilisation de certaines connaissances sémantiques.

Une interprétation possible est :

> Les contenus familiers de la condition Standard facilitent un sentiment subjectif de compréhension ou de certitude, ce qui augmente la confiance et favorise l’utilisation de la valeur maximale.

Cependant, cette interprétation reste une hypothèse psychologique. Le modèle statistique montre une différence de confiance, mais il ne démontre pas directement le mécanisme mental responsable de cette différence.

---

## 6.8 Conclusion sur la condition

La formulation la plus défendable est :

> La condition Standard était associée à une confiance plus élevée que la condition Neutral. Cette différence était relativement stable dans les modèles appliqués à l’ensemble des observations, mais elle semblait principalement portée par une utilisation plus fréquente de la valeur maximale 100 dans la condition Standard.

---

# 7. Effet de la position dans la séquence

Résultat final :

\[
\beta_{\text{séquence}}=-0{,}438
\]

\[
SE=0{,}097
\]

\[
IC_{95\%}=[-0{,}627;-0{,}249]
\]

\[
p=5{,}75\times10^{-6}
\]

---

## 7.1 Rappel de l’échelle de `sequence_c10`

La variable de séquence a été :

1. centrée autour de sa moyenne ;
2. divisée par 10.

La formule générale est :

\[
\text{sequence\_c10}
=
\frac{\text{sequence}-\overline{\text{sequence}}}{10}
\]

Le coefficient correspond donc à une progression de dix essais.

---

## 7.2 Interprétation directe

Pour dix essais supplémentaires, la confiance moyenne diminue d’environ :

\[
0{,}438\text{ point}
\]

toutes les autres variables étant maintenues constantes.

Pour 20 essais supplémentaires :

\[
2\times(-0{,}438)
=
-0{,}876
\]

Pour 50 essais supplémentaires :

\[
5\times(-0{,}438)
=
-2{,}19
\]

---

## 7.3 Différence entre le début et la fin

La séquence va de 1 à 64.

La différence est :

\[
64-1=63\text{ essais}
\]

soit :

\[
\frac{63}{10}=6{,}3
\]

unités de `sequence_c10`.

La variation prédite est donc approximativement :

\[
6{,}3\times(-0{,}438)
\approx-2{,}76
\]

Le modèle linéaire complet prédit donc environ 2,8 points de confiance en moins à la fin qu’au début, toutes choses égales par ailleurs.

---

## 7.4 Est-ce un effet important ?

À l’échelle d’un seul bloc de dix essais, l’effet est petit :

\[
-0{,}438
\]

Mais il s’accumule au cours des 64 essais.

La taille reste néanmoins modeste relativement à :

- une moyenne de confiance proche de 76 ;
- un écart-type brut de confiance proche de 22 ;
- un écart-type résiduel proche de 17.

L’effet est donc :

- statistiquement clairement détecté ;
- directionnellement stable ;
- mais quantitativement modéré.

---

## 7.5 Pourquoi la confiance pourrait-elle diminuer ?

Plusieurs mécanismes sont possibles :

- fatigue ;
- diminution de l’enthousiasme initial ;
- usage moins fréquent des réponses extrêmes ;
- apprentissage de la difficulté de la tâche ;
- réévaluation progressive de sa propre performance ;
- modification de la stratégie de réponse ;
- familiarisation avec l’échelle.

Le modèle statistique ne permet pas de choisir automatiquement entre ces explications.

---

## 7.6 Ce que montre l’analyse sans les 100

Lorsque les essais avec confiance égale à 100 sont exclus, le coefficient devient approximativement :

\[
\beta=0{,}062
\]

avec :

\[
p\approx0{,}580
\]

L’effet disparaît presque complètement.

En parallèle, le modèle logistique du plafond donne :

\[
OR_{\text{séquence}}\approx0{,}850
\]

pour dix essais supplémentaires.

Cela signifie que les odds d’utiliser la valeur 100 diminuent d’environ :

\[
1-0{,}850=0{,}150
\]

soit 15 % par tranche de dix essais.

> **Attention**
>
> Une diminution de 15 % des odds n’est pas exactement une diminution de 15 points de probabilité. Les odds et les probabilités ne sont pas identiques.

---

## 7.7 Interprétation complète

L’effet négatif de la séquence dans le modèle linéaire ne semble pas provenir d’une baisse uniforme de toutes les valeurs inférieures à 100.

Il semble plutôt correspondre à ceci :

> Au début de l’expérience, les participants utilisent plus souvent la réponse maximale 100. Au fil des essais, cette réponse extrême devient moins fréquente, ce qui fait baisser la moyenne globale de confiance.

Cette interprétation est plus précise que l’affirmation générale :

> Les participants deviennent progressivement moins confiants.

Ils deviennent peut-être moins enclins à exprimer une certitude maximale, sans que toute la distribution des valeurs inférieures à 100 se déplace fortement.

---

## 7.8 Conclusion sur la séquence

La formulation recommandée est :

> La confiance moyenne diminuait au cours de l’expérience. Les analyses de sensibilité indiquaient cependant que cette diminution était principalement liée à une réduction progressive de l’utilisation de la valeur maximale 100, plutôt qu’à une baisse générale parmi les réponses inférieures à 100.

---

# 8. Effet de la précision moyenne du participant

Résultat final :

\[
\beta_{\text{précision}}=0{,}696
\]

\[
SE=1{,}584
\]

\[
IC_{95\%}=[-2{,}408;3{,}800]
\]

\[
p=0{,}660
\]

---

## 8.1 Que représente `subject_accuracy_z` ?

Pour chaque participant, nous avons calculé :

\[
\text{subject\_accuracy}
=
\frac{\text{nombre de réponses correctes}}
{\text{nombre total de réponses analysées}}
\]

Chaque participant possédait 64 essais.

Exemple :

- 40 réponses correctes ;
- 64 essais.

\[
\text{précision}
=
\frac{40}{64}
=
0{,}625
\]

Cette variable a ensuite été standardisée :

\[
\text{subject\_accuracy\_z}
=
\frac{
\text{subject\_accuracy}
-
\overline{\text{subject\_accuracy}}
}{
SD(\text{subject\_accuracy})
}
\]

Le coefficient compare donc des participants qui diffèrent d’un écart-type de précision moyenne.

---

## 8.2 Interprétation du coefficient

Le coefficient est positif :

\[
0{,}696
\]

Il suggère qu’un participant plus précis d’un écart-type pourrait avoir une confiance supérieure d’environ 0,70 point.

Mais l’incertitude est importante :

\[
[-2{,}408;3{,}800]
\]

L’intervalle inclut :

- une association négative ;
- aucune association ;
- une association positive.

Nous ne pouvons donc pas déterminer clairement la direction réelle de l’effet.

---

## 8.3 Pourquoi dit-on que l’effet n’est pas détecté ?

Ce n’est pas simplement parce que :

\[
SE>\beta
\]

Même si cette comparaison donne une intuition, le critère est plutôt fondé sur le rapport :

\[
z=
\frac{\widehat{\beta}}{SE}
\]

Ici :

\[
z\approx\frac{0{,}696}{1{,}584}
\approx0{,}44
\]

Le coefficient est petit relativement à son incertitude.

La valeur \(p\) est élevée :

\[
p=0{,}660
\]

et l’intervalle de confiance contient largement zéro.

Ces trois expressions décrivent la même situation :

- faible rapport coefficient/incertitude ;
- intervalle comprenant zéro ;
- grande valeur \(p\).

---

## 8.4 « Non détecté » ne signifie pas « exactement nul »

Nous ne devons pas conclure :

> La précision du participant n’a absolument aucun effet sur sa confiance.

Nous pouvons conclure :

> Les données et le modèle ne permettent pas d’estimer avec suffisamment de précision une association générale entre la précision moyenne des participants et leur confiance moyenne conditionnelle.

L’intervalle montre qu’une association de quelques points dans un sens ou dans l’autre reste compatible avec les données.

---

## 8.5 Pourquoi cette association est-elle difficile à estimer ?

`subject_accuracy_z` varie uniquement entre participants.

Chaque participant possède une seule valeur de précision moyenne.

Même si le dataset comporte 9024 lignes, nous ne disposons pas de 9024 valeurs indépendantes de précision moyenne. Nous disposons essentiellement de :

\[
141
\]

profils de participants.

De plus :

- la condition varie elle aussi entre participants ;
- le nombre moyen de modèles varie entre participants ;
- l’effet aléatoire du participant capture les autres différences stables entre individus ;
- les participants diffèrent fortement dans leur utilisation générale de l’échelle.

La quantité d’information réellement disponible pour les prédicteurs interindividuels est donc bien plus proche de 141 participants que de 9024 essais indépendants.

---

## 8.6 Résultat du jackknife

Le coefficient de précision changeait de signe selon le participant retiré.

Il pouvait devenir :

- légèrement négatif ;
- ou positif.

Cela confirme que ce prédicteur est mal déterminé.

---

## 8.7 Relation avec la calibration métacognitive

On pourrait intuitivement s’attendre à ce que les participants les plus précis soient aussi les plus confiants.

Mais deux dimensions doivent être séparées :

1. le niveau moyen de confiance ;
2. la capacité à avoir davantage confiance quand on a raison.

Un participant peut :

- être très précis mais prudent ;
- être peu précis mais très confiant ;
- être globalement surconfiant ;
- discriminer correctement ses bonnes et mauvaises réponses sans être bien calibré en moyenne.

Notre résultat montre qu’il n’existe pas d’association interindividuelle claire entre précision moyenne et niveau de confiance ajusté.

---

## 8.8 Conclusion sur la précision

Formulation recommandée :

> La précision moyenne des participants n’était pas clairement associée à leur niveau de confiance après prise en compte des autres prédicteurs. Cette estimation était imprécise et sensible au retrait de certains participants.

---

# 9. Effet de l’entropie de l’item

Résultat final :

\[
\beta_{\text{entropie}}=-2{,}487
\]

\[
SE=0{,}273
\]

\[
IC_{95\%}=[-3{,}022;-1{,}953]
\]

\[
p=7{,}58\times10^{-20}
\]

Il s’agit du résultat le plus robuste du projet.

---

## 9.1 Rappel : qu’est-ce que l’entropie ?

Pour chaque item, les participants pouvaient répondre :

- Yes ;
- No.

Nous calculons :

\[
p_{\text{Yes}}
=
\frac{\text{nombre de Yes}}
{\text{nombre total de réponses}}
\]

et :

\[
p_{\text{No}}=1-p_{\text{Yes}}
\]

L’entropie binaire est :

\[
H
=
-p_{\text{Yes}}\log_2(p_{\text{Yes}})
-p_{\text{No}}\log_2(p_{\text{No}})
\]

avec la convention :

\[
0\log_2(0)=0
\]

---

## 9.2 Exemples

### Accord presque complet

Supposons :

- 70 Yes ;
- 1 No.

Alors :

\[
p_{\text{Yes}}\approx0{,}986
\]

\[
p_{\text{No}}\approx0{,}014
\]

L’entropie est faible.

Les participants donnent presque tous la même réponse.

### Désaccord maximal

Supposons :

- 35 Yes ;
- 36 No.

Les proportions sont proches de 0,50.

L’entropie est proche de :

\[
1
\]

Les participants sont presque également divisés.

---

## 9.3 Ce que mesure réellement l’entropie

L’entropie mesure le désaccord collectif sur la réponse.

Elle ne mesure pas directement :

- le temps de réponse ;
- le nombre de prémisses ;
- la validité formelle ;
- le nombre de modèles mentaux ;
- la confiance ;
- l’exactitude individuelle.

Elle répond à la question :

> Pour cet item, les réponses Yes et No sont-elles très concentrées sur une option ou réparties entre les deux options ?

---

## 9.4 Interprétation du coefficient

Comme `item_entropy_z` est standardisée :

> Une augmentation d’un écart-type de l’entropie est associée à une diminution moyenne d’environ 2,49 points de confiance.

Pour une différence de deux écarts-types :

\[
2\times(-2{,}487)
=
-4{,}974
\]

Deux items séparés de deux écarts-types d’entropie différeraient donc d’environ cinq points de confiance prédite, toutes les autres variables étant maintenues constantes.

---

## 9.5 Pourquoi l’effet est-il considéré comme très robuste ?

### Dans le modèle cognitif initial

\[
\beta\approx-2{,}437
\]

avec une valeur \(p\) extrêmement petite.

### Avec le type de tâche

\[
\beta\approx-2{,}246
\]

L’effet reste fortement négatif.

### Sans les réponses égales à 100

\[
\beta\approx-2{,}307
\]

L’effet reste fortement négatif.

### Dans le modèle logistique du plafond

\[
OR\approx0{,}740
\]

Une augmentation d’un écart-type de l’entropie est associée à une diminution d’environ 26 % des odds d’utiliser la confiance 100 :

\[
1-0{,}740=0{,}260
\]

### Dans le jackknife

Le coefficient restait approximativement entre :

\[
-2{,}529
\quad\text{et}\quad
-2{,}440
\]

selon le participant retiré.

### Avec 20 simulations MReasoner

L’effet reste pratiquement inchangé.

---

## 9.6 Pourquoi cet effet réduit-il fortement la variance entre items ?

Dans le modèle nul, la variance des items était approximativement :

\[
11{,}87
\]

Dans le modèle cognitif, elle tombait autour de :

\[
5{,}2
\]

La réduction est approximativement :

\[
11{,}87-5{,}2=6{,}67
\]

soit :

\[
\frac{6{,}67}{11{,}87}
\approx56\%
\]

Une grande partie de la différence moyenne de confiance entre les items est donc associée aux prédicteurs ajoutés, et surtout à l’entropie.

Cela est cohérent avec la nature de la variable :

- l’entropie est une propriété de l’item ;
- elle varie entre items ;
- elle explique donc principalement une partie de la variance entre items.

---

## 9.7 Interprétation cognitive

Une interprétation possible est :

> Les items qui ne conduisent pas les participants vers une réponse dominante provoquent aussi une confiance plus faible.

Autrement dit, lorsqu’un item permet plusieurs interprétations, stratégies ou intuitions concurrentes, cela peut produire :

- davantage de désaccord entre les participants ;
- davantage d’incertitude subjective au niveau individuel.

L’entropie collective peut donc agir comme un indicateur empirique de l’ambiguïté psychologique de l’item.

---

## 9.8 Entropie et difficulté ne sont pas exactement la même chose

La difficulté est souvent mesurée par le taux d’erreur :

\[
1-\text{item\_accuracy}
\]

L’entropie repose sur la répartition Yes/No.

Ces mesures peuvent être liées, mais elles ne sont pas identiques.

### Exemple 1 — Item invalide difficile

Si la bonne réponse est No, mais que :

- 50 % répondent Yes ;
- 50 % répondent No ;

alors :

- exactitude proche de 50 % ;
- entropie maximale.

### Exemple 2 — Item auquel presque tout le monde répond incorrectement

Si la bonne réponse est No, mais que :

- 95 % répondent Yes ;
- 5 % répondent No ;

alors :

- exactitude très faible ;
- entropie faible.

L’item est difficile au sens de l’exactitude, mais il ne crée pas beaucoup de désaccord : presque tout le monde commet la même erreur.

L’entropie mesure donc davantage l’incertitude collective que la simple difficulté objective.

---

## 9.9 Une limite importante : l’entropie utilise les réponses analysées

L’entropie de chaque item est calculée à partir des réponses des mêmes participants dont nous analysons la confiance.

Cela crée une dépendance empirique :

- la variable prédictive est résumée à partir de l’échantillon ;
- puis réinjectée dans un modèle portant sur ce même échantillon.

Ce procédé n’est pas nécessairement incorrect pour une analyse descriptive, mais il peut renforcer l’ajustement dans l’échantillon.

Une validation plus exigeante consisterait à calculer l’entropie sur un ensemble de participants et à prédire la confiance d’autres participants.

### Validation croisée possible

```text
Groupe A
  └── calcul de l’entropie des items

Groupe B
  └── prédiction de la confiance à partir de l’entropie du groupe A
```

On pourrait également utiliser une méthode « leave-one-subject-out » pour calculer, pour chaque participant, l’entropie sans utiliser ses propres réponses.

---

## 9.10 L’effet est-il causal ?

Non démontré.

Nous pouvons dire :

> Une entropie plus élevée est associée à une confiance plus faible.

Nous ne pouvons pas affirmer directement :

> L’entropie provoque une baisse de confiance.

Pourquoi ?

Parce que l’entropie n’a pas été manipulée expérimentalement. Elle peut résumer d’autres propriétés :

- ambiguïté du contenu ;
- difficulté ;
- conflit ;
- formulation ;
- familiarité ;
- stratégie de raisonnement ;
- hétérogénéité des interprétations.

---

## 9.11 Conclusion sur l’entropie

La formulation principale recommandée est :

> L’entropie empirique de l’item constituait le prédicteur le plus robuste de la confiance. Les participants rapportaient une confiance plus faible pour les items suscitant une plus grande dispersion des réponses Yes/No. Cette association résistait aux différentes spécifications statistiques, à l’exclusion des réponses égales à 100, au retrait successif des participants et à l’utilisation de 20 simulations de MReasoner.

---

# 10. Effet interindividuel du nombre moyen de modèles mentaux

Résultat final :

\[
\beta_{\text{modèles moyens}}=-2{,}241
\]

\[
SE=1{,}583
\]

\[
IC_{95\%}=[-5{,}343;0{,}862]
\]

\[
p=0{,}157
\]

---

## 10.1 Rappel de la variable

Pour chaque participant, nous disposons de quatre estimations :

\[
M_{i,\text{AC}},
M_{i,\text{DA}},
M_{i,\text{MP}},
M_{i,\text{MT}}
\]

La moyenne personnelle est :

\[
\overline{M}_i
=
\frac{
M_{i,\text{AC}}
+
M_{i,\text{DA}}
+
M_{i,\text{MP}}
+
M_{i,\text{MT}}
}{4}
\]

Cette variable est identique pour tous les essais du même participant.

Elle mesure une différence interindividuelle :

> Certains participants sont-ils caractérisés par un nombre moyen de modèles mentaux estimé plus élevé que d’autres ?

---

## 10.2 Interprétation du coefficient

Le coefficient négatif suggère :

> Les participants ayant un nombre moyen de modèles mentaux supérieur d’un écart-type pourraient donner une confiance inférieure d’environ 2,24 points.

Mais l’intervalle de confiance est :

\[
[-5{,}343;0{,}862]
\]

Il contient zéro.

Les données sont compatibles avec :

- une association négative assez substantielle ;
- une association faible ;
- aucune association ;
- une petite association positive.

Nous ne pouvons donc pas établir clairement l’existence d’une association interindividuelle.

---

## 10.3 Pourquoi l’incertitude est-elle importante ?

Cette variable varie entre 141 participants, et non entre 9024 essais indépendants.

De plus, elle entre en concurrence statistique avec :

- la condition ;
- la précision moyenne ;
- l’interception aléatoire du participant.

L’interception aléatoire capture les différences générales de confiance entre individus qui ne sont pas expliquées par les prédicteurs mesurés.

Le nombre moyen de modèles doit donc expliquer une partie spécifique des différences entre participants, au-delà de toutes les autres différences individuelles.

---

## 10.4 Résultat du modèle de plafond

Dans le modèle logistique de la réponse 100 :

\[
OR\approx1{,}165
\]

avec un intervalle crédible supérieur à 1.

Cela suggère qu’un nombre moyen de modèles plus élevé est associé à une plus grande probabilité d’utiliser la confiance 100.

Mais dans le modèle linéaire final, l’association avec le niveau moyen de confiance est négative et non clairement détectée.

Cette apparente différence peut venir de plusieurs facteurs :

- les modèles n’ont pas la même variable dépendante ;
- l’un étudie tout le niveau de confiance ;
- l’autre étudie uniquement le passage à la valeur exacte 100 ;
- l’approximation bayésienne variationnelle peut sous-estimer certaines incertitudes ;
- les relations peuvent différer entre le centre et l’extrémité de la distribution.

Cette divergence invite à la prudence.

---

## 10.5 Conclusion sur l’effet interindividuel

Formulation recommandée :

> Le nombre moyen de modèles mentaux du participant présentait une association négative estimée avec la confiance, mais cette association n’était pas suffisamment précise pour être clairement distinguée de zéro. Les résultats relatifs à ce prédicteur interindividuel doivent donc être considérés comme exploratoires.

---

# 11. Effet intra-individuel du nombre de modèles mentaux

Résultat final :

\[
\beta_{\text{intra}}=-0{,}485
\]

\[
SE=0{,}231
\]

\[
IC_{95\%}=[-0{,}937;-0{,}032]
\]

\[
p=0{,}036
\]

---

## 11.1 Rappel de la décomposition intra-individuelle

La variable brute `number_models_generated` mélangeait deux questions :

1. Certains participants génèrent-ils généralement plus de modèles que d’autres ?
2. Pour un même participant, les tâches nécessitant plus de modèles que sa moyenne sont-elles associées à une confiance différente ?

Nous avons séparé ces deux niveaux.

La composante intra-individuelle est :

\[
M^{\text{within}}_{it}
=
M_{it}
-
\overline{M}_i
\]

où :

- \(M_{it}\) est le nombre de modèles du participant \(i\) pour le type de tâche \(t\) ;
- \(\overline{M}_i\) est sa moyenne personnelle.

---

## 11.2 Exemple concret

Supposons qu’un participant possède :

| Tâche | Nombre de modèles |
|---|---:|
| MT | 2 |
| MP | 2,4 |
| AC | 2,6 |
| DA | 5 |

Sa moyenne est :

\[
\frac{2+2{,}4+2{,}6+5}{4}
=
3
\]

Les composantes intra-individuelles sont :

| Tâche | Calcul | Valeur intra |
|---|---:|---:|
| MT | \(2-3\) | -1 |
| MP | \(2,4-3\) | -0,6 |
| AC | \(2,6-3\) | -0,4 |
| DA | \(5-3\) | +2 |

La tâche DA demande ici davantage de modèles que la moyenne personnelle.

---

## 11.3 Interprétation du coefficient final

Le coefficient négatif suggère :

> Pour un même participant, les types de tâche associés à un nombre de modèles mentaux supérieur à sa moyenne personnelle sont associés à une confiance légèrement plus faible.

La variable ayant été standardisée, une augmentation d’un écart-type de la composante intra-individuelle correspond à une diminution moyenne d’environ :

\[
0{,}485\text{ point de confiance}
\]

---

## 11.4 Taille de l’effet

L’effet est petit :

\[
-0{,}485
\]

Il représente moins d’un demi-point sur une échelle de 0 à 100 pour une augmentation d’un écart-type.

L’effet est donc :

- statistiquement détecté dans le modèle final ;
- cohérent avec l’idée qu’un raisonnement plus complexe réduit légèrement la confiance ;
- mais quantitativement faible.

Il ne faut pas le présenter comme un changement massif de confiance.

---

## 11.5 Pourquoi cet effet exige-t-il beaucoup de prudence ?

Dans le modèle cognitif avec validité, le coefficient \(N=20\) était :

\[
-0{,}342
\]

avec :

\[
p\approx0{,}190
\]

Il n’était alors pas clairement détecté.

Dans le modèle final sans validité, il devient :

\[
-0{,}485
\]

avec :

\[
p\approx0{,}036
\]

Le signe reste négatif dans les deux modèles, mais le franchissement du seuil de 0,05 dépend de la spécification.

### Cela signifie-t-il que l’effet est faux ?

Non.

Cela signifie que :

- la direction négative semble relativement cohérente ;
- la taille est faible ;
- la précision de l’estimation est limitée ;
- la conclusion « clairement différent de zéro » dépend du modèle choisi.

Nous devons donc éviter une formulation trop forte.

---

## 11.6 Proximité avec le seuil de 0,05

La valeur \(p\) est :

\[
0{,}036
\]

Elle est inférieure à 0,05, mais pas de manière spectaculaire.

L’intervalle de confiance est :

\[
[-0{,}937;-0{,}032]
\]

Sa borne supérieure est très proche de zéro.

Une légère modification de :

- la spécification ;
- la méthode d’estimation ;
- la définition de la variable ;
- l’échantillon ;
- la prise en compte de l’incertitude de MReasoner ;

pourrait rendre l’intervalle compatible avec zéro.

---

## 11.7 Une autre limite : la variable varie surtout avec le type de tâche

MReasoner fournit une valeur par :

```text
participant × type de tâche
```

Tous les essais du même participant et du même type reçoivent donc la même estimation.

La variable intra-individuelle n’explique pas une variation fine entre items d’un même type.

Par exemple, les différents essais DA d’un participant possèdent la même estimation MReasoner dans notre dataset.

L’effet peut donc refléter partiellement des différences entre types de tâches.

Nous avons vérifié le rôle du type de tâche, mais la structure reste importante pour l’interprétation.

---

## 11.8 Effet dans le modèle de plafond

Dans le modèle logistique de la valeur 100 :

\[
OR\approx0{,}920
\]

Cela signifie qu’une augmentation d’un écart-type du nombre de modèles intra-individuel est associée à une diminution d’environ 8 % des odds d’utiliser la réponse 100 :

\[
1-0{,}920=0{,}080
\]

Cette direction négative est cohérente avec le modèle linéaire final :

> Davantage de modèles relativement à la moyenne personnelle est associé à une confiance légèrement plus faible et à une utilisation un peu moins fréquente de la valeur 100.

---

## 11.9 Interprétation cognitive possible

Selon une interprétation fondée sur les modèles mentaux, devoir maintenir ou générer davantage de représentations pourrait :

- augmenter la charge cognitive ;
- multiplier les possibilités examinées ;
- diminuer le sentiment d’évidence ;
- augmenter l’incertitude ;
- réduire légèrement la confiance.

Cependant, notre analyse n’observe pas directement les modèles mentaux dans l’esprit des participants. Elle utilise les sorties d’un modèle computationnel ajusté à leur comportement.

La conclusion doit donc rester formulée comme une association entre :

- une estimation computationnelle ;
- et une mesure déclarée de confiance.

---

## 11.10 Conclusion sur l’effet intra-individuel

Formulation recommandée :

> Dans le modèle final fondé sur 20 simulations, un nombre de modèles supérieur à la moyenne personnelle était associé à une légère diminution de la confiance. Cet effet était faible et sa détection dépendait de la spécification statistique ; il doit donc être interprété comme un résultat suggestif plutôt que comme une conclusion définitive.

---

# 12. Composantes de variance et différences individuelles

Résultats finaux :

| Composante | Variance | Écart-type | Proportion |
|---|---:|---:|---:|
| Participant | 195,815 | 13,993 | 40,3 % |
| Item | 5,213 | 2,283 | 1,1 % |
| Résiduelle | 284,781 | 16,875 | 58,6 % |

La variance totale aléatoire est :

\[
195{,}815+5{,}213+284{,}781
=
485{,}809
\]

---

## 12.1 Variance participant

La variance entre participants est :

\[
195{,}815
\]

L’écart-type correspondant est :

\[
\sqrt{195{,}815}\approx13{,}993
\]

Cela signifie qu’après prise en compte des prédicteurs fixes, les participants diffèrent encore fortement dans leur niveau général de confiance.

Un participant situé à un écart-type au-dessus de la moyenne aléatoire possède un effet d’environ :

\[
+14\text{ points}
\]

Un participant situé à un écart-type en dessous possède un effet d’environ :

\[
-14\text{ points}
\]

La différence entre ces deux profils serait d’environ :

\[
28\text{ points}
\]

---

## 12.2 Variance item

La variance entre items est :

\[
5{,}213
\]

L’écart-type est :

\[
\sqrt{5{,}213}\approx2{,}283
\]

Après prise en compte de l’entropie et des autres prédicteurs, les différences résiduelles moyennes entre items sont donc relativement modestes.

---

## 12.3 Variance résiduelle

La variance résiduelle est :

\[
284{,}781
\]

L’écart-type résiduel est :

\[
\sqrt{284{,}781}\approx16{,}875
\]

Même après avoir pris en compte :

- la condition ;
- la séquence ;
- la précision ;
- l’entropie ;
- MReasoner ;
- les différences générales entre participants ;
- les différences générales entre items ;

il reste une variabilité importante entre les essais individuels.

---

## 12.4 Interprétation des proportions

### Participants

\[
\frac{195{,}815}{485{,}809}
\approx0{,}403
\]

Environ 40,3 % de la variance aléatoire restante est associée aux différences générales entre participants.

### Items

\[
\frac{5{,}213}{485{,}809}
\approx0{,}011
\]

Environ 1,1 % est associée aux différences résiduelles entre items.

### Résidus

\[
\frac{284{,}781}{485{,}809}
\approx0{,}586
\]

Environ 58,6 % reste au niveau des essais.

---

## 12.5 Pourquoi la variance item a-t-elle diminué par rapport au modèle nul ?

Dans le modèle nul :

\[
\sigma^2_{\text{item}}\approx11{,}87
\]

Dans le modèle final :

\[
\sigma^2_{\text{item}}\approx5{,}21
\]

L’ajout des prédicteurs, principalement l’entropie, explique une grande partie des différences systématiques entre items.

Le modèle final ne dit donc pas que les items sont devenus objectivement plus semblables.

Il dit :

> Une fois l’entropie et les autres prédicteurs pris en compte, il reste moins de variation inexpliquée entre les moyennes des items.

---

## 12.6 Pourquoi la variance participant reste-t-elle élevée ?

La variance participant passe approximativement de :

\[
200
\]

dans le modèle nul à :

\[
196
\]

dans le modèle final REML.

La diminution est modeste.

Nos prédicteurs interindividuels :

- précision moyenne ;
- nombre moyen de modèles ;

n’expliquent donc qu’une petite partie des grandes différences générales dans l’utilisation de l’échelle de confiance.

D’autres caractéristiques non mesurées pourraient intervenir :

- tendance générale à utiliser des valeurs élevées ;
- compréhension de l’échelle ;
- personnalité ;
- tolérance à l’incertitude ;
- motivation ;
- stratégie de réponse ;
- capacité métacognitive ;
- expérience préalable ;
- interprétation subjective du mot « confiance ».

---

## 12.7 Conséquence scientifique

Le résultat sur la variance participant est central :

> Les différences individuelles dans l’utilisation générale de l’échelle de confiance sont beaucoup plus importantes que les différences résiduelles moyennes entre items.

Cela justifie pleinement le recours au modèle mixte.

Une régression simple qui ignorerait les participants traiterait à tort les 9024 observations comme entièrement indépendantes.

---

# 13. Qualité explicative du modèle

Dans l’analyse cognitive initiale, le modèle possédait approximativement :

\[
R^2_{\text{marginal}}\approx0{,}033
\]

\[
R^2_{\text{conditionnel}}\approx0{,}434
\]

Les valeurs finales peuvent varier légèrement selon la spécification \(N=20\), mais l’interprétation générale reste la même.

---

## 13.1 Rappel : \(R^2\) marginal

Le **\(R^2\) marginal** mesure la proportion de variance expliquée par les effets fixes.

Les effets fixes sont :

- condition ;
- séquence ;
- précision ;
- entropie ;
- modèles moyens ;
- modèles intra-individuels.

Un \(R^2\) marginal proche de 0,03 signifie que les prédicteurs fixes expliquent environ 3 % de la variance totale selon cette méthode de calcul.

---

## 13.2 Rappel : \(R^2\) conditionnel

Le **\(R^2\) conditionnel** prend en compte :

- les effets fixes ;
- les effets aléatoires du participant ;
- les effets aléatoires de l’item.

Une valeur proche de :

\[
0{,}43
\]

indique que l’ensemble du modèle, y compris les différences stables entre participants et items, représente environ 43 % de la variance totale.

---

## 13.3 Pourquoi existe-t-il une grande différence ?

Parce que les effets aléatoires des participants sont très importants.

```text
Effets fixes mesurés
        └── environ 3 % de variance

Effets fixes + identité du participant + identité de l’item
        └── environ 43 % de variance
```

Cela signifie que savoir qui est le participant aide beaucoup à prévoir son niveau de confiance, même si nous ne savons pas encore exactement quelles variables psychologiques expliquent cette différence.

---

## 13.4 Un \(R^2\) marginal de 3 % est-il mauvais ?

Pas nécessairement.

Dans les données comportementales essai par essai :

- les réponses sont très variables ;
- de nombreux facteurs ne sont pas mesurés ;
- les mesures de confiance contiennent du bruit ;
- un effet de quelques points peut être scientifiquement intéressant.

De plus, l’objectif n’était pas nécessairement de prédire parfaitement chaque valeur de confiance, mais de tester certaines associations théoriques.

Cependant, nous devons être honnêtes :

> Les prédicteurs fixes étudiés n’expliquent qu’une petite partie de la variabilité totale de la confiance.

Le modèle identifie des tendances moyennes robustes, mais il ne fournit pas une prédiction individuelle précise de chaque réponse.

---

# 14. Interprétation des analyses de sensibilité

Une **analyse de sensibilité** consiste à modifier raisonnablement une décision analytique afin de voir si la conclusion principale change.

L’idée est la suivante :

> Un résultat crédible ne devrait pas disparaître dès que l’on change légèrement une décision raisonnable du modèle.

---

## 14.1 Validité contre type de tâche

La validité était structurellement liée au type de tâche :

| Type | Validité |
|---|---|
| AC | Invalide |
| DA | Invalide |
| MP | Valide |
| MT | Valide |

Nous avons comparé :

1. un modèle sans validité ni type ;
2. un modèle avec validité ;
3. un modèle avec le type de tâche complet.

Résultats :

### Ajout de la validité

\[
LR\approx1{,}203
\]

\[
p\approx0{,}273
\]

### Ajout du type de tâche

\[
LR\approx3{,}841
\]

\[
p\approx0{,}279
\]

Aucune des deux extensions n’améliore clairement le modèle.

---

## 14.2 Interprétation

Après prise en compte :

- de la condition ;
- de la séquence ;
- de l’entropie ;
- des prédicteurs individuels ;
- de MReasoner ;

la validité ou les catégories MP/MT/AC/DA n’ajoutent pas beaucoup d’information sur la confiance moyenne.

Cela ne signifie pas que la validité est psychologiquement sans importance dans l’absolu.

Cela signifie :

> Dans cette spécification et en présence des autres prédicteurs, nous ne détectons pas une contribution supplémentaire claire de la validité ou du type de tâche à la confiance.

---

## 14.3 Tests par retrait d’un prédicteur

Chaque prédicteur a été retiré séparément, puis le modèle réduit a été comparé au modèle complet avec un test du rapport de vraisemblance.

Le résultat le plus clair concernait l’entropie :

\[
LR\approx61{,}14
\]

\[
p\approx5{,}31\times10^{-15}
\]

Le retrait de l’entropie dégrade fortement le modèle.

Pour les autres prédicteurs cognitifs :

- précision : pas de dégradation claire ;
- modèles moyens : pas de dégradation claire ;
- modèles intra : dégradation faible ;
- validité : pas de dégradation claire.

Cette analyse renforce la hiérarchie des résultats :

```text
Entropie
  └── contribution robuste et nette

MReasoner intra
  └── contribution faible et dépendante de la spécification

MReasoner inter
  └── contribution incertaine

Précision moyenne
  └── contribution non détectée
```

---

## 14.4 AIC et complexité

L’**AIC** compare l’ajustement d’un modèle tout en pénalisant le nombre de paramètres.

Une valeur plus faible est préférable.

Mais une différence très petite d’AIC n’indique pas une supériorité importante.

Par exemple, lorsqu’un modèle réduit possède un AIC légèrement inférieur au modèle complet, cela signifie que le gain d’ajustement fourni par le prédicteur supplémentaire ne compense pas son coût de complexité.

Cela soutient la suppression des termes qui n’ajoutent pas suffisamment d’information.

---

# 15. Interprétation du problème de plafond

## 15.1 Qu’est-ce qu’un effet de plafond ?

Un **effet de plafond** apparaît lorsqu’une proportion importante des observations se trouve à la valeur maximale possible.

Ici :

\[
2336
\]

essais sur :

\[
9024
\]

ont une confiance égale à 100.

Le taux est :

\[
\frac{2336}{9024}\approx25{,}89\%
\]

Environ un quart des observations se trouvent donc au plafond.

---

## 15.2 Pourquoi est-ce un problème pour un modèle linéaire normal ?

Le modèle linéaire suppose une variable théoriquement non bornée :

\[
-\infty<Y<+\infty
\]

Mais la confiance est limitée :

\[
0\leq Y\leq100
\]

Le modèle peut théoriquement prédire :

- 105 ;
- 110 ;
- ou une valeur négative.

De plus, une masse importante à 100 produit :

- une asymétrie ;
- une troncature de la variabilité ;
- des résidus non normaux ;
- une relation potentiellement différente entre le centre et le plafond.

---

## 15.3 Ce que nous avons fait

Nous avons effectué deux analyses complémentaires.

### Analyse 1 — Exclure les valeurs 100

Nous avons ajusté le modèle sur les 6688 observations strictement inférieures à 100.

Cette analyse répond à :

> Parmi les réponses qui ne sont pas au plafond, retrouve-t-on les mêmes associations ?

### Analyse 2 — Modéliser la probabilité d’utiliser 100

Nous avons défini :

\[
C_i=
\begin{cases}
1 & \text{si confiance}=100\\
0 & \text{sinon}
\end{cases}
\]

Puis nous avons ajusté un modèle logistique mixte.

Cette analyse répond à :

> Quels prédicteurs expliquent l’utilisation de la valeur maximale ?

---

## 15.4 Résultats principaux sous le plafond

Pour les valeurs inférieures à 100 :

- condition : effet non clairement détecté ;
- séquence : effet non clairement détecté ;
- précision : effet non clairement détecté ;
- entropie : effet toujours fortement négatif ;
- modèles moyens : tendance négative ;
- modèles intra : effet non clairement détecté.

Le résultat clé est donc :

> L’effet d’entropie ne dépend pas uniquement de la présence des valeurs 100.

---

## 15.5 Résultats du modèle logistique de plafond

| Prédicteur | Odds ratio | Interprétation |
|---|---:|---|
| Standard | 3,956 | Utilisation de 100 beaucoup plus probable |
| Séquence | 0,850 | Utilisation de 100 moins probable au fil des essais |
| Précision participant | 0,832 | Association négative avec l’usage de 100 |
| Entropie | 0,740 | Utilisation de 100 moins probable pour les items entropiques |
| Modèles moyens | 1,165 | Utilisation de 100 légèrement plus probable |
| Modèles intra | 0,920 | Utilisation de 100 légèrement moins probable |

---

## 15.6 Prudence avec les intervalles variationnels

Le modèle logistique a été ajusté avec une approximation bayésienne variationnelle.

Cette méthode est utile pour les modèles binomiaux mixtes complexes, mais elle peut parfois sous-estimer l’incertitude.

Les intervalles crédibles très étroits doivent donc être interprétés avec davantage de prudence que s’ils provenaient d’un échantillonnage bayésien complet correctement diagnostiqué.

---

## 15.7 Conclusion sur le plafond

Le problème de plafond est réel, mais il ne détruit pas l’ensemble des résultats.

Il permet au contraire de préciser leur mécanisme :

- l’effet Standard est en grande partie un effet sur l’utilisation de 100 ;
- l’effet de séquence est en grande partie une baisse de l’utilisation de 100 ;
- l’effet d’entropie existe à la fois sur la probabilité d’utiliser 100 et parmi les réponses inférieures à 100.

L’entropie est donc le résultat le moins dépendant du plafond.

---

# 16. Interprétation de la calibration métacognitive

## 16.1 Résumé global

| Mesure | Valeur |
|---|---:|
| Confiance moyenne | 75,7 % |
| Exactitude observée | 62,3 % |
| Biais de calibration | +13,4 points |
| Score de Brier | 0,302 |
| AUC de type 2 moyenne | 0,522 |
| Différence de confiance correct–incorrect | 1,62 point |

---

## 16.2 Surconfiance globale

Les participants expriment en moyenne :

\[
75{,}7\%
\]

de confiance, alors qu’ils répondent correctement dans :

\[
62{,}3\%
\]

des essais.

Le biais est :

\[
75{,}7-62{,}3=13{,}4
\]

Ils sont donc globalement surconfiants.

---

## 16.3 Calibration et exactitude

Il ne faut pas confondre :

- être correct ;
- être confiant ;
- être bien calibré.

### Participant A

- exactitude : 90 % ;
- confiance : 90 %.

Il est très précis et bien calibré.

### Participant B

- exactitude : 60 % ;
- confiance : 60 %.

Il est moins précis, mais également bien calibré.

### Participant C

- exactitude : 60 % ;
- confiance : 90 %.

Il est surconfiant.

La calibration ne récompense donc pas uniquement la performance. Elle évalue la correspondance entre confiance et réussite.

---

## 16.4 Faible discrimination métacognitive

L’AUC moyenne est :

\[
0{,}522
\]

soit très proche du hasard :

\[
0{,}500
\]

La confiance moyenne est seulement environ 1,62 point plus élevée sur les réponses correctes que sur les réponses incorrectes.

Le modèle logistique donne :

\[
OR_{\text{confiance}}\approx0{,}992
\]

avec un intervalle crédible contenant 1.

Ces trois résultats convergent :

> La confiance discrimine faiblement les essais corrects des essais incorrects.

---

## 16.5 Comment peut-on être surconfiant tout en ayant une faible discrimination ?

Supposons qu’un participant donne :

- 85 quand il a raison ;
- 84 quand il a tort.

Il est presque toujours très confiant.

Sa confiance moyenne peut être largement supérieure à son exactitude, donc il est surconfiant.

Mais la différence entre les essais corrects et incorrects n’est que d’un point. Sa discrimination est faible.

C’est approximativement le type de phénomène suggéré par nos données à l’échelle globale.

---

## 16.6 Variabilité entre participants

Le biais de calibration individuel varie approximativement entre :

\[
-0{,}372
\quad\text{et}\quad
0{,}500
\]

Certains participants sont donc :

- fortement sous-confiants ;
- bien calibrés ;
- fortement surconfiants.

L’AUC varie également fortement entre participants.

La moyenne globale masque donc une grande hétérogénéité individuelle.

---

## 16.7 Limites de l’interprétation probabiliste de l’échelle

Nous avons traité :

\[
\text{confiance}=75
\]

comme une probabilité subjective de :

\[
0{,}75
\]

Mais les participants n’utilisent pas forcément l’échelle de cette manière.

Pour certains :

- 100 peut signifier « très confiant » et non « impossibilité absolue d’erreur » ;
- 50 peut signifier « confiance moyenne » et non « une chance sur deux » ;
- les distances entre 60 et 70 ne sont peut-être pas psychologiquement équivalentes aux distances entre 80 et 90.

Le biais de calibration doit donc être interprété comme une mesure utile, mais dépendante de cette hypothèse de correspondance entre échelle et probabilité.

---

## 16.8 Conclusion métacognitive

Formulation recommandée :

> Les participants présentaient une surconfiance globale d’environ 13 points de pourcentage. En revanche, leur confiance distinguait peu les réponses correctes des réponses incorrectes : l’AUC métacognitive moyenne était proche du hasard et le modèle logistique ne montrait pas d’association claire entre la confiance standardisée et l’exactitude. La confiance semblait donc refléter davantage un niveau subjectif général de certitude qu’un indicateur précis de la justesse de chaque réponse.

---

# 17. Interprétation de la robustesse de MReasoner

## 17.1 Pourquoi 3 simulations étaient insuffisantes

Avec seulement trois simulations, une sortie rare peut modifier fortement la moyenne.

Cette difficulté était particulièrement importante pour DA, dont les sorties étaient très variables.

Les comparaisons indiquaient :

| Comparaison | Différence absolue moyenne |
|---|---:|
| 3 contre 10 | 0,315 |
| 3 contre 20 | 0,384 |
| 10 contre 20 | 0,187 |

Les estimations à 10 et 20 simulations sont donc nettement plus proches entre elles.

---

## 17.2 Stabilité structurelle et stabilité numérique

Deux formes de stabilité doivent être distinguées.

### Stabilité structurelle

Les combinaisons ayant des valeurs élevées dans une version ont-elles aussi des valeurs élevées dans l’autre ?

Les corrélations élevées montrent une bonne stabilité structurelle.

### Stabilité numérique

Les valeurs exactes sont-elles proches ?

Les différences absolues montrent que les estimations à trois simulations ne sont pas toujours numériquement assez proches de celles à vingt simulations.

---

## 17.3 Pourquoi utiliser \(N=20\) pour les résultats finaux ?

Parce que \(N=20\) :

- réduit l’incertitude de la moyenne simulée ;
- est plus proche de \(N=10\) que \(N=3\) ;
- révèle mieux la variabilité interne ;
- fournit des prédicteurs MReasoner plus fiables ;
- permet une analyse finale plus défendable.

Cela ne signifie pas que 20 est une valeur magique ou parfaitement suffisante.

Pour une tâche très variable comme DA, davantage de simulations pourraient encore améliorer la précision.

---

## 17.4 Les conclusions changent-elles entre \(N=3\) et \(N=20\) ?

Les conclusions principales ne changent pas :

- condition positive ;
- séquence négative ;
- entropie négative ;
- modèles moyens négatifs mais incertains ;
- composante intra négative.

Les signes sont conservés.

Le passage à \(N=20\) modifie surtout la taille précise de certains coefficients MReasoner et leur incertitude.

---

## 17.5 L’effet intra-individuel est-il validé par \(N=20\) ?

Seulement de manière prudente.

Dans le modèle final sans validité, l’effet est détecté :

\[
\beta=-0{,}485,\quad p=0{,}036
\]

Mais dans le modèle avec validité, il reste non détecté.

Nous ne pouvons donc pas dire :

> Vingt simulations ont définitivement démontré l’effet MReasoner.

Nous pouvons dire :

> Les estimations à vingt simulations conservent une direction négative et, dans la spécification finale, l’effet intra-individuel est faiblement détecté. Toutefois, sa dépendance à la spécification impose une interprétation prudente.

---

## 17.6 Une limite plus profonde : l’incertitude de MReasoner n’est pas propagée

Dans le modèle statistique, `number_models_generated` est traité comme une valeur connue.

Pourtant, cette valeur est une moyenne simulée comportant une incertitude.

Le modèle linéaire ne tient pas directement compte du fait que :

- certaines estimations MReasoner sont très stables ;
- d’autres, notamment DA, sont beaucoup plus incertaines.

Une méthode plus avancée pourrait intégrer l’erreur de mesure du prédicteur ou propager les simulations jusqu’au modèle final.

### Approche possible

1. tirer une estimation MReasoner dans chaque distribution simulée ;
2. reconstruire le prédicteur ;
3. réajuster le modèle ;
4. répéter l’opération ;
5. observer la distribution des coefficients.

Cela permettrait de transmettre l’incertitude computationnelle jusqu’à l’incertitude statistique finale.

---

## 17.7 Conclusion sur MReasoner

Formulation recommandée :

> Les estimations issues de 10 et 20 simulations étaient fortement concordantes, tandis que les estimations fondées sur trois simulations étaient moins précises, particulièrement pour la tâche DA. L’utilisation de 20 simulations a renforcé la fiabilité numérique des prédicteurs sans modifier les conclusions générales. Les associations avec la confiance demeuraient toutefois modestes et plus sensibles à la spécification que l’effet de l’entropie.

---

# 18. Ce que disent les diagnostics

## 18.1 Résidus

Les résidus présentent :

- une moyenne proche de zéro ;
- une asymétrie négative ;
- un excès de kurtosis ;
- davantage de valeurs extrêmes que sous une normalité parfaite.

Le modèle linéaire ne représente donc pas parfaitement la distribution de la confiance.

---

## 18.2 Pourquoi la moyenne nulle ne suffit-elle pas ?

Un modèle peut être correctement centré en moyenne mais mal représenter :

- les extrémités ;
- la variance ;
- la forme de la distribution ;
- la borne à 100.

La moyenne des erreurs proche de zéro est nécessairement rassurante, mais elle n’est pas suffisante.

---

## 18.3 Influence

Le jackknife montre qu’aucun participant unique ne semble être responsable de :

- l’effet d’entropie ;
- l’effet de séquence ;
- la différence Standard–Neutral.

L’effet de précision est plus instable.

---

## 18.4 Le modèle linéaire final reste-t-il utilisable ?

Oui, comme modèle principal des différences moyennes, mais avec des réserves.

Les raisons de le conserver sont :

- interprétation simple en points de confiance ;
- prise en compte des participants et items ;
- stabilité des principaux coefficients ;
- analyses de sensibilité cohérentes ;
- modélisation complémentaire du plafond.

Les raisons de rester prudent sont :

- variable bornée ;
- forte accumulation à 100 ;
- résidus non normaux ;
- effets potentiellement différents au plafond et sous le plafond.

---

## 18.5 Alternative future : modèle en deux parties

Une extension naturelle serait un modèle en deux parties :

### Partie 1

Probabilité d’être exactement à 100 :

\[
P(Y=100)
\]

### Partie 2

Distribution de la confiance lorsque :

\[
Y<100
\]

Cette approche est parfois appelée modèle à inflation au plafond ou modèle en deux parties.

Elle correspond bien à ce que nos analyses séparées ont déjà suggéré :

```text
Processus A
  └── décision d’utiliser ou non 100

Processus B
  └── choix du niveau de confiance parmi les valeurs inférieures à 100
```

Notre combinaison :

- modèle linéaire complet ;
- modèle sans les 100 ;
- modèle logistique du plafond ;

constitue déjà une approximation pédagogique de cette logique.

---

# 19. Ce que nous pouvons conclure scientifiquement

## 19.1 Première conclusion : importance des différences individuelles

Les participants diffèrent fortement dans leur utilisation générale de l’échelle de confiance.

Cette variabilité représente environ 40 % de la variance aléatoire finale.

La confiance est donc largement une caractéristique du profil individuel, au-delà des caractéristiques mesurées de chaque essai.

---

## 19.2 Deuxième conclusion : effet robuste de l’entropie

L’entropie des réponses par item est négativement associée à la confiance.

Cet effet est :

- statistiquement très clair ;
- quantitativement de quelques points ;
- stable entre les spécifications ;
- stable sous le plafond ;
- stable dans le jackknife ;
- stable avec 20 simulations.

Il s’agit du résultat principal.

---

## 19.3 Troisième conclusion : effet de la condition porté par la valeur 100

La condition Standard est associée à une confiance plus élevée.

Cependant, cette différence semble provenir en grande partie d’une utilisation plus fréquente de la réponse maximale 100.

---

## 19.4 Quatrième conclusion : évolution au cours de l’expérience

La confiance moyenne diminue avec la position dans la séquence.

Cette diminution semble surtout refléter une baisse progressive de l’utilisation de la valeur 100.

---

## 19.5 Cinquième conclusion : effets MReasoner modestes

Le nombre moyen de modèles mentaux n’est pas clairement associé à la confiance.

La composante intra-individuelle présente une association négative faible dans le modèle final, mais cette conclusion dépend de la spécification.

Ces résultats sont compatibles avec l’idée d’une confiance légèrement plus faible lorsque davantage de modèles doivent être considérés, mais ils ne constituent pas une preuve forte.

---

## 19.6 Sixième conclusion : surconfiance et faible discrimination

Les participants sont globalement surconfiants.

Mais leur confiance distingue peu les réponses correctes des réponses incorrectes.

Le niveau de confiance ne constitue donc pas ici un indicateur très fiable de l’exactitude essai par essai.

---

# 20. Ce que nous ne pouvons pas conclure

## 20.1 Nous ne pouvons pas affirmer une causalité de l’entropie

L’entropie est associée à la confiance, mais elle n’a pas été manipulée.

Nous ne savons pas si :

- l’entropie réduit la confiance ;
- une difficulté sous-jacente augmente l’entropie et réduit la confiance ;
- une ambiguïté sémantique produit les deux ;
- plusieurs stratégies concurrentes produisent les deux.

---

## 20.2 Nous ne pouvons pas affirmer que MReasoner décrit directement l’esprit des participants

MReasoner est un modèle computationnel.

Ses sorties sont des représentations théoriques ou simulées.

Elles ne constituent pas une observation directe des modèles mentaux réellement présents dans l’esprit d’un participant.

---

## 20.3 Nous ne pouvons pas conclure que la précision n’a aucun lien avec la confiance

L’effet n’est pas détecté, mais son intervalle de confiance reste relativement large.

L’absence de détection n’est pas la preuve d’une absence exacte.

---

## 20.4 Nous ne pouvons pas traiter 9024 essais comme 9024 participants

Les essais sont regroupés dans 141 participants et 128 items.

Les conclusions interindividuelles reposent principalement sur le nombre de participants.

Le grand nombre total de lignes ne crée pas artificiellement 9024 unités indépendantes.

---

## 20.5 Nous ne pouvons pas supposer une distribution parfaitement normale

Les diagnostics montrent clairement des écarts.

Les résultats du modèle linéaire doivent être lus avec les analyses de sensibilité du plafond.

---

## 20.6 Nous ne pouvons pas affirmer que \(p<0{,}05\) rend un résultat définitivement vrai

Une valeur \(p\) inférieure à 0,05 indique une incompatibilité relative avec un effet nul sous les hypothèses du modèle.

Elle ne garantit pas :

- la réplication ;
- la causalité ;
- une grande taille d’effet ;
- l’absence de biais ;
- l’exactitude de toutes les hypothèses.

---

# 21. Forces et limites du projet

# 21.1 Forces

## 21.1.1 Plan croisé riche

Les données contiennent :

- 141 participants ;
- 128 items ;
- 9024 observations ;
- quatre types de tâches ;
- deux conditions.

Cela permet d’étudier simultanément les différences entre personnes et entre items.

## 21.1.2 Modèle mixte approprié

Les intercepts aléatoires croisés évitent de supposer à tort que toutes les observations sont indépendantes.

## 21.1.3 Décomposition de MReasoner

La séparation entre :

- effet interindividuel ;
- effet intra-individuel ;

évite de confondre deux questions scientifiques différentes.

## 21.1.4 Nombreuses analyses de sensibilité

Nous avons vérifié :

- validité contre type de tâche ;
- retrait des prédicteurs ;
- exclusion des valeurs 100 ;
- modélisation de la valeur 100 ;
- retrait successif des participants ;
- 3 contre 10 contre 20 simulations.

## 21.1.5 Diagnostic computationnel

Nous n’avons pas traité aveuglément les sorties de MReasoner comme parfaitement stables.

Nous avons quantifié leur variabilité.

## 21.1.6 Calibration métacognitive

L’analyse ne se limite pas au niveau moyen de confiance. Elle examine aussi sa relation avec l’exactitude.

---

# 21.2 Limites

## 21.2.1 Distribution bornée et plafond

Un quart des observations sont égales à 100.

Le modèle linéaire normal est donc une approximation.

## 21.2.2 Entropie calculée dans le même échantillon

L’entropie n’a pas été calculée sur un échantillon indépendant.

Une validation croisée renforcerait la conclusion.

## 21.2.3 Prédicteurs MReasoner peu granulaires

Le nombre de modèles varie par participant et type de tâche, pas par item précis.

La variable ne capture donc pas toutes les différences entre essais.

## 21.2.4 Incertitude computationnelle non propagée

Le modèle final traite la moyenne MReasoner comme connue, alors qu’elle est estimée par simulation.

## 21.2.5 Intercepts aléatoires seulement

Le modèle permet aux participants et aux items d’avoir des niveaux moyens différents, mais pas nécessairement des sensibilités différentes aux prédicteurs.

Par exemple, certains participants pourraient être très sensibles à l’entropie, d’autres pas.

Un modèle plus complexe pourrait inclure une pente aléatoire de l’entropie par participant, si les données et l’optimisation le permettent.

## 21.2.6 Condition entre participants

La différence Standard–Neutral est estimée entre deux groupes de personnes différents.

Elle est donc moins précise qu’une comparaison au sein des mêmes participants.

## 21.2.7 Calibration fondée sur une hypothèse d’échelle probabiliste

Diviser la confiance par 100 suppose que les participants utilisent l’échelle comme une probabilité subjective.

Cette hypothèse n’est pas garantie.

## 21.2.8 Multiplicité des analyses

De nombreux modèles et coefficients ont été examinés.

Plus on effectue de tests, plus le risque d’obtenir au moins un résultat inférieur à 0,05 par hasard augmente.

Cette question renforce la nécessité de distinguer :

- résultats principaux planifiés ;
- analyses de sensibilité ;
- analyses exploratoires.

L’effet d’entropie est moins préoccupant à ce niveau, car sa valeur \(p\) est extrêmement petite et sa robustesse est forte. L’effet intra-individuel de MReasoner, proche du seuil, nécessite davantage de prudence.

---

# 22. Proposition de rédaction finale des résultats

La section suivante constitue une version scientifique cohérente que vous pouvez adapter au rapport.

---

## 22.1 Données et stratégie analytique

L’analyse portait sur 9024 essais issus de 141 participants et de 128 items. La confiance, mesurée sur une échelle de 0 à 100, fut analysée au moyen d’un modèle linéaire mixte comprenant des intercepts aléatoires croisés pour les participants et les items. Le modèle final incluait la condition expérimentale, la position de l’essai, la précision moyenne du participant, l’entropie empirique de l’item, le nombre moyen de modèles mentaux estimé par participant et la composante intra-individuelle du nombre de modèles. Les prédicteurs cognitifs continus furent standardisés et la position de l’essai fut centrée et exprimée par tranches de dix essais. Les comparaisons de modèles furent effectuées par maximum de vraisemblance, tandis que les estimations finales des coefficients et des composantes de variance furent obtenues par maximum de vraisemblance restreint.

---

## 22.2 Modèle principal

La condition Standard était associée à une confiance supérieure à celle de la condition Neutral, \(\beta=5{,}247\), \(SE=2{,}522\), IC à 95 % \([0{,}303;10{,}191]\), \(p=0{,}038\). La confiance diminuait également avec la progression dans l’expérience : chaque tranche supplémentaire de dix essais était associée à une diminution moyenne d’environ 0,44 point, \(\beta=-0{,}438\), \(SE=0{,}097\), IC à 95 % \([-0{,}627;-0{,}249]\), \(p=5{,}75\times10^{-6}\).

L’entropie empirique de l’item constituait le prédicteur le plus robuste. Une augmentation d’un écart-type de l’entropie était associée à une diminution moyenne de 2,49 points de confiance, \(\beta=-2{,}487\), \(SE=0{,}273\), IC à 95 % \([-3{,}022;-1{,}953]\), \(p=7{,}58\times10^{-20}\).

La précision moyenne du participant n’était pas clairement associée à la confiance, \(\beta=0{,}696\), \(SE=1{,}584\), IC à 95 % \([-2{,}408;3{,}800]\), \(p=0{,}660\). De même, l’association interindividuelle entre le nombre moyen de modèles mentaux et la confiance n’était pas clairement détectée, \(\beta=-2{,}241\), \(SE=1{,}583\), IC à 95 % \([-5{,}343;0{,}862]\), \(p=0{,}157\).

La composante intra-individuelle du nombre de modèles présentait une faible association négative avec la confiance, \(\beta=-0{,}485\), \(SE=0{,}231\), IC à 95 % \([-0{,}937;-0{,}032]\), \(p=0{,}036\). Ainsi, pour un même participant, les types de tâche associés à un nombre de modèles supérieur à sa moyenne personnelle tendaient à susciter une confiance légèrement plus faible. Cet effet était toutefois sensible à la spécification du modèle et doit être interprété prudemment.

---

## 22.3 Composantes de variance

La variance entre participants était de 195,815, correspondant à un écart-type de 13,993 et à environ 40,3 % de la variance aléatoire totale. La variance résiduelle entre items était de 5,213, soit un écart-type de 2,283 et environ 1,1 % de la variance. La variance résiduelle au niveau des essais était de 284,781, correspondant à un écart-type de 16,875 et à environ 58,6 % de la variance.

Ces résultats indiquent que les différences individuelles dans l’utilisation générale de l’échelle de confiance étaient beaucoup plus importantes que les différences résiduelles moyennes entre items. La diminution de la variance item par rapport au modèle nul suggère qu’une partie substantielle des différences de confiance entre items était expliquée par les prédicteurs du modèle, en particulier l’entropie.

---

## 22.4 Analyses de sensibilité

L’ajout de la validité n’améliorait pas clairement le modèle cognitif, \(LR=1{,}203\), \(p=0{,}273\). Le remplacement de la validité par le type de tâche complet n’apportait pas non plus d’amélioration claire, \(LR=3{,}841\), \(p=0{,}279\). L’effet de l’entropie restait fortement négatif après contrôle du type de tâche.

Comme 25,9 % des observations étaient égales à la borne supérieure 100, le modèle fut également réajusté après exclusion de ces réponses. Dans cette analyse, l’effet de l’entropie restait fortement négatif, \(\beta=-2{,}307\), \(SE=0{,}306\), IC à 95 % \([-2{,}907;-1{,}706]\). En revanche, les effets de la condition et de la séquence n’étaient plus clairement détectés.

Un modèle logistique mixte de l’utilisation de la valeur 100 montrait que les odds de choisir cette borne supérieure étaient environ quatre fois plus élevées dans la condition Standard que dans la condition Neutral, \(OR=3{,}956\), intervalle crédible à 95 % \([3{,}450;4{,}535]\). Les odds d’utiliser 100 diminuaient avec la progression dans l’expérience, \(OR=0{,}850\) par tranche de dix essais, intervalle crédible à 95 % \([0{,}819;0{,}882]\). Ces résultats suggèrent que les effets de condition et de séquence observés dans le modèle linéaire étaient en grande partie liés à l’utilisation de la borne supérieure.

---

## 22.5 Calibration métacognitive

La confiance moyenne était de 75,7 %, tandis que l’exactitude observée était de 62,3 %, ce qui correspondait à une surconfiance globale de 13,4 points de pourcentage. Le score de Brier était de 0,302.

La discrimination métacognitive était faible. L’AUC de type 2 moyenne était de 0,522, soit une valeur proche du niveau du hasard. La confiance moyenne ne dépassait que de 1,62 point les essais incorrects sur les essais corrects. Dans un modèle logistique mixte prédisant l’exactitude, une augmentation d’un écart-type de confiance n’était pas clairement associée aux odds de réponse correcte, \(OR=0{,}992\), intervalle crédible à 95 % \([0{,}942;1{,}046]\).

---

## 22.6 Robustesse de MReasoner

Les estimations MReasoner obtenues avec 10 et 20 simulations étaient fortement concordantes, avec une corrélation de Pearson de 0,981 et une différence absolue moyenne de 0,187 modèle. Les estimations fondées sur trois simulations présentaient davantage de différences par rapport à celles fondées sur vingt simulations, avec une différence absolue moyenne de 0,384. La variabilité computationnelle était particulièrement importante pour la tâche DA, tandis que la tâche MT produisait systématiquement deux modèles.

Le passage de 3 à 20 simulations ne modifiait ni le signe des coefficients ni les conclusions générales du modèle cognitif. Il améliorait toutefois la précision numérique des prédicteurs MReasoner. Les résultats finaux furent par conséquent fondés sur les estimations obtenues avec vingt simulations.

---

# 23. Proposition de discussion scientifique

## 23.1 Résultat principal

Le résultat le plus stable de l’analyse concerne l’entropie empirique des items. Les participants étaient moins confiants pour les items suscitant une répartition plus équilibrée entre les réponses Yes et No. Cette association était indépendante de la condition, de la progression dans l’expérience, des différences moyennes de précision entre participants et des prédicteurs issus de MReasoner.

Une interprétation possible est que l’entropie collective capture une forme d’ambiguïté psychologique. Lorsqu’un item active des représentations, stratégies ou intuitions concurrentes, les réponses du groupe deviennent plus dispersées et les individus ressentent simultanément davantage d’incertitude. Cette interprétation reste néanmoins associative, puisque l’entropie fut calculée à partir des réponses observées et non manipulée expérimentalement.

---

## 23.2 Condition expérimentale

La condition Standard était associée à une confiance plus élevée que la condition Neutral. L’analyse détaillée de la borne supérieure indique cependant que cette différence reposait en grande partie sur une utilisation plus fréquente de la réponse 100. Le contenu sémantiquement familier des items Standard pourrait favoriser un sentiment d’évidence ou de compréhension, sans pour autant améliorer proportionnellement l’exactitude.

Cette interprétation est cohérente avec le profil métacognitif global : les participants exprimaient une confiance élevée, mais celle-ci discriminait faiblement les réponses correctes et incorrectes.

---

## 23.3 Progression dans l’expérience

La confiance moyenne diminuait au fil des essais, mais cette diminution disparaissait lorsque les valeurs égales à 100 étaient exclues. Le résultat semble donc refléter une réduction progressive des réponses de certitude maximale.

Plusieurs mécanismes sont envisageables :

- calibration progressive à la difficulté ;
- diminution de l’enthousiasme initial ;
- fatigue ;
- usage plus modéré de l’échelle après familiarisation.

L’expérience actuelle ne permet pas de distinguer ces explications.

---

## 23.4 MReasoner

Les résultats de MReasoner fournissent un soutien limité à l’hypothèse selon laquelle la génération de davantage de modèles mentaux serait associée à une moindre confiance.

L’effet interindividuel n’était pas clairement détecté. L’effet intra-individuel était négatif et faiblement détecté dans la spécification finale, mais il dépendait de la présence ou de l’absence de certains termes de contrôle.

Ces résultats peuvent indiquer que le nombre de représentations envisagées influence légèrement le sentiment de certitude. Cependant, ils peuvent également refléter :

- des différences entre types de tâches ;
- l’incertitude des simulations ;
- une mesure computationnelle trop peu spécifique aux items ;
- une relation plus complexe que la relation linéaire testée.

---

## 23.5 Métacognition

La combinaison d’une forte surconfiance moyenne et d’une faible discrimination correct–incorrect suggère que la confiance ne constitue pas ici un suivi précis de l’exactitude.

La confiance semble davantage refléter :

- une tendance individuelle générale ;
- les propriétés subjectives des items ;
- la familiarité du contenu ;
- l’usage de la borne supérieure ;

qu’une connaissance fiable de la correction de chaque réponse.

---

## 23.6 Implication méthodologique

L’importante variance entre participants indique qu’une analyse des seules moyennes de groupe aurait été insuffisante.

Les futurs travaux devraient examiner des variables capables d’expliquer ces différences individuelles, par exemple :

- style d’utilisation de l’échelle ;
- métacognition générale ;
- besoin de cognition ;
- tolérance à l’incertitude ;
- performance logique indépendante ;
- temps de réponse ;
- stratégie de raisonnement.

---

# 24. Résumé pédagogique de l’ensemble du projet

Nous pouvons maintenant résumer toute la logique en une seule chaîne.

```text
Question initiale
    │
    └── Pourquoi la confiance varie-t-elle ?
             │
             ▼
Préparation des données
    │
    ├── harmonisation des participants
    ├── construction des items
    ├── calcul de l’exactitude
    ├── calcul de l’entropie
    └── fusion avec MReasoner
             │
             ▼
Modèle nul
    │
    └── Où se trouve la variabilité ?
             │
             ├── beaucoup entre participants
             ├── un peu entre items
             └── beaucoup entre essais
             │
             ▼
Modèle de contrôle
    │
    ├── condition
    └── séquence
             │
             ▼
Modèle cognitif
    │
    ├── précision du participant
    ├── entropie de l’item
    ├── modèles moyens
    └── modèles intra-individuels
             │
             ▼
Analyses de sensibilité
    │
    ├── validité
    ├── type de tâche
    ├── retrait des prédicteurs
    └── comparaison des modèles
             │
             ▼
Problème de plafond
    │
    ├── modèle sans les 100
    └── modèle logistique de la valeur 100
             │
             ▼
Diagnostics
    │
    ├── résidus
    ├── normalité
    ├── observations atypiques
    └── jackknife
             │
             ▼
Calibration métacognitive
    │
    ├── surconfiance
    ├── Brier
    ├── AUC de type 2
    └── modèle exactitude ~ confiance
             │
             ▼
Robustesse computationnelle
    │
    ├── 3 simulations
    ├── 10 simulations
    └── 20 simulations
             │
             ▼
Conclusion
    │
    ├── effet robuste de l’entropie
    ├── effet Standard principalement au plafond
    ├── baisse séquentielle principalement au plafond
    ├── forte variabilité individuelle
    ├── faible calibration métacognitive
    └── effets MReasoner modestes et prudents
```

---

# 25. Conclusion générale

L’analyse de l’expérience E1 montre que la confiance ne dépend pas d’un seul mécanisme.

Elle résulte de plusieurs niveaux simultanés :

- un niveau individuel, avec de fortes différences générales entre participants ;
- un niveau lié aux items, notamment à leur entropie empirique ;
- un niveau expérimental, avec une confiance plus élevée en condition Standard ;
- un niveau temporel, avec une diminution de l’utilisation de la certitude maximale ;
- un niveau computationnel potentiel, associé au nombre de modèles mentaux ;
- un niveau métacognitif, caractérisé par une surconfiance et une faible discrimination de l’exactitude.

Le résultat principal peut être formulé ainsi :

> **Les participants étaient moins confiants pour les items suscitant davantage de désaccord collectif. Cette association entre l’entropie des réponses et la confiance était le résultat le plus fort et le plus robuste de l’analyse.**

Le résultat sur la condition peut être formulé ainsi :

> **Les participants de la condition Standard exprimaient une confiance plus élevée que ceux de la condition Neutral, principalement parce qu’ils utilisaient plus souvent la valeur maximale 100.**

Le résultat temporel peut être formulé ainsi :

> **L’utilisation de la certitude maximale diminuait au cours de l’expérience, expliquant une grande partie de la baisse moyenne de confiance avec la séquence.**

Le résultat MReasoner doit être formulé avec davantage de prudence :

> **Un nombre de modèles supérieur à la moyenne personnelle était associé à une légère diminution de confiance dans la spécification finale, mais cet effet était faible et dépendait de certaines décisions de modélisation.**

Enfin, le résultat métacognitif peut être résumé ainsi :

> **Les participants étaient globalement surconfiants et leur niveau de confiance distinguait faiblement les réponses correctes des réponses incorrectes.**

La hiérarchie finale des preuves est donc :

| Niveau de confiance scientifique | Résultat |
|---|---|
| **Très fort** | Association négative entre entropie de l’item et confiance |
| **Modéré** | Confiance plus élevée en condition Standard, principalement via l’usage de 100 |
| **Modéré** | Diminution de la confiance au fil des essais, principalement via l’usage de 100 |
| **Descriptif fort** | Grande variabilité générale entre participants |
| **Descriptif fort** | Surconfiance globale et faible discrimination métacognitive |
| **Faible à exploratoire** | Association interindividuelle avec le nombre moyen de modèles |
| **Suggestif mais fragile** | Association intra-individuelle négative avec le nombre de modèles |

Cette hiérarchie est essentielle : elle permet de ne pas donner le même poids à tous les coefficients et de construire une conclusion scientifique proportionnée à la solidité réelle des résultats.