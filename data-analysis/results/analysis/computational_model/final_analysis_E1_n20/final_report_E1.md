# Analyse de l’expérience E1

## Méthodes

### Participants et plan expérimental

L’analyse portait sur 9 024 essais provenant de 141 participants et de
128 items. Chaque participant réalisa 64 essais. La condition expérimentale
était manipulée entre les participants : 71 participants furent affectés à
la condition Standard et 70 à la condition Neutral.

Les items appartenaient à quatre formes d’inférence : modus ponens (MP),
modus tollens (MT), affirmation du conséquent (AC) et déni de l’antécédent
(DA). Les formes MP et MT correspondaient aux essais logiquement valides,
tandis que les formes AC et DA correspondaient aux essais invalides. La
validité et le type de tâche étaient donc structurellement liés et ne furent
pas introduits simultanément dans le même modèle.

### Variable dépendante

La variable dépendante principale était la confiance déclarée, mesurée sur
une échelle allant de 0 à 100. La confiance moyenne était de 75,74
(ET = 22,30) et sa médiane était de 77. Un effet plafond était présent :
2 336 observations, soit 25,89 % des essais, avaient une confiance égale à
100.

### Précision individuelle

La précision moyenne de chaque participant fut calculée comme la proportion
de ses réponses correctes sur les 64 essais expérimentaux. La précision
moyenne entre participants était de 0,623.

### Entropie des items

L’incertitude empirique associée à chaque item fut mesurée par l’entropie
binaire de Shannon calculée à partir de la distribution des réponses « Yes »
et « No » produites par les participants pour cet item. Une entropie élevée
indiquait que les réponses étaient réparties de manière relativement
équilibrée entre les deux options, tandis qu’une entropie faible indiquait
un consensus plus important.

Cette variable constitue une mesure descriptive du désaccord observé dans
l’échantillon. Son association avec la confiance doit donc être interprétée
comme une association empirique et non comme un effet causal.

### Estimation du nombre de modèles mentaux

Le nombre de modèles mentaux générés par MReasoner fut estimé séparément pour
chaque combinaison participant × type de tâche. Les estimations finales
reposaient sur 20 simulations par combinaison.

Le prédicteur fut décomposé en deux composantes :

1. `subject_mean_models`, correspondant au nombre moyen de modèles générés
   par le participant, et représentant les différences interindividuelles ;
2. `models_within_subject`, correspondant à l’écart entre le nombre de
   modèles générés pour un type de tâche et la moyenne personnelle du
   participant, et représentant la variation intra-individuelle.

Une analyse de robustesse compara les estimations reposant sur 3, 10 et
20 simulations. Les estimations obtenues avec 10 et 20 simulations étaient
fortement corrélées, r = .981, et le passage de 3 à 20 simulations ne
modifia ni la direction ni le statut statistique des coefficients du modèle
cognitif initial.

### Préparation des prédicteurs

La précision individuelle, l’entropie des items, le nombre moyen de modèles
et la composante intra-individuelle du nombre de modèles furent standardisés.
Le coefficient de chacun de ces prédicteurs correspond donc à une
augmentation d’un écart-type.

La position de l’essai fut centrée sur la position moyenne et divisée par
dix. Son coefficient représente ainsi la variation moyenne de confiance
associée à dix essais supplémentaires.

### Modèle statistique

Un modèle linéaire mixte fut ajusté avec des intercepts aléatoires croisés
pour les participants et les items. Le modèle final était :

confidence ~ condition
             + sequence_c10
             + subject_accuracy_z
             + item_entropy_z
             + subject_mean_models_z
             + models_within_subject_z
             + (1 | participant)
             + (1 | item)

Les modèles différant par leurs effets fixes furent comparés par maximum de
vraisemblance. Après sélection de la spécification finale, les coefficients
et les composantes de variance furent estimés par maximum de vraisemblance
restreint. Cette séparation est appropriée, car les vraisemblances REML ne
doivent pas servir à comparer des modèles ayant des structures d’effets
fixes différentes. ([statsmodels.org](https://www.statsmodels.org/stable/generated/statsmodels.regression.mixed_linear_model.MixedLM.fit.html?utm_source=openai))

### Analyses de sensibilité

Plusieurs analyses de sensibilité furent réalisées :

1. remplacement de la validité par le type détaillé de tâche ;
2. tests de retrait individuel des prédicteurs cognitifs ;
3. réestimation du modèle après exclusion des réponses de confiance égales
   à 100 ;
4. modèle logistique mixte de la probabilité d’utiliser la valeur 100 ;
5. comparaison des estimations MReasoner fondées sur 3, 10 et 20
   simulations ;
6. analyses leave-one-subject-out, dans lesquelles le modèle fut réestimé
   après retrait successif de chacun des 141 participants.

Le modèle logistique du plafond fut estimé par une approximation
variationnelle bayésienne avec des intercepts aléatoires croisés pour les
participants et les items. Cette méthode correspond à l’une des procédures
d’estimation disponibles pour les modèles binomiaux mixtes de statsmodels.
([statsmodels.org](https://www.statsmodels.org/stable/mixed_glm.html?utm_source=openai))

### Calibration métacognitive

La calibration globale fut décrite en comparant la confiance exprimée à la
proportion de réponses correctes. Le biais de calibration fut défini comme
la différence entre la confiance moyenne, divisée par 100, et la précision
observée.

La discrimination métacognitive fut évaluée à partir :

- de la différence de confiance entre réponses correctes et incorrectes ;
- de l’AUC de type 2 calculée séparément pour chaque participant ;
- d’un modèle logistique mixte prédisant l’exactitude à partir de la
  confiance.

## Résultats

### Comparaison des modèles

Le modèle de contrôle, comprenant la condition et la position de l’essai,
s’ajustait mieux aux données que le modèle nul,

χ²(2) = 24,47, p < .001.

L’ajout conjoint de la précision individuelle, de l’entropie et des deux
composantes du nombre de modèles mentaux améliorait fortement le modèle de
contrôle,

χ²(4) = 73,10, p < .001.

Les tests de retrait individuel indiquaient que cette amélioration globale
était principalement portée par l’entropie des items. Le retrait de
l’entropie dégradait fortement l’ajustement,

χ²(1) = 61,14, p < .001, ΔAIC = 59,14.

L’ajout de la validité n’améliorait pas significativement le modèle,

χ²(1) = 1,20, p = .273,

pas plus que l’ajout du type détaillé de tâche,

χ²(3) = 3,84, p = .279.

Le modèle parcimonieux ne comprenant ni la validité ni le type de tâche fut
donc retenu pour les analyses finales.

### Effets fixes du modèle final

#### Condition expérimentale

La confiance était plus élevée dans la condition Standard que dans la
condition Neutral,

β = 5,25, SE = 2,52, z = 2,08, p = .038,
IC à 95 % [0,30 ; 10,19].

À position moyenne et pour des prédicteurs cognitifs fixés à leur moyenne,
la condition Standard était ainsi associée à une augmentation moyenne
d’environ 5,25 points de confiance.

#### Position de l’essai

La confiance diminuait au cours de l’expérience,

β = −0,438, SE = 0,097, z = −4,53, p < .001,
IC à 95 % [−0,627 ; −0,249].

Ce coefficient correspond à une diminution moyenne d’environ 0,44 point
pour dix essais supplémentaires, soit approximativement 2,76 points entre
le premier et le dernier essai.

#### Précision individuelle

La précision moyenne du participant n’était pas clairement associée à son
niveau moyen de confiance,

β = 0,70, SE = 1,58, z = 0,44, p = .660,
IC à 95 % [−2,41 ; 3,80].

#### Entropie de l’item

Une entropie plus élevée était associée à une confiance plus faible,

β = −2,49, SE = 0,27, z = −9,11, p < .001,
IC à 95 % [−3,02 ; −1,95].

Une augmentation d’un écart-type de l’entropie correspondait donc à une
diminution moyenne d’environ 2,49 points de confiance.

#### Nombre moyen de modèles mentaux

La composante interindividuelle du nombre de modèles mentaux n’était pas
significativement associée à la confiance,

β = −2,24, SE = 1,58, z = −1,42, p = .157,
IC à 95 % [−5,34 ; 0,86].

On ne peut donc pas conclure que les participants générant en moyenne
davantage de modèles mentaux exprimaient une confiance générale différente.

#### Variation intra-individuelle du nombre de modèles

La composante intra-individuelle du nombre de modèles était négativement
associée à la confiance dans le modèle final,

β = −0,485, SE = 0,231, z = −2,10, p = .036,
IC à 95 % [−0,937 ; −0,032].

Lorsqu’un type de tâche entraînait la génération d’un nombre de modèles
supérieur d’un écart-type au niveau personnel moyen, la confiance diminuait
en moyenne d’environ 0,49 point.

Cet effet est cohérent avec l’hypothèse selon laquelle la représentation de
plusieurs possibilités réduit la certitude subjective. Il doit néanmoins
être interprété avec prudence : il était faible et n’était pas significatif
dans toutes les spécifications, notamment dans le modèle comprenant la
validité.

### Composantes de variance

La variance entre participants était de 195,82, soit 40,3 % de la variance
totale du modèle. La variance entre items était de 5,21, soit 1,1 %, et la
variance résiduelle était de 284,78, soit 58,6 %.

Les différences individuelles dans l’utilisation de l’échelle de confiance
étaient donc beaucoup plus importantes que les différences moyennes
restantes entre les items.

### Stabilité leave-one-subject-out

L’effet négatif de l’entropie était particulièrement stable après retrait
successif de chacun des participants. Dans l’analyse initiale à trois
simulations, son estimation variait seulement entre −2,53 et −2,44 et ne
changeait jamais de signe.

Les effets de condition, de séquence et de la composante intra-individuelle
du nombre de modèles ne changeaient pas non plus de signe. En revanche,
l’estimation associée à la précision individuelle changeait parfois de
signe, confirmant son manque de stabilité.

### Sensibilité au plafond

Au total, 25,89 % des réponses de confiance étaient égales à 100.

Après exclusion de ces réponses, l’effet négatif de l’entropie demeurait
important,

β = −2,31, SE = 0,31, p < .001,
IC à 95 % [−2,91 ; −1,71].

En revanche, les effets de condition et de séquence n’étaient plus détectés
parmi les réponses inférieures à 100. Cette analyse conditionne cependant
l’échantillon sur la variable dépendante et doit être considérée comme une
analyse de sensibilité plutôt que comme un remplacement du modèle principal.

### Modèle logistique de l’utilisation de la borne supérieure

Le modèle logistique mixte bayésien de la probabilité de répondre 100
convergea correctement.

Les odds d’utiliser la borne supérieure étaient presque quatre fois plus
élevées dans la condition Standard que dans la condition Neutral,

β = 1,38, OR = 3,96,
IC crédible à 95 % de l’OR [3,45 ; 4,53].

Chaque tranche de dix essais était associée à une diminution d’environ 15 %
des odds d’utiliser la valeur 100,

β = −0,162, OR = 0,850,
IC crédible à 95 % [0,819 ; 0,882].

Une augmentation d’un écart-type de l’entropie diminuait les odds d’utiliser
la valeur maximale d’environ 26 %,

β = −0,301, OR = 0,740,
IC crédible à 95 % [0,693 ; 0,791].

Ces résultats indiquent que les effets de condition et de séquence observés
dans le modèle linéaire complet reflétaient en grande partie des variations
dans la propension à utiliser la borne supérieure. En revanche, l’effet
négatif de l’entropie concernait à la fois le niveau de confiance sous le
plafond et la probabilité d’utiliser la valeur 100.

### Calibration métacognitive

La confiance moyenne était de 75,7 %, alors que la précision observée était
de 62,3 %. Le biais global était donc de +13,4 points de pourcentage, ce qui
indique une surconfiance moyenne.

Le score de Brier était de 0,302 et l’erreur absolue de calibration pondérée
était de 0,210.

La discrimination métacognitive individuelle était faible :

- AUC de type 2 moyenne = 0,522 ;
- AUC de type 2 médiane = 0,511 ;
- différence moyenne de confiance entre réponses correctes et incorrectes
  = 1,62 point.

Dans le modèle logistique mixte, une augmentation d’un écart-type de
confiance n’était pas clairement associée à une augmentation des odds de
réponse correcte,

β = −0,008, OR = 0,992,
IC crédible à 95 % [0,942 ; 1,046].

La confiance semblait donc refléter davantage un niveau général de certitude
ou un style individuel d’utilisation de l’échelle qu’une capacité à
distinguer les réponses correctes des réponses incorrectes essai par essai.

### Diagnostics

Les résidus du modèle final présentaient une asymétrie négative de −1,43 et
un excès de kurtosis de 4,40. Le test de Shapiro–Wilk rejetait fortement la
normalité, mais ce résultat devait être interprété avec prudence en raison
de la grande taille de l’échantillon.

Environ 5,78 % des résidus standardisés dépassaient |2| et 1,96 %
dépassaient |3|. Les écarts à la normalité étaient principalement cohérents
avec la borne supérieure de l’échelle et l’accumulation des réponses à 100.

Les analyses spécifiques au plafond et le modèle limité aux réponses
inférieures à 100 furent utilisés pour vérifier que le principal effet
d’entropie ne dépendait pas de cette violation.

## Conclusion générale

L’entropie empirique de l’item constituait le prédicteur le plus important
et le plus robuste de la confiance. Les participants étaient moins confiants
pour les items suscitant davantage de désaccord dans l’échantillon. Cette
association subsistait après contrôle du type de tâche, après exclusion des
réponses situées au plafond et dans le modèle de la probabilité d’utiliser
la valeur 100.

Le nombre moyen de modèles mentaux ne présentait pas d’association
interindividuelle claire avec la confiance. En revanche, le modèle final
reposant sur 20 simulations indiquait un faible effet intra-individuel :
les types de tâches entraînant la génération d’un nombre de modèles
supérieur au niveau personnel moyen étaient associés à une confiance
légèrement plus faible. Cet effet étant sensible à la spécification du
modèle, il doit être considéré comme secondaire par rapport à l’effet
d’entropie.

Enfin, les participants manifestaient une surconfiance moyenne et une faible
discrimination métacognitive entre leurs réponses correctes et incorrectes.
Les différences stables entre participants expliquaient une part importante
de la variabilité de confiance.
