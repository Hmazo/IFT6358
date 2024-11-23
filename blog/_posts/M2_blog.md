# 3. Modèles de base

Dans cette troisième étape du Milestone, nous considérons les caractéristiques de distance et d'angle produites dans la section précédente afin de mettre en pratique et tester des modèles simples. L’objectif est de tester la performance des modèles de base afin d’avancer de nouveaux modèles, avec d’autres caractéristiques plus avancées.
Pour ce faire nous procédons de la manière suivante :

### 3.1 Question 1 *Évaluation sur l'ensemble de validation*

En utilisant la caractéristique de distance seulement, cette étape consiste à créer une division d'entraînement et de validation et à entraîner un classificateur de régression logistique simple. Autrement dit, un classificateur de régression logistique avec des paramètres complètement par défaut : 
clf = LogisticRegression()
clf.fit(X, y)

Évaluation de la précision du modèle sur l'ensemble de validation :  

Les résultats de ce premier modèle mettent en lumière un aspect important de nos données : 
Les métriques indiquent un problème grave avec votre modèle de régression logistique, probablement dû à un déséquilibre de classe. Bien que la précision soit élevée, à savoir 94,87 %, la précision, le rappel et le score F1 sont de 0,0, ce qui suggère que le modèle n'a pas réussi à faire de prédictions positives. L’une des explications pourrait découlée de l’existence d'une dépendance excessive à la prédiction de la classe majoritaire, due au déséquilibre de nos données (classes déséquilibrées). Bien que la précision soit élevée, elle est trompeuse, car elle reflète des prédictions correctes pour la classe majoritaire, et donc dominante seulement, mais pas pour l’autre classe qui s’avère être minoritaire. On se retrouve donc avec des résultats qui affirme que la caractéristique ‘distance’ est incapable de différencier les 2 classes en raison d’un biais des données à l’échelle de la classe regroupé par (‘is_goal=0’)

### 3.2 Questions 2 et 3 - *figures*

**Figure 1 : Les courbes Receiver Operating Characteristic (ROC)**

![Courbes ROC des modèles de base](images\roc_base.png)

L’objectif de cette partie était de générer les courbes ROC de 3 modèles de classification par régression logistique, en considérant la distance, l’angle ainsi que la distance t l’angle ensemble. Comme base de comparaison, nous considérons une ligne de base aléatoire. On se retrouve finalement avec 4 modèles. Les valeurs AUC indiquent que les modèles de classification par régression logistique, fondés sur la distance et la distance et l’angle sont plus ou moins similaires et surperforment le modèle basé sur l’angle seulement. De même, les résultats de ce dernier (modèle logistique basé sur l’angle seulement) sont plus ou moins rapprochés a ceux de la ligne de base aléatoire.

**Figure 2 : Taux de buts comme une fonction du centile de la probabilité de tir du modèle** 

![taux de buts comme une fonction du centile de la probabilité de tir donnée par le modèle de base](images\taux_de_but_base.png)

Le deuxième graphique représente le taux de buts réussis et ratés comme une fonction du centile de la probabilité de tir donnée par le modèle. Les courbes démontrent que pour les deux modèles ‘distance’ et ‘distance et angle’, les courbes sont croissantes, indiquant que les probabilités prédites correspondent bien aux taux de buts observés. Cela signifie que, contrairement au modèle 2, les modèles 1 et 3 sont capables de distinguer avec précision les tirs les plus susceptibles de se transformer en buts.

**Figure 3: Proportion cumulée de buts comme une fonction du centile de la probabilité de tir du modèle**

![Proportion cumulative de buts des modèles de base](images\cumulative_base.png)

Le graphique du taux de réussite des buts en fonction du centile de la probabilité de tir nous a permis d'identifier les zones de performance des modèles étudiés et d'évaluer leur précision. Les courbes des modèles 1 et 3 suivent une tendance croissante avec l'augmentation des centiles, ce qui signifie que ces modèles ont une bonne capacité de discrimination par rapport au modèle 2 qui possède une courbe non croissante qui se rapproche de la “ligne de base aléatoire”.

**Figure 4 : Diagramme de fiabilité** 

![Calibration (Courbe de fiabilité) modèles de base](images\calibration_base.png)

Le diagramme de fiabilité nous permet d’élucider à quel degré les probabilités prédites correspondent aux taux réels de réussite. Pour nos modèles, on remarque que ceux-ci ne dévient pas par rapport à la diagonale, c’est-à-dire qu’ils sont bien calibrés. Toutefois, ceci n’est pas le cas pour la ligne de base aléatoire où est ce qu’on remarque que ce modèle tend à sous-estimer les probabilités, du fait que la courbe est située au-dessous de la diagonale. 

 
# 6. Tentatives de plusieurs modèles

Dans cette section, nous explorons une variété de modèles. Bien entendu, l’objectif n’étant pas de proposer un modèle idéal et optimisé en fonction de nos données, nous nous efforçons à tester plusieurs modèles afin de d’explorer les divers apports de chacun, ainsi que ses failles. 

**Arbre de décision**

Pour ce faire, nous commençons par un arbre de décision, ou est-ce qu’une recherche aléatoire (‘RandomizedSearchCV’) fut favorisée afin d’optimiser les hyperparamètres du modèle. Par la suite, en raison du déséquilibre des classes, nous optons pour SMOTE afin de contrer le déséquilibre de nos données. 

La mise en parallèle du premier modèle et celui utilisant SMOTE met en évidence l'impact du traitement du déséquilibre de classes. Bien que le modèle initial (optimisation des hyperparamètres) atteint une ‘accuracy’ de 94,75 % et une précision de 47,93 %, celui-ci ne permet pas d’identifier efficacement la classe minoritaire, avec un rappel de seulement 2,51 % et un F1-score de 4,77 %. 
Après l'application de SMOTE, l'exactitude diminue à 80,21 %, reflétant le changement du modèle, en raison d’une prise en considération de la classe minoritaire. Le rappel s'est considérablement amélioré, passant de 2,51% à 47,20 %, ce qui indique que le modèle identifie désormais beaucoup plus de vrais positifs. 
Toutefois, la précision a diminué à 12,83 %, et avec un F1-score plus bas de 20,17 %, nous concluons l’existence d’un compromis entre la précision et le rappel. Autrement dit, bien que les résultats démontrent que SMOTE améliore la capacité du modèle à détecter la classe minoritaire, ceci est atteint au détriment de la précision (tradeoff précision/rappel). 
Les 4 figures générées pour les modèles de base furent de même générées pour l'arbre de décision: 

**Figure 1 : Les 4 figures d'évaluations tel que définit auparavant**
![Courbes ROC de l'arbre de décision](images\roc_tree.png)
![taux de buts comme une fonction du centile de la probabilité de tir donnée par l'arbre de décision](images\taux_de_but_tree.png)
![Proportion cumulative de buts de l'arbre de décision](images\cumulative_tree.png)
![Calibration (Courbe de fiabilité) arbre de décision](images\calibration_tree.png)

**LightGBM**
Dans la même trajetoire, nous avons considérons "LightGBM", un modèle performant pour les grands ensembles de données , en raison des caractéristiques du 'Gradient-Based One-Side Sampling' et du 'Exclusive Feature Bundling', optimisant vitesse et mémoire. Le modèle s'adapte bien aux données déséquilibrées, ce qui est le cas de nos données. 
Dans ca cadre, nous avons opté pour Dla stratégie de sélection de caractéristiques suivante : lgb.plot_importance(clf, importance_type="gain")
Cette stratégie nous permet de définir l'importance des caractéristiques en les évaluant en terme de gain moyen lorsque chaque caractérsitique est incluse dans les arbres de décision.
D'après les résultats, avec une 'accuracy' de 0.9498, une précision de 0,6392 et un score auc de 0.8388, ces score reflètent une capacité de discrimination efficace (toujours avec une marge d'amélioration)
Voici les 4 figures d'évaluation tel que les autres modèles:

**Figure 2 : Les 4 figures d'évaluations tel que définit auparavant**
![Courbes ROC de LightGBM](images\roc_lightgbm.png)
![taux de buts comme une fonction du centile de la probabilité de tir donnée par LightGBM](images\taux_de_but_lightgbm.png)
![Proportion cumulative de buts de LightGBM](images\cumulative_lightgbm.png)
![Calibration (Courbe de fiabilité) LightGBM](images\calibration_lightgbm.png)





