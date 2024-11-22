## Partie 2 Ingénierie des caractéristiques I

![Histogramme Tirs par Distance](images\fe1_histo_tir_par_distance.png) 

Ce graphique présente la distribution des tirs en fonction de leur distance au filet, en distinguant les tirs réussis (buts) et les tirs manqués. Les tirs manqués (en bleu) constituent une majorité écrasante par rapport aux tirs réussis (en vert), quelle que soit la distance. Cela reflète une réalité bien connue du hockey : la difficulté de marquer un but augmente avec la distance.

En observant les barres, on constate que la majorité des tirs sont effectués à une distance relativement courte, généralement entre 10 et 30 pieds. Les tirs à courte distance ont également un pourcentage légèrement plus élevé de succès, bien que les buts restent une minorité même dans cette zone. À mesure que la distance augmente, le nombre total de tirs diminue fortement, et les tirs réussis deviennent extrêmement rares au-delà de 50 pieds.

Ces observations renforcent l'idée que la proximité du filet est un facteur clé pour maximiser les chances de marquer, bien que d'autres éléments comme l'angle, le type de tir et la situation de jeu aient également une influence significative. Ce graphique met en évidence l'importance d'intégrer la distance comme caractéristique dans notre modèle prédictif des buts.

![Histogramme Tir par angle](images\fe1_histo_tir_par_angle.png)

En examinant l'impact de l'angle des tirs, nous observons que les tirs alignés avec l'axe central (autour de 0 degré) sont significativement plus nombreux et présentent une proportion plus élevée de buts. Cela peut être attribué à une meilleure perspective du joueur et à une diminution des interférences défensives. Les tirs effectués à des angles extrêmes, bien que rares, ont des taux de réussite beaucoup plus faibles, ce qui s'explique par des perspectives limitées et des chances de placement plus difficiles.

![Histogramme 2D](images\fe1_histo_2d_angle_distance.png) 

La visualisation 2D combinant la distance et l'angle des tirs met en évidence la répartition des tirs réussis et ratés en fonction de leur position sur la patinoire. Les tirs situés à proximité du filet (distances inférieures à 20 pieds) montrent une densité notablement plus élevée de buts, signalée par la concentration des points orange. À mesure que la distance augmente, la proportion de tirs ratés (points bleus) devient prépondérante. Cette tendance est renforcée par l'angle des tirs : les tirs effectués à des angles plus proches de l'axe central (environ 0 degré) semblent avoir une probabilité plus élevée de succès. Cela peut s'expliquer par une visibilité accrue et des opportunités de placement plus directes pour les tirs effectués dans cet intervalle.

![Histogramme Taux de but en fonction angle et distance](images\fe1_histo_taux_angle_distance.png) 

L'analyse du taux de réussite des buts met en évidence des tendances claires. Pour la distance, nous constatons une décroissance rapide du taux de réussite à mesure que la distance augmente. Les tirs effectués à moins de 10 pieds montrent un taux de réussite exceptionnellement élevé, atteignant plus de 25 %, tandis que les tirs au-delà de 40 pieds ont des taux négligeables. Concernant l'angle, une dynamique oscillante est observée : les tirs alignés avec l'axe central (0 degré) ont le plus haut taux de réussite, suivi par des pics secondaires autour de -25 et 25 degrés. Les angles extrêmes, dépassant ±75 degrés, affichent les taux les plus faibles.

Ces analyses soulignent l'importance de la proximité et de l'alignement avec l'axe central pour maximiser les chances de réussite des tirs. Ces observations nous guident dans l'extraction de caractéristiques clés et l'entraînement de modèles prédictifs robustes pour estimer la probabilité de but.

![Histogramme Buts par distance](images\fe1_histo_but_distance.png) 

L'histogramme des buts distingue deux contextes majeurs : les tirs effectués sur un filet vide et ceux réalisés contre un gardien en place. Dans le cas des filets vides, les buts sont relativement uniformément répartis quelle que soit la distance, bien que l'on note un léger pic autour des 20 pieds. Cette uniformité illustre que l'absence de gardien réduit considérablement l'impact de la distance sur le succès du tir. À l'inverse, pour les filets non vides, une forte concentration des buts est observée à des distances inférieures à 20 pieds. Au-delà de cette limite, la probabilité de succès diminue drastiquement, soulignant l'importance de la proximité dans la réussite des tirs contre un gardien actif.

## Partie 4 : Ajout de nouvelles caractéristiques liés aux évènements précédents

Dans cette partie, nous avons considéré de nouvelles caractéristiques, liés surtout aux évènements précédents. Nous considérons alors les caractéristiques suivantes :
- Game seconds : les secondes de jeu 
- Game period : la periode de jeu
- Les coorodnnées x et y
- Shot distance : la distance à laquelle le tir a été effectué
- Shot angle : l'angle du tir efectué par rapport au filet
- Last event type : le type de l'évènement précédent
- Les coordonnées de cet évènement précédent
- Le temps écoulé depuis le dernier évènement
- La distance effectuée depuis le dernier évènement
- Rebound : booléen qui renvoie Vrai si l'évnement précédent était un shot on goal, Faux sinon
- Angle change : le changement d'angle entre l'évènement et le précédent 
- Speed : La vitesse, définie comme la distance depuis l'événement précédent, divisée par le temps écoulé depuis l'événement précédent. 

Ces caractéristiques vont nous permettre de définir si un tir est bel et bien un rebond. On ne peut pas considérer qu'un tir est un rebond si, par exemple,
c'est le premier tir d'une période. 
Nous avons aussi décidé de fixer une limite temporelle entre deux tirs afin d'épurer notre appellation "rebond". Cette limite temporelle est de *** secondes.
Avec toutes ces nouvelles caractéristiques, nous pouvons alors donner un peu plus de contexte sur chacun des tirs.

### Exemple de match : Winnipeg versus Washington

Nous avons isolé un match en particulier : un match entre les Winnipeg Jets et les Washington Capitals. Pour ce match, nous avons créé une expérience sur wandb, en voici le lien : https://wandb.ai/hicham-mazouzi-university-of-montreal/IFT6758.2024-A/artifacts/dataset/wpg_v_wsh_2017021065
