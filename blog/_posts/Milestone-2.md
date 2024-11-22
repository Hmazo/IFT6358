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
