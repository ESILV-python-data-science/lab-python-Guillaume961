# Reponses examen

Classify pages :

L'objectif est de mettre au point un classifier de page afin de pr�dire pour chaque page d�un manuscrit son type.
Il faut �galement pouvoir estimer la performance des classifiers utilis�s.

On cherche � faire de la classification sur des livres. Il s'agit d'une technique supervis�e. 
On souhaite � associer une image � un label (ici le type de page) en entrainant des classifiers sur des datasets comportants les r�ponses.

Pour manipuler les images, nous avons besoin de les convertir en vecteurs. Pour ce faire, il faut d'abord r�duire l'image pour que les calculs ne prennent pas trop de temps. On constate que nos images ont une dimension rectangulaire, on essaye donc de r�duire les images � une taille de 12,16 pour garder les proportions pour le calcul des vecteurs.

Afin de ne pas avoir besoin de reg�n�rer ces vecteurs, on stock ces informations dans un fichier pickle que l'on va lire par la suite.

Pour manipuler les donn�es on utilise un �chantillon al�atoire qui permettra d'entrainer nos classifiers et d'estimer les param�tres des algorithmes. En plus de cette partition on cr�e aussi un ensemble de validation et un ensemble de test.

Les mod�les de classifiers utilis�s sont la regression logistique et le k-plus proche voisin.

Les r�sultats trouv�s au niveau de la pr�cision des predictions ne sont pas tr�s bons. Entre les differents classfiers et param�tres on trouve entre 0.71 et 0.83 de ratio de bonnes pr�dictions.

Avec plus de temps, il faudrait essayer des mod�les plus complexes tels que SVM ou random forest... toujours en recherchant � minimiser l'erreur de pr�diction.

Les commandes utilis�es sont : 
```
classify_pages.py --images-list --save-features img.pkl
classify_pages.py --load-features img.pkl
classify_pages.py --load-features img.pkl --classify --logistic-regression
classify_pages.py --load-features img.pkl --classify --optimize-nearest-neighbors
classify_pages.py --load-features img.pkl --classify --nearest-neighbors 1
```

# Quelques r�sultats console

![Regression Logistique](img_reponse/logReg.png "Regression Logistique")

![KNN1](img_reponse/NearestN1.png "KNN1")

![Multiple KNN](img_reponse/multiple_knn.png "Multiple KNN")


