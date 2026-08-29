# Apprentissage, approximation et modélisation

Ce dépôt rassemble les notebooks et les programmes associés au cours de
M2 MMS **Apprentissage, approximation et modélisation**.

Le point de vue du cours est principalement mathématique : les notebooks
servent à construire et expérimenter les machines étudiées dans le cours,
notamment avec JAX, Flax et Optax.

## Parcours

### Introduction

1. [Prérequis Python](Cours/01_Introduction/00_prerequis_python.ipynb) :
   autoévaluation rapide du niveau Python attendu ;
2. [Introduction à JAX](Cours/01_Introduction/01_introduction_jax.ipynb) :
   tableaux, différentiation automatique et transformations de fonctions ;
3. [Optimisation avec Optax](Cours/01_Introduction/02_optax.ipynb) :
   deux exemples concis d'ajustement de paramètres ;
4. [Machines mathématiques avec Flax NNX](Cours/01_Introduction/03_flax_nnx.ipynb) :
   définition et apprentissage d'un perceptron multicouche en quelques lignes ;
5. [Algèbre linéaire et calcul différentiel](Cours/01_Introduction/04_algebre_calcul_differentiel.ipynb) :
   produit de Hadamard, compression d'image par SVD et dérivées composante
   par composante.

### Machines mathématiques

1. [Gradient et Hessienne de machines simples](Cours/02_MachinesMathematiques/01_gradient_hessienne_machines.ipynb) :
   régression linéaire et perceptron ;
2. [ReLU, fonctions affines par morceaux et splines](Cours/02_MachinesMathematiques/02_relu_splines.ipynb) :
   réalisation de $\mathcal P_h^1$, interpolation et maillage adaptatif ;
3. [Log-somme-exponentielle et softmax](Cours/02_MachinesMathematiques/03_logsumexp_softmax.ipynb) :
   gradient, Hessienne, approximation régulière du maximum et stabilité numérique ;
4. [Courbes périodiques : MLP dense, MLP partagé et CNN](Cours/02_MachinesMathematiques/04_courbes_periodiques.ipynb) :
   partage des poids, nombre de paramètres et équivariance par translation ;
5. [Autoencodeur linéaire et SVD](Cours/02_MachinesMathematiques/05_autoencodeur_lineaire.ipynb) :
   projection optimale, apprentissage d'une factorisation et non-unicité des
   coordonnées latentes.

### Approximation

1. [Erreur, complexité et ordres d'approximation en dimension un](Cours/03_Approximation/01_erreur_complexite_approximation.ipynb) :
   inversion erreur–complexité et ordres $N^{-1}$ et $N^{-2}$ pour les machines $P^0$ et $P^1$ ;
2. [Profondeur et régions affines](Cours/03_Approximation/02_profondeur_regions_affines.ipynb) :
   compositions de la fonction tente et comparaison profondeur–largeur.

### Apprentissage

1. [Préconditionnement, gradient naturel et matrice de Fisher](Cours/04_Apprentissage/01_preconditionnement_fisher.ipynb) :
   métriques sur les paramètres, Hessienne, Fisher du modèle, Fisher
   empirique et dépendance au changement de coordonnées ;
2. [Gradient, accélération et gradient conjugué](Cours/04_Apprentissage/02_gradient_accelere_conjugue.ipynb) :
   conditionnement, distribution du spectre et préconditionnement d'une
   fonction quadratique ;
3. [Mini-lots, Adam et différentiation automatique](Cours/04_Apprentissage/03_mini_lots_adam_autodiff.ipynb) :
   covariance du gradient, comparaison des optimisations, changement
   d'échelle et produits jacobien--vecteur.

### Classification binaire

1. [Marge et SVM affine](Cours/05_ClassificationBinaire/01_marges_svm_affine.ipynb) :
   données séparables, marge géométrique et fonctions de coût charnière et logistique ;
2. [Données non séparables : hyperplan et MLP](Cours/05_ClassificationBinaire/02_non_separable_svm_mlp.ipynb) :
   limites des machines affines et comparaison avec un perceptron multicouche ;
3. [SVM à noyau gaussien et MLP](Cours/05_ClassificationBinaire/03_noyau_gaussien_mlp.ipynb) :
   comparaison sur les demi-lunes, complexité et sensibilité aux perturbations ;
4. [Application aux données Wisconsin](Cours/05_ClassificationBinaire/04_wisconsin.ipynb) :
   préparation des données, sélection des paramètres et analyse des erreurs.

### Classification à plusieurs classes

1. [Iris : score affine et MLP](Cours/06_ClassificationPlusieursClasses/01_iris_affine_mlp.ipynb) :
   régions de décision, softmax et matrices de confusion ;
2. [MNIST : score affine, MLP et CNN](Cours/06_ClassificationPlusieursClasses/02_mnist_affine_mlp_cnn.ipynb) :
   structure des machines, fréquences d'erreur et confusions entre chiffres.

### Régression

1. [Fonction perturbée : pertes, erreur et classes de machines](Cours/07_Regression/01_fonction_bruitee_pertes.ipynb) :
   pertes quadratique, absolue et de Huber, polynômes, fonctions P1, SVR et MLP ;
2. [Quartet d'Anscombe](Cours/07_Regression/02_quartet_anscombe.ipynb) :
   indicateurs numériques, représentations et pertes robustes ;
3. [California Housing](Cours/07_Regression/03_california_housing.ipynb) :
   normalisation, régression affine, SVR, MLP et analyse des résidus ;
4. [Apprentissage d'une application solution](Cours/07_Regression/04_application_solution.ipynb) :
   interpolation, extrapolation et sensibilité au paramètre.

### Résolution d'équations

1. [Collocation et schéma du point milieu](Cours/08_ResolutionEquations/01_collocation_point_milieu.ipynb) :
   fonction affine par morceaux, Crank--Nicolson, ordre de convergence et
   équation non linéaire ;
2. [Une expérience critique avec un résidu ponctuel](Cours/08_ResolutionEquations/02_residu_ponctuel_pinn.ipynb) :
   résidu invisible entre les points, polynômes, fonctions P1 et MLP ;
3. [Cinétique chimique : application solution et estimation de paramètres](Cours/08_ResolutionEquations/03_reactions_chimiques_estimation.ipynb) :
   intégration avec `odeint`, apprentissage de $S_h(q)$, contraintes physiques
   et problème inverse, puis préconditionnement par les sensibilités ;
4. [Poisson : énergie et résidu fort](Cours/08_ResolutionEquations/04_poisson_energie_residu.ipynb) :
   condition au bord exacte, comparaison des pertes et résidu ponctuel
   invisible ;
5. [Apprendre l'opérateur solution de Poisson](Cours/08_ResolutionEquations/05_apprentissage_operateur_poisson.ipynb) :
   machine linéaire, fréquences nouvelles, SVD et changement de grille.

### Modèles de langage

1. [Du texte aux données d'apprentissage](Cours/09_ModelesLangage/01_texte_donnees.ipynb) :
   vocabulaire, encodage, fenêtres de contexte et multiplicité des données ;
2. [Un modèle de bigrammes](Cours/09_ModelesLangage/02_bigrammes.ipynb) :
   fréquences de transition, lissage, perplexité et génération ;
3. [Plongement vectoriel et factorisation de faible rang](Cours/09_ModelesLangage/03_plongement_faible_rang.ipynb) :
   sélection des colonnes, rang, SVD et non-convexité ;
4. [Un calcul d'attention](Cours/09_ModelesLangage/04_attention.ipynb) :
   masque causal, softmax par ligne et combinaison des valeurs ;
5. [Un petit modèle de langage](Cours/09_ModelesLangage/05_petit_modele_langage.ipynb) :
   comparaison entre bigrammes et MLP avec plongement, apprentissage et
   température de génération.

### Graphes et réseaux de neurones sur graphes

1. [Laplacien, diffusion et réseau de neurones sur graphe](Cours/10_Graphes/01_laplacien_diffusion_gnn.ipynb) :
   énergie et spectre du Laplacien, diffusion, invariance par permutation et
   première classification de sommets par convolution sur graphe.

### Interprétation et sensibilité des machines

1. [Sensibilité, conditionnement, Shapley et robustesse](Cours/11_InterpretationSensibilite/01_sensibilite_shapley_robustesse.ipynb) :
   dérivées par rapport aux entrées et aux paramètres, conditionnements
   absolu et relatif, valeurs de Shapley et bornes de robustesse.

## Convention pédagogique

Chaque notebook doit rester centré sur un objectif mathématique précis et
contenir :

- une courte introduction ;
- du code directement exécutable ;
- des expériences ou questions à traiter ;
- une conclusion indiquant ce qu'il faut retenir.

Le vocabulaire français est privilégié ; les termes anglais usuels sont
indiqués entre apostrophes lors de leur première introduction importante.
