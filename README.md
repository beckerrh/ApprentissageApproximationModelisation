# Apprentissage, approximation et modélisation

Ce dépôt rassemble les notebooks et les programmes associés au cours de
M2 MMS **Apprentissage, approximation et modélisation**.

Le point de vue du cours est principalement mathématique : les notebooks
servent à construire et expérimenter les machines étudiées dans le cours,
notamment avec JAX, Flax et Optax.

## Parcours

Le parcours d'introduction comprend actuellement :

1. [Prérequis Python](Cours/Introduction/00_prerequis_python.ipynb) :
   autoévaluation rapide du niveau Python attendu ;
2. [Introduction à JAX](Cours/Introduction/01_introduction_jax.ipynb) :
   tableaux, différentiation automatique et transformations de fonctions ;
3. [Optimisation avec Optax](Cours/Introduction/02_optax.ipynb) :
   deux exemples concis d'ajustement de paramètres ;
4. [Machines mathématiques avec Flax NNX](Cours/Introduction/03_flax_nnx.ipynb) :
   définition et apprentissage d'un perceptron multicouche en quelques lignes.

Les premiers exercices du cours sont disponibles dans l'ordre suivant :

1. [Algèbre linéaire et calcul différentiel](Cours/Exercices/01_algebre_calcul_differentiel.ipynb) :
   produit de Hadamard et dérivées composante par composante ;
2. [Gradient et Hessienne de machines simples](Cours/Exercices/02_gradient_hessienne_machines.ipynb) :
   régression linéaire et perceptron ;
3. [ReLU, fonctions affines par morceaux et splines](Cours/Exercices/03_relu_splines.ipynb) :
   réalisation de $\mathcal P_h^1$, interpolation et maillage adaptatif ;
4. [Log-somme-exponentielle et softmax](Cours/Exercices/04_logsumexp_softmax.ipynb) :
   gradient, Hessienne, approximation régulière du maximum et stabilité numérique ;
5. [Courbes périodiques : MLP dense, MLP partagé et CNN](Cours/Exercices/05_courbes_periodiques.ipynb) :
   partage des poids, nombre de paramètres et équivariance par translation ;
6. [Erreur, complexité et ordres d'approximation en dimension un](Cours/Exercices/06_erreur_complexite_approximation.ipynb) :
   inversion erreur–complexité et ordres $N^{-1}$ et $N^{-2}$ pour les machines $P^0$ et $P^1$ ;
7. [Profondeur et régions affines](Cours/Exercices/07_profondeur_regions_affines.ipynb) :
   compositions de la fonction tente et comparaison profondeur–largeur.

Les répertoires **Cours/Classification** et **Cours/ODE** contiennent des
supports existants qui seront progressivement harmonisés.

## Convention pédagogique

Chaque notebook doit rester centré sur un objectif mathématique précis et
contenir :

- une courte introduction ;
- du code directement exécutable ;
- des expériences ou questions à traiter ;
- une conclusion indiquant ce qu'il faut retenir.

Le vocabulaire français est privilégié ; les termes anglais usuels sont
indiqués entre apostrophes lors de leur première introduction importante.
