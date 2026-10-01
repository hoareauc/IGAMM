# IGAMM - Analyse Isogéométrique 2D pour l'Acoustique

Projet de calcul modal pour cavités acoustiques 2D utilisant l'**Analyse Isogéométrique (IGA)** avec des fonctions de base B-splines / NURBS.

L'architecture du code est **entièrement procédurale / non orientée objet (sans classes)**, basée sur des fonctions pures et des structures de données transparentes (tableaux NumPy et matrices creuses SciPy).

---

## Structure du projet

```text
IGAMM/
├── requirements.txt                   # Dépendances scientifiques (NumPy, SciPy, Matplotlib)
├── README.md                          # Documentation et guide d'utilisation
│
├── iga/                               # Module fonctionnel d'Analyse Isogéométrique 2D
│   ├── __init__.py                    # Exports des fonctions publiques
│   ├── bspline.py                     # Algorithmes de Cox-de Boor et Piegl & Tiller (1D et 2D)
│   ├── quadrature.py                  # Quadrature de Gauss-Legendre 1D/2D et mapping par élément
│   ├── geometrie.py                   # Paramétrage géométrique, jacobienne et métrique
│   ├── assemblage.py                  # Assemblage procédural des matrices globales K et M (CSR)
│   ├── solveur.py                     # Résolution du problème aux valeurs propres et solutions analytiques
│   └── post_traitement.py             # Évaluation tensorielle sur grille, affichage et visualisations
│
└── cas_tests/                         # Dossier contenant les cas tests
    └── cavite_rectangulaire/          # Cas test 1 : Cavité rectangulaire 2D [0, Lx] x [0, Ly]
        ├── __init__.py
        ├── cas_cavite_rectangulaire.py   # Script principal de calcul et tracé des modes
        ├── validation_analytique.py      # Test unitaire de validation et non-régression
        └── modes_cavite_rectangulaire.png# Cartographies 2D des modes propres acoustiques
```

---

## Formulation Physique et Mathématique

Pour une cavité acoustique bidimensionnelle $\Omega$, la pression acoustique $p(x, y)$ satisfait l'équation de Helmholtz aux valeurs propres :
$$-\Delta p = k^2 p \quad \text{dans } \Omega$$
avec $k = \omega / c$ le nombre d'onde, $\omega = 2\pi f$ la pulsation et $c$ la célérité acoustique (343 m/s dans l'air à 20°C).

Pour des parois parfaitement rigides (conditions de Neumann homogènes) :
$$\frac{\partial p}{\partial n} = 0 \quad \text{sur } \partial\Omega$$

La formulation faible variationnelle conduit au problème matriciel généralisé :
$$K \mathbf{p} = \lambda M \mathbf{p}$$
où $\lambda = k^2 = (\omega/c)^2$, avec :
- Matrice de rigidité : $K_{ij} = \int_\Omega \nabla R_i \cdot \nabla R_j \, d\Omega$
- Matrice de masse : $M_{ij} = \int_\Omega R_i \, R_j \, d\Omega$

Pour une cavité rectangulaire $[0, L_x] \times [0, L_y]$, les fréquences propres analytiques sont :
$$f_{m, n} = \frac{c}{2} \sqrt{\left(\frac{m}{L_x}\right)^2 + \left(\frac{n}{L_y}\right)^2}, \quad m, n \in \mathbb{N}$$

---

## Installation et Prérequis

Python 3.12 (ou version 3.x stable) avec les bibliothèques scientifiques :

```bash
pip install -r requirements.txt
```

---

## Exécution du Cas Test (Cavité Rectangulaire)

Pour lancer le calcul modal et générer la figure des modes propres :

```bash
python cas_tests/cavite_rectangulaire/cas_cavite_rectangulaire.py
```

Pour lancer les tests de validation numérique automatique :

```bash
python cas_tests/cavite_rectangulaire/validation_analytique.py
```
