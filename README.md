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
    │
    ├── cavite_rectangulaire/          # Cas test 1 : Cavité rectangulaire 2D [0, Lx] x [0, Ly]
    │   ├── __init__.py
    │   ├── cas_cavite_rectangulaire.py   # Script principal de calcul et tracé des modes
    │   ├── validation_analytique.py      # Test unitaire de validation et non-régression
    │   ├── convergence_k_rafinement.py   # Analyse paramétrique de convergence et de temps CPU
    │   ├── modes_cavite_rectangulaire.png# Cartographies 2D des modes propres acoustiques
    │   ├── convergence_k_rafinement_modes_1_5_10.png # Erreur en fréquence vs DDL (p = 2 à 7)
    │   ├── temps_calcul_k_rafinement.png # Temps d'assemblage et de résolution vs DDL (p = 2 à 7)
    │   └── analyse_complete_k_rafinement.png # Planche maîtresse combinée (Précision & Temps CPU)
    │
    └── cavite_circulaire/             # Cas test 2 : Cavité circulaire 2D (Rayon R)
        ├── __init__.py
        ├── cas_cavite_circulaire.py      # Calcul modal IGA NURBS et comparaison Bessel
        ├── validation_analytique_circulaire.py # Test unitaire automatique
        ├── convergence_cavite_circulaire.py # Analyse de convergence h et scalabilité CPU
        ├── geometrie_control_net_cavite_circulaire.png # Géométrie, points de contrôle et control net
        ├── modes_cavite_circulaire.png   # Cartographies 2D des modes propres cylindriques
        ├── convergence_cavite_circulaire_modes.png # Erreur vs DDL pour modes cibles
        ├── temps_calcul_cavite_circulaire.png # Temps d'assemblage et solveur vs DDL
        └── analyse_complete_cavite_circulaire.png # Planche maîtresse complète (Précision & CPU)
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

### 1. Cavité Rectangulaire $[0, L_x] \times [0, L_y]$
Les fréquences propres analytiques sont données par :
$$f_{m, n} = \frac{c}{2} \sqrt{\left(\frac{m}{L_x}\right)^2 + \left(\frac{n}{L_y}\right)^2}, \quad m, n \in \mathbb{N}$$

### 2. Cavité Circulaire de Rayon $R$
Les modes propres sont les fonctions de Bessel de première espèce $J_m$ :
$$p_{m, n}(r, \theta) = J_m(k_{m, n} r) \cos(m \theta + \phi_0)$$
La condition de paroi rigide en $r = R$ impose $J'_m(k_{m, n} R) = 0$. Les fréquences propres exactes sont donc :
$$f_{m, n} = \frac{c \, \alpha'_{m, n}}{2\pi R}$$
où $\alpha'_{m, n}$ est la $n$-ième racine non nulle de la dérivée de la fonction de Bessel $J'_m(x) = 0$.

#### Paramétrisation NURBS du Disque (« Carré Gonflé »)
Pour éviter la singularité polaire en $r=0$ (liée aux coordonnées polaires standard), la cavité est discrétisée en un **patch NURBS unique** de degré quadratique ($p=2$) représentant un « carré gonflé » sans rotation :
- 9 points de contrôle de base avec poids $w = 1$ aux 4 coins et au centre, et $w = 1/\sqrt{2}$ aux milieux des arêtes.
- Le bord paramétrique reproduit **strictement le cercle $r=R$ à la précision machine**.
- Le raffinement $h$ par insertion de nœuds en coordonnées projectives préserve la géométrie exacte sans aucune erreur de maillage.

---

## Installation et Prérequis

Python 3.12 (ou version 3.x stable) avec les bibliothèques scientifiques :

```bash
pip install -r requirements.txt
```

---

## Exécution des Cas Tests

### 1. Cavité Rectangulaire
```bash
# Calcul modal et cartographies 2D
python cas_tests/cavite_rectangulaire/cas_cavite_rectangulaire.py

# Validation numérique automatique (unittest)
python cas_tests/cavite_rectangulaire/validation_analytique.py

# Analyse paramétrique de convergence du k-rafinement (p = 2 à 7)
python cas_tests/cavite_rectangulaire/convergence_k_rafinement.py
```

### 2. Cavité Circulaire (NURBS Patch Unique)
```bash
# Calcul modal, réseau de contrôle et modes propres
python cas_tests/cavite_circulaire/cas_cavite_circulaire.py

# Validation numérique automatique par rapport aux zéros de Bessel
python cas_tests/cavite_circulaire/validation_analytique_circulaire.py

# Analyse paramétrique de convergence h (NURBS) et temps CPU
python cas_tests/cavite_circulaire/convergence_cavite_circulaire.py
```

