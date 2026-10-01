"""
Package IGA (Analyse Isogéométrique 2D) - Programmation Procédurale / Fonctionnelle
Ce module regroupe les fonctions mathématiques et numériques pour l'IGA 2D :
- Fonctions B-Splines et calcul des dérivées (Piegl & Tiller)
- Intégration par quadrature de Gauss-Legendre
- Paramétrage et métrique géométrique 2D
- Assemblage des matrices globales de rigidité et de masse
- Résolution du problème aux valeurs propres acoustique
- Post-traitement, calcul d'erreur et visualisations des modes
"""

from .bspline import (
    generer_vecteur_noeuds_uniforme,
    trouver_intervalle,
    fonctions_base_1d,
    derivees_fonctions_base_1d,
    fonctions_base_et_derivees_2d,
)
from .quadrature import (
    quadrature_gauss_1d,
    quadrature_gauss_2d,
    mapper_quadrature_vers_element,
)
from .geometrie import (
    creer_geometrie_rectangle,
    creer_geometrie_disque_nurbs,
    creer_geometrie_ellipse_nurbs,
    evaluer_jacobienne_et_gradients_physiques,
)
from .assemblage import (
    assembler_systeme_acoustique_2d,
)
from .solveur import (
    resoudre_modes_acoustiques_2d,
    frequences_analytiques_cavite_rectangulaire,
    frequences_analytiques_cavite_circulaire,
)
from .post_traitement import (
    evaluer_champ_pression_grille,
    afficher_comparaison_modes,
    visualiser_modes_acoustiques,
    evaluer_mode_sur_grille_nurbs,
    creer_grille_pyvista_mode,
    visualiser_mode_pyvista_3d,
    visualiser_planche_modes_pyvista_3d,
)

__all__ = [
    "generer_vecteur_noeuds_uniforme",
    "trouver_intervalle",
    "fonctions_base_1d",
    "derivees_fonctions_base_1d",
    "fonctions_base_et_derivees_2d",
    "quadrature_gauss_1d",
    "quadrature_gauss_2d",
    "mapper_quadrature_vers_element",
    "creer_geometrie_rectangle",
    "creer_geometrie_disque_nurbs",
    "creer_geometrie_ellipse_nurbs",
    "evaluer_jacobienne_et_gradients_physiques",
    "assembler_systeme_acoustique_2d",
    "resoudre_modes_acoustiques_2d",
    "frequences_analytiques_cavite_rectangulaire",
    "frequences_analytiques_cavite_circulaire",
    "evaluer_champ_pression_grille",
    "afficher_comparaison_modes",
    "visualiser_modes_acoustiques",
    "evaluer_mode_sur_grille_nurbs",
    "creer_grille_pyvista_mode",
    "visualiser_mode_pyvista_3d",
    "visualiser_planche_modes_pyvista_3d",
]
